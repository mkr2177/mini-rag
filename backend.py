
import os
import io
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class RetrievedChunk:
    doc_name: str
    chunk_index: int
    text: str
    score: float


# -----------------------------
# Utility functions
# -----------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _now_ts() -> int:
    return int(time.time())


# -----------------------------
# RAG Backend
# -----------------------------

class RAGBackend:
    def __init__(self, db_path: str = "rag.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._create_schema()

        # Lazy-loaded models
        self._embedder: Optional[SentenceTransformer] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None

    # -------------------------
    # Schema
    # -------------------------
    def _create_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                name TEXT PRIMARY KEY,
                created_at INTEGER
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at INTEGER,
                FOREIGN KEY(doc_name) REFERENCES documents(name)
            );
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_doc_chunk
            ON chunks(doc_name, chunk_index);
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER
            );
            """
        )
        self._conn.commit()

    # -------------------------
    # Models
    # -------------------------
    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._embedder

    def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        if self._text_splitter is None:
            # Approximate ~1000 tokens per chunk with small overlap
            self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=100,
            )
        return self._text_splitter

    # -------------------------
    # Document ingestion
    # -------------------------
    def _read_pdf(self, file_like: io.BytesIO) -> str:
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed. Add pypdf to requirements.")
        reader = PdfReader(file_like)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts).strip()

    def _read_text(self, file_bytes: bytes) -> str:
        # Try utf-8, fallback latin-1
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")

    def ingest_file(self, file_bytes: bytes, filename: str) -> Tuple[int, int]:
        """
        Ingest a single file. Returns (num_chunks, num_inserted).
        """
        name = os.path.basename(filename)
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""

        if ext == "pdf":
            text = self._read_pdf(io.BytesIO(file_bytes))
        else:
            # txt, md, markdown, others treated as plain text
            text = self._read_text(file_bytes)

        text = (text or "").strip()
        if not text:
            return (0, 0)

        # Upsert document record
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO documents(name, created_at) VALUES (?, ?)",
            (name, _now_ts()),
        )
        self._conn.commit()

        # Chunk
        splitter = self._get_text_splitter()
        chunks = splitter.split_text(text)
        if not chunks:
            return (0, 0)

        # Embed
        embedder = self._get_embedder()
        embeddings = embedder.encode(chunks, normalize_embeddings=False)

        # Insert chunks with embeddings
        inserted = 0
        for idx, (chunk_text, emb_vec) in enumerate(zip(chunks, embeddings)):
            emb_json = json.dumps(emb_vec.tolist())
            try:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO chunks(doc_name, chunk_index, text, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (name, idx, chunk_text, emb_json, _now_ts()),
                )
                inserted += 1
            except Exception:
                continue
        self._conn.commit()

        return (len(chunks), inserted)

    def ingest_text(self, text: str, doc_name: str) -> Tuple[int, int]:
        """
        Ingest raw text as a document. Returns (num_chunks, num_inserted).
        """
        name = os.path.basename(doc_name) if doc_name else f"pasted-{_now_ts()}"
        text = (text or "").strip()
        if not text:
            return (0, 0)

        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO documents(name, created_at) VALUES (?, ?)",
            (name, _now_ts()),
        )
        self._conn.commit()

        splitter = self._get_text_splitter()
        chunks = splitter.split_text(text)
        if not chunks:
            return (0, 0)

        embedder = self._get_embedder()
        embeddings = embedder.encode(chunks, normalize_embeddings=False)

        inserted = 0
        for idx, (chunk_text, emb_vec) in enumerate(zip(chunks, embeddings)):
            emb_json = json.dumps(emb_vec.tolist())
            try:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO chunks(doc_name, chunk_index, text, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (name, idx, chunk_text, emb_json, _now_ts()),
                )
                inserted += 1
            except Exception:
                continue
        self._conn.commit()
        return (len(chunks), inserted)

    # -------------------------
    # Retrieval
    # -------------------------
    def search(self, query: str, top_k: int = 4, doc_names: Optional[List[str]] = None) -> List[RetrievedChunk]:
        embedder = self._get_embedder()
        q_emb = np.array(embedder.encode([query])[0], dtype=np.float32)

        cur = self._conn.cursor()
        if doc_names:
            placeholders = ",".join(["?"] * len(doc_names))
            cur.execute(
                f"SELECT doc_name, chunk_index, text, embedding FROM chunks WHERE doc_name IN ({placeholders})",
                tuple(doc_names),
            )
        else:
            cur.execute("SELECT doc_name, chunk_index, text, embedding FROM chunks")
        rows = cur.fetchall()
        if not rows:
            return []

        scored: List[RetrievedChunk] = []
        for doc_name, chunk_index, text, emb_json in rows:
            try:
                vec = np.array(json.loads(emb_json), dtype=np.float32)
            except Exception:
                continue
            score = _cosine_similarity(q_emb, vec)
            scored.append(RetrievedChunk(doc_name=doc_name, chunk_index=int(chunk_index), text=text, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(1, int(top_k))]

    # -------------------------
    # LLM Answering (Groq)
    # -------------------------
    def answer(self, query: str, top_k: int = 4, model: str = "llama-3.1-8b-instant", doc_names: Optional[List[str]] = None) -> Tuple[str, List[RetrievedChunk]]:
        retrieved = self.search(query, top_k=top_k, doc_names=doc_names)

        if not retrieved:
            return ("Not found in provided text.", [])

        # Build labeled context; encourage citations by using the [doc:idx] tag
        context_lines = []
        for r in retrieved:
            tag = f"[{r.doc_name}:{r.chunk_index}]"
            context_lines.append(f"{tag} {r.text}")
        context_text = "\n\n".join(context_lines)

        system_prompt = (
            "You are a helpful assistant.\n"
            "Use ONLY the context below to answer the question.\n"
            "If the answer is not in the context, say: \"Not found in provided text.\"\n\n"
            "Answer with citations in [doc:chunk] format."
        )
        user_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question:\n{query}"
        )

        chat = ChatGroq(temperature=0.0, model_name=model)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        resp = chat.invoke(messages)
        answer_text = resp.content if hasattr(resp, "content") else str(resp)
        return (answer_text.strip(), retrieved)

    def answer_with_text(self, context_text: str, query: str, model: str = "llama-3.1-8b-instant") -> str:
        """Answer using only provided raw context text (no retrieval)."""
        context_text = (context_text or "").strip()
        if not context_text:
            return "Not found in provided text."

        system_prompt = (
            "You are a helpful assistant.\n"
            "Use ONLY the context below to answer the question.\n"
            "If the answer is not in the context, say: \"Not found in provided text.\"\n\n"
            "Answer with citations in [doc:chunk] format if labels are provided."
        )
        user_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question:\n{query}"
        )

        chat = ChatGroq(temperature=0.0, model_name=model)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        resp = chat.invoke(messages)
        return (resp.content if hasattr(resp, "content") else str(resp)).strip()

    # -------------------------
    # Documents & Chunks
    # -------------------------
    def list_documents(self) -> List[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT name FROM documents ORDER BY created_at DESC")
        return [row[0] for row in cur.fetchall()]

    def get_chunks_for_document(self, doc_name: str) -> List[Tuple[int, str]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT chunk_index, text FROM chunks WHERE doc_name = ? ORDER BY chunk_index ASC",
            (doc_name,),
        )
        return [(int(idx), text) for idx, text in cur.fetchall()]

    def delete_document(self, doc_name: str) -> int:
        """
        Delete a document and its chunks. Returns number of chunks deleted.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(1) FROM chunks WHERE doc_name = ?", (doc_name,))
        count = int(cur.fetchone()[0])
        cur.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
        cur.execute("DELETE FROM documents WHERE name = ?", (doc_name,))
        self._conn.commit()
        return count

    # -------------------------
    # Thread persistence (messages)
    # -------------------------
    def save_message(self, session_id: str, role: str, content: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now_ts()),
        )
        self._conn.commit()

    def load_messages(self, session_id: str) -> List[Tuple[str, str]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        return [(role, content) for role, content in cur.fetchall()]


# Convenience singleton for simple scripts
_backend_singleton: Optional[RAGBackend] = None


def get_backend(db_path: str = "rag.db") -> RAGBackend:
    global _backend_singleton
    if _backend_singleton is None:
        _backend_singleton = RAGBackend(db_path=db_path)
    return _backend_singleton


