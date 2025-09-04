
import os
import uuid
import streamlit as st

from backend import get_backend, RetrievedChunk
from dotenv import load_dotenv


st.set_page_config(page_title="Local RAG (SQLite + Groq)", layout="wide")
load_dotenv()
# Also allow Streamlit secrets to provide the key
if not os.getenv("GROQ_API_KEY"):
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# Ensure a session id for message persistence
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

backend = get_backend(db_path="rag.db")


def sidebar_ui() -> None:
    with st.sidebar:
        st.title("RAG")
        # No API key prompt; loaded from .env

        # Documents list with filtering and deletion
        st.subheader("Documents")
        docs = backend.list_documents()
        if not docs:
            st.info("No documents uploaded yet.")
        else:
            selected = st.multiselect("Limit queries to:", options=docs, default=docs)
            st.session_state["selected_docs"] = selected
            del_choice = st.selectbox("Delete document", options=["(select)"] + docs, index=0)
            if st.button("Delete") and del_choice and del_choice != "(select)":
                deleted = backend.delete_document(del_choice)
                st.success(f"Deleted '{del_choice}' ({deleted} chunks)")

        # Keep sidebar minimal; hide thread rendering


# Removed command bar for a simpler UI


def uploader_ui() -> None:
    st.subheader("Add Documents")
    tab1, tab2 = st.tabs(["Upload file", "Paste text"])
    with tab1:
        uploaded = st.file_uploader(
            "Upload PDF/TXT/MD files",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
        )
        if uploaded:
            for file in uploaded:
                data = file.read()
                num_chunks, num_inserted = backend.ingest_file(data, file.name)
                st.success(f"Ingested {file.name}: {num_inserted}/{num_chunks} chunks stored")
    with tab2:
        doc_name = st.text_input("Document name for pasted text", placeholder="e.g., notes.txt")
        text_area = st.text_area("Paste text to store as a document", height=180)
        if st.button("Ingest pasted text"):
            if not text_area.strip():
                st.warning("Please paste some text.")
            else:
                n_chunks, n_inserted = backend.ingest_text(text_area, doc_name or "pasted.txt")
                st.success(f"Stored '{doc_name or 'pasted.txt'}': {n_inserted}/{n_chunks} chunks")


def query_ui() -> None:
    st.subheader("Ask a Question")
    default_model = "llama-3.1-8b-instant"
    model = st.selectbox(
        "Groq Model",
        options=["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
        index=0,
    )

    query = st.text_input("Your question", placeholder="Ask about your uploaded docs...")
    top_k = st.slider("Top-k chunks", min_value=3, max_value=5, value=4)

    if st.button("Search & Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
            return
        try:
            backend.save_message(st.session_state.session_id, "user", query)
            selected_docs = st.session_state.get("selected_docs") or None
            answer, retrieved = backend.answer(query, top_k=top_k, model=model, doc_names=selected_docs)
            backend.save_message(st.session_state.session_id, "assistant", answer)

            st.markdown("### Answer")
            st.write(answer)

        except Exception as e:
            st.error(str(e))


def paste_qa_ui() -> None:
    st.subheader("Quick QA on Pasted Text")
    with st.expander("Open pasted-text QA"):
        col1, col2 = st.columns([1, 1])
        with col1:
            context_text = st.text_area("Paste context text here", height=200)
        with col2:
            question = st.text_input("Your question about the pasted text")
            model = st.selectbox(
                "Groq Model (pasted text)",
                options=["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
                index=0,
            )
            if st.button("Answer from pasted text"):
                if not context_text.strip() or not question.strip():
                    st.warning("Provide both context text and a question.")
                else:
                    try:
                        answer = backend.answer_with_text(context_text, question, model=model)
                        st.markdown("### Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(str(e))

# Removed chat UI for simplicity


def main() -> None:
    st.title("MINI-RAG")
    # Key is loaded from .env; no user prompt

    sidebar_ui()
    uploader_ui()
    query_ui()
    # Removed separate pasted-text QA; ingestion happens in Add Documents


if __name__ == "__main__":
    main()


