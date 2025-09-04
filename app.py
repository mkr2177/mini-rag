
import os
import uuid
import streamlit as st

from backend import get_backend, RetrievedChunk


st.set_page_config(page_title="Local RAG (SQLite + Groq)", layout="wide")

# Ensure a session id for message persistence
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

backend = get_backend(db_path="rag.db")


def sidebar_ui() -> None:
    with st.sidebar:
        st.title("RAG")
        # Groq API Key setter (session-only)
        st.subheader("Groq API")
        current_key = os.getenv("GROQ_API_KEY", "")
        key_input = st.text_input("GROQ_API_KEY", type="password", value=current_key)
        if st.button("Set API Key"):
            if key_input:
                os.environ["GROQ_API_KEY"] = key_input
                st.success("GROQ_API_KEY set for this session.")
            else:
                st.warning("Empty key not saved.")

        # Documents list (compact)
        st.subheader("Documents")
        docs = backend.list_documents()
        if not docs:
            st.info("No documents uploaded yet.")
        else:
            sel = st.selectbox("View document chunks", options=["(select)"] + docs, index=0)
            if sel and sel != "(select)":
                chunks = backend.get_chunks_for_document(sel)
                st.caption(f"Showing {len(chunks)} chunks from {sel}")
                for idx, text in chunks[:100]:
                    with st.expander(f"Chunk {idx}"):
                        st.write(text)

        # Keep sidebar minimal; hide thread rendering


# Removed command bar for a simpler UI


def uploader_ui() -> None:
    st.subheader("Upload Documents")
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
            answer, retrieved = backend.answer(query, top_k=top_k, model=model)
            backend.save_message(st.session_state.session_id, "assistant", answer)

            st.markdown("### Answer")
            st.write(answer)

        except Exception as e:
            st.error(str(e))


# Removed chat UI for simplicity


def main() -> None:
    st.title("Retrieval-Augmented Generation: SQLite + Groq")
    groq_key_set = bool(os.getenv("GROQ_API_KEY"))
    if not groq_key_set:
        st.warning("GROQ_API_KEY is not set. Set it in environment before asking questions.")

    sidebar_ui()
    uploader_ui()
    query_ui()


if __name__ == "__main__":
    main()


