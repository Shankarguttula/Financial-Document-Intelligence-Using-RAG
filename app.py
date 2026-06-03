"""
Streamlit web UI.

Run:
    streamlit run app.py

Features:
  - Upload PDFs/TXT/MD and index them on the fly.
  - Ask questions; answers stream back with an expandable Sources panel.
  - Inspect index status in the sidebar.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from config import settings
from src.rag_pipeline import FinancialRAG
from src.vectorstore import VectorStore


st.set_page_config(page_title="Financial Document Intelligence",
                   page_icon="📊", layout="wide")


@st.cache_resource
def get_rag() -> FinancialRAG:
    return FinancialRAG()


rag = get_rag()

st.title("📊 Financial Document Intelligence")
st.caption("RAG-based GenAI Q&A over your financial documents — grounded, "
           "cited, and built to refuse rather than guess.")

# --------------------------------------------------------------------- #
# Sidebar: ingestion + status
# --------------------------------------------------------------------- #
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload filings / reports (PDF, TXT, MD)",
        type=["pdf", "txt", "md"], accept_multiple_files=True,
    )
    if uploads and st.button("Index uploaded files", type="primary"):
        with st.spinner("Ingesting, chunking, embedding…"):
            total = 0
            with tempfile.TemporaryDirectory() as tmp:
                for up in uploads:
                    fp = Path(tmp) / up.name
                    fp.write_bytes(up.getbuffer())
                    stats = rag.index(fp)
                    total += stats["chunks"]
        st.success(f"Indexed {len(uploads)} file(s), {total} chunks.")

    st.divider()
    store = VectorStore()
    st.metric("Chunks in index", store.count())
    files = sorted({d["metadata"].get("source", "?")
                    for d in store.all_documents()})
    if files:
        st.write("**Indexed documents**")
        for f in files:
            st.write(f"• {f}")
    if st.button("Clear index"):
        store.reset()
        st.rerun()

    st.divider()
    st.caption(f"LLM: `{settings.llm.provider}` / `{settings.llm.active_model}`")
    st.caption(f"Embeddings: `{settings.embedding.provider}`")

# --------------------------------------------------------------------- #
# Main: chat
# --------------------------------------------------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(
                        f"**[{s.marker}]** `{s.source}` — page {s.page} — "
                        f"*{s.section}* — {s.element_type} "
                        f"(score {s.score})\n\n> {s.preview}"
                    )

if prompt := st.chat_input("Ask about revenue, risks, cash flow, guidance…"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if VectorStore().count() == 0:
            msg = "Please upload and index at least one document first."
            st.warning(msg)
            st.session_state.history.append(
                {"role": "assistant", "content": msg})
        else:
            with st.spinner("Retrieving and reasoning…"):
                ans = rag.ask(prompt)
            st.markdown(ans.answer)
            if ans.sources:
                with st.expander("Sources"):
                    for s in ans.sources:
                        st.markdown(
                            f"**[{s.marker}]** `{s.source}` — page {s.page} — "
                            f"*{s.section}* — {s.element_type} "
                            f"(score {s.score})\n\n> {s.preview}"
                        )
            st.session_state.history.append({
                "role": "assistant",
                "content": ans.answer,
                "sources": ans.sources,
            })
