"""
RAG orchestration.

`FinancialRAG` exposes two main verbs:

  * index(paths)  — ingest -> chunk -> embed -> store.
  * ask(question) — retrieve -> build grounded context -> generate answer.

The system prompt is the heart of trustworthiness here. It forces the model to:
  - answer ONLY from the retrieved context,
  - cite every figure with [S#] source markers,
  - say "not found in the provided documents" instead of guessing,
  - never invent or extrapolate numbers,
  - flag when figures appear to conflict across sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import settings
from src.ingestion import load_file, load_directory
from src.chunking import chunk_elements
from src.embeddings import get_embedder
from src.vectorstore import VectorStore
from src.retriever import Retriever, Retrieved
from src.llm import BaseLLM, get_llm


SYSTEM_PROMPT = """You are a meticulous financial document analyst. You answer \
questions strictly using the CONTEXT excerpts provided to you, which come from \
the user's financial documents (filings, reports, statements, transcripts).

Rules you must follow without exception:
1. Use ONLY information present in the CONTEXT. Do not use outside knowledge \
about the company, market, or figures.
2. Every numeric figure, date, or factual claim must be followed by a source \
marker like [S1] or [S2, S3] identifying which excerpt(s) it came from.
3. If the answer is not contained in the CONTEXT, say exactly: \
"I could not find this in the provided documents." Do not guess or estimate.
4. Never fabricate, round, extrapolate, or compute figures that are not \
explicitly supported, unless the user asks for a calculation — in which case \
show each input figure with its [S#] source and the arithmetic.
5. If sources disagree on a figure, point out the discrepancy and cite each.
6. Be concise and precise. Preserve units, currencies, and reporting periods \
exactly as written in the source.
7. You are not providing investment advice; report what the documents say."""


@dataclass
class Source:
    marker: str          # e.g. "S1"
    source: str
    page: int | str
    section: str
    element_type: str
    score: float
    preview: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source]


class FinancialRAG:
    def __init__(self, store: VectorStore | None = None,
                 model: str | None = None):
        self.store = store or VectorStore()
        self.embedder = get_embedder()
        self._retriever: Retriever | None = None
        self._llm: BaseLLM | None = None
        self._model = model

    # ------------------------- indexing ------------------------------- #
    def index(self, path: str | Path) -> dict:
        path = Path(path)
        elements = load_directory(path) if path.is_dir() else load_file(path)
        chunks = chunk_elements(elements)
        if not chunks:
            return {"files": 0, "chunks": 0}

        cfg = settings.embedding
        for i in range(0, len(chunks), cfg.batch_size):
            batch = chunks[i:i + cfg.batch_size]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            self.store.add(batch, vectors)

        # Invalidate retriever so its BM25 index rebuilds with new docs.
        self._retriever = None
        sources = {e.source for e in elements}
        return {"files": len(sources), "chunks": len(chunks),
                "total_in_store": self.store.count()}

    # ------------------------- querying ------------------------------- #
    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.store)
        return self._retriever

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = get_llm(self._model)
        return self._llm

    @staticmethod
    def _build_context(hits: list[Retrieved]) -> tuple[str, list[Source]]:
        blocks, sources = [], []
        for i, h in enumerate(hits, start=1):
            marker = f"S{i}"
            m = h.metadata
            page = m.get("page", "") or "n/a"
            section = m.get("section", "") or "n/a"
            header = (f"[{marker}] source={m.get('source', '?')} "
                      f"page={page} section=\"{section}\" "
                      f"type={m.get('element_type', 'text')}")
            blocks.append(f"{header}\n{h.text}")
            sources.append(Source(
                marker=marker,
                source=m.get("source", "?"),
                page=page,
                section=section,
                element_type=m.get("element_type", "text"),
                score=h.score,
                preview=(h.text[:240] + "…") if len(h.text) > 240 else h.text,
            ))
        return "\n\n---\n\n".join(blocks), sources

    def ask(self, question: str, top_k: int | None = None,
            final_k: int | None = None) -> Answer:
        hits = self.retriever.retrieve(question, top_k, final_k)
        if not hits:
            return Answer(question,
                          "I could not find this in the provided documents.",
                          [])
        context, sources = self._build_context(hits)
        user = (f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                f"Answer using only the context above, with [S#] citations.")
        answer = self.llm.generate(SYSTEM_PROMPT, user)
        return Answer(question, answer, sources)
