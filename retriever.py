"""
Retrieval layer.

Pipeline:
  1. Dense retrieval from the vector store (semantic similarity).
  2. Optional hybrid blend with BM25 keyword scores. Financial queries often
     hinge on exact terms ("free cash flow", "goodwill impairment", a year),
     where keyword matching rescues recall that pure embeddings miss.
  3. Optional cross-encoder reranking for the highest precision on the final
     shortlist handed to the LLM.

Returns a ranked list of `Retrieved` hits with merged metadata + scores.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import settings
from src.embeddings import get_embedder
from src.vectorstore import VectorStore


@dataclass
class Retrieved:
    chunk_id: str
    text: str
    metadata: dict
    score: float


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class Retriever:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self.cfg = settings.retrieval
        self.embedder = get_embedder()
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._corpus: dict[str, dict] = {}
        if self.cfg.use_hybrid:
            self._build_bm25()
        self._reranker = None

    # ----------------------------------------------------------------- #
    def _build_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            # Hybrid silently degrades to dense-only if rank_bm25 is missing.
            self.cfg = type(self.cfg)(**{**self.cfg.__dict__, "use_hybrid": False}) \
                if hasattr(self.cfg, "__dict__") else self.cfg
            return
        docs = self.store.all_documents()
        if not docs:
            return
        self._corpus = {d["chunk_id"]: d for d in docs}
        self._bm25_ids = [d["chunk_id"] for d in docs]
        self._bm25 = BM25Okapi([_tokenize(d["text"]) for d in docs])

    def _keyword_scores(self, query: str, pool: set[str]) -> dict[str, float]:
        if self._bm25 is None:
            return {}
        raw = self._bm25.get_scores(_tokenize(query))
        return {cid: float(s) for cid, s in zip(self._bm25_ids, raw)
                if cid in pool}

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.cfg.reranker_model)
        return self._reranker

    # ----------------------------------------------------------------- #
    def retrieve(self, query: str, top_k: int | None = None,
                 final_k: int | None = None) -> list[Retrieved]:
        top_k = top_k or self.cfg.top_k
        final_k = final_k or self.cfg.final_k

        q_vec = self.embedder.embed_query(query)
        dense_hits = self.store.query(q_vec, top_k)
        if not dense_hits:
            return []

        by_id = {h["chunk_id"]: h for h in dense_hits}
        dense_scores = {h["chunk_id"]: h["score"] for h in dense_hits}

        # --- Hybrid blend ------------------------------------------------- #
        if self.cfg.use_hybrid and self._bm25 is not None:
            kw_scores = self._keyword_scores(query, set(by_id))
            d_norm = _minmax(dense_scores)
            k_norm = _minmax(kw_scores)
            w = self.cfg.keyword_weight
            blended = {
                cid: (1 - w) * d_norm.get(cid, 0.0) + w * k_norm.get(cid, 0.0)
                for cid in by_id
            }
        else:
            blended = dense_scores

        ranked_ids = sorted(blended, key=blended.get, reverse=True)

        # --- Optional cross-encoder rerank -------------------------------- #
        if self.cfg.use_reranker:
            ce = self._get_reranker()
            pairs = [(query, by_id[cid]["text"]) for cid in ranked_ids]
            ce_scores = ce.predict(pairs)
            order = sorted(range(len(ranked_ids)),
                           key=lambda i: ce_scores[i], reverse=True)
            ranked_ids = [ranked_ids[i] for i in order]
            for i, cid in enumerate(ranked_ids):
                blended[cid] = float(ce_scores[order.index(i)]) \
                    if cid in ranked_ids else blended[cid]

        results = []
        for cid in ranked_ids[:final_k]:
            h = by_id[cid]
            results.append(Retrieved(cid, h["text"], h["metadata"],
                                     round(float(blended[cid]), 4)))
        return results
