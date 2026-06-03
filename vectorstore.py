"""
Vector store layer.

A thin wrapper over a persistent ChromaDB collection. We supply our own
embeddings (so the same vectors are used for indexing and querying regardless
of provider) and store rich metadata for citation.
"""
from __future__ import annotations

from pathlib import Path

from config import settings, STORE_DIR
from src.chunking import Chunk


class VectorStore:
    def __init__(self, persist_dir: Path | None = None,
                 collection_name: str | None = None):
        try:
            import chromadb
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "chromadb is required. Run: pip install chromadb"
            ) from e

        persist_dir = Path(persist_dir or STORE_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ----------------------------------------------------------------- #
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        if not res["ids"] or not res["ids"][0]:
            return hits
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0],
        ):
            hits.append({
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                # cosine distance -> similarity in [0, 1]
                "score": 1.0 - float(dist),
            })
        return hits

    def all_documents(self) -> list[dict]:
        """Return every stored chunk (used for BM25 keyword index)."""
        res = self._collection.get(include=["documents", "metadatas"])
        return [
            {"chunk_id": cid, "text": doc, "metadata": meta}
            for cid, doc, meta in zip(res["ids"], res["documents"],
                                      res["metadatas"])
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"},
        )
