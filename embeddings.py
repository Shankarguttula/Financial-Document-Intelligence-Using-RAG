"""
Embeddings layer.

Two interchangeable providers behind one interface:

  * "local"  — sentence-transformers, runs fully offline, zero API keys.
               Great default for getting started and for private data.
  * "voyage" — Voyage AI (Anthropic's recommended embedding partner). The
               `voyage-finance-2` model is domain-tuned for filings, earnings
               calls, and similar financial text.

Switch via FDI_EMBED_PROVIDER in the environment or config.py.
"""
from __future__ import annotations

import os
from functools import lru_cache

from config import settings


class BaseEmbedder:
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for the local embedder. "
                "Run: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=settings.embedding.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


class OpenAIEmbedder(BaseEmbedder):
    # Output dimensions for the text-embedding-3 family.
    _DEFAULT_DIMS = {"text-embedding-3-small": 1536,
                     "text-embedding-3-large": 3072}

    def __init__(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "openai is required for the OpenAI embedder. "
                "Run: pip install openai"
            ) from e
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI embedder.")
        self._client = OpenAI(api_key=api_key)
        self._model = model_name
        self.dim = self._DEFAULT_DIMS.get(model_name, 1536)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        bs = settings.embedding.batch_size
        for i in range(0, len(texts), bs):
            batch = [t.replace("\n", " ") for t in texts[i:i + bs]]
            resp = self._client.embeddings.create(model=self._model,
                                                  input=batch)
            out.extend(d.embedding for d in resp.data)
        if out:
            self.dim = len(out[0])
        return out


class VoyageEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        try:
            import voyageai
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "voyageai is required for the Voyage embedder. "
                "Run: pip install voyageai"
            ) from e
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("Set VOYAGE_API_KEY to use the Voyage embedder.")
        self._client = voyageai.Client(api_key=api_key)
        self._model = model_name
        # voyage-finance-2 -> 1024 dims; confirmed lazily after first call.
        self.dim = 1024

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        bs = settings.embedding.batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            resp = self._client.embed(batch, model=self._model,
                                      input_type=input_type)
            out.extend(resp.embeddings)
        if out:
            self.dim = len(out[0])
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    cfg = settings.embedding
    if cfg.provider == "openai":
        return OpenAIEmbedder(cfg.openai_model)
    if cfg.provider == "voyage":
        return VoyageEmbedder(cfg.voyage_model)
    return LocalEmbedder(cfg.local_model)
