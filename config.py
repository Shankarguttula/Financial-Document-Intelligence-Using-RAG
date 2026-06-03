"""
Central configuration for the Financial Document Intelligence system.

Every tunable lives here so you can change behavior without touching logic.
Values can be overridden by environment variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Auto-load a .env file sitting next to this config, so users never have to
# manually `export`/`set` environment variables. Silently skipped if
# python-dotenv isn't installed.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
RAW_DIR = Path(os.getenv("FDI_RAW_DIR", ROOT / "data" / "raw"))
STORE_DIR = Path(os.getenv("FDI_STORE_DIR", ROOT / "data" / "store"))


@dataclass(frozen=True)
class EmbeddingConfig:
    # "local"  -> sentence-transformers, runs offline, no API key needed.
    # "openai" -> OpenAI text-embedding-3 models (needs OPENAI_API_KEY).
    # "voyage" -> Voyage AI (Anthropic's recommended embeddings partner).
    provider: str = os.getenv("FDI_EMBED_PROVIDER", "local")
    local_model: str = os.getenv("FDI_LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
    openai_model: str = os.getenv("FDI_OPENAI_EMBED_MODEL", "text-embedding-3-small")
    voyage_model: str = os.getenv("FDI_VOYAGE_MODEL", "voyage-finance-2")
    # Embed in batches to control memory / rate limits.
    batch_size: int = int(os.getenv("FDI_EMBED_BATCH", "64"))


@dataclass(frozen=True)
class ChunkConfig:
    # Chunk sizes are measured in tokens (approximate, via tiktoken if present).
    max_tokens: int = int(os.getenv("FDI_CHUNK_TOKENS", "550"))
    overlap_tokens: int = int(os.getenv("FDI_CHUNK_OVERLAP", "80"))
    # Tables are kept whole up to this token budget before being split.
    table_max_tokens: int = int(os.getenv("FDI_TABLE_TOKENS", "900"))


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = int(os.getenv("FDI_TOP_K", "8"))          # dense candidates
    final_k: int = int(os.getenv("FDI_FINAL_K", "5"))      # after reranking
    use_hybrid: bool = os.getenv("FDI_HYBRID", "true").lower() == "true"
    # Weight given to keyword (BM25) score when blending with dense score.
    keyword_weight: float = float(os.getenv("FDI_KEYWORD_WEIGHT", "0.35"))
    use_reranker: bool = os.getenv("FDI_RERANK", "false").lower() == "true"
    reranker_model: str = os.getenv(
        "FDI_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


@dataclass(frozen=True)
class LLMConfig:
    # "anthropic" or "openai" — picks which API the generation step uses.
    provider: str = os.getenv("FDI_LLM_PROVIDER", "anthropic")

    # --- Anthropic ---
    # Sonnet is the high-volume default; Opus for hard, multi-document analysis.
    model: str = os.getenv("FDI_LLM_MODEL", "claude-sonnet-4-6")
    heavy_model: str = os.getenv("FDI_LLM_HEAVY_MODEL", "claude-opus-4-8")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")

    # --- OpenAI ---
    # Mini is the cheap high-volume default; full GPT-5.4 for harder analysis.
    openai_model: str = os.getenv("FDI_OPENAI_MODEL", "gpt-5.4-mini")
    openai_heavy_model: str = os.getenv("FDI_OPENAI_HEAVY_MODEL", "gpt-5.4")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # --- Shared generation params ---
    max_tokens: int = int(os.getenv("FDI_LLM_MAX_TOKENS", "1500"))
    # Low temperature: financial answers should be deterministic and grounded.
    temperature: float = float(os.getenv("FDI_LLM_TEMPERATURE", "0.0"))

    @property
    def active_model(self) -> str:
        return self.openai_model if self.provider == "openai" else self.model

    @property
    def active_heavy_model(self) -> str:
        return (self.openai_heavy_model if self.provider == "openai"
                else self.heavy_model)


@dataclass(frozen=True)
class Settings:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    collection_name: str = os.getenv("FDI_COLLECTION", "financial_docs")


settings = Settings()
