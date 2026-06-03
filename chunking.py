"""
Chunking layer.

Splits `Element`s into retrieval-sized `Chunk`s while respecting two rules
that matter for financial documents:

  1. Never split a table across chunks if it fits the table budget — a row of
     numbers detached from its header is worse than useless.
  2. Carry page + section metadata onto every chunk so the LLM can cite
     precisely and the user can verify against the source filing.

Token counting uses tiktoken when available, otherwise a ~4-chars-per-token
heuristic. Exact token counts are not required; stable, sensible splits are.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import settings
from src.ingestion import Element

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover - heuristic fallback
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    element_type: str
    page: int | None
    section: str | None
    source: str

    def to_metadata(self) -> dict:
        return {
            "element_type": self.element_type,
            "page": self.page if self.page is not None else "",
            "section": self.section if self.section is not None else "",
            "source": self.source,
        }


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(0-9])")


def _split_sentences(text: str) -> list[str]:
    # Split on paragraph breaks first, then sentences, to keep coherence.
    pieces: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        pieces.extend(s for s in _SENT_SPLIT.split(para) if s.strip())
    return pieces or [text]


def _pack(units: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack text units into ~max_tokens windows with overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        ut = count_tokens(unit)
        if current and current_tokens + ut > max_tokens:
            chunks.append(" ".join(current).strip())
            # Build overlap tail from the end of the previous chunk.
            tail, tail_tokens = [], 0
            for prev in reversed(current):
                pt = count_tokens(prev)
                if tail_tokens + pt > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += pt
            current = tail[:]
            current_tokens = tail_tokens
        current.append(unit)
        current_tokens += ut

    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def chunk_elements(elements: list[Element]) -> list[Chunk]:
    cfg = settings.chunk
    chunks: list[Chunk] = []
    counter = 0

    for el in elements:
        tokens = count_tokens(el.text)

        if el.element_type == "table":
            # Keep small/medium tables whole; only split oversized ones.
            if tokens <= cfg.table_max_tokens:
                parts = [el.text]
            else:
                parts = _pack(el.text.splitlines(), cfg.table_max_tokens, 0)
        else:
            if tokens <= cfg.max_tokens:
                parts = [el.text]
            else:
                parts = _pack(_split_sentences(el.text),
                              cfg.max_tokens, cfg.overlap_tokens)

        for part in parts:
            cid = f"{el.source}::p{el.page or 0}::#{counter}"
            chunks.append(Chunk(
                chunk_id=cid,
                text=part,
                element_type=el.element_type,
                page=el.page,
                section=el.section,
                source=el.source,
            ))
            counter += 1

    return chunks
