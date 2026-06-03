"""
Ingestion layer.

Turns raw files (PDF, TXT, MD) into a flat list of `Element` objects that
carry enough metadata for trustworthy financial citations:

    - text         : the content
    - element_type : "text" | "table"
    - page         : 1-indexed source page (None for plaintext)
    - section      : nearest detected heading (e.g. "Item 1A. Risk Factors")
    - source       : file name

Tables are extracted separately and rendered as Markdown so numbers and their
column headers stay attached — losing that alignment is the classic way RAG
systems hallucinate financial figures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class Element:
    text: str
    element_type: str  # "text" or "table"
    page: int | None
    section: str | None
    source: str

    def to_metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        # Chroma metadata values must be str/int/float/bool (no None).
        return {k: (v if v is not None else "") for k, v in d.items()}


# Heading patterns common in 10-K / 10-Q / annual reports and generic docs.
_HEADING_PATTERNS = [
    re.compile(r"^\s*(item\s+\d+[a-z]?\.?.*)$", re.IGNORECASE),
    re.compile(r"^\s*(part\s+[ivx]+\.?.*)$", re.IGNORECASE),
    re.compile(r"^\s*(management'?s discussion.*)$", re.IGNORECASE),
    re.compile(r"^\s*(notes to .*financial statements.*)$", re.IGNORECASE),
    re.compile(r"^\s*(consolidated .*(statements?|sheets?|operations).*)$", re.IGNORECASE),
    # Generic: a short, mostly-uppercase line is probably a heading.
    re.compile(r"^\s*([A-Z][A-Z0-9 ,&\-/]{6,80})\s*$"),
]


def _detect_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 100:
        return None
    for pat in _HEADING_PATTERNS:
        m = pat.match(line)
        if m:
            return m.group(1).strip()
    return None


def _table_to_markdown(rows: list[list]) -> str:
    """Render a list-of-rows table as GitHub-flavored Markdown."""
    cleaned = [
        [("" if c is None else str(c).replace("\n", " ").strip()) for c in row]
        for row in rows
        if any(c not in (None, "") for c in row)
    ]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    header, *body = cleaned
    md = ["| " + " | ".join(header) + " |",
          "| " + " | ".join(["---"] * width) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md)


def _load_pdf(path: Path) -> list[Element]:
    try:
        import pdfplumber
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pdfplumber is required to read PDFs. Run: pip install pdfplumber"
        ) from e

    elements: list[Element] = []
    current_section: str | None = None

    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            # 1) Tables first, so we can subtract them from the prose later.
            table_bboxes = []
            for tbl in page.find_tables():
                rows = tbl.extract()
                md = _table_to_markdown(rows) if rows else ""
                if md:
                    elements.append(Element(md, "table", page_no,
                                            current_section, path.name))
                    table_bboxes.append(tbl.bbox)

            # 2) Prose, excluding regions already captured as tables.
            def _outside_tables(obj):
                cy = (obj["top"] + obj["bottom"]) / 2
                cx = (obj["x0"] + obj["x1"]) / 2
                for (x0, top, x1, bottom) in table_bboxes:
                    if x0 <= cx <= x1 and top <= cy <= bottom:
                        return False
                return True

            text = page.filter(_outside_tables).extract_text() or ""
            for raw_line in text.splitlines():
                heading = _detect_heading(raw_line)
                if heading:
                    current_section = heading
            if text.strip():
                elements.append(Element(text.strip(), "text", page_no,
                                        current_section, path.name))
    return elements


def _load_text(path: Path) -> list[Element]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    elements: list[Element] = []
    current_section: str | None = None
    buffer: list[str] = []

    def flush():
        if buffer:
            block = "\n".join(buffer).strip()
            if block:
                elements.append(Element(block, "text", None,
                                        current_section, path.name))
            buffer.clear()

    for line in text.splitlines():
        heading = _detect_heading(line)
        if heading:
            flush()
            current_section = heading
        buffer.append(line)
    flush()
    return elements


def load_file(path: str | Path) -> list[Element]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return _load_text(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_directory(directory: str | Path) -> list[Element]:
    directory = Path(directory)
    elements: list[Element] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}:
            elements.extend(load_file(path))
    return elements


def iter_sources(elements: Iterable[Element]) -> set[str]:
    return {e.source for e in elements}
