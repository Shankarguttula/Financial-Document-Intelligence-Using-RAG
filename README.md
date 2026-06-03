# Financial Document Intelligence — RAG-based GenAI System

A retrieval-augmented question-answering system purpose-built for financial
documents (10-Ks, 10-Qs, annual reports, earnings transcripts, statements). You
load your filings, ask questions in plain English, and get **grounded, cited
answers** — or an honest "not found" instead of a hallucinated number.

It is designed around the failure mode that matters most in finance: a model
confidently inventing or mis-attributing a figure. Every defense in this system
points at that.

---

## Why this is built the way it is

| Concern in financial RAG | How the system handles it |
|---|---|
| Tables get shredded, numbers lose their headers | PDF tables are extracted whole (pdfplumber) and kept intact through chunking |
| Models invent figures | System prompt forbids outside knowledge and requires `[S#]` citations on every figure; refuses when context lacks the answer |
| Exact terms ("free cash flow", a fiscal year) missed by embeddings | Hybrid retrieval blends dense vectors with BM25 keyword scores |
| Can't verify an answer | Every answer ships with sources: file, page, section, and a preview |
| Conflicting figures across filings | Prompt instructs the model to surface discrepancies and cite each source |
| Private data | Default embeddings run **fully offline** (sentence-transformers) |

---

## Architecture

```
        ┌──────────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────────┐
files → │  Ingestion   │ → │  Chunking │ → │  Embeddings  │ → │ Vector Store │
        │ PDF/TXT/MD   │   │ token +   │   │ local /      │   │  (ChromaDB)  │
        │ tables+pages │   │ table aware│  │ Voyage       │   │  cosine      │
        └──────────────┘   └───────────┘   └──────────────┘   └──────┬───────┘
                                                                      │
                                              ┌───────────────────────┘
                                              ▼
   question → ┌───────────┐   dense + BM25   ┌───────────────┐   grounded   ┌─────────┐
              │ Retriever │ ───(+ rerank)──→ │  RAG Pipeline  │ ──prompt──→ │ Claude  │ → cited answer
              └───────────┘                  │ context+cite   │            │ Messages │
                                             └───────────────┘            └─────────┘
```

Modules (`src/`): `ingestion` → `chunking` → `embeddings` → `vectorstore` →
`retriever` → `rag_pipeline` (orchestration + prompt), with `llm` wrapping the
Anthropic Messages API. Everything tunable lives in `config.py`.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

The system supports **OpenAI or Anthropic** for generation, and **OpenAI,
local, or Voyage** for embeddings. Since you have an OpenAI key, the shipped
`.env.example` leads with the OpenAI setup — just paste your key:

```bash
FDI_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
FDI_OPENAI_MODEL=gpt-5.4-mini          # cheap, high-volume default
FDI_OPENAI_HEAVY_MODEL=gpt-5.4         # for harder analysis

FDI_EMBED_PROVIDER=openai              # embed with OpenAI too
FDI_OPENAI_EMBED_MODEL=text-embedding-3-small
```

That single key now powers both the embedding step and the answer generation —
no other accounts needed. (Embeddings can also stay `local` and offline if you
prefer to keep your documents off any API; only generation would then use
OpenAI.)

> **Switching embedding providers?** Run `python cli.py reset` first. Different
> models produce different vector dimensions, so you must re-index when you
> change `FDI_EMBED_PROVIDER`.

**Models** (override in `.env`): OpenAI generation defaults to `gpt-5.4-mini`
(fast/cheap) with `gpt-5.4` as the heavy option; Anthropic defaults to
`claude-sonnet-4-6` / `claude-opus-4-8`. The OpenAI wrapper auto-adapts to the
GPT-5.x reasoning models (handles `max_completion_tokens` and temperature
constraints), so newer model names work without code changes.

---

## Usage

### Command line
```bash
# Index a folder or a single file
python cli.py index data/raw
python cli.py index data/raw/acme_10k_sample.txt

# Ask a question
python cli.py ask "What was total revenue in fiscal 2024 and how did it change?"

# Interactive session
python cli.py chat

# Inspect or clear the index
python cli.py status
python cli.py reset
```

### Web app
```bash
streamlit run app.py
```
Upload documents in the sidebar, index them, and chat. Each answer has an
expandable **Sources** panel.

### As a library
```python
from src.rag_pipeline import FinancialRAG

rag = FinancialRAG()
rag.index("data/raw/acme_10k_sample.txt")

ans = rag.ask("What is the fiscal 2025 revenue guidance?")
print(ans.answer)
for s in ans.sources:
    print(s.marker, s.source, "page", s.page, "-", s.section)
```

### Try the included sample
```bash
python examples/sample_query.py
```
Indexes a synthetic ACME Robotics 10-K and runs four questions — including one
whose answer is *not* in the document, to demonstrate the refusal behavior.

---

## Configuration knobs (`config.py` / env)

- **Generation**: `FDI_LLM_PROVIDER=openai|anthropic`. OpenAI uses
  `FDI_OPENAI_MODEL` / `FDI_OPENAI_HEAVY_MODEL`; Anthropic uses
  `FDI_LLM_MODEL` / `FDI_LLM_HEAVY_MODEL`.
- **Embeddings**: `FDI_EMBED_PROVIDER=openai|local|voyage`. OpenAI uses
  `text-embedding-3-small` (1536-d, cheap) or `text-embedding-3-large` (3072-d).
  Local runs offline via sentence-transformers. Voyage uses the finance-tuned
  `voyage-finance-2`.
- **Chunking**: `FDI_CHUNK_TOKENS`, `FDI_CHUNK_OVERLAP`, `FDI_TABLE_TOKENS`.
- **Retrieval**: `FDI_TOP_K`, `FDI_FINAL_K`, `FDI_HYBRID`,
  `FDI_KEYWORD_WEIGHT`, `FDI_RERANK` (cross-encoder precision pass).
- **LLM params**: `FDI_LLM_TEMPERATURE` (default `0.0` for determinism),
  `FDI_LLM_MAX_TOKENS`.

---

## Extending it

- **More formats**: add a loader in `src/ingestion.py` (e.g. XBRL, HTML
  filings, `.xlsx`) returning `Element`s — the rest of the pipeline is agnostic.
- **Metric extraction**: add a pass that pulls structured KPIs into metadata for
  filtered retrieval ("only FY2024 income-statement chunks").
- **Multi-company / multi-year filtering**: ChromaDB supports metadata `where`
  filters; expose `source`/`section`/`page` filters at query time.
- **Evaluation**: add a small Q/A set with gold sources and measure retrieval
  recall@k and citation accuracy.

---

## Notes & limitations

- This reports what the documents say; it is **not investment advice**.
- Quality depends on parse quality. Scanned PDFs need OCR first (not included).
- The included sample document is synthetic and for demonstration only.
```
