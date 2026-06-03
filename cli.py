#!/usr/bin/env python3
"""
Command-line interface for the Financial Document Intelligence system.

Examples:
    # Index a folder of filings
    python cli.py index data/raw

    # Index a single file
    python cli.py index data/raw/acme_10k.pdf

    # Ask a question
    python cli.py ask "What was total revenue in fiscal 2024 and how did it change?"

    # Interactive Q&A session
    python cli.py chat

    # Show index status / wipe it
    python cli.py status
    python cli.py reset
"""
from __future__ import annotations

import argparse
import sys

from src.rag_pipeline import FinancialRAG, Answer
from src.vectorstore import VectorStore


def _print_answer(ans: Answer) -> None:
    print("\n" + "=" * 70)
    print(ans.answer)
    print("=" * 70)
    if ans.sources:
        print("\nSources:")
        for s in ans.sources:
            print(f"  [{s.marker}] {s.source} | page {s.page} | "
                  f"{s.section} | {s.element_type} | score {s.score}")
            print(f"        {s.preview}")
    print()


def cmd_index(args) -> None:
    rag = FinancialRAG()
    print(f"Indexing: {args.path} ...")
    stats = rag.index(args.path)
    print(f"Done. files={stats['files']} new_chunks={stats['chunks']} "
          f"total_in_store={stats.get('total_in_store', '?')}")


def cmd_ask(args) -> None:
    rag = FinancialRAG()
    if rag.store.count() == 0:
        print("Index is empty. Run `python cli.py index <path>` first.")
        sys.exit(1)
    _print_answer(rag.ask(args.question))


def cmd_chat(args) -> None:
    rag = FinancialRAG()
    if rag.store.count() == 0:
        print("Index is empty. Run `python cli.py index <path>` first.")
        sys.exit(1)
    print("Financial RAG chat. Type 'exit' to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            _print_answer(rag.ask(q))


def cmd_status(args) -> None:
    store = VectorStore()
    docs = store.all_documents()
    files = sorted({d["metadata"].get("source", "?") for d in docs})
    print(f"Chunks in store: {store.count()}")
    print(f"Documents ({len(files)}):")
    for f in files:
        print(f"  - {f}")


def cmd_reset(args) -> None:
    VectorStore().reset()
    print("Index cleared.")


def main() -> None:
    p = argparse.ArgumentParser(description="Financial Document Intelligence (RAG)")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="Index a file or directory")
    pi.add_argument("path")
    pi.set_defaults(func=cmd_index)

    pa = sub.add_parser("ask", help="Ask one question")
    pa.add_argument("question")
    pa.set_defaults(func=cmd_ask)

    pc = sub.add_parser("chat", help="Interactive Q&A")
    pc.set_defaults(func=cmd_chat)

    ps = sub.add_parser("status", help="Show index status")
    ps.set_defaults(func=cmd_status)

    pr = sub.add_parser("reset", help="Clear the index")
    pr.set_defaults(func=cmd_reset)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
