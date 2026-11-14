"""CLI: python -m agent {ingest,ask,eval}"""

from __future__ import annotations

import argparse
import sys

from . import embeddings
from .evals import evaluate, format_report, load_cases
from .graph import build_graph, run
from .index import VectorIndex
from .models import load_chat_model
from .tools import make_tools


def _graph(index_dir: str, max_steps: int):
    index = VectorIndex.load(embeddings.from_env(), index_dir)
    return build_graph(load_chat_model(), make_tools(index), max_steps=max_steps)


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; model output often contains Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("--index", default=".index", help="index directory")
    parser.add_argument("--max-steps", type=int, default=6)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="chunk and embed a folder of markdown")
    ingest.add_argument("docs")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--trace", action="store_true", help="print every message")
    ev = sub.add_parser("eval", help="run an eval dataset")
    ev.add_argument("dataset")
    ev.add_argument("--min-pass-rate", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.command == "ingest":
        index = VectorIndex.from_directory(embeddings.from_env(), args.docs)
        index.save(args.index)
        print(f"indexed {len(index)} chunks with {index.embedder.name} -> {args.index}")
        return 0

    graph = _graph(args.index, args.max_steps)
    if args.command == "ask":
        result = run(graph, args.question)
        if args.trace:
            for message in result.messages:
                message.pretty_print()
        print(result.answer)
        print(f"\ntools: {[n for n, _ in result.tool_calls]}  grounded: {result.grounded}  steps: {result.steps}")
        return 0

    results = evaluate(graph, load_cases(args.dataset))
    print(format_report(results))
    rate = sum(r.passed for r in results) / max(len(results), 1)
    return 0 if rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
