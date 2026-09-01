"""Command-line entry: ``sage-kg generate``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="sage-kg",
        description="SAGE-KG: document to knowledge graph via Sequential Uncertainty Resolution.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Extract a KG from text or a document folder")
    generate.add_argument("source", help="Text file, folder of .md/.txt files, or '-' for stdin")
    generate.add_argument("--model", default="qwen2.5:14b")
    generate.add_argument("--output", "-o", default="./output")
    generate.add_argument("--chunk", type=int, default=200)
    generate.add_argument("--overlap", type=int, default=25)

    args = parser.parse_args(argv)

    if args.command == "generate":
        from sage_kg.pipeline import SAGEKG

        kg = SAGEKG(
            model=args.model,
            output_dir=args.output,
            chunk_size=args.chunk,
            overlap=args.overlap,
        )
        if args.source == "-":
            source = sys.stdin.read()
        else:
            source = args.source
        graph = kg.generate(source, output_folder=args.output)
        print(f"entities={len(graph.entities)} relations={len(graph.relations)}")
        print(f"wrote {Path(args.output) / 'graph.json'}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
