"""Command-line entry point: ``sage-kg extract|construct|query``."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="sage-kg",
        description="SAGE-KG: extract triples, build a graph, and answer questions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="SUR triplet extraction")
    extract.add_argument("models", nargs="+", help="Ollama model names, e.g. qwen2.5:14b")
    extract.add_argument("--data", "-d", default="./data")
    extract.add_argument("--output", "-o", default="./output")
    extract.add_argument("--patterns", nargs="+", default=["*.md", "*.txt"])
    extract.add_argument("--chunk", type=int, default=200)
    extract.add_argument("--overlap", type=int, default=25)

    construct = sub.add_parser("construct", help="Build graph + indexes from triples JSON")
    construct.add_argument("--input-triplets", required=True)
    construct.add_argument("--graph-file", default="knowledge_graph.pickle")
    construct.add_argument("--chunk-file", default="chunk_data.pickle")
    construct.add_argument("--tfidf-file", default="tfidf_data.joblib")
    construct.add_argument("--embedding-model", default="all-mpnet-base-v2")

    query = sub.add_parser("query", help="Hybrid retrieval + answering")
    query.add_argument("--qa-file", required=True)
    query.add_argument("--output-file", default="output.md")
    query.add_argument("--graph-file", default="knowledge_graph.pickle")
    query.add_argument("--chunk-file", default="chunk_data.pickle")
    query.add_argument("--tfidf-file", default="tfidf_data.joblib")
    query.add_argument("--embedding-model", default="all-mpnet-base-v2")
    query.add_argument("--llm-model", default="qwen2.5:14b")
    query.add_argument("--max-hops", type=int, default=3)

    args = parser.parse_args(argv)

    if args.command == "extract":
        from sage_kg.extraction.agents import TripleProcessor, setup_logging
        from langchain_community.chat_models import ChatOllama
        from pathlib import Path

        for model_name in args.models:
            setup_logging(model_name, Path(args.output) / "logs")
            TripleProcessor(
                llm=ChatOllama(model=model_name, temperature=0),
                data_folder=args.data,
                chunk_size=args.chunk,
                overlap=args.overlap,
                output_dir=args.output,
            ).run(args.patterns)
        return 0

    if args.command == "construct":
        from sage_kg.construction.create_kg import build_knowledge_graph

        G = build_knowledge_graph(
            args.input_triplets,
            graph_file=args.graph_file,
            chunk_file=args.chunk_file,
            tfidf_file=args.tfidf_file,
            embedding_model=args.embedding_model,
        )
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return 0

    if args.command == "query":
        from sentence_transformers import SentenceTransformer
        from langchain_community.llms import Ollama
        from sage_kg.querying.query_kg import (
            NetworkXRetriever,
            read_sample_file,
            save_to_markdown,
        )

        qa_pairs = read_sample_file(args.qa_file)
        if not qa_pairs:
            print(f"No Question/Answer pairs found in {args.qa_file}")
            return 1
        queries = [q for q, _ in qa_pairs]
        ground_truths = [a for _, a in qa_pairs]
        retriever = NetworkXRetriever(max_hop_depth=args.max_hops).load(
            args.graph_file, args.chunk_file, args.tfidf_file
        )
        encoder = SentenceTransformer(args.embedding_model)
        llm = Ollama(model=args.llm_model, temperature=0)
        results = {}
        for query_text in queries:
            answer, _ = retriever.ask(query_text, encoder, llm)
            results[query_text] = answer
        save_to_markdown(queries, ground_truths, results, args.output_file)
        print(f"Wrote {len(results)} answers to {args.output_file}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
