"""High-level document → KG API, following the kg-gen ``generate()`` shape."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from sage_kg.llm import OllamaCompleter, chat_llm
from sage_kg.models import Graph


class SAGEKG:
    """Extract a knowledge graph from text or a document folder using SUR.

    Example::

        from sage_kg import SAGEKG

        kg = SAGEKG(model="qwen2.5:14b")
        graph = kg.generate("Christopher Nolan directed Inception in 2010.")
        print(graph.relations)
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        output_dir: Union[str, Path] = "output",
        chunk_size: int = 200,
        overlap: int = 25,
        max_hops: int = 3,
        embedding_model: str = "all-mpnet-base-v2",
        api_base: str = "http://127.0.0.1:11434",
    ):
        self.model = model
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_hops = max_hops
        self.embedding_model = embedding_model
        self.api_base = api_base
        self.graph: Optional[Graph] = None

    def _chat(self):
        return chat_llm(self.model, temperature=0, base_url=self.api_base)

    def _processor(self, data_folder: Optional[str] = None, output_dir: Optional[str] = None):
        from sage_kg.extraction.agents import TripleProcessor

        return TripleProcessor(
            llm=self._chat(),
            data_folder=data_folder,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            output_dir=output_dir or str(self.output_dir),
        )

    @staticmethod
    def _normalize_input(input_data: Union[str, Path, List[Dict], Sequence]) -> str:
        if isinstance(input_data, list):
            lines = []
            for message in input_data:
                if isinstance(message, dict) and "content" in message:
                    role = message.get("role", "user")
                    lines.append(f"{role}: {message['content']}")
                else:
                    lines.append(str(message))
            return "\n".join(lines)
        return str(input_data)

    def generate(
        self,
        input_data: Union[str, Path, List[Dict]],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        output_folder: Optional[Union[str, Path]] = None,
        patterns: Optional[Sequence[str]] = None,
        context: Optional[str] = None,
    ) -> Graph:
        """Build a Graph from a string, message list, or document directory.

        Matches kg-gen's ``generate(input_data=...)`` entry point. SUR is used
        for extraction; no embedding model is loaded.
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if overlap is not None:
            self.overlap = overlap
        out = Path(output_folder).expanduser().resolve() if output_folder else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        source = Path(input_data) if isinstance(input_data, (str, Path)) else None
        processor = self._processor(
            data_folder=str(source) if source and source.is_dir() else str(out),
            output_dir=str(out),
        )

        if source is not None and source.is_dir():
            triples, _ = processor.run(list(patterns or ["*.md", "*.txt"]))
        elif source is not None and source.is_file():
            text = source.read_text(encoding="utf-8", errors="replace")
            if context:
                text = f"{context.strip()}\n\n{text}"
            triples = processor.extract_text(text, file_id=source.name)
            processor._save_results(triples)
        else:
            text = self._normalize_input(input_data)
            if context:
                text = f"{context.strip()}\n\n{text}"
            triples = processor.extract_text(text)
            processor._save_results(triples)

        self.graph = Graph.from_triples(triples or [])
        self.graph.save(out / "graph.json")
        return self.graph

    @staticmethod
    def from_triples(triples: Iterable) -> Graph:
        """Assemble a Graph from already extracted triples (no LLM)."""
        from sage_kg.construction.create_kg import assemble_graph

        return assemble_graph(triples)

    @staticmethod
    def aggregate(graphs: Sequence[Graph]) -> Graph:
        return Graph.aggregate(graphs)

    def to_nx(self, graph: Optional[Graph] = None):
        graph = graph or self.graph
        if graph is None:
            raise ValueError("No graph to convert. Call generate() first.")
        return graph.to_nx()

    def export_graph(self, graph: Optional[Graph] = None, path: Optional[Union[str, Path]] = None) -> Path:
        graph = graph or self.graph
        if graph is None:
            raise ValueError("No graph to export. Call generate() first.")
        return graph.save(path or (self.output_dir / "graph.json"))

    def build_indexes(
        self,
        graph: Optional[Graph] = None,
        graph_file: Optional[Union[str, Path]] = None,
        chunk_file: Optional[Union[str, Path]] = None,
        tfidf_file: Optional[Union[str, Path]] = None,
    ):
        """Optional retrieval indexes. Requires ``pip install 'sage-kg[retrieve]'``."""
        from sage_kg.construction.create_kg import build_knowledge_graph

        graph = graph or self.graph
        if graph is None:
            raise ValueError("No graph to index. Call generate() first.")
        return build_knowledge_graph(
            graph.triples,
            graph_file=str(graph_file or self.output_dir / "knowledge_graph.pickle"),
            chunk_file=str(chunk_file or self.output_dir / "chunk_data.pickle"),
            tfidf_file=str(tfidf_file or self.output_dir / "tfidf_data.joblib"),
            embedding_model=self.embedding_model,
        )

    def ask(self, question: str, graph: Optional[Graph] = None) -> str:
        """Answer from retrieved triples. Needs indexes from ``build_indexes()``."""
        from sage_kg.querying.query_kg import NetworkXRetriever

        graph_file = self.output_dir / "knowledge_graph.pickle"
        if not graph_file.exists():
            self.build_indexes(graph)
        from sentence_transformers import SentenceTransformer

        retriever = NetworkXRetriever(max_hop_depth=self.max_hops).load(
            str(graph_file),
            str(self.output_dir / "chunk_data.pickle"),
            str(self.output_dir / "tfidf_data.joblib"),
        )
        answer, _ = retriever.ask(
            question,
            SentenceTransformer(self.embedding_model),
            OllamaCompleter(self.model, temperature=0, base_url=self.api_base),
        )
        return answer
