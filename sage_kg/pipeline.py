"""High-level extract → construct → query API."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer

from sage_kg.construction.create_kg import build_knowledge_graph
from sage_kg.extraction.agents import Triple, TripleProcessor
from sage_kg.querying.query_kg import NetworkXRetriever


class SAGEKG:
    """Build a knowledge graph from text with SUR, then answer questions from it.

    Example::

        from sage_kg import SAGEKG

        kg = SAGEKG(model="qwen2.5:14b")
        kg.extract("./docs")
        kg.build()
        print(kg.ask("Who directed Inception?"))
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        embedding_model: str = "all-mpnet-base-v2",
        output_dir: Union[str, Path] = "output",
        chunk_size: int = 200,
        overlap: int = 25,
        max_hops: int = 3,
    ):
        self.model = model
        self.embedding_model_name = embedding_model
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_hops = max_hops

        self.graph_file = self.output_dir / "knowledge_graph.pickle"
        self.chunk_file = self.output_dir / "chunk_data.pickle"
        self.tfidf_file = self.output_dir / "tfidf_data.joblib"

        self.triples: List[Triple] = []
        self.triples_path: Optional[Path] = None
        self._encoder = None
        self._retriever: Optional[NetworkXRetriever] = None

    def _chat(self):
        return ChatOllama(model=self.model, temperature=0)

    def _llm(self):
        return Ollama(model=self.model, temperature=0)

    def _embedder(self):
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def extract(
        self,
        data_folder: Union[str, Path],
        patterns: Optional[Sequence[str]] = None,
    ) -> List[Triple]:
        """Extract triples from a folder of ``.md`` / ``.txt`` files."""
        processor = TripleProcessor(
            llm=self._chat(),
            data_folder=str(data_folder),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            output_dir=str(self.output_dir),
        )
        triples, json_path = processor.run(list(patterns or ["*.md", "*.txt"]))
        self.triples = triples or []
        self.triples_path = Path(json_path) if json_path else None
        return self.triples

    def extract_text(self, text: str, file_id: str = "inline") -> List[Triple]:
        """Extract triples from a single string and remember them for ``build()``."""
        processor = TripleProcessor(
            llm=self._chat(),
            data_folder=str(self.output_dir),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            output_dir=str(self.output_dir),
        )
        self.triples = processor.extract_text(text, file_id=file_id)
        self.triples_path = processor._save_results(self.triples)
        return self.triples

    def build(self, triples: Optional[Union[str, Path, Iterable]] = None):
        """Assemble the NetworkX graph and retrieval indexes from extracted triples."""
        source = triples if triples is not None else self.triples_path
        if source is None and self.triples:
            source = [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "file_id": t.file_id,
                    "chunk_id": t.chunk_id,
                }
                for t in self.triples
            ]
        if source is None:
            raise ValueError("No triples to build from. Call extract() first or pass triples=.")

        G = build_knowledge_graph(
            source,
            graph_file=str(self.graph_file),
            chunk_file=str(self.chunk_file),
            tfidf_file=str(self.tfidf_file),
            embedding_model=self._embedder(),
        )
        self._retriever = None
        return G

    def load(self, graph_file=None, chunk_file=None, tfidf_file=None) -> "SAGEKG":
        """Load a previously built graph instead of extracting again."""
        self.graph_file = Path(graph_file or self.graph_file)
        self.chunk_file = Path(chunk_file or self.chunk_file)
        self.tfidf_file = Path(tfidf_file or self.tfidf_file)
        self._retriever = NetworkXRetriever(max_hop_depth=self.max_hops).load(
            str(self.graph_file), str(self.chunk_file), str(self.tfidf_file)
        )
        return self

    def _get_retriever(self) -> NetworkXRetriever:
        if self._retriever is None:
            if not self.graph_file.exists():
                raise FileNotFoundError(
                    f"No graph at {self.graph_file}. Call build() or load() first."
                )
            self.load()
        return self._retriever

    def retrieve(self, question: str) -> List[str]:
        """Return the multi-hop triples retrieved for ``question``."""
        return self._get_retriever().retrieve_triplets(question, self._embedder())

    def ask(self, question: str) -> str:
        """Retrieve triples and generate a short factual answer."""
        answer, _ = self._get_retriever().ask(question, self._embedder(), self._llm())
        return answer
