"""Construction tests that must not load sentence-transformers / Pillow."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage_kg.models import Graph
from sage_kg.construction.create_kg import assemble_graph


def test_graph_from_triples():
    triples = [
        {"subject": "christopher nolan", "predicate": "directed", "object": "inception"},
        {"s": "inception", "p": "released_in", "o": "2010"},
        {"subject": "christopher nolan", "predicate": "directed", "object": "inception"},
    ]
    graph = Graph.from_triples(triples)
    assert "christopher nolan" in graph.entities
    assert "inception" in graph.entities
    assert ("christopher nolan", "directed", "inception") in graph.relations
    assert len(graph.relations) == 2
    nx_g = graph.to_nx()
    assert nx_g.number_of_nodes() == 3
    assert nx_g.number_of_edges() == 2


def test_assemble_graph_from_json(tmp_path):
    path = tmp_path / "triples.json"
    path.write_text(
        '[{"subject": "a", "predicate": "knows", "object": "b"}]',
        encoding="utf-8",
    )
    graph = assemble_graph(path)
    assert graph.relations == {("a", "knows", "b")}


def test_graph_roundtrip(tmp_path):
    graph = Graph.from_triples([{"subject": "x", "predicate": "rel", "object": "y"}])
    out = graph.save(tmp_path / "graph.json")
    loaded = Graph.load(out)
    assert loaded.relations == graph.relations


def test_import_construction_does_not_load_sentence_transformers():
    import sage_kg
    import sage_kg.construction.create_kg as mod

    assert sage_kg.__version__
    assert "sentence_transformers" not in sys.modules
    assert "torchvision" not in sys.modules
    assert "PIL" not in sys.modules
    assert hasattr(mod, "assemble_graph")


def test_aggregate_and_from_triples():
    from sage_kg.pipeline import SAGEKG

    a = SAGEKG.from_triples([{"subject": "a", "predicate": "knows", "object": "b"}])
    b = SAGEKG.from_triples([{"s": "b", "p": "knows", "o": "c"}])
    combined = SAGEKG.aggregate([a, b])
    assert len(combined.relations) == 2
    assert "a" in combined.entities and "c" in combined.entities


if __name__ == "__main__":
    test_graph_from_triples()
    test_assemble_graph_from_json(Path("/tmp"))
    print("ok")
