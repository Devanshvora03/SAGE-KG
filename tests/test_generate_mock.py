"""generate() wiring without calling an LLM."""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage_kg.pipeline import SAGEKG


def test_generate_from_string(tmp_path):
    triples = [
        {"subject": "Christopher Nolan", "predicate": "directed", "object": "Inception", "file_id": "inline", "chunk_id": "c0"},
        {"subject": "Inception", "predicate": "released_in", "object": "2010", "file_id": "inline", "chunk_id": "c0"},
    ]
    kg = SAGEKG(model="qwen2.5:14b", output_dir=tmp_path)

    with patch.object(SAGEKG, "_processor") as proc_factory:
        processor = MagicMock()
        processor.extract_text.return_value = triples
        proc_factory.return_value = processor
        graph = kg.generate(input_data="Christopher Nolan directed Inception in 2010.")

    processor.extract_text.assert_called_once()
    assert ("christopher nolan", "directed", "inception") in graph.relations
    assert (tmp_path / "graph.json").exists()


def test_generate_from_directory(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Christopher Nolan directed Inception in 2010.", encoding="utf-8")
    triples = [{"subject": "nolan", "predicate": "directed", "object": "inception", "file_id": "a.txt", "chunk_id": "c0"}]
    kg = SAGEKG(output_dir=tmp_path / "out")

    with patch.object(SAGEKG, "_processor") as proc_factory:
        processor = MagicMock()
        processor.run.return_value = (triples, None)
        proc_factory.return_value = processor
        graph = kg.generate(input_data=docs)

    processor.run.assert_called_once()
    assert ("nolan", "directed", "inception") in graph.relations


def test_generate_from_messages(tmp_path):
    triples = [{"subject": "France", "predicate": "has capital", "object": "Paris"}]
    kg = SAGEKG(output_dir=tmp_path)

    with patch.object(SAGEKG, "_processor") as proc_factory:
        processor = MagicMock()
        processor.extract_text.return_value = triples
        proc_factory.return_value = processor
        graph = kg.generate(input_data=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ])

    text = processor.extract_text.call_args[0][0]
    assert "France" in text and "Paris" in text
    assert ("france", "has capital", "paris") in graph.relations
