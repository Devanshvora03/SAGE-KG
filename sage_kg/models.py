"""In-memory knowledge graph, aligned with kg-gen's Graph object."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple, Union

import networkx as nx


def _triple_key(item) -> Tuple[str, str, str]:
    if hasattr(item, "subject"):
        return (
            str(item.subject).strip().lower(),
            str(item.predicate).strip().lower(),
            str(item.object).strip().lower(),
        )
    return (
        str(item.get("subject") or item.get("s") or "").strip().lower(),
        str(item.get("predicate") or item.get("p") or "").strip().lower(),
        str(item.get("object") or item.get("o") or "").strip().lower(),
    )


@dataclass
class Graph:
    """Document-level KG: entities, (s, p, o) relations, and predicate types."""

    entities: Set[str] = field(default_factory=set)
    relations: Set[Tuple[str, str, str]] = field(default_factory=set)
    edges: Set[str] = field(default_factory=set)
    triples: List[dict] = field(default_factory=list)

    @classmethod
    def from_triples(cls, triples: Iterable) -> "Graph":
        graph = cls()
        seen = set()
        for item in triples:
            s, p, o = _triple_key(item)
            if not (s and p and o):
                continue
            rec = {
                "subject": s,
                "predicate": p,
                "object": o,
                "file_id": getattr(item, "file_id", None)
                if hasattr(item, "file_id")
                else item.get("file_id") or item.get("file"),
                "chunk_id": getattr(item, "chunk_id", None)
                if hasattr(item, "chunk_id")
                else item.get("chunk_id") or item.get("chunk"),
            }
            key = (s, p, o)
            graph.entities.add(s)
            graph.entities.add(o)
            graph.edges.add(p)
            graph.relations.add(key)
            if key not in seen:
                seen.add(key)
                graph.triples.append(rec)
        return graph

    def to_nx(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        for entity in self.entities:
            G.add_node(entity, node_type="entity")
        for s, p, o in self.relations:
            G.add_edge(s, o, predicate=p, original_predicate=p)
        return G

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entities": sorted(self.entities),
            "relations": [list(r) for r in sorted(self.relations)],
            "edges": sorted(self.edges),
            "triples": self.triples,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Graph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            entities=set(data.get("entities") or []),
            relations={tuple(r) for r in data.get("relations") or []},
            edges=set(data.get("edges") or []),
            triples=list(data.get("triples") or []),
        )

    @classmethod
    def aggregate(cls, graphs: Sequence["Graph"]) -> "Graph":
        """Union several graphs (kg-gen's ``aggregate``)."""
        triples: List[dict] = []
        for graph in graphs:
            triples.extend(graph.triples)
        return cls.from_triples(triples)

    def __len__(self) -> int:
        return len(self.relations)
