"""Foundation graph loader functions."""

from __future__ import annotations

import logging
from pathlib import Path

from rdflib import Graph, Namespace

from foundation.manifest import PROJECT_ROOT, default_foundation_paths

logger = logging.getLogger(__name__)
MSH = Namespace("http://example.org/ontologies/MovieSHACL3#")

_ENTITY_QUERIES = {
    "Nomination": "SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE { ?x a msh:Nomination }",
    "Person": "SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE { ?x a msh:Person }",
    "Film": "SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE { ?x a msh:Film }",
    "AwardCeremony": "SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE { ?x a msh:AwardCeremony }",
    "AwardCategory": "SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE { ?x a msh:AwardCategory }",
    "AwardSystem": "SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE { ?x a msh:AwardSystem }",
}


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _parse_ttl(graph: Graph, path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Foundation graph file not found: {path}")

    before = len(graph)
    try:
        graph.parse(str(path), format="turtle")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse Turtle file: {path}") from exc
    after = len(graph)
    return after - before


def load_foundation_graph(
    *extra_paths: str | Path,
    include_ontology: bool = True,
) -> Graph:
    """Load ontology + foundation instance TTL files into one Graph."""
    graph = Graph()

    ordered_paths: list[Path] = list(default_foundation_paths(include_ontology=include_ontology))
    ordered_paths.extend(_resolve_path(path) for path in extra_paths)

    for path in ordered_paths:
        added = _parse_ttl(graph, path)
        logger.info("Loaded %s (+%s triples, total=%s)", path, added, len(graph))

    try:
        validate_foundation_graph(graph)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Foundation graph validation failed: %s", exc)

    return graph


def load_enriched_graph(
    *enrichment_paths: str | Path,
    foundation: Graph | None = None,
) -> Graph:
    """Load enrichment overlays on top of foundation graph data."""
    graph = foundation if foundation is not None else load_foundation_graph()

    for path in enrichment_paths:
        resolved = _resolve_path(path)
        added = _parse_ttl(graph, resolved)
        logger.info("Loaded enrichment %s (+%s triples, total=%s)", resolved, added, len(graph))

    return graph


def validate_foundation_graph(graph: Graph) -> dict[str, int]:
    """Run diagnostic entity counts on the loaded foundation graph."""
    counts: dict[str, int] = {}
    init_bindings = {"msh": MSH}

    for entity_type, query in _ENTITY_QUERIES.items():
        result = graph.query(query, initNs=init_bindings)
        row = next(iter(result), None)
        count = int(row[0]) if row is not None else 0
        counts[entity_type] = count

    logger.info(
        (
            "Foundation counts: Nomination=%d Person=%d Film=%d "
            "AwardCeremony=%d AwardCategory=%d AwardSystem=%d"
        ),
        counts["Nomination"],
        counts["Person"],
        counts["Film"],
        counts["AwardCeremony"],
        counts["AwardCategory"],
        counts["AwardSystem"],
    )
    if counts["Nomination"] < 1000:
        logger.warning("Nomination count is suspiciously low: %d", counts["Nomination"])

    return counts


__all__ = [
    "load_enriched_graph",
    "load_foundation_graph",
    "validate_foundation_graph",
]
