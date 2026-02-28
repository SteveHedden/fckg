"""Canonical manifest for foundation graph files."""

from __future__ import annotations

from pathlib import Path

# example_model/foundation/manifest.py -> project root (three levels up)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Ontology/schema first, then award-system instance files.
ONTOLOGY_PATH = PROJECT_ROOT / "movieontology.ttl"
INSTANCE_PATHS = [
    PROJECT_ROOT / "data" / "instances" / "oscar_nominations.ttl",
    PROJECT_ROOT / "data" / "instances" / "films.ttl",
    PROJECT_ROOT / "data" / "instances" / "people.ttl",
    PROJECT_ROOT / "data" / "instances" / "bafta_nominations.ttl",
    PROJECT_ROOT / "data" / "instances" / "sag_nominations.ttl",
    PROJECT_ROOT / "data" / "instances" / "dga_nominations.ttl",
    PROJECT_ROOT / "data" / "instances" / "pga_nominations.ttl",
    PROJECT_ROOT / "data" / "instances" / "golden_globes_nominations.ttl",
    PROJECT_ROOT / "data" / "instances" / "category_alignments.ttl",
]


def default_foundation_paths(*, include_ontology: bool = True) -> list[Path]:
    """Return ordered default paths for loading the foundation graph."""
    paths: list[Path] = []
    if include_ontology:
        paths.append(ONTOLOGY_PATH)
    paths.extend(INSTANCE_PATHS)
    return paths


__all__ = [
    "PROJECT_ROOT",
    "ONTOLOGY_PATH",
    "INSTANCE_PATHS",
    "default_foundation_paths",
]

