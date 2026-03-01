"""Main feature extractor combining canonical graph features with optional enrichment.

Usage:
    from features.extractor import FeatureExtractor

    # Canonical foundation graph (default manifest)
    fx = FeatureExtractor()
    df = fx.extract()

    # Override with explicit TTL path(s) for tests or targeted runs
    fx = FeatureExtractor("data/instances/oscar_nominations.ttl")
    df = fx.extract()

    # With graph-native enrichment overlays
    fx = FeatureExtractor(
        enrichment_ttl_paths=[
            "data/enriched/film_metadata.ttl",
            "data/enriched/film_credits.ttl",
            "data/enriched/person_metadata.ttl",
            "data/enriched/precursor_awards.ttl",
        ],
    )
    df = fx.extract()
    fx.to_csv("features_enriched.csv")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from foundation.loader import load_foundation_graph
from .canonical import extract_canonical_features, load_graph
from .graph_enrichment import extract_graph_enrichment


class FeatureExtractor:
    """Extracts Best Picture prediction features from the FCKG knowledge graph.

    Canonical features are computed from one RDF nomination graph.
    By default this is loaded via the foundation manifest (all award systems).
    When explicit instance paths are provided, those are loaded directly for
    testability/backward compatibility.
    Enrichment overlays (TMDB/OMDB metadata, Wikidata precursor awards) are
    merged from enrichment TTL overlays only.
    """

    def __init__(
        self,
        *instance_paths: str | Path,
        ontology_path: str | Path | None = None,
        precursor_ttl_paths: list[str | Path] | None = None,
        enrichment_ttl_paths: str | Path | list[str | Path] | None = None,
    ):
        self._instance_paths = instance_paths
        self._ontology_path = ontology_path
        self._precursor_ttl_paths = precursor_ttl_paths or []
        if enrichment_ttl_paths is None:
            self._enrichment_ttl_paths: list[str | Path] | None = None
        elif isinstance(enrichment_ttl_paths, (str, Path)):
            self._enrichment_ttl_paths = [enrichment_ttl_paths]
        else:
            self._enrichment_ttl_paths = list(enrichment_ttl_paths)
        self._result: pd.DataFrame | None = None

    def extract(self) -> pd.DataFrame:
        """Run the full extraction pipeline and return the feature DataFrame."""
        # Default path: unified foundation loader.
        # Explicit paths keep existing behavior for tests/targeted runs.
        if self._instance_paths:
            paths = list(self._instance_paths)
            if self._ontology_path:
                paths.insert(0, self._ontology_path)
            for p in self._precursor_ttl_paths:
                p = Path(p)
                if p.exists():
                    paths.append(p)
            graph = load_graph(*paths)
        else:
            graph = load_foundation_graph(*self._precursor_ttl_paths, include_ontology=True)

        # Enrichment source priority:
        # 1) explicit enrichment TTL paths
        # 2) auto-detected enrichment TTLs in data/enriched/
        explicit_ttls = self._enrichment_ttl_paths
        auto_ttls: list[Path] = []
        if explicit_ttls:
            for p in explicit_ttls:
                path = Path(p)
                if path.exists():
                    graph.parse(str(path), format="turtle")
        elif not self._instance_paths:
            enriched_dir = Path("data/enriched")
            if enriched_dir.exists():
                auto_ttls = sorted(enriched_dir.glob("*.ttl"))
                for ttl_file in auto_ttls:
                    graph.parse(str(ttl_file), format="turtle")

        # Canonical features (Oscar nominations + precursor indicators)
        canonical_df = extract_canonical_features(graph)
        if canonical_df.empty:
            self._result = canonical_df
            return canonical_df

        if explicit_ttls or auto_ttls:
            self._result = extract_graph_enrichment(graph, canonical_df)
            return self._result

        self._result = canonical_df
        return self._result

    @property
    def features(self) -> pd.DataFrame:
        """Access the extracted features (calls extract() if not yet run)."""
        if self._result is None:
            self.extract()
        return self._result

    def to_csv(self, path: str | Path, **kwargs) -> None:
        """Export features to CSV."""
        df = self.features
        kwargs.setdefault("index", False)
        df.to_csv(str(path), **kwargs)

    def to_json(self, path: str | Path, **kwargs) -> None:
        """Export features to JSON (records orientation)."""
        df = self.features
        kwargs.setdefault("orient", "records")
        kwargs.setdefault("indent", 2)
        df.to_json(str(path), **kwargs)

    def feature_columns(self) -> list[str]:
        """Return the list of feature column names (excluding identifiers)."""
        df = self.features
        exclude = {"film", "year_film", "tmdb_id", "imdb_id", "nom", "nominee", "nominee_type"}
        return [c for c in df.columns if c not in exclude]
