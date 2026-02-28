
#!/usr/bin/env python3
"""
enrichment_runner.py
────────────────────
Enriches the FCKG data foundation with metadata from TMDB, OMDb, and Wikidata.
For each film in the graph, it extracts external IDs from msh:hasIdentifier,
calls the relevant APIs, caches the raw JSON responses, and writes four additive
RDF overlay files (Turtle format) into data/enriched/.

Output files (written to data/enriched/)
─────────────────────────────────────────
  film_metadata.ttl    — budget, revenue, runtime, genres, ratings, release info
  film_credits.ttl     — director, cast, and crew links to Person nodes
  person_metadata.ttl  — birth year, biography, known-for titles
  person_identifiers.ttl — TMDB and Wikidata IDs for Person nodes

These files are additive overlays: they reference the same Film and Person URIs
defined in data/instances/ but do not modify the foundation files.

API keys required
─────────────────
Copy .env.example to .env and fill in your keys:
  TMDB_API_KEY    — https://www.themoviedb.org/settings/api  (free)
  OMDB_API_KEY    — https://www.omdbapi.com/apikey.aspx      (free tier: 1000/day)
  Wikidata uses the public SPARQL endpoint — no key required.

Usage
─────
  # First time: enrich all films
  python enrichment_runner.py

  # Enrich only a specific release year
  python enrichment_runner.py --year 2024

  # Rebuild TTLs from cached API responses (no new API calls)
  python enrichment_runner.py --cache-only

  # Preview what would be processed without making any calls or writes
  python enrichment_runner.py --dry-run

  # Re-fetch everything, ignoring the cache
  python enrichment_runner.py --force

Requirements
────────────
  pip install rdflib pyshacl python-dotenv requests

Note: API calls are throttled automatically. Enriching all ~6,300 films for the
first time takes roughly 2–3 hours depending on API tier limits.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure example_model/ is on the path so the local foundation/ and enrichment/
# packages are found before any system-level install.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from enrichment.pipeline import run_enrichment_pipeline

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich FCKG films from TMDB / OMDb / Wikidata."
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Only process films with this release year.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cache and re-fetch all API responses.",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Regenerate TTL overlays from cached responses without making API calls.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed; make no API calls or file writes.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    summary = run_enrichment_pipeline(
        year=args.year,
        force=args.force,
        cache_only=args.cache_only,
        dry_run=args.dry_run,
    )

    print(
        f"films_seen={summary.films_seen}  processed={summary.films_processed}  "
        f"api_fetched={summary.api_fetched}  cache_hits={summary.cache_hits}  "
        f"errors={summary.errors}"
    )
    if not args.dry_run:
        print(f"\nOutput files (data/enriched/):")
        print(f"  film_metadata.ttl        — {summary.metadata_films_enriched} films")
        print(f"  film_credits.ttl         — {summary.credits_films_enriched} films")
        print(f"  person_metadata.ttl      — {summary.people_enriched} persons")
        print(
            f"  person_identifiers.ttl   — "
            f"tmdb={summary.person_tmdb_identifiers_enriched}  "
            f"wikidata={summary.person_wikidata_identifiers_enriched}"
        )


if __name__ == "__main__":
    main()
