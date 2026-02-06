# fckg

Core FilmClub knowledge graph assets.

## Contents
- `movieontology.ttl` — core ontology for the FilmClub KG
- `oscar_nominated_movies.ttl` — reference dataset of Oscar-nominated movies

## Usage (TopBraid EDG)
1. Create or open an EDG project.
2. Import `movieontology.ttl` as the ontology.
3. Import `oscar_nominated_movies.ttl` as reference data.

## Notes
- This repo is intended to be the canonical home for KG reference assets.
- Downstream projects (e.g., FilmClub) can consume these files as a submodule or by syncing releases.
