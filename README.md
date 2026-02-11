# Film Club Knowledge Graph (FCKG)

This repository contains the core Film Club knowledge graph assets used for lightweight querying and demos.

## Core Assets

- Ontology schema: `movieontology.ttl`
- Canonical instance datasets under `data/instances/`
- Simple demo notebook: `fckg_demo_simply.ipynb`

## Instance Datasets

The repository includes the following canonical instance TTL files:

- `data/instances/oscar_nominations.ttl`
- `data/instances/bafta_nominations.ttl`
- `data/instances/sag_nominations.ttl`
- `data/instances/dga_nominations.ttl`
- `data/instances/pga_nominations.ttl`
- `data/instances/golden_globes_nominations.ttl`

## Notebook

Run `fckg_demo_simply.ipynb` to:

1. Load `movieontology.ttl` and `data/instances/*.ttl`
2. Execute example local SPARQL queries
3. See single-title API examples for TMDB, OMDb (queried by IMDb ID), and Wikidata

## License

MIT. See `LICENSE`.
