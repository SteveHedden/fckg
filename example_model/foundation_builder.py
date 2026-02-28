#!/usr/bin/env python3
"""
foundation_builder.py
─────────────────────
Combines the FCKG ontology and all instance TTL files into a single validated
RDF graph, expands it with SHACL-AF inference rules and RDFS entailment, and
writes the result to a Turtle file.

What gets inferred
──────────────────
1. SHACL SPARQLRule (sh:advanced):
   For every Person who appears as msh:hasNominee on a Nomination, a direct
   shortcut triple is constructed:
       <Person> msh:nominatedFor <Film>
   This lets you query Person→Film links without traversing the Nomination
   reification.

2. RDFS entailment (inference='rdfs'):
   Uses rdfs:domain, rdfs:range, and rdfs:subClassOf axioms in the ontology
   to type-assert nodes from property usage. For example, if a node appears
   as the object of msh:hasNominee (range: msh:Person), it is asserted to be
   rdf:type msh:Person even if that triple wasn't explicit in the instance data.

Usage
─────
    python foundation_builder.py
    python foundation_builder.py --output /path/to/output.ttl

Requirements
────────────
    pip install rdflib pyshacl

Note: the output file will be large (~50–100 MB). It is gitignored by default.
"""

import argparse
from pathlib import Path

from rdflib import Graph, URIRef, BNode
from rdflib.collection import Collection
from pyshacl import validate


PROJECT_ROOT   = Path(__file__).resolve().parent.parent
ONTOLOGY_PATH  = PROJECT_ROOT / "movieontology.ttl"
INSTANCES_DIR  = PROJECT_ROOT / "data" / "instances"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "foundation" / "foundation.ttl"

SH_OR = URIRef("http://www.w3.org/ns/shacl#or")


def _patch_sh_or(shapes_graph: Graph) -> int:
    """Wrap bare sh:or <URI> values in proper RDF lists.

    The ontology uses DASH shortcuts like `sh:or dash:StringOrLangString`
    (a bare IRI) instead of the required RDF list `sh:or (dash:StringOrLangString)`.
    pyshacl is strict about this; patching in memory avoids modifying the source file.
    Returns the number of triples patched.
    """
    to_fix = [
        (s, o) for s, p, o in shapes_graph.triples((None, SH_OR, None))
        if isinstance(o, URIRef)
    ]
    for shape, obj in to_fix:
        shapes_graph.remove((shape, SH_OR, obj))
        list_node = BNode()
        Collection(shapes_graph, list_node, [obj])
        shapes_graph.add((shape, SH_OR, list_node))
    return len(to_fix)


def build_foundation(output_path: Path, verbose: bool = False) -> None:

    # ── 1. Load ontology (shapes + schema) ───────────────────────────────────
    print("Loading ontology...")
    shapes_graph = Graph()
    shapes_graph.parse(ONTOLOGY_PATH, format="turtle")
    print(f"  {len(shapes_graph):,} triples")

    # ── 2. Load all instance files ────────────────────────────────────────────
    print("\nLoading instance data...")
    data_graph = Graph()
    total_before = 0
    for ttl in sorted(INSTANCES_DIR.glob("*.ttl")):
        before = len(data_graph)
        data_graph.parse(ttl, format="turtle")
        print(f"  {ttl.name}: +{len(data_graph) - before:,} triples")
    print(f"\n  Instance total: {len(data_graph):,} triples")

    # ── 3. SHACL validation + inference ───────────────────────────────────────
    print("\nRunning SHACL validation and inference...")

    # The ontology uses DASH shortcuts (sh:or dash:StringOrLangString) written
    # as bare IRIs; patch them into proper RDF lists before passing to pyshacl.
    n_patched = _patch_sh_or(shapes_graph)
    if n_patched:
        print(f"  Patched {n_patched} sh:or bare-URI shortcuts for pyshacl compatibility")

    before_inference = len(data_graph)

    conforms, results_graph, results_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        advanced=True,    # fires sh:SPARQLRule → adds msh:nominatedFor triples
        inference="rdfs", # RDFS entailment from domain/range/subClassOf axioms
        inplace=True,     # inferred triples are added directly to data_graph
        js=False,
    )

    inferred = len(data_graph) - before_inference
    print(f"  Inference added {inferred:,} triples")

    if conforms:
        print("  Validation: PASSED")
    else:
        # Summarise violations by source shape rather than dumping all results.
        # (The most common issue is hasIdentifier pointing to raw URL IRIs instead
        # of typed msh:Identifier nodes — a known design choice in the instance data.)
        SH = URIRef("http://www.w3.org/ns/shacl#")
        SH_result      = URIRef("http://www.w3.org/ns/shacl#result")
        SH_sourceShape = URIRef("http://www.w3.org/ns/shacl#sourceShape")
        from collections import Counter
        counts = Counter(
            str(results_graph.value(r, SH_sourceShape))
            for r in results_graph.objects(None, SH_result)
        )
        total = sum(counts.values())
        print(f"  Validation: {total} violations (use --verbose to see full report)")
        for shape, n in counts.most_common(10):
            shape_name = shape.split("#")[-1] if "#" in shape else shape.split("/")[-1]
            print(f"    {n:>5}  {shape_name}")
        if verbose:
            print(results_text)

    # ── 4. Combine data + ontology into one graph ─────────────────────────────
    combined = data_graph + shapes_graph
    print(f"\n  Combined graph: {len(combined):,} triples")

    # ── 5. Write output ────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {output_path} ...")
    combined.serialize(destination=str(output_path), format="turtle")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the FCKG data foundation with SHACL validation and inference."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output TTL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full SHACL validation report when violations are found.",
    )
    args = parser.parse_args()
    build_foundation(args.output, verbose=args.verbose)
