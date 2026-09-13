# Local Family/FHKB reconstruction

## Scope and rights

The Family portion of CORE-LLM-Bench is a modified/adapted version of the Family
History Knowledge Base (FHKB). The *Manchester Family History Advanced OWL
Tutorial* and its downloadable FHKB resources are licensed CC BY-SA 3.0 and
credit Robert Stevens, Margaret Stevens, Nicolas Matentzoglu, and Simon Jupp.
The authoritative upstream pages are the
[tutorial and resource page](https://oboacademy.github.io/obook/tutorial/fhkb/),
Robert Stevens's [ontology page](https://www.cs.man.ac.uk/~stevensr/menupages/ontologies.php),
and [FHKB materials page](https://www.cs.man.ac.uk/~stevensr/menupages/fhkb.php).
The ontology page links the source artifact as `family.rdf.owl`.

Redistribution and adaptation are permitted under CC BY-SA 3.0, subject to its
attribution, license-link or license-copy, change-identification, ShareAlike,
and no-additional-restrictions requirements. CORE's Family/FHKB-derived
ontology, fragments, contexts, and conservatively source-derived benchmark
records remain under those applicable terms and are not relicensed as CC BY
4.0. See `NOTICE.md` for attribution and the recorded modifications. This
procedure is an additional reproducibility path, not a licensing workaround.

## Reconstruct locally

The repository pipeline parses the supplied ontology, extracts deterministic
individual-centred 1-hop and 2-hop Turtle subgraphs, uses the Java reasoner to
generate questions and proof explanations, stratifies questions, constructs
entity-abstracted views, verbalizes ontology contexts, generates NL questions,
and combines aligned NL, FS, and AR fields into the final JSON files.

1. Obtain `family.rdf.owl` from the authoritative page and record its hash.
2. Save it outside the repository, or under ignored `local_sources/`.
3. Install Java 17, Maven, and the Python dependencies used by the generation
   scripts. Configure the OpenAI credentials required by the existing NL
   generation stage; this stage sends generation inputs to the selected model
   provider and can incur cost.
4. From the repository root, run:

```console
python scripts/run_final_benchmark_pipeline.py --input-owl local_sources/family.rdf.owl --dataset FamilyOWL --hops 1hop 2hop
python scripts/validate_family_reconstruction.py --input-dir final_benchmark
```

The expected validation totals are:

| Output | Instances |
| --- | ---: |
| `FamilyOWL_1hop.json` | 1,880 |
| `FamilyOWL_2hop.json` | 1,880 |
| Total | 3,760 |
| BQA | 2,544 |
| OEQA | 1,216 |

## Reproducibility qualification

Technical reconstruction is possible, but exact byte-for-byte reproduction is
not established from the upstream download alone. The release source is a
transformed/adapted FHKB artifact and is not byte-identical to the authoritative
download identified during the release audit. The NL generation stage is also
model-dependent. Record the source hash, model identifier, and all generated
intermediate manifests. Treat a count mismatch as a provenance or
pipeline-version failure; do not edit or subsample outputs merely to reach the
expected totals.
