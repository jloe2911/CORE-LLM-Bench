# Licensing and provenance notice

The repository's software/code is offered under the MIT License in `LICENSE`. Original CORE-LLM-Bench benchmark questions and metadata created by the authors are intended for release under CC BY 4.0. Neither license replaces licenses or rights attached to third-party ontologies or source-derived content.

The manuscript-frozen SAGE-QA answer evaluator is vendored unmodified under
`scripts/llm_pipeline/vendor/` from SAGE-QA commit
`dbdbb50708bdc6c686ef82518ec71c1d1bf55985`. It is redistributed under its
MIT license, preserved as `scripts/llm_pipeline/vendor/SAGE-QA-LICENSE.txt`.

## Source ontologies

| Material | Provenance found in this audit | Release treatment |
| --- | --- | --- |
| Family / FamilyOWL | CORE's `data/input/family.owl` is a transformed and adapted version of the Family History Knowledge Base (FHKB) distributed with the *Manchester Family History Advanced OWL Tutorial*. The tutorial and its downloadable FHKB OWL resources are licensed CC BY-SA 3.0. Authors: Robert Stevens, Margaret Stevens, Nicolas Matentzoglu, and Simon Jupp. | Family/FHKB-derived ontology content, fragments, contexts, and source-derived benchmark material are distributed under the applicable [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) terms. They are not relicensed as CC BY 4.0. |
| Pizza 100 / Pizza 250 | Both variants derive from the University of Manchester Pizza ontology. The checked-in ontology declares Creative Commons Attribution 3.0 and names Alan Rector, Chris Wroe, Matthew Horridge, Nick Drummond, and Robert Stevens as contributors. | Retain attribution and the CC BY 3.0 notice. The benchmark variants add generated individuals and metadata; the upstream attribution still applies. |
| OWL2Bench | The official `kracr/owl2bench` repository identifies Apache-2.0 as its license. CORE-LLM-Bench uses the OWL2DL-1 generated ontology. | Preserve OWL2Bench attribution and Apache-2.0 terms/notices when redistributing this source material. |
| CORE-LLM-Bench questions and metadata | Original benchmark material created by the CORE-LLM-Bench authors, including natural-language and abstract representations generated through the benchmark pipeline. | CC BY 4.0 where separable from source-derived material. Family records that reproduce, transform, or build on FHKB content remain subject to CC BY-SA 3.0. This does not assert copyright in bare facts or independently authored expression where applicable law says otherwise. |

## Important scope note

`final_benchmark/*.zip` embeds source-derived ontology contexts. Redistribution is permitted only under the applicable component terms: Family/FHKB-derived material under CC BY-SA 3.0, Pizza-derived material under CC BY 3.0, OWL2Bench-derived material under Apache-2.0, separable original CORE benchmark material under CC BY 4.0, and software under MIT. The repository is not uniformly MIT or CC BY 4.0. This notice is a provenance record and not legal advice.

## Family/FHKB attribution and modifications

CORE-LLM-Bench uses an adapted FHKB artifact rather than a byte-identical copy.
The source work is the *Manchester Family History Advanced OWL Tutorial* and
its FHKB OWL resources by Robert Stevens, Margaret Stevens, Nicolas
Matentzoglu, and Simon Jupp.

- Tutorial, license notice, and FHKB downloads:
  `https://oboacademy.github.io/obook/tutorial/fhkb/`
- Robert Stevens's official FHKB materials page:
  `https://www.cs.man.ac.uk/~stevensr/menupages/fhkb.php`
- License: Creative Commons Attribution-ShareAlike 3.0 Unported:
  `https://creativecommons.org/licenses/by-sa/3.0/`

Observable CORE modifications include re-serialization with OWL API 4.5.26,
changing the entity namespace from the CO-ODE family-tree namespace to
`http://www.example.com/genealogy.owl#`, selecting and reorganizing a narrower
TBox, treating six upstream data-style fields as annotation properties, and
generating individual-centred ontology fragments, benchmark contexts,
questions, answers, abstractions, verbalizations, and metadata. The original
FHKB ontology IRI and descriptive comment are retained. Family-derived
material is distributed subject to the applicable CC BY-SA 3.0 attribution,
change-identification, ShareAlike, license-link, and no-additional-restrictions
requirements. Attribution does not imply endorsement by the FHKB authors or
the University of Manchester.

Upstream references:

- Family ontology IRI: `http://www.co-ode.org/roberts/family-tree.owl`
- Robert Stevens's Family ontology page: `https://www.cs.man.ac.uk/~stevensr/menupages/ontologies.php`
- Robert Stevens's FHKB tutorial/materials page: `https://www.cs.man.ac.uk/~stevensr/menupages/fhkb.php`
- Manchester/OBO Academy FHKB tutorial: `https://oboacademy.github.io/obook/tutorial/fhkb/`
- Stevens and Stevens (2008), *A Family History Knowledge Base Using OWL 2*: `https://ceur-ws.org/Vol-432/owled2008eu_submission_29.pdf`
- Pizza ontology: `http://www.co-ode.org/ontologies/pizza/`, version 2.0.0; embedded license `CC BY 3.0`
- OWL2Bench: `https://github.com/kracr/owl2bench`, Apache-2.0
