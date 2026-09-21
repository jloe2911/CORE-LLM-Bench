# Corrected explanation semantics for v1.1

OEQA explanations are disjunctive only within a single answer group. A
complete proof for an OEQA is the conjunction of one selected alternative for
every gold answer. The builder assigns proofs to answers exclusively through
the exact `inferred.object` provenance in the source `Explanations.json`; it
does not use token, substring, or fuzzy matching.

Each OEQA therefore carries `Answer Explanations`, an ordered list of records
with `Answer`, `Alternatives`, `Alternative Count`, minimum/maximum proof axiom
counts, and minimum/maximum primitive tag counts. Complete-answer-set metadata
enumerates all valid selections and reports minimum/maximum union axiom counts
and primitive tag counts. A shared ontology axiom is counted once in a selected
complete proof union. Complete-union metadata also reports the minimum and
maximum number of distinct non-direct tag types and whether `M` is always,
never, or selection-dependent across valid conjunctive selections.

The old `Minimum Explanation`, `Explanations`, `Explanation Count`,
`Explanation Min`, and `Explanation Max` fields are retained as flattened
legacy fields. They are insufficient for OEQA because they erase which answer
each proof establishes, and they must not be used for corrected scientific
complexity calculations.

The canonical tag taxonomy remains `D H T S A J N E ∩ ¬ I F V Y Q R C L U M`.
OWLAPI class expressions are traversed recursively and one axiom may contribute
multiple primitive tags. Repeated occurrences mean distinct axioms or nested
construct occurrences, not duplicate OWL/text serializations.

`M` means that the complete explanation contains at least two distinct
non-direct TBox reasoning types. It is categorical metadata about symbolic
heterogeneity, not a primitive inference operation. Consequently `M` is always
excluded from primitive tag counts. Repeating one type (for example `H H`) does
not trigger `M`, while `D H R` and a single axiom containing `Q ∩ E` do.
