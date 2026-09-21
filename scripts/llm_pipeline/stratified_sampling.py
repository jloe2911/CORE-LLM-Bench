"""
stratified_sampling.py
---------------------
Stratified sampling of SPARQL questions for LLM pipeline.

Usage:
    python stratified_sampling.py input.csv output.csv
    python stratified_sampling.py input.csv output.csv --test-size 0.8 --random-state 42
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .benchmark_corrections import (
        add_group_keys,
        assert_no_broken_source_pairs,
        assert_owl2bench_membership_pairs,
        audit_bqa_pairing,
        canonical_dataset_name,
        prepare_eligible_rows,
        print_domainconcept_report,
        print_pairing_report,
    )
except ImportError:  # Direct script execution.
    from benchmark_corrections import (
        add_group_keys,
        assert_no_broken_source_pairs,
        assert_owl2bench_membership_pairs,
        audit_bqa_pairing,
        canonical_dataset_name,
        prepare_eligible_rows,
        print_domainconcept_report,
        print_pairing_report,
    )


SAMPLING_COLUMNS = [
    "Task ID",
    "Root Entity",
    "Task Type",
    "Answer Type",
    "SPARQL Query",
    "Answer",
    "Size of ontology ABox",
    "Max Tag Length",
]

CORRECTED_COMPLEXITY_COLUMN = "Corrected Complexity"
COMPLEXITY_BIN_COLUMN = "Complexity Bin"
SAMPLING_MODE_COLUMN = "Sampling Mode"
TASK_CLASS_COLUMN = "Sampling Task"


def final_complexity_bin(task: str, complexity: int) -> str:
    """Return the scientifically frozen v1.1 task-specific complexity bin."""

    value = int(complexity)
    if value < 1:
        raise ValueError(f"Complexity must be a positive integer, got {value}")
    task = str(task).upper()
    if task == "BQA":
        return "Low" if value == 1 else "Medium" if value == 2 else "High"
    if task == "OEQA":
        return "Low" if value <= 3 else "Medium" if value <= 5 else "High"
    raise ValueError(f"Unsupported task class: {task!r}")


def _task_class(answer_type: object) -> str:
    return "BQA" if str(answer_type).upper() == "BIN" else "OEQA"


def _stable_group_text(value: object) -> str:
    if isinstance(value, tuple):
        value = list(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def build_corrected_sampling_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Build atomic v1.1 sampling groups with raw and categorical complexity.

    The caller must attach Phase 4's corrected integer complexity to every row.
    A BQA TRUE/FALSE pair is one group and is rejected if its members disagree
    on task class or corrected complexity.
    """

    required = {
        "Sampling Group Key",
        "Answer Type",
        "Size of ontology ABox",
        CORRECTED_COMPLEXITY_COLUMN,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing corrected sampling columns: {missing}")

    working = df.copy()
    working[TASK_CLASS_COLUMN] = working["Answer Type"].map(_task_class)
    consistency = working.groupby("Sampling Group Key", sort=False).agg(
        task_classes=(TASK_CLASS_COLUMN, "nunique"),
        complexities=(CORRECTED_COMPLEXITY_COLUMN, "nunique"),
    )
    invalid = consistency[
        (consistency["task_classes"] != 1)
        | (consistency["complexities"] != 1)
    ]
    if not invalid.empty:
        raise ValueError(
            "Sampling groups disagree on task class or corrected complexity: "
            f"{list(invalid.index[:5])}"
        )

    groups = (
        working.groupby("Sampling Group Key", sort=False, as_index=False)
        .agg(
            {
                "Size of ontology ABox": "max",
                CORRECTED_COMPLEXITY_COLUMN: "first",
                TASK_CLASS_COLUMN: "first",
            }
        )
        .copy()
    )
    groups[CORRECTED_COMPLEXITY_COLUMN] = groups[
        CORRECTED_COMPLEXITY_COLUMN
    ].astype(int)
    groups["Size of ontology ABox"] = pd.to_numeric(
        groups["Size of ontology ABox"], errors="raise"
    )
    groups[COMPLEXITY_BIN_COLUMN] = [
        final_complexity_bin(task, complexity)
        for task, complexity in zip(
            groups[TASK_CLASS_COLUMN],
            groups[CORRECTED_COMPLEXITY_COLUMN],
            strict=True,
        )
    ]

    # Preserve the benchmark's existing deterministic NumPy "auto" ABox
    # discretization across the complete dataset/hop pool. Task remains an
    # explicit component of the stratum; it does not redefine the ABox edges.
    values = groups["Size of ontology ABox"]
    edges = np.histogram_bin_edges(values, bins="auto")
    groups["ABox Bin"] = pd.cut(
        values,
        bins=edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    ).astype(int)
    groups["Combined Stratum"] = [
        f"{task}|abox={abox}|complexity={complexity}"
        for task, abox, complexity in zip(
            groups[TASK_CLASS_COLUMN],
            groups["ABox Bin"],
            groups[COMPLEXITY_BIN_COLUMN],
            strict=True,
        )
    ]
    return groups


def select_corrected_groups(
    df: pd.DataFrame,
    *,
    benchmark_fraction: float = 0.25,
    random_state: int = 42,
) -> tuple[set[object], pd.DataFrame]:
    """Select v1.1 groups without dropping sparse strata.

    Combined ABox/complexity strata with at least two groups are normal.
    Singleton combined strata are pooled only with other fallback candidates
    in the same task/complexity bin.  A remaining singleton is assigned by a
    stable SHA-256 order so the final partition count is closest to the
    requested fraction.  Exact half-count ties round toward the benchmark.
    """

    if not 0 < benchmark_fraction < 1:
        raise ValueError("benchmark_fraction must be between zero and one")
    groups = build_corrected_sampling_groups(df)
    decisions: list[pd.DataFrame] = []
    selected: set[object] = set()

    for task, partition in groups.groupby(TASK_CLASS_COLUMN, sort=True):
        partition = partition.copy()
        combined_counts = partition["Combined Stratum"].value_counts()
        is_normal = partition["Combined Stratum"].map(combined_counts).ge(2)
        partition.loc[is_normal, SAMPLING_MODE_COLUMN] = "normal-stratified"

        fallback = partition.loc[~is_normal].copy()
        fallback_counts = fallback[COMPLEXITY_BIN_COLUMN].value_counts()
        fallback_supported = fallback[COMPLEXITY_BIN_COLUMN].map(
            fallback_counts
        ).ge(2)
        fallback.loc[
            fallback_supported, SAMPLING_MODE_COLUMN
        ] = "complexity-fallback"
        fallback.loc[
            ~fallback_supported, SAMPLING_MODE_COLUMN
        ] = "deterministic-singleton-fallback"
        partition.loc[fallback.index, SAMPLING_MODE_COLUMN] = fallback[
            SAMPLING_MODE_COLUMN
        ]

        partition["Effective Stratum"] = partition["Combined Stratum"]
        supported_fallback = partition[SAMPLING_MODE_COLUMN].eq(
            "complexity-fallback"
        )
        partition.loc[supported_fallback, "Effective Stratum"] = [
            f"{task}|complexity={value}|fallback"
            for value in partition.loc[
                supported_fallback, COMPLEXITY_BIN_COLUMN
            ]
        ]
        singleton_mask = partition[SAMPLING_MODE_COLUMN].eq(
            "deterministic-singleton-fallback"
        )
        supported = partition.loc[~singleton_mask].copy()
        singletons = partition.loc[singleton_mask].copy()

        supported_selected: set[object] = set()
        if not supported.empty:
            supported = supported.assign(
                _stable_key=supported["Sampling Group Key"].map(
                    _stable_group_text
                )
            ).sort_values("_stable_key", kind="stable")
            class_count = supported["Effective Stratum"].nunique()
            requested = _half_up(benchmark_fraction * len(supported))
            selected_count = min(
                max(requested, class_count), len(supported) - class_count
            )
            if selected_count <= 0:
                raise ValueError(
                    f"Stratified selection is impossible for {task}: "
                    f"{len(supported)} groups and {class_count} strata"
                )
            chosen, _ = train_test_split(
                supported["Sampling Group Key"],
                train_size=selected_count,
                test_size=len(supported) - selected_count,
                stratify=supported["Effective Stratum"],
                random_state=random_state,
            )
            supported_selected = set(chosen)

        target = _half_up(benchmark_fraction * len(partition))
        needed_singletons = min(
            max(target - len(supported_selected), 0), len(singletons)
        )
        singleton_selected: set[object] = set()
        if needed_singletons:
            ranked = sorted(
                singletons["Sampling Group Key"],
                key=lambda key: (
                    hashlib.sha256(
                        f"{random_state}|{_stable_group_text(key)}".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    _stable_group_text(key),
                ),
            )
            singleton_selected = set(ranked[:needed_singletons])

        partition_selected = supported_selected | singleton_selected
        partition["Selected"] = partition["Sampling Group Key"].isin(
            partition_selected
        )
        partition["Requested Benchmark Fraction"] = benchmark_fraction
        partition["Requested Partition Group Count"] = target
        decisions.append(partition.drop(columns=["_stable_key"], errors="ignore"))
        selected.update(partition_selected)

    decision_frame = pd.concat(decisions, ignore_index=True)
    if len(decision_frame) != len(groups):
        raise AssertionError("Every eligible group must receive one decision")
    return selected, decision_frame


def build_sampling_groups(df):
    """Build one stratification row per sampling unit.

    If rows in a BQA pair disagree, the group uses the maximum ABox size and
    maximum explanation complexity (``Max Tag Length``).  This deterministic,
    conservative rule prevents either side from understating group complexity.
    """

    df_groups = (
        df.groupby("Sampling Group Key", sort=False, as_index=False)
        .agg(
            {
                "Size of ontology ABox": "max",
                "Max Tag Length": "max",
            }
        )
        .copy()
    )

    bin_edges = np.histogram_bin_edges(
        df_groups["Size of ontology ABox"], bins="auto"
    )
    df_groups["Bin_Size of ontology ABox"] = pd.cut(
        df_groups["Size of ontology ABox"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    bin_edges = np.histogram_bin_edges(df_groups["Max Tag Length"], bins="auto")
    df_groups["Bin_Max Tag Length"] = pd.cut(
        df_groups["Max Tag Length"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    # Combine bins into a single stratification key
    df_groups["strata"] = (
        df_groups["Bin_Size of ontology ABox"].astype(str)
        + "_"
        + df_groups["Bin_Max Tag Length"].astype(str)
    )
    return df_groups


def select_train_groups(df, test_size, random_state):
    df_groups = build_sampling_groups(df)
    strata_counts = df_groups["strata"].value_counts()
    valid_strata = strata_counts[strata_counts >= 2].index
    df_filtered = df_groups[df_groups["strata"].isin(valid_strata)].copy()

    n_groups = len(df_filtered)
    n_classes = df_filtered["strata"].nunique()

    print(f"Groups after filtering: {n_groups}")
    print(f"Number of strata: {n_classes}")

    if n_groups == 0:
        raise ValueError(
            "No valid groups remain after filtering strata with at least 2 samples."
        )

    if n_groups < 2 * n_classes:
        raise ValueError(
            f"Stratified split impossible: {n_groups} groups for {n_classes} strata. "
            f"Need at least {2 * n_classes} groups total."
        )

    requested_test_size = int(round(test_size * n_groups))

    # Ensure both splits can contain all strata
    min_test_size = n_classes
    max_test_size = n_groups - n_classes
    adjusted_test_size = min(max(requested_test_size, min_test_size), max_test_size)
    adjusted_train_size = n_groups - adjusted_test_size

    if adjusted_test_size != requested_test_size:
        print(
            f"Adjusted test size from {requested_test_size} to {adjusted_test_size} "
            f"so both splits contain at least one sample per stratum."
        )

    train_groups, _ = train_test_split(
        df_filtered["Sampling Group Key"],
        train_size=adjusted_train_size,
        test_size=adjusted_test_size,
        stratify=df_filtered["strata"],
        random_state=random_state,
    )
    return set(train_groups)


def limit_rows_by_group(df, max_rows, random_state):
    if max_rows is None or len(df) <= max_rows:
        return df

    task_groups = df["Sampling Group Key"]
    group_sizes = task_groups.value_counts()
    rng = np.random.default_rng(random_state)
    shuffled_groups = rng.permutation(group_sizes.index.to_numpy())

    selected_groups = []
    selected_rows = 0
    for group in shuffled_groups:
        group_size = group_sizes[group]
        if selected_rows and selected_rows + group_size > max_rows:
            continue
        if group_size > max_rows:
            continue

        selected_groups.append(group)
        selected_rows += group_size
        if selected_rows == max_rows:
            break

    limited_df = df[task_groups.isin(selected_groups)].copy()
    print(
        f"Limiting sampled rows from {len(df)} to {len(limited_df)} "
        f"while preserving {len(selected_groups)} task groups."
    )
    return limited_df


def stream_sampled_rows(
    input_file,
    output_file,
    train_groups,
    random_state,
    dataset,
    hop,
    eligible_source,
    max_rows=None,
    chunksize=50000,
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_chunks = []
    sampled_rows = 0
    print(f"Streaming selected rows to: {output_file}")

    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk, _ = prepare_eligible_rows(chunk, dataset, hop)
        chunk = add_group_keys(chunk, dataset, hop)
        selected = chunk[chunk["Sampling Group Key"].isin(train_groups)].copy()
        if selected.empty:
            continue

        selected_chunks.append(selected)
        sampled_rows += len(selected)

    if selected_chunks:
        train_df = pd.concat(selected_chunks, ignore_index=True).drop_duplicates()
    else:
        train_df = pd.read_csv(input_file, nrows=0)
        train_df["Sampling Group Key"] = pd.Series(dtype=object)

    train_df = limit_rows_by_group(train_df, max_rows, random_state)

    pairing_report = audit_bqa_pairing(
        eligible_source, train_df, dataset=dataset, hop=hop
    )
    print_pairing_report(pairing_report)
    assert_no_broken_source_pairs(pairing_report)
    if canonical_dataset_name(dataset) == "OWL2Bench":
        assert_owl2bench_membership_pairs(pairing_report)

    train_df = train_df.drop(columns=["Sampling Group Key"], errors="ignore")

    print(f"Sampled rows: {len(train_df)}")
    print(f"Writing output: {output_file}")
    train_df.to_csv(output_file, index=False)
    print("Done.")
    return train_df


def stratified_sample(
    input_file,
    output_file,
    test_size=0.75,
    random_state=42,
    max_rows=None,
    dataset=None,
    hop=None,
):
    print(f"Reading input: {input_file}")
    input_path = Path(input_file)
    dataset = dataset or input_path.parent.parent.name
    hop = hop or input_path.parent.name
    df = pd.read_csv(input_file, usecols=SAMPLING_COLUMNS)
    print(f"Initial rows: {len(df)}")

    df = df.drop_duplicates()
    df, filter_report = prepare_eligible_rows(df, dataset, hop)
    print_domainconcept_report(filter_report)
    df = add_group_keys(df, dataset, hop)
    train_groups = select_train_groups(df, test_size, random_state)
    return stream_sampled_rows(
        input_file,
        output_file,
        train_groups,
        random_state,
        dataset,
        hop,
        df.drop(columns=["Sampling Group Key"]),
        max_rows=max_rows,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Perform stratified sampling on a SPARQL questions CSV file."
    )
    parser.add_argument("--input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to output CSV file")
    parser.add_argument(
        "--dataset",
        help=(
            "Dataset name for structural grouping/filtering; inferred from "
            "input path if omitted."
        ),
    )
    parser.add_argument(
        "--hop",
        choices=("1hop", "2hop"),
        help="Hop for structural grouping; inferred from input path if omitted.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.75,
        help="Fraction assigned to test split (default: 0.75)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional hard cap on sampled output rows.",
    )

    args = parser.parse_args()

    stratified_sample(
        input_file=args.input_file,
        output_file=args.output_file,
        test_size=args.test_size,
        random_state=args.random_state,
        max_rows=args.max_rows,
        dataset=args.dataset,
        hop=args.hop,
    )


if __name__ == "__main__":
    main()
