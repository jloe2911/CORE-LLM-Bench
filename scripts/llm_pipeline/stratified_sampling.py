"""
stratified_sampling.py
---------------------
Stratified sampling of SPARQL questions for LLM pipeline.

Usage:
    python stratified_sampling.py input.csv output.csv
    python stratified_sampling.py input.csv output.csv --test-size 0.8 --random-state 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SAMPLING_COLUMNS = [
    "Task ID",
    "Task Type",
    "Size of ontology ABox",
    "Max Tag Length",
]


def add_sampling_columns(df):
    df = df[df["Task Type"].isin(["Membership", "Property Assertion"])].copy()
    df["Task ID temp"] = df["Task ID"].str.replace(
        r"(-BIN-.+-NEG-BIN|-(BIN|MC))$", "", regex=True
    )

    # Bin variables
    bin_edges = np.histogram_bin_edges(df["Size of ontology ABox"], bins="auto")
    df["Bin_Size of ontology ABox"] = pd.cut(
        df["Size of ontology ABox"], bins=bin_edges, labels=False, include_lowest=True
    )

    bin_edges = np.histogram_bin_edges(df["Max Tag Length"], bins="auto")
    df["Bin_Max Tag Length"] = pd.cut(
        df["Max Tag Length"], bins=bin_edges, labels=False, include_lowest=True
    )

    # Combine bins into a single stratification key
    df["strata"] = (
        df["Bin_Size of ontology ABox"].astype(str)
        + "_"
        + df["Bin_Max Tag Length"].astype(str)
    )
    return df


def select_train_groups(df, test_size, random_state):
    # Group by Task ID temp
    df_groups = df.groupby("Task ID temp").first().reset_index()
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
        df_filtered["Task ID temp"],
        train_size=adjusted_train_size,
        test_size=adjusted_test_size,
        stratify=df_filtered["strata"],
        random_state=random_state,
    )
    return set(train_groups)


def limit_rows_by_group(df, max_rows, random_state):
    if max_rows is None or len(df) <= max_rows:
        return df

    task_groups = df["Task ID"].str.replace(
        r"(-BIN-.+-NEG-BIN|-(BIN|MC))$", "", regex=True
    )
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
    input_file, output_file, train_groups, random_state, max_rows=None, chunksize=50000
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_chunks = []
    sampled_rows = 0
    print(f"Streaming selected rows to: {output_file}")

    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk = chunk[
            chunk["Task Type"].isin(["Membership", "Property Assertion"])
        ].copy()
        chunk["Task ID temp"] = chunk["Task ID"].str.replace(
            r"(-BIN-.+-NEG-BIN|-(BIN|MC))$", "", regex=True
        )
        selected = chunk[chunk["Task ID temp"].isin(train_groups)].copy()
        if selected.empty:
            continue

        selected = selected.drop(columns=["Task ID temp"])
        selected_chunks.append(selected)
        sampled_rows += len(selected)

    if selected_chunks:
        train_df = pd.concat(selected_chunks, ignore_index=True).drop_duplicates()
    else:
        train_df = pd.DataFrame()

    train_df = limit_rows_by_group(train_df, max_rows, random_state)

    print(f"Sampled rows: {len(train_df)}")
    print(f"Writing output: {output_file}")
    train_df.to_csv(output_file, index=False)
    print("Done.")
    return train_df


def stratified_sample(
    input_file, output_file, test_size=0.75, random_state=42, max_rows=None
):
    print(f"Reading input: {input_file}")
    df = pd.read_csv(input_file, usecols=SAMPLING_COLUMNS)
    print(f"Initial rows: {len(df)}")

    df = df.drop_duplicates()
    df = add_sampling_columns(df)
    train_groups = select_train_groups(df, test_size, random_state)
    return stream_sampled_rows(
        input_file, output_file, train_groups, random_state, max_rows=max_rows
    )


def main():
    parser = argparse.ArgumentParser(
        description="Perform stratified sampling on a SPARQL questions CSV file."
    )
    parser.add_argument("--input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to output CSV file")
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
    )


if __name__ == "__main__":
    main()
