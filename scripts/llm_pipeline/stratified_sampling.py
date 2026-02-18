"""
stratified_sampling.py
---------------------
Stratified sampling of SPARQL questions for LLM pipeline.
- Input: SPARQL_questions.csv
- Output: SPARQL_questions_sampling.csv
- Can be run from IDE (main function)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# Navigate to project root (2 levels up from current script)
script_dir = Path(__file__).resolve().parent  # scripts/llm_pipeline/
project_root = script_dir.parent.parent  # owl-inference-explainer/
os.chdir(project_root)

print(f"Working directory set to: {os.getcwd()}")


def stratified_sample(input_csv, output_csv, test_size=0.95, random_state=42):
    print(f"Reading input: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Initial rows: {len(df)}")

    df = df.drop_duplicates()
    df = df[df["Task Type"].isin(["Membership", "Property Assertion"])]
    df.loc[:, "Task ID temp"] = df["Task ID"].str.replace(r"-(BIN|MC)$", "", regex=True)

    # Bin variables
    bin_edges = np.histogram_bin_edges(df["Size of ontology ABox"], bins="auto")
    df["Bin_Size of ontology ABox"] = pd.cut(
        df["Size of ontology ABox"], bins=bin_edges, labels=False
    )
    bin_edges = np.histogram_bin_edges(df["Max Tag Length"], bins="auto")
    df["Bin_Max Tag Length"] = pd.cut(
        df["Max Tag Length"], bins=bin_edges, labels=False
    )

    # Combine bins into a single stratification key
    df["strata"] = (
        df["Bin_Size of ontology ABox"].astype(str)
        + "_"
        + df["Bin_Max Tag Length"].astype(str)
    )

    # Group by Task ID temp
    df_groups = df.groupby("Task ID temp").first().reset_index()
    strata_counts = df_groups["strata"].value_counts()
    valid_strata = strata_counts[strata_counts >= 2].index
    df_filtered = df_groups[df_groups["strata"].isin(valid_strata)]

    # Stratified split at group level
    train_groups, test_groups = train_test_split(
        df_filtered["Task ID temp"],
        test_size=test_size,
        stratify=df_filtered["strata"],
        random_state=random_state,
    )

    # Assign split back to original df rows based on group membership
    df["split"] = "test"
    df.loc[df["Task ID temp"].isin(train_groups), "split"] = "train"

    train_df = df[df["split"] == "train"]

    # Drop the temporary columns
    columns_to_drop = [
        "Task ID temp",
        "Bin_Size of ontology ABox",
        "Bin_Max Tag Length",
        "strata",
        "split",
    ]
    train_df = train_df.drop(columns=columns_to_drop)

    print(f"Sampled rows: {len(train_df)}")
    print(f"Writing output: {output_csv}")
    train_df.to_csv(output_csv, index=False)
    print("Done.")
    return train_df


def main():
    # Default file locations (can be changed as needed)
    input_csv = "output/OWL2Bench/1hop/SPARQL_questions_with_tags.csv"
    output_csv = "output/OWL2Bench/1hop/SPARQL_questions_sampling2.csv"
    stratified_sample(input_csv, output_csv)


if __name__ == "__main__":
    main()
