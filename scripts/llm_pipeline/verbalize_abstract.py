import argparse
from pathlib import Path
import pandas as pd
import re


SECTION_HEADERS = {
    "=== CLASSES ===",
    "=== OBJECT PROPERTIES ===",
    "=== DATA PROPERTIES ===",
    "=== INDIVIDUALS ===",
    "=== ONTOLOGY ABSTRACTION MAPPINGS ===",
}


def parse_mapping_file(mapping_file: Path) -> dict[str, str]:
    """
    Parse a mapping file of the form:
    OriginalName -> <http://...#Class1>

    Returns a dictionary like:
    {
        "Ancestor": "Class1",
        "hasParent": "Property10",
        ...
    }
    """
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file does not exist: {mapping_file}")

    mappings = {}

    with open(mapping_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line in SECTION_HEADERS:
                continue

            if "->" not in line:
                continue

            left, right = line.split("->", 1)
            original = left.strip()
            abstract_uri = right.strip()

            # Extract fragment after '#', e.g. <...#Class1> -> Class1
            match = re.search(r"#([^>]+)>?$", abstract_uri)
            if match:
                abstract_name = match.group(1).strip()
            else:
                # Fallback: remove angle brackets if fragment is missing
                abstract_name = abstract_uri.strip("<>").strip()

            if original:
                mappings[original] = abstract_name

    return mappings


def build_replacement_pattern(mapping_keys: list[str]) -> re.Pattern:
    """
    Build a regex that matches any mapping key.
    Sort by length descending so longer terms are replaced first.
    """
    sorted_keys = sorted(mapping_keys, key=len, reverse=True)
    escaped = [re.escape(k) for k in sorted_keys]
    return re.compile(r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)")


def abstract_question(text: str, mappings: dict[str, str], pattern: re.Pattern) -> str:
    """
    Replace ontology terms in a question using the abstraction mapping.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    def replacer(match: re.Match) -> str:
        original = match.group(1)
        return mappings.get(original, original)

    return pattern.sub(replacer, text)


def process_csv(
    input_file: Path,
    output_file: Path,
    mapping_file: Path,
    question_column: str = "Question",
    answer_column: str = "Answer",
) -> None:
    """
    Read input CSV, replace the Question column with its abstracted version,
    and write a new CSV.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_file}")

    df = pd.read_csv(input_file)

    if question_column not in df.columns:
        raise ValueError(
            f"Column '{question_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    mappings = parse_mapping_file(mapping_file)
    if not mappings:
        raise ValueError(f"No mappings found in mapping file: {mapping_file}")

    pattern = build_replacement_pattern(list(mappings.keys()))

    # Replace the question and, when present, open-ended gold answers so
    # abstract evaluation compares abstract labels against abstract labels.
    df[question_column] = df[question_column].apply(
        lambda q: abstract_question(q, mappings, pattern)
    )
    if answer_column in df.columns:
        answer_type = (
            df["Answer Type"].astype(str).str.upper()
            if "Answer Type" in df.columns
            else pd.Series("", index=df.index)
        )
        mc_mask = answer_type.isin(["MC", "MULTI CHOICE", "MULTICHOICE"])
        df.loc[mc_mask, answer_column] = df.loc[mc_mask, answer_column].apply(
            lambda answer: abstract_question(answer, mappings, pattern)
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Input CSV:    {input_file}")
    print(f"Mapping file: {mapping_file}")
    print(f"Output CSV:   {output_file}")
    print(f"Processed rows: {len(df)}")
    print(f"Replaced column: {question_column}")
    if answer_column in df.columns:
        print(f"Replaced MC answers in column: {answer_column}")
    print(f"Loaded mappings: {len(mappings)}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace the Question column in a CSV using an ontology abstraction mapping file."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--mapping-file",
        required=True,
        help="Path to the ontology abstraction mapping text file",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--question-column",
        default="SPARQL Query",
        help="Name of the question column (default: Question)",
    )
    parser.add_argument(
        "--answer-column",
        default="Answer",
        help="Name of the answer column to abstract for MC rows (default: Answer)",
    )

    args = parser.parse_args()

    process_csv(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        mapping_file=Path(args.mapping_file),
        question_column=args.question_column,
        answer_column=args.answer_column,
    )


if __name__ == "__main__":
    main()
