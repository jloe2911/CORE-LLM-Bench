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
    "=== TEXT ALIASES ===",
}


def deduplicate_sentences(descriptions) -> list[str]:
    """Keep each exact rendered sentence once, preserving first occurrence."""
    output = []
    seen = set()
    for description in descriptions:
        sentences = re.findall(r".*?[.!?](?=\s|$)|.+$", str(description).strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                output.append(sentence)
    return output


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

    with open(mapping_file, "r", encoding="utf-8") as f:
        return parse_mapping_lines(f)


def parse_mapping_lines(lines) -> dict[str, str]:
    mappings = {}
    for raw_line in lines:
        line = raw_line.strip()

        if not line or line in SECTION_HEADERS:
            continue

        if "->" not in line:
            continue

        left, right = line.split("->", 1)
        original = left.strip()
        if original.startswith("@alias "):
            original = original[len("@alias ") :].strip()
        elif original.startswith("<") and original.endswith(">"):
            # Full URI mappings are consumed separately and resolved via
            # formal-query identity. Only explicit safe aliases belong in
            # the global textual replacement table.
            continue
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


def parse_uri_mapping_file(mapping_file: Path) -> dict[str, str]:
    """Return full original URI to abstract local-name mappings."""
    with open(mapping_file, "r", encoding="utf-8") as handle:
        return parse_uri_mapping_lines(handle)


def parse_uri_mapping_lines(lines) -> dict[str, str]:
    mappings = {}
    for raw_line in lines:
        line = raw_line.strip()
        if "->" not in line or line.startswith("@alias "):
            continue
        left, right = (part.strip() for part in line.split("->", 1))
        if not (left.startswith("<") and left.endswith(">")):
            continue
        match = re.search(r"#([^>]+)>?$", right)
        if match:
            mappings[left[1:-1]] = match.group(1).strip()
    return mappings


def query_text_mappings(query: str, uri_mappings: dict[str, str]) -> dict[str, str]:
    """Derive display aliases only for entities identified by this query."""
    result = {}
    for uri in re.findall(r"<([^>]+)>", str(query or "")):
        target = uri_mappings.get(uri)
        if not target:
            # Generated questions can use a synthetic namespace while retaining
            # the ontology entity's local name.
            local = uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            matches = {
                mapped
                for original, mapped in uri_mappings.items()
                if original.rsplit("#", 1)[-1].rsplit("/", 1)[-1] == local
            }
            if len(matches) != 1:
                continue
            target = next(iter(matches))
        else:
            local = uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        display_bases = {local, re.sub(r"_dynamic_\d+$", "", local, flags=re.I)}
        for display_base in display_bases:
            suffix_pattern = r"_\d{4}$|_\d+$|_v\d+$"
            if target.startswith("Individual"):
                suffix_pattern += r"|_\w{2,3}$"
            cleaned = re.sub(suffix_pattern, "", display_base)
            cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
            cleaned = cleaned.replace("_", " ").replace("-", " ")
            cleaned = " ".join(word.capitalize() for word in cleaned.split())
            for alias in (display_base, cleaned):
                if alias:
                    result[alias] = target
    return result


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
    uri_mappings = parse_uri_mapping_file(mapping_file)
    if not mappings:
        raise ValueError(f"No mappings found in mapping file: {mapping_file}")

    def abstract_row_value(row, column):
        row_mappings = dict(mappings)
        if "SPARQL Query" in row:
            row_mappings.update(query_text_mappings(row["SPARQL Query"], uri_mappings))
        pattern = build_replacement_pattern(list(row_mappings.keys()))
        return abstract_question(row[column], row_mappings, pattern)

    # Replace the question and, when present, open-ended gold answers so
    # abstract evaluation compares abstract labels against abstract labels.
    df[question_column] = df.apply(
        lambda row: abstract_row_value(row, question_column), axis=1
    )
    if answer_column in df.columns:
        answer_type = (
            df["Answer Type"].astype(str).str.upper()
            if "Answer Type" in df.columns
            else pd.Series("", index=df.index)
        )
        mc_mask = answer_type.isin(["MC", "MULTI CHOICE", "MULTICHOICE"])
        df.loc[mc_mask, answer_column] = df.loc[mc_mask].apply(
            lambda row: abstract_row_value(row, answer_column), axis=1
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
