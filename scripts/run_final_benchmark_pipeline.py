import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_executable(*names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the benchmark generation pipeline from an OWL file through "
            "final_benchmark/create_final_bench.py."
        )
    )
    parser.add_argument(
        "--input-owl",
        required=True,
        help="Path to the source OWL ontology file.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name used in data/resources, data/output, and final_benchmark.",
    )
    parser.add_argument(
        "--hops",
        nargs="+",
        choices=["1hop", "2hop"],
        default=["1hop", "2hop"],
        help="Hop datasets to build. Default: 1hop 2hop.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model for SPARQL-to-NL conversion. Default: gpt-4.1-mini.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stages whose expected output already exists.",
    )
    parser.add_argument(
        "--sampling-test-size",
        type=float,
        default=0.95,
        help=(
            "Fraction assigned to the stratified test split; the remaining "
            "rows are used for the benchmark sample. Default: 0.95."
        ),
    )
    parser.add_argument(
        "--max-sampled-rows",
        type=int,
        default=None,
        help="Optional hard cap on rows emitted by stratified sampling.",
    )
    parser.add_argument(
        "--no-maven-build",
        action="store_true",
        help="Do not run mvn package before Java stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def run_command(command, dry_run=False):
    print("\n" + "=" * 80)
    print("Running:", " ".join(str(part) for part in command))
    print("=" * 80)

    if dry_run:
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def should_skip(path, skip_existing):
    return skip_existing and Path(path).exists()


def is_valid_json(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        return True
    except json.JSONDecodeError:
        return False


def has_data_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle) > 1


def core_outputs_ready(sparql_questions, explanations):
    return has_data_rows(sparql_questions) and is_valid_json(explanations)


def py_script(script, *args):
    return [sys.executable, str(PROJECT_ROOT / script), *map(str, args)]


def main():
    args = parse_args()

    input_owl = Path(args.input_owl)
    if not input_owl.is_absolute():
        input_owl = PROJECT_ROOT / input_owl

    if not input_owl.exists():
        raise FileNotFoundError(f"Input OWL file not found: {input_owl}")

    if not args.dry_run:
        os.makedirs(PROJECT_ROOT / "data" / "resources", exist_ok=True)
        os.makedirs(PROJECT_ROOT / "data" / "output", exist_ok=True)
        os.makedirs(PROJECT_ROOT / "final_benchmark", exist_ok=True)

    mvn = find_executable("mvn.cmd", "mvn.bat", "mvn")
    if not mvn and not args.no_maven_build:
        raise FileNotFoundError(
            "Maven was not found on PATH. Install Maven or add its bin directory "
            "to PATH, then reopen PowerShell. If the Java project is already built, "
            "rerun with --no-maven-build."
        )

    if not args.no_maven_build:
        run_command([mvn, "package", "-DskipTests"], dry_run=args.dry_run)

    resource_dirs = {
        "1hop": PROJECT_ROOT / "data" / "resources" / f"{args.dataset}_1hop",
        "2hop": PROJECT_ROOT / "data" / "resources" / f"{args.dataset}_2hop",
    }

    extraction_outputs_ready = all(
        resource_dirs[hop].exists() and any(resource_dirs[hop].glob("*.ttl"))
        for hop in args.hops
    )

    if args.skip_existing and extraction_outputs_ready:
        print(
            "Skipping subgraph extraction; resource directories already contain TTL files."
        )
    else:
        if not mvn:
            raise FileNotFoundError(
                "Maven was not found on PATH, but subgraph extraction needs Maven. "
                "Install Maven or add its bin directory to PATH, then reopen PowerShell."
            )

        run_command(
            [
                mvn,
                "exec:java",
                "-Dexec.mainClass=SmallOntologyExtractor",
                "-Dexec.args="
                + " ".join(
                    [
                        str(input_owl),
                        str(resource_dirs["1hop"]),
                        str(resource_dirs["2hop"]),
                    ]
                ),
            ],
            dry_run=args.dry_run,
        )

    jar_path = PROJECT_ROOT / "target" / "llm-orbench-1.0-SNAPSHOT.jar"

    for hop in args.hops:
        resources_dir = resource_dirs[hop]
        output_dir = PROJECT_ROOT / "data" / "output" / args.dataset / hop
        abstracted_dir = (
            PROJECT_ROOT
            / "data"
            / "output"
            / "abstracted_ontologies"
            / f"{args.dataset}_{hop}"
        )
        verbalized_dir = (
            PROJECT_ROOT
            / "data"
            / "output"
            / "verbalized_ontologies"
            / f"{args.dataset}_{hop}"
        )
        final_json = PROJECT_ROOT / "final_benchmark" / f"{args.dataset}_{hop}.json"

        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        sparql_questions = output_dir / "SPARQL_questions.csv"
        explanations = output_dir / "Explanations.json"
        if args.skip_existing and core_outputs_ready(sparql_questions, explanations):
            print(
                f"Skipping core benchmark generation for {hop}; "
                f"{sparql_questions} and {explanations} are complete."
            )
        else:
            if args.skip_existing and sparql_questions.exists():
                print(
                    f"Regenerating core benchmark generation for {hop}; "
                    f"existing core outputs are incomplete or invalid."
                )
            run_command(
                ["java", "-jar", str(jar_path), str(resources_dir), str(output_dir)],
                dry_run=args.dry_run,
            )

        sampled_questions = output_dir / "SPARQL_questions_sampling.csv"
        if should_skip(sampled_questions, args.skip_existing):
            print(
                f"Skipping stratified sampling for {hop}; {sampled_questions} exists."
            )
        else:
            sampling_command = py_script(
                "scripts/llm_pipeline/stratified_sampling.py",
                "--input_file",
                sparql_questions,
                "--output_file",
                sampled_questions,
                "--test-size",
                args.sampling_test_size,
            )
            if args.max_sampled_rows is not None:
                sampling_command.extend(["--max-rows", str(args.max_sampled_rows)])

            run_command(
                sampling_command,
                dry_run=args.dry_run,
            )

        mappings_file = abstracted_dir / "abstraction_mappings.txt"
        abstracted_ontologies_dir = abstracted_dir / "abstracted_ontologies"
        if (
            should_skip(mappings_file, args.skip_existing)
            and abstracted_ontologies_dir.exists()
        ):
            print(
                f"Skipping ontology abstraction for {hop}; abstraction output exists."
            )
        else:
            run_command(
                py_script(
                    "scripts/ontology_tools/abstraction/Usage.py",
                    "--input-directory",
                    resources_dir,
                    "--output-directory",
                    abstracted_dir,
                ),
                dry_run=args.dry_run,
            )

        if should_skip(verbalized_dir, args.skip_existing):
            print(
                f"Skipping ontology verbalization for {hop}; {verbalized_dir} exists."
            )
        else:
            run_command(
                py_script(
                    "scripts/llm_pipeline/verbalize_ontologies.py",
                    "--input-dir",
                    resources_dir,
                    "--output-dir",
                    verbalized_dir,
                    "--file-pattern",
                    "*.ttl",
                ),
                dry_run=args.dry_run,
            )

        abstracted_verbalized_dir = verbalized_dir / "abstracted"
        if should_skip(abstracted_verbalized_dir, args.skip_existing):
            print(
                f"Skipping abstract ontology verbalization for {hop}; "
                f"{abstracted_verbalized_dir} exists."
            )
        else:
            run_command(
                py_script(
                    "scripts/llm_pipeline/verbalize_ontologies.py",
                    "--input-dir",
                    abstracted_ontologies_dir,
                    "--output-dir",
                    abstracted_verbalized_dir,
                    "--file-pattern",
                    "*.ttl",
                ),
                dry_run=args.dry_run,
            )

        abstract_temp = output_dir / "SPARQL_questions_sampling_abs_temp.csv"
        if should_skip(abstract_temp, args.skip_existing):
            print(
                f"Skipping abstract question conversion for {hop}; {abstract_temp} exists."
            )
        else:
            run_command(
                py_script(
                    "scripts/llm_pipeline/verbalize_abstract.py",
                    "--input-file",
                    sampled_questions,
                    "--mapping-file",
                    mappings_file,
                    "--output-file",
                    abstract_temp,
                ),
                dry_run=args.dry_run,
            )

        nl_questions = output_dir / "SPARQL_questions_sampling_nl.csv"
        if should_skip(nl_questions, args.skip_existing):
            print(f"Skipping SPARQL-to-NL for {hop}; {nl_questions} exists.")
        else:
            run_command(
                py_script(
                    "scripts/llm_pipeline/sparql_to_nl.py",
                    "--input-csv",
                    sampled_questions,
                    "--output-directory",
                    output_dir,
                    "--output-file",
                    nl_questions.name,
                    "--model",
                    args.model,
                ),
                dry_run=args.dry_run,
            )

        abs_questions = output_dir / "SPARQL_questions_sampling_abs.csv"
        if should_skip(abs_questions, args.skip_existing):
            print(f"Skipping abstract SPARQL-to-NL for {hop}; {abs_questions} exists.")
        else:
            run_command(
                py_script(
                    "scripts/llm_pipeline/sparql_to_nl.py",
                    "--input-csv",
                    abstract_temp,
                    "--output-directory",
                    output_dir,
                    "--output-file",
                    abs_questions.name,
                    "--model",
                    args.model,
                ),
                dry_run=args.dry_run,
            )

        if args.skip_existing and is_valid_json(final_json):
            print(
                f"Skipping final benchmark creation for {hop}; {final_json} is valid."
            )
        else:
            if args.skip_existing and final_json.exists():
                print(
                    f"Regenerating final benchmark creation for {hop}; "
                    f"{final_json} is missing, empty, or invalid."
                )
            run_command(
                py_script(
                    "final_benchmark/create_final_bench.py",
                    "--dataset",
                    args.dataset,
                    "--hop",
                    hop,
                ),
                dry_run=args.dry_run,
            )

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
