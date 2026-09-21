import argparse
import json
import os
import random
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
        default=0.75,
        help=(
            "Fraction assigned to the stratified test split; the remaining "
            "rows are used for the benchmark sample. Default: 0.75."
        ),
    )
    parser.add_argument(
        "--max-sampled-rows",
        type=int,
        default=None,
        help="Optional hard cap on rows emitted by stratified sampling.",
    )
    parser.add_argument(
        "--max-subgraphs-per-hop",
        type=int,
        default=None,
        help=(
            "Optional deterministic pre-sample cap for extracted TTL subgraphs "
            "before Java core generation. Applies to every requested hop."
        ),
    )
    parser.add_argument(
        "--max-2hop-subgraphs",
        type=int,
        default=None,
        help=(
            "Optional deterministic pre-sample cap for extracted 2-hop TTL "
            "subgraphs before Java core generation."
        ),
    )
    parser.add_argument(
        "--subgraph-sample-seed",
        type=int,
        default=13,
        help="Random seed for deterministic subgraph pre-sampling. Default: 13.",
    )
    parser.add_argument(
        "--max-subgraph-file-size-mb",
        type=float,
        default=None,
        help=(
            "Optional maximum TTL file size, in MB, allowed in the paired "
            "subgraph pre-sample. Useful for avoiding pathological 2-hop "
            "OWL2Bench subgraphs during explanation generation."
        ),
    )
    parser.add_argument(
        "--focus-root-individual-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During Java core generation, extract inferred triples only for the "
            "root individual named by each subgraph file. Default: true. "
            "Use --no-focus-root-individual-only to process all individuals."
        ),
    )
    parser.add_argument(
        "--max-individuals-per-ontology",
        type=int,
        default=None,
        help=(
            "Optional cap on individuals processed per ontology during Java "
            "core generation. Used after root-focus fallback, if any."
        ),
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
    parser.add_argument(
        "--no-explanations",
        action="store_true",
        help=(
            "Reuse an existing valid Explanations.json and regenerate missing "
            "SPARQL questions without recomputing explanations. If explanations "
            "are missing, explanations are generated."
        ),
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


def get_subgraph_limit(args, hop):
    if hop == "2hop" and args.max_2hop_subgraphs is not None:
        return args.max_2hop_subgraphs
    return args.max_subgraphs_per_hop


def get_paired_subgraph_limit(args):
    if args.max_2hop_subgraphs is not None:
        return args.max_2hop_subgraphs
    return args.max_subgraphs_per_hop


def get_max_subgraph_file_size_bytes(args):
    if args.max_subgraph_file_size_mb is None:
        return None
    return int(args.max_subgraph_file_size_mb * 1024 * 1024)


def select_paired_subgraph_names(resource_dirs, limit, seed, max_file_size_bytes=None):
    one_hop_files = {path.name: path for path in resource_dirs["1hop"].glob("*.ttl")}
    two_hop_files = {path.name: path for path in resource_dirs["2hop"].glob("*.ttl")}
    common_names = sorted(set(one_hop_files) & set(two_hop_files))

    if not common_names:
        raise FileNotFoundError(
            "No matching TTL subgraph filenames found between 1hop and 2hop."
        )
    total_common = len(common_names)
    if max_file_size_bytes is not None:
        common_names = [
            name
            for name in common_names
            if one_hop_files[name].stat().st_size <= max_file_size_bytes
            and two_hop_files[name].stat().st_size <= max_file_size_bytes
        ]
        if not common_names:
            raise FileNotFoundError(
                "No paired TTL subgraphs remain after applying "
                f"--max-subgraph-file-size-mb={max_file_size_bytes / 1024 / 1024:.2f}."
            )
        print(
            "Subgraph size filter kept "
            f"{len(common_names)}/{total_common} paired TTL filenames."
        )
    if limit is None or limit <= 0 or limit >= len(common_names):
        return common_names

    rng = random.Random(seed)
    return sorted(rng.sample(common_names, limit))


def find_existing_paired_manifest(dataset, requested_hops):
    if len(requested_hops) != 1:
        return None

    requested_hop = requested_hops[0]
    counterpart_hop = "1hop" if requested_hop == "2hop" else "2hop"
    pattern = f"{dataset}_{counterpart_hop}_paired_sampled_*_seed_*"
    candidates = []

    for sampled_dir in (PROJECT_ROOT / "data" / "resources").glob(pattern):
        if sampled_dir.name.endswith("_tmp"):
            continue
        manifest_path = sampled_dir / "sample_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except json.JSONDecodeError:
            continue
        if manifest.get("paired") and manifest.get("selected_files"):
            candidates.append((manifest_path.stat().st_mtime, manifest_path, manifest))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda item: item[0])
    if len(candidates) > 1:
        print(
            f"Found {len(candidates)} existing paired {counterpart_hop} samples; "
            f"reusing the newest manifest: {candidates[0][1]}."
        )
    return candidates[0][2]


def prepare_sampled_resources(
    source_dir,
    dataset,
    hop,
    limit,
    seed,
    dry_run=False,
    selected_names=None,
    paired=False,
    max_file_size_bytes=None,
):
    source_dir = Path(source_dir)
    ttl_files = sorted(source_dir.glob("*.ttl"))
    if not ttl_files:
        raise FileNotFoundError(f"No TTL subgraphs found in {source_dir}")

    if selected_names is None and (
        limit is None or limit <= 0 or limit >= len(ttl_files)
    ):
        return source_dir

    ttl_by_name = {path.name: path for path in ttl_files}
    if selected_names is None:
        rng = random.Random(seed)
        selected = sorted(rng.sample(ttl_files, limit), key=lambda path: path.name)
        selected_names = [path.name for path in selected]
    else:
        missing_names = sorted(set(selected_names) - set(ttl_by_name))
        if missing_names:
            raise FileNotFoundError(
                f"{hop} is missing {len(missing_names)} paired sampled TTL files; "
                f"first missing file: {missing_names[0]}"
            )
        selected = [ttl_by_name[name] for name in selected_names]
        limit = len(selected_names)

    sample_kind = "paired_sampled" if paired else "sampled"
    sampled_dir = (
        PROJECT_ROOT
        / "data"
        / "resources"
        / f"{dataset}_{hop}_{sample_kind}_{limit}_seed_{seed}"
    )
    manifest = {
        "source_dir": str(source_dir),
        "hop": hop,
        "limit": limit,
        "seed": seed,
        "paired": paired,
        "max_file_size_bytes": max_file_size_bytes,
        "total_available": len(ttl_files),
        "selected_files": selected_names,
    }
    manifest_path = sampled_dir / "sample_manifest.json"

    if dry_run:
        print(
            f"Would pre-sample {limit}/{len(ttl_files)} {hop} subgraphs "
            f"from {source_dir} into {sampled_dir}."
        )
        return sampled_dir

    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing_manifest = json.load(handle)
            files_ready = all((sampled_dir / name).exists() for name in selected_names)
            if existing_manifest == manifest and files_ready:
                print(
                    f"Reusing sampled {hop} resources: "
                    f"{limit}/{len(ttl_files)} TTL files in {sampled_dir}."
                )
                return sampled_dir
        except json.JSONDecodeError:
            pass

    if sampled_dir.exists():
        shutil.rmtree(sampled_dir)
    sampled_dir.mkdir(parents=True, exist_ok=True)

    for path in selected:
        shutil.copy2(path, sampled_dir / path.name)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(
        f"Prepared sampled {hop} resources: "
        f"{limit}/{len(ttl_files)} TTL files in {sampled_dir}."
    )
    return sampled_dir


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

    jar_path = PROJECT_ROOT / "target" / "llm-orbench-1.0.0.jar"
    paired_subgraph_names = None
    paired_subgraph_limit = None
    paired_subgraph_seed = args.subgraph_sample_seed
    paired_max_file_size_bytes = get_max_subgraph_file_size_bytes(args)
    requested_hops = list(args.hops)
    explicit_paired_limit = get_paired_subgraph_limit(args)
    if (
        explicit_paired_limit is not None
        and explicit_paired_limit > 0
        or paired_max_file_size_bytes is not None
    ):
        paired_subgraph_limit = explicit_paired_limit
        paired_subgraph_names = select_paired_subgraph_names(
            resource_dirs,
            paired_subgraph_limit,
            paired_subgraph_seed,
            paired_max_file_size_bytes,
        )
        print(
            "Using paired subgraph pre-sample: "
            f"{len(paired_subgraph_names)} matching TTL filenames."
        )
    else:
        existing_manifest = find_existing_paired_manifest(args.dataset, requested_hops)
        if existing_manifest:
            paired_subgraph_names = existing_manifest["selected_files"]
            paired_subgraph_limit = len(paired_subgraph_names)
            paired_subgraph_seed = existing_manifest.get("seed", paired_subgraph_seed)
            paired_max_file_size_bytes = existing_manifest.get("max_file_size_bytes")
            print(
                "Reusing paired subgraph selection from existing counterpart "
                f"manifest: {paired_subgraph_limit} matching TTL filenames."
            )

    for hop in args.hops:
        selected_names = paired_subgraph_names if paired_subgraph_names else None
        resources_dir = prepare_sampled_resources(
            resource_dirs[hop],
            args.dataset,
            hop,
            paired_subgraph_limit if selected_names else get_subgraph_limit(args, hop),
            paired_subgraph_seed if selected_names else args.subgraph_sample_seed,
            dry_run=args.dry_run,
            selected_names=selected_names,
            paired=selected_names is not None,
            max_file_size_bytes=paired_max_file_size_bytes if selected_names else None,
        )
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
            java_command = [
                "java",
                "-jar",
                str(jar_path),
                str(resources_dir),
                str(output_dir),
            ]
            explanations_ready = is_valid_json(explanations)
            if args.no_explanations and explanations_ready:
                print(
                    f"Reusing existing {explanations}; "
                    "regenerating missing core questions without recomputing explanations."
                )
                java_command.append("--no-explanations")
            elif args.no_explanations:
                print(
                    f"{explanations} is missing or invalid; "
                    "generating explanations for this run."
                )
            java_command.append(
                "--processing.focus-root-individual-only="
                + str(args.focus_root_individual_only).lower()
            )
            if args.max_individuals_per_ontology is not None:
                java_command.append(
                    "--processing.max-individuals-per-ontology="
                    + str(args.max_individuals_per_ontology)
                )
            run_command(
                java_command,
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
                "--dataset",
                args.dataset,
                "--hop",
                hop,
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
