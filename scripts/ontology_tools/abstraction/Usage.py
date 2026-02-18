# usage.py
from OntologyAbstractor import process_ontology_abstraction
from pathlib import Path
import os

# Navigate to project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
os.chdir(project_root)


def main():
    # Your original input files
    ontology_directory = "src/main/resources/OWL2Bench_2hop"
    output_directory = "output/OWL2Bench/2hop/abstracted"

    # Run comprehensive abstraction
    results = process_ontology_abstraction(
        ontology_dir=ontology_directory, output_dir=output_directory
    )

    print("Comprehensive abstraction completed!")
    print(f"Abstracted ontologies: {results['ontologies_dir']}")
    print(f"Mappings: {results['mappings_file']}")


if __name__ == "__main__":
    main()
