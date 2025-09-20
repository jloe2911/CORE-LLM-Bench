# LLM-ORBench: A Comprehensive Framework for Ontology-Based Reasoning Evaluation

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-blue)](link-to-paper)
[![Data](https://img.shields.io/badge/Data-Google%20Drive-green)](https://drive.google.com/drive/u/0/folders/182veDWX2hfMtyOrFztZkctzJNp7U3FgF)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LLM-ORBench is a systematic framework for evaluating Large Language Models on complex ontology-based reasoning tasks. The framework generates verifiable multi-step inferences using symbolic reasoners and provides comprehensive evaluation across natural language, formal SPARQL, and abstracted representations.

**Works with any OWL ontology** - the framework automatically processes and generates reasoning benchmarks from standard OWL ontologies.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Complete Pipeline](#complete-pipeline)
- [Java Pipeline: Dataset Generation](#java-pipeline-dataset-generation)
- [Python Pipeline: LLM Evaluation](#python-pipeline-llm-evaluation)
- [Data and Reproducibility](#data-and-reproducibility)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Overview

LLM-ORBench consists of two main components:

1. **Java Pipeline**: Generates reasoning benchmarks from OWL ontologies using symbolic reasoners
2. **Python Pipeline**: Evaluates LLMs across multiple settings with comprehensive metrics

The framework implements our novel **Explanation Tagging System** that analyzes formal proofs to create fine-grained complexity metrics, enabling systematic evaluation of reasoning capabilities.

### Key Features

- **Universal OWL Support**: Compatible with any valid OWL ontology
- **Verifiable Ground Truth**: Uses Pellet reasoner for gold-standard inferences
- **Multi-dimensional Evaluation**: Natural language, formal SPARQL, and abstracted reasoning
- **Explanation Tagging**: 20-tag complexity system analyzing formal proofs
- **Comprehensive Metrics**: Accuracy, confidence calibration, hallucination detection
- **Systematic Abstraction**: Tests pure logical reasoning by removing semantic content

## Repository Structure

```
LLM-ORBench/
├── README.md
├── LICENSE
├── pom.xml                          # Java dependencies
├── .env.example                     # API keys template
├── data/                            # Data directories (created during execution)
│   ├── input/                       # Place your OWL ontologies here
│   ├── output/                      # Generated benchmark datasets
│   └── results/                     # Evaluation results
├── src/main/                        # Java pipeline for dataset generation
│   ├── java/
│   │   ├── SmallOntologyExtractor.java          # Subgraph extraction
│   │   ├── com/example/application/
│   │   │   └── OwlSparqlGenerator.java          # Main processing application
│   │   └── com/example/                         # Core framework components
│   │       ├── config/ProcessingConfiguration.java
│   │       ├── explanation/                     # Explanation analysis
│   │       ├── reasoning/PelletReasoningService.java
│   │       ├── query/SparqlQueryGenerationService.java
│   │       └── output/StreamingOutputService.java
│   └── resources/
│       └── application.properties               # Java configuration
└── scripts/                        # Python evaluation pipeline
    ├── llm_pipeline/               # LLM evaluation components
    │   ├── requirements.txt
    │   ├── stratified_sampling.py              # Dataset sampling
    │   ├── sparql_to_nl.py                     # SPARQL to natural language
    │   ├── verbalize_ontologies.py             # Ontology verbalization
    │   ├── verbalize_abstract.py               # Abstract verbalization
    │   ├── api_calls.py                        # Core LLM testing
    │   ├── run_nl_verbalized.py                # Natural language evaluation
    │   ├── run_sparql_ttl.py                   # Formal-symbolic evaluation
    │   └── complete_evaluation.py              # Final analysis
    └── ontology_tools/             # Ontology processing utilities
        ├── requirements.txt
        └── abstraction/
            ├── OntologyAbstractor.py
            └── Usage.py                         # Abstraction execution
```

## Prerequisites

- **Java**: JDK 17 or higher
- **Maven**: 3.6 or higher
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for large ontologies)
- **API Keys**: OpenAI, DeepSeek, or other LLM providers

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/SaraFlht/LLM-ORBench.git
cd LLM-ORBench
```

### 2. Build Java Components
```bash
mvn clean install
```

### 3. Setup Python Environment
```bash
cd scripts/llm_pipeline/
pip install -r requirements.txt

# Setup API keys
cp ../../.env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key
# DEEPSEEK_API_KEY=your_deepseek_key
# OPENROUTER_API_KEY=your_openrouter_key
```

### 4. Create Data Directories
```bash
mkdir -p data/input data/output data/results
```

## Complete Pipeline

### Quick Start: End-to-End Execution

For users who want to run the complete pipeline:

```bash
# 1. Place your OWL ontologies in data/input/
cp your_ontology.owl data/input/

# 2. Configure paths in application.properties
# processing.ontologies-directory=data/input
# processing.output-directory=data/output

# 3. Run complete pipeline
./run_complete_pipeline.sh your_ontology.owl
```

### Step-by-Step Execution

For users who want to run individual components:

## Java Pipeline: Dataset Generation

### Overview
The Java pipeline processes OWL ontologies to generate comprehensive reasoning benchmarks with verifiable ground truth using the Pellet reasoner.

### Step 1: Subgraph Extraction

**Purpose**: Extract manageable 1-hop and 2-hop subgraphs from large ontologies while preserving complete terminological knowledge (TBox).

```bash
# Extract subgraphs from your ontology
java -cp target/llm-orbench-1.0-SNAPSHOT.jar SmallOntologyExtractor

# Or with custom paths:
java -cp target/llm-orbench-1.0-SNAPSHOT.jar SmallOntologyExtractor \
  data/input/your_ontology.owl \
  data/extracted/1hop/ \
  data/extracted/2hop/
```

**Output**: Individual-centric ontology files maintaining complete logical structure.

**Parameters**:
- **Input**: Large OWL ontology file
- **Output 1-hop**: Directory for 1-hop subgraphs
- **Output 2-hop**: Directory for 2-hop subgraphs

### Step 2: Core Pipeline Execution

**Purpose**: Generate comprehensive reasoning benchmark with explanations and complexity analysis.

```bash
# Run main processing pipeline
java -jar target/llm-orbench-1.0-SNAPSHOT.jar

# Or with custom directories:
java -jar target/llm-orbench-1.0-SNAPSHOT.jar data/extracted/1hop/ data/output/
```

**Processing Steps**:
1. **Symbolic Reasoning**: Apply Pellet reasoner to derive all valid inferences
2. **Explanation Generation**: Create formal proofs for each inference
3. **Complexity Analysis**: Apply 20-tag system to analyze reasoning steps
4. **Query Generation**: Create SPARQL ASK (binary) and SELECT (multiple-choice) queries
5. **Dataset Output**: Generate structured CSV and JSON files

**Output Files**:
- `SPARQL_questions.csv`: Complete query dataset with metadata
- `Explanations.json`: Detailed reasoning explanations with complexity tags

### Configuration

Edit `src/main/resources/application.properties`:

```properties
# Required: Input and output paths
processing.ontologies-directory=data/extracted/1hop
processing.output-directory=data/output

# Processing parameters
processing.max-explanations-per-inference=20
processing.timeout-hours=4
processing.batch-size=50
processing.enable-detailed-logging=true

# Memory management
processing.thread-pool-size=1
processing.enable-gc-logging=false

# Output configuration
output.csv.path=SPARQL_questions.csv
output.json.path=Explanations.json
```

### Memory Optimization

For large ontologies:

```bash
# Set JVM options
export MAVEN_OPTS="-Xmx16g -Xms8g -XX:+UseG1GC"

# Run with memory monitoring
java -Xmx16g -Xms8g -XX:+UseG1GC \
  -jar target/llm-orbench-1.0-SNAPSHOT.jar
```

## Python Pipeline: LLM Evaluation

### Overview
The Python pipeline evaluates LLMs across multiple settings using the generated benchmark dataset.

### Setup
```bash
cd scripts/llm_pipeline/

# Ensure API keys are configured
cat .env  # Verify your API keys are set
```

### Phase 1: Data Preparation

#### Stratified Sampling
```bash
python stratified_sampling.py
```

**Purpose**: Balance dataset across complexity and size dimensions while preventing data leakage.

**Input**: `../../data/output/SPARQL_questions.csv`
**Output**: `SPARQL_questions_sampling.csv`

**Configuration**:
- Stratifies by reasoning complexity (tag length) and ontology size
- Maintains representative distribution across reasoning patterns
- Groups related questions to prevent data leakage

#### Ontology Abstraction
```bash
cd ../ontology_tools/abstraction/
python Usage.py
```

**Purpose**: Create abstracted versions that remove semantic content while preserving logical structure.

**Functionality**:
- Classes: `ex:Person` → `ex:Class1`
- Properties: `ex:hasParent` → `ex:Property8`
- Individuals: `ex:JohnSmith` → `ex:Individual12`

**Configuration**:
```python
# Edit Usage.py for custom paths
ontology_directory = "../../data/extracted/1hop"
output_directory = "../../data/abstracted"
```

### Phase 2: Verbalization

#### Semantic Ontology Verbalization
```bash
cd ../../llm_pipeline/
python verbalize_ontologies.py \
  --input-dir ../../data/extracted/1hop/ \
  --output-dir ../../data/verbalized/ \
  --file-pattern "*.ttl"
```

#### Abstract Ontology Verbalization
```bash
python verbalize_abstract.py \
  --input-dir ../../data/abstracted/ \
  --output-dir ../../data/verbalized_abstract/ \
  --file-pattern "*.ttl"
```

#### SPARQL to Natural Language Conversion
```bash
python sparql_to_nl.py
```

**Purpose**: Convert formal SPARQL queries to human-readable questions using GPT-4o-mini.

**Configuration**:
- Temperature: 0.0 (deterministic)
- Supports binary and multiple-choice question types
- Maintains logical equivalence between formats

### Phase 3: Multi-Setting LLM Evaluation

#### Setting 1: Natural Language + Verbalized Context / Abstracted Questuons + Verbalized Abstracted Context
```bash
python run_nl_verbalized.py
```

**Evaluation Protocol**:
- **Input**: Natural language questions + verbalized ontology context
- **Models**: Configurable LLM providers (OpenAI, DeepSeek, Meta)
- **Output**: Performance across semantic and abstracted reasoning

**Configuration**:
```python
MODELS_CONFIG = {
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "deepseek-chat": "deepseek-chat",
    "llama-4-maverick": "meta-llama/llama-4-maverick"
}

CONFIG = {
    'max_workers': 8,
    'batch_size': 25,
    'checkpoint_frequency': 50,
    'silent_mode': False  # Set True for faster processing
}
```

#### Setting 2: Formal SPARQL + TTL Context
```bash
python run_sparql_ttl.py
```

**Evaluation Protocol**:
- **Input**: SPARQL queries + raw OWL/TTL ontology context
- **Purpose**: Test symbolic reasoning and formal language understanding
- **Comparison**: Direct performance comparison with Setting 1

### Phase 4: Comprehensive Analysis

```bash
python complete_evaluation.py \
  --csv-file data/results/experiment_results.csv \
  --explanations-file ../../data/output/Explanations.json \
  --output-dir ./evaluation_report/
```

**Generated Metrics**:

1. **Jaccard Accuracy**:
   ```
   J(A,E) = |A ∩ E| / |A ∪ E|
   ```

2. **Confidence Calibration**:
   ```
   Calibration = 1 - |Confidence - Accuracy|
   ```

3. **Hallucination Rate** (Multiple-choice only):
   ```
   H = Invalid_Answers / Total_Answers
   ```

4. **Tag Group Analysis**: Performance breakdown by reasoning complexity categories

5. **Statistical Analysis**: Bootstrap confidence intervals, Mann-Whitney U tests

**Output Structure**:
```
evaluation_report/
├── key_findings_summary.json        # Executive summary
├── detailed_performance.csv         # Full results
├── statistical_analysis.json        # Significance tests
└── tag_group_breakdown.csv         # Complexity analysis
```

## Explanation Tagging System

Our complexity metric analyzes formal proofs using 20 distinct tags:

| Tag | Description | Reasoning Type |
|-----|-------------|----------------|
| **D** | Direct Assertion | Fact retrieval |
| **H** | Hierarchy | rdfs:subClassOf, rdfs:subPropertyOf |
| **T** | Transitivity | Transitive property chains |
| **S** | Symmetry | Symmetric property reasoning |
| **I** | Inverse | Inverse property pairs |
| **R** | Domain/Range | Property domain/range constraints |
| **N** | Property Chain | Multi-step property composition |
| **C** | Cardinality | Cardinality restrictions |
| **E** | Existential | owl:someValuesFrom restrictions |
| **L** | Universal | owl:allValuesFrom restrictions |
| **Q** | Equivalence | owl:equivalentClass relationships |
| **J** | Disjointness | owl:disjointWith constraints |
| **∩** | Intersection | owl:intersectionOf |
| **∪** | Union | owl:unionOf |
| **¬** | Complement | owl:complementOf |
| **V** | Reflexivity | Reflexive properties |
| **Y** | Irreflexivity | Irreflexive properties |
| **A** | Asymmetry | Asymmetric properties |
| **F** | Functional | Functional properties |
| **M** | Multi-step | Combination of multiple axiom types |

**Complexity Measurement**: Tag string length serves as proxy for reasoning difficulty (e.g., "DHT" = 3-step inference requiring direct assertion, hierarchy navigation, and transitivity).

## Data and Reproducibility

### Benchmark Data Access

**Complete datasets, model outputs, and evaluation results** from our validation study are available in the [Google Drive folder](https://drive.google.com/drive/u/0/folders/182veDWX2hfMtyOrFztZkctzJNp7U3FgF).

**Available Resources**:
- Input ontologies (Family, OWL2Bench)
- Generated benchmark datasets
- Complete model evaluation results
- Supplementary analysis materials

### Reproducing Paper Results

To reproduce the exact results from our paper:

1. **Download data** from Google Drive folder
2. **Place datasets** in appropriate directories
3. **Run evaluation pipeline** with provided configurations
4. **Compare outputs** using provided reference results

### Using Your Own Ontologies

The framework supports any valid OWL ontology:

1. **Place your ontology** in `data/input/`
2. **Configure paths** in `application.properties`
3. **Run the complete pipeline** following the steps above
4. **Customize evaluation** by modifying Python script configurations

## Configuration Reference

### Java Configuration (`application.properties`)

```properties
# Required paths
processing.ontologies-directory=data/input
processing.output-directory=data/output

# Processing limits
processing.max-explanations-per-inference=20
processing.timeout-hours=4
processing.batch-size=50

# Performance tuning
processing.thread-pool-size=1
processing.enable-gc-logging=false
processing.enable-detailed-logging=true

# Output files
output.csv.path=SPARQL_questions.csv
output.json.path=Explanations.json
```

### Python Configuration

**Model Configuration** (`api_calls.py`):
```python
MODELS = {
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "deepseek-chat": "deepseek-chat",
    "llama-4-maverick": "meta-llama/llama-4-maverick"
}
```

**Processing Configuration**:
```python
CONFIG = {
    'max_workers': 8,           # Parallel workers
    'batch_size': 25,           # Questions per batch
    'checkpoint_frequency': 50, # Save progress frequency
    'memory_threshold': 85,     # Memory usage limit (%)
    'silent_mode': False        # Disable verbose output
}
```

## Troubleshooting

### Common Issues

**Java OutOfMemoryError**:
```bash
export MAVEN_OPTS="-Xmx16g -Xms8g"
java -Xmx16g -jar target/llm-orbench-1.0-SNAPSHOT.jar
```

**Reasoner timeout on large ontologies**:
```properties
# Increase timeout in application.properties
processing.timeout-hours=8
```

**Python API authentication errors**:
```bash
# Verify API keys
cd scripts/llm_pipeline/
cat .env
# Ensure all required keys are set
```

**Memory issues during Python evaluation**:
```python
# Reduce batch size in script configuration
CONFIG = {
    'batch_size': 10,  # Reduce from default 25
    'max_workers': 4   # Reduce parallel workers
}
```

### Performance Optimization

**For large ontologies** (>50MB):
- Increase Java heap size to 16GB+
- Use sequential processing (thread-pool-size=1)
- Enable GC logging for monitoring
- Process in smaller batches

**For many small ontologies**:
- Use parallel processing with multiple workers
- Increase batch size for efficiency
- Monitor memory usage with checkpoints

## Citation

If you use LLM-ORBench in your research, please cite our paper:

```bibtex
@article{falahatkar2025llmorbench,
  title={LLM-ORBench: Designing a Benchmark Dataset for Complex Ontology-Based Reasoning Tasks in Large Language Models},
  author={Falahatkar, Sara},
  journal={Under review at ICLR 2026},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Contact**: For questions about the framework or to request specific benchmark configurations, please open an issue in this repository.
