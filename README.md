# LLM-ORBench: A Comprehensive Framework for Ontology-Based Reasoning Evaluation

LLM-ORBench is a systematic framework for evaluating Large Language Models on complex ontology-based reasoning tasks. The framework generates verifiable multi-step inferences using symbolic reasoners (Pellet/OpenLLet) and provides comprehensive evaluation across multiple dimensions including abstraction, reasoning depth, and knowledge representation formats.

## Table of Contents

- Key Features
- Framework Architecture
- Explanation Tagging System
- Quick Start
- Java Pipeline
- Python Evaluation Pipeline
- Dataset Structure
- Evaluation Metrics
- Research Applications
- Configuration
- Performance Benchmarks
- Troubleshooting
- Citation

## Key Features

- Verifiable Ground Truth: Uses symbolic reasoners to generate gold-standard inferences with formal proofs
- Fine-grained Complexity Analysis: Novel explanation tagging system based on OWL axioms and reasoning patterns
- Multi-dimensional Evaluation: Tests across natural language, formal SPARQL, and abstracted representations
- Systematic Abstraction: Removes semantic content to test pure logical reasoning capabilities
- Comprehensive Metrics: Accuracy, confidence calibration, and hallucination detection
- Scalable Pipeline: Processes multiple ontologies with memory-efficient sequential processing

## Framework Architecture

The LLM-ORBench framework consists of two main phases:

### Phase 1: Java-Based Dataset Generation

1. SmallOntologyExtractor: Creates 1-hop and 2-hop resource-centric subgraphs
2. TBox Preservation: Maintains complete terminological knowledge
3. Individual-Centric Extraction: Focuses reasoning context around specific entities

### Phase 2: Benchmark Dataset Generation

1. Symbolic Reasoning: Apply Pellet reasoner to derive ground-truth inferences
2. Explanation Analysis: Generate and tag formal proofs using our 20-tag system
3. Query Generation: Create both SPARQL ASK and SELECT queries
4. Multi-format Output: Natural language and formal representations
5. Stratified Sampling: Balanced representation across complexity levels

### Phase 3: Python-Based Evaluation Pipeline

1. Data Preparation: Stratified sampling and abstraction generation
2. Verbalization: Convert formal ontologies to natural language
3. LLM Testing: Multi-setting evaluation across different models
4. Results Analysis: Comprehensive statistical evaluation

## Explanation Tagging System

Our novel complexity metric analyzes formal proofs using 20 distinct tags:

Tag D: Direct Assertion - Explicitly stated facts
Tag H: Hierarchy - rdfs:subClassOf, rdfs:subPropertyOf
Tag T: Transitivity - Transitive property reasoning
Tag S: Symmetry - Symmetric property reasoning
Tag I: Inverse - Inverse property pairs
Tag R: Domain/Range - Property domain/range constraints
Tag N: Property Chain - Multi-step property composition
Tag M: Multi-step - Combination of multiple axiom types
Tag E: Existential - restrictions (owl:someValuesFrom)
Tag L: Universal - restrictions (owl:allValuesFrom)
Tag C: Cardinality - Cardinality restrictions
Tag Q: Equivalence - owl:equivalentClass relationships
Tag ∩: Intersection - owl:intersectionOf
Tag ∪: Union - owl:unionOf
Tag ¬: Complement - owl:complementOf

Tag String Length serves as our primary proxy for reasoning complexity, with longer strings indicating more complex inferences requiring multiple logical steps.

## Quick Start

### Prerequisites

- Java 17 or higher
- Maven 3.6 or higher
- Python 3.8 or higher
- Minimum 8GB RAM (16GB recommended for large ontologies)

### Installation

```
git clone https://github.com/SaraFlht/LLM-ORBench.git
cd LLM-ORBench
mvn clean install
cd scripts/llm_pipeline/
pip install -r requirements.txt
```

### Basic Usage Workflow

```
# Step 1: Generate subgraphs from large ontologies
java -cp target/llm-orbench-1.0-SNAPSHOT.jar SmallOntologyExtractor \
  --input src/main/resources/large-ontology.owl \
  --output-1hop src/main/resources/ontologies_1hop/ \
  --output-2hop src/main/resources/ontologies_2hop/

# Step 2: Generate benchmark dataset
java -jar target/llm-orbench-1.0-SNAPSHOT.jar \
  --ontologies-dir src/main/resources/ontologies_1hop/ \
  --output-dir ./output/

# Step 3: Run complete evaluation pipeline
cd scripts/llm_pipeline/
python complete_evaluation.py \
  --input ../../output/ \
  --models gpt-4,claude-3,llama-2 \
  --output ./evaluation_results/
```

## Java Pipeline

### Core Components

The Java pipeline implements the core dataset generation functionality using Spring Boot architecture:

#### 1. Ontology Subgraph Extraction

```
# Extract 1-hop and 2-hop subgraphs from large ontologies
java -cp target/llm-orbench-1.0-SNAPSHOT.jar SmallOntologyExtractor \
  --input src/main/resources/OWL2DL-1.owl \
  --output-1hop src/main/resources/OWL2Bench_1hop/ \
  --output-2hop src/main/resources/OWL2Bench_2hop/
```

#### 2. Benchmark Dataset Generation

```
# Generate questions and explanations from subgraphs
java -jar target/llm-orbench-1.0-SNAPSHOT.jar \
  src/main/resources/ontologies_1hop/ \
  ./output/
```

### Configuration

Edit src/main/resources/application.properties:

```
# Input/Output Configuration
processing.ontologies-directory=src/main/resources/ontologies_1hop
processing.output-directory=./output

# Processing Parameters
processing.max-explanations-per-inference=20
processing.timeout-hours=2
processing.enable-detailed-logging=true

# Memory Management
processing.batch-size=50
processing.thread-pool-size=1
```

### Memory Management

For large ontology processing:

```
export MAVEN_OPTS="-Xmx16g -Xms8g -XX:+UseG1GC"
java -Xmx16g -jar target/llm-orbench-1.0-SNAPSHOT.jar
```

## Python Evaluation Pipeline

The scripts/ directory contains the complete evaluation pipeline for testing Large Language Models on the generated benchmark datasets.

### Directory Structure

```
scripts/
├── llm_pipeline/
│   ├── requirements.txt
│   ├── stratified_sampling.py
│   ├── sparql_to_nl.py
│   ├── verbalize_ontologies.py
│   ├── verbalize_abstract.py
│   ├── api_calls.py
│   ├── run_nl_verbalized.py
│   ├── run_sparql_ttl.py
│   └── complete_evaluation.py
└── ontology_tools/
    ├── requirements.txt
    └── abstraction/
        ├── OntologyAbstractor.py
        └── Usage.py
```

### Setup

```
cd scripts/llm_pipeline/
pip install -r requirements.txt

# Create environment file for API keys
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
EOF
```

### Phase 1: Data Preparation and Sampling

#### Stratified Sampling
```
python stratified_sampling.py \
  --input ../../output/SPARQL_questions.csv \
  --output ./sampled_questions.csv \
  --sample-size 1000 \
  --stratify-by complexity,size
```

### Phase 2: Ontology Abstraction

#### Generate Abstract Ontologies
```
cd ../ontology_tools/abstraction/

python Usage.py \
  --input ../../../src/main/resources/ontologies_1hop/ \
  --output ../../../src/main/resources/ontologies_abstract/ \
  --abstraction-mode complete \
  --preserve-structure
```

The abstraction process systematically replaces:
- Classes: ex:Person → ex:Class1
- Properties: ex:hasParent → ex:Property8
- Individuals: ex:JohnSmith → ex:Individual12

### Phase 3: Verbalization Pipeline

#### Natural Language Verbalization
```
cd ../../llm_pipeline/

# Verbalize semantic ontologies
python verbalize_ontologies.py \
  --input ../../../src/main/resources/ontologies_1hop/ \
  --output ./verbalized_ontologies/ \
  --format natural_language

# Verbalize abstracted ontologies
python verbalize_abstract.py \
  --input ../../../src/main/resources/ontologies_abstract/ \
  --output ./verbalized_abstract/ \
  --preserve-logical-structure
```

#### SPARQL to Natural Language Conversion
```
python sparql_to_nl.py \
  --input ./sampled_questions.csv \
  --output ./nl_questions.csv \
  --model gpt-4o-mini \
  --temperature 0
```

### Phase 4: LLM Evaluation

#### Natural Language Reasoning Evaluation
```
python run_nl_verbalized.py \
  --questions ./nl_questions.csv \
  --ontologies ./verbalized_ontologies/ \
  --models gpt-4,claude-3-sonnet,llama-2-70b \
  --output ./results_nl/ \
  --include-abstraction
```

#### Formal-Symbolic Reasoning Evaluation
```
python run_sparql_ttl.py \
  --questions ./sampled_questions.csv \
  --ontologies ../../../src/main/resources/ontologies_1hop/ \
  --models gpt-4,claude-3-sonnet,llama-2-70b \
  --output ./results_sparql/ \
  --format turtle
```

### Phase 5: Comprehensive Results Analysis

```
python complete_evaluation.py \
  --results-dir ./results_nl/ ./results_sparql/ \
  --explanations ../../output/Explanations.json \
  --output ./final_evaluation_report/ \
  --include-statistical-tests
```

## Dataset Structure

### Generated Outputs

- SPARQL_questions.csv: Comprehensive query dataset with metadata
- Explanations.json: Detailed reasoning explanations with complexity tags
- Evaluation Results: Model performance across multiple dimensions

### CSV Schema

```
"Task ID","Root Entity","Size of ontology TBox","Size of ontology ABox",
"Task Type","Answer Type","SPARQL Query","Predicate","Answer",
"Min Tag Length","Max Tag Length"
```

### JSON Schema

```
{
  "subject|predicate|object": {
    "inferred": {
      "subject": "individual_name",
      "predicate": "property_name",
      "object": "target_value"
    },
    "explanations": [
      ["axiom1_description", "axiom2_description", "TAG:DHT"],
      ["alternative_path", "TAG:HR"]
    ],
    "size": {"min": 2, "max": 4},
    "explanationCount": 2,
    "taskIds": ["1hop-entity_subject-subject-predicate-BIN"],
    "sparqlQueries": ["ASK WHERE { ... }", "SELECT ?x WHERE { ... }"]
  }
}
```

## Evaluation Metrics

### Primary Metrics (DeepEval Framework)

```
# Jaccard Accuracy
J(A, E) = |A ∩ E| / |A ∪ E|

# Confidence Calibration Score
Calibration = 1 - |Confidence - Accuracy|

# Hallucination Rate (Multi-choice only)
H = Number_of_Invalid_Answers / Total_Answers_Provided
```

### Complexity Analysis
- Tag Length: Proxy for reasoning difficulty (1-5+ steps)
- Reasoning Groups: Basic Hierarchy, Property Characteristics, Domain/Range, Class Relations
- Multi-step Detection: Inferences requiring multiple axiom types

### Three-Setting Evaluation Protocol

1. Setting 1 (Natural Language): Verbalized ontology context + natural language questions
2. Setting 2 (Formal-Symbolic): Raw OWL syntax + SPARQL queries
3. Setting 3 (Abstract): Semantically stripped versions testing pure logical reasoning

## Research Applications

### Included Ontologies
- Family Ontology: 2,527 axioms modeling genealogical relationships
- OWL2Bench: 60,573 axioms designed for reasoner stress-testing

### Supported Question Types
- Binary Questions: TRUE/FALSE answers (ASK queries)
- Open-ended Questions: Free-form responses (SELECT queries)
- Multi-choice Questions: Selection from valid answer sets

### Research Questions Addressed

1. RQ1: Can LLMs handle logical challenges such as abstraction, contradiction, and multi-step reasoning?
2. RQ2: Under what conditions do LLMs show inconsistent reasoning behavior?
3. RQ3: How does accuracy differ between single-step and multi-step inference tasks?
4. RQ4: How do different LLMs compare in logical reasoning capabilities?

## Configuration

### Memory Management
```
# JVM options for large ontology processing
-Xmx16g -Xms8g
-XX:+UseG1GC
-Djava.util.concurrent.ForkJoinPool.common.parallelism=1
```

### Processing Limits
```
processing.max-explanations-per-inference=20
processing.timeout-hours=2
processing.batch-size=50
```

## Performance Benchmarks

Ontology Type | Avg Processing Time | Memory Usage | Generated Queries
Family (1-hop) | 45 seconds | 2.1 GB | 1,424
Family (2-hop) | 78 seconds | 3.2 GB | 2,183
OWL2Bench (1-hop) | 312 seconds | 8.7 GB | 2,461
OWL2Bench (2-hop) | 567 seconds | 12.4 GB | 4,573

## Troubleshooting

### Common Issues

OutOfMemoryError during processing:
```
# Increase heap size
export MAVEN_OPTS="-Xmx16g -Xms8g"
java -Xmx16g -jar target/llm-orbench-1.0-SNAPSHOT.jar
```

Reasoner timeout on large ontologies:
```
# Increase timeout in application.properties
processing.timeout-hours=6
```

Duplicate ontology IRI errors:
```
# Clear maven cache and rebuild
mvn clean install -U
```

Python API key issues:
```
# Ensure .env file is in the correct directory
cd scripts/llm_pipeline/
cat .env  # Verify API keys are set
```

## Data and Reproducibility

Test Data Access: Complete input datasets, model outputs, and evaluation results from our experiments are available in the Google Drive folder (https://drive.google.com/drive/u/0/folders/182veDWX2hfMtyOrFztZkctzJNp7U3FgF) for reference and reproducibility.

Supported Models: The evaluation pipeline supports API-based models (GPT-4, GPT-3.5, Claude-3, DeepSeek, LLaMA, etc.) and can be extended for local model evaluation.

## Citation

If you use LLM-ORBench in your research, please cite:

```

@article{falahatkar2025llmorbench_paper,
  title={LLM-ORBench: Designing a Benchmark Dataset for Complex Ontology-Based Reasoning Tasks in Large Language Models},
  author={Falahatkar, Sara},
  journal={Under review at ICLR 2026},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


---

Note: This framework represents a comprehensive approach to evaluating LLM reasoning capabilities through systematic, ontology-grounded assessment. The complete pipeline enables researchers to systematically benchmark model performance across multiple dimensions of logical reasoning complexity.
