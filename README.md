# LLM-ORBench: A Comprehensive Framework for Ontology-Based Reasoning Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java 17+](https://img.shields.io/badge/Java-17+-orange.svg)](https://openjdk.java.net/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.2-green.svg)](https://spring.io/projects/spring-boot)
[![OWL API](https://img.shields.io/badge/OWL%20API-5.1.20-blue.svg)](https://github.com/owlcs/owlapi)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

LLM-ORBench is a systematic framework for evaluating Large Language Models on complex ontology-based reasoning tasks. The framework generates verifiable multi-step inferences using symbolic reasoners (Pellet/OpenLLet) and provides comprehensive evaluation across multiple dimensions including abstraction, reasoning depth, and knowledge representation formats.

## Table of Contents

- [Key Features](#key-features)
- [Framework Architecture](#framework-architecture)
- [Explanation Tagging System](#explanation-tagging-system)
- [Quick Start](#quick-start)
- [Java Pipeline](#java-pipeline)
- [Python Evaluation Pipeline](#python-evaluation-pipeline)
- [Dataset Structure](#dataset-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Research Applications](#research-applications)
- [Configuration](#configuration)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contributing](#contributing)
- [Contact](#contact)

## Key Features

- **Verifiable Ground Truth**: Uses symbolic reasoners to generate gold-standard inferences with formal proofs
- **Fine-grained Complexity Analysis**: Novel explanation tagging system based on OWL axioms and reasoning patterns
- **Multi-dimensional Evaluation**: Tests across natural language, formal SPARQL, and abstracted representations
- **Systematic Abstraction**: Removes semantic content to test pure logical reasoning capabilities
- **Comprehensive Metrics**: Accuracy, confidence calibration, and hallucination detection
- **Scalable Pipeline**: Processes multiple ontologies with memory-efficient sequential processing

## Framework Architecture

The LLM-ORBench framework consists of two main phases:

### Phase 1: Java-Based Dataset Generation

1. **SmallOntologyExtractor**: Creates 1-hop and 2-hop resource-centric subgraphs
2. **TBox Preservation**: Maintains complete terminological knowledge
3. **Individual-Centric Extraction**: Focuses reasoning context around specific entities

### Phase 2: Benchmark Dataset Generation

1. **Symbolic Reasoning**: Apply Pellet reasoner to derive ground-truth inferences
2. **Explanation Analysis**: Generate and tag formal proofs using our 20-tag system
3. **Query Generation**: Create both SPARQL ASK and SELECT queries
4. **Multi-format Output**: Natural language and formal representations
5. **Stratified Sampling**: Balanced representation across complexity levels

### Phase 3: Python-Based Evaluation Pipeline

1. **Data Preparation**: Stratified sampling and abstraction generation
2. **Verbalization**: Convert formal ontologies to natural language
3. **LLM Testing**: Multi-setting evaluation across different models
4. **Results Analysis**: Comprehensive statistical evaluation

## Explanation Tagging System

Our novel complexity metric analyzes formal proofs using 20 distinct tags:

| Tag | Description | Example Usage |
|-----|-------------|---------------|
| **D** | Direct Assertion | Explicitly stated facts |
| **H** | Hierarchy | rdfs:subClassOf, rdfs:subPropertyOf |
| **T** | Transitivity | Transitive property reasoning |
| **S** | Symmetry | Symmetric property reasoning |
| **I** | Inverse | Inverse property pairs |
| **R** | Domain/Range | Property domain/range constraints |
| **N** | Property Chain | Multi-step property composition |
| **M** | Multi-step | Combination of multiple axiom types |
| **E** | Existential | ∃ restrictions (owl:someValuesFrom) |
| **L** | Universal | ∀ restrictions (owl:allValuesFrom) |
| **C** | Cardinality | Cardinality restrictions |
| **Q** | Equivalence | owl:equivalentClass relationships |
| **∩** | Intersection | owl:intersectionOf |
| **∪** | Union | owl:unionOf |
| **¬** | Complement | owl:complementOf |

**Tag String Length** serves as our primary proxy for reasoning complexity, with longer strings indicating more complex inferences requiring multiple logical steps.

## Quick Start

### Prerequisites

- Java 17 or higher
- Maven 3.6 or higher
- Python 3.8 or higher
- Minimum 8GB RAM (16GB recommended for large ontologies)

### Installation
```bash
git clone https://github.com/SaraFlht/LLM-ORBench.git
cd LLM-ORBench
mvn clean install
cd scripts/llm_pipeline/
pip install -r requirements.txt
