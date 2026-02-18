import pandas as pd
import json
import numpy as np
import re
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path
import os
from scipy import stats

# Navigate to project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)

# Define tag groupings for analysis
TAG_GROUPS = {
    "Basic_Hierarchy": ["H", "I"],  # Hierarchy and Inverse
    "Property_Characteristics": ["T", "S"],  # Transitivity and Symmetry
    "Class_Relations": ["Q", "J"],  # Equivalence and Disjointness
    "Restrictions": ["C", "E", "L"],  # Cardinality, Existential, Universal
    "Advanced_Properties": ["V", "Y", "A"],  # Reflexivity, Irreflexivity, Asymmetry
    "Boolean_Logic": ["∩", "∪", "¬"],  # Intersection, Union, Complement
    "Domain_Range": ["R"],  # Domain/Range (separate due to frequency)
    "Property_Chains": ["N"],  # Property chains (complex reasoning)
    "Functional": ["F"],  # Functional properties
    "Multi_Step": ["M"],  # Multi-step combination (analyze separately)
}

TAG_DESCRIPTIONS = {
    "D": "Direct Assertion",
    "H": "Hierarchy",
    "T": "Transitivity",
    "S": "Symmetry",
    "I": "Inverse",
    "F": "Functional Property",
    "Q": "Equivalence",
    "J": "Disjointness",
    "R": "Domain/Range",
    "N": "Property Chain",
    "C": "Cardinality",
    "E": "Existential Restriction",
    "L": "Universal Restriction",
    "V": "Reflexivity",
    "Y": "Irreflexivity",
    "A": "Asymmetry",
    "∩": "Intersection",
    "∪": "Union",
    "¬": "Complement",
    "M": "Multi-step Combination",
}


class JaccardAccuracyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.async_mode = False

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase):
        expected = str(test_case.expected_output).strip()
        actual = str(test_case.actual_output).strip()
        answer_type = test_case.metadata.get("answer_type", "BIN")

        # Clean and normalize answers
        def clean_answer(text):
            # Convert to lowercase and remove extra spaces
            text = text.lower().strip()
            text = re.sub(r"\s+", " ", text)

            # Handle different separators
            if ";" in text:
                items = text.split(";")
            elif "," in text:
                items = text.split(",")
            else:
                items = [text]

            # Clean each item thoroughly
            cleaned_items = []
            for item in items:
                item = item.strip()

                # Remove underscores, asterisks, and other special characters
                item = re.sub(r'[_*#@$%^&()+=\[\]{}|\\:";\'<>?/~`]', "", item)

                # Remove extra whitespace
                item = re.sub(r"\s+", " ", item).strip()

                # Remove numbers from names (e.g., "1887" from "violet heath 1887")
                item = re.sub(r"\s+\d{4}$", "", item)
                item = re.sub(r"\s+\d{4}\s+", " ", item)

                # Remove standalone numbers
                item = re.sub(r"^\d+$", "", item)

                # Final cleanup - remove any remaining special chars except letters, numbers, and spaces
                item = re.sub(r"[^a-z0-9\s]", "", item)
                item = re.sub(r"\s+", " ", item).strip()

                if item and len(item) > 0:
                    cleaned_items.append(item)

            return set(cleaned_items)

        expected_set = clean_answer(expected)
        actual_set = clean_answer(actual)

        # Calculate Jaccard similarity
        if len(expected_set) == 0 and len(actual_set) == 0:
            score = 1.0
        elif len(expected_set) == 0 or len(actual_set) == 0:
            score = 0.0
        else:
            intersection = len(expected_set.intersection(actual_set))
            union = len(expected_set.union(actual_set))
            jaccard_score = intersection / union if union > 0 else 0.0

            # For binary questions, convert to binary score
            if answer_type == "BIN":
                score = 1.0 if jaccard_score == 1.0 else 0.0
            else:
                score = jaccard_score

        self.success = score >= self.threshold
        self.score = score
        return score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Jaccard Accuracy"


class ConfidenceCalibrationMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.async_mode = False
        self.model_name = None  # Will be set when creating the metric

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase):
        # Get confidence from the stored metadata (from CSV column)
        confidence = test_case.metadata.get("confidence_score", 0.5)
        confidence = max(0.0, min(1.0, confidence))  # Ensure it's between 0 and 1

        # Get the Jaccard accuracy score directly from the test case
        # This should have been calculated by JaccardAccuracyMetric first
        jaccard_accuracy = test_case.metrics_data.get("Jaccard Accuracy", {}).get(
            "score", 0
        )

        # For calibration, we need binary correctness (1 if perfect, 0 otherwise)
        answer_type = test_case.metadata.get("answer_type", "BIN")

        if answer_type == "BIN":
            # For binary questions, correctness is 1 if Jaccard = 1.0, else 0
            is_correct = 1.0 if jaccard_accuracy >= 1.0 else 0.0
        else:  # MC questions
            # For MC questions, you could use either:
            # Option 1: Binary (1 if perfect, 0 otherwise)
            is_correct = 1.0 if jaccard_accuracy >= 1.0 else 0.0
            # Option 2: Use the Jaccard score directly (partial credit)
            # is_correct = jaccard_accuracy

        # Calibration error: |confidence - correctness|
        calibration_error = abs(confidence - is_correct)
        calibration_score = 1.0 - calibration_error

        self.score = calibration_score
        self.success = calibration_score >= self.threshold

        return calibration_score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Confidence Calibration"


class OntologyHallucinationMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.async_mode = False

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase):
        actual_output = str(test_case.actual_output).strip()
        expected_output = str(test_case.expected_output).strip()
        answer_type = test_case.metadata.get("answer_type", "BIN")

        # For binary questions, skip hallucination calculation entirely
        if answer_type == "BIN":
            self.score = None  # Don't include in calculations
            self.success = True
            return None

        # For multiple-choice questions, calculate hallucination rate
        hallucination_score = self._detect_mc_hallucinations(
            actual_output, expected_output, test_case.context
        )

        self.score = hallucination_score
        self.success = hallucination_score >= self.threshold
        return hallucination_score

    def _detect_mc_hallucinations(self, actual_output, expected_output, context):
        """
        Detect hallucinations in multiple-choice answers.
        Hallucination = providing answers that don't exist as valid options.
        """
        # Parse the actual answers provided by the model
        actual_answers = self._parse_answers(actual_output)

        if not actual_answers:
            # If no answers provided, no hallucination but wrong answer
            return 1.0

        # Parse the expected/valid answers
        expected_answers = self._parse_answers(expected_output)

        # Get all valid entities from context (for the specific query)
        valid_entities = self._extract_valid_entities_from_context(
            context, expected_answers
        )

        # Count hallucinated answers (answers not in valid entities)
        hallucinated_count = 0
        for answer in actual_answers:
            answer_normalized = self._normalize_answer(answer)

            # Check if this answer exists in valid entities
            is_valid = False
            for valid_entity in valid_entities:
                if self._answers_match(
                    answer_normalized, self._normalize_answer(valid_entity)
                ):
                    is_valid = True
                    break

            if not is_valid:
                hallucinated_count += 1

        # Calculate hallucination score (1 - hallucination rate)
        total_answers = len(actual_answers)
        hallucination_rate = (
            hallucinated_count / total_answers if total_answers > 0 else 0
        )

        return hallucination_rate

    def _parse_answers(self, answer_text):
        """Parse multiple answers from text"""
        # Clean and normalize the answer text
        text = answer_text.lower().strip()

        # Split by common delimiters
        if ";" in text:
            answers = text.split(";")
        elif "," in text:
            answers = text.split(",")
        elif "\n" in text:
            answers = text.split("\n")
        else:
            answers = [text]

        # Clean each answer
        cleaned_answers = []
        for ans in answers:
            ans = ans.strip()
            # Remove special characters but keep underscores and numbers
            ans = re.sub(r"[^\w\s]", "", ans)
            ans = re.sub(r"\s+", " ", ans).strip()

            if ans and len(ans) > 0:
                cleaned_answers.append(ans)

        return cleaned_answers

    def _extract_valid_entities_from_context(self, context, expected_answers):
        """
        Extract all valid entities that could be answers for this query.
        This includes the expected answers plus any other valid entities
        of the same type mentioned in the context.
        """
        valid_entities = set(expected_answers)

        if context:
            context_text = (
                " ".join(context) if isinstance(context, list) else str(context)
            )

            # Extract entity patterns from context
            # Pattern for individuals (e.g., john_smith_1850)
            individual_pattern = r"\b[a-z]+(?:_[a-z]+)*_\d{4}\b"
            individuals = re.findall(individual_pattern, context_text.lower())
            valid_entities.update(individuals)

            # Pattern for classes (e.g., Person, Woman)
            class_pattern = r"\b[A-Z][a-zA-Z]+\b"
            classes = re.findall(class_pattern, context_text)
            valid_entities.update([c.lower() for c in classes])

            # Also check for entities in SPARQL query if present
            sparql_pattern = r"<[^>]+#([^>]+)>"
            sparql_entities = re.findall(sparql_pattern, context_text)
            valid_entities.update([e.lower() for e in sparql_entities])

        return valid_entities

    def _normalize_answer(self, answer):
        """Normalize answer for comparison"""
        # Convert to lowercase and remove extra spaces
        normalized = answer.lower().strip()
        # Remove underscores for comparison
        normalized = normalized.replace("_", " ")
        # Remove numbers at the end (years)
        normalized = re.sub(r"\s*\d{4}$", "", normalized)
        # Remove extra spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _answers_match(self, answer1, answer2):
        """Check if two normalized answers match"""
        # Exact match
        if answer1 == answer2:
            return True

        # Check if one is contained in the other (for partial matches)
        if answer1 in answer2 or answer2 in answer1:
            return True

        # Check without spaces
        if answer1.replace(" ", "") == answer2.replace(" ", ""):
            return True

        return False

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Ontology Hallucination Detection"


class CompleteEvaluator:
    def __init__(self, csv_file: str, explanations_file: str, output_dir: str):
        self.csv_file = Path(csv_file)
        self.explanations_file = Path(explanations_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract prefix from CSV filename for output naming
        csv_filename = self.csv_file.stem  # Gets filename without extension
        self.file_prefix = csv_filename.split("_")[
            0
        ]  # Gets first part before underscore

        # Load data
        self.df = pd.read_csv(self.csv_file)
        with open(self.explanations_file, "r") as f:
            self.explanations = json.load(f)

        # Detect models
        self.models = self._detect_models()

        # Create SPARQL to explanations mapping
        self.sparql_to_explanations = self._create_sparql_mapping()

        # Calculate tag group distribution once for all questions
        self.tag_group_distribution = self._analyze_tag_group_distribution()

        print(f"📊 Loaded {len(self.df)} questions")
        print(f"📁 Found explanations for {len(self.explanations)} inferences")
        print(
            f"🔗 Created SPARQL mappings for {len(self.sparql_to_explanations)} queries"
        )
        print(f"🤖 Detected models: {list(self.models.keys())}")
        print(f"📊 Tag Group Distribution:")
        for group_name, stats in sorted(
            self.tag_group_distribution.items(),
            key=lambda x: x[1]["percentage_of_total"],
            reverse=True,
        ):
            print(f"   {group_name:<20}: {stats['percentage_of_total']:.1f}%")

    def add_statistical_analysis(self, all_results):
        """Add bootstrap confidence intervals and basic comparisons"""

        print("\n📊 STATISTICAL ANALYSIS")
        print("=" * 50)

        # Bootstrap confidence intervals for each model
        print("\n📈 Performance Confidence Intervals (95% CI):")
        model_stats = {}

        for model_name, results in all_results.items():
            jaccard_scores = [
                r.metrics_data.get("Jaccard Accuracy", {}).get("score", 0)
                for r in results["eval_results"]
            ]

            if len(jaccard_scores) > 30:
                # Bootstrap resampling
                bootstrap_means = []
                n = len(jaccard_scores)

                for _ in range(1000):
                    bootstrap_sample = np.random.choice(
                        jaccard_scores, size=n, replace=True
                    )
                    bootstrap_means.append(np.mean(bootstrap_sample))

                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                mean_score = np.mean(jaccard_scores)

                model_stats[model_name] = {
                    "mean": mean_score,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "margin_of_error": (ci_upper - ci_lower) / 2,
                    "n": len(jaccard_scores),
                }

                print(
                    f"   {model_name}: {mean_score:.1%} [{ci_lower:.1%}, {ci_upper:.1%}] (n={len(jaccard_scores)})"
                )

        # Pairwise comparisons
        print("\n🔄 Pairwise Comparisons (Mann-Whitney U test):")
        pairwise_comparisons = []
        model_names = list(model_stats.keys())

        if len(model_names) >= 2:
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]

                    scores1 = [
                        r.metrics_data.get("Jaccard Accuracy", {}).get("score", 0)
                        for r in all_results[model1]["eval_results"]
                    ]
                    scores2 = [
                        r.metrics_data.get("Jaccard Accuracy", {}).get("score", 0)
                        for r in all_results[model2]["eval_results"]
                    ]

                    if len(scores1) > 30 and len(scores2) > 30:
                        _, p_value = stats.mannwhitneyu(
                            scores1, scores2, alternative="two-sided"
                        )
                        significance_level = (
                            "***"
                            if p_value < 0.001
                            else "**"
                            if p_value < 0.01
                            else "*"
                            if p_value < 0.05
                            else "ns"
                        )

                        mean_diff = np.mean(scores1) - np.mean(scores2)

                        comparison = {
                            "models": f"{model1} vs {model2}",
                            "mean_difference": mean_diff,
                            "p_value": p_value,
                            "significance": significance_level,
                            "is_significant": p_value < 0.05,
                        }
                        pairwise_comparisons.append(comparison)

                        print(
                            f"   {model1} vs {model2}: Δ={mean_diff:+.1%}, p={p_value:.3f} {significance_level}"
                        )

        return model_stats, pairwise_comparisons

    def _detect_models(self):
        """Detect which models were tested"""
        models = {}
        for col in self.df.columns:
            if col.endswith("_final_answer"):
                model_name = col.replace("_final_answer", "")
                models[model_name] = model_name
        return models

    def _normalize_sparql(self, sparql_query: str) -> str:
        """Normalize SPARQL query for comparison"""
        if not sparql_query:
            return ""
        normalized = re.sub(r"\s+", " ", sparql_query.strip())
        normalized = normalized.rstrip(". ")
        return normalized

    def _create_sparql_mapping(self):
        """Create mapping from SPARQL queries to all their explanations"""
        sparql_mapping = defaultdict(list)

        for explanation_key, explanation_data in self.explanations.items():
            sparql_queries = explanation_data.get("sparqlQueries", [])

            for sparql_query in sparql_queries:
                normalized_query = self._normalize_sparql(sparql_query)

                explanation_info = {
                    "explanation_key": explanation_key,
                    "inferred": explanation_data.get("inferred", {}),
                    "explanations": explanation_data.get("explanations", []),
                    "explanation_count": explanation_data.get("explanationCount", 0),
                    "size": explanation_data.get("size", {}),
                    "task_ids": explanation_data.get("taskIds", []),
                }

                sparql_mapping[normalized_query].append(explanation_info)

        return dict(sparql_mapping)

    def _find_explanations_for_sparql(self, sparql_query: str):
        """Find all explanations for a given SPARQL query"""
        normalized_query = self._normalize_sparql(sparql_query)

        if normalized_query in self.sparql_to_explanations:
            return self.sparql_to_explanations[normalized_query]

        # Fuzzy matching for slight variations
        for stored_query, explanations in self.sparql_to_explanations.items():
            if self._queries_similar(normalized_query, stored_query):
                return explanations

        return []

    def _queries_similar(
        self, query1: str, query2: str, threshold: float = 0.9
    ) -> bool:
        """Check if two SPARQL queries are similar"""
        tokens1 = set(re.findall(r"\w+", query1.lower()))
        tokens2 = set(re.findall(r"\w+", query2.lower()))

        if not tokens1 or not tokens2:
            return False

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

    def _extract_shortest_explanation_tags(self, explanations_list: List[dict]) -> dict:
        """Extract the shortest tag from EACH inference separately"""
        all_shortest_tags = []
        inference_details = []

        for explanation_info in explanations_list:
            explanations = explanation_info.get("explanations", [])
            inferred = explanation_info.get("inferred", {})

            if not explanations:
                continue

            # Find the shortest explanation for THIS specific inference
            shortest_explanation = None
            shortest_tag_length = float("inf")

            for explanation in explanations:
                if len(explanation) > 2 and explanation[2].startswith("TAG:"):
                    tag = explanation[2].replace("TAG:", "")
                    # Remove 'D' from analysis as it's in everything
                    tag_without_d = tag.replace("D", "")
                    tag_length = len(tag_without_d)

                    if tag_length < shortest_tag_length:
                        shortest_tag_length = tag_length
                        shortest_explanation = {
                            "tag": tag,
                            "tag_without_d": tag_without_d,
                            "tag_length": tag_length,
                            "explanation_text": explanation[0]
                            if len(explanation) > 0
                            else "",
                            "rule_text": explanation[1] if len(explanation) > 1 else "",
                            "inferred_triple": inferred,
                            "inference_key": explanation_info.get(
                                "explanation_key", ""
                            ),
                        }

            if shortest_explanation:
                all_shortest_tags.append(shortest_explanation["tag_without_d"])
                inference_details.append(shortest_explanation)

        if not all_shortest_tags:
            return {
                "all_shortest_tags": [],
                "combined_tag_string": "",
                "tag_groups": [],
                "individual_tags": [],
                "inference_details": [],
                "inference_count": 0,
            }

        # Combine all individual tags
        all_individual_tags = []
        for tag in all_shortest_tags:
            all_individual_tags.extend(list(tag))

        # Analyze tag groups from all tags
        combined_tag_string = "".join(all_shortest_tags)
        tag_groups = self._analyze_tag_groups(combined_tag_string)

        return {
            "all_shortest_tags": all_shortest_tags,
            "combined_tag_string": combined_tag_string,
            "tag_groups": tag_groups,
            "individual_tags": list(set(all_individual_tags)),
            "inference_details": inference_details,
            "inference_count": len(inference_details),
        }

    def _analyze_tag_groups(self, tag_string: str) -> List[str]:
        """Analyze which tag groups this tag string belongs to"""
        if not tag_string:
            return []

        found_groups = []
        tag_chars = list(tag_string)

        for group_name, group_tags in TAG_GROUPS.items():
            if any(tag in tag_chars for tag in group_tags):
                found_groups.append(group_name)

        return found_groups

    def _analyze_tag_group_distribution(self):
        """Calculate tag group distribution for all questions (model-independent)"""
        total_queries = len(self.df)
        group_distribution = defaultdict(int)

        # Go through all questions and count tag groups
        for idx, row in self.df.iterrows():
            sparql_query = row.get("SPARQL Query", "")
            explanations_list = self._find_explanations_for_sparql(sparql_query)

            if explanations_list:
                tag_info = self._extract_shortest_explanation_tags(explanations_list)
                tag_groups = tag_info["tag_groups"]

                # Count each group this query belongs to
                for group in tag_groups:
                    group_distribution[group] += 1

        # Convert to percentages
        tag_group_percentages = {}
        for group_name, count in group_distribution.items():
            percentage = (count / total_queries) * 100
            tag_group_percentages[group_name] = {"percentage_of_total": percentage}

        return tag_group_percentages

    def _create_test_cases(self, model_name: str):
        """Create DeepEval test cases for a model"""
        test_cases = []
        final_answer_col = f"{model_name}_final_answer"
        confidence_col = f"{model_name}_confidence_score"
        time_col = f"{model_name}_response_time"
        tokens_col = f"{model_name}_token_count"

        # Add counters for debugging
        total_rows = len(self.df)
        bin_total = len(self.df[self.df["Answer Type"] == "BIN"])
        mc_total = len(self.df[self.df["Answer Type"] == "MC"])

        bin_skipped = 0
        mc_skipped = 0
        bin_processed = 0
        mc_processed = 0

        found_explanations = 0
        missing_explanations = 0

        for idx, row in self.df.iterrows():
            answer_type = row.get("Answer Type", "BIN")

            # Skip empty or error responses
            if (
                pd.isna(row[final_answer_col])
                or row[final_answer_col] == ""
                or str(row[final_answer_col]).startswith("[ERROR]")
                or str(row[final_answer_col]).startswith("ERROR")
            ):
                if answer_type == "BIN":
                    bin_skipped += 1
                else:
                    mc_skipped += 1
                continue

            # Count processed cases by type
            if answer_type == "BIN":
                bin_processed += 1
            else:
                mc_processed += 1

            # Get SPARQL query
            sparql_query = row.get("SPARQL Query", "")

            # Find explanations for this SPARQL query
            explanations_list = self._find_explanations_for_sparql(sparql_query)

            if explanations_list:
                found_explanations += 1
                tag_info = self._extract_shortest_explanation_tags(explanations_list)
            else:
                missing_explanations += 1
                tag_info = {
                    "all_shortest_tags": [],
                    "combined_tag_string": "",
                    "tag_groups": [],
                    "individual_tags": [],
                    "inference_details": [],
                    "inference_count": 0,
                }

            # Create rich context
            context_parts = []
            context_parts.append(f"Task Type: {row.get('Task Type', '')}")
            context_parts.append(f"Answer Type: {row.get('Answer Type', 'BIN')}")
            context_parts.append(f"Ontology: {row.get('Root Entity', '')}")
            context_parts.append(f"SPARQL Query: {sparql_query}")

            if tag_info["inference_count"] > 0:
                context_parts.append(
                    f"Number of Inferences: {tag_info['inference_count']}"
                )
                context_parts.append(
                    f"Shortest Tags per Inference: {', '.join(tag_info['all_shortest_tags'])}"
                )
                context_parts.append(
                    f"Combined Reasoning Groups: {', '.join(tag_info['tag_groups'])}"
                )

            test_case = LLMTestCase(
                input=sparql_query,
                actual_output=row[final_answer_col],
                expected_output=row.get("Answer", ""),
                context=context_parts,
            )

            # Store enriched metadata
            test_case.metadata = {
                "task_id": row.get("Task ID", ""),
                "root_entity": row.get("Root Entity", ""),
                "answer_type": row.get("Answer Type", "BIN"),
                "task_type": row.get("Task Type", ""),
                "all_shortest_tags": tag_info["all_shortest_tags"],
                "combined_tag_string": tag_info["combined_tag_string"],
                "tag_groups": tag_info["tag_groups"],
                "individual_tags": tag_info["individual_tags"],
                "inference_count": tag_info["inference_count"],
                "inference_details": tag_info["inference_details"],
                "original_index": idx,
                "explanations_found": len(explanations_list) > 0,
                "confidence_score": row.get(confidence_col, 0.5),
            }

            # Add time and token count if available
            if time_col in row and not pd.isna(row[time_col]):
                test_case.metadata["response_time"] = row[time_col]
            if tokens_col in row and not pd.isna(row[tokens_col]):
                test_case.metadata["token_count"] = row[tokens_col]

            test_cases.append(test_case)

        # Enhanced debugging output
        print(f"📊 {model_name} Test Case Creation:")
        print(
            f"   📋 Total dataset: {total_rows} questions ({bin_total} BIN, {mc_total} MC)"
        )
        print(
            f"   ❌ Skipped (errors): {bin_skipped + mc_skipped} ({bin_skipped} BIN, {mc_skipped} MC)"
        )
        print(
            f"   ✅ Processed: {bin_processed + mc_processed} ({bin_processed} BIN, {mc_processed} MC)"
        )
        print(
            f"   📊 Final distribution: {bin_processed / (bin_processed + mc_processed) * 100:.1f}% BIN, {mc_processed / (bin_processed + mc_processed) * 100:.1f}% MC"
        )
        print(f"   🔍 Found explanations: {found_explanations}")
        print(f"   ❓ Missing explanations: {missing_explanations}")

        # Additional check - show which specific questions were skipped
        if (bin_skipped + mc_skipped) > 0:
            print(
                f"   ⚠️  Error rate by type: BIN {bin_skipped}/{bin_total} ({bin_skipped / bin_total * 100:.1f}%), MC {mc_skipped}/{mc_total} ({mc_skipped / mc_total * 100:.1f}%)"
            )

        return test_cases

    def evaluate_model(self, model_name: str):
        """Evaluate a single model"""
        print(f"\n🔍 Evaluating {model_name}...")

        test_cases = self._create_test_cases(model_name)

        if not test_cases:
            print(f"❌ No valid test cases for {model_name}")
            return None

        # Define metrics
        metrics = [
            JaccardAccuracyMetric(threshold=1.0),
            ConfidenceCalibrationMetric(threshold=0.7),
            OntologyHallucinationMetric(threshold=0.7),
        ]

        # Run evaluation manually
        print(f"⚙️ Running manual evaluation...")

        eval_results = []
        for i, test_case in enumerate(test_cases):
            result = test_case
            result.metrics_data = {}

            for metric in metrics:
                try:
                    score = metric.measure(test_case)
                    result.metrics_data[metric.__name__] = {
                        "score": score,
                        "success": metric.is_successful(),
                    }
                except Exception as e:
                    print(
                        f"\n    Warning: Error in {metric.__name__} for case {i}: {e}"
                    )
                    result.metrics_data[metric.__name__] = {
                        "score": 0,
                        "success": False,
                    }

            eval_results.append(result)

        print(f"  ✅ Evaluated {len(eval_results)} test cases")

        # Calculate summary statistics
        summary = self._calculate_summary(eval_results, model_name)

        # Perform tag group analysis
        tag_group_analysis = self._analyze_by_tag_groups(eval_results, model_name)

        results = {
            "model_name": model_name,
            "eval_results": eval_results,
            "summary": summary,
            "tag_group_analysis": tag_group_analysis,
        }

        return results

    def _analyze_by_tag_groups(self, eval_results, model_name):
        """Analyze performance by tag groups - using ALL inference tags"""
        group_analysis = defaultdict(list)

        for result in eval_results:
            tag_groups = result.metadata.get("tag_groups", [])
            jaccard_score = result.metrics_data.get("Jaccard Accuracy", {}).get(
                "score", 0
            )
            calibration_score = result.metrics_data.get(
                "Confidence Calibration", {}
            ).get("score", 0)
            hallucination_score = result.metrics_data.get(
                "Ontology Hallucination Detection", {}
            ).get("score", 0)
            answer_type = result.metadata.get("answer_type", "BIN")
            all_shortest_tags = result.metadata.get("all_shortest_tags", [])
            inference_count = result.metadata.get("inference_count", 0)

            # Add to each group this query belongs to
            for group in tag_groups:
                group_analysis[group].append(
                    {
                        "jaccard": jaccard_score,
                        "calibration": calibration_score,
                        "hallucination": hallucination_score,
                        "answer_type": answer_type,
                        "all_shortest_tags": all_shortest_tags,
                        "inference_count": inference_count,
                        "is_correct": jaccard_score >= 1.0,
                    }
                )

        # Aggregate results by group
        aggregated_groups = {}
        for group_name, results in group_analysis.items():
            if len(results) >= 3:  # Only analyze groups with enough samples
                correct_count = sum(1 for r in results if r["is_correct"])
                total_count = len(results)

                # Get pre-calculated percentage distribution (same for all models)
                percentage_of_total = self.tag_group_distribution.get(
                    group_name, {}
                ).get("percentage_of_total", 0.0)

                # Filter hallucination scores to exclude None values (from BIN questions)
                valid_hallucination_scores = [
                    r["hallucination"]
                    for r in results
                    if r["hallucination"] is not None
                ]

                aggregated_groups[group_name] = {
                    "percentage_of_total": percentage_of_total,  # Same for all models
                    "accuracy_rate": correct_count / total_count,
                    "jaccard_mean": np.mean([r["jaccard"] for r in results]),
                    "jaccard_std": np.std([r["jaccard"] for r in results]),
                    "calibration_mean": np.mean([r["calibration"] for r in results]),
                    "calibration_std": np.std([r["calibration"] for r in results]),
                    "hallucination_mean": np.mean(valid_hallucination_scores)
                    if valid_hallucination_scores
                    else 0,
                    "hallucination_std": np.std(valid_hallucination_scores)
                    if valid_hallucination_scores
                    else 0,
                    "binary_count": sum(
                        1 for r in results if r["answer_type"] == "BIN"
                    ),
                    "mc_count": sum(1 for r in results if r["answer_type"] == "MC"),
                    "avg_inference_count": np.mean(
                        [r["inference_count"] for r in results]
                    ),
                }

        return aggregated_groups

    def _calculate_summary(self, eval_results, model_name):
        """Calculate overall summary statistics and also by answer type"""

        # Get original dataset counts (same for all models)
        total_dataset_questions = len(self.df)
        bin_dataset_questions = len(self.df[self.df["Answer Type"] == "BIN"])
        mc_dataset_questions = len(self.df[self.df["Answer Type"] == "MC"])

        dataset_counts = {
            "total": total_dataset_questions,
            "bin": bin_dataset_questions,
            "mc": mc_dataset_questions,
        }

        summary_results = {}

        # Overall Summary
        summary_results["overall"] = self._calculate_summary_for_subset(
            eval_results, dataset_counts
        )

        # Add dataset composition info
        summary_results["dataset_composition"] = {
            "total_questions": total_dataset_questions,
            "bin_questions": bin_dataset_questions,
            "mc_questions": mc_dataset_questions,
            "bin_percentage": (bin_dataset_questions / total_dataset_questions) * 100,
            "mc_percentage": (mc_dataset_questions / total_dataset_questions) * 100,
        }

        # Summary by Answer Type
        bin_cases = [r for r in eval_results if r.metadata.get("answer_type") == "BIN"]
        mc_cases = [r for r in eval_results if r.metadata.get("answer_type") == "MC"]

        summary_results["binary"] = self._calculate_summary_for_subset(
            bin_cases, {"total": bin_dataset_questions}
        )
        summary_results["mc"] = self._calculate_summary_for_subset(
            mc_cases, {"total": mc_dataset_questions}
        )

        return summary_results

    def _calculate_summary_for_subset(
        self, test_cases: List[LLMTestCase], dataset_counts: dict = None
    ):
        """Helper to calculate summary for a subset of test cases"""
        total_cases = len(test_cases)
        if total_cases == 0:
            return {
                "total_test_cases": 0,
                "jaccard_accuracy": {
                    "mean": 0,
                    "std": 0,
                    "perfect_answers": 0,
                    "partial_answers": 0,
                    "wrong_answers": 0,
                    "perfect_rate": 0,
                    "partial_rate": 0,
                    "wrong_rate": 0,
                },
                "confidence_calibration": {"mean": 0, "std": 0, "well_calibrated": 0},
                "hallucination_detection": {
                    "mean": 0,
                    "std": 0,
                    "clean_responses": 0,
                    "hallucination_rate": 0,
                    "severe_hallucinations": 0,
                },
                "response_time_ms": {"mean": 0, "std": 0},
                "token_count": {"mean": 0, "std": 0},
                "dataset_percentage": 0.0,
            }

        jaccard_scores = []
        answer_categories = {"perfect": 0, "partial": 0, "wrong": 0}
        calibration_scores = []
        hallucination_scores = []
        response_times = []
        token_counts = []

        for result in test_cases:
            metrics_data = result.metrics_data
            answer_type = result.metadata.get("answer_type", "BIN")  # Get answer type

            if "Jaccard Accuracy" in metrics_data:
                score = metrics_data["Jaccard Accuracy"]["score"]
                jaccard_scores.append(score)

                # Fixed categorization logic based on answer type
                if answer_type == "BIN":
                    # For binary questions, only perfect or wrong
                    if score == 1.0:
                        answer_categories["perfect"] += 1
                    else:
                        answer_categories["wrong"] += 1
                else:  # MC questions
                    if score == 1.0:
                        answer_categories["perfect"] += 1
                    elif score > 0.0:
                        answer_categories["partial"] += 1
                    else:
                        answer_categories["wrong"] += 1

            if "Confidence Calibration" in metrics_data:
                calibration_scores.append(
                    metrics_data["Confidence Calibration"]["score"]
                )

            if "Ontology Hallucination Detection" in metrics_data:
                score = metrics_data["Ontology Hallucination Detection"]["score"]
                if score is not None:  # Only include MC answers
                    hallucination_scores.append(score)

            if "response_time" in result.metadata and not pd.isna(
                result.metadata["response_time"]
            ):
                response_times.append(result.metadata["response_time"])
            if "token_count" in result.metadata and not pd.isna(
                result.metadata["token_count"]
            ):
                token_counts.append(result.metadata["token_count"])

        # Calculate dataset percentage if counts provided
        dataset_percentage = 0.0
        if dataset_counts:
            total_dataset = dataset_counts.get("total", 0)
            if total_dataset > 0:
                dataset_percentage = (total_cases / total_dataset) * 100

        summary = {
            "total_test_cases": total_cases,
            "dataset_percentage": dataset_percentage,
            "jaccard_accuracy": {
                "mean": np.mean(jaccard_scores) if jaccard_scores else 0,
                "std": np.std(jaccard_scores) if jaccard_scores else 0,
                "perfect_answers": answer_categories["perfect"],
                "partial_answers": answer_categories["partial"],
                "wrong_answers": answer_categories["wrong"],
                "perfect_rate": answer_categories["perfect"] / total_cases
                if total_cases > 0
                else 0,
                "partial_rate": answer_categories["partial"] / total_cases
                if total_cases > 0
                else 0,
                "wrong_rate": answer_categories["wrong"] / total_cases
                if total_cases > 0
                else 0,
            },
            "confidence_calibration": {
                "mean": np.mean(calibration_scores) if calibration_scores else 0,
                "std": np.std(calibration_scores) if calibration_scores else 0,
                "well_calibrated": sum(1 for s in calibration_scores if s >= 0.7)
                if calibration_scores
                else 0,
            },
            "hallucination_detection": {
                "mean": np.mean(hallucination_scores) if hallucination_scores else 0,
                "std": np.std(hallucination_scores) if hallucination_scores else 0,
                "clean_responses": sum(1 for s in hallucination_scores if s >= 0.7)
                if hallucination_scores
                else 0,
                "hallucination_rate": sum(1 for s in hallucination_scores if s < 0.7)
                / len(hallucination_scores)
                if hallucination_scores
                else 0,
                "severe_hallucinations": sum(1 for s in hallucination_scores if s < 0.3)
                if hallucination_scores
                else 0,
            },
            "response_time_ms": {
                "mean": np.mean(response_times) if response_times else 0,
                "std": np.std(response_times) if response_times else 0,
            },
            "token_count": {
                "mean": np.mean(token_counts) if token_counts else 0,
                "std": np.std(token_counts) if token_counts else 0,
            },
        }
        return summary

    def _create_summary_json(self, all_results: Dict):
        """Saves a summary of key findings from all models to a JSON file."""
        print("\n💾 Saving key findings summary to JSON...")

        if not all_results:
            print("❌ No results to save - all models failed evaluation")
            return

        # Run statistical analysis if multiple models
        statistical_analysis = {}
        if len(all_results) > 1:
            model_stats, pairwise_comparisons = self.add_statistical_analysis(
                all_results
            )

            # Format for JSON
            statistical_analysis = {
                "confidence_intervals": {
                    model: {
                        "mean_accuracy": f"{stats['mean']:.1%}",
                        "confidence_interval_lower": f"{stats['ci_lower']:.1%}",
                        "confidence_interval_upper": f"{stats['ci_upper']:.1%}",
                        "margin_of_error": f"{stats['margin_of_error']:.1%}",
                        "sample_size": stats["n"],
                    }
                    for model, stats in model_stats.items()
                },
                "pairwise_comparisons": [
                    {
                        "comparison": comp["models"],
                        "mean_difference": f"{comp['mean_difference']:+.1%}",
                        "p_value": f"{comp['p_value']:.6f}",
                        "significance_level": comp["significance"],
                        "statistically_significant": bool(comp["is_significant"]),
                    }
                    for comp in pairwise_comparisons
                ],
                "interpretation": {
                    "confidence_interval_meaning": "95% confidence interval - we are 95% confident the true performance lies within this range",
                    "significance_levels": {
                        "***": "p < 0.001 (highly significant)",
                        "**": "p < 0.01 (very significant)",
                        "*": "p < 0.05 (significant)",
                        "ns": "not significant",
                    },
                },
            }

        # Get dataset composition from any model (should be same for all)
        first_model = next(iter(all_results.values()))
        dataset_composition = first_model["summary"]["dataset_composition"]

        summary_data = {
            "dataset_composition": {
                "total_questions": dataset_composition["total_questions"],
                "binary_questions": f"{dataset_composition['bin_questions']} ({dataset_composition['bin_percentage']:.1f}%)",
                "mc_questions": f"{dataset_composition['mc_questions']} ({dataset_composition['mc_percentage']:.1f}%)",
            },
            "statistical_analysis": statistical_analysis,  # Add this line
            "key_findings_summary": {},
        }
        for model_name, model_results in all_results.items():
            total_test_cases = model_results["summary"]["overall"]["total_test_cases"]

            # Prepare tag group analysis - include ALL groups
            tag_group_data = {}
            tag_groups = model_results["tag_group_analysis"]
            for group, stats in tag_groups.items():
                tag_group_data[group] = {
                    "accuracy": f"{stats['jaccard_mean']:.1%}",
                    "percentage_of_total": f"{stats['percentage_of_total']:.1f}%",
                }

            summary_data["key_findings_summary"][model_name] = {
                "overall_metrics": {
                    "average_accuracy": f"{model_results['summary']['overall']['jaccard_accuracy']['mean']:.1%}",
                    "perfect_answers": f"{model_results['summary']['overall']['jaccard_accuracy']['perfect_rate']:.1%}",
                    "partial_answers": f"{model_results['summary']['overall']['jaccard_accuracy']['partial_rate']:.1%}",
                    "wrong_answers": f"{model_results['summary']['overall']['jaccard_accuracy']['wrong_rate']:.1%}",
                    "confidence_calibration": f"{model_results['summary']['overall']['confidence_calibration']['mean']:.1%}",
                    "hallucination_score": f"{model_results['summary']['overall']['hallucination_detection']['mean']:.1%}",
                    "average_response_time_ms": f"{model_results['summary']['overall']['response_time_ms']['mean']:.2f}",
                    "average_token_count": f"{model_results['summary']['overall']['token_count']['mean']:.2f}",
                },
                "tag_group_analysis": tag_group_data,
                "performance_by_answer_type": {
                    "binary": {
                        "dataset_questions": f"{dataset_composition['bin_questions']} ({dataset_composition['bin_percentage']:.1f}%)",
                        "successfully_evaluated": f"{model_results['summary']['binary']['total_test_cases']} ({model_results['summary']['binary']['dataset_percentage']:.1f}%)",
                        "average_accuracy": f"{model_results['summary']['binary']['jaccard_accuracy']['mean']:.1%}",
                        "average_response_time_ms": f"{model_results['summary']['binary']['response_time_ms']['mean']:.2f}",
                        "average_token_count": f"{model_results['summary']['binary']['token_count']['mean']:.2f}",
                    },
                    "mc": {
                        "dataset_questions": f"{dataset_composition['mc_questions']} ({dataset_composition['mc_percentage']:.1f}%)",
                        "successfully_evaluated": f"{model_results['summary']['mc']['total_test_cases']} ({model_results['summary']['mc']['dataset_percentage']:.1f}%)",
                        "average_accuracy": f"{model_results['summary']['mc']['jaccard_accuracy']['mean']:.1%}",
                        "average_response_time_ms": f"{model_results['summary']['mc']['response_time_ms']['mean']:.2f}",
                        "average_token_count": f"{model_results['summary']['mc']['token_count']['mean']:.2f}",
                    },
                },
            }

        # Use the extracted prefix for the output filename
        output_path = self.output_dir / f"{self.file_prefix}_key_findings_summary.json"
        with open(output_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"✅ Summary JSON saved to: {output_path}")

    def evaluate_all_models(self):
        """Evaluate all detected models"""
        all_results = {}

        for model_name in self.models.keys():
            try:
                results = self.evaluate_model(model_name)
                if results:
                    all_results[model_name] = results
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                continue

        # Add statistical analysis
        if len(all_results) > 1:
            self.add_statistical_analysis(all_results)  # <-- Add this line

        self._create_summary_json(all_results)
        return all_results


def main():
    """Main function to run complete evaluation"""
    csv_file = "output/llm_results/FamilyOWL_2hop/abstract_results_FINAL.csv"
    explanations_file = "output/FamilyOWL/2hop/Explanations.json"
    output_dir = "output/llm_results/FamilyOWL_2hop"

    print("🚀 Starting Ontology Reasoning Evaluation with Tag Group Analysis")
    print("=" * 70)
    print("📋 Focus: Analyzing performance by SHORTEST explanation tags")
    print("🎯 Groupings:")
    for group_name, tags in TAG_GROUPS.items():
        tag_descriptions = [TAG_DESCRIPTIONS.get(tag, tag) for tag in tags]
        print(f"   {group_name:<20}: {tags} ({', '.join(tag_descriptions)})")
    print("=" * 70)

    evaluator = CompleteEvaluator(
        csv_file=csv_file, explanations_file=explanations_file, output_dir=output_dir
    )

    # Run evaluation
    results = evaluator.evaluate_all_models()

    print("\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {evaluator.output_dir}")

    # Print final summary
    print(f"\n🎯 KEY FINDINGS SUMMARY:")
    print("=" * 50)

    for model_name, model_results in results.items():
        print(f"\n🤖 {model_name.upper()}:")

        # Overall Metrics
        summary_overall = model_results["summary"]["overall"]
        jaccard_overall = summary_overall["jaccard_accuracy"]
        print(f"   Overall:")
        print(f"     Average Accuracy: {jaccard_overall['mean']:.1%}")
        print(
            f"     Perfect Answers: {jaccard_overall['perfect_answers']}/{summary_overall['total_test_cases']} ({jaccard_overall['perfect_rate']:.1%})"
        )
        print(
            f"     Confidence Calibration: {summary_overall['confidence_calibration']['mean']:.1%}"
        )
        print(
            f"     Hallucination Score: {summary_overall['hallucination_detection']['mean']:.1%}"
        )

        # Show top performing tag groups
        tag_groups = model_results["tag_group_analysis"]
        sorted_groups = sorted(
            tag_groups.items(), key=lambda x: x[1]["jaccard_mean"], reverse=True
        )

        if len(sorted_groups) >= 1:
            print(f"\n   💪 Best performing groups:")
            for group, stats in sorted_groups[: min(3, len(sorted_groups))]:
                print(
                    f"      {group}: {stats['jaccard_mean']:.1%} accuracy ({stats['percentage_of_total']:.1f}% of queries)"
                )


if __name__ == "__main__":
    main()
