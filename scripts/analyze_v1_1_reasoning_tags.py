#!/usr/bin/env python3
"""Offline reasoning-tag difficulty analysis for CORE-LLM-Bench v1.1.

The analysis reads only the finalized public explanation schema and the frozen
corrected observation-level scores.  It never parses legacy TAG strings and it
never selects one tied minimum explanation arbitrarily.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PRIMITIVE_TAGS = tuple("DHINRST")
ADJUSTED_TAGS = ("H", "I", "R")
MODEL_REFERENCE = "GPT-5 mini"
REPRESENTATION_REFERENCE = "NL"
DATASET_REFERENCE = "FamilyOWL"
HOP_REFERENCE = "1hop"
MIN_INFERENTIAL_QUESTIONS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("release/v1.1.0-staging/benchmark"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("results/v1.1.0-evaluation-corrected-v1/csv/per_observation_scores.csv"),
    )
    parser.add_argument(
        "--defective-ar",
        type=Path,
        default=Path(
            "results/v1.1.0-evaluation-corrected-v1/audit/"
            "confirmed_defective_ar_questions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v1.1.0-reasoning-tag-analysis"),
    )
    return parser.parse_args()


def _canonical_tags(values: Iterable[str]) -> str:
    tags = set(values)
    invalid = tags - set(PRIMITIVE_TAGS)
    if invalid:
        raise ValueError(f"Invalid primitive reasoning tags: {sorted(invalid)}")
    return "".join(tag for tag in PRIMITIVE_TAGS if tag in tags)


def _proof_tags(proof: dict) -> set[str]:
    tags: list[str] = []
    for axiom in proof.get("axioms", []):
        tag = axiom.get("tag")
        if not isinstance(tag, str) or len(tag) != 1:
            raise ValueError(f"Invalid canonical axiom tag: {tag!r}")
        tags.append(tag)
    _canonical_tags(tags)
    return set(tags)


def _question_rows(benchmark_dir: Path) -> tuple[pd.DataFrame, dict]:
    paths = sorted(benchmark_dir.glob("*.json"))
    if len(paths) != 8:
        raise ValueError(f"Expected 8 benchmark JSON files, found {len(paths)}")

    raw_questions: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for context in payload:
            raw_questions.extend(context.get("QAs", []))
    if len(raw_questions) != 9_048:
        raise ValueError(f"Expected 9,048 questions, found {len(raw_questions):,}")

    by_id = {str(q["task_id"]): q for q in raw_questions}
    if len(by_id) != len(raw_questions):
        raise ValueError("Canonical task_id values are not unique")

    rows: list[dict] = []
    bqa_positive_basis = 0
    for q in raw_questions:
        task_id = str(q["task_id"])
        task = str(q["task_type"])
        label = str(q["gold_answer"]) if task == "BQA" else ""

        # FALSE rows carry paired-positive structural annotation in the public
        # schema. Resolve the positive explicitly so it can never be described
        # as a proof of FALSE.
        basis = q
        if task == "BQA" and label == "FALSE":
            positive_id = str(q.get("positive_task_id", ""))
            if positive_id not in by_id:
                raise ValueError(f"Missing paired positive for FALSE task {task_id}")
            basis = by_id[positive_id]
            if basis.get("gold_answer") != "TRUE":
                raise ValueError(f"Paired-positive target is not TRUE for task {task_id}")
            if basis.get("pair_group_id") != q.get("pair_group_id"):
                raise ValueError(f"BQA pair identifier mismatch for task {task_id}")
            bqa_positive_basis += 1

        complete = basis.get("complete_explanation")
        minima = complete.get("minimum_explanations", []) if isinstance(complete, dict) else []
        if not minima:
            raise ValueError(f"No tied/global minimum explanations for task {task_id}")
        proof_tag_sets = [_proof_tags(proof) for proof in minima]
        every_tags = set.intersection(*proof_tag_sets)
        any_tags = set.union(*proof_tag_sets)
        min_counts = {len(proof.get("axioms", [])) for proof in minima}
        if len(min_counts) != 1:
            raise ValueError(f"Tied minima have unequal size for task {task_id}")
        minimum_count = min_counts.pop()
        declared = int(q["raw_minimum_complete_primitive_tag_complexity"])
        if minimum_count != declared:
            raise ValueError(
                f"Declared minimum complexity mismatch for task {task_id}: "
                f"schema={minimum_count}, declared={declared}"
            )

        declared_tags = _canonical_tags(q["primitive_reasoning_tags"])
        if not set(any_tags).issubset(set(declared_tags)):
            raise ValueError(f"Minimum tags exceed declared tags for task {task_id}")
        if "M" in declared_tags or any("M" in tags for tags in proof_tag_sets):
            raise ValueError(f"M was encoded as primitive at task {task_id}")

        answer_groups = basis.get("answer_explanations", [])
        gold_size = len(answer_groups) if task == "OEQA" else math.nan
        if task == "OEQA" and gold_size < 1:
            raise ValueError(f"OEQA task {task_id} has no answer explanation groups")

        cluster_id = str(q.get("pair_group_id", "")) if task == "BQA" else f"oeqa-{task_id}"
        if not cluster_id:
            raise ValueError(f"Missing BQA pair_group_id for task {task_id}")
        rows.append(
            {
                "task_id": task_id,
                "semantic_key": str(q["semantic_key"]),
                "task": task,
                "dataset": str(q["dataset"]),
                "hop": str(q["hop"]),
                "bqa_label": label if task == "BQA" else "",
                "pair_group_id": str(q.get("pair_group_id", "")),
                "positive_task_id": str(q.get("positive_task_id", "")),
                "cluster_id": cluster_id,
                "gold_answer_set_size": gold_size,
                "tied_minimum_explanation_count": len(minima),
                "minimum_complete_primitive_tag_count": minimum_count,
                "tags_every_tied_minimum": _canonical_tags(every_tags),
                "tags_any_tied_minimum": _canonical_tags(any_tags),
                "m_status": str(q.get("m_status", "")),
                "structural_annotation_basis": (
                    "paired_positive_entailment" if task == "BQA" and label == "FALSE"
                    else "positive_entailment" if task == "BQA"
                    else "minimum_complete_gold_answer_set"
                ),
            }
        )

    frame = pd.DataFrame(rows).sort_values("task_id", key=lambda s: s.astype(int))
    if set("".join(frame["tags_any_tied_minimum"])) != set(PRIMITIVE_TAGS):
        raise ValueError("The seven expected primitive reasoning tags were not exercised")

    # TRUE/FALSE rows must share one structural annotation and cluster.
    bqa = frame[frame["task"] == "BQA"]
    pair_sizes = bqa.groupby("pair_group_id").size()
    labels = bqa.groupby("pair_group_id")["bqa_label"].agg(lambda x: set(x))
    structural = bqa.groupby("pair_group_id").agg(
        every=("tags_every_tied_minimum", "nunique"),
        any=("tags_any_tied_minimum", "nunique"),
        count=("minimum_complete_primitive_tag_count", "nunique"),
        ties=("tied_minimum_explanation_count", "nunique"),
    )
    if not (pair_sizes == 2).all() or not labels.map(lambda x: x == {"TRUE", "FALSE"}).all():
        raise ValueError("BQA TRUE/FALSE pairing is incomplete")
    if (structural != 1).any().any():
        raise ValueError("BQA TRUE/FALSE structural annotations differ within a pair")

    audit = {
        "benchmark_files": [str(path.as_posix()) for path in paths],
        "question_count": len(frame),
        "task_counts": frame["task"].value_counts().sort_index().to_dict(),
        "bqa_pair_count": int(bqa["pair_group_id"].nunique()),
        "false_questions_resolved_to_paired_positive": bqa_positive_basis,
        "questions_with_tied_minima": int((frame["tied_minimum_explanation_count"] > 1).sum()),
        "maximum_tied_minima": int(frame["tied_minimum_explanation_count"].max()),
        "declared_minimum_complexity_mismatches": 0,
        "primitive_tags_exercised": list(PRIMITIVE_TAGS),
        "m_excluded_from_primitive_tags": True,
    }
    return frame, audit


def _load_scores(path: Path, questions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scores = pd.read_csv(path, dtype={"task_id": str, "semantic_key": str})
    if len(scores) != 81_432:
        raise ValueError(f"Expected 81,432 corrected observations, found {len(scores):,}")
    keys = ["task_id", "model", "representation"]
    duplicate_count = int(scores.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate corrected observations")

    expected_models = {"GPT-5 mini", "Gemini 2.5 Flash-Lite", "Qwen3-30B-A3B-Instruct"}
    expected_reps = {"NL", "FS", "AR"}
    if set(scores["model"]) != expected_models or set(scores["representation"]) != expected_reps:
        raise ValueError("Corrected observations do not cover the expected models/representations")
    if set(scores["task"]) != {"BQA", "OEQA"} or set(scores["hop"]) != {"1hop", "2hop"}:
        raise ValueError("Corrected observations do not cover both tasks and extraction depths")

    qkeys = questions[["task_id", "semantic_key"]]
    merged = scores.merge(
        questions,
        on=["task_id", "semantic_key"],
        how="left",
        validate="many_to_one",
        suffixes=("_score", ""),
        indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"Found {unmatched} corrected observations without a benchmark match")
    merged = merged.drop(columns="_merge")
    if len(qkeys.merge(scores[["task_id", "semantic_key"]].drop_duplicates(), how="left", indicator=True).query("_merge != 'both'")):
        raise ValueError("Some benchmark questions have no corrected observations")
    for column in ("task", "dataset", "hop"):
        score_column = f"{column}_score"
        if score_column in merged and not (merged[score_column] == merged[column]).all():
            raise ValueError(f"Corrected observation {column} disagrees with canonical benchmark")

    for metric in ("answer_f1", "answer_exact_match"):
        merged[metric] = pd.to_numeric(merged[metric], errors="raise")
        if merged[metric].isna().any() or not merged[metric].between(0, 1).all():
            raise ValueError(f"Invalid corrected {metric} values")

    per_question = merged.groupby("task_id").size()
    if not (per_question == 9).all():
        raise ValueError("Each question must have exactly 3 models x 3 representations")
    audit = {
        "observation_count": len(merged),
        "duplicate_observation_count": duplicate_count,
        "unmatched_observation_count": unmatched,
        "models": sorted(merged["model"].unique()),
        "representations": sorted(merged["representation"].unique()),
        "tasks": sorted(merged["task"].unique()),
        "hops": sorted(merged["hop"].unique()),
        "malformed_but_accepted_observations_retained": int(
            (merged["observation_status"] == "malformed_response").sum()
        ),
    }
    return merged, audit


def _clustered_mean_ci(group: pd.DataFrame, metric: str) -> tuple[float, float, float, float, int]:
    values = group[metric].to_numpy(dtype=float)
    mean = float(values.mean())
    clusters = group["cluster_id"].astype(str).to_numpy()
    unique = np.unique(clusters)
    g = len(unique)
    n = len(values)
    if g < 2:
        return mean, math.nan, math.nan, math.nan, g
    score_sums = np.array([np.sum(values[clusters == cluster] - mean) for cluster in unique])
    se = float(np.sqrt((g / (g - 1)) * np.sum(score_sums**2) / (n**2)))
    critical = float(stats.t.ppf(0.975, df=g - 1))
    return mean, se, max(0.0, mean - critical * se), min(1.0, mean + critical * se), g


def _descriptive(
    observations: pd.DataFrame,
    group_columns: Sequence[str],
    basis: str,
    scope: str,
) -> pd.DataFrame:
    tag_column = f"tags_{basis}_tied_minimum"
    rows: list[dict] = []
    for tag in PRIMITIVE_TAGS:
        selected = observations[observations[tag_column].str.contains(tag, regex=False)]
        if selected.empty:
            continue
        grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
        for key, group in selected.groupby(grouper, observed=True, sort=True):
            key = (key,) if len(group_columns) == 1 else key
            row = dict(zip(group_columns, key))
            f1, f1_se, f1_low, f1_high, n_clusters = _clustered_mean_ci(group, "answer_f1")
            em, em_se, em_low, em_high, _ = _clustered_mean_ci(group, "answer_exact_match")
            row.update(
                {
                    "scope": scope,
                    "tag_basis": basis,
                    "tag": tag,
                    "n_unique_questions": int(group["task_id"].nunique()),
                    "n_observations": len(group),
                    "n_clusters": n_clusters,
                    "answer_f1_mean": f1,
                    "answer_f1_se_clustered": f1_se,
                    "answer_f1_ci95_low": f1_low,
                    "answer_f1_ci95_high": f1_high,
                    "answer_exact_match_mean": em,
                    "answer_exact_match_se_clustered": em_se,
                    "answer_exact_match_ci95_low": em_low,
                    "answer_exact_match_ci95_high": em_high,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _frequency_tables(questions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frequency_rows: list[dict] = []
    co_rows: list[dict] = []
    for basis in ("every", "any"):
        column = f"tags_{basis}_tied_minimum"
        for task, task_frame in questions.groupby("task", sort=True):
            total = len(task_frame)
            counts = {tag: int(task_frame[column].str.contains(tag, regex=False).sum()) for tag in PRIMITIVE_TAGS}
            for tag, count in counts.items():
                frequency_rows.append(
                    {
                        "tag_basis": basis,
                        "task": task,
                        "tag": tag,
                        "n_unique_questions": count,
                        "total_questions": total,
                        "question_percent": 100 * count / total,
                        "sparse_for_inference": count < MIN_INFERENTIAL_QUESTIONS,
                        "ubiquitous": count == total,
                    }
                )
            for tag_a, tag_b in combinations_with_replacement(PRIMITIVE_TAGS, 2):
                both = int(
                    (
                        task_frame[column].str.contains(tag_a, regex=False)
                        & task_frame[column].str.contains(tag_b, regex=False)
                    ).sum()
                )
                co_rows.append(
                    {
                        "tag_basis": basis,
                        "task": task,
                        "tag_a": tag_a,
                        "tag_b": tag_b,
                        "n_unique_questions": both,
                        "total_questions": total,
                        "question_percent": 100 * both / total,
                        "percent_of_tag_a_questions": 100 * both / counts[tag_a] if counts[tag_a] else math.nan,
                        "percent_of_tag_b_questions": 100 * both / counts[tag_b] if counts[tag_b] else math.nan,
                    }
                )
    return pd.DataFrame(frequency_rows), pd.DataFrame(co_rows)


def _raw_contrasts(observations: pd.DataFrame, basis: str, scope: str) -> pd.DataFrame:
    rows: list[dict] = []
    column = f"tags_{basis}_tied_minimum"
    for (task, representation), frame in observations.groupby(["task", "representation"], sort=True):
        question_frame = frame.drop_duplicates("task_id")
        for tag in PRIMITIVE_TAGS:
            n_present = int(question_frame[column].str.contains(tag, regex=False).sum())
            n_absent = len(question_frame) - n_present
            for metric in ("answer_f1", "answer_exact_match"):
                row = {
                    "scope": scope,
                    "tag_basis": basis,
                    "task": task,
                    "representation": representation,
                    "tag": tag,
                    "outcome": metric,
                    "reference_group": f"questions without {tag}",
                    "n_present_questions": n_present,
                    "n_absent_questions": n_absent,
                    "estimable": False,
                    "reason_not_estimable": "",
                }
                if n_present == len(question_frame):
                    row["reason_not_estimable"] = "tag ubiquitous"
                elif n_present < MIN_INFERENTIAL_QUESTIONS:
                    row["reason_not_estimable"] = "tag sparse; descriptive only"
                elif n_absent < MIN_INFERENTIAL_QUESTIONS:
                    row["reason_not_estimable"] = "absence group sparse; descriptive only"
                else:
                    work = frame.copy()
                    work["tag_present"] = work[column].str.contains(tag, regex=False).astype(int)
                    fit = smf.ols(f"{metric} ~ tag_present", data=work).fit(
                        cov_type="cluster", cov_kwds={"groups": work["cluster_id"]}
                    )
                    row.update(
                        {
                            "estimable": True,
                            "effect_present_minus_absent": float(fit.params["tag_present"]),
                            "standard_error_clustered": float(fit.bse["tag_present"]),
                            "ci95_low": float(fit.conf_int().loc["tag_present", 0]),
                            "ci95_high": float(fit.conf_int().loc["tag_present", 1]),
                            "p_value": float(fit.pvalues["tag_present"]),
                        }
                    )
                rows.append(row)
    result = pd.DataFrame(rows)
    result["p_value_holm"] = math.nan
    for _, indexes in result[result["estimable"]].groupby(
        ["scope", "tag_basis", "task", "outcome"]
    ).groups.items():
        pvalues = result.loc[indexes, "p_value"].to_numpy(float)
        result.loc[indexes, "p_value_holm"] = multipletests(pvalues, method="holm")[1]
    return result


def _adjusted_models(observations: pd.DataFrame, basis: str, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    column = f"tags_{basis}_tied_minimum"
    rep_term = f'C(representation, Treatment(reference="{REPRESENTATION_REFERENCE}"))'
    model_term = f'C(model, Treatment(reference="{MODEL_REFERENCE}"))'
    base_terms = [
        *[f"tag_{tag}" for tag in ADJUSTED_TAGS],
        model_term,
        rep_term,
        f'C(dataset, Treatment(reference="{DATASET_REFERENCE}"))',
        f'C(hop, Treatment(reference="{HOP_REFERENCE}"))',
        *[f"tag_{tag}:{rep_term}" for tag in ADJUSTED_TAGS],
        *[f"tag_{tag}:{model_term}" for tag in ADJUSTED_TAGS],
    ]
    for task, frame in observations.groupby("task", sort=True):
        work = frame.copy()
        for tag in ADJUSTED_TAGS:
            work[f"tag_{tag}"] = work[column].str.contains(tag, regex=False).astype(int)
        if task == "BQA":
            task_terms = ['C(bqa_label, Treatment(reference="FALSE"))']
        else:
            work["gold_answer_set_size_centered"] = (
                work["gold_answer_set_size"] - work["gold_answer_set_size"].mean()
            )
            task_terms = ["gold_answer_set_size_centered"]
        for outcome in ("answer_f1", "answer_exact_match"):
            formula = f"{outcome} ~ " + " + ".join(base_terms + task_terms)
            ordinary_fit = smf.ols(formula, data=work).fit()
            rank = int(np.linalg.matrix_rank(ordinary_fit.model.exog))
            columns = int(ordinary_fit.model.exog.shape[1])
            condition = float(np.linalg.cond(ordinary_fit.model.exog))
            counts = {
                tag: int(work.drop_duplicates("task_id")[f"tag_{tag}"].sum())
                for tag in ADJUSTED_TAGS
            }
            estimable = rank == columns and condition < 1_000 and all(
                count >= MIN_INFERENTIAL_QUESTIONS for count in counts.values()
            )
            diagnostic = {
                "scope": scope,
                "tag_basis": basis,
                "task": task,
                "outcome": outcome,
                "formula": formula,
                "cluster_unit": "BQA pair_group_id" if task == "BQA" else "OEQA question",
                "n_observations": len(work),
                "n_clusters": int(work["cluster_id"].nunique()),
                "design_rank": rank,
                "design_columns": columns,
                "condition_number": condition,
                "tag_question_counts": json.dumps(counts, sort_keys=True),
                "estimable": estimable,
                "status": "estimated" if estimable else "not estimated: unstable or sparse design",
            }
            diagnostic_rows.append(diagnostic)
            if not estimable:
                continue
            fit = ordinary_fit.get_robustcov_results(
                cov_type="cluster", groups=work["cluster_id"], use_correction=True
            )
            conf = fit.conf_int()
            for index, term in enumerate(fit.model.exog_names):
                focal = "tag_" in term
                if focal and "C(representation" in term:
                    term_family = "representation_by_tag_interaction"
                elif focal and "C(model" in term:
                    term_family = "model_by_tag_interaction"
                elif focal:
                    term_family = "tag_main_association"
                else:
                    term_family = "adjustment"
                coefficient_rows.append(
                    {
                        "scope": scope,
                        "tag_basis": basis,
                        "task": task,
                        "outcome": outcome,
                        "term": term,
                        "term_family": term_family,
                        "reference_groups": (
                            f"model={MODEL_REFERENCE}; representation={REPRESENTATION_REFERENCE}; "
                            f"dataset={DATASET_REFERENCE}; hop={HOP_REFERENCE}; "
                            + ("BQA label=FALSE" if task == "BQA" else "gold-answer-set size at task mean")
                        ),
                        "effect": float(fit.params[index]),
                        "standard_error_clustered": float(fit.bse[index]),
                        "ci95_low": float(conf[index, 0]),
                        "ci95_high": float(conf[index, 1]),
                        "p_value": float(fit.pvalues[index]),
                        "p_value_holm_focal": math.nan,
                    }
                )
    coefficients = pd.DataFrame(coefficient_rows)
    if not coefficients.empty:
        focal = coefficients["term_family"] != "adjustment"
        for _, indexes in coefficients[focal].groupby(
            ["scope", "tag_basis", "task", "outcome"]
        ).groups.items():
            pvalues = coefficients.loc[indexes, "p_value"].to_numpy(float)
            coefficients.loc[indexes, "p_value_holm_focal"] = multipletests(
                pvalues, method="holm"
            )[1]
    return coefficients, pd.DataFrame(diagnostic_rows)


def _format_ci(row: pd.Series, metric: str = "answer_f1") -> str:
    return (
        f"{row[f'{metric}_mean']:.3f} "
        f"[{row[f'{metric}_ci95_low']:.3f}, {row[f'{metric}_ci95_high']:.3f}]"
    )


def _latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def _write_latex_tables(
    output_dir: Path,
    primary: pd.DataFrame,
    frequencies: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> None:
    latex_dir = output_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    main = primary[
        (primary["scope"] == "all") & (primary["tag_basis"] == "every")
    ].copy()
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Task & Tag & NL F1 (95\% CI) & FS F1 (95\% CI) & AR F1 (95\% CI) \\",
        r"\midrule",
    ]
    for task in ("BQA", "OEQA"):
        for tag in PRIMITIVE_TAGS:
            subset = main[(main["task"] == task) & (main["tag"] == tag)]
            if subset.empty:
                continue
            by_rep = subset.set_index("representation")
            cells = [_format_ci(by_rep.loc[rep]) if rep in by_rep.index else "--" for rep in ("NL", "FS", "AR")]
            lines.append(
                f"{task} & {tag} & " + " & ".join(cells) + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (latex_dir / "main_tag_performance.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    freq = frequencies[(frequencies["tag_basis"] == "every")]
    lines = [r"\begin{tabular}{llrr}", r"\toprule", r"Task & Tag & Questions & Percent \\", r"\midrule"]
    for row in freq.itertuples(index=False):
        lines.append(
            f"{row.task} & {row.tag} & {row.n_unique_questions:,} & {row.question_percent:.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (latex_dir / "tag_frequencies.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    focal = adjusted[
        (adjusted["scope"] == "all")
        & (adjusted["tag_basis"] == "every")
        & (adjusted["outcome"] == "answer_f1")
        & (adjusted["term_family"] != "adjustment")
    ]
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Task & Term & Effect & 95\% CI & Holm $p$ \\",
        r"\midrule",
    ]
    for row in focal.itertuples(index=False):
        lines.append(
            f"{row.task} & {_latex_escape(row.term)} & {row.effect:.3f} & "
            f"[{row.ci95_low:.3f}, {row.ci95_high:.3f}] & {row.p_value_holm_focal:.3g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (latex_dir / "adjusted_tag_associations.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_main(output_dir: Path, primary: pd.DataFrame, frequency: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = primary[(primary["scope"] == "all") & (primary["tag_basis"] == "every")]
    counts = frequency[frequency["tag_basis"] == "every"].set_index(["task", "tag"])
    colors = {"NL": "#1f77b4", "FS": "#ff7f0e", "AR": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    offsets = {"NL": -0.22, "FS": 0.0, "AR": 0.22}
    for axis, task in zip(axes, ("BQA", "OEQA")):
        present_tags = [tag for tag in PRIMITIVE_TAGS if ((task, tag) in counts.index and counts.loc[(task, tag), "n_unique_questions"] > 0)]
        x = np.arange(len(present_tags))
        for rep in ("NL", "FS", "AR"):
            subset = data[(data["task"] == task) & (data["representation"] == rep)].set_index("tag")
            y = np.array([subset.loc[tag, "answer_f1_mean"] for tag in present_tags])
            lower = np.array([subset.loc[tag, "answer_f1_ci95_low"] for tag in present_tags])
            upper = np.array([subset.loc[tag, "answer_f1_ci95_high"] for tag in present_tags])
            axis.errorbar(
                x + offsets[rep], y, yerr=np.vstack([y - lower, upper - y]),
                marker="o", linestyle="none", capsize=3, color=colors[rep], label=rep,
            )
        labels = [f"{tag}\n(n={int(counts.loc[(task, tag), 'n_unique_questions']):,})" for tag in present_tags]
        axis.set_xticks(x, labels)
        axis.set_title(task)
        axis.set_xlabel("Tag in every tied minimum explanation")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, 1.03)
    axes[0].set_ylabel("Answer F1 (pair/question-clustered 95% CI)")
    axes[1].legend(title="Representation", loc="lower left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"reasoning_tag_f1_by_representation.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_models(output_dir: Path, model_table: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    data = model_table[(model_table["scope"] == "all") & (model_table["tag_basis"] == "every")]
    models = sorted(data["model"].unique())
    colors = {model: color for model, color in zip(models, ("#4c78a8", "#f58518", "#54a24b"))}
    offsets = {model: offset for model, offset in zip(models, (-0.18, 0.0, 0.18))}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
    for row_index, task in enumerate(("BQA", "OEQA")):
        for col_index, rep in enumerate(("NL", "FS", "AR")):
            axis = axes[row_index, col_index]
            subset = data[(data["task"] == task) & (data["representation"] == rep)]
            tags = [tag for tag in PRIMITIVE_TAGS if tag in set(subset["tag"])]
            x = np.arange(len(tags))
            for model_index, model in enumerate(models):
                selected = subset[subset["model"] == model].set_index("tag")
                y = np.array([selected.loc[tag, "answer_f1_mean"] for tag in tags])
                lower = np.array([selected.loc[tag, "answer_f1_ci95_low"] for tag in tags])
                upper = np.array([selected.loc[tag, "answer_f1_ci95_high"] for tag in tags])
                axis.errorbar(
                    x + offsets[model], y, yerr=np.vstack([y - lower, upper - y]),
                    marker="o", linestyle="none", capsize=2, label=model,
                    color=colors[model], alpha=0.9,
                )
            axis.set_xticks(x, tags)
            axis.set_title(f"{task} / {rep}")
            axis.grid(axis="y", alpha=0.2)
            axis.set_ylim(0, 1.03)
    axes[0, 0].legend(fontsize=8)
    fig.supylabel("Mean Answer F1")
    fig.supxlabel("Tag in every tied minimum explanation")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"model_specific_tag_f1.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(
    output_dir: Path,
    primary: pd.DataFrame,
    frequencies: pd.DataFrame,
    adjusted: pd.DataFrame,
    diagnostics: pd.DataFrame,
    defective_sensitivity: pd.DataFrame,
) -> None:
    main = primary[(primary["scope"] == "all") & (primary["tag_basis"] == "every")]
    contrast_lines: list[str] = []
    for task in ("BQA", "OEQA"):
        task_rows = main[
            (main["task"] == task)
            & (main["tag"] != "D")
            & (main["n_unique_questions"] >= MIN_INFERENTIAL_QUESTIONS)
        ]
        for rep in ("NL", "FS", "AR"):
            selected = task_rows[task_rows["representation"] == rep]
            if selected.empty:
                continue
            lowest = selected.loc[selected["answer_f1_mean"].idxmin()]
            highest = selected.loc[selected["answer_f1_mean"].idxmax()]
            contrast_lines.append(
                f"- {task}/{rep}: lowest tag-specific mean was {lowest['tag']} "
                f"({lowest['answer_f1_mean']:.3f}, n={int(lowest['n_unique_questions']):,}); "
                f"highest was {highest['tag']} ({highest['answer_f1_mean']:.3f})."
            )
    sparse = frequencies[
        (frequencies["tag_basis"] == "every")
        & frequencies["sparse_for_inference"]
        & (frequencies["n_unique_questions"] > 0)
    ]
    sparse_text = ", ".join(
        f"{row.task}-{row.tag} (n={row.n_unique_questions})" for row in sparse.itertuples(index=False)
    )
    max_delta = defective_sensitivity["absolute_f1_mean_change"].max()
    every_rows = primary[
        (primary["scope"] == "all") & (primary["tag_basis"] == "every")
    ]
    any_rows = primary[
        (primary["scope"] == "all") & (primary["tag_basis"] == "any")
    ]
    tied_compare = every_rows.merge(
        any_rows,
        on=["task", "representation", "tag"],
        suffixes=("_every", "_any"),
    )
    max_tied_delta = (
        tied_compare["answer_f1_mean_any"] - tied_compare["answer_f1_mean_every"]
    ).abs().max()
    rep_interactions = adjusted[
        (adjusted["scope"] == "all")
        & (adjusted["tag_basis"] == "every")
        & (adjusted["term_family"] == "representation_by_tag_interaction")
        & (adjusted["outcome"] == "answer_f1")
        & (adjusted["p_value_holm_focal"] < 0.05)
    ]
    model_interactions = adjusted[
        (adjusted["scope"] == "all")
        & (adjusted["tag_basis"] == "every")
        & (adjusted["term_family"] == "model_by_tag_interaction")
        & (adjusted["outcome"] == "answer_f1")
        & (adjusted["p_value_holm_focal"] < 0.05)
    ]
    rep_interaction_text = (
        "; ".join(f"{r.task}: {r.term} ({r.effect:+.3f})" for r in rep_interactions.itertuples(index=False))
        if len(rep_interactions) else "None survived Holm correction."
    )
    model_interaction_text = (
        "; ".join(f"{r.task}: {r.term} ({r.effect:+.3f})" for r in model_interactions.itertuples(index=False))
        if len(model_interactions) else "None survived Holm correction."
    )
    report = f"""# CORE-LLM-Bench v1.1 reasoning-tag difficulty analysis

## Scope and method

This offline analysis joins 9,048 canonical questions to all 81,432 corrected accepted observations. It uses the final `complete_explanation.minimum_explanations` schema. The primary tag definition is presence in every tied minimum; presence in any tied minimum is reported as sensitivity. BQA FALSE rows use their paired TRUE entailment only as structural annotation, never as a proof of FALSE. OEQA minima cover the complete gold-answer set and retain shared-axiom deduplication. `M` is excluded.

Intervals for descriptive means cluster BQA at the TRUE/FALSE pair and OEQA at the question. Adjusted linear models cluster on the same units and include tag, model, representation, dataset, hop, task-specific covariates, representation-by-tag interactions, and model-by-tag interactions. Coefficients are associations, not causal effects or evidence that a model executed the tagged operation.

## Descriptive findings

{chr(10).join(contrast_lines)}

The comparisons above omit `D` and sparse tags. `D` occurs in every question and is not discriminating. Sparse task-tag cells are {sparse_text}; they are descriptive only. Raw tag means remain confounded by tag co-occurrence and benchmark composition.

## Adjusted interactions

Holm-corrected representation-by-tag F1 interactions: {rep_interaction_text}

Holm-corrected model-by-tag F1 interactions: {model_interaction_text}

All adjusted-design diagnostics are recorded in `csv/model_diagnostics.csv`; {int(diagnostics['estimable'].sum())} of {len(diagnostics)} planned fits passed the rank, condition-number, and frequency gates. Reference groups are explicit in `csv/adjusted_associations.csv`.

## Sensitivity and limitations

Changing the tag definition from EVERY to ANY tied minimum changed a tag/representation F1 mean by at most {max_tied_delta:.6f}; full membership and estimate changes are in `csv/tied_minimum_sensitivity.csv`. Excluding the three confirmed defective AR questions (nine observations) changed any primary tag/representation F1 mean by at most {max_delta:.6f}. The analysis does not imply exhaustive semantic validation of other AR contexts. Existing complexity-bin outputs were not modified and remain optional supplementary context.
"""
    (output_dir / "INTERPRETATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output_dir
    csv_dir = output / "csv"
    audit_dir = output / "audit"
    csv_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    questions, question_audit = _question_rows(args.benchmark_dir)
    observations, observation_audit = _load_scores(args.scores, questions)
    defective = pd.read_csv(args.defective_ar, dtype={"task_id": str})
    defective_ids = set(defective["task_id"])
    if defective_ids != {"1274", "4094", "5165"}:
        raise ValueError(f"Unexpected confirmed-defective AR task IDs: {sorted(defective_ids)}")
    flagged_ids = set(
        observations.loc[observations["confirmed_defective_ar_prompt"] == 1, "task_id"]
    )
    if flagged_ids != defective_ids:
        raise ValueError("Corrected observation defect flags disagree with the audit list")
    without_defective = observations[
        ~(
            observations["task_id"].isin(defective_ids)
            & (observations["representation"] == "AR")
        )
    ].copy()
    if len(observations) - len(without_defective) != 9:
        raise ValueError("Expected exactly 9 model observations for three defective AR questions")

    frequencies, cooccurrence = _frequency_tables(questions)
    tables: dict[str, pd.DataFrame] = {}
    groupings = {
        "descriptive_by_task_tag_representation": ["task", "representation"],
        "descriptive_by_model": ["task", "representation", "model"],
        "descriptive_by_dataset_hop": ["task", "representation", "dataset", "hop"],
        "bqa_label_supplement": ["task", "representation", "bqa_label"],
        "oeqa_gold_set_size": ["task", "representation", "gold_answer_set_size"],
    }
    for name, columns in groupings.items():
        source = observations
        if name == "bqa_label_supplement":
            source = source[source["task"] == "BQA"]
        elif name == "oeqa_gold_set_size":
            source = source[source["task"] == "OEQA"]
        pieces = [_descriptive(source, columns, basis, "all") for basis in ("every", "any")]
        tables[name] = pd.concat(pieces, ignore_index=True)

    primary_excluded = _descriptive(
        without_defective, ["task", "representation"], "every", "exclude_confirmed_defective_ar"
    )
    primary_all = tables["descriptive_by_task_tag_representation"]
    primary_every = primary_all[(primary_all["scope"] == "all") & (primary_all["tag_basis"] == "every")]
    sensitivity = primary_every.merge(
        primary_excluded,
        on=["task", "representation", "tag"],
        suffixes=("_all", "_without_defective"),
        validate="one_to_one",
    )
    sensitivity["answer_f1_mean_change"] = (
        sensitivity["answer_f1_mean_without_defective"] - sensitivity["answer_f1_mean_all"]
    )
    sensitivity["absolute_f1_mean_change"] = sensitivity["answer_f1_mean_change"].abs()
    sensitivity["answer_exact_match_mean_change"] = (
        sensitivity["answer_exact_match_mean_without_defective"]
        - sensitivity["answer_exact_match_mean_all"]
    )

    every = primary_all[primary_all["tag_basis"] == "every"]
    any_ = primary_all[primary_all["tag_basis"] == "any"]
    tied_sensitivity = every.merge(
        any_, on=["scope", "task", "representation", "tag"], suffixes=("_every", "_any")
    )
    tied_sensitivity["answer_f1_mean_any_minus_every"] = (
        tied_sensitivity["answer_f1_mean_any"] - tied_sensitivity["answer_f1_mean_every"]
    )
    tied_sensitivity["n_unique_questions_any_minus_every"] = (
        tied_sensitivity["n_unique_questions_any"] - tied_sensitivity["n_unique_questions_every"]
    )

    raw_parts: list[pd.DataFrame] = []
    adjusted_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    for scope, frame in (("all", observations), ("exclude_confirmed_defective_ar", without_defective)):
        for basis in ("every", "any"):
            raw_parts.append(_raw_contrasts(frame, basis, scope))
            coefficients, diagnostics = _adjusted_models(frame, basis, scope)
            adjusted_parts.append(coefficients)
            diagnostic_parts.append(diagnostics)
    raw_contrasts = pd.concat(raw_parts, ignore_index=True)
    adjusted = pd.concat(adjusted_parts, ignore_index=True)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True)

    questions.to_csv(csv_dir / "per_question_minimum_explanation_tags.csv", index=False)
    frequencies.to_csv(csv_dir / "tag_frequencies.csv", index=False)
    cooccurrence.to_csv(csv_dir / "tag_cooccurrence.csv", index=False)
    for name, table in tables.items():
        table.to_csv(csv_dir / f"{name}.csv", index=False)
    raw_contrasts.to_csv(csv_dir / "raw_tag_contrasts.csv", index=False)
    adjusted.to_csv(csv_dir / "adjusted_associations.csv", index=False)
    diagnostics.to_csv(csv_dir / "model_diagnostics.csv", index=False)
    tied_sensitivity.to_csv(csv_dir / "tied_minimum_sensitivity.csv", index=False)
    sensitivity.to_csv(csv_dir / "defective_ar_sensitivity.csv", index=False)

    _write_latex_tables(output, primary_all, frequencies, adjusted)
    _plot_main(output, primary_all, frequencies)
    _plot_models(output, tables["descriptive_by_model"])
    _write_report(output, primary_all, frequencies, adjusted, diagnostics, sensitivity)

    validation = {
        **question_audit,
        **observation_audit,
        "confirmed_defective_ar_task_ids": sorted(defective_ids, key=int),
        "observations_excluded_in_defective_ar_sensitivity": 9,
        "primary_tag_basis": "present in every tied minimum complete explanation",
        "sensitivity_tag_basis": "present in any tied minimum complete explanation",
        "descriptive_interval": "95% t interval using pair/question-clustered sandwich SE",
        "adjusted_analysis": "cluster-robust OLS associations; Holm correction for focal terms",
        "rare_tag_threshold_questions": MIN_INFERENTIAL_QUESTIONS,
        "adjusted_tags": list(ADJUSTED_TAGS),
        "rare_tags_descriptive_only": ["N", "S", "T"],
        "d_ubiquitous_and_excluded_from_adjusted_models": True,
        "canonical_inputs_modified": False,
    }
    (audit_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    input_paths = [*sorted(args.benchmark_dir.glob("*.json")), args.scores, args.defective_ar]
    output_paths = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "analysis": "CORE-LLM-Bench v1.1 reasoning-tag difficulty",
        "inputs": [
            {"path": path.as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in input_paths
        ],
        "outputs": [
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
            if path.name != "REPRODUCIBILITY_MANIFEST.json"
        ],
    }
    (output / "REPRODUCIBILITY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
