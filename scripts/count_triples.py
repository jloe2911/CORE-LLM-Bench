# -*- coding: utf-8 -*-
"""
Count raw OWL/RDF triples in TTL context files and SPARQL triple patterns in questions.
For benchmark context size tracking.
"""
import os
import sys
import re
import csv
import json
from pathlib import Path
from collections import defaultdict

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).resolve().parent.parent


def count_ttl_triples_rdflib(ttl_path):
    """Count triples using rdflib (most accurate)."""
    try:
        from rdflib import Graph
        g = Graph()
        g.parse(ttl_path, format='turtle')
        return len(g), "rdflib"
    except ImportError:
        return None, None
    except Exception as e:
        print(f"  rdflib error for {ttl_path}: {e}")
        return None, None


def count_ttl_triples_manual(ttl_path):
    """
    Count RDF triples in a Turtle file by parsing statement structure.
    Each ';' continues the same subject (new pred-obj = new triple).
    Each ',' continues the same subject+pred (new obj = new triple).
    Each '.' ends a statement group.
    """
    with open(ttl_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove full-line comments and directives
    lines = content.split('\n')
    cleaned = []
    for line in lines:
        s = line.strip()
        if s.startswith('#') or s.startswith('###'):
            continue
        if s.startswith('@prefix') or s.startswith('@base'):
            continue
        cleaned.append(line)
    content = '\n'.join(cleaned)

    # Remove string literals to avoid confusing punctuation
    content = re.sub(r'"[^"]*"(?:\^\^[^ ;.,\]]*)?', '""', content)
    content = re.sub(r"'[^']*'", "''", content)

    triple_count = 0
    in_statement = False

    for ch in content:
        if ch == '.':
            if in_statement:
                triple_count += 1
                in_statement = False
        elif ch == ';':
            if in_statement:
                triple_count += 1
        elif ch == ',':
            if in_statement:
                triple_count += 1
        elif ch in '[](){}':
            pass
        elif not ch.isspace():
            in_statement = True

    return triple_count, "manual"


def count_sparql_triple_patterns(sparql_query):
    """Count triple patterns in a SPARQL query.
    
    Each basic graph pattern {?s ?p ?o} = 1 triple pattern.
    Must not split on '.' inside URIs like <http://www.example.com/...>.
    """
    match = re.search(r'\{([^}]*)\}', sparql_query)
    if not match:
        return 0
    where_body = match.group(1).strip()
    if not where_body:
        return 0
    # Remove URIs (content inside angle brackets) to avoid splitting on dots within URIs
    cleaned = re.sub(r'<[^>]*>', '<URI>', where_body)
    # Remove string literals
    cleaned = re.sub(r'"[^"]*"', '""', cleaned)
    # Now split on '.' as triple pattern terminator
    patterns = [p.strip() for p in cleaned.split('.') if p.strip()]
    return len(patterns)


def main():
    print("=" * 80)
    print("CORE-LLM-Bench Triple Counter")
    print("=" * 80)

    # Check rdflib
    use_rdflib = False
    try:
        from rdflib import Graph
        use_rdflib = True
        print("[OK] rdflib available - using precise triple counting")
    except ImportError:
        print("[!!] rdflib not available - using manual TTL parser")

    # ---- TTL directories ----
    ttl_dirs = {}
    toy1 = project_root / "data" / "resources" / "toy_example_1hop"
    toy2 = project_root / "data" / "resources" / "toy_example_2hop"
    if toy1.exists():
        ttl_dirs["toy_example_1hop"] = toy1
    if toy2.exists():
        ttl_dirs["toy_example_2hop"] = toy2

    # ---- CSV files ----
    csv_files = {
        "toy_1hop_sampling": project_root / "data" / "output" / "toy_example" / "1hop" / "SPARQL_questions_sampling.csv",
    }

    results = {}

    # ==== CONTEXT TRIPLES ====
    print("\n" + "=" * 80)
    print("CONTEXT TRIPLES (OWL/RDF from TTL files)")
    print("=" * 80)

    for dir_name, ttl_dir in ttl_dirs.items():
        print(f"\n[DIR] {dir_name}: {ttl_dir}")
        print("-" * 60)

        ttl_files = sorted(ttl_dir.glob("*.ttl"))
        if not ttl_files:
            print("  No TTL files found")
            continue

        dir_results = {}
        for ttl_file in ttl_files:
            if use_rdflib:
                count, method = count_ttl_triples_rdflib(ttl_file)
                if count is None:
                    count, method = count_ttl_triples_manual(ttl_file)
            else:
                count, method = count_ttl_triples_manual(ttl_file)

            entity_name = ttl_file.stem
            dir_results[entity_name] = {
                "count": count, "method": method,
                "size_bytes": ttl_file.stat().st_size
            }
            print(f"  {entity_name:45s} | {count:4d} triples ({method}) | {ttl_file.stat().st_size:>6,d} bytes")

        counts = [r["count"] for r in dir_results.values()]
        print(f"\n  Summary for {dir_name}:")
        print(f"    Files:  {len(dir_results)}")
        print(f"    Total:  {sum(counts)} triples (across all files)")
        print(f"    Min:    {min(counts)}")
        print(f"    Max:    {max(counts)}")
        print(f"    Avg:    {sum(counts)/len(counts):.1f}")

        results[dir_name] = dir_results

    # ==== QUESTION TRIPLE PATTERNS ====
    print("\n" + "=" * 80)
    print("QUESTION TRIPLE PATTERNS (from SPARQL queries)")
    print("=" * 80)

    for csv_name, csv_path in csv_files.items():
        if not csv_path.exists():
            print(f"\n[!!] {csv_name}: not found at {csv_path}")
            continue

        print(f"\n[CSV] {csv_name}: {csv_path.name}")
        print("-" * 60)

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        pattern_counts = defaultdict(int)
        total_patterns = 0
        ask_count = 0
        select_count = 0

        for row in rows:
            sparql = row.get("SPARQL Query", "")
            n = count_sparql_triple_patterns(sparql)
            total_patterns += n
            pattern_counts[n] += 1
            if sparql.strip().upper().startswith("ASK"):
                ask_count += 1
            elif sparql.strip().upper().startswith("SELECT"):
                select_count += 1

        print(f"  Total questions:             {len(rows)}")
        print(f"  ASK queries:                 {ask_count}")
        print(f"  SELECT queries:              {select_count}")
        print(f"  Total SPARQL triple patterns: {total_patterns}")
        print(f"  Patterns per query:")
        for n, c in sorted(pattern_counts.items()):
            print(f"    {n} pattern(s): {c} queries")

        # Per-entity breakdown
        entity_q = defaultdict(lambda: {"questions": 0, "patterns": 0, "tbox": "?", "abox": "?"})
        for row in rows:
            entity = row.get("Root Entity", "Unknown")
            sparql = row.get("SPARQL Query", "")
            n = count_sparql_triple_patterns(sparql)
            entity_q[entity]["questions"] += 1
            entity_q[entity]["patterns"] += n
            entity_q[entity]["tbox"] = row.get("Size of ontology TBox", "?")
            entity_q[entity]["abox"] = row.get("Size of ontology ABox", "?")

        print(f"\n  Per-entity breakdown:")
        for entity, info in sorted(entity_q.items()):
            print(f"    {entity:45s} | {info['questions']:2d} Qs | {info['patterns']:2d} patterns | TBox={info['tbox']} ABox={info['abox']}")

    # ==== PER-QUERY TOTAL ====
    print("\n" + "=" * 80)
    print("PER-QUERY TOTAL: CONTEXT + QUESTION (toy_example_1hop)")
    print("=" * 80)

    csv_path = csv_files.get("toy_1hop_sampling")
    ttl_key = "toy_example_1hop"

    if csv_path and csv_path.exists() and ttl_key in results:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"\n{'Q#':>3s} | {'Root Entity':45s} | {'Ctx':>5s} | {'QP':>3s} | {'Total':>6s} | Type")
        print("-" * 85)

        for i, row in enumerate(rows):
            entity = row.get("Root Entity", "Unknown")
            sparql = row.get("SPARQL Query", "")
            n_patterns = count_sparql_triple_patterns(sparql)
            ctx = results[ttl_key].get(entity, {}).get("count", "?")
            total = (ctx + n_patterns) if isinstance(ctx, int) else "?"
            qtype = "ASK" if sparql.strip().upper().startswith("ASK") else "SEL"
            atype = row.get("Answer Type", "?")
            print(f"{i+1:3d} | {entity:45s} | {ctx:>5} | {n_patterns:>3d} | {total:>6} | {qtype}({atype})")

    # ==== INSTRUCTION ANALYSIS ====
    print("\n" + "=" * 80)
    print("INSTRUCTION / PROMPT TEMPLATE ANALYSIS")
    print("=" * 80)

    print("""
Source: api_calls.py :: create_context_specific_prompt()

EXAMPLES: There are NO few-shot examples in any prompt variant.

INSTRUCTIONS vary by TWO dimensions:
  1. context_mode: 'ttl' (SPARQL+TTL) vs 'json' (NL+Verbalized)
  2. answer_type:  'BIN' (binary TRUE/FALSE) vs 'MC' (multi-choice)

This creates 4 template combinations (2 x 2).
Within each combo, the instruction text is IDENTICAL for every query.

Variable parts per query are ONLY:
  - {query}:            the SPARQL query or NL question
  - {ontology_context}: the TTL file content or verbalized JSON content

TEMPLATE DIFFERENCES:
  - TTL mode opening:  "You are an expert in SPARQL and OWL ontologies..."
  - NL mode opening:   "You are an expert in ontologies..."
  - BIN format:        "ANSWER: [TRUE or FALSE]"
  - MC format:         "ANSWER: [Use LOCAL NAMES only, comma-separated]"
  + Both include CONFIDENCE and REASONING_STEPS fields.
""")

    # Instruction sizes
    templates = {
        "TTL+BIN": (
            "You are an expert in SPARQL and OWL ontologies. Analyze the TTL ontology "
            "and answer the SPARQL query precisely.\n\n"
            "CRITICAL: You MUST respond in exactly this format:\n"
            "ANSWER: [TRUE or FALSE]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "REASONING_STEPS: [distinct number of reasoning steps you used to get to the answer, "
            "indicating complexity of reasoning needed]\n\n"
            "ANSWER section: ONLY write TRUE or FALSE.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
            "REASONING_STEPS section: 1 = trivial/direct lookup, 10+ = complex multi-step reasoning.\n\n"
            "DO NOT include any additional text before or after this format.\n\n"
            "Question: \nContext: "
        ),
        "TTL+MC": (
            "You are an expert in SPARQL and OWL ontologies. Analyze the TTL ontology "
            "and answer the SPARQL query precisely.\n\n"
            "CRITICAL: You MUST respond in exactly this format:\n"
            "ANSWER: [Use LOCAL NAMES only, comma-separated]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "REASONING_STEPS: [distinct number of reasoning steps you used to get to the answer, "
            "indicating complexity of reasoning needed]\n\n"
            "ANSWER section: Use LOCAL NAMES only (e.g., 'Person', 'U0C4', 'caroline_lavinia_tubb_1840'), "
            "give all the possible answers.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
            "REASONING_STEPS section: 1 = trivial/direct lookup, 10+ = complex multi-step reasoning.\n\n"
            "DO NOT include any additional text before or after this format.\n\n"
            "Question: \nContext: "
        ),
        "NL+BIN": (
            "You are an expert in ontologies, answer the following question based on the provided "
            "ontological relationships. Reason through the ontological context and answer based on "
            "what you can infer from the context.\n\n"
            "CRITICAL: You MUST respond in exactly this format:\n"
            "ANSWER: [TRUE or FALSE]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "REASONING_STEPS: [distinct number of reasoning steps you used to get to the answer, "
            "indicating complexity of reasoning needed]\n\n"
            "ANSWER section: ONLY write TRUE or FALSE.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
            "REASONING_STEPS section: 1 = trivial/direct lookup, 10+ = complex multi-step reasoning.\n\n"
            "DO NOT include any additional text before or after this format.\n\n"
            "Question: \nContext: "
        ),
        "NL+MC": (
            "You are an expert in ontologies, answer the following question based on the provided "
            "ontological relationships. Reason through the ontological context and answer based on "
            "what you can infer from the context.\n\n"
            "CRITICAL: You MUST respond in exactly this format:\n"
            "ANSWER: [Use LOCAL NAMES only, comma-separated]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "REASONING_STEPS: [distinct number of reasoning steps you used to get to the answer, "
            "indicating complexity of reasoning needed]\n\n"
            "ANSWER section: Use LOCAL NAMES only (e.g., 'Person', 'U0C4', 'caroline_lavinia_tubb_1840'), "
            "give all the possible answers.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
            "REASONING_STEPS section: 1 = trivial/direct lookup, 10+ = complex multi-step reasoning.\n\n"
            "DO NOT include any additional text before or after this format.\n\n"
            "Question: \nContext: "
        ),
    }

    print("  Instruction overhead (fixed text, excluding query & context):")
    print(f"  {'Template':12s} | {'Chars':>6s} | {'Words':>5s} | {'~Tokens':>7s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}")
    for name, text in templates.items():
        chars = len(text)
        words = len(text.split())
        tokens = chars // 4  # rough estimate
        print(f"  {name:12s} | {chars:6d} | {words:5d} | {tokens:7d}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
