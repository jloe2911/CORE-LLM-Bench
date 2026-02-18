import pandas as pd
import json
import os


def verbalize_abox(json_data):
    output = []
    individuals = json_data.get("individuals", [])
    for ind in individuals:
        desc = ind.get("description", "")
        output.append(f"{desc}")
    output.append("")
    return "\n".join(output)


def parse_root_entity_and_get_verbalized_ont(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return verbalize_abox(data)
    except FileNotFoundError:
        return f"File not found: {path}"
    except json.JSONDecodeError:
        return f"Invalid JSON in file: {path}"
    except Exception as e:
        return f"Error processing {path}: {str(e)}"


def df_to_json(df):
    result = []

    group_cols = ["Task Type", "Answer Type", "Root Entity", "NL Context"]
    qa_cols = ["SPARQL Query", "NL Question", "Answer"]

    grouped = df.groupby(group_cols)

    for group_keys, group_df in grouped:
        task_type, answer_type, root_entity, nl_context = group_keys

        qas = group_df[qa_cols].to_dict(orient="records")

        entry = {
            "Task Type": task_type,
            "Answer Type": answer_type,
            "Root Entity": root_entity,
            "NL Context": nl_context,
            "QAs": qas,
        }

        result.append(entry)

    return result


def process_dataset(dataset, hop):
    print(f"Processing dataset={dataset}, hop={hop}")

    base_path = f"../data/output/{dataset}/{hop}"

    q_nl = pd.read_excel(f"{base_path}/SPARQL_questions_sampling_nl.xlsx")

    df = q_nl[
        [
            "Root Entity",
            "Task Type",
            "Answer Type",
            "SPARQL Query",
            "Question",
            "Answer",
        ]
    ].copy()
    df["NL Question"] = q_nl["Question"].values

    verbalized_path = f"../data/output/verbalized_ontologies/{dataset}_{hop}"

    df["NL Context"] = df["Root Entity"].apply(
        lambda root: parse_root_entity_and_get_verbalized_ont(
            os.path.join(verbalized_path, f"{root}.json")
        )
    )

    final_json = df_to_json(df)

    output_file = f"{dataset}_{hop}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

    print(f"Saved → {output_file}\n")


# =========================
# 🔥 Run Multiple Variants
# =========================

datasets = ["FamilyOWL", "OWL2Bench"]
hops = ["1hop", "2hop"]

for dataset in datasets:
    for hop in hops:
        try:
            process_dataset(dataset, hop)
        except Exception as e:
            print(f"Skipped {dataset}-{hop}: {e}")
