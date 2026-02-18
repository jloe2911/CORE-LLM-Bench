import os
import time
import sys
import pandas as pd
from datetime import datetime
import openai
from dotenv import load_dotenv
from tqdm import tqdm
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
CONFIG = {
    "temperature": 0.0,
    "max_tokens": 4096,
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "checkpoint_frequency": 50,
    "api_delay": 0.1,
    "max_retries": 3,
    "retry_delay": 2.0,
}

# Set your API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

# Prompt templates
PROMPT_TEMPLATES = {
    "binary": """Assume you have to rephrase the following SPARQL query as a true/false (binary) natural-language question for non-technical humans. Do not include birth dates or birth years in the question.
Use the {language_string} language. SPARQL query: 
{sparql}""",
    "membership": """Assume you have to rephrase the following SPARQL query as a multiple or single choice natural-language question for non-technical humans. Avoid technical phrases like 'type of entity' or 'rdf:type'. Ask what class or classes the individual belongs to. Do not include birth dates or birth years in the question. 
Use the {language_string} language. SPARQL query:
{sparql}""",
    "property": """Assume you have to rephrase the following SPARQL query as a multiple or single choice natural-language question for non-technical humans. Describe the relationship or attributes involved, and avoid SPARQL or ontology jargon. Do not include birth dates or birth years in the question.
Use the {language_string} language. SPARQL query:
{sparql}""",
}


def get_model_version_info(response):
    """Extract model version information from response."""
    model_version = "Unknown"
    if hasattr(response, "model"):
        model_version = response.model
    if hasattr(response, "system_fingerprint"):
        model_version += f" (fingerprint: {response.system_fingerprint})"
    return model_version


def get_prompt_template(answer_type, task_type=None):
    """Get appropriate prompt template based on answer type and task type."""
    if answer_type == "Binary":
        return PROMPT_TEMPLATES["binary"]
    elif answer_type == "Multi Choice":
        if task_type and task_type.strip() == "Membership":
            return PROMPT_TEMPLATES["membership"]
        else:
            return PROMPT_TEMPLATES["property"]
    else:
        raise ValueError(f"Unknown Answer Type: {answer_type}")


def make_api_call_with_retry(client, messages, gen_params, max_retries=3):
    """Make API call with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, **gen_params
            )
            return response.choices[0].message.content, get_model_version_info(response)
        except Exception as e:
            logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(CONFIG["retry_delay"] * (attempt + 1))
            else:
                logger.error(f"All {max_retries} API call attempts failed")
                return f"[ERROR] {str(e)}", "Error - Unable to retrieve"


def validate_dataframe(df):
    """Validate input DataFrame has required columns."""
    required_columns = ["SPARQL Query"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV must contain columns: {missing_cols}")

    df["Language"] = df.get("Language", "English")
    df["Answer Type"] = df.get("Answer Type", "Binary")

    logger.info(f"Loaded DataFrame with {len(df)} rows")
    return df


def save_checkpoint(df, logs, timestamp, checkpoint_num):
    """Save intermediate results."""
    checkpoint_dir = Path(f"checkpoints_{timestamp}")
    checkpoint_dir.mkdir(exist_ok=True)

    df.to_csv(
        checkpoint_dir / f"checkpoint_{checkpoint_num}_questions.csv", index=False
    )
    pd.DataFrame(logs).to_csv(
        checkpoint_dir / f"checkpoint_{checkpoint_num}_log.csv", index=False
    )
    logger.info(f"Checkpoint {checkpoint_num} saved")


def create_output_directory(timestamp):
    """Create organized output directory."""
    output_dir = Path(f"2hop_family_output_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def main():
    # Load and validate data
    try:
        df = pd.read_csv("../SPARQL_questions_sampling_2hop.csv")
        df = validate_dataframe(df)
    except FileNotFoundError:
        logger.error("Input CSV file not found")
        return
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_output_directory(timestamp)
    logs = []

    # Track statistics
    stats = {"total": len(df), "processed": 0, "errors": 0, "start_time": time.time()}

    # Process each SPARQL question with progress bar
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing SPARQL queries"
    ):
        try:
            sparql = row["SPARQL Query"]
            language_string = row.get("Language", "English")
            answer_type = row.get("Answer Type", "Binary")
            task_type = row.get("Task Type", "")

            # Get appropriate prompt
            prompt_template = get_prompt_template(answer_type, task_type)
            prompt = prompt_template.format(
                language_string=language_string, sparql=sparql
            )

            print(f"\n🔎 Processing question #{idx + 1} in {language_string}")

            start_time = time.time()
            messages = [{"role": "user", "content": prompt}]
            gen_params = {
                k: v
                for k, v in CONFIG.items()
                if k
                in [
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "presence_penalty",
                    "frequency_penalty",
                ]
            }

            # Make API call with retry
            content, model_version = make_api_call_with_retry(
                client, messages, gen_params, CONFIG["max_retries"]
            )

            end_time = time.time()

            # Show result
            if content.startswith("[ERROR]"):
                print(f"❌ Error: {content}")
                stats["errors"] += 1
            else:
                print(f"✅ Generated Question: {content}")
                stats["processed"] += 1

            # Save response to DataFrame
            df.at[idx, "Question"] = content

            # Create comprehensive log entry
            log_entry = {
                "SPARQL_index": idx,
                "SPARQL_query": sparql,
                "generated_question": content,
                "language": language_string,
                "answer_type": answer_type,
                "task_type": task_type,
                "model": "gpt-4o-mini",
                "model_version": model_version,
                "python_version": sys.version.split()[0],
                "timestamp_request": datetime.now().isoformat(),
                "response_time_sec": round(end_time - start_time, 3),
                "response_preview": content[:100],
                "success": not content.startswith("[ERROR]"),
            }

            log_entry.update(gen_params)
            logs.append(log_entry)

            # Rate limiting
            time.sleep(CONFIG["api_delay"])

        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            stats["errors"] += 1

        # Save checkpoint
        if (idx + 1) % CONFIG["checkpoint_frequency"] == 0:
            save_checkpoint(
                df, logs, timestamp, (idx + 1) // CONFIG["checkpoint_frequency"]
            )

    # Save final outputs
    output_filename = output_dir / "generated_questions.csv"
    log_filename = output_dir / "processing_log.csv"
    stats_filename = output_dir / "processing_stats.txt"

    df.to_csv(output_filename, index=False)
    pd.DataFrame(logs).to_csv(log_filename, index=False)

    # Save processing statistics
    total_time = time.time() - stats["start_time"]
    with open(stats_filename, "w") as f:
        f.write(f"Processing Statistics\n")
        f.write(f"====================\n")
        f.write(f"Total questions: {stats['total']}\n")
        f.write(f"Successfully processed: {stats['processed']}\n")
        f.write(f"Errors: {stats['errors']}\n")
        f.write(f"Success rate: {stats['processed'] / stats['total'] * 100:.1f}%\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(
            f"Average time per question: {total_time / stats['total']:.2f} seconds\n"
        )

    print(f"\n✅ Done! Results saved to {output_dir}/")
    print(
        f"📊 Success rate: {stats['processed']}/{stats['total']} ({stats['processed'] / stats['total'] * 100:.1f}%)"
    )

    logger.info(
        f"Processing completed. Success rate: {stats['processed']}/{stats['total']}"
    )


if __name__ == "__main__":
    main()
