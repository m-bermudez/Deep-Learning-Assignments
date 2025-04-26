from utils import dataset_name, dataset_text_field, dataset_summary_field, logger
from transformers import AutoTokenizer
from datasets import load_dataset
import re

# --- Load Dataset ---
try:
    dataset = load_dataset(dataset_name, split="train[:60%]")
    logger.info(f"Successfully loaded dataset: {dataset_name}")
    logger.info(f"Dataset size: {len(dataset)} examples")
    logger.info(f"Example data point: {dataset[0]}")

    def clean_text(text):
        """Clean text by removing \n and extra spaces."""
        text = text.replace('\n', ' ')  # Remove newlines
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()

    def format_instruction(sample):
        """Format the dataset into instruction-response pairs for summarization."""
        cleaned_text = clean_text(sample[dataset_text_field])
        cleaned_summary = clean_text(sample[dataset_summary_field])
        return {
            "text": (
                f"### Instruction:\nSummarize the following legal bill.\n\n"
                f"### Input:\n{cleaned_text}\n\n"
                f"### Response:\n{cleaned_summary}"
            )
        }

    dataset = dataset.map(format_instruction)
    logger.info(f"Formatted dataset. Example: {dataset[0]['text'][:200]}...")

except Exception as e:
    logger.error(f"Error loading dataset {dataset_name}: {e}")
    logger.error("Ensure the dataset exists and is accessible, or prepare your data manually.")
    raise RuntimeError(f"Failed to load dataset: {e}")