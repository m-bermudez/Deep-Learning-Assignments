import re
from utils import config, logger
from transformers import AutoTokenizer
from datasets import load_dataset

# --- Load Dataset ---
try:
    dataset = load_dataset(config.dataset_name, split="train[:90%]")
    logger.info(f"Successfully loaded dataset: {config.dataset_name}")
    logger.info(f"Dataset size: {len(dataset)} examples")
    # logger.info(f"Example data point: {dataset[0]}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    def clean_text(text):
        """Clean text by removing \n and extra spaces."""
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def truncate_middle(text, max_tokens):
        """Take the beginning and end parts of the text if it exceeds max_tokens."""
        tokens = tokenizer.tokenize(text)
        if len(tokens) <= max_tokens:
            return text  

        half = max_tokens // 2
        first_part = tokens[:half]
        last_part = tokens[-half:]
        truncated_tokens = first_part + last_part
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        return truncated_text

    def preprocess_function(sample):
        """Prepare the input (Bill) and target (Summary)."""
        bill_text = clean_text(sample[config.dataset_text_field])
        summary_text = clean_text(sample[config.dataset_summary_field])

        bill_text = truncate_middle(bill_text, config.max_input_length)
        summary_text = truncate_middle(summary_text, config.max_target_length)

        return {
            "Bill": bill_text,
            "Summary": summary_text,
        }

    dataset = dataset.map(preprocess_function)

    logger.info(f"Formatted dataset. Example:\nBill: {dataset[0]['Bill'][:200]}...\nSummary: {dataset[0]['Summary'][:200]}...")

except Exception as e:
    logger.error(f"Error loading dataset {config.dataset_name}: {e}")
    logger.error("Ensure the dataset exists and is accessible, or prepare your data manually.")
    raise RuntimeError(f"Failed to load dataset: {e}")