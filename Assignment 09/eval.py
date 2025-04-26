import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from utils import logger, dataset_name, model_id, peft_output_dir
from rouge_score import rouge_scorer
import re

def clean_text(text):
    """Clean text by removing \n and extra spaces."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def run_evaluation():
    logger.info("Running evaluation on test set...")

    try:
        # Load base model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )

        model = PeftModel.from_pretrained(model, peft_output_dir)
        model = model.merge_and_unload()  # Merge LoRA weights
        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load and format test set
        test_dataset = load_dataset(dataset_name, split="test")

        def format_instruction(sample):
            """Format and clean text for summarization."""
            cleaned_text = clean_text(sample["text"])
            return (
                f"### Instruction:\nSummarize the following legal bill.\n\n"
                f"### Input:\n{cleaned_text}\n\n"
                f"### Response:\n"
            )

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        num_samples = min(500, len(test_dataset))

        for i, sample in enumerate(test_dataset):
            if i >= num_samples:
                break

            # --- Prepare input ---
            formatted_input = format_instruction(sample)
            ground_truth = sample.get("summary", "")

            inputs = tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=128,
                    num_beams=4,
                )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # --- Score ---
            scores = scorer.score(generated_text, ground_truth)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

            if i < 3:
                logger.info(f"\nExample {i+1}:")
                logger.info(f"Input: {formatted_input[:300]}...")
                logger.info(f"Ground truth: {ground_truth}")
                logger.info(f"Generated: {generated_text}")
                logger.info(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
                logger.info(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
                logger.info(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

        # --- Average scores ---
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

        logger.info(f"\nAverage ROUGE scores on {num_samples} samples:")
        logger.info(f"ROUGE-1: {avg_rouge1:.4f}")
        logger.info(f"ROUGE-2: {avg_rouge2:.4f}")
        logger.info(f"ROUGE-L: {avg_rougeL:.4f}")

        logger.info("Evaluation complete!")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise RuntimeError("Evaluation failed")