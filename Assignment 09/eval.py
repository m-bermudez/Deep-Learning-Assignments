# eval.py

import torch
import re
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from utils import config, logger
from rouge_score import rouge_scorer

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def run_evaluation():
    logger.info("Starting evaluation...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_id,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=False
        )

        model = PeftModel.from_pretrained(model, config.peft_output_dir)
        model = model.merge_and_unload()
        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        test_dataset = load_dataset(config.dataset_name, split="test")

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        num_samples = min(400, len(test_dataset))  

        for i, sample in enumerate(test_dataset):
            if i >= num_samples:
                break

            bill_text = clean_text(sample["text"])
            ground_truth = sample.get("summary", "")

            encoding = tokenizer(
                bill_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=config.max_input_length
            )

            inputs = {
                "input_ids": encoding.input_ids.to(device),
                "attention_mask": encoding.attention_mask.to(device)
            }

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=config.max_new_tokens,
                    num_beams=config.num_beams,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    repetition_penalty=config.repetition_penalty,
                    length_penalty=config.length_penalty,
                    early_stopping=config.early_stopping
                )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            scores = scorer.score(generated_text, ground_truth)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

            if i < 3:
                logger.info(f"\nExample {i+1}:")
                logger.info(f"Bill (truncated): {bill_text[:300]}...")
                logger.info(f"Ground truth summary: {ground_truth}")
                logger.info(f"Generated summary: {generated_text}")
                logger.info(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
                logger.info(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
                logger.info(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

        logger.info(f"\nAverage ROUGE on {num_samples} samples:")
        logger.info(f"ROUGE-1: {avg_rouge1:.4f}")
        logger.info(f"ROUGE-2: {avg_rouge2:.4f}")
        logger.info(f"ROUGE-L: {avg_rougeL:.4f}")

        logger.info("Evaluation finished successfully!")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise RuntimeError("Evaluation failed")