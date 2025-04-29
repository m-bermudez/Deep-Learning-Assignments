# train.py

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from utils import logger, config

def run_training(model, tokenizer, dataset):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

        logger.info(f"Training for {config.num_epochs} epochs with {len(train_loader)} batches per epoch.")

        global_step = 0
        running_loss = 0.0

        for epoch in range(config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
            model.train()

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            optimizer.zero_grad()

            for step, batch in enumerate(progress_bar):
                bills = batch["Bill"]
                summaries = batch["Summary"]

                model_inputs = tokenizer(
                    bills,
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_input_length,
                    return_tensors="pt"
                )

                labels = tokenizer(
                    summaries,
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_target_length,
                    return_tensors="pt"
                ).input_ids

                
                labels[labels == tokenizer.pad_token_id] = -100

                input_ids = model_inputs.input_ids.to(device)
                attention_mask = model_inputs.attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

                running_loss += loss.item()

                if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % config.logging_steps == 0:
                        avg_loss = running_loss / config.logging_steps
                        logger.info(f"Step {global_step}: Avg Loss = {avg_loss:.4f}")
                        running_loss = 0.0

            save_path = os.path.join(config.peft_output_dir, f"epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Saved model checkpoint to {save_path}")

        final_path = config.peft_output_dir
        os.makedirs(final_path, exist_ok=True)
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Training complete! Final model saved to {final_path}")

        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise RuntimeError("Training failed")