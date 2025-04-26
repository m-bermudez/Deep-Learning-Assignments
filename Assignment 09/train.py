import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from utils import (
    logger, output_dir, peft_output_dir,
    gradient_accumulation_steps, learning_rate,
    num_epochs, warmup_steps, logging_steps, batch_size
)

def run_training(model, tokenizer, dataset):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        logger.info(f"Training for {num_epochs} epochs, {len(train_loader)} batches per epoch.")

        global_step = 0
        running_loss = 0.0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            model.train()

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            optimizer.zero_grad()

            for step, batch in enumerate(progress_bar):
                texts = batch["text"]

                # Tokenize inside the loop
                tokenized = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)

                # The label is the expected summary (in instruction format, it's in "text" too)
                labels = input_ids.clone()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                loss.backward()

                running_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % logging_steps == 0:
                        avg_loss = running_loss / logging_steps
                        logger.info(f"Step {global_step}: Avg Loss = {avg_loss:.4f}")
                        running_loss = 0.0

            # Save LoRA adapter checkpoint after every epoch
            save_path = os.path.join(peft_output_dir, f"epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Saved model checkpoint to {save_path}")

        # Final save
        final_path = peft_output_dir
        os.makedirs(final_path, exist_ok=True)
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Training complete! Final model saved to {final_path}")

        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in manual training loop: {e}")
        raise RuntimeError(f"Training failed: {e}")