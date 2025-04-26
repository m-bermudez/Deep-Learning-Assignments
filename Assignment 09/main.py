import argparse
from utils import logger
from model import model, tokenizer  # ensures model is loaded before training/eval
from data import dataset
from train import run_training
from eval import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model with LoRA")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "both"], default="both",
                        help="Operation mode: train, eval, or both")

    args = parser.parse_args()

    logger.info(f"Running in {args.mode} mode")

    if args.mode in ["train", "both"]:
        run_training(model, tokenizer, dataset)

    if args.mode in ["eval", "both"]:
        run_evaluation()


if __name__ == "__main__":
    main()