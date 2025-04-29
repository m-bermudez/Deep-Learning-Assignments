import os
import random
import logging
import torch
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq  
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Seed setting for reproducibility ---
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# --- GPU availability check ---
def check_gpu_availability():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            logger.info(f"\u2705 GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        logger.warning("\u274C No NVIDIA GPU detected! Training will be extremely slow on CPU.")
        return False

def require_gpu():
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for training. Please use a GPU-enabled environment.")

# --- Package version checking ---
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # For Python < 3.8

def get_package_version(package_name):
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "Not installed"

def log_package_versions(packages):
    for package in packages:
        version = get_package_version(package)
        logger.info(f"{package} version: {version}")

# Log important package versions
log_package_versions(["torch", "transformers", "peft", "datasets", "bitsandbytes", "accelerate"])

# --- Configurations ---
@dataclass
class Config:
    model_id: str = "t5-large"
    #model_id: str = "facebook/bart-large"
    dataset_name: str = "billsum"
    dataset_text_field: str = "text"
    dataset_summary_field: str = "summary"
    output_dir: str = "./fine_tuned_model"
    peft_output_dir: str = "./peft_adapter"

    # Training hyperparameters
    num_epochs: int = 2
    batch_size: int = 4 if torch.cuda.is_available() else 2
    learning_rate: float = 4e-5
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 200
    logging_steps: int = 50

    # Input/output max lengths
    max_input_length: int = 512
    max_target_length: int = 256

    # Generation parameters
    max_new_tokens: int = 256
    num_beams: int = 6
    repetition_penalty: float = 1.8
    length_penalty: float = 1.1
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3

    # Other options
    skip_quantization: bool = False
    label_smoothing_factor: float = 0.1

config = Config()