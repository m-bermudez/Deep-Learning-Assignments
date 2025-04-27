import os
import random
import logging
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# --- Check for GPU availability ---
def check_gpu_availability():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"\u2705 {gpu_count} NVIDIA GPU(s) detected: {gpu_name}")
        for i in range(gpu_count):
            if i > 0:
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        logger.warning("\u274C No NVIDIA GPU detected! Training will be extremely slow on CPU.")
        logger.warning("Consider using a GPU-enabled environment (Google Colab, Kaggle, etc.)")
        return False

# Check GPU
has_gpu = check_gpu_availability()
if not has_gpu:
    user_response = input("Continue without GPU? (y/n): ")
    if user_response.lower() != 'y':
        logger.info("Exiting as requested.")
        raise SystemExit("Exiting due to no GPU available")
    logger.warning("Continuing without GPU, but training will be extremely slow.")

# --- Check package versions ---
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

def get_package_version(package_name):
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "Not installed"

# Log versions
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Transformers version: {get_package_version('transformers')}")
logger.info(f"PEFT version: {get_package_version('peft')}")
logger.info(f"Datasets version: {get_package_version('datasets')}")
logger.info(f"BitsAndBytes version: {get_package_version('bitsandbytes')}")
logger.info(f"Accelerate version: {get_package_version('accelerate')}")

# --- Config ---
model_id = "t5-large" 
dataset_name = "billsum"         
dataset_text_field = "text"        
dataset_summary_field = "summary"  
output_dir = "./fine_tuned_model"
peft_output_dir = "./peft_adapter"

# Training params
num_epochs = 2
batch_size = 4 if torch.cuda.is_available() else 2
learning_rate = 1e-5
gradient_accumulation_steps = 4
warmup_steps = 200
logging_steps = 50

# Max lengths
max_input_length = 1024
max_target_length = 512

# Evaluation generation params
max_new_tokens = 256
num_beams = 4
repetition_penalty = 1.4
length_penalty = 1.2
early_stopping = True

# Flag to control quantization
skip_quantization = False