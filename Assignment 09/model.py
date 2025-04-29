# model.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import config, logger 

model_loaded = False
skip_quantization = getattr(config, "skip_quantization", False)

try:
    logger.info(f"Loading model: {config.model_id}")

    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        skip_quantization = True
        logger.warning("Running on CPU, setting skip_quantization=True")

    if not skip_quantization:
        try:
            import bitsandbytes as bnb
            logger.info(f"Using BitsAndBytes for 4-bit quantization, version: {bnb.__version__}")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.model_id,
                device_map=device_map,
                quantization_config=bnb_config,
                trust_remote_code=False
            )
            logger.info("Successfully loaded model with 4-bit quantization")

        except Exception as bnb_error:
            logger.warning(f"BitsAndBytes quantization failed: {bnb_error}")
            logger.warning("Falling back to standard loading")
            skip_quantization = True

    if skip_quantization:
        logger.info("Loading model without quantization")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=False
        )
        logger.info("Successfully loaded model without quantization")

    model.config.use_cache = False
    model.config.max_length = config.max_target_length  # <- Important for generation (Step 3)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # <- Important (Step 1)

    logger.info("Successfully loaded model and tokenizer")
    model_loaded = True

except Exception as e:
    logger.error(f"Error loading model {config.model_id}: {e}")
    raise RuntimeError("Model loading failed")

if not model_loaded:
    raise RuntimeError("Model was not loaded successfully, cannot continue")

# Apply LoRA
target_modules = ["q", "v"]
#target_modules = ["q_proj", "v_proj"]
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

try:
    if not skip_quantization:
        model = prepare_model_for_kbit_training(model)
        logger.info("Prepared model for k-bit training")

    model = get_peft_model(model, lora_config)
    logger.info("Applied LoRA configuration")

    model.print_trainable_parameters()

except Exception as e:
    logger.error(f"Error configuring PEFT: {e}")
    raise RuntimeError("Failed to configure LoRA PEFT")