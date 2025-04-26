import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import model_id, skip_quantization, logger

model_loaded = False

try:
    logger.info(f"Loading model: {model_id}")

    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        skip_quantization = True
        logger.warning("Running on CPU - forcing skip_quantization=True")

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
                model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=False  # trust_remote_code is not needed for normal T5
            )
            logger.info("Successfully loaded model with 4-bit quantization")
        except Exception as bnb_error:
            logger.warning(f"BitsAndBytes quantization failed: {bnb_error}")
            logger.warning("Falling back to standard precision loading")
            skip_quantization = True

    if skip_quantization:
        logger.info("Loading model without quantization")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=False
        )
        logger.info("Successfully loaded model without quantization")

    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Successfully loaded model and tokenizer")
    model_loaded = True

except Exception as e:
    logger.error(f"Error loading model {model_id}: {e}")
    logger.error("Try a different model or check your internet connection.")
    model_loaded = False

if not model_loaded or 'model' not in locals():
    logger.error("Model loading failed. Cannot continue with fine-tuning.")
    raise RuntimeError("Model loading failed")

# --- Configure LoRA for T5 (vanilla) ---
target_modules = ["q", "v"]  # Only q, v are best for normal t5 (not q, k, v, o)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=target_modules,
    lora_dropout=0.03,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

try:
    if not skip_quantization:
        model = prepare_model_for_kbit_training(model)
        logger.info("Prepared model for k-bit training")

    model = get_peft_model(model, lora_config)
    logger.info("Applied LoRA configuration to model")

    logger.info("Trainable parameters:")
    model.print_trainable_parameters()

except Exception as e:
    logger.error(f"Error configuring PEFT: {e}")
    raise RuntimeError(f"Failed to configure PEFT: {e}")