# Assignment 09 - Instructions

## Environment Setup

Please use the provided YAML file to create the environment required for testing the code.

## Running the Code

1. **Activate** the environment you created from the YAML file.
2. **Navigate** to the `Assignment09` directory.
3. **Execute** the following command in the terminal:

   ```bash
   python main.py --mode both

To run the BART model instead of the T5 model:

Open utils.py:

Comment out the model_id line for T5.

Uncomment the model_id line for BART.

Open model.py:

Update the target_modules as follows:

For T5 model:
target_modules = ["q", "v"]

For BART model:
target_modules = ["q_proj", "v_proj"]

the code structure has been developed such that:
utils.py provide global configuration variables used on the other python files as well importing essential libraries
data.py loads and preprocess the data into tokens and ensures format fits the task to do.
model.py makes sure the model is load correctly and performs the configuration for the PEFT (applying Lora)
train.py includes the training loop as well tokenization verification and saving modeling steps.
eval.py does the inference step using the test set from the dataset chosen. It provides evaluation metrics such as ROUGUE and provides an estimate across 400 samples.