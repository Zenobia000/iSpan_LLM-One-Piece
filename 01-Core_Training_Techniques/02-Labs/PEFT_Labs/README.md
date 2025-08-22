# PEFT (Parameter-Efficient Fine-Tuning) Labs

This directory contains a series of hands-on labs designed to provide a deep, practical understanding of various Parameter-Efficient Fine-Tuning (PEFT) techniques using the Hugging Face ecosystem.

Each lab focuses on a specific PEFT method, guiding you from environment setup to training and inference, with a strong emphasis on explaining the role and usage of key components from the `transformers` and `peft` libraries.

## Lab Structure

- **Lab-01-LoRA**: Covers the most popular PEFT method, Low-Rank Adaptation (LoRA), and its quantized variant, QLoRA.
- **Lab-02-AdapterLayers**: Explores the classic "Adapter" method, which involves inserting small, trainable modules into the model architecture.
- **Lab-03-Prompt_Tuning**: Focuses on Prompt Tuning, a method that learns "soft prompts" to guide the model's behavior without altering its weights.
- **Lab-04-Prefix_Tuning**: Implements Prefix-Tuning, which is similar to Prompt Tuning but adds trainable prefixes to the model's internal states.
- **Lab-05-IA3**: Demonstrates (IA)³, an extremely parameter-efficient method that learns to scale internal activations.
- **Lab-06-BitFit**: Implements BitFit, an extremely parameter-efficient method that only fine-tunes bias parameters.
- **Lab-07-P_Tuning**: Explores P-Tuning, which uses trainable virtual tokens with a prompt encoder for natural language understanding tasks.
- **Lab-08-P_Tuning_v2**: Covers P-Tuning v2, an advanced version that applies deep prompts to every transformer layer for universal task effectiveness.

## Learning Approach

In each lab, you will find a consistent set of Jupyter Notebooks:

- `01-Setup.ipynb`: Sets up the environment and installs all necessary dependencies.
- `02-Train.ipynb`: The core of the lab. It covers loading the dataset and base model, configuring the specific PEFT method, and running the training process.
- `03-Inference.ipynb`: Shows how to load the trained PEFT adapter and use the fine-tuned model for inference on new data.
- `04-Merge_and_Deploy.ipynb` (where applicable): Demonstrates how to merge the PEFT adapter weights back into the base model for standalone deployment.

Start with `Lab-01-LoRA` to build a strong foundation, then proceed to the other labs to explore the diverse landscape of parameter-efficient fine-tuning.
