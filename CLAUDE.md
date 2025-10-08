# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced LLM (Large Language Model) engineering course repository focused on providing systematic learning paths for LLM training, optimization, deployment, and evaluation. The course is designed for software engineers, AI researchers, and technical professionals seeking to master LLM engineering practices.

## Repository Structure

The repository follows a modular, theory-practice separation design:

```
iSpan_LLM-One-Piece/
├── 00-Course_Setup/          # Environment setup using Poetry
├── 01-Core_Training_Techniques/
│   ├── 01-Theory/           # PEFT, Distributed Training, Alignment theory
│   └── 02-Labs/             # PEFT hands-on labs (LoRA, Adapters, etc.)
├── 02-Efficient_Inference_and_Serving/
├── 03-Model_Compression/
├── 04-Evaluation_and_Data_Engineering/
├── common_utils/            # Shared utility functions
└── datasets/               # Course datasets
```

## Environment Setup and Development Commands

### Initial Setup
The project uses Poetry for dependency management. To set up the environment:

```bash
# Navigate to setup directory
cd 00-Course_Setup/

# Install dependencies (includes PyTorch, transformers, PEFT, DeepSpeed, etc.)
poetry install --no-root --all-extras

# Activate environment
poetry env activate
# Then run the command it outputs, e.g.:
source /path/to/.venv/bin/activate

# Alternative activation
source ./.venv/bin/activate
```

### Common Development Tasks

```bash
# Check GPU availability
python 01-PyTorch_Basics/check_gpu.py

# Run Jupyter Lab for notebooks
jupyter lab

# Example: Start with LoRA training lab
cd 01-Core_Training_Techniques/02-Labs/PEFT_Labs/Lab-01-LoRA/
jupyter lab 01-Setup.ipynb
```

## Core Technologies and Dependencies

The project focuses on modern LLM engineering stack:

- **Training**: PyTorch, Transformers, PEFT, DeepSpeed, Accelerate
- **Inference**: vLLM, TensorRT-LLM, Triton Server
- **Compression**: BitsAndBytes, Auto-GPTQ, AutoAWQ, Optimum
- **Evaluation**: OpenCompass evaluation framework
- **Data**: Datasets library, standard ML data processing tools

## Architecture Patterns

### Lab Structure
Each lab follows a consistent 4-stage pattern:
1. `01-Setup.ipynb` - Environment and model setup
2. `02-Train.ipynb` - Training/fine-tuning implementation
3. `03-Inference.ipynb` - Model inference and testing
4. `04-Merge_and_Deploy.ipynb` - Model merging and deployment (where applicable)

### PEFT Labs Organization
The PEFT labs are organized by technique:
- Lab-01-LoRA: Low-Rank Adaptation
- Lab-02-AdapterLayers: Adapter-based fine-tuning
- Lab-03-Prompt_Tuning: Soft prompt tuning
- Lab-04-Prefix_Tuning: Prefix-based tuning
- Lab-05-IA3: (IA)³ method
- Lab-06-BitFit: Bias-term fine-tuning
- Lab-07-P_Tuning: P-Tuning v1
- Lab-08-P_Tuning_v2: P-Tuning v2

### Shared Utilities
- `common_utils/model_helpers.py` - Model loading and configuration helpers
- `common_utils/data_loaders.py` - Dataset loading utilities

## Key Development Guidelines

### Environment Consistency
- Always use the Poetry-managed virtual environment
- GPU dependencies (TensorRT, vLLM) may require manual installation with specific CUDA versions
- Check `.cursorrules` for detailed development guidelines and coding standards

### Course Philosophy
The course follows a "fundamentals-first" approach based on:
1. **First Principles**: Understanding core LLM engineering concepts
2. **Modular Learning**: Each module can be studied independently
3. **Theory-Practice Integration**: Every theory section has corresponding hands-on labs
4. **Progressive Difficulty**: From basic PEFT to advanced distributed training

### Working with Labs
- Start with setup notebooks to understand dependencies
- Follow the numbered sequence within each lab
- Each lab includes comprehensive README with theory background
- Labs are designed to build upon each other within modules

## Important Notes

- The course content is primarily in Traditional Chinese
- Environment setup instructions support Windows, macOS, Linux, and WSL
- GPU acceleration is recommended for training labs
- Some advanced features may require specific hardware configurations

## Testing and Validation

No automated test framework is currently configured. Validation is done through:
- Individual notebook execution
- Manual verification of training results
- Performance benchmarking in evaluation labs