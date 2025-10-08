# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced LLM (Large Language Model) engineering course repository focused on providing systematic learning paths for LLM training, optimization, deployment, and evaluation. The course is designed for software engineers, AI researchers, and technical professionals seeking to master LLM engineering practices.

## Repository Structure

The repository follows a modular, theory-practice separation design:

```
iSpan_LLM-One-Piece/
├── 00-Course_Setup/          # Poetry environment (pyproject.toml, .venv)
├── 01-Core_Training_Techniques/
│   ├── 01-Theory/           # PEFT, Distributed Training, Alignment theory
│   └── 02-Labs/
│       ├── PEFT_Labs/       # 8 technique-specific labs (Lab-01 to Lab-08)
│       ├── Lab-1.1-PEFT_with_HuggingFace/
│       ├── Lab-1.2-PyTorch_DDP_Basics/
│       └── Lab-1.3-Finetune_Alpaca_with_DeepSpeed/
├── 02-Efficient_Inference_and_Serving/
│   ├── 01-Theory/
│   └── 02-Labs/
├── 03-Model_Compression/
├── 04-Evaluation_and_Data_Engineering/
├── common_utils/            # model_helpers.py, data_loaders.py
├── datasets/                # alpaca_data/, etc.
├── 01-PyTorch_Basics/       # check_gpu.py, fundamentals
└── docs/                    # Course documentation
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
# Verify GPU and CUDA setup
python 01-PyTorch_Basics/check_gpu.py

# Launch Jupyter Lab (from any directory after activating venv)
jupyter lab

# Navigate to specific lab (example: LoRA training)
cd 01-Core_Training_Techniques/02-Labs/PEFT_Labs/Lab-01-LoRA/
jupyter lab

# Or run distributed training examples
cd 01-Core_Training_Techniques/02-Labs/Lab-1.2-PyTorch_DDP_Basics/
# (follow lab-specific instructions)
```

## Core Technologies and Dependencies

The project uses Poetry for dependency management with CUDA 12.1 optimized builds. Key dependencies in `00-Course_Setup/pyproject.toml`:

- **Training**: PyTorch 2.5.1+cu121, Transformers 4.57+, PEFT, DeepSpeed, Accelerate
- **Inference**: vLLM (optional via `--all-extras`), FastAPI, Uvicorn
- **Development**: JupyterLab 4.4+, ipywidgets, matplotlib, seaborn
- **Data**: Datasets 2.14+, pandas 2.0+, scikit-learn 1.3+
- **Compression**: BitsAndBytes, Auto-GPTQ, AutoAWQ, Optimum
- **Evaluation**: OpenCompass evaluation framework

**Important**: PyTorch is installed from explicit CUDA 12.1 source. GPU libraries like vLLM and flash-attention may require manual installation matching your CUDA version.

## Architecture Patterns

### Lab Structure
Each PEFT lab follows a consistent 4-stage pattern designed for progressive learning:
1. `01-Setup.ipynb` - Environment verification, model/tokenizer loading, dataset preparation
2. `02-Train.ipynb` - PEFT configuration, training loop implementation, checkpoint saving
3. `03-Inference.ipynb` - Adapter loading, inference testing, output comparison
4. `04-Merge_and_Deploy.ipynb` - Adapter merging (LoRA/QLoRA), model export, deployment preparation

This structure allows students to understand each phase independently while seeing the complete workflow.

### PEFT Labs Organization
The PEFT labs in `01-Core_Training_Techniques/02-Labs/PEFT_Labs/` are organized by technique category:

**Reparameterization Methods** (industry standard, most versatile):
- `Lab-01-LoRA`: Low-Rank Adaptation with QLoRA (0.1-1% params, Llama-2-7B + guanaco)

**Additive Methods** (modular, multi-task friendly):
- `Lab-02-AdapterLayers`: Bottleneck adapters (0.5-5% params, BERT + MRPC)
- `Lab-03-Prompt_Tuning`: Soft prompts (0.01-0.1% params, best for large models)
- `Lab-04-Prefix_Tuning`: Multi-layer prefixes (0.1-1% params, excels at generation)
- `Lab-07-P_Tuning`: MLP prompt encoder (0.1% params, NLU specialized)
- `Lab-08-P_Tuning_v2`: Deep prompt mechanism (0.1% params, most general)

**Selective Methods** (ultra-efficient):
- `Lab-05-IA3`: Activation scaling (~0.01% params, extreme efficiency)
- `Lab-06-BitFit`: Bias-only tuning (0.08% params, resource-constrained)

### Shared Utilities
The `common_utils/` directory contains reusable utilities designed for all PEFT labs:

- **`model_helpers.py`**: Comprehensive model management utilities including:
  - Model loading with quantization support (INT8, FP4, NF4)
  - PEFT adapter initialization and configuration
  - Model merging for LoRA/QLoRA adapters
  - Memory monitoring and GPU resource management
  - Support for ModelType enum: LLAMA, MISTRAL, QWEN, GEMMA

- **`data_loaders.py`**: Dataset utilities with multi-format support:
  - `InstructionDataset` class for instruction fine-tuning
  - Multiple prompt templates (Alpaca, Dolly, ChatML)
  - `load_alpaca_dataset()` for common dataset loading
  - `InstructionDataCollator` for PEFT-optimized batching
  - Automatic tokenization and padding handling

These utilities are imported across all PEFT labs to maintain consistency.

## Key Development Guidelines

### Environment Consistency
- Always use the Poetry-managed virtual environment located in `00-Course_Setup/.venv`
- Activate before any development work: `source 00-Course_Setup/.venv/bin/activate`
- GPU dependencies (TensorRT, vLLM) may require manual installation with specific CUDA versions
- The project is optimized for CUDA 12.1; other CUDA versions may require dependency adjustments

### Course Philosophy
The course follows a "fundamentals-first" approach based on:
1. **First Principles**: Understanding core LLM engineering concepts
2. **Modular Learning**: Each module can be studied independently
3. **Theory-Practice Integration**: Every theory section has corresponding hands-on labs
4. **Progressive Difficulty**: From basic PEFT to advanced distributed training

### Working with Labs
- **Always activate the virtual environment first**: `source 00-Course_Setup/.venv/bin/activate`
- Start with `01-Setup.ipynb` to verify dependencies and understand the data pipeline
- Follow the numbered sequence (01→02→03→04) within each lab
- Each lab directory contains a `README.md` with theoretical background, research papers, and methodology
- Import shared utilities from `common_utils/` rather than duplicating code
- Labs within PEFT_Labs are independent but share common patterns for easier learning transfer

## Important Notes

- **Language**: Course content (notebooks, READMEs, comments) is primarily in Traditional Chinese (繁體中文)
- **Multi-platform support**: Environment setup works on Windows, macOS, Linux, and WSL (WSL is the development environment)
- **GPU requirements**:
  - GPU with CUDA 12.1+ is strongly recommended for training labs
  - Quantization labs (QLoRA, INT8) require GPU with sufficient memory
  - CPU-only mode possible but extremely slow for model training
- **Hardware considerations**:
  - LoRA labs with 7B models require ~16GB GPU memory (with quantization)
  - Full precision fine-tuning may require 24GB+ VRAM
  - Distributed training labs require multi-GPU setup

## Testing and Validation

No automated test framework is currently configured. Validation is done through:
- Individual notebook execution
- Manual verification of training results
- Performance benchmarking in evaluation labs