"""
Model Helper Functions for PEFT Training and Inference

This module provides comprehensive utilities for model loading, PEFT adapter management,
model merging, and inference optimization tailored for educational PEFT experiments.

Author: Claude Code Assistant
Project: iSpan LLM One-Piece Educational Course
Module: Core Training Techniques - Model Management Utilities
"""

import os
import json
import torch
import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig, TrainingArguments,
    PreTrainedModel, PreTrainedTokenizer
)
from peft import (
    LoraConfig, AdapterConfig, PrefixTuningConfig, PromptTuningConfig,
    TaskType, PeftModel, PeftConfig, get_peft_model, prepare_model_for_kbit_training
)
import accelerate
from accelerate import Accelerator
import psutil
import GPUtil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class ModelType(Enum):
    """Supported model types for PEFT experiments"""
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    GEMMA = "gemma"
    PHI = "phi"
    BAICHUAN = "baichuan"


class PEFTMethod(Enum):
    """Supported PEFT methods"""
    LORA = "lora"
    ADAPTER = "adapter"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"
    IA3 = "ia3"
    BITFIT = "bitfit"
    P_TUNING = "p_tuning"
    P_TUNING_V2 = "p_tuning_v2"


@dataclass
class ModelConfig:
    """Configuration for model loading and PEFT setup"""
    model_name_or_path: str
    model_type: ModelType = ModelType.LLAMA
    peft_method: Optional[PEFTMethod] = None

    # Quantization settings
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16

    # Model loading settings
    torch_dtype: torch.dtype = torch.float16
    device_map: Union[str, Dict] = "auto"
    trust_remote_code: bool = False
    use_flash_attention: bool = True

    # PEFT specific settings
    target_modules: Optional[List[str]] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM

    # Training settings
    gradient_checkpointing: bool = True
    use_cache: bool = False

    def __post_init__(self):
        """Post-initialization validation and default settings"""
        if self.target_modules is None:
            self.target_modules = self._get_default_target_modules()

    def _get_default_target_modules(self) -> List[str]:
        """Get default target modules based on model type"""
        target_module_map = {
            ModelType.LLAMA: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ModelType.MISTRAL: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ModelType.QWEN: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ModelType.GEMMA: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ModelType.PHI: ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            ModelType.BAICHUAN: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
        return target_module_map.get(self.model_type, ["q_proj", "v_proj"])


class SystemResourceManager:
    """Manage system resources and provide recommendations"""

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {"available": False, "count": 0}

            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "memory_util": gpu.memoryUtil,
                    "load": gpu.load
                })

            return {
                "available": True,
                "count": len(gpus),
                "gpus": gpu_info,
                "total_memory": sum(gpu["memory_total"] for gpu in gpu_info)
            }
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            return {"available": torch.cuda.is_available(), "count": torch.cuda.device_count()}

    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get system memory information"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percentage": memory.percent
        }

    @staticmethod
    def recommend_model_config(model_size_gb: float, available_gpu_memory_gb: float) -> Dict[str, Any]:
        """Recommend model configuration based on available resources"""
        recommendations = {
            "quantization": None,
            "gradient_checkpointing": True,
            "batch_size": 1,
            "micro_batch_size": 1,
            "warnings": []
        }

        # Estimate memory requirements (rough estimates)
        base_memory = model_size_gb
        training_overhead = model_size_gb * 2  # Gradients + optimizer states
        activation_memory = 1.0  # Estimate for activations

        total_required = base_memory + training_overhead + activation_memory

        if total_required > available_gpu_memory_gb:
            if total_required * 0.5 <= available_gpu_memory_gb:  # 4-bit can reduce by ~50%
                recommendations["quantization"] = "4bit"
                recommendations["warnings"].append("Using 4-bit quantization due to memory constraints")
            elif total_required * 0.75 <= available_gpu_memory_gb:  # 8-bit can reduce by ~25%
                recommendations["quantization"] = "8bit"
                recommendations["warnings"].append("Using 8-bit quantization due to memory constraints")
            else:
                recommendations["warnings"].append("Model may not fit in available GPU memory even with quantization")

        # Adjust batch size recommendations
        if available_gpu_memory_gb < 8:
            recommendations["batch_size"] = 1
            recommendations["micro_batch_size"] = 1
        elif available_gpu_memory_gb < 16:
            recommendations["batch_size"] = 2
            recommendations["micro_batch_size"] = 1
        else:
            recommendations["batch_size"] = 4
            recommendations["micro_batch_size"] = 2

        return recommendations


class ModelLoader:
    """Advanced model loading with automatic configuration"""

    @staticmethod
    def load_model_and_tokenizer(
        config: ModelConfig,
        cache_dir: Optional[str] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer with automatic configuration

        Args:
            config: ModelConfig instance with loading parameters
            cache_dir: Optional cache directory for model files

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {config.model_name_or_path}")

        # Setup quantization config if needed
        quantization_config = None
        if config.load_in_4bit or config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            cache_dir=cache_dir,
            padding_side="left"  # Important for generation
        )

        # Add pad token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.unk_token

        # Setup model loading arguments
        model_kwargs = {
            "torch_dtype": config.torch_dtype,
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": cache_dir,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        # Add attention implementation if supported
        if config.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                logger.warning("Flash Attention 2 not available, using default attention")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs
        )

        # Configure model for training
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if not config.use_cache:
            model.config.use_cache = False

        # Prepare for k-bit training if quantized
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)

        logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")

        return model, tokenizer

    @staticmethod
    def get_model_info(model_name_or_path: str) -> Dict[str, Any]:
        """Get model information without loading the full model"""
        try:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

            # Estimate model size
            num_params = getattr(config, 'num_parameters', None)
            if num_params is None:
                # Estimate based on hidden size and layers
                hidden_size = getattr(config, 'hidden_size', 4096)
                num_layers = getattr(config, 'num_hidden_layers', 32)
                vocab_size = getattr(config, 'vocab_size', 32000)

                # Rough estimation: embedding + transformer layers + output
                embedding_params = vocab_size * hidden_size
                layer_params = num_layers * (hidden_size * hidden_size * 12)  # Rough estimate
                num_params = embedding_params + layer_params

            model_size_gb = (num_params * 2) / (1024**3)  # FP16 = 2 bytes per parameter

            return {
                "model_type": config.model_type,
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_layers": getattr(config, 'num_hidden_layers', None),
                "num_attention_heads": getattr(config, 'num_attention_heads', None),
                "vocab_size": getattr(config, 'vocab_size', None),
                "estimated_params": num_params,
                "estimated_size_gb": model_size_gb,
                "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
            }
        except Exception as e:
            logger.error(f"Could not get model info: {e}")
            return {"error": str(e)}


class PEFTManager:
    """Manage PEFT configurations and adapter operations"""

    @staticmethod
    def create_peft_config(config: ModelConfig) -> PeftConfig:
        """Create PEFT configuration based on method"""
        if config.peft_method == PEFTMethod.LORA:
            return LoraConfig(
                task_type=config.task_type,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias=config.bias,
                target_modules=config.target_modules,
            )
        elif config.peft_method == PEFTMethod.ADAPTER:
            return AdapterConfig(
                task_type=config.task_type,
                adapter_hidden_size=config.lora_r * 4,  # Reasonable default
                target_modules=config.target_modules,
            )
        elif config.peft_method == PEFTMethod.PREFIX_TUNING:
            return PrefixTuningConfig(
                task_type=config.task_type,
                num_virtual_tokens=config.lora_r,  # Use lora_r as virtual token count
                prefix_projection=True,
            )
        elif config.peft_method == PEFTMethod.PROMPT_TUNING:
            return PromptTuningConfig(
                task_type=config.task_type,
                num_virtual_tokens=config.lora_r,
                prompt_tuning_init="RANDOM",
            )
        else:
            raise ValueError(f"Unsupported PEFT method: {config.peft_method}")

    @staticmethod
    def apply_peft(model: PreTrainedModel, config: ModelConfig) -> PeftModel:
        """Apply PEFT to a base model"""
        if config.peft_method is None:
            raise ValueError("PEFT method must be specified")

        peft_config = PEFTManager.create_peft_config(config)
        model = get_peft_model(model, peft_config)

        # Print trainable parameters
        model.print_trainable_parameters()

        return model

    @staticmethod
    def load_peft_model(
        base_model_path: str,
        adapter_path: str,
        config: ModelConfig
    ) -> Tuple[PeftModel, PreTrainedTokenizer]:
        """Load a PEFT model with adapters"""
        logger.info(f"Loading PEFT model from {adapter_path}")

        # Load base model and tokenizer
        model, tokenizer = ModelLoader.load_model_and_tokenizer(config)

        # Load PEFT model
        model = PeftModel.from_pretrained(model, adapter_path)

        return model, tokenizer

    @staticmethod
    def merge_and_save(
        peft_model: PeftModel,
        output_path: str,
        tokenizer: PreTrainedTokenizer,
        safe_serialization: bool = True
    ) -> None:
        """Merge PEFT adapters and save the merged model"""
        logger.info(f"Merging adapters and saving to {output_path}")

        # Merge adapters
        merged_model = peft_model.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(
            output_path,
            safe_serialization=safe_serialization
        )

        # Save tokenizer
        tokenizer.save_pretrained(output_path)

        logger.info("Model and tokenizer saved successfully")

    @staticmethod
    def save_adapter_only(
        peft_model: PeftModel,
        output_path: str,
        safe_serialization: bool = True
    ) -> None:
        """Save only the PEFT adapters"""
        logger.info(f"Saving adapter to {output_path}")

        peft_model.save_pretrained(
            output_path,
            safe_serialization=safe_serialization
        )

        logger.info("Adapter saved successfully")


class InferenceOptimizer:
    """Optimize models for inference"""

    @staticmethod
    def optimize_for_inference(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimization_level: str = "medium"
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Optimize model for inference

        Args:
            model: The model to optimize
            tokenizer: The tokenizer
            optimization_level: "light", "medium", "aggressive"

        Returns:
            Optimized model and tokenizer
        """
        logger.info(f"Optimizing model for inference (level: {optimization_level})")

        # Enable cache for generation
        model.config.use_cache = True

        # Disable gradient computation
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if optimization_level in ["medium", "aggressive"]:
            # Compile model if PyTorch 2.0+
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")

        if optimization_level == "aggressive":
            # Additional optimizations
            try:
                # Convert to half precision if not already
                if model.dtype != torch.float16:
                    model = model.half()
                    logger.info("Converted model to FP16")
            except Exception as e:
                logger.warning(f"Could not convert to FP16: {e}")

        return model, tokenizer

    @staticmethod
    def benchmark_generation(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        test_prompts: List[str],
        max_new_tokens: int = 50,
        num_runs: int = 5
    ) -> Dict[str, float]:
        """Benchmark model generation performance"""
        import time

        logger.info("Benchmarking generation performance...")

        times = []
        token_counts = []

        for i in range(num_runs):
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                start_time = time.time()

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                end_time = time.time()

                # Count generated tokens
                generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]

                times.append(end_time - start_time)
                token_counts.append(generated_tokens)

        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_second = avg_tokens / avg_time

        return {
            "avg_generation_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "total_runs": len(times)
        }


class ModelAnalyzer:
    """Analyze model architecture and parameters"""

    @staticmethod
    def analyze_model_parameters(model: PreTrainedModel) -> Dict[str, Any]:
        """Analyze model parameters distribution"""
        param_info = {}
        total_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count

            if param.requires_grad:
                trainable_params += param_count

            # Group by layer type
            layer_type = name.split('.')[0] if '.' in name else name
            if layer_type not in param_info:
                param_info[layer_type] = {
                    "total_params": 0,
                    "trainable_params": 0,
                    "layers": []
                }

            param_info[layer_type]["total_params"] += param_count
            if param.requires_grad:
                param_info[layer_type]["trainable_params"] += param_count

            param_info[layer_type]["layers"].append({
                "name": name,
                "shape": list(param.shape),
                "params": param_count,
                "trainable": param.requires_grad,
                "dtype": str(param.dtype)
            })

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "parameter_breakdown": param_info,
            "model_size_mb": (total_params * 4) / (1024 * 1024),  # Assume FP32
        }

    @staticmethod
    def analyze_memory_usage(model: PreTrainedModel) -> Dict[str, float]:
        """Analyze model memory usage"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        # Get GPU memory before
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()

        # Move model to GPU if not already there
        device = next(model.parameters()).device
        if device.type != 'cuda':
            model = model.cuda()

        memory_after = torch.cuda.memory_allocated()
        model_memory = memory_after - memory_before

        return {
            "model_memory_mb": model_memory / (1024 * 1024),
            "total_allocated_mb": memory_after / (1024 * 1024),
            "total_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "gpu_utilization": torch.cuda.utilization()
        }


class ModelUtils:
    """Utility functions for model operations"""

    @staticmethod
    def get_model_dtype_distribution(model: PreTrainedModel) -> Dict[str, int]:
        """Get distribution of parameter data types"""
        dtype_counts = {}
        for param in model.parameters():
            dtype = str(param.dtype)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + param.numel()
        return dtype_counts

    @staticmethod
    def estimate_training_memory(
        model: PreTrainedModel,
        batch_size: int = 1,
        sequence_length: int = 2048,
        optimizer: str = "adamw"
    ) -> Dict[str, float]:
        """Estimate memory required for training"""
        param_count = sum(p.numel() for p in model.parameters())

        # Model parameters (FP16)
        model_memory = param_count * 2

        # Gradients (FP16)
        gradient_memory = param_count * 2

        # Optimizer states
        if optimizer.lower() == "adamw":
            # Adam: momentum + variance (FP32)
            optimizer_memory = param_count * 8
        elif optimizer.lower() == "sgd":
            # SGD: momentum (FP32)
            optimizer_memory = param_count * 4
        else:
            optimizer_memory = param_count * 4  # Conservative estimate

        # Activations (rough estimate)
        hidden_size = getattr(model.config, 'hidden_size', 4096)
        activation_memory = batch_size * sequence_length * hidden_size * 2  # FP16

        total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory

        return {
            "model_memory_gb": model_memory / (1024**3),
            "gradient_memory_gb": gradient_memory / (1024**3),
            "optimizer_memory_gb": optimizer_memory / (1024**3),
            "activation_memory_gb": activation_memory / (1024**3),
            "total_memory_gb": total_memory / (1024**3),
            "parameters": param_count
        }

    @staticmethod
    def save_model_info(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_path: str,
        additional_info: Optional[Dict] = None
    ) -> None:
        """Save comprehensive model information"""
        info = {
            "model_name": model.config._name_or_path if hasattr(model.config, '_name_or_path') else "unknown",
            "model_type": model.config.model_type,
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "parameters": ModelAnalyzer.analyze_model_parameters(model),
            "config": model.config.to_dict(),
        }

        if additional_info:
            info.update(additional_info)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info(f"Model info saved to {output_path}")

    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """Validate model configuration and return warnings/errors"""
        warnings = []

        # Check quantization compatibility
        if config.load_in_4bit and config.load_in_8bit:
            warnings.append("Both 4-bit and 8-bit quantization enabled, 4-bit will take precedence")

        # Check target modules
        if config.peft_method and not config.target_modules:
            warnings.append("PEFT method specified but no target modules defined")

        # Check memory requirements
        if config.load_in_4bit or config.load_in_8bit:
            if config.torch_dtype == torch.float32:
                warnings.append("Using FP32 with quantization may not provide expected memory savings")

        # Check gradient checkpointing
        if config.gradient_checkpointing and config.use_cache:
            warnings.append("Gradient checkpointing with use_cache=True may cause issues")

        return warnings


# Convenience functions for common operations
def load_model_for_training(
    model_name: str,
    peft_method: str = "lora",
    quantization: str = "none",
    **kwargs
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """
    Convenience function to load a model ready for PEFT training

    Args:
        model_name: HuggingFace model name or path
        peft_method: PEFT method to use ("lora", "adapter", etc.)
        quantization: Quantization method ("none", "4bit", "8bit")
        **kwargs: Additional configuration parameters

    Returns:
        Tuple of (peft_model, tokenizer)
    """
    config = ModelConfig(
        model_name_or_path=model_name,
        peft_method=PEFTMethod(peft_method),
        load_in_4bit=(quantization == "4bit"),
        load_in_8bit=(quantization == "8bit"),
        **kwargs
    )

    # Validate configuration
    warnings = ModelUtils.validate_model_config(config)
    for warning in warnings:
        logger.warning(warning)

    # Load base model
    model, tokenizer = ModelLoader.load_model_and_tokenizer(config)

    # Apply PEFT
    model = PEFTManager.apply_peft(model, config)

    return model, tokenizer


def load_model_for_inference(
    model_name: str,
    adapter_path: Optional[str] = None,
    quantization: str = "none",
    optimization_level: str = "medium",
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Convenience function to load a model optimized for inference

    Args:
        model_name: HuggingFace model name or path
        adapter_path: Optional path to PEFT adapters
        quantization: Quantization method ("none", "4bit", "8bit")
        optimization_level: Inference optimization level
        **kwargs: Additional configuration parameters

    Returns:
        Tuple of (model, tokenizer)
    """
    config = ModelConfig(
        model_name_or_path=model_name,
        load_in_4bit=(quantization == "4bit"),
        load_in_8bit=(quantization == "8bit"),
        use_cache=True,
        gradient_checkpointing=False,
        **kwargs
    )

    if adapter_path:
        # Load PEFT model
        model, tokenizer = PEFTManager.load_peft_model(model_name, adapter_path, config)
    else:
        # Load base model
        model, tokenizer = ModelLoader.load_model_and_tokenizer(config)

    # Optimize for inference
    model, tokenizer = InferenceOptimizer.optimize_for_inference(
        model, tokenizer, optimization_level
    )

    return model, tokenizer


def get_recommended_config(model_name: str) -> ModelConfig:
    """
    Get recommended configuration for a model based on system resources

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Recommended ModelConfig
    """
    # Get system info
    gpu_info = SystemResourceManager.get_gpu_info()
    memory_info = SystemResourceManager.get_memory_info()
    model_info = ModelLoader.get_model_info(model_name)

    # Get recommendations
    if gpu_info["available"]:
        gpu_memory = gpu_info["gpus"][0]["memory_total"] / 1024  # Convert to GB
        recommendations = SystemResourceManager.recommend_model_config(
            model_info.get("estimated_size_gb", 4.0),
            gpu_memory
        )
    else:
        recommendations = {"quantization": "4bit", "warnings": ["No GPU detected"]}

    # Create config based on recommendations
    config = ModelConfig(
        model_name_or_path=model_name,
        load_in_4bit=(recommendations.get("quantization") == "4bit"),
        load_in_8bit=(recommendations.get("quantization") == "8bit"),
    )

    # Log recommendations
    for warning in recommendations.get("warnings", []):
        logger.warning(warning)

    return config


# Example usage and testing functions
def test_model_loading():
    """Test basic model loading functionality"""
    try:
        # Test with a small model
        model_name = "microsoft/DialoGPT-small"

        logger.info("Testing basic model loading...")
        config = ModelConfig(model_name_or_path=model_name)
        model, tokenizer = ModelLoader.load_model_and_tokenizer(config)

        logger.info("Testing model analysis...")
        param_info = ModelAnalyzer.analyze_model_parameters(model)
        logger.info(f"Model has {param_info['total_parameters']:,} parameters")

        logger.info("Testing inference optimization...")
        model, tokenizer = InferenceOptimizer.optimize_for_inference(model, tokenizer)

        logger.info("All tests passed!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # Print system information
    print("=== System Information ===")
    gpu_info = SystemResourceManager.get_gpu_info()
    memory_info = SystemResourceManager.get_memory_info()

    print(f"GPU Available: {gpu_info['available']}")
    if gpu_info["available"]:
        print(f"GPU Count: {gpu_info['count']}")
        if "gpus" in gpu_info:
            for i, gpu in enumerate(gpu_info["gpus"]):
                print(f"  GPU {i}: {gpu['name']} ({gpu['memory_total']}MB)")

    print(f"System Memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available")

    # Run basic tests
    print("\n=== Running Tests ===")
    test_model_loading()