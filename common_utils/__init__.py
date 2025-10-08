"""
Common Utilities for PEFT Training Experiments
PEFT 實驗室共用工具模組

提供模型載入、數據處理、視覺化、訓練輔助等功能。
"""

__version__ = "1.1.0"
__author__ = "LLM Teaching Project Team"

# 導入核心模組
from .model_helpers import (
    ModelType,
    PEFTMethod,
    load_model_with_peft,
    create_peft_config,
    merge_and_save_model,
    SystemResourceManager
)

from .data_loaders import (
    InstructionDataset,
    InstructionDataCollator,
    PromptTemplate,
    load_alpaca_dataset,
    create_prompt
)

from .visualization import (
    plot_training_curves,
    plot_loss_comparison,
    plot_peft_comparison,
    plot_parameter_efficiency,
    plot_parameter_distribution,
    plot_inference_benchmark,
    plot_memory_usage,
    PEFT_COLORS
)

from .training_helpers import (
    check_gpu_availability,
    get_device,
    clear_gpu_cache,
    print_gpu_memory_usage,
    load_latest_checkpoint,
    save_checkpoint_safe,
    validate_training_config,
    print_trainable_parameters,
    get_layer_wise_parameters,
    TrainingMonitor,
    safe_load_model,
    safe_load_dataset,
    pre_training_checklist,
    analyze_training_results
)

__all__ = [
    # Model helpers
    'ModelType',
    'PEFTMethod',
    'load_model_with_peft',
    'create_peft_config',
    'merge_and_save_model',
    'SystemResourceManager',

    # Data loaders
    'InstructionDataset',
    'InstructionDataCollator',
    'PromptTemplate',
    'load_alpaca_dataset',
    'create_prompt',

    # Visualization
    'plot_training_curves',
    'plot_loss_comparison',
    'plot_peft_comparison',
    'plot_parameter_efficiency',
    'plot_parameter_distribution',
    'plot_inference_benchmark',
    'plot_memory_usage',
    'PEFT_COLORS',

    # Training helpers
    'check_gpu_availability',
    'get_device',
    'clear_gpu_cache',
    'print_gpu_memory_usage',
    'load_latest_checkpoint',
    'save_checkpoint_safe',
    'validate_training_config',
    'print_trainable_parameters',
    'get_layer_wise_parameters',
    'TrainingMonitor',
    'safe_load_model',
    'safe_load_dataset',
    'pre_training_checklist',
    'analyze_training_results',
]

print(f"✅ Common Utils v{__version__} 載入完成")
print(f"可用模組: model_helpers, data_loaders, visualization, training_helpers")
