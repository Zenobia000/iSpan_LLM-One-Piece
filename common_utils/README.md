# Common Utilities 使用指南
## PEFT 實驗室共用工具模組

**版本**: v1.1.0
**最後更新**: 2025-10-09

---

## 📚 模組概述

`common_utils` 提供所有 PEFT 實驗室共用的工具函數，包含模型載入、數據處理、視覺化、訓練輔助等功能，確保代碼一致性與可維護性。

### 模組結構

```
common_utils/
├── __init__.py              # 模組入口
├── model_helpers.py         # 模型管理工具 (878行)
├── data_loaders.py          # 數據載入工具 (996行)
├── visualization.py         # 視覺化工具 (新增, 370行)
├── training_helpers.py      # 訓練輔助工具 (新增, 380行)
└── README.md               # 本文檔
```

---

## 🚀 快速開始

### 基本導入

```python
# 導入所有工具
from common_utils import *

# 或選擇性導入
from common_utils import load_model_with_peft, plot_training_curves
from common_utils.visualization import PEFT_COLORS
```

### 典型使用流程

```python
# 1. 檢查 GPU
gpu_info = check_gpu_availability()
device = get_device()

# 2. 載入模型
model = load_model_with_peft(
    model_name="meta-llama/Llama-2-7b-hf",
    peft_method=PEFTMethod.LORA,
    peft_config={'r': 8, 'lora_alpha': 16}
)

# 3. 載入數據
dataset = load_alpaca_dataset(num_samples=1000)

# 4. 訓練前檢查
passed, issues = pre_training_checklist(model, dataset, "./output")

# 5. 訓練（使用 HuggingFace Trainer）
trainer = Trainer(...)
trainer.train()

# 6. 視覺化結果
plot_training_curves(trainer.state.log_history)

# 7. 分析結果
results = analyze_training_results(trainer, "./output")
```

---

## 📦 模組詳細說明

### 1. model_helpers.py

**主要功能**: 模型載入、PEFT 配置、量化、記憶體監控

#### 核心類別

**ModelType (Enum)**
```python
class ModelType(Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    GEMMA = "gemma"
    BERT = "bert"
    GPT2 = "gpt2"
```

**PEFTMethod (Enum)**
```python
class PEFTMethod(Enum):
    LORA = "lora"
    QLORA = "qlora"
    ADAPTER = "adapter"
    PREFIX_TUNING = "prefix_tuning"
    # ... 其他方法
```

#### 主要函數

**load_model_with_peft()**
```python
model = load_model_with_peft(
    model_name="meta-llama/Llama-2-7b-hf",
    peft_method=PEFTMethod.LORA,
    peft_config={'r': 8, 'lora_alpha': 16, 'target_modules': ['q_proj', 'v_proj']},
    quantization_config={'load_in_4bit': True}
)
```

**create_peft_config()**
```python
config = create_peft_config(
    method=PEFTMethod.LORA,
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16
)
```

---

### 2. data_loaders.py

**主要功能**: 指令數據集處理、多格式提示模板、批次收集

#### 核心類別

**InstructionDataset**
```python
dataset = InstructionDataset(
    data=alpaca_data,
    tokenizer=tokenizer,
    template_type=PromptTemplate.ALPACA,
    max_length=512
)
```

**InstructionDataCollator**
```python
collator = InstructionDataCollator(
    tokenizer=tokenizer,
    padding=True,
    max_length=512
)
```

#### 提示模板

支援多種格式:
- `PromptTemplate.ALPACA`: Alpaca 格式
- `PromptTemplate.DOLLY`: Dolly 格式
- `PromptTemplate.CHATML`: ChatML 格式
- `PromptTemplate.CUSTOM`: 自定義格式

---

### 3. visualization.py ⭐ 新增

**主要功能**: 統一的視覺化工具，確保所有實驗室圖表風格一致

#### 配色方案

```python
PEFT_COLORS = {
    'LoRA': '#FF6B6B',
    'Adapter': '#4ECDC4',
    'IA3': '#45B7D1',
    'train': '#3498db',
    'eval': '#e74c3c'
}
```

#### 核心函數

**plot_training_curves()** - 訓練曲線
```python
history = {
    'train_loss': [3.2, 2.8, 2.5, 2.3],
    'eval_loss': [3.5, 3.0, 2.7, 2.6],
    'eval_perplexity': [33.1, 20.1, 14.9, 13.5]
}

plot_training_curves(history, save_path="training_curves.png")
```

**plot_peft_comparison()** - PEFT 方法對比
```python
results = [
    {'method': 'LoRA', 'trainable_params_%': 0.5, 'performance': 85.2},
    {'method': 'Adapter', 'trainable_params_%': 2.0, 'performance': 86.1},
    {'method': 'IA3', 'trainable_params_%': 0.01, 'performance': 84.5}
]

plot_peft_comparison(results, metrics=['trainable_params_%', 'performance'])
```

**plot_parameter_distribution()** - 參數分佈
```python
stats = print_trainable_parameters(model, verbose=False)
plot_parameter_distribution(stats)
```

**快速使用**: Trainer 整合
```python
from common_utils.visualization import quick_plot_trainer_history

trainer.train()
quick_plot_trainer_history(trainer, save_dir="./output")
```

---

### 4. training_helpers.py ⭐ 新增

**主要功能**: 錯誤處理、資源管理、訓練監控

#### GPU 管理

**check_gpu_availability()**
```python
gpu_info = check_gpu_availability(verbose=True)
# 輸出:
# ============================================================
# GPU 環境檢查
# ============================================================
# ✅ CUDA 可用
# GPU 數量: 1
# GPU 型號: NVIDIA RTX 4060 Ti
# 總記憶體: 16.00 GB
# CUDA 版本: 12.1
```

**get_device()** - 自動選擇設備
```python
device = get_device()  # 自動選擇 cuda/mps/cpu
```

#### 檢查點管理

**load_latest_checkpoint()**
```python
checkpoint_path = load_latest_checkpoint(
    output_dir="./output",
    prefix="checkpoint-"
)
# ✅ 載入檢查點: ./output/checkpoint-1000
```

#### 訓練驗證

**validate_training_config()**
```python
config = {
    'learning_rate': 5e-5,
    'num_train_epochs': 3,
    'per_device_train_batch_size': 4
}

is_valid, warnings = validate_training_config(config)
for warning in warnings:
    print(warning)
```

**pre_training_checklist()**
```python
passed, issues = pre_training_checklist(
    model=model,
    train_dataset=dataset,
    output_dir="./output"
)

if passed:
    # 開始訓練
    trainer.train()
```

#### 訓練監控

**TrainingMonitor**
```python
monitor = TrainingMonitor(log_interval=100)

for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    monitor.log_step(loss, metrics={'lr': current_lr})

history = monitor.get_history()
plot_training_curves(history)
```

---

## 💡 最佳實踐

### 在實驗室中使用

**推薦的導入結構** (在 notebook 頂部):

```python
# 01-Setup.ipynb
from common_utils import (
    check_gpu_availability,
    get_device,
    safe_load_model,
    safe_load_dataset
)

# 02-Train.ipynb
from common_utils import (
    load_model_with_peft,
    create_peft_config,
    load_alpaca_dataset,
    InstructionDataCollator,
    print_trainable_parameters,
    plot_training_curves,
    analyze_training_results
)

# 03-Inference.ipynb
from common_utils import (
    load_latest_checkpoint,
    plot_inference_benchmark
)

# 04-Merge_and_Deploy.ipynb
from common_utils import (
    merge_and_save_model,
    plot_parameter_distribution
)
```

### 標準化訓練流程

```python
# Step 1: 環境檢查
check_gpu_availability()
device = get_device()

# Step 2: 載入模型與數據
model = load_model_with_peft(...)
dataset = load_alpaca_dataset(...)

# Step 3: 訓練前檢查
passed, issues = pre_training_checklist(model, dataset, output_dir)
if not passed:
    for issue in issues:
        print(f"⚠️  {issue}")
    # 決定是否繼續

# Step 4: 訓練
trainer = Trainer(...)
trainer.train()

# Step 5: 視覺化與分析
plot_training_curves(trainer.state.log_history)
results = analyze_training_results(trainer, output_dir)
```

---

## 🔧 高級用法

### 自定義視覺化

```python
from common_utils.visualization import plt, PEFT_COLORS

# 使用統一配色
plt.plot(data, color=PEFT_COLORS['LoRA'], label='LoRA r=8')
plt.plot(data2, color=PEFT_COLORS['train'], label='Training')
```

### 記憶體監控

```python
# 訓練前
print_gpu_memory_usage("訓練前: ")

# 訓練中
for step in range(num_steps):
    loss = train_step()
    if step % 100 == 0:
        print_gpu_memory_usage(f"Step {step}: ")

# 訓練後
clear_gpu_cache()
```

### 組合使用範例

```python
# 完整的訓練與分析流程
def train_and_analyze(model, dataset, config):
    # 驗證配置
    is_valid, warnings = validate_training_config(config)

    # 訓練前檢查
    passed, issues = pre_training_checklist(model, dataset, config['output_dir'])

    # 訓練
    trainer = Trainer(model, args=config, train_dataset=dataset)
    trainer.train()

    # 視覺化
    plot_training_curves(trainer.state.log_history)

    # 分析
    results = analyze_training_results(trainer, config['output_dir'])

    return results
```

---

## 📊 模組統計

| 模組 | 代碼行數 | 函數數 | 類別數 | 用途 |
|------|---------|--------|--------|------|
| model_helpers.py | 878 | 15+ | 3 | 模型管理 |
| data_loaders.py | 996 | 10+ | 3 | 數據處理 |
| visualization.py | 370 | 12 | 0 | 視覺化 |
| training_helpers.py | 380 | 15+ | 1 | 訓練輔助 |
| **總計** | **2,624** | **52+** | **7** | - |

---

## 🎯 更新日誌

### v1.1.0 (2025-10-09)
- ✅ 新增 `visualization.py` - 統一視覺化工具
- ✅ 新增 `training_helpers.py` - 訓練輔助與錯誤處理
- ✅ 更新 `__init__.py` - 整合新模組
- ✅ 新增本 README 文檔

### v1.0.0 (2025-10-08)
- ✅ 完成 `model_helpers.py` - 模型載入與 PEFT 配置
- ✅ 完成 `data_loaders.py` - 指令數據集處理

---

## 📝 貢獻指南

### 新增工具函數

1. 確定所屬模組 (model/data/viz/training)
2. 遵循現有命名規範
3. 添加完整的 docstring
4. 包含使用範例
5. 更新 `__init__.py` 的 `__all__`

### 代碼風格

- 遵循 PEP 8
- 使用 Type Hints
- 中文註解 + 英文 docstring
- 完整的錯誤處理

---

## 🆘 常見問題

### Q: 為什麼導入失敗?
```python
# 錯誤
from common_utils import plot_training_curves  # ModuleNotFoundError

# 解決方案: 確保在正確目錄
import sys
sys.path.append('/path/to/iSpan_LLM-One-Piece')
from common_utils import plot_training_curves
```

### Q: 如何自定義視覺化風格?
```python
from common_utils.visualization import PEFT_COLORS

# 修改配色
PEFT_COLORS['LoRA'] = '#YOUR_COLOR'

# 或使用自己的配色
my_colors = {'method1': '#color1', 'method2': '#color2'}
```

### Q: GPU 記憶體不足怎麼辦?
```python
# 使用工具函數診斷
print_gpu_memory_usage()

# 清空緩存
clear_gpu_cache()

# 檢查配置
validate_training_config(config)  # 會給出記憶體警告
```

---

## 🔗 相關資源

- **PEFT Labs**: `01-Core_Training_Techniques/02-Labs/PEFT_Labs/`
- **理論文件**: `01-Core_Training_Techniques/01-Theory/1.1-PEFT.md`
- **專案文檔**: `docs/`

---

**維護者**: LLM 教學專案團隊
**授權**: MIT License
**貢獻**: 歡迎提交 PR 改進工具函數
