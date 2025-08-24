# Lab 1: LoRA & QLoRA - 高效微調大型語言模型

## 概述

**低秩適應 (LoRA, Low-Rank Adaptation)** 是當前最流行、應用最廣泛的參數高效微調 (PEFT) 方法之一。其核心思想是，在不改變預訓練模型原始權重的前提下，通過注入可訓練的低秩矩陣來適應新任務。**QLoRA** 則是 LoRA 的一種高效變體，它結合了 4-bit 量化技術，極大地降低了微調大型語言模型（如 Llama-2）所需的記憶體。

本實驗將深入探討 LoRA 與 QLoRA 的原理，並實際操作使用 QLoRA 微調一個 70 億參數的 Llama-2 模型。

![LoRA 原理示意圖](https://pic3.zhimg.com/v2-1ee7ae98a860e5f9aff51d5b1c833296_1440w.jpg)

---

## 1. 技術背景與動機

對大型語言模型 (LLM) 進行全參數微調 (Full Fine-Tuning) 成本極其高昂：

- **巨大的記憶體需求**：微調一個 7B 參數的模型通常需要數十 GB 的 GPU 記憶體。
- **高昂的儲存成本**：每微調一個任務，就需要保存一份完整的模型副本（例如，Llama-2-7B 的 `bfloat16` 權重約 14 GB）。
- **部署困難**：在多任務場景下，管理和提供多個大型模型副本變得非常複雜。

LoRA 的出現旨在解決這些問題，它允許我們以極小的成本（僅保存數十 MB 的適配器權重）來微調模型，並實現與全參數微調相近的性能。

---

## 2. LoRA 核心原理

### 2.1 低秩假設

LoRA 的理論基礎來源於一個關鍵假設：**模型在適應下游任務時，其權重矩陣的變化量（`ΔW`）具有很低的“內在秩” (intrinsic rank)**。換言之，儘管 `ΔW` 的維度很大，但它可以用兩個更小的矩陣相乘來高效地表示，而不會損失太多信息。

### 2.2 技術實現

LoRA 的實現方式如下：
1.  **凍結原始權重**：在微調過程中，預訓練模型的原始權重 \( W_0 \) 保持不變。
2.  **注入低秩矩陣**：在模型的特定層（通常是 Transformer 的注意力層），注入兩個可訓練的低秩矩陣 \( A \) 和 \( B \)。
3.  **計算權重變化**：權重的變化量 \( \Delta W \) 由這兩個小矩陣相乘得到：\( \Delta W = B \cdot A \)。其中，\( A \) 的維度是 \( d \times r \)，\( B \) 的維度是 \( r \times k \)，而秩 \( r \) 遠小於原始維度 \( d \) 和 \( k \)。
4.  **前向傳播**：模型的前向傳播結果由 \( h = W_0 x + \Delta W x = W_0 x + B(Ax) \) 計算得出。

通過這種方式，需要訓練的參數數量從 \( d \times k \) 大幅減少到 \( r \times (d+k) \)，極大地提高了參數效率。

### 2.3 QLoRA：量化與 LoRA 的結合

QLoRA 進一步提升了效率：
- **4-bit NormalFloat (NF4)**：將凍結的基礎模型權重從 16-bit 或 32-bit 量化到 4-bit，極大降低了模型載入所需的記憶體。
- **Double Quantization**：對量化常數本身再次進行量化，進一步節省記憶體。
- **Paged Optimizers**：利用 NVIDIA 統一記憶體特性，防止在處理長序列時因梯度檢查點而導致的記憶體不足問題。

---

## 3. 實現原理與步驟

使用 Hugging Face `peft` 庫，可以輕鬆實現 LoRA 和 QLoRA。

### 3.1 關鍵配置 `LoraConfig`

```python
from peft import LoraConfig, TaskType

# 定義 LoRA 配置
lora_config = LoraConfig(
    r=8,  # LoRA 適配器的秩 (rank)
    lora_alpha=16,  # 縮放因子，類似學習率
    target_modules=["q_proj", "v_proj"], # 要應用 LoRA 的模組
    lora_dropout=0.05, # LoRA 層的 dropout 概率
    bias="none", # 不訓練偏置項
    task_type=TaskType.CAUSAL_LM, # 任務類型
)
```

### 3.2 關鍵參數說明
- `r`: LoRA 的核心超參數，即低秩矩陣的秩。較大的 `r` 意味著更多的可訓練參數和更強的表達能力，但也可能導致過擬合。通常在 8 到 64 之間選擇。
- `lora_alpha`: 縮放因子。LoRA 的輸出會乘以 `lora_alpha / r`。這允許我們在不改變 `r` 的情況下調整 LoRA 適配器的權重大小，類似於一種學習率。
- `target_modules`: 一個列表，指定要將 LoRA 適配器應用於哪些模組。對於 Transformer 模型，通常是注意力機制中的查詢 (`q_proj`) 和值 (`v_proj`) 投影層。
- `lora_dropout`: 在 LoRA 層上應用的 dropout，用於正則化。
- `bias`: 控制是否訓練偏置參數。`"none"` 表示不訓練，`"all"` 表示全部訓練，`"lora_only"` 表示僅訓練 LoRA 層的偏置。

### 3.3 部署：合併權重

LoRA 的一個巨大優勢是，在推理時，可以將適配器權重 \( \Delta W \) 與原始權重 \( W_0 \) 合併，得到一個新的權重矩陣 \( W' = W_0 + \Delta W \)。這樣做的好處是：
- **無推理延遲**：合併後，模型結構與原始模型完全相同，不會引入任何額外的計算開銷。
- **部署簡潔**：只需部署一個單一的模型文件，無需管理額外的適配器權重。

```python
# 將適配器權重合併到基礎模型中
merged_model = peft_model.merge_and_unload()
```

---

## 4. 性能表現與對比

LoRA 在性能、效率和易用性之間取得了出色的平衡。

![PEFT 方法對比](https://pic4.zhimg.com/v2-43e234f226c6a965cfab4c2a8173cc75_1440w.jpg)

| 方法 | 參數效率 | 訓練速度 | 推理延遲 | 性能 |
|:---|:---|:---|:---|:---|
| **LoRA/QLoRA** | **0.1-1%** | **快** | **無 (合併後)** | **非常接近全量微調** |
| Adapter | 2-4% | 中等 | 有 | 較好 |
| Prefix-Tuning| ~0.1% | 慢 | 有 | 適用於生成任務 |
| BitFit | ~0.08% | 快 | 無 | 性能有一定差距 |
| 全參數微調 | 100% | 慢 | 無 | 基準性能 |

---

## 5. 技術優勢

| 優勢項目 | 說明 |
|:---|:---|
| **高效的參數** | 僅需訓練極少數參數，極大降低了計算和儲存成本。 |
| **無推理延遲** | 適配器權重可以合併回基礎模型，不影響推理速度。 |
| **任務切換靈活** | 可以為多個任務訓練不同的 LoRA 適配器，並在推理時動態加載。 |
| **性能卓越** | 在多數任務上，性能與全參數微調相當或非常接近。 |
| **易於實現** | Hugging Face `peft` 庫提供了非常簡潔的 API。 |

---

## 6. 實驗設計與實作

### 6.1 實驗環境

- **模型**: `NousResearch/Llama-2-7b-hf`
- **任務**: 指令微調
- **數據集**: `mlabonne/guanaco-llama2-1k`
- **核心技術**: QLoRA (4-bit 量化)

### 6.2 實驗流程

1.  **環境準備** (`01-Setup.ipynb`)
    -   安裝 `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes`。
    -   檢查 GPU 可用性。

2.  **模型訓練** (`02-Train.ipynb`)
    -   使用 `BitsAndBytesConfig` 以 4-bit 精度加載 Llama-2 模型。
    -   定義 `LoraConfig`，指定 `r`, `lora_alpha`, `target_modules` 等。
    -   使用 `transformers.Trainer` 對 PEFT 模型進行微調。

3.  **推理測試** (`03-Inference.ipynb`)
    -   加載基礎模型和訓練好的 LoRA 適配器。
    -   使用 `PeftModel` 進行推理，驗證微調效果。

4.  **合併與部署** (`04-Merge_and_Deploy.ipynb`)
    -   使用 `merge_and_unload()` 將適配器權重合併到基礎模型中。
    -   將合併後的模型保存為標準的 Hugging Face 模型，以便於部署。

---

## 7. 實戰參數調優策略 (2024 年行業最佳實踐)

### 7.1 基於資料量規模的參數配置

| 資料集大小 | Rank (r) | Alpha | Learning Rate | Batch Size | Epochs |
|:---|:---|:---|:---|:---|:---|
| **小型資料集** (<1K 樣本) | 4-8 | 8-16 | 3e-4 | 4-8 | 3-5 |
| **中型資料集** (1K-10K 樣本) | 8-16 | 16-32 | 2e-4 | 8-16 | 2-3 |
| **大型資料集** (>10K 樣本) | 16-64 | 32-128 | 1e-4 | 16-32 | 1-2 |

### 7.2 不同業務場景的調優策略

#### 對話機器人 / 客服系統
```python
lora_config = LoraConfig(
    r=16,                    # 中等複雜度
    lora_alpha=32,          # Alpha = 2×rank 經驗法則
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,       # 防止過擬合
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

#### 代碼生成 / 技術文檔
```python
lora_config = LoraConfig(
    r=32,                   # 需要較強表達能力
    lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,      # 較低 dropout
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

#### 領域知識問答 / RAG 系統  
```python
lora_config = LoraConfig(
    r=8,                    # 較簡單的適應需求
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 專注注意力機制
    lora_dropout=0.1,
    bias="none", 
    task_type=TaskType.CAUSAL_LM
)
```

### 7.3 記憶體最佳化配置

#### 極限記憶體模式 (<16GB GPU)
```python
# QLoRA 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)
```

#### 平衡性能模式 (24-48GB GPU)
```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,      # 使用 8-bit 量化
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05
)
```

### 7.4 收斂監控與早停策略

```python
training_args = TrainingArguments(
    evaluation_strategy="steps",
    eval_steps=100,                # 每100步評估一次
    save_strategy="steps", 
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    early_stopping_patience=3       # 3輪無改善即停止
)
```

### 7.5 多任務場景的適配器管理

```python
# 為不同任務創建不同的適配器
task_configs = {
    "chat": LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]),
    "code": LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
    "summary": LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
}

# 動態加載不同任務的適配器
def switch_adapter(model, task_name):
    model.load_adapter(f"./adapters/{task_name}_adapter")
    return model
```

### 7.6 故障診斷與調優指南

| 問題現象 | 可能原因 | 解決方案 |
|:---|:---|:---|
| 訓練損失不下降 | 學習率過低/Rank 過小 | 提高學習率至 5e-4 或增加 Rank |
| 快速過擬合 | Rank 過大/缺少正則化 | 降低 Rank 或增加 dropout |
| GPU 記憶體不足 | 批次大小過大/精度過高 | 使用梯度累積/4-bit 量化 |
| 推理效果差 | 目標模組選擇不當 | 增加更多 target_modules |

---

## 8. 模型部署與生產環境最佳實踐

### 8.1 合併適配器權重的時機選擇

```python
# 開發階段：保持適配器分離便於調試
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# 生產部署：合併權重提升推理效率
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

### 8.2 多版本適配器管理

```python
# 版本化適配器管理
adapter_versions = {
    "v1.0": "./adapters/chat_v1.0", 
    "v1.1": "./adapters/chat_v1.1",
    "v2.0": "./adapters/chat_v2.0"
}

# A/B 測試支援
def load_adapter_by_version(model, version="latest"):
    adapter_path = adapter_versions.get(version, adapter_versions["v2.0"])
    return PeftModel.from_pretrained(model, adapter_path)
```

### 8.3 推理效能監控

```python
import time
import psutil

def monitor_inference(model, inputs):
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / 1024**3
    
    outputs = model.generate(**inputs)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used / 1024**3
    
    metrics = {
        "inference_time": end_time - start_time,
        "memory_usage": end_memory - start_memory,
        "tokens_per_second": len(outputs[0]) / (end_time - start_time)
    }
    return outputs, metrics
```

---

## 9. 結論與學習成果

通過本實驗，您將獲得以下核心能力：

1.  **深度理解** LoRA 和 QLoRA 的技術原理與數學基礎
2.  **實戰經驗** 在真實大型語言模型上應用 QLoRA 進行高效微調
3.  **參數調優** 掌握基於資料量和業務場景的參數最佳化策略
4.  **工程實踐** 熟練使用 `peft` 庫進行完整的訓練-推理-部署流程
5.  **生產部署** 理解多適配器管理和性能監控的工程最佳實踐

LoRA/QLoRA 已成為 LLM 個性化和領域適應的行業黃金標準。掌握這些技術和最佳實踐，將為您在資源有限條件下釋放大型模型潛力提供強有力的工具。

---

## 10. 技術限制與改進方向

### 10.1 訓練階段限制分析

| 限制項目 | 具體表現 | 效能影響 | 解決方案 |
|:---|:---|:---|:---|
| **記憶體瓶頸** | QLoRA 4-bit 量化仍需 13-15GB GPU 記憶體 | 限制模型規模和批次大小 | 使用梯度累積、更小 rank |
| **訓練不穩定** | 量化誤差導致梯度噪聲 | 收斂困難、性能波動 | 調整學習率、使用 warmup |
| **超參敏感性** | rank 和 alpha 對性能影響巨大 | 需大量調參實驗 | 使用經驗法則 alpha=2×rank |
| **資料相依性** | 小資料集容易過擬合 | 泛化能力差 | 降低 rank、增加 dropout |
| **量化精度損失** | 4-bit NF4 引入量化誤差 | 訓練精度下降 1-2% | 使用 double quantization |

### 10.2 推理階段限制分析

| 限制項目 | 具體表現 | 效能影響 | 解決方案 |
|:---|:---|:---|:---|
| **動態載入延遲** | 切換 adapter 需要載入權重 | 首次推理延遲 100-500ms | 預載入常用 adapter |
| **記憶體碎片** | 多 adapter 同時載入 | 記憶體使用效率降低 | 使用 adapter 池化管理 |
| **併發限制** | 同一模型難以並行處理不同任務 | 吞吐量受限 | 權重合併或多實例部署 |
| **精度退化** | 量化推理可能累積誤差 | 長序列生成品質下降 | 關鍵路徑使用 FP16 |
| **模型大小** | 多個 adapter 累積存儲開銷 | 磁盤空間和載入時間增加 | adapter 壓縮和剪枝 |

### 10.3 效能瓶頸深度分析

#### 訓練效能基準測試 (Llama-2-7B)

| 配置 | GPU 記憶體 | 訓練速度 | 收斂輪數 | 最終性能 |
|:---|:---|:---|:---|:---|
| **全參數微調** | 80GB | 0.5 samples/s | 1-2 epochs | 100% (基準) |
| **LoRA r=8** | 24GB | 2.1 samples/s | 2-3 epochs | 98.5% |
| **LoRA r=16** | 28GB | 1.8 samples/s | 2-3 epochs | 99.1% |
| **QLoRA r=16** | 15GB | 1.2 samples/s | 3-4 epochs | 98.8% |
| **QLoRA r=32** | 18GB | 0.9 samples/s | 3-4 epochs | 99.0% |

#### 推理效能基準測試

| 場景 | 延遲 | 吞吐量 | 記憶體占用 | 備註 |
|:---|:---|:---|:---|:---|
| **原模型** | 45ms | 22.2 tokens/s | 13GB | 基準性能 |
| **動態 LoRA** | 48ms (+7%) | 20.8 tokens/s (-6%) | 13.2GB | 熱切換 overhead |
| **合併 LoRA** | 45ms (0%) | 22.1 tokens/s (-0.5%) | 13GB | 無額外開銷 |
| **多 adapter 並行** | 52ms (+16%) | 19.2 tokens/s (-14%) | 15.5GB | 資源競爭 |

### 10.4 瓶頸突破策略與未來方向

#### 訓練最佳化
- **記憶體最佳化**: 透過梯度檢查點 (gradient checkpointing)、梯度累積 (gradient accumulation)、混合精度 (fp16/bf16) 和 8-bit 優化器 (adamw_bnb_8bit) 等技術，可以在有限的硬體上訓練更大的模型。
- **訓練穩定性**: 採用 warmup 策略、餘弦學習率退火 (cosine scheduler) 和梯度裁剪 (max_grad_norm) 對於穩定 QLoRA 等量化訓練至關重要。

#### 推理最佳化
- **Adapter 池化**: 對於需要動態切換多個 LoRA 適配器的場景，可以實現 LRU (Least Recently Used) 緩存池來管理適配器，減少載入延遲和記憶體碎片。
- **批次推理**: 將相同任務（使用相同 LoRA 適配器）的請求分組進行批次推理，可以顯著減少適配器切換開銷，提高吞吐量。

#### 未來研究方向
- **自動化 Rank 選擇**: 研究基於任務複雜度和數據集大小自動選擇最優 `r` 值的算法。
- **結構化 LoRA**: 探索將 LoRA 與其他結構（如 Adapter）結合，或在 LoRA 矩陣中引入稀疏性，以進一步提升效率。
- **LoRA 與量化結合**: QLoRA 是成功的開端，未來將探索更先進的量化技術與 LoRA 的結合，以在更低比特下保持性能。
- **跨層 Rank 分配**: 研究為 Transformer 的不同層分配不同的 `r` 值，以實現更精細的參數效率控制。

---

## 11. 參考資料

### 核心論文
- **LoRA**: Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- **QLoRA**: Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.

### 延伸閱讀與工具
- **Hugging Face PEFT**: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- **QLoRA 官方實現**: [https://github.com/artidoro/qlora](https://github.com/artidoro/qlora) 
- **Databricks LoRA 調優指南**: [https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- **行業最佳實踐 2024**: Lightning AI LoRA Insights

### 模型與資料集
- **基礎模型**: [NousResearch/Llama-2-7b-hf](https://huggingface.co/NousResearch/Llama-2-7b-hf)
- **訓練資料**: [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)
- **更大訓練集**: [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

