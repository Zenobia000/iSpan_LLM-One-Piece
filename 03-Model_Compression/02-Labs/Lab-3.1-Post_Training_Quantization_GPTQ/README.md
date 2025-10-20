# Lab 3.1: GPTQ 訓練後量化 - 高效壓縮大型語言模型

## 概述

**GPTQ (Generative Pre-trained Transformer Quantization)** 是當前最先進、應用最廣泛的訓練後量化 (Post-Training Quantization, PTQ) 方法之一。其核心思想是,通過 **Hessian 矩陣指導的逐層誤差補償**,在不需要重新訓練的情況下,將 FP16 權重量化為 INT4/INT8,實現 3-4 倍的模型壓縮,同時保持精度損失 <2%。

本實驗將深入探討 GPTQ 的原理,並實際操作使用 GPTQ 量化一個 70 億參數的 Llama-2 模型,從 13.5GB 壓縮至 3.5GB,推理速度提升 2.8 倍。

![GPTQ 量化示意圖](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Thumbnail_blue.png)

---

## 1. 技術背景與動機

### 1.1 為何需要模型量化?

在大型語言模型 (LLM) 的時代,全精度推理 (FP16/FP32) 面臨三大挑戰:

- **記憶體瓶頸**: Llama-2-70B 的 FP16 權重需要 140GB 記憶體,超過單卡 GPU 容量。
- **部署成本**: 邊緣設備 (如移動端、IoT) 無法負擔數十 GB 的模型載入。
- **推理延遲**: 高精度浮點運算速度慢,無法滿足實時應用需求 (如聊天機器人)。

量化技術通過降低數值精度 (FP16 → INT4),可以:
- **減少模型大小 75%** (13.5GB → 3.5GB for Llama-2-7B)
- **加速推理 2-4x** (利用低位元運算的硬體加速)
- **降低記憶體占用 70%** (GPU 記憶體從 15GB → 5GB)

### 1.2 量化技術分類

```
量化方法
├─ 訓練後量化 (PTQ) ← 本實驗重點
│   ├─ 簡單量化 (Round-to-Nearest, RTN)
│   ├─ GPTQ (Hessian 誤差補償)
│   ├─ AWQ (激活感知權重量化)
│   └─ SmoothQuant (異常值平滑)
│
└─ 量化感知訓練 (QAT)
    ├─ QLoRA (4-bit 量化 + LoRA 微調)
    └─ LLM-QAT (完整重訓練)
```

**GPTQ 的優勢**:
- ✅ 無需重新訓練 (幾小時內完成量化)
- ✅ 精度損失極小 (<2% perplexity 增加)
- ✅ 支持極低位元 (3-4 bit)
- ✅ 工業界成熟方案 (Hugging Face 原生支持)

---

## 2. GPTQ 核心原理

### 2.1 數學基礎: 最優化量化問題

GPTQ 的目標是求解以下最優化問題:

```
最小化: ||W·X - Ŵ·X||²
其中:
  W  = 原始 FP16 權重矩陣
  Ŵ  = 量化後 INT4 權重矩陣
  X  = 校準數據的激活值
```

**關鍵洞察**: 該問題等價於在「二階泰勒展開」的誤差約束下,尋找最優量化參數。

### 2.2 Hessian 矩陣與敏感度

GPTQ 使用 **Hessian 矩陣 (二階導數)** 來衡量權重的重要性:

```python
# Hessian 近似 (簡化形式)
H = 2 · X @ X.T

# 權重敏感度: Hessian 對角線值越大,權重越敏感
sensitivity = diag(H)
```

**物理意義**:
- 敏感度高 → 該權重對輸出影響大 → 量化時需更小心
- 敏感度低 → 可以更激進地量化

### 2.3 逐層量化與誤差補償

GPTQ 的創新點在於 **逐列量化 + 誤差傳播**:

```python
for i in range(num_columns):
    # 1. 量化第 i 列
    w_quantized[i] = quantize(w[i])

    # 2. 計算量化誤差
    error = w[i] - w_quantized[i]

    # 3. 將誤差分攤到後續未量化的列 (關鍵步驟!)
    w[i+1:] -= (error / H[i,i]) * H[i, i+1:]
```

**為何有效?**
- 誤差不會累積 → 每層保持「局部最優」
- Hessian 指導補償 → 敏感權重獲得更多補償
- 逐層獨立 → 可並行處理,速度快

### 2.4 分組量化 (Group Quantization)

為了進一步降低精度損失,GPTQ 使用 **分組量化**:

```python
# 每 128 個權重共享一個縮放因子
group_size = 128
for group in split(weights, group_size):
    scale = max(abs(group)) / (2^(bits-1) - 1)
    quantized_group = round(group / scale).clamp(-128, 127)
```

**效果**: 相比全張量量化 (Per-Tensor),精度損失從 5% 降至 <2%。

---

## 3. 實現原理與步驟

### 3.1 關鍵配置 `GPTQConfig`

```python
from transformers import GPTQConfig

# GPTQ 量化配置
quantization_config = GPTQConfig(
    bits=4,                    # 量化位元數: 3/4/8 bit
    group_size=128,            # 分組大小 (影響精度 vs 速度)
    desc_act=True,             # 激活值降序排序 (提升精度)
    sym=True,                  # 對稱量化 ([-127, 127])
    dataset="c4",              # 校準數據集 (越大越準確)
    tokenizer=tokenizer,       # 分詞器
    damp_percent=0.01,         # Hessian 阻尼係數 (穩定性)
    use_exllama=True,          # 使用 ExLlama 內核加速
)
```

### 3.2 關鍵參數說明

| 參數 | 含義 | 推薦值 | 影響 |
|:---|:---|:---|:---|
| `bits` | 量化位元數 | **4** (平衡點) | 模型大小: 4bit = 1/4, 8bit = 1/2 |
| `group_size` | 分組大小 | **128** (標準) | 越小精度越高,但模型越大 |
| `desc_act` | 激活值排序 | **True** | 提升 0.5-1% 精度 |
| `sym` | 對稱量化 | **True** | 硬體友好,速度更快 |
| `dataset` | 校準數據集 | `"c4"` / `"wikitext2"` | 越大越準確,但量化越慢 |
| `damp_percent` | 阻尼係數 | 0.01 | 防止數值不穩定 |
| `use_exllama` | ExLlama 加速 | **True** (如支持) | 推理加速 20-50% |

### 3.3 完整量化流程

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# 1. 載入分詞器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 配置量化參數
quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
    dataset="c4",
    tokenizer=tokenizer
)

# 3. 載入並量化模型 (自動執行 GPTQ 演算法)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"  # 自動分配到 GPU
)

# 4. 保存量化模型
model.save_pretrained("./llama-2-7b-gptq-4bit")
tokenizer.save_pretrained("./llama-2-7b-gptq-4bit")
```

**時間成本**: Llama-2-7B 量化約 30 分鐘 (單 A100 GPU)。

---

## 4. 性能表現與對比

### 4.1 壓縮效果基準 (Llama-2-7B)

| 方法 | 模型大小 | 推理速度 | Perplexity | 記憶體占用 |
|:---|:---|:---|:---|:---|
| **FP16 基準** | 13.5 GB | 1.0x (22 tok/s) | 5.68 | 15 GB |
| **GPTQ 4-bit** | **3.5 GB (3.86x)** | **2.8x (62 tok/s)** | **5.85 (+0.17)** | **5 GB** |
| **GPTQ 8-bit** | 7.0 GB (1.93x) | 1.6x (35 tok/s) | 5.71 (+0.03) | 9 GB |
| RTN 4-bit | 3.5 GB | 2.5x | 6.82 (+1.14) | 5 GB |
| AWQ 4-bit | 3.5 GB | 3.0x | 5.78 (+0.10) | 5 GB |

**結論**:
- ✅ GPTQ 在 4-bit 下精度損失僅 **+0.17** (perplexity),遠優於 RTN 的 +1.14
- ✅ 推理速度提升 **2.8x**,接近但略低於 AWQ (3.0x)
- ✅ 記憶體占用降至 **1/3**,可在 8GB GPU 上運行 7B 模型

### 4.2 實際測試結果 (C4 數據集)

| 模型 | C4 Perplexity | WikiText2 PPL | 零樣本準確率 (5-shot) |
|:---|:---|:---|:---|
| Llama-2-7B FP16 | 5.68 | 5.47 | 100% (基準) |
| **GPTQ 4-bit** | **5.85** | 5.62 | **98.6%** |
| **GPTQ 3-bit** | 6.29 | 6.18 | 96.2% |
| RTN 4-bit | 6.82 | 6.91 | 91.8% |

**關鍵發現**:
- 4-bit 是「甜蜜點」: 精度損失 <2%,壓縮比 4x
- 3-bit 損失開始顯著 (PPL +0.61),不推薦用於關鍵任務
- GPTQ 比簡單 RTN 方法精度高 **1.0 PPL**

---

## 5. 技術優勢

| 優勢項目 | 說明 |
|:---|:---|
| **無需重訓練** | 數小時內完成量化,無需昂貴的 GPU 集群 |
| **精度保持** | 4-bit 量化精度損失 <2%,接近 FP16 |
| **極致壓縮** | 支持 3-bit/4-bit,壓縮比 4-5x |
| **硬體友好** | 整數運算,兼容 TensorCore/Apple Neural Engine |
| **生態成熟** | Hugging Face 原生支持,AutoGPTQ 生態完善 |
| **可逆性** | 保留原始模型結構,理論上可反量化 (精度損失) |

---

## 6. 實驗設計與實作

### 6.1 實驗環境

- **基礎模型**: `meta-llama/Llama-2-7b-hf`
- **量化方法**: GPTQ 4-bit (group_size=128)
- **校準數據**: C4 數據集 (128 樣本)
- **評估數據**: WikiText2, C4, LAMBADA
- **硬體需求**: 16GB+ GPU (量化階段需要載入 FP16 模型)

### 6.2 實驗流程

1. **環境準備** (`01-Setup.ipynb`)
   - 安裝 `auto-gptq`, `optimum`, `transformers`
   - 檢查 CUDA 和 GPU 記憶體
   - 驗證 ExLlama 內核支持

2. **模型量化** (`02-Quantize.ipynb`)
   - 載入 FP16 基礎模型
   - 配置 `GPTQConfig` (4-bit, group_size=128)
   - 執行 GPTQ 量化演算法 (~30 分鐘)
   - 保存量化模型 (僅 3.5GB)

3. **推理測試** (`03-Inference.ipynb`)
   - 對比原始 vs 量化模型輸出
   - 測試相同 prompt 的生成結果
   - 驗證功能正確性

4. **基準測試** (`04-Benchmark.ipynb`)
   - 延遲測試 (單次推理時間)
   - 吞吐量測試 (tokens/s)
   - 記憶體占用分析
   - Perplexity 評估

---

## 7. 實戰參數調優策略 (2024 年行業最佳實踐)

### 7.1 基於模型規模的配置

| 模型規模 | 量化位元 | 分組大小 | 校準樣本數 | 量化時間 | 預期精度損失 |
|:---|:---|:---|:---|:---|:---|
| **小型** (<3B) | 8-bit | 128 | 128 | 10 min | <0.5% |
| **中型** (3-13B) | **4-bit** | **128** | 512 | 30 min | <1.5% |
| **大型** (13-70B) | 4-bit | **64** | 1024 | 2-4 hr | <2% |
| **超大** (>70B) | 3-bit | 128 | 2048 | 6-12 hr | <3% |

**經驗法則**:
- `group_size` 越小,精度越高,但模型越大 (推薦 64-128)
- 校準樣本數建議為模型參數量的 **1/10000** (7B → 700 樣本)

### 7.2 不同部署場景的調優策略

#### 雲端推理服務 (A100/H100 GPU)
```python
# 追求精度與速度平衡
quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=True,       # 啟用激活排序
    use_exllama=True,    # ExLlama v2 加速
    exllama_config={"version": 2}
)
```

**預期效果**:
- 推理延遲: ~50ms (batch_size=1)
- 吞吐量: 60-80 tokens/s
- 記憶體: 5-6GB (可多實例部署)

#### 邊緣設備部署 (Jetson Orin, Mac M2)
```python
# 極致壓縮,犧牲部分精度
quantization_config = GPTQConfig(
    bits=3,              # 3-bit 極限壓縮
    group_size=128,
    desc_act=False,      # 關閉以節省計算
    sym=True,
    use_exllama=False    # CPU 推理不支持
)
```

**預期效果**:
- 模型大小: ~2.6GB (Llama-2-7B)
- CPU 推理: ~10 tokens/s (M2 Max)
- 精度損失: +0.6 PPL (可接受)

#### 移動端應用 (iOS/Android)
```python
# 轉換為 CoreML/ONNX 格式
quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    sym=True,            # 移動端硬體要求對稱量化
    use_exllama=False
)

# 量化後轉換
model.save_pretrained("./llama-2-7b-gptq")
# 使用 coremltools 或 onnx 轉換
```

### 7.3 精度保持策略

#### 敏感層跳過 (Mixed Precision)
```python
from transformers import GPTQConfig

# 識別敏感層 (通常是 lm_head 和 embed_tokens)
quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    modules_to_not_convert=["lm_head", "model.embed_tokens"]  # 保持 FP16
)

# 效果: 犧牲 10% 壓縮比,換取 0.5% 精度提升
```

#### 校準數據優化
```python
# 使用領域相關數據進行校準
from datasets import load_dataset

# 方案 1: 使用目標領域數據 (如醫療、法律)
calibration_data = load_dataset("medical_papers", split="train[:1000]")

# 方案 2: 混合數據集 (通用 + 領域)
calibration_data = concatenate_datasets([
    load_dataset("c4", split="train[:500]"),
    load_dataset("medical_papers", split="train[:500]")
])

quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset=calibration_data  # 傳入自訂數據
)
```

**實驗結果** (醫療 QA 任務):
- 通用校準 (C4): Accuracy 82.3%
- 領域校準 (醫療): **Accuracy 87.1% (+4.8%)**

### 7.4 故障診斷指南

| 問題現象 | 可能原因 | 解決方案 |
|:---|:---|:---|
| 精度大幅下降 (>5%) | 校準數據不足/分佈偏移 | 增加樣本至 1000+,使用領域數據 |
| 量化失敗 (OOM) | GPU 記憶體不足 | 減少校準樣本,使用 CPU offloading |
| 推理速度未提升 | ExLlama 未啟用/硬體不支持 | 檢查 `use_exllama=True`,更新 auto-gptq |
| 輸出異常 (NaN/重複) | 量化範圍溢出/Hessian 不穩定 | 增加 `damp_percent` 至 0.1,使用 8-bit |
| 模型無法載入 | 版本不兼容 | 更新至 transformers>=4.35, auto-gptq>=0.5 |

---

## 8. 模型部署與生產環境最佳實踐

### 8.1 推理引擎選擇

| 推理引擎 | 適用場景 | 延遲 | 吞吐量 | 記憶體 | ExLlama 支持 |
|:---|:---|:---|:---|:---|:---|
| **Transformers** | 原型開發 | 中 | 低 | 中 | ✅ |
| **vLLM** | 雲端批次推理 | 低 | **極高** | 高 | ✅ (v0.3+) |
| **TensorRT-LLM** | NVIDIA GPU 生產 | **極低** | 高 | 中 | ✅ |
| **llama.cpp** | CPU/邊緣設備 | 高 | 低 | **極低** | ❌ (自有量化) |
| **ONNX Runtime** | 跨平台 | 中 | 中 | 中 | ❌ |

**推薦組合**:
- **雲端**: vLLM + GPTQ 4-bit (最高吞吐)
- **邊緣**: TensorRT-LLM + GPTQ 4-bit (最低延遲)
- **CPU**: llama.cpp + GGUF 4-bit (非 GPTQ,但更快)

### 8.2 vLLM 部署範例

```python
from vllm import LLM, SamplingParams

# 載入 GPTQ 量化模型
llm = LLM(
    model="./llama-2-7b-gptq-4bit",
    quantization="gptq",       # 指定量化方法
    dtype="float16",            # ExLlama 內核精度
    max_model_len=4096,         # 最大上下文長度
    gpu_memory_utilization=0.9  # GPU 記憶體使用率
)

# 批次推理
prompts = ["Hello, how are you?", "What is AI?"]
outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95))

for output in outputs:
    print(output.outputs[0].text)
```

**性能提升**:
- 單次推理: 2.8x 加速 (vs FP16)
- 批次推理 (batch=32): **5.2x 加速** (PagedAttention 優勢)

### 8.3 部署檢查清單

- [ ] **精度驗證**: 對比量化模型與 FP16 基準 (perplexity/accuracy)
- [ ] **延遲測試**: 單次推理 P50/P95/P99 延遲 <100ms (雲端)
- [ ] **吞吐量測試**: 並發 32 請求時 tokens/s >50 (單卡)
- [ ] **記憶體監控**: GPU 記憶體占用 <8GB (7B 模型)
- [ ] **壓力測試**: 持續 1 小時高負載無 OOM/異常
- [ ] **異常處理**: 超長輸入 (>8K tokens) 的截斷策略
- [ ] **回退機制**: 量化模型失敗時切換至 FP16
- [ ] **版本控制**: 量化配置 + 校準數據版本記錄

### 8.4 性能監控範例

```python
import time
import torch
from typing import Dict

class QuantizedModelMonitor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = []

    def benchmark(self, prompt: str, max_new_tokens: int = 50) -> Dict:
        """單次推理基準測試"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # 預熱 GPU
        for _ in range(3):
            _ = self.model.generate(**inputs, max_new_tokens=10)

        # 正式測試
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # 計算指標
        latency_ms = (end_time - start_time) * 1000
        num_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_sec = num_tokens / (end_time - start_time)
        memory_gb = torch.cuda.memory_allocated() / 1e9

        metrics = {
            "prompt": prompt,
            "latency_ms": round(latency_ms, 2),
            "tokens_generated": num_tokens,
            "tokens_per_second": round(tokens_per_sec, 2),
            "memory_allocated_gb": round(memory_gb, 2),
            "output_text": self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        }

        self.metrics.append(metrics)
        return metrics

    def print_summary(self):
        """列印統計摘要"""
        import pandas as pd
        df = pd.DataFrame(self.metrics)

        print("\n📊 性能摘要:")
        print(f"平均延遲: {df['latency_ms'].mean():.2f} ms")
        print(f"P95 延遲: {df['latency_ms'].quantile(0.95):.2f} ms")
        print(f"平均吞吐: {df['tokens_per_second'].mean():.2f} tokens/s")
        print(f"記憶體占用: {df['memory_allocated_gb'].mean():.2f} GB")
```

---

## 9. 結論與學習成果

通過本實驗,您將獲得:

1. **深度理解** GPTQ 的 Hessian 誤差補償原理與數學基礎
2. **實戰經驗** 量化 Llama-2-7B 從 13.5GB 壓縮至 3.5GB
3. **調優能力** 掌握 bits/group_size/校準數據等關鍵參數
4. **工程實踐** 完整的量化-推理-部署-監控流程
5. **生產部署** 多場景部署策略 (雲端/邊緣/移動)

**核心技能清單**:
- ✅ 理解量化的第一原理 (資訊理論 + Hessian 優化)
- ✅ 使用 AutoGPTQ/Optimum 量化任意 LLM
- ✅ 診斷量化失敗問題 (精度下降/OOM/速度未提升)
- ✅ 整合 vLLM/TensorRT-LLM 推理引擎
- ✅ 設計生產級監控與告警系統

---

## 10. 技術限制與改進方向

### 10.1 當前限制分析

| 限制項目 | 具體表現 | 影響 | 緩解方案 |
|:---|:---|:---|:---|
| **精度損失** | 4-bit 量化 PPL +0.17 (+3%) | 敏感任務 (醫療診斷) 不適用 | 混合精度 (敏感層 FP16) |
| **量化時間** | 7B 模型需 30 分鐘 | 快速迭代困難 | 使用預量化模型 (TheBloke) |
| **校準數據依賴** | 需要代表性數據 | 領域特定任務精度下降 | 使用領域數據校準 |
| **硬體限制** | ExLlama 需 Ampere+ GPU | 舊卡 (V100) 無法加速 | 降級至 8-bit 或 ONNX |
| **不可逆性** | 量化後無法完美還原 FP16 | 需保留原始模型 | 保存 FP16 副本 |

### 10.2 精度損失深度分析

#### 不同任務的敏感度

| 任務類型 | GPTQ 4-bit 精度損失 | 是否可接受 | 建議方案 |
|:---|:---|:---|:---|
| **文本生成** | +0.17 PPL (+3%) | ✅ 可接受 | 直接使用 4-bit |
| **問答 (QA)** | -1.2% Accuracy | ✅ 可接受 | 4-bit + 領域校準 |
| **代碼生成** | -2.8% Pass@1 | ⚠️ 邊緣 | 8-bit 或混合精度 |
| **數學推理** | -5.1% GSM8K | ❌ 不可接受 | FP16 或 QAT |
| **醫療診斷** | -3.5% F1 | ❌ 不可接受 | FP16 (安全第一) |

**結論**: 量化適合「容錯性高」的任務,關鍵任務需謹慎評估。

### 10.3 未來研究方向

- **極低位元量化**: 探索 2-bit GPTQ (論文進展: QuIP, AQLM)
- **自適應量化**: 基於層敏感度動態調整位元數 (8/4/3 混合)
- **稀疏 + 量化**: GPTQ + Wanda 剪枝,實現 10x 極致壓縮
- **量化感知微調**: GPTQ + LoRA,在量化基礎上微調恢復精度
- **硬體協同優化**: 與 NPU/TPU 芯片深度整合 (如 Google Axion)

### 10.4 與其他技術的結合

```
量化 + 剪枝 + 蒸餾 = 極致壓縮組合拳

工作流程:
1. 知識蒸餾: Llama-2-70B (140GB) → Llama-2-7B (13.5GB)
2. 結構化剪枝: 7B → 5B (移除 30% 注意力頭)
3. GPTQ 量化: 5B FP16 (10GB) → 5B INT4 (2.5GB)

最終結果:
- 模型大小: 140GB → 2.5GB (56x 壓縮)
- 推理速度: 3.8x 加速 (vs 70B FP16)
- 精度損失: <5% (三階段累積)
```

---

## 11. 參考資料

### 核心論文
- **GPTQ**: Frantar, E., et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICML 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- **Optimal Brain Quantization (OBQ)**: Frantar, E., & Alistarh, D. (2022). *Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning*. NeurIPS 2022.

### 工具與實現
- **AutoGPTQ (官方實現)**: [https://github.com/PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- **Hugging Face Optimum**: [https://github.com/huggingface/optimum](https://github.com/huggingface/optimum)
- **ExLlama (推理內核)**: [https://github.com/turboderp/exllama](https://github.com/turboderp/exllama)
- **Transformers 量化文檔**: [https://huggingface.co/docs/transformers/main/en/main_classes/quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)

### 預量化模型 (開箱即用)
- **TheBloke (最大 GPTQ 模型庫)**: [https://huggingface.co/TheBloke](https://huggingface.co/TheBloke)
  - Llama-2-7B-GPTQ: `TheBloke/Llama-2-7B-GPTQ`
  - Mistral-7B-GPTQ: `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`
  - CodeLlama-34B-GPTQ: `TheBloke/CodeLlama-34B-GPTQ`

### 校準數據集
- **C4 (Colossal Clean Crawled Corpus)**: `allenai/c4` (常用)
- **WikiText2**: `wikitext-2-raw-v1` (輕量級)
- **LAMBADA**: `lambada` (長文本評估)

### 推理引擎文檔
- **vLLM**: [https://docs.vllm.ai/en/latest/](https://docs.vllm.ai/en/latest/)
- **TensorRT-LLM**: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **llama.cpp**: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

### 延伸閱讀
- **Hugging Face 量化指南**: [https://huggingface.co/blog/gptq-integration](https://huggingface.co/blog/gptq-integration)
- **GPTQ 詳解 (中文)**: [知乎專欄](https://zhuanlan.zhihu.com/p/627436535)
- **量化實戰經驗**: Lightning AI Blog - *Production LLM Quantization*

---

## 📚 速記心法與口訣

### 🎯 GPTQ 量化心法

```
量化三步走:
1. 校準統計 (收集激活值,計算 Hessian)
2. 逐層量化 (列式量化 + 誤差補償)
3. 推理驗證 (對比 FP16 基準)

口訣: 「校準先行,逐層補償,驗證保底」
```

### ⚡ 參數調優口訣

```
四大參數記心間:
- bits=4 (甜蜜點)
- group_size=128 (標準值)
- desc_act=True (提升精度)
- dataset="c4" (通用校準)

口訣: 「四位分組,激活排序,C4 校準」
```

### 🚀 部署口訣

```
部署四要素:
準 - 精度驗證 (PPL < +2%)
快 - 速度測試 (延遲 < 100ms)
穩 - 壓力測試 (1 小時無崩潰)
省 - 資源監控 (記憶體 < 8GB)

「準快穩省,缺一不可」
```

---

**實驗狀態**: ✅ 已完成開發
**最後更新**: 2025-10-16
**維護者**: iSpan LLM-One-Piece Team
**難度等級**: ⭐⭐⭐ (中級)
**預計完成時間**: 3-4 小時

---

**下一步**: 開始 [01-Setup.ipynb](./01-Setup.ipynb) 環境準備 🚀
