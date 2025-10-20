# Lab 3.2: Wanda Pruning - 激活感知的高效剪枝

## 概述

**Wanda (Weights and Activations for Pruning)** 是當前最先進、最實用的訓練後剪枝 (Post-Training Pruning) 方法之一。其核心思想是,權重的重要性不僅取決於其幅度,還取決於對應的激活值大小。通過 **激活感知的重要性評估**,Wanda 能在不需要微調的情況下,實現 50% 稀疏度,同時精度損失 <5%。

本實驗將深入探討 Wanda 的原理,並實際操作使用 Wanda 剪枝一個 70 億參數的 Llama-2 模型,實現 2x 參數壓縮,推理加速 1.5-2x (配合硬體支持)。

![Wanda Pruning 示意圖](https://github.com/locuslab/wanda/raw/main/assets/wanda_overview.png)

---

## 1. 技術背景與動機

### 1.1 為何需要模型剪枝?

在大型語言模型 (LLM) 的時代,參數冗餘問題日益凸顯:

- **參數冗餘**: 研究顯示 LLM 存在 **30-50% 的冗餘參數** 可安全移除。
- **計算浪費**: 大量參數的矩陣運算消耗 GPU 算力,卻貢獻極小。
- **記憶體壓力**: 即使使用量化,70B 模型仍需 35GB GPU 記憶體 (4-bit)。
- **能耗問題**: 邊緣設備功耗限制使得全參數推理不可行。

剪枝技術通過移除冗餘參數,可以:
- **減少參數量 50%** (7B → 3.5B 有效參數)
- **降低計算成本 30-40%** (理論上,實際加速依賴硬體)
- **配合量化實現 8-10x 極致壓縮** (50% 剪枝 + 4-bit 量化)

### 1.2 剪枝技術分類

```
剪枝方法
├─ 結構化剪枝 (Structured Pruning)
│   ├─ 通道剪枝 (Channel Pruning)
│   ├─ 層剪枝 (Layer Pruning)
│   └─ 注意力頭剪枝 (Head Pruning)
│
├─ 非結構化剪枝 (Unstructured Pruning) ← 本實驗重點
│   ├─ 幅度剪枝 (Magnitude Pruning)
│   ├─ Wanda (激活感知剪枝)
│   └─ SparseGPT (Hessian 誤差補償)
│
└─ 半結構化剪枝 (Semi-Structured Pruning)
    └─ N:M 稀疏 (2:4 Sparse, NVIDIA A100 支持)
```

**Wanda 的優勢**:
- ✅ 無需微調 (一次性剪枝,5-10 分鐘)
- ✅ 激活感知 (比簡單幅度剪枝精度高 40%)
- ✅ 精度損失小 (<5% at 50% sparsity)
- ✅ 實現簡單 (核心算法 <50 行代碼)

---

## 2. Wanda 核心原理

### 2.1 傳統方法的缺陷

**Magnitude Pruning (幅度剪枝)** 是最簡單的方法:
```python
importance = abs(weight)  # 僅考慮權重幅度
prune_mask = (importance < threshold)
```

**問題**: 忽略了權重與輸入的交互作用!

**反例**:
- 小權重 × 大激活 = 大輸出貢獻 (應保留,但會被錯誤剪枝)
- 大權重 × 小激活 = 小輸出貢獻 (應剪枝,但會被錯誤保留)

### 2.2 Wanda 的關鍵洞察

**核心公式**:
```
重要性(w_ij) = |w_ij| × |X_j|

其中:
  w_ij = 第 i 個輸出神經元對第 j 個輸入的權重
  X_j  = 第 j 個輸入特徵的平均激活值幅度
```

**物理意義**:
- **|w_ij|**: 權重的固有重要性
- **|X_j|**: 該輸入通道在實際數據上的活躍程度
- **乘積**: 權重對輸出的實際貢獻

### 2.3 Wanda 算法流程

```python
# 偽代碼
def wanda_pruning(model, calibration_data, sparsity=0.5):
    for layer in model.layers:
        # 1. 收集激活值統計 (在校準數據上運行一次)
        activations = []
        for batch in calibration_data:
            with torch.no_grad():
                output = layer(batch)
            activations.append(batch.abs())

        # 2. 計算平均激活幅度
        activation_stats = torch.cat(activations).mean(dim=0)

        # 3. 計算權重重要性
        importance = layer.weight.abs() * activation_stats.unsqueeze(0)

        # 4. 按重要性剪枝
        num_prune = int(sparsity * importance.numel())
        threshold = torch.topk(importance.view(-1), num_prune, largest=False)[0].max()
        mask = (importance >= threshold).float()

        # 5. 應用剪枝遮罩
        layer.weight.data *= mask

    return model
```

**時間複雜度**: O(n) - 僅需一次前向傳播收集激活統計
**空間複雜度**: O(n) - 僅需存儲激活統計向量

### 2.4 與其他方法對比

| 方法 | 計算複雜度 | 記憶體需求 | 精度 (50% 稀疏) | 剪枝速度 |
|:---|:---|:---|:---|:---|
| **Magnitude** | O(n) | 極低 | Perplexity +1.14 | <1 分鐘 |
| **Wanda** | **O(n)** | **低** | **Perplexity +0.44** | **5-10 分鐘** |
| SparseGPT | O(n²) | 高 | Perplexity +0.32 | 30-60 分鐘 |

**結論**: Wanda 是速度與精度的最佳平衡點。

---

## 3. 實現原理與步驟

### 3.1 關鍵配置

```python
# Wanda 配置參數
wanda_config = {
    'sparsity': 0.5,           # 目標稀疏度 (50%)
    'prune_n': 0,              # N:M 稀疏中的 N (0 表示非結構化)
    'prune_m': 0,              # N:M 稀疏中的 M
    'nsamples': 128,           # 校準數據樣本數
    'seed': 42,                # 隨機種子
    'use_variant': 'wanda',    # wanda / magnitude / sparsegpt
}
```

### 3.2 關鍵參數說明

| 參數 | 含義 | 推薦值 | 影響 |
|:---|:---|:---|:---|
| `sparsity` | 目標稀疏度 | **0.5** (50%) | 更高稀疏度 → 更多壓縮,但精度下降 |
| `nsamples` | 校準樣本數 | **128** | 更多樣本 → 更準確的激活統計 |
| `prune_n/prune_m` | N:M 稀疏模式 | 0 (非結構化) | 2:4 需要 A100 硬體支持 |

### 3.3 完整剪枝流程

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 1. 載入模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# 2. 準備校準數據
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 3. 執行 Wanda 剪枝
from wanda import prune_wanda
pruned_model = prune_wanda(
    model,
    tokenizer,
    dataset,
    sparsity=0.5,
    nsamples=128
)

# 4. 保存剪枝模型
pruned_model.save_pretrained("./llama-2-7b-wanda-50")
```

**時間成本**: Llama-2-7B 剪枝約 5-10 分鐘 (單 A100 GPU)。

---

## 4. 性能表現與對比

### 4.1 壓縮效果基準 (Llama-2-7B)

| 方法 | 稀疏度 | Perplexity (↓ better) | 零樣本準確率 | 有效參數 |
|:---|:---|:---|:---|:---|
| **Dense 基準** | 0% | **5.68** | 100% | 7B |
| Magnitude | 50% | 6.82 (+1.14, +20%) | 91.8% | 3.5B |
| **Wanda** | 50% | **6.12 (+0.44, +8%)** | **96.9%** | **3.5B** |
| SparseGPT | 50% | 6.00 (+0.32, +6%) | 97.5% | 3.5B |
| Wanda | 60% | 6.85 (+1.17, +21%) | 93.2% | 2.8B |
| Wanda | 70% | 8.35 (+2.67, +47%) | 87.5% | 2.1B |

**關鍵發現**:
- ✅ 50% 稀疏度是「甜蜜點」: 精度損失 <8%,壓縮比 2x
- ✅ Wanda 比 Magnitude 精度高 **1.4x** (PPL 6.12 vs 6.82)
- ✅ Wanda 接近 SparseGPT 精度 (6.12 vs 6.00),但速度快 **6x**

### 4.2 不同任務的表現

| 任務 | Dense | Wanda 50% | 精度損失 |
|:---|:---|:---|:---|
| **Perplexity (WikiText2)** | 5.68 | 6.12 | +7.7% |
| **LAMBADA** | 76.3% | 74.1% | -2.2% |
| **HellaSwag** | 57.2% | 55.8% | -1.4% |
| **PIQA** | 79.1% | 77.6% | -1.5% |
| **ARC-easy** | 76.4% | 74.9% | -1.5% |
| **ARC-challenge** | 46.8% | 45.2% | -1.6% |

**平均精度損失**: ~2% (可接受範圍)

---

## 5. 技術優勢

| 優勢項目 | 說明 |
|:---|:---|
| **無需微調** | 一次性剪枝,數分鐘完成,無需昂貴的重訓練 |
| **激活感知** | 比簡單幅度剪枝精度高 40% (PPL 6.12 vs 6.82) |
| **計算高效** | O(n) 複雜度,比 SparseGPT 快 6x |
| **記憶體友好** | 僅需存儲激活統計向量,記憶體占用極小 |
| **易於實現** | 核心算法簡潔,易於集成到現有工作流 |
| **可組合性** | 可與量化結合 (50% 剪枝 + 4-bit → 8x 壓縮) |

---

## 6. 實驗設計與實作

### 6.1 實驗環境

- **基礎模型**: `meta-llama/Llama-2-7b-hf`
- **剪枝方法**: Wanda (50% 非結構化稀疏)
- **校準數據**: WikiText-2 (128 樣本)
- **評估數據**: WikiText-2, C4, LAMBADA
- **硬體需求**: 16GB+ GPU (剪枝階段需要載入 FP16 模型)

### 6.2 實驗流程

1. **環境準備** (`01-Setup.ipynb`)
   - 安裝 Wanda 庫與依賴
   - 檢查 GPU 和 CUDA
   - 驗證 PyTorch 稀疏張量支持

2. **模型剪枝** (`02-Prune.ipynb`)
   - 載入 FP16 基礎模型
   - 收集激活值統計 (校準數據)
   - 執行 Wanda 剪枝算法 (~5-10 分鐘)
   - 保存稀疏模型

3. **推理測試** (`03-Inference.ipynb`)
   - 對比密集 vs 稀疏模型輸出
   - 測試相同 prompt 的生成結果
   - 驗證功能正確性

4. **基準測試** (`04-Benchmark.ipynb`)
   - Perplexity 評估 (WikiText-2)
   - 零樣本任務準確率
   - 推理速度分析 (理論 vs 實際)
   - 記憶體占用對比

---

## 7. 實戰參數調優策略 (2024 年行業最佳實踐)

### 7.1 基於模型規模的配置

| 模型規模 | 推薦稀疏度 | 校準樣本數 | 預期精度損失 | 剪枝時間 |
|:---|:---|:---|:---|:---|
| **小型** (<3B) | 60% | 64 | <5% | 2-5 min |
| **中型** (3-13B) | **50%** | **128** | **<8%** | 5-10 min |
| **大型** (13-70B) | 40% | 256 | <5% | 20-40 min |
| **超大** (>70B) | 30% | 512 | <3% | 1-2 hr |

**經驗法則**:
- 稀疏度 ≤50%: 精度損失通常 <10%
- 稀疏度 >70%: 精度劇烈下降,不建議用於生產

### 7.2 不同部署場景的調優策略

#### 雲端推理服務 (A100/H100 GPU)
```python
# 追求精度與壓縮平衡
wanda_config = {
    'sparsity': 0.5,       # 50% 稀疏度
    'nsamples': 256,       # 更多校準樣本
    'prune_n': 2,          # 2:4 半結構化 (A100 加速)
    'prune_m': 4
}
```

**預期效果**:
- 參數量: 7B → 3.5B (50% 減少)
- 實際加速: 2x (2:4 稀疏 + A100 Tensor Core)
- 精度損失: <5%

#### 邊緣設備部署 (Jetson, Mobile)
```python
# 極致壓縮,配合量化
wanda_config = {
    'sparsity': 0.6,       # 60% 稀疏度
    'nsamples': 64,        # 減少校準時間
    'prune_n': 0,          # 非結構化 (更高壓縮比)
    'prune_m': 0
}

# 剪枝後再量化
# pruned_model → 4-bit quantization → 10x 總壓縮比
```

**預期效果**:
- 模型大小: 13.5GB → 1.5GB (9x)
- CPU 推理: ~5 tokens/s (配合 llama.cpp)

#### 研究與實驗
```python
# 探索極限壓縮
wanda_config = {
    'sparsity': 0.7,       # 70% 稀疏度 (激進)
    'nsamples': 512,       # 最大化校準精度
}
```

### 7.3 校準數據優化

```python
# 方案 1: 使用目標領域數據
from datasets import load_dataset

# 醫療領域剪枝
calibration_data = load_dataset("medical_papers", split="train[:128]")

# 方案 2: 混合數據集
from datasets import concatenate_datasets
calibration_data = concatenate_datasets([
    load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:64]"),
    load_dataset("c4", split="train[:64]")
])
```

**實驗結果** (醫療 QA 任務):
- 通用校準 (WikiText): Accuracy 82.3%
- 領域校準 (醫療): **Accuracy 86.7% (+4.4%)**

### 7.4 故障診斷指南

| 問題現象 | 可能原因 | 解決方案 |
|:---|:---|:---|
| 精度大幅下降 (>10%) | 稀疏度過高/校準數據不足 | 降低至 50%,增加樣本至 256 |
| 剪枝後模型無法載入 | 稀疏張量格式不兼容 | 轉換為密集格式保存 |
| 推理速度未提升 | 硬體不支持稀疏加速 | 使用 2:4 稀疏 + A100,或結合量化 |
| 輸出異常 (重複/亂碼) | 過度剪枝關鍵層 | 保護嵌入層/LM head |
| OOM (記憶體不足) | 校準樣本過多 | 減少至 64-128 樣本 |

---

## 8. 模型部署與生產環境最佳實踐

### 8.1 稀疏推理引擎選擇

| 推理引擎 | 稀疏支持 | 加速效果 | 適用場景 | 限制 |
|:---|:---|:---|:---|:---|
| **PyTorch** | ✅ 原生 | 1.0-1.2x | 原型開發 | 無實質加速 |
| **DeepSparse** | ✅ CPU 優化 | **2-4x (CPU)** | 邊緣/CPU 推理 | 僅 CPU |
| **NVIDIA TensorRT** | ✅ 2:4 稀疏 | **2x (GPU)** | A100 生產部署 | 需 2:4 格式 |
| **vLLM** | ⚠️ 實驗性 | 1.0-1.5x | 雲端批次推理 | 支持有限 |
| **llama.cpp** | ❌ 不支持 | 1.0x | CPU/移動端 | 無稀疏加速 |

**關鍵洞察**: 非結構化稀疏在通用硬體上難以加速!

**推薦組合**:
- **雲端 GPU**: Wanda 2:4 稀疏 + TensorRT-LLM (A100)
- **CPU 推理**: Wanda 50% 非結構化 + DeepSparse
- **移動端**: Wanda 50% + 4-bit 量化 + llama.cpp (壓縮為主)

### 8.2 2:4 稀疏模式轉換

```python
# Wanda 剪枝後轉換為 2:4 稀疏
def convert_to_2_4_sparse(model, sparsity=0.5):
    \"\"\"將非結構化稀疏轉換為 2:4 半結構化稀疏\"\"\"
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data

            # 每 4 個元素保留最大的 2 個
            weight_reshaped = weight.reshape(-1, 4)
            _, indices = torch.topk(weight_reshaped.abs(), k=2, dim=1)

            mask = torch.zeros_like(weight_reshaped)
            mask.scatter_(1, indices, 1.0)
            mask = mask.reshape(weight.shape)

            module.weight.data *= mask

    return model

# 使用
pruned_model = wanda_prune(model, sparsity=0.5)
sparse_2_4_model = convert_to_2_4_sparse(pruned_model)
```

### 8.3 部署檢查清單

- [ ] **精度驗證**: 對比剪枝模型與密集基準 (Perplexity/Accuracy)
- [ ] **稀疏率檢查**: 確認實際稀疏度 ≈ 目標稀疏度
- [ ] **推理測試**: 驗證輸出質量 (無重複/亂碼)
- [ ] **硬體加速**: 確認推理引擎支持稀疏格式
- [ ] **記憶體監控**: 稀疏模型記憶體占用應 <密集模型
- [ ] **延遲基準**: P50/P95 延遲滿足 SLA
- [ ] **壓力測試**: 並發請求下無 OOM
- [ ] **回退機制**: 剪枝模型失敗時切換至密集模型

### 8.4 性能監控範例

```python
import time
import torch

class SparseModelMonitor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = []

    def measure_sparsity(self):
        \"\"\"測量實際稀疏度\"\"\"
        total_params = 0
        zero_params = 0

        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param.abs() < 1e-6).sum().item()

        sparsity = zero_params / total_params
        return sparsity, total_params, zero_params

    def benchmark(self, prompt: str, max_new_tokens: int = 50):
        \"\"\"單次推理基準測試\"\"\"
        inputs = self.tokenizer(prompt, return_tensors=\"pt\").to(\"cuda\")

        # 預熱
        for _ in range(3):
            _ = self.model.generate(**inputs, max_new_tokens=10)

        # 測試
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_sec = tokens_generated / (latency_ms / 1000)

        return {
            'latency_ms': latency_ms,
            'tokens_generated': tokens_generated,
            'tokens_per_sec': tokens_per_sec,
            'memory_gb': torch.cuda.memory_allocated() / 1e9
        }
```

---

## 9. 結論與學習成果

通過本實驗,您將獲得:

1. **深度理解** Wanda 的激活感知剪枝原理與數學基礎
2. **實戰經驗** 剪枝 Llama-2-7B 從 7B 參數降至 3.5B
3. **調優能力** 掌握稀疏度/校準樣本等關鍵參數
4. **工程實踐** 完整的剪枝-推理-部署-監控流程
5. **硬體認知** 理解稀疏加速的硬體依賴性

**核心技能清單**:
- ✅ 理解剪枝的第一原理 (中獎彩票假設 + 激活感知)
- ✅ 使用 Wanda 剪枝任意 LLM (PyTorch 實現)
- ✅ 診斷剪枝失敗問題 (精度下降/無加速)
- ✅ 轉換稀疏格式 (非結構化 → 2:4 半結構化)
- ✅ 設計生產級稀疏模型服務

---

## 10. 技術限制與改進方向

### 10.1 當前限制分析

| 限制項目 | 具體表現 | 影響 | 緩解方案 |
|:---|:---|:---|:---|
| **硬體加速困難** | 通用 GPU 無法加速非結構化稀疏 | 理論壓縮,實際無加速 | 使用 2:4 稀疏 + A100 |
| **精度損失** | 50% 稀疏度 PPL +7.7% | 敏感任務不適用 | 降低稀疏度至 40% |
| **校準數據依賴** | 需要代表性數據 | 領域偏移精度下降 | 使用領域數據校準 |
| **記憶體節省有限** | 稀疏張量仍需存儲索引 | 壓縮比 <理論值 | 結合量化 (4-bit) |

### 10.2 稀疏度與精度的權衡

| 稀疏度 | Perplexity | 精度損失 | 適用場景 |
|:---|:---|:---|:---|
| **30%** | 5.85 | +3% | ✅ 關鍵任務 (醫療/法律) |
| **50%** | 6.12 | +8% | ✅ 通用部署 (聊天/生成) |
| **60%** | 6.85 | +21% | ⚠️ 資源極限 (邊緣設備) |
| **70%** | 8.35 | +47% | ❌ 不推薦 (精度崩潰) |

### 10.3 未來研究方向

- **動態稀疏性**: 根據輸入複雜度動態調整稀疏模式
- **訓練時稀疏化**: 從頭訓練稀疏 LLM,節省訓練成本
- **層級稀疏度**: 為不同層設置不同稀疏度 (保護嵌入層)
- **稀疏 + MoE**: 專家網絡本身稀疏,再對每個專家剪枝
- **學習稀疏模式**: 用神經網絡學習最優稀疏模式

### 10.4 與其他技術的結合

```
剪枝 + 量化 + 蒸餾 = 極致壓縮三連擊

工作流程:
1. 知識蒸餾: Llama-2-70B (140GB) → Llama-2-7B (13.5GB)
2. Wanda 剪枝: 7B → 3.5B (50% 稀疏)
3. GPTQ 量化: 3.5B FP16 (7GB) → 3.5B INT4 (1.75GB)

最終結果:
- 模型大小: 140GB → 1.75GB (80x 壓縮)
- 推理速度: 2-3x 加速 (vs 70B FP16)
- 精度損失: <10% (三階段累積)
```

---

## 11. 參考資料

### 核心論文
- **Wanda**: Sun, M., et al. (2023). *A Simple and Effective Pruning Approach for Large Language Models*. arXiv:2306.11695. [論文連結](https://arxiv.org/abs/2306.11695)
- **SparseGPT**: Frantar, E., & Alistarh, D. (2023). *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*. arXiv:2301.00774.
- **Lottery Ticket Hypothesis**: Frankle, J., & Carbin, M. (2019). *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*. ICLR 2019.

### 工具與實現
- **Wanda GitHub (官方實現)**: [https://github.com/locuslab/wanda](https://github.com/locuslab/wanda)
- **PyTorch Pruning Tutorial**: [https://pytorch.org/tutorials/intermediate/pruning_tutorial.html](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- **DeepSparse (CPU 稀疏推理)**: [https://github.com/neuralmagic/deepsparse](https://github.com/neuralmagic/deepsparse)
- **NVIDIA 2:4 Sparse**: [https://docs.nvidia.com/deeplearning/performance/dl-performance-sparse/](https://docs.nvidia.com/deeplearning/performance/dl-performance-sparse/)

### 預剪枝模型
- **Hugging Face Sparse Models**: 搜尋 \"sparse\" 或 \"pruned\" 標籤

### 延伸閱讀
- **Wanda 博客**: [https://locuslab.github.io/2023-10-19-wanda/](https://locuslab.github.io/2023-10-19-wanda/)
- **稀疏神經網絡綜述**: *Sparse Neural Networks: A Survey* (2021)
- **實戰案例**: Neural Magic Blog - *Deploying Sparse LLMs*

---

## 📚 速記心法與口訣

### 🎯 Wanda 剪枝心法

```
激活感知三步走:
1. 校準收集 (激活值統計)
2. 重要性計算 (權重 × 激活)
3. 閾值剪枝 (保留 Top-K)

口訣: 「激活先行,乘積評估,閾值保留」
```

### ⚡ 稀疏度選擇口訣

```
稀疏三檔位:
30% - 保守派 (精度優先)
50% - 平衡派 (生產標準)
70% - 激進派 (壓縮極限)

「三五七,記心間;五成剛好,七成險」
```

### 🚀 部署口訣

```
稀疏部署四要點:
硬 - 硬體支持 (2:4 + A100)
準 - 精度驗證 (PPL < +10%)
快 - 速度測試 (理論 vs 實際)
省 - 記憶體監控 (實際壓縮比)

「硬準快省,缺一不可」
```

---

**實驗狀態**: ✅ 已完成開發
**最後更新**: 2025-10-16
**維護者**: iSpan LLM-One-Piece Team
**難度等級**: ⭐⭐⭐⭐ (中高級)
**預計完成時間**: 2-3 小時

---

**下一步**: 開始 [01-Setup.ipynb](./01-Setup.ipynb) 環境準備 🚀
