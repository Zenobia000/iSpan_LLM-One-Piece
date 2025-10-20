# 📘 模型壓縮章節撰文風格規範

**適用範圍**: `03-Model_Compression` 目錄下所有文檔與實驗室
**參考來源**: `01-Core_Training_Techniques` 已驗證的撰文模式
**最後更新**: 2025-10-16

---

## 🎯 核心原則

### 1. 一致性優先
- 與 `01-Core_Training_Techniques` 保持高度一致的結構和風格
- 學習者應能無縫切換章節而不感到困惑

### 2. 理論與實踐並重
- 每個 Lab 都有對應的理論基礎
- 理論文檔獨立於實驗室,便於快速查閱

### 3. 工程導向
- 不僅講「是什麼」,更講「怎麼用」
- 包含生產環境最佳實踐和故障診斷

---

## 📂 目錄結構規範

```
03-Model_Compression/
├── 01-Theory/
│   ├── 3.1-Quantization.md              # 量化技術理論
│   ├── 3.2-Pruning.md                   # 剪枝技術理論
│   └── 3.3-Knowledge_Distillation.md    # 知識蒸餾理論
├── 02-Labs/
│   ├── Lab-3.1-Post_Training_Quantization_GPTQ/
│   │   ├── README.md                    # 實驗說明文檔
│   │   ├── 01-Setup.ipynb
│   │   ├── 02-Train.ipynb               # 或 02-Quantize.ipynb (視技術而定)
│   │   ├── 03-Inference.ipynb
│   │   └── 04-Deploy.ipynb              # 或 04-Benchmark.ipynb
│   ├── Lab-3.2-Pruning_with_Wanda/
│   │   ├── README.md
│   │   ├── 01-Setup.ipynb
│   │   ├── 02-Prune.ipynb
│   │   ├── 03-Inference.ipynb
│   │   └── 04-Benchmark_and_Analysis.ipynb
│   └── Lab-3.3-Knowledge_Distillation_MiniLM/
│       ├── README.md
│       ├── 01-Setup.ipynb
│       ├── 02-Train_Teacher.ipynb       # 或直接載入已訓練的 Teacher
│       ├── 03-Distill_Student.ipynb
│       ├── 04-Compare_and_Deploy.ipynb
└── WRITING_STYLE_GUIDE.md               # 本文檔
```

---

## 📄 README.md 撰文模板

### 結構大綱 (11個必要章節)

```markdown
# Lab 3.X: [技術名稱] - [簡短描述]

## 概述
**[核心技術]** 是... [技術背景 1-2段]
本實驗將... [實驗目標聲明]

![技術示意圖](url_or_local_path)

---

## 1. 技術背景與動機

### 1.1 為何需要模型壓縮?
- **部署限制**: 邊緣設備、移動端、嵌入式系統
- **成本考量**: 推理延遲、吞吐量、記憶體占用
- **環境約束**: 功耗、散熱、儲存空間

### 1.2 壓縮技術分類
[橫向比較: 量化 vs 剪枝 vs 蒸餾]

---

## 2. [技術名稱] 核心原理

### 2.1 理論基礎
[數學原理、演算法基礎]

### 2.2 技術實現
[具體實現方式、關鍵步驟]

### 2.3 變體與改進
[該技術的不同變體: PTQ vs QAT, GPTQ vs AWQ...]

---

## 3. 實現原理與步驟

### 3.1 關鍵配置
\```python
# 配置範例 (帶詳細註解)
config = QuantizationConfig(
    bits=4,  # 量化位元數: 4/8 bit
    group_size=128,  # 分組大小: 影響精度
    ...
)
\```

### 3.2 關鍵參數說明
| 參數名稱 | 含義 | 推薦值 | 影響 |
|:---|:---|:---|:---|
| `bits` | 量化位元數 | 4/8 | 模型大小 vs 精度 |

### 3.3 工作流程
1. 步驟1: [詳細說明]
2. 步驟2: [詳細說明]
...

---

## 4. 性能表現與對比

### 4.1 壓縮效果基準

| 方法 | 模型大小 | 推理速度 | 精度損失 | 記憶體占用 |
|:---|:---|:---|:---|:---|
| **原始模型** | 100% | 1x | 0% | 基準 |
| **本技術** | **25%** | **2-4x** | **<2%** | **30%** |
| 其他方法1 | 50% | 1.5x | 1% | 55% |

### 4.2 實際測試結果
[基於真實模型的測試數據]

---

## 5. 技術優勢

| 優勢項目 | 說明 |
|:---|:---|
| **部署友好** | 大幅降低模型大小,適合邊緣設備 |
| **推理加速** | 低位元計算加速推理 |
| **精度保持** | 先進技術精度損失 <2% |
| **易於整合** | 兼容主流推理引擎 |

---

## 6. 實驗設計與實作

### 6.1 實驗環境
- **模型**: `meta-llama/Llama-2-7b-hf` (或其他)
- **任務**: 文本生成 / 分類 / QA
- **數據集**: [具體數據集]
- **核心技術**: [技術名稱 + 關鍵庫]

### 6.2 實驗流程
1. **環境準備** (`01-Setup.ipynb`)
   - 安裝依賴
   - 驗證環境

2. **模型壓縮** (`02-[操作].ipynb`)
   - 載入基礎模型
   - 應用壓縮技術
   - 保存壓縮模型

3. **推理測試** (`03-Inference.ipynb`)
   - 對比原始模型 vs 壓縮模型
   - 性能評估

4. **基準測試** (`04-Benchmark.ipynb`)
   - 延遲測試
   - 吞吐量測試
   - 記憶體分析

---

## 7. 實戰參數調優策略 (2024 年行業最佳實踐)

### 7.1 基於模型規模的配置

| 模型規模 | 量化位元 | 分組大小 | 校準樣本數 | 預期精度損失 |
|:---|:---|:---|:---|:---|
| **小型** (<1B) | 8-bit | 128 | 128 | <0.5% |
| **中型** (1-10B) | 4-bit | 128 | 512 | <1.5% |
| **大型** (>10B) | 4-bit | 64 | 1024 | <2% |

### 7.2 不同部署場景的調優策略

#### 雲端推理服務
\```python
# 配置範例
config = {
    'quantization': '8-bit',  # 平衡精度與速度
    'batch_size': 32,         # 高吞吐
    'use_flash_attention': True
}
\```

#### 邊緣設備部署
\```python
config = {
    'quantization': '4-bit',  # 極致壓縮
    'batch_size': 1,          # 低延遲
    'optimize_for_latency': True
}
\```

#### 移動端應用
\```python
config = {
    'quantization': 'int8',
    'model_format': 'onnx',   # 跨平台
    'optimize_memory': True
}
\```

### 7.3 精度保持策略

\```python
# 敏感層檢測與保護
sensitive_layers = ['lm_head', 'embed_tokens']
quantization_config = QuantizationConfig(
    skip_modules=sensitive_layers,  # 跳過敏感層
    calibration_samples=1024,       # 增加校準樣本
)
\```

### 7.4 故障診斷指南

| 問題現象 | 可能原因 | 解決方案 |
|:---|:---|:---|
| 精度大幅下降 (>5%) | 校準樣本不足/分佈不均 | 增加樣本量、確保樣本多樣性 |
| 推理速度未提升 | 硬體不支持低位元運算 | 檢查硬體支持、使用專用推理引擎 |
| 量化失敗 | OOM / 版本不兼容 | 減少批次大小、更新庫版本 |
| 輸出異常 (NaN/Inf) | 量化範圍溢出 | 調整量化範圍、使用對稱量化 |

---

## 8. 模型部署與生產環境最佳實踐

### 8.1 推理引擎選擇

| 推理引擎 | 適用場景 | 優勢 | 限制 |
|:---|:---|:---|:---|
| **vLLM** | 雲端批次推理 | 高吞吐、PagedAttention | 需要較多記憶體 |
| **TensorRT-LLM** | NVIDIA GPU | 極致性能、低延遲 | NVIDIA 專屬 |
| **ONNX Runtime** | 跨平台部署 | 通用性強 | 性能次優 |
| **llama.cpp** | CPU/邊緣設備 | 低資源、跨平台 | 速度較慢 |

### 8.2 部署檢查清單

- [ ] 驗證量化模型精度 (與原始模型對比)
- [ ] 基準測試 (延遲、吞吐量、記憶體)
- [ ] 壓力測試 (並發請求、長文本)
- [ ] 異常處理 (OOM、超時、錯誤輸出)
- [ ] 監控指標 (P50/P95/P99 延遲)

### 8.3 性能監控範例

\```python
import time
import torch

class InferenceMonitor:
    def __init__(self):
        self.metrics = []

    def measure(self, model, inputs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(**inputs)

        torch.cuda.synchronize()
        latency = time.perf_counter() - start

        metrics = {
            'latency_ms': latency * 1000,
            'tokens_generated': len(outputs[0]),
            'tokens_per_second': len(outputs[0]) / latency,
            'memory_allocated_gb': torch.cuda.memory_allocated() / 1e9
        }

        self.metrics.append(metrics)
        return outputs, metrics
\```

---

## 9. 結論與學習成果

通過本實驗,您將獲得:

1. **深度理解** [技術名稱] 的原理與數學基礎
2. **實戰經驗** 在真實 LLM 上應用該技術
3. **調優能力** 掌握參數配置與故障診斷
4. **工程實踐** 完整的壓縮-推理-部署流程
5. **生產部署** 多場景部署策略與性能優化

---

## 10. 技術限制與改進方向

### 10.1 當前限制分析

| 限制項目 | 具體表現 | 影響 | 緩解方案 |
|:---|:---|:---|:---|
| **精度損失** | 4-bit 量化精度下降 1-3% | 敏感任務不適用 | 混合精度、敏感層跳過 |
| **硬體依賴** | 需要特定硬體加速 | 部署靈活性降低 | 使用通用推理引擎 |
| **校準成本** | PTQ 需要代表性數據 | 數據收集成本 | 使用公開數據集 |

### 10.2 未來研究方向

- **更低位元量化**: 探索 2-bit / 1-bit 量化
- **自適應量化**: 基於層敏感度動態調整位元數
- **量化感知訓練**: QAT 進一步降低精度損失
- **硬體協同設計**: 與推理芯片深度整合

### 10.3 與其他壓縮技術結合

\```
量化 + 剪枝 + 蒸餾 = 極致壓縮
- 剪枝移除冗餘參數 (30% 稀疏度)
- 量化降低位元精度 (4-bit)
- 蒸餾保持性能 (學生模型更小)
→ 最終壓縮比 10-20x,精度損失 <3%
\```

---

## 11. 參考資料

### 核心論文
- **[技術名稱原始論文]**: Author et al. (Year). *Title*. Conference/Journal.
- **相關改進論文**: [列出 2-3 篇關鍵論文]

### 工具與實現
- **官方實現**: [GitHub 連結]
- **Hugging Face 整合**: [文檔連結]
- **推理引擎文檔**: [相關連結]

### 模型與資料集
- **基礎模型**: [Hugging Face 模型卡]
- **校準數據**: [數據集連結]
- **基準測試**: [公開 benchmark 結果]

### 延伸閱讀
- **技術博客**: [優質博客文章]
- **實戰案例**: [工業界應用案例]
- **社群討論**: [論壇/Discord 連結]

---

## 📚 速記心法與口訣 (可選,視技術特性添加)

### 🎯 [技術名稱] 心法
\```
量化三步走:
1. 校準統計 (Calibration)
2. 範圍量化 (Quantization)
3. 推理驗證 (Validation)

口訣: 「校準先行,範圍適中,驗證保底」
\```

### ⚡ 部署口訣
\```
部署四要素:
準 - 精度驗證
快 - 速度測試
穩 - 壓力測試
省 - 資源監控

「準快穩省,缺一不可」
\```

---

**狀態**: ✅ 已完成 / ⏸️ 開發中 / 📝 規劃中
**最後更新**: YYYY-MM-DD
**維護者**: [維護者名稱]
```

---

## 📓 Jupyter Notebook 撰文規範

### Notebook 命名與結構

#### 01-Setup.ipynb
```markdown
# Lab 3.X: [技術名稱] - Environment Setup

**Goal:** Prepare the environment for [compression technique]

**You will learn to:**
- Verify GPU and CUDA compatibility
- Install quantization/pruning/distillation libraries
- Load baseline model for compression

---

## Step 1: Hardware Verification

[說明為何需要檢查硬體,不同硬體的影響]
```

#### 02-[操作].ipynb (視技術而定)
- **量化**: `02-Quantize.ipynb`
- **剪枝**: `02-Prune.ipynb`
- **蒸餾**: `02-Distill_Student.ipynb`

```markdown
# Lab 3.X: [技術名稱] - [操作階段]

**Goal:** Apply [technique] to compress the model

**Key concepts:**
- [核心概念1]
- [核心概念2]

---

## Step 1: Load Base Model

Before compression, we load the original model...
```

#### 03-Inference.ipynb
```markdown
# Lab 3.X: [技術名稱] - Inference and Comparison

**Goal:** Compare compressed model with baseline

**Metrics:**
- Latency (ms)
- Throughput (tokens/s)
- Memory usage (GB)
- Accuracy (perplexity/F1/...)

---

## Step 1: Load Both Models

We'll load both the original and compressed models...
```

#### 04-Deploy.ipynb / 04-Benchmark.ipynb
```markdown
# Lab 3.X: [技術名稱] - Production Deployment

**Goal:** Prepare compressed model for production

**Deployment targets:**
- vLLM server
- TensorRT-LLM
- ONNX Runtime
- Mobile devices (optional)
```

---

### Code Cell 規範

#### 1. 安裝依賴
```python
# 每個依賴都要有註解說明用途
%pip install -q auto-gptq optimum  # GPTQ 量化核心庫
%pip install -q transformers>=4.35.0  # 支持量化模型載入
```

#### 2. 環境檢查
```python
# 檢查 CUDA 和 GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU detected. Compression may be slow.")
```

#### 3. 配置定義
```python
from transformers import GPTQConfig

# GPTQ 量化配置
quantization_config = GPTQConfig(
    bits=4,                    # 量化位元數
    group_size=128,            # 分組大小 (影響精度)
    desc_act=True,             # 啟用描述性激活排序
    sym=True,                  # 對稱量化
    dataset="c4",              # 校準數據集
    tokenizer=tokenizer        # 分詞器
)

print("✅ Quantization config created")
```

#### 4. 性能對比
```python
# 對比表格
import pandas as pd

results = pd.DataFrame({
    'Model': ['Original', 'Quantized'],
    'Size (GB)': [13.5, 3.5],
    'Latency (ms)': [156, 58],
    'Throughput (tok/s)': [22.3, 61.7],
    'Perplexity': [5.68, 5.82]
})

print(results.to_markdown(index=False))
```

---

## 🎨 視覺元素規範

### Emoji 使用指南

```markdown
## 📊 性能分析          # 數據、圖表相關
## 🔧 配置與調優        # 技術配置
## ⚡ 優化策略          # 性能優化
## 🚀 部署指南          # 生產部署
## 💡 實戰技巧          # 經驗分享
## ⚠️  限制與注意事項   # 警告聲明
## 📚 參考資源          # 文檔、論文
## 🎯 學習目標          # 目標聲明
## ✅ 成功/完成         # 驗證通過
## ❌ 錯誤/禁止         # 錯誤示例
## 🏗️ 架構設計         # 系統架構
## 🔬 實驗設計          # 實驗相關
```

### 表格視覺化

#### 對比表格 (加粗重點)
```markdown
| 方法 | 壓縮比 | 速度提升 | 精度損失 |
|:---|:---|:---|:---|
| **GPTQ** | **4x** | **2.8x** | **1.2%** |
| AWQ | 4x | 2.5x | 0.9% |
| SmoothQuant | 2x | 1.6x | 0.5% |
```

#### 配置表格 (清晰分類)
```markdown
| 參數 | 含義 | 推薦值 | 影響 |
|:---|:---|:---|:---|
| `bits` | 量化位元數 | 4/8 | 模型大小 vs 精度 |
| `group_size` | 分組大小 | 128 | 精度 vs 速度 |
```

---

## ✍️ 文字風格規範

### 語言特徵

1. **專業但易懂**
   - ✅ "量化通過降低權重精度來壓縮模型"
   - ❌ "量化就是把浮點數變成整數"

2. **主動語態**
   - ✅ "本實驗將展示..."
   - ❌ "將會被展示..."

3. **具體數據**
   - ✅ "GPTQ 可將 7B 模型從 13.5GB 壓縮至 3.5GB (4x)"
   - ❌ "GPTQ 可以大幅壓縮模型"

4. **避免過度承諾**
   - ✅ "在大多數任務上,精度損失 <2%"
   - ❌ "完全無損壓縮"

### 術語一致性

#### 模型壓縮專用術語
```
量化 (Quantization)
  - 訓練後量化 (Post-Training Quantization, PTQ)
  - 量化感知訓練 (Quantization-Aware Training, QAT)
  - GPTQ, AWQ, SmoothQuant (保持英文縮寫)

剪枝 (Pruning)
  - 結構化剪枝 (Structured Pruning)
  - 非結構化剪枝 (Unstructured Pruning)
  - 稀疏度 (Sparsity)
  - Wanda, SparseGPT, Magnitude Pruning

知識蒸餾 (Knowledge Distillation)
  - 教師模型 (Teacher Model)
  - 學生模型 (Student Model)
  - 軟標籤 (Soft Labels)
  - 溫度 (Temperature)
```

---

## 🔍 品質檢查清單

### README.md 檢查
- [ ] 包含所有 11 個核心章節
- [ ] 有清晰的技術背景說明
- [ ] 提供代碼範例 (帶詳細註解)
- [ ] 包含性能對比表格
- [ ] 列出實戰調優策略
- [ ] 有故障診斷指南
- [ ] 包含完整參考資料
- [ ] 使用一致的 emoji 標記
- [ ] 專業術語翻譯一致

### Notebook 檢查
- [ ] 每個 code cell 前有 markdown 說明
- [ ] 使用 `print("✅ ...")` 驗證輸出
- [ ] 包含環境檢查步驟
- [ ] 有性能對比與可視化
- [ ] 註解清晰 (英文)
- [ ] 變數命名規範
- [ ] 可重現執行 (固定隨機種子)

### 整體一致性
- [ ] README 與 notebook 內容對應
- [ ] 與理論文檔保持一致
- [ ] 與其他 Lab 風格統一
- [ ] 圖片和表格正確顯示
- [ ] 超連結有效

---

## 📖 範例參考

### 完整 README 範例
參考: `01-Core_Training_Techniques/02-Labs/PEFT_Labs/Lab-01-LoRA/README.md`

### 完整 Notebook 範例
參考: `01-Core_Training_Techniques/02-Labs/PEFT_Labs/Lab-01-LoRA/01-Setup.ipynb`

---

## 🛠️ 工具與資源

### Markdown 編輯器
- **VS Code**: Markdown Preview Enhanced 插件
- **Typora**: 所見即所得編輯器
- **在線工具**: StackEdit, Dillinger

### 表格生成器
- [Tables Generator](https://www.tablesgenerator.com/markdown_tables)
- [Markdown Tables](https://tabletomarkdown.com/)

### Emoji 參考
- [Emojipedia](https://emojipedia.org/)
- [GitHub Emoji Cheat Sheet](https://github.com/ikatyang/emoji-cheat-sheet)

---

**維護者**: Claude Code
**最後更新**: 2025-10-16
**版本**: 1.0

---

**使用建議**:
1. 開發新 Lab 前,先閱讀本規範
2. 參考已有 Lab 的結構和風格
3. 保持與 `01-Core_Training_Techniques` 一致
4. 定期更新最佳實踐

**問題反饋**:
如發現規範不清楚或需要補充,請在專案中提出 issue。
