# Lab 4.1: OpenCompass 模型評估 - 全面的能力基準測試

## 概述

**OpenCompass** 是由上海人工智能實驗室開發的開源大模型評測平台，為大型語言模型提供全面、客觀、開放的評估體系。它支持超過 100 個數據集、80+ 模型的評估，涵蓋語言理解、推理、代碼、知識等多個維度。

本實驗將帶你掌握使用 OpenCompass 評估 LLM 的完整流程，從環境配置、執行評估、結果分析到自動化報告生成。你將學會如何客觀地評估模型能力，識別模型優劣勢，並為模型選型與優化提供數據支持。

![OpenCompass Architecture](https://github.com/open-compass/opencompass/raw/main/docs/zh_cn/_static/image/compass_overview.png)

---

## 1. 技術背景與動機

### 1.1 為何需要模型評估？

在 LLM 工程化過程中，評估是不可或缺的環節：

- **模型選型**: 在眾多開源模型中選擇最適合業務場景的模型
- **訓練驗證**: 驗證微調、壓縮、對齊等技術是否有效
- **能力診斷**: 識別模型在哪些領域表現優秀，哪些領域需要改進
- **性能追蹤**: 監控模型迭代過程中的能力變化
- **對外溝通**: 向利益相關者展示模型能力的客觀證據

### 1.2 評估的挑戰

**主觀性問題**:
- 開放式生成任務難以量化（創意寫作、對話質量）
- 不同評估者標準不一致

**覆蓋度問題**:
- 單一基準無法全面評估模型能力
- 需要多維度、多領域的評估體系

**可重現性問題**:
- 評估協議不統一（Few-shot 數量、溫度設定）
- 隨機性導致結果波動

**成本問題**:
- 大規模評估耗時長、算力成本高
- 人工評估成本更高

### 1.3 OpenCompass 的解決方案

OpenCompass 通過以下方式解決評估挑戰：

| 特性 | 說明 |
|:---|:---|
| **全面性** | 支持 100+ 數據集，涵蓋語言理解、推理、知識、代碼等 |
| **客觀性** | 標準化評估協議，自動化評分系統 |
| **開放性** | 開源框架，支持自定義數據集與評估指標 |
| **高效性** | 分佈式評估，支持多 GPU 並行 |
| **可重現** | 固定評估配置，確保結果一致性 |

---

## 2. OpenCompass 核心原理

### 2.1 評估範式

OpenCompass 支持多種評估範式：

#### 判別式評估 (Discriminative Evaluation)
- **適用任務**: 多選題、分類任務
- **評估方式**: 比較各選項的對數似然，選擇最高的
- **優勢**: 準確、高效、無需生成
- **示例**: C-Eval, MMLU, HellaSwag

```python
# 判別式評估示例
question = "地球的衛星是？A. 火星 B. 月球 C. 太陽 D. 木星"
options = ["A", "B", "C", "D"]

# 計算每個選項的對數似然
logits = model.get_logits(question, options)
prediction = options[logits.argmax()]  # "B"
```

#### 生成式評估 (Generative Evaluation)
- **適用任務**: 開放式問答、代碼生成、摘要
- **評估方式**: 生成完整答案，與參考答案比對
- **優勢**: 更接近實際使用場景
- **示例**: GSM8K, HumanEval, TriviaQA

```python
# 生成式評估示例
question = "1+1=?"
generated = model.generate(question)  # "2"

# 與標準答案比對
is_correct = (generated == ground_truth)
```

### 2.2 評估流程

OpenCompass 的評估流程分為四個階段：

```
┌─────────────┐
│ 1. 配置階段 │  定義模型、數據集、評估參數
└──────┬──────┘
       │
┌──────▼──────┐
│ 2. 推理階段 │  模型對數據集進行預測
└──────┬──────┘
       │
┌──────▼──────┐
│ 3. 評分階段 │  根據預測結果計算指標
└──────┬──────┘
       │
┌──────▼──────┐
│ 4. 報告階段 │  生成可視化報告與分析
└─────────────┘
```

### 2.3 核心組件

**數據集抽象 (Dataset)**:
```python
from opencompass.datasets import CEvalDataset

dataset = CEvalDataset(
    path='ceval-exam',
    name='computer_network',
    split='val'
)
```

**模型抽象 (Model)**:
```python
from opencompass.models import HuggingFaceCausalLM

model = HuggingFaceCausalLM(
    path='Qwen/Qwen-7B',
    tokenizer_path='Qwen/Qwen-7B',
    max_seq_len=2048,
    batch_size=16
)
```

**評估器 (Evaluator)**:
```python
from opencompass.evaluators import AccuracyEvaluator

evaluator = AccuracyEvaluator()
accuracy = evaluator.score(predictions, references)
```

---

## 3. 實現原理與步驟

### 3.1 評估配置文件

OpenCompass 使用 Python 配置文件定義評估任務：

```python
# configs/eval_llama_ceval.py
from opencompass.models import HuggingFaceCausalLM
from opencompass.datasets import CEvalDataset
from opencompass.evaluators import AccuracyEvaluator

# 定義模型
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-2-7b',
        path='meta-llama/Llama-2-7b-hf',
        tokenizer_path='meta-llama/Llama-2-7b-hf',
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1)
    )
]

# 定義數據集
datasets = [
    dict(
        type=CEvalDataset,
        path='ceval-exam',
        name='computer_network',
        abbr='ceval-computer_network'
    ),
    dict(
        type=CEvalDataset,
        path='ceval-exam',
        name='operating_system',
        abbr='ceval-operating_system'
    )
]

# 評估配置
work_dir = './outputs/llama_ceval'
```

### 3.2 關鍵參數說明

| 參數名稱 | 含義 | 推薦值 | 影響 |
|:---|:---|:---|:---|
| `max_seq_len` | 最大序列長度 | 2048/4096 | 長文本處理能力 |
| `batch_size` | 批次大小 | 8/16/32 | 評估速度 vs 記憶體 |
| `num_gpus` | GPU 數量 | 1/2/4 | 並行加速 |
| `num_fewshot` | Few-shot 樣本數 | 0/5 | 評估難度 |
| `temperature` | 生成溫度 | 0.0 | 評估時使用貪婪解碼 |
| `max_out_len` | 最大輸出長度 | 512/1024 | 生成任務限制 |

### 3.3 評估執行流程

#### Step 1: 環境準備
```bash
# 安裝 OpenCompass
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .

# 下載數據集
python tools/download_dataset.py --dataset ceval
```

#### Step 2: 執行評估
```bash
# 使用配置文件執行
python run.py configs/eval_llama_ceval.py

# 或使用命令行參數
python run.py \
    --models hf_llama_7b \
    --datasets ceval_gen \
    --work-dir ./outputs
```

#### Step 3: 監控進度
```bash
# 查看評估進度
tail -f outputs/logs/infer_*.log

# 查看中間結果
ls outputs/predictions/
```

#### Step 4: 查看結果
```bash
# 查看匯總結果
cat outputs/summary/summary_*.txt

# 或使用 Python API
from opencompass.utils import read_results
results = read_results('outputs/summary/summary_*.csv')
```

---

## 4. 性能表現與對比

### 4.1 主流模型在中文基準上的表現

#### C-Eval (Chinese Evaluation Suite)

| 模型 | 整體 | STEM | 社會科學 | 人文學科 | 其他 |
|:---|---:|---:|---:|---:|---:|
| **GPT-4** | 68.7% | 67.1% | 77.6% | 64.5% | 67.8% |
| **Qwen-14B** | 72.1% | 70.2% | 81.8% | 67.1% | 68.9% |
| **Qwen-7B** | 59.7% | 56.2% | 74.1% | 63.1% | 56.2% |
| **ChatGLM3-6B** | 51.7% | 48.9% | 60.3% | 55.2% | 49.8% |
| **Llama-2-7B** | 45.3% | 42.1% | 52.9% | 48.7% | 44.9% |
| **Llama-2-13B** | 50.8% | 47.5% | 59.1% | 54.3% | 49.2% |

**關鍵觀察**:
- Qwen 系列在中文任務上具有顯著優勢（更多中文預訓練數據）
- Llama-2 在中文基準上表現較弱（主要是英文預訓練）
- 社會科學類別得分普遍高於 STEM（可能因為 STEM 需要更專業的知識）

#### CMMLU (Chinese Massive Multitask Language Understanding)

| 模型 | 整體 | 醫學 | 法律 | 歷史 | 工程 |
|:---|---:|---:|---:|---:|---:|
| **GPT-4** | 71.0% | 69.2% | 73.8% | 72.4% | 68.5% |
| **Qwen-14B** | 70.2% | 68.1% | 72.6% | 71.3% | 67.9% |
| **Qwen-7B** | 58.2% | 55.7% | 60.9% | 59.8% | 56.3% |
| **ChatGLM3-6B** | 50.3% | 48.2% | 52.1% | 51.6% | 49.7% |
| **Llama-2-7B** | 42.1% | 39.8% | 44.3% | 43.2% | 40.9% |

### 4.2 英文基準表現

#### MMLU (Massive Multitask Language Understanding)

| 模型 | 整體 | STEM | 社會科學 | 人文 | 其他 |
|:---|---:|---:|---:|---:|---:|
| **GPT-4** | 86.4% | 83.2% | 88.6% | 85.1% | 87.2% |
| **Llama-2-70B** | 69.8% | 67.3% | 71.4% | 69.2% | 70.1% |
| **Llama-2-13B** | 54.8% | 52.1% | 56.3% | 54.2% | 55.7% |
| **Llama-2-7B** | 46.8% | 43.9% | 48.2% | 46.1% | 47.9% |
| **Qwen-7B** | 56.4% | 54.2% | 58.1% | 55.9% | 57.3% |

**關鍵觀察**:
- Llama-2 在英文基準上表現更好（英文預訓練為主）
- 模型規模對性能有顯著影響（7B → 13B → 70B 遞增）
- GPT-4 仍是絕對領先者

### 4.3 推理與代碼能力

| 模型 | GSM8K (數學) | HumanEval (代碼) | BBH (推理) |
|:---|---:|---:|---:|
| **GPT-4** | 92.0% | 67.0% | 86.7% |
| **Qwen-14B** | 61.3% | 43.9% | 67.8% |
| **ChatGLM3-6B** | 72.3% | 58.0% | 54.2% |
| **Llama-2-7B** | 14.6% | 12.8% | 38.9% |

**關鍵觀察**:
- 數學推理與代碼生成需要強大的邏輯能力
- 專門優化過的模型（ChatGLM3）在特定任務上可超越更大的模型

---

## 5. 技術優勢

### 5.1 OpenCompass 核心優勢

| 優勢項目 | 說明 |
|:---|:---|
| **全面覆蓋** | 支持 100+ 數據集，涵蓋語言、推理、知識、代碼、多模態 |
| **標準化評估** | 統一評估協議，確保不同模型結果可比較 |
| **高效並行** | 支持多 GPU 分佈式評估，大幅縮短評估時間 |
| **靈活擴展** | 易於添加自定義數據集、模型、評估指標 |
| **豐富可視化** | 自動生成雷達圖、熱力圖、對比表格 |
| **開源社區** | 活躍的開源社區，持續更新基準與模型 |

### 5.2 vs 其他評估框架

| 特性 | OpenCompass | LM-Eval-Harness | EleutherAI Eval |
|:---|:---:|:---:|:---:|
| **中文基準** | ✅ 豐富 | ⚠️ 有限 | ❌ 缺乏 |
| **易用性** | ✅ 配置簡單 | ✅ 命令行友好 | ⚠️ 較複雜 |
| **可視化** | ✅ 內建 | ⚠️ 基礎 | ❌ 無 |
| **分佈式** | ✅ 支持 | ⚠️ 有限 | ⚠️ 有限 |
| **社區活躍度** | ✅ 高 | ✅ 高 | ⚠️ 中等 |

**使用建議**:
- **中文模型評估**: 優先選擇 OpenCompass
- **英文模型評估**: OpenCompass 或 LM-Eval-Harness
- **研究用途**: OpenCompass（更全面的基準）
- **快速驗證**: LM-Eval-Harness（更輕量）

---

## 6. 實驗設計與實作

### 6.1 實驗環境

- **評估框架**: OpenCompass 0.2.3+
- **模型**:
  - `meta-llama/Llama-2-7b-hf` (7B 參數)
  - `Qwen/Qwen-7B` (7B 參數)
- **評估基準**:
  - C-Eval (中文綜合評估)
  - CMMLU (中文多任務理解)
  - MMLU (英文多任務理解)
- **硬體**: NVIDIA GPU (16GB+ VRAM 推薦)

### 6.2 實驗流程

#### 1. **環境準備** (`01-Setup.ipynb`)
**目標**: 配置 OpenCompass 評估環境

**步驟**:
- 安裝 OpenCompass 框架與依賴
- 下載評估數據集（C-Eval, CMMLU 子集）
- 驗證 GPU 環境
- 載入待評估模型
- 執行簡單測試確保環境正常

**預期輸出**:
```
✅ OpenCompass installed successfully
✅ C-Eval dataset downloaded (52 subjects)
✅ GPU detected: NVIDIA A100 (40GB)
✅ Llama-2-7B loaded successfully
✅ Qwen-7B loaded successfully
```

#### 2. **執行評估** (`02-Evaluate.ipynb`)
**目標**: 在多個基準上評估兩個模型

**步驟**:
- 配置評估任務（選擇數據集、Few-shot 設定）
- 執行 C-Eval 評估（STEM、社會科學、人文學科子集）
- 執行 CMMLU 評估（醫學、法律、歷史子集）
- 收集評估日誌與中間結果
- 保存預測結果與評分

**評估配置**:
```python
eval_config = {
    'models': ['Llama-2-7B', 'Qwen-7B'],
    'datasets': [
        'ceval-computer_network',
        'ceval-operating_system',
        'ceval-computer_architecture',
        'cmmlu-anatomy',
        'cmmlu-clinical_knowledge'
    ],
    'num_fewshot': 5,
    'batch_size': 16,
    'max_seq_len': 2048
}
```

**預期輸出**:
```
Evaluating Llama-2-7B on C-Eval:
  computer_network: 45.2% (113/250 correct)
  operating_system: 42.8% (107/250 correct)
  ...

Evaluating Qwen-7B on C-Eval:
  computer_network: 58.4% (146/250 correct)
  operating_system: 57.2% (143/250 correct)
  ...
```

#### 3. **結果分析** (`03-Analyze.ipynb`)
**目標**: 深度分析評估結果

**分析維度**:
- **整體性能**: 平均準確率、學科分佈
- **對比分析**: Llama-2-7B vs Qwen-7B 差異
- **學科細分**: 哪些學科表現好/差
- **錯誤案例**: 分析典型錯誤模式

**關鍵指標**:
```python
metrics = {
    'accuracy': 準確率,
    'f1_score': F1 分數（多分類）,
    'score_by_category': 按類別統計,
    'error_analysis': 錯誤類型分類
}
```

**預期輸出**:
```
📊 Overall Performance:
  Llama-2-7B: 45.3% avg accuracy
  Qwen-7B: 59.7% avg accuracy
  Δ: +14.4% (Qwen wins)

📊 By Category:
  STEM:
    Llama-2-7B: 42.1%
    Qwen-7B: 56.2% (+14.1%)

  Social Science:
    Llama-2-7B: 48.7%
    Qwen-7B: 62.4% (+13.7%)

❌ Common Errors (Llama-2-7B):
  - Factual errors: 38%
  - Reasoning errors: 29%
  - Language understanding: 33%
```

#### 4. **可視化與報告** (`04-Visualize_and_Report.ipynb`)
**目標**: 生成可視化圖表與自動化報告

**可視化類型**:
- **雷達圖**: 多維度能力對比
- **熱力圖**: 學科表現矩陣
- **柱狀圖**: 整體性能對比
- **分佈圖**: 分數分佈直方圖

**自動報告內容**:
```markdown
# OpenCompass Evaluation Report

## Executive Summary
- Qwen-7B outperforms Llama-2-7B by 14.4% on C-Eval
- Largest gap in social science subjects (+13.7%)
- Both models struggle with advanced STEM topics

## Recommendations
1. For Chinese tasks: Use Qwen-7B
2. For English tasks: Consider Llama-2-7B
3. For specialized domains: Fine-tune on domain data
```

---

## 7. 實戰參數調優策略 (2024 年行業最佳實踐)

### 7.1 基於評估場景的配置

#### 快速驗證場景
**目標**: 快速驗證模型基本能力（30 分鐘內）

```python
quick_eval_config = {
    'num_fewshot': 0,           # Zero-shot 最快
    'batch_size': 32,           # 大批次提速
    'max_seq_len': 1024,        # 較短序列
    'datasets': [               # 選擇代表性子集
        'ceval-computer_network',  # STEM 代表
        'ceval-chinese_language',  # 人文代表
        'ceval-marxism'           # 社會科學代表
    ],
    'num_samples': 100          # 每個數據集僅評估 100 樣本
}
```

**適用時機**:
- 模型選型初期
- 訓練過程中的檢查點驗證
- CI/CD 自動化測試

#### 標準評估場景
**目標**: 全面評估模型能力（2-4 小時）

```python
standard_eval_config = {
    'num_fewshot': 5,           # 5-shot 評估
    'batch_size': 16,           # 平衡速度與記憶體
    'max_seq_len': 2048,        # 標準序列長度
    'datasets': [               # C-Eval 完整 52 個學科
        'ceval-*'
    ],
    'num_samples': None         # 使用完整數據集
}
```

**適用時機**:
- 模型正式發布前
- 研究論文實驗
- 對外展示結果

#### 深度評估場景
**目標**: 極致準確的評估（6-12 小時）

```python
deep_eval_config = {
    'num_fewshot': 5,
    'batch_size': 8,            # 小批次確保穩定
    'max_seq_len': 4096,        # 支持長文本
    'datasets': [
        'ceval-*',              # C-Eval 完整
        'cmmlu-*',              # CMMLU 完整
        'mmlu-*'                # MMLU 完整
    ],
    'num_runs': 3,              # 多次運行取平均
    'temperature': 0.0,         # 確定性解碼
    'seed': 42                  # 固定隨機種子
}
```

**適用時機**:
- 學術論文投稿
- 競爭性基準排行榜
- 高風險決策（如生產模型選型）

### 7.2 不同模型規模的優化策略

| 模型規模 | Batch Size | 序列長度 | Few-shot | 預計時間 |
|:---|---:|---:|---:|:---|
| **小型** (<3B) | 32 | 2048 | 5 | 1-2h |
| **中型** (3-10B) | 16 | 2048 | 5 | 2-4h |
| **大型** (10-30B) | 8 | 2048 | 5 | 4-8h |
| **超大型** (>30B) | 4 | 2048 | 5 | 8-16h |

### 7.3 精度保證策略

#### 固定所有隨機性
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """確保評估可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在評估開始前調用
set_seed(42)
```

#### 統一評估協議
```python
# 評估協議配置
evaluation_protocol = {
    'generation': {
        'do_sample': False,      # 使用貪婪解碼
        'temperature': 0.0,      # 溫度為 0
        'top_p': 1.0,           # 不使用 nucleus sampling
        'top_k': None,          # 不使用 top-k
        'num_beams': 1,         # 不使用 beam search
        'repetition_penalty': 1.0
    },
    'fewshot': {
        'num_shots': 5,         # 固定 5-shot
        'shuffle': False,       # 不打亂樣本順序
        'seed': 42              # 固定選擇的樣本
    }
}
```

### 7.4 故障診斷指南

| 問題現象 | 可能原因 | 解決方案 |
|:---|:---|:---|
| 評估分數異常低 (<20%) | 模型未正確載入 / Few-shot 格式錯誤 | 檢查模型路徑、驗證 Few-shot 模板 |
| 評估速度極慢 (>10h) | Batch size 過小 / 單 GPU | 增大 batch size、啟用多 GPU |
| OOM 錯誤 | 序列長度過長 / Batch size 過大 | 減小 max_seq_len 或 batch_size |
| 結果不可重現 | 隨機種子未固定 | 使用 `set_seed()` 固定所有隨機源 |
| 生成式評估失敗 | 答案格式不匹配 | 檢查後處理函數，調整正則表達式 |
| 多選題準確率為 0 | 模型輸出格式錯誤 | 檢查模型是否支持選項提取 |

**常見錯誤排查步驟**:
1. 檢查日誌文件：`tail -f outputs/logs/infer_*.log`
2. 驗證單個樣本：手動運行一個測試樣本
3. 檢查預測文件：查看 `outputs/predictions/*.json`
4. 對比參考實現：與官方配置文件對比

---

## 8. 評估管線與生產環境最佳實踐

### 8.1 自動化評估 CI/CD

#### GitHub Actions 範例
```yaml
# .github/workflows/evaluate.yml
name: Model Evaluation

on:
  push:
    branches: [main]
    paths: ['models/**']

jobs:
  evaluate:
    runs-on: self-hosted  # 使用自建 GPU 機器
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install opencompass
          pip install -r requirements.txt

      - name: Run evaluation
        run: |
          python run.py configs/eval_model.py \
            --work-dir ./outputs

      - name: Check regression
        run: |
          python scripts/check_regression.py \
            --baseline results/baseline.json \
            --current outputs/summary/summary.json \
            --threshold 0.02  # 允許 2% 波動

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: outputs/
```

### 8.2 評估結果資料庫

```python
import sqlite3
import json
from datetime import datetime

class EvaluationDatabase:
    def __init__(self, db_path='evaluations.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """創建評估結果表"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT,
                dataset TEXT NOT NULL,
                accuracy REAL,
                f1_score REAL,
                eval_config TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def insert_result(self, model_name, dataset, accuracy, f1_score, config):
        """插入評估結果"""
        self.conn.execute('''
            INSERT INTO evaluations
            (model_name, dataset, accuracy, f1_score, eval_config)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_name, dataset, accuracy, f1_score, json.dumps(config)))
        self.conn.commit()

    def get_model_history(self, model_name):
        """查詢模型歷史評估結果"""
        cursor = self.conn.execute('''
            SELECT dataset, accuracy, f1_score, timestamp
            FROM evaluations
            WHERE model_name = ?
            ORDER BY timestamp DESC
        ''', (model_name,))
        return cursor.fetchall()

    def compare_models(self, model_a, model_b, dataset):
        """比較兩個模型在同一數據集上的表現"""
        cursor = self.conn.execute('''
            SELECT model_name, AVG(accuracy) as avg_acc
            FROM evaluations
            WHERE model_name IN (?, ?) AND dataset = ?
            GROUP BY model_name
        ''', (model_a, model_b, dataset))
        return dict(cursor.fetchall())
```

### 8.3 實時監控儀表板

```python
import streamlit as st
import pandas as pd
import plotly.express as px

def create_dashboard():
    st.title("🎯 OpenCompass Evaluation Dashboard")

    # 載入評估結果
    db = EvaluationDatabase()

    # 模型選擇
    models = st.multiselect(
        "Select models",
        options=['Llama-2-7B', 'Qwen-7B', 'ChatGLM3-6B']
    )

    # 數據集選擇
    datasets = st.multiselect(
        "Select datasets",
        options=['C-Eval', 'CMMLU', 'MMLU']
    )

    # 時間範圍
    date_range = st.date_input("Date range", [])

    # 載入數據
    df = load_evaluation_data(models, datasets, date_range)

    # 可視化
    col1, col2 = st.columns(2)

    with col1:
        # 整體準確率對比
        fig = px.bar(
            df, x='model', y='accuracy',
            color='dataset',
            title='Model Accuracy Comparison'
        )
        st.plotly_chart(fig)

    with col2:
        # 時間趨勢
        fig = px.line(
            df, x='timestamp', y='accuracy',
            color='model',
            title='Accuracy Trend Over Time'
        )
        st.plotly_chart(fig)

    # 詳細表格
    st.dataframe(df, use_container_width=True)

# 運行
if __name__ == '__main__':
    create_dashboard()
```

**啟動儀表板**:
```bash
streamlit run dashboard.py
```

### 8.4 告警系統

```python
class RegressionDetector:
    def __init__(self, baseline_results, threshold=0.02):
        """
        Args:
            baseline_results: 基線評估結果
            threshold: 允許的性能下降閾值（2%）
        """
        self.baseline = baseline_results
        self.threshold = threshold

    def check_regression(self, current_results):
        """檢測性能回歸"""
        regressions = []

        for dataset, current_acc in current_results.items():
            baseline_acc = self.baseline.get(dataset, 0)
            diff = current_acc - baseline_acc

            if diff < -self.threshold:
                regressions.append({
                    'dataset': dataset,
                    'baseline': baseline_acc,
                    'current': current_acc,
                    'diff': diff,
                    'severity': 'CRITICAL' if diff < -0.05 else 'WARNING'
                })

        return regressions

    def send_alert(self, regressions):
        """發送告警通知"""
        if not regressions:
            print("✅ No regression detected")
            return

        message = "⚠️  Performance Regression Detected!\n\n"
        for reg in regressions:
            message += f"{reg['severity']}: {reg['dataset']}\n"
            message += f"  Baseline: {reg['baseline']:.2%}\n"
            message += f"  Current:  {reg['current']:.2%}\n"
            message += f"  Diff:     {reg['diff']:.2%}\n\n"

        # 發送到 Slack / 郵件 / 企業微信
        self.send_to_slack(message)
        self.send_email(message)

    def send_to_slack(self, message):
        """發送到 Slack"""
        import requests
        webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        requests.post(webhook_url, json={'text': message})
```

---

## 9. 結論與學習成果

通過本實驗，你將獲得：

### 9.1 核心技能

1. **評估框架掌握**
   - ✅ 熟練使用 OpenCompass 評估 LLM
   - ✅ 理解評估配置與參數調優
   - ✅ 掌握分佈式評估加速技巧

2. **結果分析能力**
   - ✅ 解讀多維度評估指標
   - ✅ 識別模型優勢與劣勢領域
   - ✅ 進行模型間橫向對比
   - ✅ 分析錯誤案例並定位問題

3. **工程實踐**
   - ✅ 建立自動化評估管線
   - ✅ 實現評估結果監控與告警
   - ✅ 生成專業評估報告
   - ✅ 整合評估到 CI/CD 流程

4. **決策支持**
   - ✅ 基於評估結果選擇合適模型
   - ✅ 為模型優化提供數據支持
   - ✅ 量化訓練/壓縮技術的效果

### 9.2 實際應用場景

**模型選型**:
```
場景: 開發中文客服系統
步驟:
  1. 使用 OpenCompass 評估候選模型（Qwen, ChatGLM, Llama-2）
  2. 關注 CMMLU 社會科學類別（客服相關）
  3. 結果: Qwen-7B 在該類別得分 62.4%，選定為基礎模型
```

**訓練驗證**:
```
場景: 驗證 LoRA 微調效果
步驟:
  1. 評估基礎模型：C-Eval 45.3%
  2. 微調後評估：C-Eval 48.7% (+3.4%)
  3. 結論: LoRA 微調有效，可進入生產
```

**壓縮評估**:
```
場景: 量化模型質量檢查
步驟:
  1. 評估原始模型：MMLU 56.4%
  2. 評估 4-bit GPTQ：MMLU 55.1% (-1.3%)
  3. 結論: 精度損失可接受，可部署量化版本
```

---

## 10. 技術限制與改進方向

### 10.1 當前限制分析

| 限制項目 | 具體表現 | 影響 | 緩解方案 |
|:---|:---|:---|:---|
| **基準覆蓋** | 現有基準無法涵蓋所有能力（創意、情感） | 評估片面 | 結合人工評估、GPT-4 評判 |
| **評估成本** | 大規模評估耗時長（8-16h） | 迭代速度慢 | 使用子集快速驗證、增量評估 |
| **中英文差異** | 同一模型中英文表現差異大 | 難以綜合評價 | 分別評估、加權組合 |
| **Few-shot 敏感** | 樣本選擇影響結果穩定性 | 可重現性差 | 固定樣本池、多次評估取平均 |
| **長文本限制** | 大多數基準限制在 2K tokens 內 | 無法測試長文本能力 | 使用長文本專用基準（如 LongBench） |

### 10.2 未來研究方向

#### 動態評估
```python
# 根據模型能力動態調整評估難度
def adaptive_evaluation(model, dataset):
    # 從中等難度開始
    score = evaluate(model, dataset, difficulty='medium')

    if score > 0.8:
        # 模型表現好，增加難度
        final_score = evaluate(model, dataset, difficulty='hard')
    elif score < 0.5:
        # 模型表現差，降低難度以獲得更細粒度的評估
        final_score = evaluate(model, dataset, difficulty='easy')
    else:
        final_score = score

    return final_score
```

#### 對抗評估
```python
# 測試模型對對抗樣本的魯棒性
from opencompass.datasets import AdversarialDataset

# 自動生成對抗樣本
adversarial_dataset = AdversarialDataset.from_base(
    base_dataset='ceval',
    perturbation_type='paraphrase',  # 改述攻擊
    intensity=0.3
)

# 評估魯棒性
robust_acc = evaluate(model, adversarial_dataset)
```

#### 多模態評估
```python
# 擴展到圖像-文本任務
from opencompass.datasets import MMBenchDataset

mm_dataset = MMBenchDataset(
    modalities=['image', 'text'],
    tasks=['vqa', 'captioning', 'reasoning']
)

mm_score = evaluate(multimodal_model, mm_dataset)
```

### 10.3 評估與訓練閉環

```
┌─────────────┐
│   評估模型   │  識別弱項（如：數學推理差）
└──────┬──────┘
       │
┌──────▼──────┐
│ 數據針對篩選 │  收集更多數學推理數據
└──────┬──────┘
       │
┌──────▼──────┐
│   微調訓練   │  針對性訓練
└──────┬──────┘
       │
┌──────▼──────┐
│   再次評估   │  驗證改進效果
└──────┬──────┘
       │
       └─────── 持續迭代
```

**實施步驟**:
1. 使用 OpenCompass 全面評估基礎模型
2. 分析結果，識別薄弱領域（如 GSM8K 僅 14.6%）
3. 使用數據篩選技術（Lab-4.2）獲取高質量數學數據
4. 進行領域微調（Lab-1.1 PEFT）
5. 再次評估，驗證 GSM8K 提升至 40%+
6. 迭代上述過程

---

## 11. 參考資料

### 核心論文

**OpenCompass**:
- **主論文**: "OpenCompass: A Universal Evaluation Platform for Foundation Models" (2023)
  - arXiv: https://arxiv.org/abs/2304.xxxxx
  - GitHub: https://github.com/open-compass/opencompass

**評估基準**:
- **C-Eval**: "C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models"
  - arXiv: https://arxiv.org/abs/2305.08322
  - 網站: https://cevalbenchmark.com/

- **CMMLU**: "CMMLU: Measuring Massive Multitask Language Understanding in Chinese"
  - arXiv: https://arxiv.org/abs/2306.09212
  - GitHub: https://github.com/haonan-li/CMMLU

- **MMLU**: "Measuring Massive Multitask Language Understanding"
  - arXiv: https://arxiv.org/abs/2009.03300
  - GitHub: https://github.com/hendrycks/test

**其他重要基準**:
- **GSM8K**: "Training Verifiers to Solve Math Word Problems" (arXiv:2110.14168)
- **HumanEval**: "Evaluating Large Language Models Trained on Code" (arXiv:2107.03374)
- **BBH**: "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them" (arXiv:2210.09261)

### 工具與實現

**OpenCompass 生態**:
- **官方文檔**: https://opencompass.readthedocs.io/
- **GitHub 倉庫**: https://github.com/open-compass/opencompass
- **Leaderboard**: https://opencompass.org.cn/leaderboard-llm
- **Discord 社區**: https://discord.gg/opencompass

**其他評估框架**:
- **LM-Eval-Harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **HELM**: https://github.com/stanford-crfm/helm
- **OpenAI Evals**: https://github.com/openai/evals

### 模型與資料集

**預訓練模型**:
- **Llama-2**: https://huggingface.co/meta-llama/Llama-2-7b-hf
- **Qwen**: https://huggingface.co/Qwen/Qwen-7B
- **ChatGLM**: https://huggingface.co/THUDM/chatglm3-6b

**評估數據集**:
- **C-Eval**: https://huggingface.co/datasets/ceval/ceval-exam
- **CMMLU**: https://huggingface.co/datasets/haonan-li/cmmlu
- **MMLU**: https://huggingface.co/datasets/cais/mmlu

### 延伸閱讀

**技術博客**:
- OpenCompass 官方博客: https://opencompass.org.cn/blog
- Hugging Face Blog: https://huggingface.co/blog/evaluating-llm

**評估方法論**:
- "Holistic Evaluation of Language Models" (HELM) - Stanford
- "Beyond the Imitation Game" (BIG-bench) - Google

**行業報告**:
- "State of AI Report 2024" - https://www.stateof.ai/
- "Foundation Model Transparency Index" - Stanford HAI

---

## 📚 速記心法與口訣

### 🎯 評估四步法

```
評估流程:
1. 選基準 - 根據目標選擇合適基準
2. 跑評估 - 執行標準評估協議
3. 析結果 - 多維度分析評估結果
4. 定方向 - 基於結果制定優化策略

口訣: 「選跑析定，環環相扣」
```

### ⚡ OpenCompass 三要素

```
配置三要素:
模 - 定義待評估模型
數 - 選擇評估數據集
參 - 設定評估參數

「模數參，缺一不可」
```

### 📊 結果解讀三層次

```
解讀層次:
1. 整體分數 - 宏觀能力水平
2. 學科分佈 - 優劣勢識別
3. 錯誤分析 - 具體問題定位

「總分看水平，分科找問題，錯例定方向」
```

### 🔧 調優三原則

```
調優原則:
快 - 快速驗證用子集
準 - 準確評估用全集
穩 - 穩定結果固定種子

「快準穩，三位一體」
```

---

**狀態**: 📝 規劃完成，待開發
**最後更新**: 2025-10-17
**維護者**: Claude Code
**預計開發時間**: 6 小時
