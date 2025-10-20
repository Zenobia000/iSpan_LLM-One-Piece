# 📘 評估與數據工程章節撰文風格規範

**適用範圍**: `04-Evaluation_and_Data_Engineering` 目錄下所有文檔與實驗室
**參考來源**: `01-Core_Training_Techniques` 和 `03-Model_Compression` 已驗證的撰文模式
**最後更新**: 2025-10-17

---

## 🎯 核心原則

### 1. 一致性優先
- 與 `01-Core_Training_Techniques` 和 `03-Model_Compression` 保持高度一致的結構和風格
- 學習者應能無縫切換章節而不感到困惑

### 2. 評估思維優先
- 強調「如何評估」而非「如何訓練」
- 關注數據質量對模型性能的影響
- 培養客觀評估與數據驅動決策思維

### 3. 工程導向
- 不僅講「如何評估」,更講「如何解讀結果」
- 包含生產環境評估管線與數據處理最佳實踐
- 強調自動化與可重現性

---

## 📂 目錄結構規範

```
04-Evaluation_and_Data_Engineering/
├── 01-Theory/
│   ├── 4.1-Evaluation_Benchmarks.md      # 評估基準理論
│   └── 4.2-Data_Engineering.md           # 數據工程理論
├── 02-Labs/
│   ├── Lab-4.1-Evaluate_with_OpenCompass/
│   │   ├── README.md                     # 實驗說明文檔
│   │   ├── 01-Setup.ipynb
│   │   ├── 02-Evaluate.ipynb
│   │   ├── 03-Analyze.ipynb
│   │   └── 04-Visualize_and_Report.ipynb
│   └── Lab-4.2-Efficient_Data_Filtering/
│       ├── README.md
│       ├── 01-Setup.ipynb
│       ├── 02-Filter.ipynb
│       ├── 03-Validate.ipynb
│       └── 04-Pipeline.ipynb
└── WRITING_STYLE_GUIDE.md                # 本文檔
```

---

## 📄 README.md 撰文模板

### 結構大綱 (11個必要章節)

```markdown
# Lab 4.X: [技術名稱] - [簡短描述]

## 概述
**[核心技術]** 是... [技術背景 1-2段]
本實驗將... [實驗目標聲明]

![技術示意圖](url_or_local_path)

---

## 1. 技術背景與動機

### 1.1 為何需要評估/數據工程?
- **評估必要性**: 客觀衡量模型能力,識別優勢與不足
- **數據重要性**: 高質量數據是模型性能的基石
- **工程化需求**: 自動化評估管線提升研發效率

### 1.2 評估/數據工程技術分類
[橫向比較: 能力評估 vs 性能評估 vs 數據質量評估]

---

## 2. [技術名稱] 核心原理

### 2.1 理論基礎
[數學原理、演算法基礎]

### 2.2 技術實現
[具體實現方式、關鍵步驟]

### 2.3 主流方法對比
[OpenCompass vs LM-Eval-Harness / IFD vs DEITA vs LESS]

---

## 3. 實現原理與步驟

### 3.1 關鍵配置
\```python
# 配置範例 (帶詳細註解)
config = EvaluationConfig(
    datasets=['ceval', 'cmmlu'],  # 評估基準
    batch_size=16,                # 批次大小
    num_fewshot=5,                # Few-shot 樣本數
    ...
)
\```

### 3.2 關鍵參數說明
| 參數名稱 | 含義 | 推薦值 | 影響 |
|:---|:---|:---|:---|
| `batch_size` | 批次大小 | 16/32 | 評估速度 vs 記憶體 |
| `num_fewshot` | Few-shot 數量 | 0/5 | 測試難度與真實性 |

### 3.3 工作流程
1. 步驟1: [詳細說明]
2. 步驟2: [詳細說明]
...

---

## 4. 性能表現與對比

### 4.1 評估基準結果

| 模型 | C-Eval | CMMLU | MMLU | 綜合得分 |
|:---|:---|:---|:---|:---|
| **Llama-2-7B** | 45.3% | 42.1% | 46.8% | 44.7% |
| **Qwen-7B** | 59.7% | 58.2% | 56.4% | 58.1% |
| **差異** | **+14.4%** | **+16.1%** | **+9.6%** | **+13.4%** |

### 4.2 數據篩選效果

| 數據集 | 樣本數 | 訓練時間 | C-Eval 準確率 | 效率提升 |
|:---|:---|:---|:---|:---|
| **原始數據** | 52K | 12h | 45.3% | - |
| **篩選後** | 15.6K | 3.5h | 47.1% | **3.4x** |

---

## 5. 技術優勢

| 優勢項目 | 說明 |
|:---|:---|
| **客觀性** | 基於標準基準,避免主觀偏見 |
| **全面性** | 多維度評估,涵蓋能力與性能 |
| **可重現** | 固定評估協議,結果可驗證 |
| **自動化** | 完整管線,高效批量評估 |

---

## 6. 實驗設計與實作

### 6.1 實驗環境
- **評估框架**: OpenCompass / IFD+DEITA
- **模型**: Llama-2-7B, Qwen-7B
- **基準**: C-Eval, CMMLU, MMLU
- **數據集**: Alpaca, Dolly, Self-Instruct

### 6.2 實驗流程
1. **環境準備** (`01-Setup.ipynb`)
   - 安裝評估框架
   - 下載基準數據集
   - 驗證模型載入

2. **執行評估/篩選** (`02-[操作].ipynb`)
   - 配置評估任務
   - 執行評估/篩選
   - 收集結果

3. **結果分析** (`03-Analyze.ipynb` / `03-Validate.ipynb`)
   - 解析評估結果
   - 識別優劣勢領域
   - 驗證篩選效果

4. **可視化與報告** (`04-Visualize.ipynb` / `04-Pipeline.ipynb`)
   - 生成可視化圖表
   - 自動生成報告
   - 建立自動化管線

---

## 7. 實戰參數調優策略 (2024 年行業最佳實踐)

### 7.1 評估配置優化

| 場景 | Few-shot | Batch Size | 評估模式 | 預期時間 |
|:---|:---|:---|:---|:---|
| **快速驗證** | 0 | 32 | 生成式 | 30min |
| **標準評估** | 5 | 16 | 生成式 | 2-3h |
| **精確評估** | 5 | 8 | 多樣本 | 4-6h |

### 7.2 數據篩選策略

#### 高質量數據優先
\```python
# IFD + DEITA 融合評分
def calculate_data_score(sample):
    ifd_score = calculate_ifd(sample)        # 困難度
    complexity = analyze_complexity(sample)  # 複雜度
    diversity = calculate_diversity(sample)  # 多樣性

    # 加權融合
    final_score = (
        0.4 * ifd_score +
        0.3 * complexity +
        0.3 * diversity
    )
    return final_score
\```

#### 多樣性保持
\```python
# K-means 聚類確保覆蓋
from sklearn.cluster import KMeans

embeddings = embed_instructions(dataset)
clusters = KMeans(n_clusters=20).fit(embeddings)

# 從每個聚類選擇 Top-K
filtered_data = select_top_k_per_cluster(
    dataset, clusters, k=10
)
\```

### 7.3 評估結果解讀指南

| 分數範圍 | 能力等級 | 適用場景 | 改進建議 |
|:---|:---|:---|:---|
| **<40%** | 基礎級 | 簡單對話、基礎分類 | 增加訓練數據、延長訓練時間 |
| **40-60%** | 中級 | 通用助手、信息檢索 | 優化數據質量、調整超參數 |
| **60-80%** | 高級 | 專業領域、複雜推理 | 領域微調、專家數據 |
| **>80%** | 專家級 | 關鍵任務、高風險場景 | 持續監控、數據迭代 |

### 7.4 故障診斷指南

| 問題現象 | 可能原因 | 解決方案 |
|:---|:---|:---|
| 評估分數異常低 | 模型未正確載入 / Few-shot 樣本錯誤 | 檢查模型配置、驗證樣本格式 |
| 評估速度極慢 | Batch size 過小 / GPU 未使用 | 增大 batch size、啟用 GPU |
| 結果不可重現 | 隨機種子未固定 / 評估協議不一致 | 固定種子、統一評估配置 |
| 數據篩選後性能下降 | 篩選過度 / 多樣性損失 | 降低篩選比例、增加多樣性權重 |

---

## 8. 評估管線與生產環境最佳實踐

### 8.1 評估框架選擇

| 評估框架 | 適用場景 | 優勢 | 限制 |
|:---|:---|:---|:---|
| **OpenCompass** | 中文模型評估 | 中文基準豐富、易於配置 | 英文基準較少 |
| **LM-Eval-Harness** | 英文模型評估 | 國際標準、基準最全 | 中文支持有限 |
| **自建評估系統** | 定制化需求 | 靈活可控 | 開發成本高 |

### 8.2 評估 CI/CD 管線

```python
# 自動化評估管線範例
class EvaluationPipeline:
    def __init__(self, config):
        self.config = config
        self.evaluator = load_evaluator(config)

    def run(self, model_path, datasets):
        """執行完整評估流程"""
        # 1. 載入模型
        model = self.load_model(model_path)

        # 2. 執行評估
        results = {}
        for dataset in datasets:
            score = self.evaluator.evaluate(model, dataset)
            results[dataset] = score

        # 3. 生成報告
        report = self.generate_report(results)
        self.save_report(report)

        # 4. 觸發告警 (如性能下降)
        if self.check_regression(results):
            self.send_alert(results)

        return results
```

### 8.3 數據質量監控

```python
# 數據質量儀表板
class DataQualityMonitor:
    def __init__(self):
        self.metrics = {
            'avg_ifd_score': [],
            'avg_complexity': [],
            'diversity_score': [],
            'data_size': []
        }

    def track(self, data_batch):
        """追蹤數據質量指標"""
        self.metrics['avg_ifd_score'].append(
            np.mean([calc_ifd(s) for s in data_batch])
        )
        self.metrics['avg_complexity'].append(
            np.mean([analyze_complexity(s) for s in data_batch])
        )
        # ... 其他指標

    def visualize(self):
        """生成可視化儀表板"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 繪製趨勢圖
        axes[0, 0].plot(self.metrics['avg_ifd_score'])
        axes[0, 0].set_title('IFD Score Trend')
        # ... 其他圖表

        plt.tight_layout()
        plt.show()
```

---

## 9. 結論與學習成果

通過本實驗,您將獲得:

1. **評估體系理解** 掌握主流評估基準與指標體系
2. **實戰經驗** 使用業界標準工具評估真實模型
3. **分析能力** 解讀評估結果,識別模型優劣勢
4. **數據工程** 實施高效數據篩選與質量優化
5. **工程實踐** 建立自動化評估與數據管線

---

## 10. 技術限制與改進方向

### 10.1 當前限制分析

| 限制項目 | 具體表現 | 影響 | 緩解方案 |
|:---|:---|:---|:---|
| **基準覆蓋** | 現有基準無法涵蓋所有能力 | 評估片面性 | 多基準組合評估 |
| **主觀任務評估** | 創意寫作、對話質量難以自動化 | 需要人工評估 | 使用 GPT-4 作為評判 |
| **數據偏見** | 篩選可能引入領域偏見 | 模型泛化能力下降 | 保持多樣性、分層採樣 |
| **評估成本** | 大規模評估耗時長、成本高 | 開發迭代速度慢 | 使用子集快速驗證 |

### 10.2 未來研究方向

- **動態評估**: 根據模型能力動態調整評估難度
- **對抗評估**: 測試模型對對抗樣本的魯棒性
- **多模態評估**: 擴展到圖像、視頻、音頻
- **在線評估**: 實時監控生產環境模型性能

### 10.3 評估與訓練閉環

```
評估 → 發現不足 → 數據篩選 → 訓練優化 → 再評估
  ↑                                        ↓
  ←──────────────── 持續迭代 ────────────────
```

---

## 11. 參考資料

### 核心論文

**評估基準**:
- **C-Eval**: "C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models" (arXiv:2305.08322)
- **CMMLU**: "CMMLU: Measuring Massive Multitask Language Understanding in Chinese" (arXiv:2306.09212)
- **MMLU**: "Measuring Massive Multitask Language Understanding" (arXiv:2009.03300)

**數據工程**:
- **DEITA**: "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning" (arXiv:2312.15685)
- **IFD**: "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning" (arXiv:2308.12032)
- **LIMA**: "LIMA: Less Is More for Alignment" (arXiv:2305.11206)

### 工具與實現

**評估框架**:
- **OpenCompass**: https://github.com/open-compass/opencompass
- **LM-Eval-Harness**: https://github.com/EleutherAI/lm-evaluation-harness

**數據工程工具**:
- **Sentence-BERT**: https://www.sbert.net/
- **DVC (Data Version Control)**: https://dvc.org/

### 延伸閱讀
- **Phi-1.5 Technical Report**: https://arxiv.org/abs/2309.05463
- **OpenAI Evals**: https://github.com/openai/evals

---

## 📚 速記心法與口訣

### 🎯 評估四步法
```
評估流程:
1. 選基準 - 根據目標選擇合適基準
2. 跑評估 - 執行標準評估協議
3. 析結果 - 多維度分析評估結果
4. 定方向 - 基於結果制定優化策略

口訣: 「選跑析定,環環相扣」
```

### ⚡ 數據篩選三原則
```
篩選原則:
質 - 高質量優先 (IFD + DEITA)
多 - 保持多樣性 (聚類覆蓋)
驗 - 驗證效果 (A/B 測試)

「質多驗,缺一不可」
```

### 📊 結果解讀三層次
```
解讀層次:
1. 整體分數 - 宏觀能力水平
2. 學科分佈 - 優劣勢識別
3. 錯誤分析 - 具體問題定位

「總分看水平,分科找問題,錯例定方向」
```

---

**狀態**: ✅ 已完成 / ⏸️ 開發中 / 📝 規劃中
**最後更新**: YYYY-MM-DD
**維護者**: [維護者名稱]
```

---

## 📓 Jupyter Notebook 撰文規範

### Notebook 命名與結構

#### Lab-4.1: OpenCompass 評估

**01-Setup.ipynb**
```markdown
# Lab 4.1: OpenCompass Model Evaluation - Environment Setup

**Goal:** Prepare OpenCompass evaluation environment

**You will learn to:**
- Install OpenCompass framework
- Download and prepare evaluation datasets (C-Eval, CMMLU)
- Load models for evaluation
- Verify configuration

---

## Step 1: Install OpenCompass

OpenCompass is a comprehensive evaluation platform for foundation models...
```

**02-Evaluate.ipynb**
```markdown
# Lab 4.1: OpenCompass Model Evaluation - Execute Evaluation

**Goal:** Run evaluation on multiple benchmarks

**Key concepts:**
- Few-shot evaluation
- Batch inference optimization
- Result collection

---

## Step 1: Configure Evaluation Tasks

Before evaluation, we need to configure...
```

**03-Analyze.ipynb**
```markdown
# Lab 4.1: OpenCompass Model Evaluation - Result Analysis

**Goal:** Analyze evaluation results in depth

**Analysis dimensions:**
- Overall accuracy by benchmark
- Performance by subject category
- Model comparison
- Error case analysis

---

## Step 1: Load Evaluation Results

We'll load and parse the evaluation outputs...
```

**04-Visualize_and_Report.ipynb**
```markdown
# Lab 4.1: OpenCompass Model Evaluation - Visualization and Reporting

**Goal:** Generate visualizations and automated reports

**Deliverables:**
- Radar charts (multi-dimensional capability)
- Heatmaps (subject performance matrix)
- Comparison bar charts
- Automated evaluation report

---

## Step 1: Prepare Data for Visualization

First, we aggregate results across models and benchmarks...
```

#### Lab-4.2: 高效數據篩選

**01-Setup.ipynb**
```markdown
# Lab 4.2: Efficient Data Filtering - Data Preparation

**Goal:** Load and analyze instruction datasets

**You will learn to:**
- Load instruction datasets (Alpaca, Dolly)
- Analyze data distribution
- Prepare quality evaluation models
- Set filtering goals

---

## Step 1: Load Instruction Datasets

We'll work with popular instruction datasets...
```

**02-Filter.ipynb**
```markdown
# Lab 4.2: Efficient Data Filtering - Apply Filtering

**Goal:** Implement IFD and DEITA filtering methods

**Key algorithms:**
- IFD (Instruction Following Difficulty)
- DEITA complexity scoring
- Multi-objective optimization

---

## Step 1: Calculate IFD Scores

IFD measures instruction difficulty by semantic similarity...
```

**03-Validate.ipynb**
```markdown
# Lab 4.2: Efficient Data Filtering - Validation Experiments

**Goal:** Validate filtering effectiveness through training

**Experiments:**
- Baseline: Full dataset training
- Experiment: Filtered dataset training
- Comparison: Performance and efficiency

---

## Step 1: Prepare Training Configurations

We'll conduct controlled experiments...
```

**04-Pipeline.ipynb**
```markdown
# Lab 4.2: Efficient Data Filtering - Automated Pipeline

**Goal:** Build end-to-end data processing pipeline

**Pipeline components:**
- Data ingestion
- Quality scoring
- Filtering logic
- Incremental processing
- Quality monitoring dashboard

---

## Step 1: Define Pipeline Architecture

An automated data pipeline consists of...
```

---

### Code Cell 規範

#### 1. 評估結果解析
```python
# 解析 OpenCompass 評估結果
import json
import pandas as pd

def parse_evaluation_results(result_file):
    """解析評估結果文件"""
    with open(result_file, 'r') as f:
        results = json.load(f)

    # 提取關鍵指標
    metrics = {
        'model': results['model_name'],
        'c_eval': results['c_eval']['accuracy'],
        'cmmlu': results['cmmlu']['accuracy'],
        'mmlu': results['mmlu']['accuracy']
    }

    return metrics

# 使用範例
results = parse_evaluation_results('outputs/results.json')
print(f"✅ Evaluation results loaded: {results}")
```

#### 2. IFD 計算實現
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 載入嵌入模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def calculate_ifd(instruction, response):
    """計算指令跟隨難度"""
    # 生成語義嵌入
    instr_emb = model.encode([instruction])
    resp_emb = model.encode([response])

    # 計算餘弦相似度
    similarity = cosine_similarity(instr_emb, resp_emb)[0][0]

    # IFD = 1 - similarity (低相似度 = 高難度)
    ifd = 1 - similarity

    return ifd

# 測試範例
sample_instr = "Analyze the causes of the French Revolution"
sample_resp = "The French Revolution was caused by..."
ifd_score = calculate_ifd(sample_instr, sample_resp)
print(f"✅ IFD Score: {ifd_score:.4f}")
```

#### 3. 可視化評估結果
```python
import matplotlib.pyplot as plt
import numpy as np

# 準備數據
models = ['Llama-2-7B', 'Qwen-7B', 'ChatGLM-6B']
subjects = ['STEM', 'Social Science', 'Humanities', 'Others']

scores = np.array([
    [42.1, 48.7, 44.9, 43.2],  # Llama-2-7B
    [56.2, 62.4, 60.1, 58.7],  # Qwen-7B
    [51.3, 55.8, 53.4, 52.1]   # ChatGLM-6B
])

# 繪製雷達圖
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False).tolist()
angles += angles[:1]  # 閉合圖形

for i, model in enumerate(models):
    values = scores[i].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.15)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), subjects)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.set_title('Model Performance by Subject Category', pad=20, fontsize=14, fontweight='bold')
ax.grid(True)

plt.tight_layout()
plt.show()

print("✅ Radar chart generated")
```

---

## 🎨 視覺元素規範

### Emoji 使用指南

```markdown
## 📊 評估結果分析        # 數據分析
## 📈 性能趨勢            # 趨勢圖表
## 🎯 評估目標            # 目標聲明
## 🔍 深度分析            # 詳細分析
## 📋 評估清單            # 檢查清單
## 🏆 模型排名            # 排行榜
## 📉 性能下降            # 負面趨勢
## 🔧 數據處理            # 數據工程
## ⚡ 效率優化            # 性能提升
## 💡 評估洞察            # 關鍵發現
## ⚠️  限制與注意         # 警告
## ✅ 驗證通過            # 成功
## ❌ 評估失敗            # 錯誤
## 🔬 實驗設計            # 實驗
## 📝 報告生成            # 文檔
```

### 表格視覺化

#### 評估結果表格
```markdown
| 模型 | C-Eval | CMMLU | MMLU | 綜合 | 排名 |
|:---|---:|---:|---:|---:|:---:|
| **Qwen-7B** | **59.7%** | **58.2%** | **56.4%** | **58.1%** | 🥇 |
| ChatGLM-6B | 51.7% | 50.3% | 48.9% | 50.3% | 🥈 |
| Llama-2-7B | 45.3% | 42.1% | 46.8% | 44.7% | 🥉 |
```

#### 數據篩選效果表格
```markdown
| 方法 | 數據量 | 訓練時間 | C-Eval | 效率 |
|:---|---:|---:|---:|---:|
| **DEITA** | **15.6K** | **3.5h** | **47.1%** | **3.4x** |
| IFD only | 18.2K | 4.1h | 46.3% | 2.9x |
| Random | 52K | 12h | 45.3% | 1.0x |
```

---

## ✍️ 文字風格規範

### 語言特徵

1. **客觀中性**
   - ✅ "在 C-Eval 基準上,Qwen-7B 得分 59.7%,超過 Llama-2-7B 14.4%"
   - ❌ "Qwen-7B 遠遠超過 Llama-2-7B"

2. **數據驅動**
   - ✅ "篩選後數據訓練的模型在 C-Eval 上提升 1.8%,同時訓練時間減少 70%"
   - ❌ "篩選後的數據效果更好"

3. **結論明確**
   - ✅ "基於評估結果,Qwen-7B 在中文理解任務上具有顯著優勢"
   - ❌ "Qwen-7B 似乎表現還不錯"

4. **因果分析**
   - ✅ "由於 Qwen-7B 使用了更多中文語料,其在 C-Eval 上表現更優"
   - ❌ "Qwen-7B 比較好"

### 術語一致性

#### 評估專用術語
```
評估基準 (Benchmark)
  - C-Eval (Chinese Evaluation)
  - CMMLU (Chinese Massive Multitask Language Understanding)
  - MMLU (Massive Multitask Language Understanding)
  - HumanEval (代碼生成評估)

評估指標 (Metrics)
  - 準確率 (Accuracy)
  - F1 分數 (F1-Score)
  - 困惑度 (Perplexity)
  - 通過率 (Pass@K)

評估模式 (Evaluation Mode)
  - Zero-shot (零樣本)
  - Few-shot (少樣本)
  - Chain-of-Thought (思維鏈)
```

#### 數據工程術語
```
數據質量 (Data Quality)
  - IFD (Instruction Following Difficulty)
  - 複雜度 (Complexity)
  - 多樣性 (Diversity)
  - 一致性 (Consistency)

篩選方法 (Filtering Methods)
  - DEITA
  - LESS (Low-Effort Score Sampling)
  - MoDS (Mixture of Data Selection)
  - CaR (Context-aware Reweighting)

數據處理 (Data Processing)
  - 清洗 (Cleaning)
  - 去重 (Deduplication)
  - 採樣 (Sampling)
  - 增強 (Augmentation)
```

---

## 🔍 品質檢查清單

### README.md 檢查
- [ ] 包含所有 11 個核心章節
- [ ] 評估基準介紹清晰準確
- [ ] 數據工程方法論完整
- [ ] 提供代碼範例 (帶詳細註解)
- [ ] 包含評估結果對比表格
- [ ] 列出實戰調優策略
- [ ] 有結果解讀指南
- [ ] 包含完整參考資料
- [ ] 使用一致的 emoji 標記
- [ ] 專業術語翻譯一致

### Notebook 檢查
- [ ] 每個 code cell 前有 markdown 說明
- [ ] 評估結果解析清晰
- [ ] 可視化圖表美觀易懂
- [ ] 包含統計顯著性檢驗
- [ ] 數據篩選邏輯正確
- [ ] 驗證實驗對比公平
- [ ] 註解清晰 (英文)
- [ ] 變數命名規範
- [ ] 可重現執行 (固定隨機種子)

### 評估實驗特定檢查
- [ ] 評估協議統一 (Few-shot, 溫度等)
- [ ] 結果可重現 (固定種子)
- [ ] 錯誤案例分析深入
- [ ] 多維度結果展示
- [ ] 統計顯著性驗證

### 數據實驗特定檢查
- [ ] 篩選前後數據對比
- [ ] 多樣性保留驗證
- [ ] A/B 測試控制變量
- [ ] 訓練曲線可視化
- [ ] 效率提升量化

### 整體一致性
- [ ] README 與 notebook 內容對應
- [ ] 與理論文檔保持一致
- [ ] 與其他 Lab 風格統一
- [ ] 圖片和表格正確顯示
- [ ] 超連結有效

---

## 📖 範例參考

### 完整 README 範例
參考: `03-Model_Compression/02-Labs/Lab-3.1-Post_Training_Quantization_GPTQ/README.md`

### 完整 Notebook 範例
參考: `03-Model_Compression/02-Labs/Lab-3.1-Post_Training_Quantization_GPTQ/01-Setup.ipynb`

---

## 🛠️ 工具與資源

### 評估工具
- **OpenCompass**: https://opencompass.org.cn/
- **LM-Eval-Harness**: https://github.com/EleutherAI/lm-evaluation-harness

### 數據工程工具
- **Sentence-BERT**: https://www.sbert.net/
- **scikit-learn**: https://scikit-learn.org/

### 可視化工具
- **Matplotlib**: https://matplotlib.org/
- **Seaborn**: https://seaborn.pydata.org/
- **Plotly**: https://plotly.com/python/

### Markdown 編輯器
- **VS Code**: Markdown Preview Enhanced 插件
- **Typora**: 所見即所得編輯器
- **在線工具**: StackEdit, Dillinger

### 表格生成器
- [Tables Generator](https://www.tablesgenerator.com/markdown_tables)
- [Markdown Tables](https://tabletomarkdown.com/)

---

## 🎯 評估實驗特殊注意事項

### 1. 評估協議統一性
```python
# 固定評估配置
EVAL_CONFIG = {
    'num_fewshot': 5,
    'temperature': 0.0,      # 評估時使用貪婪解碼
    'max_length': 2048,
    'batch_size': 16,
    'seed': 42
}
```

### 2. 結果可重現性
```python
# 固定所有隨機源
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### 3. 統計顯著性檢驗
```python
from scipy import stats

# 使用 t-test 比較兩個模型
def compare_models(scores_a, scores_b, alpha=0.05):
    """比較兩個模型性能差異的顯著性"""
    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

    if p_value < alpha:
        print(f"✅ Significant difference (p={p_value:.4f})")
    else:
        print(f"⚠️  No significant difference (p={p_value:.4f})")

    return p_value
```

---

## 🔄 數據實驗特殊注意事項

### 1. 數據版本管理
```python
# 記錄數據版本與篩選參數
DATA_VERSION = {
    'source': 'alpaca_52k',
    'filtering_method': 'DEITA',
    'ifd_threshold': 0.4,
    'complexity_threshold': 3.0,
    'final_size': 15600,
    'timestamp': '2025-10-17'
}

# 保存數據版本信息
with open('data_version.json', 'w') as f:
    json.dump(DATA_VERSION, f, indent=2)
```

### 2. A/B 測試公平性
```python
# 確保對比實驗的公平性
TRAINING_CONFIG = {
    'model': 'Llama-2-7B',
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'batch_size': 16,
    'seed': 42,
    'optimizer': 'AdamW',
    'scheduler': 'linear'
}

# 僅改變數據集,其他參數保持一致
experiments = {
    'baseline': {'data': 'full_dataset', **TRAINING_CONFIG},
    'filtered': {'data': 'filtered_dataset', **TRAINING_CONFIG}
}
```

---

**維護者**: Claude Code
**最後更新**: 2025-10-17
**版本**: 1.0

---

**使用建議**:
1. 開發新 Lab 前,先閱讀本規範
2. 參考已有 Lab 的結構和風格
3. 保持與其他章節一致
4. 重視評估結果的客觀性與可重現性
5. 強調數據質量對模型性能的關鍵影響
6. 定期更新最佳實踐

**問題反饋**:
如發現規範不清楚或需要補充,請在專案中提出 issue。
