# Lab 4.2: 高效數據篩選與優化

**難度**: ⭐⭐⭐⭐  
**預計時間**: 4-6 小時  
**更新日期**: 2025-10-17

---

## 📋 目錄

1. [實驗概述](#1-實驗概述)
2. [理論背景](#2-理論背景)
3. [技術架構](#3-技術架構)
4. [環境需求](#4-環境需求)
5. [快速開始](#5-快速開始)
6. [核心概念](#6-核心概念)
7. [實作流程](#7-實作流程)
8. [關鍵技術點](#8-關鍵技術點)
9. [實驗結果](#9-實驗結果)
10. [故障排除](#10-故障排除)
11. [延伸閱讀](#11-延伸閱讀)

---

## 1. 實驗概述

### 1.1 學習目標

本實驗將帶你掌握 LLM 數據工程的核心技術,特別是如何使用 **IFD** 和 **DEITA** 方法自動篩選高質量微調數據。

**核心能力**:
- ✅ 理解數據質量對模型性能的關鍵影響
- ✅ 掌握 IFD (Instruction Following Difficulty) 指標的計算與應用
- ✅ 能夠使用 DEITA 方法自動化數據篩選
- ✅ 建立端到端的數據處理管線
- ✅ 驗證數據篩選對模型性能的提升效果

### 1.2 為什麼需要數據篩選?

**經典研究案例**:

#### LIMA (Less Is More for Alignment)
- 僅用 **1,000 條**精心篩選的高質量數據
- 達到與 **52,000 條**數據訓練的模型相當的性能
- **核心洞察**: 數據質量 > 數據數量

#### DEITA 研究
- 篩選 **6K 數據**(從 52K Alpaca)
- 在多個基準上超越全量數據訓練的模型
- **訓練時間**: 減少 **85%**

#### Phi-1.5 (Microsoft Research)
- 1.3B 參數的小型模型
- 僅使用 7B tokens 的高質量「教科書級」數據
- 在推理任務上超越許多 10B+ 參數模型

**關鍵結論**:
> "1000 條高質量數據 > 10000 條低質量數據"

### 1.3 實驗設計

本實驗將通過以下步驟驗證數據質量的重要性:

1. **數據準備**: 加載 Alpaca 指令數據集 (52K 樣本)
2. **數據篩選**: 使用 IFD + DEITA 篩選出 30% 高質量數據
3. **效果驗證**: 對比全量數據 vs 篩選數據的訓練效果
4. **管線建立**: 建立自動化數據處理管線

**預期成果**:
- 數據量減少 **70%**
- 訓練時間減少 **65%**
- 模型性能提升或持平

---

## 2. 理論背景

### 2.1 數據質量的三個維度

#### 2.1.1 複雜度 (Complexity)
衡量任務的難度與推理深度。

**特徵**:
- 多步驟推理要求
- 條件分支數量
- 領域知識深度
- 語言表達精確性

**範例對比**:
```
低複雜度 (分數: 2/10):
  指令: "將 'hello' 翻譯成中文"
  回應: "你好"

高複雜度 (分數: 8/10):
  指令: "分析量子計算對密碼學的潛在影響,並討論後量子密碼的發展趨勢"
  回應: [需要多段落深入分析]
```

#### 2.1.2 多樣性 (Diversity)
衡量數據集的覆蓋廣度。

**重要性**:
- 避免模型過度擬合特定任務
- 提升泛化能力
- 確保均衡的領域覆蓋

**測量方法**:
- K-means 聚類識別數據分布
- 計算嵌入向量的相似度矩陣
- 評估任務類型分布

#### 2.1.3 準確性 (Accuracy)
衡量回應的正確性與質量。

**評估維度**:
- 事實準確性
- 邏輯一致性
- 格式規範性
- 完整性

### 2.2 IFD (Instruction Following Difficulty)

#### 核心思想
測量指令與回應之間的**語義距離**,距離越大表示任務越困難。

**數學表示**:
```
IFD = 1 - cosine_similarity(embedding(instruction), embedding(response))
```

**假設**:
- **簡單任務**: 回應與指令高度相關 (如「翻譯」、「重複」)
  - 指令: "總結這段文字"
  - 回應: [包含原文關鍵詞的總結]
  - IFD: 低 (0.1-0.3)

- **困難任務**: 回應需要推理,與指令語義距離較大 (如「分析」、「創造」)
  - 指令: "分析量子計算的未來發展"
  - 回應: [需要背景知識、多步驟推理的分析]
  - IFD: 高 (0.6-0.9)

**IFD 分佈特性**:
```
IFD < 0.3:  簡單任務 (翻譯、格式轉換) - 過濾掉
0.3 ≤ IFD < 0.6:  中等任務 (摘要、比較) - 保留
0.6 ≤ IFD < 0.9:  困難任務 (分析、推理) - 保留
IFD ≥ 0.9:  過於不相關或噪聲 - 過濾掉
```

#### 優勢與限制

**優勢**:
- ✅ 無需訓練,直接計算
- ✅ 計算效率高 (批量嵌入)
- ✅ 適用於各種語言
- ✅ 不依賴外部 API

**限制**:
- ❌ 依賴嵌入模型質量
- ❌ 無法捕捉邏輯錯誤
- ❌ 對某些任務類型不準確 (如創意寫作)

### 2.3 DEITA (Data-Efficient Instruction Tuning)

#### 綜合評分框架

DEITA 結合三個維度進行全面評估:

**評分公式**:
```
DEITA_score = α × Complexity + β × Quality + γ × Diversity

常用權重: α=0.4, β=0.4, γ=0.2
```

#### 三個維度的計算

**1. Complexity (複雜度)**:
- 使用 LLM (如 GPT-4) 評估指令複雜度
- 1-10 打分制
- 考慮推理步驟、知識深度、條件分支

```python
prompt = f"""請評估以下指令的複雜度,從 1 到 10 打分:
1-3: 簡單 (翻譯單詞、基礎計算)
4-6: 中等 (段落翻譯、多步驟推理)
7-10: 困難 (複雜分析、創意寫作)

指令: {instruction}
"""
```

**2. Quality (質量)**:
- 使用 LLM 評估回應質量
- 評估準確性、完整性、清晰度、實用性

```python
prompt = f"""請評估以下回應的質量,從 1 到 10 打分:
- 準確性: 回應是否正確
- 完整性: 是否充分回答問題
- 清晰度: 表達是否清晰
- 實用性: 是否有實際價值

指令: {instruction}
回應: {output}
"""
```

**3. Diversity (多樣性)**:
- 計算樣本與已選擇樣本的最小相似度
- 使用 Sentence-BERT 生成嵌入
- 鼓勵選擇與已有數據不同的樣本

```python
diversity = 1 - max(cosine_similarity(sample, selected_samples))
```

#### DEITA 選擇流程

```
1. 計算所有樣本的 Complexity 分數
   ↓
2. 迭代選擇樣本:
   - 計算當前樣本的 Quality 分數
   - 計算當前樣本與已選樣本的 Diversity 分數
   - 計算 DEITA 綜合分數
   ↓
3. 選擇 DEITA 分數最高的 k 個樣本
   ↓
4. 輸出篩選後的數據集
```

### 2.4 LESS (Low-Effort Score Sampling)

#### 基於梯度的數據選擇

**核心思想**:
- 對模型影響大的樣本 = 高梯度範數
- 選擇梯度範數最大的 k 個樣本

**優勢**:
- ✅ 直接基於模型訓練信號
- ✅ 不依賴外部 LLM
- ✅ 計算效率高 (單次前向+反向)

**計算流程**:
```python
for sample in dataset:
    loss = model(sample)
    loss.backward()
    gradient_norm = compute_gradient_norm(model)
    scores.append((sample, gradient_norm))

selected = top_k(scores, k)
```

---

## 3. 技術架構

### 3.1 整體流程圖

```
┌─────────────────────────────────────────────────────────────┐
│                     原始數據集 (52K)                          │
│                 (Alpaca / Dolly / ShareGPT)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              01-Setup: 數據加載與統計分析                      │
│  ✓ 加載原始數據集                                             │
│  ✓ 基礎統計 (長度、領域分布)                                   │
│  ✓ 準備評估模型 (Sentence-BERT)                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           02-Filter: IFD + DEITA 篩選                        │
│  ✓ IFD 計算 (語義距離)                                        │
│  ✓ DEITA 評分 (複雜度 + 質量 + 多樣性)                         │
│  ✓ 應用篩選閾值 (保留 Top-30%)                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                篩選後數據集 (15.6K, 30%)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              03-Validate: 篩選效果驗證                         │
│  ✓ 對比實驗 (全量 vs 篩選)                                    │
│  ✓ 訓練資源對比 (時間、GPU)                                   │
│  ✓ 評估指標對比 (C-Eval)                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          04-Pipeline: 自動化數據管線                           │
│  ✓ 端到端管線建立                                             │
│  ✓ 增量數據處理                                               │
│  ✓ 數據版本管理                                               │
│  ✓ 質量監控儀表板                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 技術棧

| 組件 | 技術 | 用途 |
|:---|:---|:---|
| **語義嵌入** | Sentence-BERT | 生成指令與回應的向量表示 |
| **相似度計算** | scikit-learn | 計算餘弦相似度 |
| **聚類分析** | K-means | 識別數據分布與多樣性 |
| **LLM 評估** | OpenAI API / 本地模型 | 評估複雜度與質量 |
| **數據處理** | pandas, numpy | 數據清洗與統計分析 |
| **可視化** | matplotlib, seaborn | 數據分布與評估結果可視化 |
| **模型訓練** | Transformers, PEFT | 驗證篩選效果 |

---

## 4. 環境需求

### 4.1 硬體需求

| 組件 | 最低配置 | 推薦配置 |
|:---|:---|:---|
| **GPU** | 無 (CPU 可執行篩選) | RTX 3090 / A100 (用於驗證訓練) |
| **VRAM** | - | 24GB+ (用於驗證訓練) |
| **RAM** | 16GB | 32GB+ |
| **磁碟空間** | 10GB | 50GB+ |

**說明**:
- 數據篩選階段 (01-02) 可在 CPU 上執行
- 驗證訓練階段 (03) 需要 GPU
- 如無 GPU,可使用預計算的驗證結果

### 4.2 軟體依賴

```bash
# 核心依賴
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# 訓練驗證 (可選)
transformers>=4.35.0
peft>=0.6.0
accelerate>=0.24.0
torch>=2.0.0

# 可視化增強
plotly>=5.17.0
ipywidgets>=8.1.0
```

### 4.3 安裝指令

```bash
# 方法 1: 使用 Poetry (推薦)
cd 00-Course_Setup/
poetry install --all-extras

# 方法 2: 使用 pip
pip install sentence-transformers scikit-learn pandas numpy \
            matplotlib seaborn transformers peft accelerate
```

---

## 5. 快速開始

### 5.1 完整流程 (4-6 小時)

```bash
# 1. 啟動 Jupyter Lab
cd 04-Evaluation_and_Data_Engineering/02-Labs/Lab-4.2-Efficient_Data_Filtering/
jupyter lab

# 2. 按順序執行 Notebooks
# 01-Setup.ipynb        (30-45 分鐘)
# 02-Filter.ipynb       (1-1.5 小時)
# 03-Validate.ipynb     (2-3 小時) - 需要 GPU
# 04-Pipeline.ipynb     (30-45 分鐘)
```

### 5.2 快速測試 (僅數據篩選,30 分鐘)

如果沒有 GPU 或時間有限,可以只執行數據篩選部分:

```bash
# 執行 01-Setup.ipynb 和 02-Filter.ipynb
# 跳過 03-Validate.ipynb (訓練驗證)
# 查看 02-Filter.ipynb 的篩選結果
```

### 5.3 預期輸出

完成所有 notebooks 後,你將獲得:

```
Lab-4.2-Efficient_Data_Filtering/
├── data/
│   ├── alpaca_raw.json              (52K 原始數據)
│   └── alpaca_filtered.json         (15.6K 篩選後數據)
├── analysis/
│   ├── ifd_distribution.png         (IFD 分布圖)
│   ├── deita_scores.csv             (DEITA 評分表)
│   ├── quality_comparison.png       (質量對比圖)
│   └── filtering_summary.md         (篩選摘要報告)
├── validation/
│   ├── training_logs/               (訓練日誌)
│   ├── full_data_results.json       (全量數據結果)
│   ├── filtered_data_results.json   (篩選數據結果)
│   └── comparison_report.md         (對比報告)
└── pipeline/
    ├── data_pipeline.py             (自動化管線代碼)
    └── quality_dashboard.html       (質量監控儀表板)
```

---

## 6. 核心概念

### 6.1 IFD 計算範例

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 初始化嵌入模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 範例 1: 簡單任務 (低 IFD)
instruction_1 = "將 'hello' 翻譯成中文"
response_1 = "你好"

emb_i1 = model.encode([instruction_1])
emb_r1 = model.encode([response_1])
similarity_1 = cosine_similarity(emb_i1, emb_r1)[0][0]
ifd_1 = 1 - similarity_1

print(f"簡單任務 IFD: {ifd_1:.3f}")  # 預期: 0.2-0.4

# 範例 2: 困難任務 (高 IFD)
instruction_2 = "分析量子計算對現代密碼學的影響"
response_2 = """量子計算的發展對現代密碼學構成了重大挑戰...
[詳細分析內容]"""

emb_i2 = model.encode([instruction_2])
emb_r2 = model.encode([response_2])
similarity_2 = cosine_similarity(emb_i2, emb_r2)[0][0]
ifd_2 = 1 - similarity_2

print(f"困難任務 IFD: {ifd_2:.3f}")  # 預期: 0.6-0.8
```

### 6.2 DEITA 評分範例

```python
def calculate_deita_score(sample, reference_samples, alpha=0.4, beta=0.4, gamma=0.2):
    """
    計算 DEITA 綜合評分
    
    Args:
        sample: 當前樣本
        reference_samples: 已選擇的參考樣本
        alpha, beta, gamma: 權重參數
    
    Returns:
        DEITA 分數 (0-1)
    """
    # 1. 複雜度評分 (使用 LLM 或規則)
    complexity = evaluate_complexity(sample['instruction'])
    
    # 2. 質量評分 (使用 LLM 或規則)
    quality = evaluate_quality(sample['instruction'], sample['output'])
    
    # 3. 多樣性評分 (與已選樣本的最小相似度)
    if not reference_samples:
        diversity = 1.0
    else:
        sample_emb = model.encode([sample['instruction'] + ' ' + sample['output']])
        ref_embs = model.encode([
            s['instruction'] + ' ' + s['output'] for s in reference_samples
        ])
        similarities = cosine_similarity(sample_emb, ref_embs)[0]
        diversity = 1 - np.max(similarities)
    
    # 4. 加權求和
    deita_score = alpha * complexity + beta * quality + gamma * diversity
    
    return deita_score
```

### 6.3 數據篩選決策樹

```
開始評估樣本
    │
    ├─ 計算 IFD 分數
    │    │
    │    ├─ IFD < 0.3? → 太簡單,過濾掉
    │    ├─ IFD > 0.9? → 不相關,過濾掉
    │    └─ 0.3 ≤ IFD ≤ 0.9? → 繼續
    │
    ├─ 計算 DEITA 分數
    │    │
    │    ├─ 複雜度 < 3? → 過濾掉
    │    ├─ 質量 < 5? → 過濾掉
    │    └─ 分數 ≥ 閾值? → 保留
    │
    └─ 檢查多樣性
         │
         ├─ 與已選樣本相似度 > 0.9? → 過濾掉 (重複)
         └─ 否則 → 加入篩選集
```

---

## 7. 實作流程

### 7.1 Notebook 01: 數據準備與環境配置

**目標**: 加載原始數據集,進行統計分析,準備評估模型

**核心任務**:
1. 加載 Alpaca 數據集 (52K 樣本)
2. 基礎統計分析:
   - 指令長度分布
   - 回應長度分布
   - 任務類型分布
3. 準備 Sentence-BERT 模型
4. 設定篩選目標 (保留 30%)

**預期輸出**:
```
數據統計:
  總樣本數: 52,002
  平均指令長度: 15.3 詞
  平均回應長度: 78.4 詞
  任務類型: 
    - 開放式 QA: 35%
    - 創意寫作: 25%
    - 分類/標註: 20%
    - 其他: 20%
```

### 7.2 Notebook 02: 數據篩選與評分

**目標**: 使用 IFD 和 DEITA 方法篩選高質量數據

**核心任務**:

#### 步驟 1: IFD 計算
```python
# 批量計算 IFD
ifd_calculator = IFDCalculator()
samples_with_ifd = ifd_calculator.calculate_batch_ifd(raw_data)

# IFD 過濾 (保留 0.3-0.9)
ifd_filtered = [
    s for s, ifd in samples_with_ifd 
    if 0.3 <= ifd <= 0.9
]

print(f"IFD 過濾: {len(raw_data)} → {len(ifd_filtered)}")
# 預期: 52,002 → 41,000 (過濾掉 21%)
```

#### 步驟 2: DEITA 評分
```python
# 計算 DEITA 分數
deita_scorer = DEITAScorer(
    alpha=0.4,  # 複雜度權重
    beta=0.4,   # 質量權重
    gamma=0.2   # 多樣性權重
)

scored_samples = []
for sample in ifd_filtered:
    score = deita_scorer.score_sample(sample, scored_samples)
    scored_samples.append(score)

# 排序並選擇 Top-30%
scored_samples.sort(key=lambda x: x['deita_score'], reverse=True)
final_filtered = scored_samples[:int(len(scored_samples) * 0.3)]

print(f"DEITA 篩選: {len(ifd_filtered)} → {len(final_filtered)}")
# 預期: 41,000 → 15,600 (保留 30%)
```

**預期輸出**:
```
篩選結果:
  原始數據: 52,002 樣本
  IFD 過濾後: 41,000 樣本 (過濾 21%)
  DEITA 篩選後: 15,600 樣本 (保留 30%)

質量提升:
  平均 IFD: 0.23 → 0.41 (+78%)
  平均複雜度: 2.1 → 3.7 (+76%)
  多樣性 (聚類數): 87% → 94%
```

### 7.3 Notebook 03: 篩選效果驗證

**目標**: 通過對比訓練實驗驗證篩選效果

**核心任務**:

#### 實驗設計
```
基線實驗 (Baseline):
  數據: 全量 52K
  模型: Llama-2-7B
  訓練輪數: 3 epochs
  評估: C-Eval

對比實驗 (Filtered):
  數據: 篩選後 15.6K (30%)
  模型: Llama-2-7B (相同初始權重)
  訓練輪數: 3 epochs
  評估: C-Eval (相同測試集)
```

#### 對比維度
1. **模型性能**: C-Eval 準確率
2. **訓練效率**: 訓練時間、GPU 利用率
3. **收斂速度**: Loss 曲線
4. **泛化能力**: 驗證集性能

**預期輸出**:
```
對比結果:

模型性能:
  全量數據 (52K): C-Eval 45.3%
  篩選數據 (15.6K): C-Eval 47.1% (+1.8%)

訓練效率:
  全量數據: 12 小時, GPU 90%
  篩選數據: 3.5 小時, GPU 88%
  效率提升: 3.4x 更快

結論:
  ✅ 數據量減少 70%
  ✅ 訓練時間減少 71%
  ✅ 模型性能提升 1.8%
```

### 7.4 Notebook 04: 自動化數據管線

**目標**: 建立端到端的自動化數據處理管線

**核心任務**:

#### 管線設計
```python
class DataFilteringPipeline:
    def __init__(self, config):
        self.ifd_calculator = IFDCalculator()
        self.deita_scorer = DEITAScorer()
        self.config = config
    
    def run(self, input_data):
        # 1. IFD 過濾
        ifd_filtered = self.ifd_filter(input_data)
        
        # 2. DEITA 評分
        scored = self.deita_score(ifd_filtered)
        
        # 3. Top-K 選擇
        selected = self.select_top_k(scored, k=self.config['target_size'])
        
        # 4. 保存結果
        self.save_results(selected)
        
        return selected
```

#### 增量處理
```python
# 處理新增數據
new_data = load_new_data()
existing_data = load_filtered_data()

# 僅對新數據進行篩選
new_filtered = pipeline.run(new_data)

# 合併並重新評估多樣性
combined = combine_and_rerank(existing_data, new_filtered)
```

---

## 8. 關鍵技術點

### 8.1 嵌入模型選擇

| 模型 | 參數量 | 速度 | 質量 | 推薦場景 |
|:---|---:|:---:|:---:|:---|
| `all-MiniLM-L6-v2` | 22M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 快速實驗 |
| `all-mpnet-base-v2` | 110M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 平衡選擇 (推薦) |
| `text-embedding-ada-002` | - | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 最佳質量 (需 API) |

**推薦**: `all-mpnet-base-v2` (質量與速度平衡)

### 8.2 批量處理優化

```python
# ✅ 好的做法: 批量編碼
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

# ❌ 不好的做法: 逐個編碼
embeddings = [model.encode(text) for text in texts]  # 慢 10-50 倍
```

### 8.3 記憶體優化

```python
# 分批處理大型數據集
def process_large_dataset(dataset, batch_size=1000):
    results = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)
        
        # 釋放記憶體
        del batch_results
        gc.collect()
    
    return results
```

### 8.4 多樣性保證

使用**貪婪選擇演算法**確保多樣性:

```python
def greedy_selection(scored_samples, k):
    """
    貪婪選擇演算法,確保多樣性
    
    Args:
        scored_samples: 已評分的樣本列表
        k: 目標數量
    
    Returns:
        選擇的樣本列表
    """
    selected = []
    
    # 1. 選擇分數最高的第一個樣本
    selected.append(scored_samples[0])
    remaining = scored_samples[1:]
    
    # 2. 迭代選擇
    while len(selected) < k and remaining:
        # 重新計算每個候選樣本的多樣性分數
        for sample in remaining:
            diversity = calculate_diversity(sample, selected)
            sample['final_score'] = (
                0.6 * sample['deita_score'] + 
                0.4 * diversity
            )
        
        # 選擇分數最高的
        remaining.sort(key=lambda x: x['final_score'], reverse=True)
        selected.append(remaining[0])
        remaining = remaining[1:]
    
    return selected
```

---

## 9. 實驗結果

### 9.1 預期結果

基於 DEITA 論文的結果,我們預期:

| 指標 | 全量數據 (52K) | 篩選數據 (15.6K) | 變化 |
|:---|---:|---:|:---:|
| **數據量** | 52,002 | 15,600 | -70% |
| **平均 IFD** | 0.23 | 0.41 | +78% |
| **平均複雜度** | 2.1 | 3.7 | +76% |
| **多樣性** | 87% | 94% | +7% |
| **C-Eval 準確率** | 45.3% | 47.1% | +1.8% |
| **訓練時間** | 12h | 3.5h | -71% |
| **GPU 記憶體** | 22GB | 22GB | - |

### 9.2 關鍵觀察

**數據質量提升**:
- IFD 分數顯著提高 (+78%),表示保留了更困難的任務
- 複雜度提升 (+76%),過濾掉簡單重複的任務
- 多樣性改善 (+7%),確保領域覆蓋

**訓練效率**:
- 訓練時間減少 71%,大幅提升迭代速度
- 模型性能反而提升 1.8%,驗證「質量 > 數量」

**成本效益**:
- 每次訓練成本降低 70%
- 實驗迭代速度提升 3.4 倍
- ROI 顯著提升

---

## 10. 故障排除

### 10.1 常見問題

#### Q1: Sentence-BERT 記憶體不足

**症狀**: `CUDA out of memory` 或 `RuntimeError: out of memory`

**解決方案**:
```python
# 方法 1: 減小批次大小
embeddings = model.encode(texts, batch_size=32)  # 降低至 32 或 16

# 方法 2: 使用 CPU
model = SentenceTransformer('all-mpnet-base-v2', device='cpu')

# 方法 3: 使用更小的模型
model = SentenceTransformer('all-MiniLM-L6-v2')  # 22M 參數
```

#### Q2: DEITA 評分速度慢

**症狀**: 處理 52K 樣本需要超過 10 小時

**解決方案**:
```python
# 方法 1: 不使用 LLM,僅使用規則評分
def rule_based_complexity(instruction):
    # 基於長度、關鍵詞等規則
    score = calculate_rule_score(instruction)
    return score

# 方法 2: 使用本地模型而非 API
local_model = AutoModelForCausalLM.from_pretrained("Qwen-7B")

# 方法 3: 先用 IFD 過濾,減少需要評分的樣本
ifd_filtered = ifd_filter(raw_data, min_ifd=0.3, max_ifd=0.9)
deita_scored = deita_score(ifd_filtered)  # 數量減少 ~20%
```

#### Q3: 訓練驗證實驗無 GPU

**症狀**: 沒有 GPU 無法執行 03-Validate.ipynb

**解決方案**:
```python
# 選項 1: 使用預計算的結果
# 下載預計算的驗證結果
!wget https://example.com/validation_results.json

# 選項 2: 使用更小的模型 (CPU 可行)
model_name = "EleutherAI/pythia-160m"  # 160M 參數,CPU 可訓練

# 選項 3: 跳過驗證,查看 Lab 提供的結果摘要
```

### 10.2 除錯技巧

#### 逐步檢查數據質量
```python
# 檢查 IFD 分布
import matplotlib.pyplot as plt

ifd_scores = [s['ifd'] for s in samples_with_ifd]
plt.hist(ifd_scores, bins=50)
plt.xlabel('IFD Score')
plt.ylabel('Count')
plt.title('IFD Distribution')
plt.show()

# 預期: 正態分布,峰值在 0.4-0.6
```

#### 檢查篩選邏輯
```python
# 打印篩選統計
print(f"原始數據: {len(raw_data)}")
print(f"IFD < 0.3 (過濾): {sum(1 for s, ifd in samples_with_ifd if ifd < 0.3)}")
print(f"IFD > 0.9 (過濾): {sum(1 for s, ifd in samples_with_ifd if ifd > 0.9)}")
print(f"0.3 ≤ IFD ≤ 0.9 (保留): {len(ifd_filtered)}")
```

---

## 11. 延伸閱讀

### 11.1 核心論文

**DEITA**:
- 標題: "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning"
- 作者: Liu et al. (HKUST)
- arXiv: [2312.15685](https://arxiv.org/abs/2312.15685)
- GitHub: [https://github.com/hkust-nlp/deita](https://github.com/hkust-nlp/deita)

**LIMA**:
- 標題: "LIMA: Less Is More for Alignment"
- 作者: Zhou et al. (Meta AI)
- arXiv: [2305.11206](https://arxiv.org/abs/2305.11206)

**IFD**:
- 標題: "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning"
- arXiv: [2308.12032](https://arxiv.org/abs/2308.12032)

**LESS**:
- 標題: "LESS: Selecting Influential Data for Targeted Instruction Tuning"
- arXiv: [2402.04333](https://arxiv.org/abs/2402.04333)

### 11.2 相關資源

**數據集**:
- Alpaca: [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- Dolly: [https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- ShareGPT: [https://sharegpt.com/](https://sharegpt.com/)

**工具與框架**:
- Sentence-Transformers: [https://www.sbert.net/](https://www.sbert.net/)
- OpenCompass: [https://opencompass.org.cn/](https://opencompass.org.cn/)

### 11.3 進階主題

**多模態數據篩選**:
- 圖文指令數據的質量評估
- 跨模態相似度計算

**強化學習數據選擇**:
- 使用 RL 動態調整篩選策略
- 在線學習與數據選擇

**聯邦學習場景**:
- 分散式數據篩選
- 隱私保護的數據質量評估

---

## 📝 總結

本實驗將帶你完整掌握 LLM 數據工程的核心技術:

1. **理論理解**: IFD、DEITA、LESS 的原理與應用
2. **實作能力**: 從數據篩選到驗證的完整流程
3. **工程經驗**: 自動化管線建立與優化技巧
4. **實驗分析**: 數據質量對模型性能的量化影響

**核心洞察**:
> "在 LLM 訓練中,數據質量往往比數量更重要。通過科學的數據篩選方法,我們可以用 30% 的數據達到甚至超越全量數據的訓練效果,同時大幅降低計算成本。"

**下一步**:
- 嘗試不同的篩選閾值 (10%, 20%, 40%)
- 應用到自己的數據集
- 探索其他數據選擇方法 (LESS, MoDS)
- 建立生產級數據管線

---

**祝學習順利!** 🎉

如有問題,請參考:
- [理論文件: 4.2-Data_Engineering.md](../../01-Theory/4.2-Data_Engineering.md)
- [故障排除](#10-故障排除)
- [GitHub Issues](https://github.com/Zenobia000/iSpan_LLM-One-Piece/issues)
