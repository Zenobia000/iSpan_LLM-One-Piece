# 第四章開發計劃：LLM 評估與數據工程

**狀態**: 規劃階段

**規劃日期**: 2025-10-17

**預計完成時間**: 2025-10-20

---

## 📋 章節概述

### 核心目標

建立完整的 LLM 評估與數據工程體系，涵蓋模型能力評估、性能基準測試、數據質量優化等關鍵環節。本章旨在幫助學員掌握：

1. **模型評估體系**：如何客觀、全面地評估 LLM 的能力與性能
2. **數據工程技術**：如何高效處理、篩選和優化訓練數據
3. **工業級實踐**：如何在生產環境中建立評估與數據管線

### 技術範圍

**評估技術**：
- 能力評估基準（C-Eval, CMMLU, MMLU, OpenCompass）
- 性能評估指標（吞吐量、延遲、TTFT、ITL）
- 多維度評估框架（準確性、安全性、效率）

**數據工程**：
- 預訓練語料處理（清洗、去重、質量篩選）
- 微調數據優化（DEITA, IFD, LESS, CaR）
- 數據質量評估與迭代

### 與其他章節的關聯

```
第一章：核心訓練技術 ────┐
                        ├─→ 第四章：評估與數據工程 ─→ 完整工程閉環
第二章：高效推理部署 ────┤      ↑ 提供評估基準
                        │      ↓ 提供優化數據
第三章：模型壓縮技術 ────┘
```

- **第一章關聯**：訓練後的模型需要評估，評估結果指導數據篩選
- **第二章關聯**：推理性能評估是部署前的關鍵驗證
- **第三章關聯**：壓縮後的模型需要評估能力保留度

---

## 🎯 學習目標

### 總體目標

掌握 LLM 評估與數據工程的完整工作流，能夠：
1. 使用業界標準工具評估模型能力與性能
2. 理解評估指標的選擇與解讀
3. 實施高效的數據篩選與優化策略
4. 建立端到端的評估與數據管線

### 具體能力

**評估能力**：
- ✅ 使用 OpenCompass 評估模型在多個基準上的表現
- ✅ 理解並計算推理性能指標（TTFT, ITL, 吞吐量）
- ✅ 分析評估結果，識別模型優勢與不足
- ✅ 選擇合適的評估基準與指標組合

**數據工程能力**：
- ✅ 實施預訓練語料的清洗與去重
- ✅ 使用 IFD (Instruction Following Difficulty) 篩選高質量指令數據
- ✅ 應用 DEITA 方法評估數據複雜度與質量
- ✅ 理解數據質量對模型性能的影響

**工程實踐能力**：
- ✅ 搭建自動化評估管線
- ✅ 設計數據處理工作流
- ✅ 優化評估效率與資源使用
- ✅ 建立評估結果可視化儀表板

---

## 📚 實驗室設計

### Lab-4.1: OpenCompass 模型評估

**目標**: 掌握使用 OpenCompass 評估 LLM 在多個基準上的能力

**技術棧**:
- OpenCompass 評估框架
- C-Eval, CMMLU, MMLU 基準
- Transformers, PyTorch
- 結果分析與可視化

**學習成果**:
- 理解主流評估基準的設計與意義
- 掌握 OpenCompass 的配置與使用
- 能夠解讀評估結果並生成報告
- 理解不同模型在不同任務上的優劣勢

**實驗設計**:

1. **01-Setup.ipynb**（環境配置與基準準備）
   - 安裝 OpenCompass 框架
   - 下載並準備評估數據集（C-Eval, CMMLU 子集）
   - 加載待評估模型（Llama-2-7B, Qwen-7B）
   - 驗證環境與依賴

2. **02-Evaluate.ipynb**（執行評估）
   - 配置評估參數（batch size, 推理模式）
   - 執行 C-Eval 評估（STEM, 社會科學, 人文學科）
   - 執行 CMMLU 評估（多任務測試）
   - 收集評估日誌與中間結果

3. **03-Analyze.ipynb**（結果分析）
   - 解析評估結果（準確率、F1、困惑度）
   - 按學科分類比較模型表現
   - 識別模型優勢與劣勢領域
   - 錯誤案例分析（為什麼模型答錯？）

4. **04-Visualize_and_Report.ipynb**（可視化與報告生成）
   - 生成雷達圖（多維度能力分佈）
   - 生成熱力圖（學科表現矩陣）
   - 生成對比柱狀圖（模型間比較）
   - 自動生成評估報告（Markdown/HTML）

**關鍵技術點**:
- OpenCompass 配置文件編寫
- 多模型並行評估
- 評估結果標準化與解析
- 統計顯著性檢驗

**預期輸出**:
```
Llama-2-7B C-Eval Results:
  Overall: 45.3%
  STEM: 42.1%
  Social Science: 48.7%
  Humanities: 44.9%

Qwen-7B C-Eval Results:
  Overall: 59.7%
  STEM: 56.2%
  Social Science: 62.4%
  Humanities: 60.1%

Qwen-7B outperforms Llama-2-7B by 14.4% overall
```

---

### Lab-4.2: 高效數據篩選與優化

**目標**: 掌握使用 IFD 和 DEITA 方法篩選高質量微調數據

**技術棧**:
- DEITA (Data-Efficient Instruction Tuning)
- IFD (Instruction Following Difficulty)
- Sentence-BERT（語義嵌入）
- K-means 聚類
- 數據質量評分系統

**學習成果**:
- 理解數據質量對模型性能的關鍵影響
- 掌握 IFD 指標的計算與應用
- 能夠使用 DEITA 方法自動化數據篩選
- 建立數據質量評估體系

**實驗設計**:

1. **01-Setup.ipynb**（數據準備與環境配置）
   - 加載原始指令數據集（Alpaca, Dolly, Self-Instruct）
   - 統計數據分佈（長度、領域、複雜度）
   - 準備質量評估模型（Sentence-BERT）
   - 設定篩選目標（保留 30% 高質量數據）

2. **02-Filter.ipynb**（數據篩選與評分）
   - **IFD 計算**：
     - 語義嵌入生成
     - 指令-響應相似度計算
     - 困難度評分（低相似度 = 高難度）
   - **DEITA 複雜度評分**：
     - 指令複雜度（條件數量、多步驟推理）
     - 響應質量（完整性、連貫性）
   - **多維度評分融合**：
     - IFD 分數（40%）
     - 複雜度分數（30%）
     - 多樣性分數（30%）
   - 應用篩選閾值，保留 Top-30% 數據

3. **03-Validate.ipynb**（篩選效果驗證）
   - 對比訓練實驗：
     - 基線模型：使用全量數據微調
     - 實驗模型：使用篩選後數據微調
   - 評估指標：
     - C-Eval 準確率
     - 訓練時間與資源消耗
     - 模型泛化能力
   - 統計分析：
     - 數據質量與模型性能的相關性
     - 資料量與性能的權衡曲線

4. **04-Pipeline.ipynb**（自動化數據管線）
   - 建立端到端數據處理管線
   - 實現增量數據篩選（處理新數據）
   - 數據版本管理（DVC 風格）
   - 質量監控儀表板（實時數據質量可視化）

**關鍵技術點**:
- 語義嵌入與相似度計算
- K-means 聚類識別數據分佈
- 多目標優化（質量 vs 多樣性）
- A/B 測試驗證篩選效果

**預期輸出**:
```
Data Filtering Results:
  Original size: 52K samples
  Filtered size: 15.6K samples (30%)

Quality Metrics:
  Avg IFD score: 0.23 → 0.41 (+78%)
  Avg complexity: 2.1 → 3.7 (+76%)
  Diversity (unique clusters): 87% → 94%

Model Performance:
  Full data (52K): C-Eval 45.3%, Training 12h
  Filtered (15.6K): C-Eval 47.1%, Training 3.5h

Efficiency: 3.4x faster training, +1.8% accuracy
```

---

## 🗂️ 文件結構

```
04-Evaluation_and_Data_Engineering/
├── DEVELOPMENT_PLAN.md          # 本文件
├── WRITING_STYLE_GUIDE.md       # 寫作風格指南
├── 01-Theory/
│   ├── 4.1-Evaluation_Benchmarks.md      # 評估基準理論
│   └── 4.2-Data_Engineering.md           # 數據工程理論
├── 02-Labs/
│   ├── Lab-4.1-Evaluate_with_OpenCompass/
│   │   ├── README.md
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
└── common_utils/                # 共享工具（複用其他章節）
    ├── evaluation_helpers.py    # 評估輔助函數
    └── data_helpers.py          # 數據處理輔助函數
```

---

## 🔧 技術實現細節

### OpenCompass 評估框架

**核心組件**:
```python
from opencompass.datasets import CEvalDataset, CMMLUDataset
from opencompass.models import HuggingFaceCausalLM
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

# 配置評估任務
config = {
    'models': ['Llama-2-7B', 'Qwen-7B'],
    'datasets': ['ceval', 'cmmlu'],
    'eval_type': 'gen',
    'batch_size': 16
}
```

**評估流程**:
1. 模型加載與配置
2. 數據集準備與分片
3. 推理執行（支持多 GPU 並行）
4. 結果聚合與標準化
5. 報告生成

### IFD (Instruction Following Difficulty)

**核心算法**:
```python
from sentence_transformers import SentenceTransformer

def calculate_ifd(instruction, response, model):
    """計算指令跟隨難度"""
    # 1. 生成嵌入
    instr_emb = model.encode(instruction)
    resp_emb = model.encode(response)

    # 2. 計算餘弦相似度
    similarity = cosine_similarity(instr_emb, resp_emb)

    # 3. IFD = 1 - similarity（低相似度 = 高難度）
    ifd = 1 - similarity

    return ifd
```

**難度等級**:
- **Low (IFD < 0.3)**: 簡單指令（"總結這段文字"）
- **Medium (0.3 ≤ IFD < 0.6)**: 中等指令（"比較 A 和 B 的優缺點"）
- **High (IFD ≥ 0.6)**: 困難指令（"分析 X 的歷史影響並預測未來趨勢"）

### DEITA 複雜度評分

**評分維度**:
```python
def calculate_deita_score(sample):
    """計算 DEITA 綜合評分"""
    # 1. 指令複雜度
    complexity = analyze_instruction_complexity(sample['instruction'])

    # 2. 響應質量
    quality = analyze_response_quality(sample['response'])

    # 3. 多樣性貢獻
    diversity = calculate_diversity_contribution(sample, dataset)

    # 4. 加權融合
    score = (
        0.4 * complexity +
        0.3 * quality +
        0.3 * diversity
    )

    return score
```

**指令複雜度特徵**:
- 條件分支數量（if-else, 多情境）
- 多步驟推理要求
- 領域知識深度
- 語言表達精確性要求

---

## 📊 評估指標體系

### 能力評估指標

#### C-Eval (Chinese Evaluation)
- **覆蓋範圍**: 52 個學科（STEM, 社會科學, 人文）
- **題型**: 多選題（4 選項）
- **難度**: 包含初中、高中、大學、專業級別
- **評分**: 準確率（% correct）

#### CMMLU (Chinese Massive Multitask Language Understanding)
- **覆蓋範圍**: 67 個任務（醫學、法律、工程、文化）
- **題型**: 多選題 + 開放式問答
- **語言**: 繁體中文 + 簡體中文
- **評分**: 準確率 + F1 分數

#### MMLU (Massive Multitask Language Understanding)
- **覆蓋範圍**: 57 個學科（英文）
- **題型**: 多選題（4 選項）
- **難度**: 涵蓋基礎到專家級別
- **評分**: 準確率

### 性能評估指標

#### 推理延遲指標
- **TTFT (Time To First Token)**: 第一個 token 生成時間
  - 衡量模型響應速度
  - 影響用戶體驗的關鍵指標
- **ITL (Inter-Token Latency)**: token 間平均延遲
  - 衡量生成流暢度
  - P50, P95, P99 百分位數
- **E2E Latency**: 端到端延遲（完整響應時間）

#### 吞吐量指標
- **TPS (Tokens Per Second)**: 每秒生成 token 數
- **QPS (Queries Per Second)**: 每秒處理請求數
- **Batch Throughput**: 批量處理吞吐量

#### 資源效率指標
- **GPU 利用率**: CUDA 核心使用率
- **記憶體使用**: VRAM 峰值與平均使用
- **能耗**: 每個 token 的能量消耗

---

## 🎓 理論補充

### 4.1 評估基準設計原則

**MMLU 範式**:
- ✅ **多領域覆蓋**: 避免單一領域偏見
- ✅ **難度分層**: 涵蓋不同知識深度
- ✅ **標準化格式**: 多選題易於自動化評估
- ✅ **人類基準**: 提供專家水平參考

**開放式評估挑戰**:
- ❌ 自動化評分困難（需要 GPT-4 作為評判）
- ❌ 答案多樣性（同一問題多個正確答案）
- ❌ 評估成本高（需要大量人工標註）

### 4.2 數據質量與模型性能關係

**核心洞察**:
> "1000 條高質量數據 > 10000 條低質量數據"

**實證研究**:
- **LIMA (Less Is More for Alignment)**:
  - 僅使用 1000 條精選數據微調 LLaMA-65B
  - 達到 GPT-4 水平對話能力的 90%

- **DEITA 研究**:
  - 篩選 6K 數據（從 52K Alpaca）
  - 在多個基準上超越全量數據訓練的模型

- **Phi 系列模型**:
  - Phi-1.5 (1.3B): 使用高質量合成數據
  - 在某些任務上超越 7B 參數模型

**數據質量維度**:
1. **複雜度**: 推理步驟、知識深度
2. **多樣性**: 領域覆蓋、任務類型
3. **準確性**: 響應正確性、格式規範性
4. **難度**: 適當挑戰性（不太簡單也不過難）

---

## 🛠️ 開發檢查清單

### Lab-4.1 開發檢查

**理論準備**:
- [ ] 閱讀 OpenCompass 官方文檔
- [ ] 理解 C-Eval, CMMLU, MMLU 基準設計
- [ ] 研究評估結果解讀方法
- [ ] 分析主流模型的評估報告

**環境準備**:
- [ ] 安裝 OpenCompass 框架
- [ ] 下載 C-Eval, CMMLU 數據集
- [ ] 準備待評估模型（Llama-2-7B, Qwen-7B）
- [ ] 驗證 GPU 資源（推薦 24GB+ VRAM）

**Notebook 開發**:
- [ ] 01-Setup.ipynb: 環境配置與數據準備
- [ ] 02-Evaluate.ipynb: 執行評估
- [ ] 03-Analyze.ipynb: 結果分析
- [ ] 04-Visualize_and_Report.ipynb: 可視化與報告

**質量檢查**:
- [ ] 所有 notebook 可獨立執行
- [ ] 評估結果可重現
- [ ] 可視化圖表清晰美觀
- [ ] 生成的報告完整準確

### Lab-4.2 開發檢查

**理論準備**:
- [ ] 閱讀 DEITA 論文（arXiv:2312.15685）
- [ ] 理解 IFD 計算方法
- [ ] 研究數據篩選的實證研究（LIMA, Phi）
- [ ] 分析數據質量評估體系

**環境準備**:
- [ ] 安裝 Sentence-BERT
- [ ] 準備指令數據集（Alpaca, Dolly）
- [ ] 配置數據處理環境（pandas, scikit-learn）
- [ ] 準備訓練資源（用於驗證篩選效果）

**Notebook 開發**:
- [ ] 01-Setup.ipynb: 數據加載與統計分析
- [ ] 02-Filter.ipynb: IFD + DEITA 篩選
- [ ] 03-Validate.ipynb: 篩選效果驗證
- [ ] 04-Pipeline.ipynb: 自動化管線

**質量檢查**:
- [ ] 篩選邏輯正確實現
- [ ] 驗證實驗對比公平（控制變量）
- [ ] 統計分析嚴謹（顯著性檢驗）
- [ ] 管線可複用且可擴展

### 文檔開發檢查

**README.md (每個 Lab)**:
- [ ] 章節完整（11 個標準章節）
- [ ] 理論深度適中（既有深度又易懂）
- [ ] 代碼示例可執行
- [ ] 參考文獻齊全

**WRITING_STYLE_GUIDE.md**:
- [ ] 定義寫作規範
- [ ] 提供模板與示例
- [ ] 明確質量標準
- [ ] 統一術語翻譯

**Theory 文件**:
- [ ] 4.1-Evaluation_Benchmarks.md: 評估基準全景
- [ ] 4.2-Data_Engineering.md: 數據工程方法論

---

## 📈 進度追蹤

### 當前階段: 規劃完成

**已完成 ✅**:
- [x] 章節整體規劃
- [x] Lab 實驗設計
- [x] 技術棧選型
- [x] 文件結構設計

**開發中 🚧**:
- [ ] 無

**待開發 📝**:
- [ ] Lab-4.1: OpenCompass 評估（4 notebooks + README）
- [ ] Lab-4.2: 高效數據篩選（4 notebooks + README）
- [ ] 理論文件（4.1, 4.2）
- [ ] 寫作風格指南

### 預計時間線

| 階段 | 任務 | 預計工時 | 完成日期 |
|------|------|----------|----------|
| 1 | Lab-4.1 README.md | 2h | 2025-10-17 |
| 2 | Lab-4.1 01-Setup.ipynb | 1.5h | 2025-10-17 |
| 3 | Lab-4.1 02-Evaluate.ipynb | 2h | 2025-10-18 |
| 4 | Lab-4.1 03-Analyze.ipynb | 1.5h | 2025-10-18 |
| 5 | Lab-4.1 04-Visualize.ipynb | 2h | 2025-10-18 |
| 6 | Lab-4.2 README.md | 2h | 2025-10-19 |
| 7 | Lab-4.2 01-Setup.ipynb | 1.5h | 2025-10-19 |
| 8 | Lab-4.2 02-Filter.ipynb | 2.5h | 2025-10-19 |
| 9 | Lab-4.2 03-Validate.ipynb | 2h | 2025-10-20 |
| 10 | Lab-4.2 04-Pipeline.ipynb | 2h | 2025-10-20 |
| 11 | 理論文件與風格指南 | 3h | 2025-10-20 |
| **總計** | | **22h** | |

---

## 🔗 參考資源

### 評估框架

**OpenCompass**:
- 官方文檔: https://opencompass.org.cn/
- GitHub: https://github.com/open-compass/opencompass
- 論文: "OpenCompass: A Universal Evaluation Platform for Foundation Models"

**評估基準**:
- C-Eval: https://cevalbenchmark.com/
- CMMLU: https://github.com/haonan-li/CMMLU
- MMLU: https://arxiv.org/abs/2009.03300

### 數據工程

**DEITA**:
- 論文: "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning" (arXiv:2312.15685)
- GitHub: https://github.com/hkust-nlp/deita

**LIMA**:
- 論文: "LIMA: Less Is More for Alignment" (arXiv:2305.11206)

**IFD**:
- 論文: "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning" (arXiv:2308.12032)

**其他重要工作**:
- LESS: "LESS: Selecting Influential Data for Targeted Instruction Tuning" (arXiv:2402.04333)
- MoDS: "Mixture of Data Selection" (arXiv:2310.xxxxx)
- CaR: "Context-aware Reweighting" (arXiv:2311.xxxxx)

### 數據質量研究

- **Phi-1.5 Technical Report**: https://arxiv.org/abs/2309.05463
- **Alpaca**: https://github.com/tatsu-lab/stanford_alpaca
- **Dolly**: https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm

---

## 💡 開發注意事項

### 評估實驗注意事項

1. **模型選擇**:
   - 優先使用 7B 參數模型（Llama-2-7B, Qwen-7B）
   - 確保模型可本地運行（避免 API 依賴）
   - 準備量化版本（GPTQ 4-bit）降低資源需求

2. **評估效率**:
   - 優先使用子集進行開發測試
   - 完整評估預留充足時間（C-Eval 約 2-3 小時）
   - 支持斷點續傳（避免重複計算）

3. **結果可重現性**:
   - 固定隨機種子
   - 記錄所有超參數
   - 保存完整評估日誌

### 數據篩選注意事項

1. **數據版權**:
   - 使用開源數據集（Alpaca, Dolly）
   - 遵守數據集許可證
   - 不使用受版權保護的數據

2. **篩選公平性**:
   - 避免引入領域偏見
   - 保持數據分佈平衡
   - 驗證多樣性保留

3. **驗證實驗**:
   - 控制變量（僅數據質量變化）
   - 使用相同訓練超參數
   - 多次實驗取平均值

### 代碼規範

1. **可重現性**:
   ```python
   import random
   import numpy as np
   import torch

   def set_seed(seed=42):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
   ```

2. **錯誤處理**:
   ```python
   try:
       result = evaluate_model(model, dataset)
   except Exception as e:
       print(f"❌ Evaluation failed: {e}")
       # 保存部分結果
       save_checkpoint(partial_result)
   ```

3. **進度追蹤**:
   ```python
   from tqdm import tqdm

   for sample in tqdm(dataset, desc="Evaluating"):
       result = model.evaluate(sample)
   ```

---

## 🎯 成功標準

### Lab-4.1 成功標準

- ✅ 成功運行 OpenCompass 評估 Llama-2-7B 和 Qwen-7B
- ✅ 在 C-Eval 至少 3 個子類別上獲得結果
- ✅ 生成清晰的對比分析圖表（雷達圖、柱狀圖）
- ✅ 自動生成完整評估報告（包含結論與建議）
- ✅ 評估結果與官方基準誤差 < 2%

### Lab-4.2 成功標準

- ✅ 實現 IFD 計算，成功評分 Alpaca 數據集
- ✅ 應用 DEITA 方法篩選出 30% 高質量數據
- ✅ 驗證實驗：篩選後數據訓練的模型性能≥基線
- ✅ 建立自動化數據管線，支持增量處理
- ✅ 生成數據質量可視化儀表板

### 整體成功標準

- ✅ 所有 notebook 可獨立運行，無依賴錯誤
- ✅ README.md 完整清晰，理論與實踐結合
- ✅ 代碼註解充分，關鍵邏輯有詳細說明
- ✅ 實驗結果可重現，提供完整復現步驟
- ✅ 遵循 WRITING_STYLE_GUIDE.md 規範

---

## 🔄 迭代計劃

### Version 1.0 (MVP)
- 核心功能實現
- 基本評估與篩選流程
- 簡單可視化

### Version 1.1 (Enhancement)
- 新增更多評估基準（HumanEval, GSM8K）
- 支持更多數據篩選方法（LESS, MoDS）
- 改進可視化儀表板

### Version 2.0 (Advanced)
- 整合自動化評估 CI/CD
- 實時數據質量監控
- 多模態數據處理支持

---

## 📝 變更日誌

| 日期 | 版本 | 變更內容 |
|------|------|----------|
| 2025-10-17 | 1.0 | 初始規劃完成 |

---

**文檔維護者**: Claude Code

**最後更新**: 2025-10-17
