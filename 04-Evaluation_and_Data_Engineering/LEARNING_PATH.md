# 評估與數據工程認知建構學習路徑
# (Evaluation and Data Engineering Cognitive Construction Learning Path)

## 模組元資料 (Module Metadata)

```json
{
  "id": "evaluation-data-engineering",
  "title": "評估與數據工程 (Evaluation and Data Engineering)",
  "category": "llm-quality-assurance",
  "difficulty_levels": {
    "elementary": 2,
    "intermediate": 3,
    "advanced": 4,
    "research": 5
  },
  "estimated_time": {
    "reading": "8hours",
    "practice": "20hours",
    "mastery": "40hours"
  },
  "tags": ["evaluation", "benchmarking", "data-engineering", "quality-assessment", "opencompass"],
  "version": "2.0",
  "last_updated": "2025-01-27"
}
```

---

## 認知建構路徑 (Cognitive Construction Path)

### Level 1: 直覺層 (Intuitive Level) - 理解評估與數據的根本價值

**目標**：為什麼評估和數據工程是 LLM 工程的基石？建立品質保證的直覺認知

#### 核心問題
- 如何判斷一個 LLM 是「好」的？
- 為什麼「垃圾進，垃圾出」在 LLM 中特別重要？
- 評估基準如何避免「考試導向」的問題？

#### 直覺理解
```
LLM 的評估與數據工程就像醫學檢查：
1. 評估基準 = 不同的檢查項目（血壓、心電圖、X光）
2. 多維評估 = 全身健康檢查（不只看一個指標）
3. 數據工程 = 體檢前的準備（空腹、充足睡眠）
4. 質量控制 = 檢查設備的校準（確保結果可信）
5. 基準更新 = 醫學標準的進步（與時俱進）

核心洞察：沒有評估就沒有改進的方向，沒有好數據就沒有好模型
```

#### 評估維度的直觀理解
```python
class LLMCapabilityMap:
    def __init__(self):
        self.core_abilities = {
            "理解能力": {
                "閱讀理解": "理解文本含義",
                "邏輯推理": "從前提得出結論",
                "常識判斷": "運用日常常識"
            },
            "生成能力": {
                "創意寫作": "原創內容創作",
                "文本摘要": "提取關鍵資訊",
                "對話交流": "自然對話互動"
            },
            "專業能力": {
                "數學計算": "數值推理與計算",
                "程式編寫": "代碼生成與除錯",
                "多語言": "跨語言理解與翻譯"
            },
            "實用能力": {
                "指令跟隨": "準確執行指令",
                "安全對齊": "避免有害輸出",
                "事實準確": "提供正確資訊"
            }
        }
```

#### 數據品質的直觀認知
```python
class DataQualityIntuition:
    def __init__(self):
        self.quality_aspects = {
            "準確性": "資訊是否正確",
            "完整性": "資訊是否充分",
            "一致性": "資訊是否矛盾",
            "時效性": "資訊是否過時",
            "相關性": "資訊是否有用",
            "多樣性": "是否涵蓋各種情況"
        }

        self.quality_impact = {
            "低質量數據": "模型能力上限低",
            "高質量數據": "事半功倍的學習效果",
            "數據偏見": "模型輸出偏見",
            "數據重複": "浪費計算資源"
        }
```

#### 視覺化輔助
- 不同評估基準的能力雷達圖
- 數據質量對模型性能的影響曲線
- 評估分數與實際應用能力的相關性

#### 自我驗證問題
1. 為什麼單一指標無法全面評估 LLM？
2. 數據量和數據質量哪個更重要？
3. 如何設計一個「作弊難度高」的評估基準？

---

### Level 2: 概念層 (Conceptual Level) - 掌握評估體系與數據工程技術

**目標**：理解主流評估框架和數據處理技術的工作原理

#### 關鍵概念架構

##### 2.1 評估基準分類體系
```python
class EvaluationTaxonomy:
    def __init__(self):
        self.by_capability = {
            "語言理解": {
                "基準": ["GLUE", "SuperGLUE", "CLUE", "C-Eval"],
                "任務": ["閱讀理解", "自然語言推理", "情感分析"],
                "指標": ["準確率", "F1分數", "Matthews相關係數"]
            },
            "知識推理": {
                "基準": ["MMLU", "GSM8K", "MATH", "CommonsenseQA"],
                "任務": ["常識推理", "數學問題", "科學知識"],
                "指標": ["準確率", "Pass@K"]
            },
            "代碼能力": {
                "基準": ["HumanEval", "MBPP", "CodeXGLUE"],
                "任務": ["代碼生成", "代碼理解", "bug修復"],
                "指標": ["Pass@1", "Pass@10", "編譯通過率"]
            }
        }

        self.by_paradigm = {
            "判別式評估": "選擇正確答案",
            "生成式評估": "生成完整回答",
            "互動式評估": "多輪對話評估",
            "對抗式評估": "抗攻擊能力測試"
        }

        self.evaluation_frameworks = {
            "OpenCompass": "開源全面評估平台",
            "EleutherAI": "Language Model Evaluation Harness",
            "BigBench": "Beyond the Imitation Game",
            "HELM": "Holistic Evaluation of Language Models"
        }
```

##### 2.2 數據工程技術棧
```python
class DataEngineeringStack:
    def __init__(self):
        self.preprocessing = {
            "數據清洗": {
                "去噪": "移除無意義字符",
                "格式標準化": "統一文本格式",
                "編碼處理": "處理特殊字符",
                "語言檢測": "識別文本語言"
            },
            "去重技術": {
                "精確去重": "完全相同文本",
                "近似去重": "MinHash, SimHash",
                "語義去重": "基於嵌入的相似性",
                "跨語言去重": "多語言重複檢測"
            }
        }

        self.quality_assessment = {
            "自動評估": {
                "困惑度過濾": "語言模型困惑度",
                "分類器評分": "品質分類模型",
                "統計特徵": "長度、複雜度統計",
                "毒性檢測": "有害內容識別"
            },
            "人工評估": {
                "專家標註": "領域專家評估",
                "眾包標註": "大規模人工評估",
                "一致性檢查": "標註者間一致性",
                "品質控制": "標註品質保證"
            }
        }

        self.selection_strategies = {
            "IFD": "Instruction Following Difficulty",
            "DEITA": "Data-Efficient Instruction Tuning",
            "LESS": "LEss Selection Strategy",
            "Self_Instruct": "自指導數據生成",
            "Evol_Instruct": "進化式指令複雜化"
        }
```

##### 2.3 評估可信度框架
```python
class EvaluationReliability:
    def __init__(self):
        self.validity = {
            "內容效度": "評估內容是否代表目標能力",
            "建構效度": "評估是否測量預期建構",
            "預測效度": "評估結果是否預測實際表現",
            "面向效度": "評估表面上是否合理"
        }

        self.reliability = {
            "內部一致性": "測試項目間的相關性",
            "重測信度": "重複評估的一致性",
            "評估者信度": "不同評估者的一致性",
            "標準誤差": "測量誤差的估計"
        }

        self.bias_sources = {
            "選擇偏見": "基準樣本不代表總體",
            "確認偏見": "傾向於支持預期結果",
            "文化偏見": "特定文化背景的偏向",
            "時間偏見": "基準隨時間過時"
        }
```

#### 技術選擇決策框架
```
評估目標分析:
模型選型 → 綜合基準評估 (MMLU + HumanEval + GSM8K)
能力分析 → 細分領域基準 (具體任務評估)
研究發表 → 標準化基準 (可比較結果)
產品部署 → 實際場景評估 (用戶測試)

數據處理策略:
預訓練數據 → 大規模清洗 + 去重 + 困惑度過濾
微調數據 → 高質量篩選 + 多樣性確保 + 人工審核
對齊數據 → 偏好標註 + 安全性檢查 + 價值觀對齊
```

#### 理解驗證問題
1. OpenCompass 相比其他評估框架的優勢？
2. IFD 與 DEITA 數據選擇策略的差異？
3. 如何設計一個抗「數據污染」的評估基準？

---

### Level 3: 形式化層 (Formalization Level) - 數學原理與評估統計學

**目標**：掌握評估的統計學基礎和數據工程的數學原理

#### 3.1 評估統計學基礎

**信度分析 (Reliability Analysis)**：
$$\alpha = \frac{k}{k-1} \left(1 - \frac{\sum_{i=1}^k \sigma_{Y_i}^2}{\sigma_X^2}\right)$$

其中 $\alpha$ 是 Cronbach's alpha 係數，$k$ 是測試項目數

**效度分析 (Validity Analysis)**：
$$r_{xy} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

**標準測量誤差**：
$$SEM = \sigma_X \sqrt{1 - r_{xx}}$$

其中 $r_{xx}$ 是信度係數

#### 3.2 數據品質度量數學模型

**數據多樣性測量**：
$$H(X) = -\sum_{i=1}^n p(x_i) \log p(x_i)$$（Shannon 熵）

**數據重複度計算**：
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$（Jaccard 相似度）

**品質評估函數**：
$$Q(D) = \alpha \cdot Accuracy(D) + \beta \cdot Diversity(D) + \gamma \cdot Novelty(D)$$

#### 3.3 評估結果的統計推論

**信賴區間計算**：
$$CI = \bar{x} \pm t_{\alpha/2, df} \cdot \frac{s}{\sqrt{n}}$$

**效應大小 (Effect Size)**：
$$d = \frac{\bar{x_1} - \bar{x_2}}{s_{pooled}}$$（Cohen's d）

**顯著性檢定**：
$$t = \frac{\bar{x_1} - \bar{x_2}}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

#### 3.4 數據選擇最佳化數學框架

**資訊增益最大化**：
$$IG(D, A) = H(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} H(D_v)$$

**困惑度閾值最佳化**：
$$PPL(x) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log p(x_i|x_{<i})\right)$$

**多目標最佳化** (Pareto Front)：
$$\min_{x \in X} (f_1(x), f_2(x), ..., f_k(x))$$

#### 複雜度分析
```python
def evaluation_complexity():
    return {
        "基準評估": "O(N × T)",      # N: 樣本數, T: 每樣本處理時間
        "困惑度計算": "O(N × L)",    # L: 序列長度
        "去重算法": "O(N²)",         # 兩兩比較
        "語義相似度": "O(N × d)",    # d: 嵌入維度
        "品質評估": "O(N × M)",      # M: 評估指標數
    }
```

#### 形式化驗證問題
1. 如何計算評估基準的最小樣本量？
2. 推導數據選擇的資訊理論最優解
3. 分析評估結果的統計顯著性

---

### Level 4: 理論層 (Theoretical Level) - 評估科學與數據理論

**目標**：理解評估的科學原理和數據工程的理論基礎

#### 4.1 測量理論在 LLM 評估中的應用

##### 經典測試理論 (Classical Test Theory)
```python
class ClassicalTestTheory:
    def __init__(self):
        self.core_equation = "X = T + E"  # 觀測分數 = 真分數 + 誤差

        self.assumptions = {
            "真分數恆定": "個體的真實能力是固定的",
            "誤差隨機": "測量誤差是隨機的",
            "誤差無關": "誤差與真分數無關",
            "誤差獨立": "不同測試的誤差無關"
        }

        self.applications = {
            "LLM能力測量": "語言能力的量化評估",
            "基準設計": "測試項目的選擇與權重",
            "分數解釋": "評估結果的意義解讀",
            "誤差估計": "評估不確定性的量化"
        }
```

##### 項目反應理論 (Item Response Theory)
```python
class ItemResponseTheory:
    def __init__(self):
        self.models = {
            "1PL": "Rasch模型，只考慮項目難度",
            "2PL": "考慮項目難度和區分度",
            "3PL": "增加猜測參數",
            "4PL": "增加上限參數"
        }

        self.item_parameters = {
            "difficulty": "項目難度參數 b",
            "discrimination": "區分度參數 a",
            "guessing": "猜測參數 c",
            "upper_asymptote": "上限參數 d"
        }
```

#### 4.2 資訊理論在數據工程中的應用

##### 數據值的資訊理論量化
```python
class DataValueTheory:
    def __init__(self):
        self.information_measures = {
            "互資訊": "I(X;Y) = H(X) - H(X|Y)",
            "條件熵": "H(X|Y) = H(X,Y) - H(Y)",
            "相對熵": "D_KL(P||Q) = ∑P(x)log(P(x)/Q(x))",
            "交叉熵": "H(P,Q) = -∑P(x)logQ(x)"
        }

        self.data_selection_principles = {
            "最大熵原理": "選擇資訊熵最大的數據",
            "最小冗餘": "避免重複資訊",
            "最大覆蓋": "涵蓋所有重要模式",
            "平衡採樣": "保持數據分佈平衡"
        }
```

##### 學習理論中的數據複雜度
- **PAC 學習框架**：Probably Approximately Correct Learning
- **VC 維理論**：Vapnik-Chervonenkis Dimension
- **Rademacher 複雜度**：泛化誤差的上界估計

#### 4.3 評估的認知科學基礎

##### 能力建構理論
```python
class AbilityConstructTheory:
    def __init__(self):
        self.cognitive_architectures = {
            "流體智力": "抽象推理和問題解決",
            "結晶智力": "知識和經驗的積累",
            "工作記憶": "暫時儲存和處理資訊",
            "長期記憶": "知識的永久儲存"
        }

        self.transfer_theory = {
            "近遷移": "相似情境間的能力轉移",
            "遠遷移": "不同領域間的能力轉移",
            "元認知": "對認知過程的認知",
            "策略知識": "解決問題的策略"
        }
```

##### 多元智力理論在 LLM 評估中的啟示
- **語言智力**：詞彙、語法、語用能力
- **邏輯數學智力**：抽象推理、數量關係
- **空間智力**：圖像理解、空間關係
- **人際智力**：社交互動、情感理解

#### 4.4 評估倫理與公平性理論

##### 算法公平性的數學定義
```python
class AlgorithmicFairness:
    def __init__(self):
        self.fairness_metrics = {
            "統計平等": "P(Ŷ=1|A=0) = P(Ŷ=1|A=1)",
            "機會平等": "P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)",
            "校準性": "P(Y=1|Ŷ=s,A=a) = s for all s,a",
            "個體公平": "相似個體得到相似結果"
        }

        self.bias_sources = {
            "歷史偏見": "訓練數據中的社會偏見",
            "代表性偏見": "某些群體代表不足",
            "測量偏見": "評估指標的系統性偏差",
            "評估偏見": "評估過程中的主觀偏見"
        }
```

#### 理論探索問題
1. LLM 能力的認知結構是什麼？
2. 評估基準的理論效度如何建立？
3. 數據工程的資訊理論最佳化原理？

---

### Level 5: 創新層 (Innovative Level) - 前沿評估技術與數據科學

**目標**：掌握最新評估方法，具備創新評估體系設計能力

#### 5.1 下一代評估範式

##### 動態評估與適應性測試
```python
class AdaptiveEvaluation:
    def __init__(self):
        self.adaptive_testing = {
            "計算機適應性測試": "根據回答調整題目難度",
            "動態基準": "根據模型能力調整評估內容",
            "個性化評估": "針對特定應用場景的評估",
            "進化式基準": "隨模型發展而進化的基準"
        }

        self.dynamic_benchmarks = {
            "持續學習評估": "評估模型的學習能力",
            "少樣本適應": "評估快速適應新任務的能力",
            "遷移學習": "評估跨域遷移能力",
            "元學習": "評估學會學習的能力"
        }
```

##### 多模態整合評估
```python
class MultimodalEvaluation:
    def __init__(self):
        self.modality_integration = {
            "視覺語言": "圖像理解與描述",
            "音頻語言": "語音理解與生成",
            "感知推理": "多感官資訊整合",
            "具身智能": "物理世界互動能力"
        }

        self.complex_reasoning = {
            "因果推理": "理解因果關係",
            "反事實推理": "假設情景分析",
            "類比推理": "抽象模式識別",
            "創造性推理": "原創性思維評估"
        }
```

#### 5.2 智能數據工程

##### 自主數據策劃 (Autonomous Data Curation)
```python
class AutonomousDataCuration:
    def __init__(self):
        self.ai_assisted_curation = {
            "數據品質預測": "AI 預測數據對訓練的價值",
            "自動標註": "多模型協作標註",
            "品質控制": "自動檢測標註錯誤",
            "數據合成": "生成高質量合成數據"
        }

        self.active_learning = {
            "不確定性採樣": "選擇模型最不確定的樣本",
            "多樣性採樣": "確保數據多樣性",
            "代表性採樣": "選擇代表性樣本",
            "難例挖掘": "找出最具挑戰性的樣本"
        }
```

##### 聯邦數據工程
```python
class FederatedDataEngineering:
    def __init__(self):
        self.privacy_preserving = {
            "差分隱私": "數據隱私保護",
            "同態加密": "加密狀態下的計算",
            "安全多方計算": "協作計算不洩露數據",
            "知識蒸餾": "模型知識而非數據的共享"
        }

        self.distributed_curation = {
            "去中心化品質評估": "分散式數據品質評估",
            "聯邦學習評估": "跨機構協作評估",
            "隱私保護基準": "保護隱私的評估基準",
            "激勵機制": "鼓勵高質量數據貢獻"
        }
```

#### 5.3 可解釋評估與診斷

##### 評估結果的可解釋性
```python
class ExplainableEvaluation:
    def __init__(self):
        self.diagnostic_evaluation = {
            "能力解構": "分解評估為細粒度能力",
            "錯誤分析": "系統性分析失敗模式",
            "能力圖譜": "繪製模型能力的全景圖",
            "改進建議": "基於評估結果的改進方向"
        }

        self.causal_evaluation = {
            "因果介入": "通過介入理解因果關係",
            "反事實分析": "分析「如果...會怎樣」",
            "機制理解": "理解模型內部機制",
            "可控實驗": "控制變量的評估實驗"
        }
```

##### 評估偏見的檢測與緩解
```python
class BiasDetectionMitigation:
    def __init__(self):
        self.bias_detection = {
            "統計檢測": "統計方法檢測偏見",
            "對抗檢測": "對抗樣本檢測偏見",
            "因果檢測": "因果推理檢測偏見",
            "眾包檢測": "人群智慧檢測偏見"
        }

        self.bias_mitigation = {
            "數據去偏": "預處理階段去除偏見",
            "演算法去偏": "訓練階段的去偏技術",
            "後處理去偏": "輸出階段的偏見校正",
            "公平約束": "約束最佳化確保公平性"
        }
```

#### 5.4 評估生態系統的未來

##### 開放科學評估平台
```python
class OpenScienceEvaluation:
    def __init__(self):
        self.open_platforms = {
            "眾包評估": "全球研究者協作評估",
            "開放數據": "共享高質量評估數據",
            "標準化協議": "統一評估標準和流程",
            "可重現性": "確保評估結果可重現"
        }

        self.continuous_evaluation = {
            "即時評估": "模型發布即評估",
            "動態基準": "基準隨技術發展演進",
            "社群驅動": "社群共同維護基準",
            "版本控制": "評估結果的版本管理"
        }
```

#### 創新研究方向
1. **認知啟發的評估設計**：基於人類認知科學設計評估
2. **生態化評估環境**：在真實應用環境中評估模型
3. **協作智能評估**：評估人機協作的效果

#### 開放性挑戰
- **評估的評估**：如何評估評估基準本身的品質
- **動態能力測量**：如何測量快速變化的 AI 能力
- **社會影響評估**：如何評估 AI 對社會的長期影響

---

## 學習時程規劃 (Learning Schedule)

### 第 1-2 天：直覺建構期
**目標**：建立評估與數據工程的重要性認知
- **Day 1**: 評估的必要性與多維能力理解（3小時理論 + 2小時基準體驗）
- **Day 2**: 數據品質的重要性與工程挑戰（3小時理論 + 2小時數據分析）

### 第 3-5 天：技術深化期
**目標**：掌握主流評估和數據處理技術
- **Day 3**: OpenCompass 評估實踐（Lab-4.1）
- **Day 4**: 數據過濾與品質評估（Lab-4.2）
- **Day 5**: 自定義評估基準設計

### 第 6-7 天：形式化掌握期
**目標**：理解評估統計學與數據科學原理
- **Day 6**: 評估統計學與測量理論
- **Day 7**: 數據工程的數學基礎

### 第 8-9 天：理論探索期
**目標**：深入理解評估科學與數據理論
- **Day 8**: 測量理論與認知科學基礎
- **Day 9**: 評估倫理與公平性理論

### 第 10-12 天：創新實踐期
**目標**：前沿技術掌握與創新能力培養
- **Day 10**: 動態評估與適應性測試
- **Day 11**: 智能數據工程技術
- **Day 12**: 創新評估體系設計

---

## 依賴關係網路 (Dependency Network)

### 前置知識 (Prerequisites)
```yaml
硬依賴:
  - id: llm-fundamentals
    reason: "理解語言模型的基本概念和能力"
  - id: machine-learning-basics
    reason: "掌握機器學習的基礎理論"
  - id: statistics-probability
    reason: "理解統計學和概率論基礎"

軟依賴:
  - id: data-science
    reason: "理解數據分析和處理方法"
  - id: experimental-design
    reason: "理解實驗設計和統計推論"
  - id: cognitive-science
    reason: "理解人類認知和學習理論"
```

### 後續知識 (Enables)
```yaml
直接促成:
  - id: model-development
    reason: "評估指導模型改進方向"
  - id: production-deployment
    reason: "評估確保部署模型品質"
  - id: ai-safety
    reason: "評估是 AI 安全的重要保障"

間接影響:
  - id: research-methodology
    reason: "評估方法影響研究品質"
  - id: ai-governance
    reason: "評估支撐 AI 治理決策"
```

### 知識整合點 (Integration Points)
- **與模型訓練的協同**：評估指導訓練策略調整
- **與推理系統的配合**：評估驗證推理系統效果
- **與應用開發的結合**：評估確保應用品質

---

## 實驗環境與工具鏈 (Experimental Environment)

### 必需工具
```bash
# 基礎環境
poetry install --all-extras

# 評估框架
pip install opencompass  # 主要評估平台
pip install lm-evaluation-harness  # EleutherAI 評估工具

# 數據處理
pip install datasets pandas numpy
pip install nltk spacy  # 自然語言處理
pip install scikit-learn  # 機器學習工具

# 統計分析
pip install scipy statsmodels
pip install matplotlib seaborn  # 視覺化
```

### 評估數據集
```bash
# 主要基準數據集
wget https://huggingface.co/datasets/mmlu
wget https://huggingface.co/datasets/gsm8k
wget https://huggingface.co/datasets/humaneval

# 中文評估基準
wget https://huggingface.co/datasets/ceval
wget https://huggingface.co/datasets/clue
```

### 推薦硬體配置
- **最低配置**：RTX 3060 12GB (基礎評估實驗)
- **推薦配置**：RTX 4070 12GB (完整評估流程)
- **理想配置**：A100 40GB (大規模評估實驗)

---

## 評估體系 (Assessment Framework)

### Level 1: 基礎概念掌握 (30%)
- [ ] 理解評估基準的分類和用途
- [ ] 掌握數據品質的評估方法
- [ ] 能使用 OpenCompass 進行基礎評估

### Level 2: 技術實現能力 (40%)
- [ ] 實現自定義評估基準
- [ ] 執行數據清洗和過濾流程
- [ ] 分析評估結果的統計意義

### Level 3: 理論分析能力 (20%)
- [ ] 分析評估的信度和效度
- [ ] 理解數據工程的理論基礎
- [ ] 設計公平性評估方案

### Level 4: 創新設計能力 (10%)
- [ ] 設計新的評估範式
- [ ] 提出數據工程創新方法
- [ ] 探索評估的前沿技術

---

## 常見誤區與解決方案 (Common Pitfalls)

### 誤區 1: 過度依賴單一評估指標
**問題**：用 MMLU 分數判斷模型的全部能力
**解決**：建立多維度評估體系

### 誤區 2: 忽視評估基準的局限性
**問題**：認為高分數等於真實能力強
**解決**：理解基準與實際應用的差距

### 誤區 3: 數據越多越好的迷思
**問題**：忽視數據品質只追求數量
**解決**：建立品質優先的數據策略

### 誤區 4: 忽視評估的公平性
**問題**：評估結果存在系統性偏見
**解決**：納入公平性和多樣性考量

---

## 延伸閱讀與研究資源 (Extended Resources)

### 核心論文
- **OpenCompass**: "OpenCompass: A Universal Evaluation Platform for Foundation Models"
- **MMLU**: "Measuring Massive Multitask Language Understanding"
- **HELM**: "Holistic Evaluation of Language Models"
- **BIG-bench**: "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models"

### 開源專案
- **OpenCompass**: 通用基礎模型評估平台
- **EleutherAI LM Evaluation Harness**: 語言模型評估工具
- **HuggingFace Evaluate**: 評估指標庫
- **DataComp**: 大規模數據集過濾工具

### 評估基準
- **MMLU**: 大規模多任務語言理解
- **GSM8K**: 小學數學文字題
- **HumanEval**: 程式碼生成評估
- **C-Eval**: 中文綜合評估基準

### 進階課程
- Measurement Theory in Psychology (Various Universities)
- Statistical Methods for Data Science (Various Universities)
- AI Safety and Evaluation (Anthropic, OpenAI)

---

**最後更新**: 2025-01-27
**維護者**: LLM Engineering Team
**版本**: 2.0