# 大模型（LLM）工程化高階速成課綱

## 課程介紹

本課程旨在為具備一定技術背景的學員（如軟體工程師、AI 研究員）提供一個基於麥肯錫金字塔原理的結構化學習路徑。學員將透過四大核心模組，系統性地掌握從 **模型訓練、高效推理、全面優化** 到 **評估驗證** 的完整生命週期，最終具備獨立完成 LLM 從原型到部署的工程化能力。

### 課程總目標 (Top-Level Idea)

建立一套完整且互斥（MECE）的 LLM 工程化知識框架，使學員能迅速、高效地掌握 LLM 工程化的核心知識體系。

### 目標學員
- 希望轉向 LLM 領域的軟體工程師與後端開發者。
- 專注於模型研究，期望補強工程化實踐能力的 AI 研究員。
- 負責 AI 產品開發的技術經理與架構師。

### 先備知識
- 熟悉 Python 程式設計。
- 具備深度學習與神經網路基礎知識。
- 了解 Linux 環境與 Shell 操作。

---

## 課程結構

### 第一章：LLM 核心訓練技術 (Model Training)
**模組目標：** 掌握 LLM 的生命週期起點——如何從無到有地訓練及微調一個強大的基礎模型。本章聚焦於核心演算法、分散式架構與前沿優化技術。

#### 1.1 參數高效微調 (Parameter-Efficient Fine-Tuning, PEFT)
- **1.1.1 理論基礎：**
  - **核心思想：** BitFit, Prefix Tuning, Prompt Tuning, P-Tuning v1/v2
  - **主流框架：** Adapter Tuning, LoRA, AdaLoRA, QLoRA
  - **高階變體：** MAM Adapter, UniPELT
- **1.1.2 PEFT 框架實戰 (HuggingFace PEFT)：**
  - Prompt Tuning / P-Tuning / Prefix Tuning
  - LoRA / IA3
  - INT8/FP4/NF4 低精度微調
  - 多模態模型微調

#### 1.2 分散式訓練 (Distributed Training)
- **1.2.1 並行技術原理：**
  - **數據與模型並行：** 數據並行、流水線並行、張量並行
  - **進階並行策略：** 序列並行、多維混合並行、自動並行
  - **專家混合模型：** MOE 並行
- **1.2.2 主流分散式框架：**
  - **PyTorch DDP**
  - **Megatron-LM**
  - **DeepSpeed**
  - **Megatron-DeepSpeed**

#### 1.3 訓練優化與對齊 (Training Optimization & Alignment)
- **1.3.1 訓練優化技術：**
  - **注意力機制優化：** FlashAttention V1/V2
  - **記憶體與計算優化：** 混合精度訓練、重計算、梯度累積
  - **架構優化：** MQA/GQA
- **1.3.2 對齊技術：**
  - **強化學習對齊：** PPO
  - **無獎勵函數對齊：** DPO, ORPO

#### 1.4 經典模型訓練案例分析
- **從 0 到 1 完整復現：** 斯坦福 Alpaca, BELLE, Chinese-LLaMA-Alpaca
- **高效微調與推理：** Alpaca-LoRA, ChatGLM, Vicuna, LLaMA
- **前沿模型探索：** OPT (RLHF), MiniGPT-4 (多模態)

---

### 第二章：LLM 高效推理部署 (Efficient Inference & Serving)
**模組目標：** 學習如何將訓練好的模型部署為高效、穩定且可擴展的線上服務。本章重點在於推理引擎、服務框架與性能優化。

#### 2.1 推理引擎核心
- **2.1.1 框架與引擎概覽**
- **2.1.2 NVIDIA 生態系統：** FasterTransformer, TensorRT-LLM
- **2.1.3 熱門開源引擎：** vLLM, SGLang, LightLLM, MNN-LLM

#### 2.2 模型推理服務
- **2.2.1 服務工具概覽**
- **2.2.2 Triton 推理伺服器：** 架構解析與開發實踐

#### 2.3 推理性能優化
- **2.3.1 核心優化技術概覽**
- **2.3.2 記憶體優化：** KV Cache, PagedAttention, Offload
- **2.3.3 解碼與吞吐量優化：** Continuous Batching, Speculative Decoding
- **2.3.4 特殊場景優化：** 結構化文本生成

---

### 第三章：LLM 模型壓縮技術 (Model Compression)
**模組目標：** 掌握縮小模型體積、降低部署成本的核心技術。本章涵蓋量化、稀疏化、蒸餾等關鍵方法。

#### 3.1 模型量化 (Quantization)
- **3.1.1 量化技術概覽**
- **3.1.2 訓練後量化 (PTQ)：** GPTQ, SmoothQuant, AWQ, SpQR
- **3.1.3 量化感知訓練 (QAT)：** QLoRA, PEQA
- **3.1.4 前沿量化技術：** FP8, FP6, FP4

#### 3.2 模型稀疏化/剪枝 (Sparsification/Pruning)
- **3.2.1 稀疏化技術概覽**
- **3.2.2 結構化剪枝：** LLM-Pruner, SliceGPT
- **3.2.3 非結構化剪枝：** SparseGPT, Wanda

#### 3.3 知識蒸餾 (Knowledge Distillation)
- **3.3.1 蒸餾技術概覽**
- **3.3.2 標準蒸餾：** MINILLM, GKD
- **3.3.3 基於湧現能力的蒸餾 (EA-based KD)**

#### 3.4 低秩分解 (Low-Rank Decomposition)
- **3.4.1 核心思想與應用**
- **3.4.2 混合壓縮技術：** 低秩+量化, 低秩+剪枝

---

### 第四章：LLM 評估與數據工程 (Evaluation & Data Engineering)
**模組目標：** 建立模型與數據的品質保證體系。本章涵蓋如何客觀評估模型效果，以及如何高效處理與篩選數據。

#### 4.1 模型效果評估
- **4.1.1 主流評估基準：** C-Eval, CMMLU, SuperCLUE, OpenCompass

#### 4.2 模型性能評估
- **4.2.1 推理性能指標：** GenAI-Perf (吞吐量, 延遲), TTFT, ITL

#### 4.3 LLM 數據工程
- **4.3.1 預訓練語料處理**
- **4.3.2 高效微調數據篩選技術：** DEITA, MoDS, IFD, CaR, LESS

---

## 課程總結與展望
- **回顧 LLM 工程化全景圖：** 整合四大模組，形成從數據到產品的閉環思維。
- **未來趨勢：** 探討多模態、端側 LLM、AI Agent 等前沿方向的工程挑戰。
- **期末專案：** 設計一個動手實踐專案，要求學員選擇一個場景，完整實踐模型微調、壓縮、部署與評估的全過程。
