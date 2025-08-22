# LLM 工程化課程 - 資料夾結構規劃

## 一、資料夾規劃設計理念

本課程的資料夾結構旨在提供一個清晰、模組化且循序漸進的學習路徑。其核心設計理念如下：

1.  **單元式學習 (Modular Learning):**
    每個核心章節（訓練、推理、壓縮、評估）都是一個獨立的模組，學員可以專注於單一領域的知識，同時也能看到各模組之間的關聯。

2.  **理論與實務分離 (Theory vs. Practice):**
    每個模組內部嚴格區分 `01-Theory` 和 `02-Labs`。理論資料夾提供教學文件（簡報、Markdown筆記），而實務資料夾則提供可執行的程式碼、Jupyter Notebooks 和練習，確保學員在理解概念後能立即動手實作。

3.  **漸進式難度 (Progressive Difficulty):**
    實務練習（Labs）按照從基礎到進階的順序進行編號。例如，從單一技術的簡單應用開始，逐步過渡到多種技術結合的複雜案例。

4.  **專案導向整合 (Project-Oriented Integration):**
    課程設有期中和期末專案，要求學員整合多個模組的知識來解決一個完整的工程問題（例如，從微調到部署一個高效的問答機器人），模擬真實世界的工作流程。

5.  **差異化比較分析 (Comparative Analysis):**
    特別設立 `Comparative_Analysis` 資料夾，提供對比不同技術方案（如不同推理引擎的性能、不同壓縮方法的效果）的實驗代碼與報告，培養學員的技術選型與決策能力。

## 二、整體資料夾結構

```
LLM_Engineering_Course/
├── 00-Course_Setup/
│   ├── pyproject.toml          # Poetry 專案配置與依賴
│   └── README.md               # Poetry 環境設定說明
│
├── 01-Core_Training_Techniques/
│   ├── 01-Theory/
│   │   ├── 1.1-PEFT.md
│   │   ├── 1.2-Distributed_Training.md
│   │   └── 1.3-Optimization_and_Alignment.md
│   └── 02-Labs/
│       ├── Lab-1.1-PEFT_with_HuggingFace/
│       ├── Lab-1.2-PyTorch_DDP_Basics/
│       └── Lab-1.3-Finetune_Alpaca_with_DeepSpeed/
│
├── 02-Efficient_Inference_and_Serving/
│   ├── 01-Theory/
│   │   ├── 2.1-Inference_Engines.md
│   │   └── 2.2-Serving_and_Optimization.md
│   └── 02-Labs/
│       ├── Lab-2.1-TensorRT-LLM_Quickstart/
│       ├── Lab-2.2-Deploy_with_Triton_Server/
│       └── Lab-2.3-vLLM_for_High_Throughput/
│
├── 03-Model_Compression/
│   ├── 01-Theory/
│   │   ├── 3.1-Quantization.md
│   │   ├── 3.2-Pruning.md
│   │   └── 3.3-Knowledge_Distillation.md
│   └── 02-Labs/
│       ├── Lab-3.1-Post_Training_Quantization_GPTQ/
│       ├── Lab-3.2-Pruning_with_Wanda/
│       └── Lab-3.3-Knowledge_Distillation_MiniLM/
│
├── 04-Evaluation_and_Data_Engineering/
│   ├── 01-Theory/
│   │   ├── 4.1-Evaluation_Benchmarks.md
│   │   └── 4.2-Data_Engineering.md
│   └── 02-Labs/
│       ├── Lab-4.1-Evaluate_with_OpenCompass/
│       └── Lab-4.2-Efficient_Data_Filtering/
│
├── 05-Projects/
│   ├── Project-01-Finetune_and_Deploy_Chatbot/
│   └── Project-02-Compress_and_Evaluate_Model/
│
├── 06-Comparative_Analysis/
│   ├── Inference_Engine_Showdown/
│   └── Compression_Techniques_Tradeoffs/
│
├── common_utils/
│   ├── data_loaders.py
│   └── model_helpers.py
│
├── datasets/
│   ├── alpaca_data/
│   └── README.md               # 資料集說明
│
└── README.md                   # 課程大綱
```

## 三、各資料夾詳細說明

- **`00-Course_Setup/`**:
  - **用途**: 提供課程開始前所需的環境設定檔。學員可使用 Poetry 快速建立一致的 Python 開發環境，避免因環境問題影響學習。

- **`01-Core_Training_Techniques/`**:
  - **`01-Theory/`**: 包含參數高效微調（PEFT）、分散式訓練、對齊技術等核心理論的教學文件。
  - **`02-Labs/`**: 提供從基礎的 PEFT 實作到複雜的 DeepSpeed 微調專案的程式碼。

- **`02-Efficient_Inference_and_Serving/`**:
  - **`01-Theory/`**: 講解 TensorRT-LLM、vLLM 等推理引擎的原理，以及 Triton 服務化框架和 KV Cache 等優化技術。
  - **`02-Labs/`**: 動手實踐主流推理引擎的部署與性能測試。

- **`03-Model_Compression/`**:
  - **`01-Theory/`**: 涵蓋量化、剪枝、知識蒸餾等模型壓縮方法的原理與最新進展。
  - **`02-Labs/`**: 提供 GPTQ 量化、Wanda 剪枝等實用壓縮技術的程式碼範例。

- **`04-Evaluation_and_Data_Engineering/`**:
  - **`01-Theory/`**: 介紹 OpenCompass 等評估框架以及高效數據篩選的理論。
  - **`02-Labs/`**: 學習如何使用標準化工具評估模型，並實作數據處理流程。

- **`05-Projects/`**:
  - **用途**: 綜合性專案區。`Project-01` 整合第一、二章知識，`Project-02` 整合第三、四章知識，旨在培養學員解決完整問題的能力。

- **`06-Comparative_Analysis/`**:
  - **用途**: 提供橫向評測專案。學員可在此比較不同技術方案的優劣（例如，vLLM vs. TensorRT-LLM 的吞吐量），深化對技術選型的理解。

- **`common_utils/`**:
  - **用途**: 存放可在多個 Labs 和專案中重用的輔助函數，如數據載入器、模型設定腳本等，遵循 DRY (Don't Repeat Yourself) 原則。

- **`datasets/`**:
  - **用途**: 存放課程中使用的公開資料集。`README.md` 中會說明各資料集的來源與用途。

- **`README.md`**:
  - **用途**: 課程的總體大綱與介紹文件。
