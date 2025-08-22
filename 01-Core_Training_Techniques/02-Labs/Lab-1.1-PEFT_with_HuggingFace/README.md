# Lab 1.1: 使用 Hugging Face PEFT 函式庫進行參數高效微調

本實驗旨在提供一份循序漸進的 Jupyter Notebook 教學，引導您從零開始，學習如何運用 Hugging Face 生態系中的 `peft` 函式庫，對大型語言模型 (LLM) 進行參數高效微調 (Parameter-Efficient Fine-Tuning)。

## 學習目標

完成本實驗後，您將能夠：
1.  理解參數高效微調 (PEFT) 的核心概念與動機。
2.  設置必要的 Python 環境並安裝相關函式庫。
3.  使用 `transformers` 載入預訓練模型與資料集。
4.  掌握如何使用 `peft` 函式庫中的 `LoraConfig` 來設定 LoRA (Low-Rank Adaptation) 微調。
5.  對模型應用 LoRA 並進行訓練。
6.  使用微調後的模型進行推理與評估。
7.  了解如何將 LoRA adapter 的權重合併回主模型並儲存。

## 實驗內容大綱

本實驗由一系列 Jupyter Notebook 組成，請依照以下順序執行：

1.  `01-environment-setup.ipynb`:
    *   **內容**: 檢查並引導安裝實驗所需的所有 Python 函式庫，包括 `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes` 等。

2.  `02-load-model-and-dataset.ipynb`:
    *   **內容**: 學習如何從 Hugging Face Hub 載入一個預訓練的大型語言模型 (以 `meta-llama/Llama-2-7b-chat-hf` 為例) 及其對應的 Tokenizer。同時，載入一個用於微調的資料集。

3.  `03-apply-lora.ipynb`:
    *   **內容**: 介紹 `peft` 的核心組件，包括 `get_peft_model` 函式與 `LoraConfig`。您將學習如何定義 LoRA 的各項超參數 (如 `r`, `lora_alpha`, `target_modules`)，並將其應用到預訓練模型上。

4.  `04-inference-and-evaluation.ipynb`:
    *   **內容**: 在模型經過 LoRA 微調後，本節將示範如何使用帶有 PEFT adapter 的模型進行推理，並觀察其在特定任務上的表現。

5.  `05-merge-and-save.ipynb`:
    *   **內容**: 學習 PEFT 的一個重要功能——權重合併。本節將示範如何調用 `merge_and_unload()` 方法將 LoRA 的權重合併回原始模型，並將其儲存為一個標準的、可直接部署的 Hugging Face 模型。

## 環境設定

在開始之前，請確保您已安裝 Python 3.8+。建議使用虛擬環境 (如 `venv` 或 `conda`) 來管理專案依賴。

您可以直接執行第一個 Notebook `01-environment-setup.ipynb`，它將引導您完成所有必要函式庫的安裝。

---
*本實驗內容參考並綜合了以下文章：*
- [知乎專欄：PEFT - 参数高效微调](https://zhuanlan.zhihu.com/p/646748939)
- [掘金：PEFT库：参数高效微调llm模型](https://juejin.cn/post/7257895211710627901)
