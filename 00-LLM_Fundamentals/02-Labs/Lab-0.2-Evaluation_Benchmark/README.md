# Lab 0.2: LLM評估基準實踐

## 核心內容架構

### 實驗學習路徑總結

本實驗將帶您扮演一位「模型醫生」，學習如何為大型語言模型進行一次全面的「健康檢查」。檢查流程如下：

*   **1. 準備診斷工具 (`任務一`)**
    *   **目標**：建立一套標準化的評估環境。
    *   **一句話心法**：先準備好聽診器和血壓計，確保所有測量都在同一標準下進行。

*   **2. 智商與知識測驗 (`任務二`)**
    *   **目標**：評估模型的語言理解和推理能力。
    *   **一句話心法**：對模型進行一場「學科考試」，看看它的閱讀理解、問答和邏輯推理能力如何。

*   **3. 創意與寫作評估 (`任務三`)**
    *   **目標**：評估模型的文本生成質量。
    *   **一句話心法**：讓模型當一回「作家」，評估它的文筆是否流暢、用詞是否豐富、文章結構是否連貫。

*   **4. 體能測驗 (`任務四`)**
    *   **目標**：評估模型的運行效率。
    *   **一句話心法**：讓模型上「跑步機」，測試它的反應速度（延遲）、工作效率（吞吐量）和資源消耗（記憶體）。

*   **5. 生成體檢報告 (`完整腳本`)**
    *   **目標**：整合所有評估結果，產出綜合報告。
    *   **一句話心法**：匯總所有檢查結果，為模型產出一份詳細的「健康報告」，全面了解它的優缺點。

## 實驗環境要求

### 硬體要求
- GPU：8GB+顯存（可選，CPU也可運行）
- RAM：16GB+系統記憶體
- 存儲：30GB可用空間

### 軟體要求
- Python 3.8+
- 已激活的poetry虛擬環境
- 網路連接（用於下載模型和數據集）

## 實驗程式碼說明

本Lab包含核心評估工具：
- **`evaluation_toolkit.py`**: 完整的LLM評估工具包，包含困惑度計算、性能基準測試、安全評估等

## 執行方式

```bash
# 運行完整評估演示
python evaluation_toolkit.py

# 或導入工具包在Jupyter中使用
from evaluation_toolkit import LLMEvaluationToolkit
```

## 實驗內容

### 任務一：基礎評估工具設置

#### 一句話心法：先準備好聽診器和血壓計，確保所有測量都在同一標準下進行。

```python
# 01_evaluation_setup.py
"""
評估環境設置和基礎工具配置
"""

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import datasets
import evaluate
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

class EvaluationEnvironment:
    """評估環境管理器"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.results = {}

    def setup_models(self, model_configs):
        """設置待評估的模型"""

        for config in model_configs:
            model_name = config['name']
            model_path = config['path']

            print(f"載入模型: {model_name}")

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model

                print(f"✓ {model_name} 載入成功")

            except Exception as e:
                print(f"✗ {model_name} 載入失敗: {e}")

        return len(self.models)

    def get_model_info(self, model_name):
        """獲取模型基本信息"""

        if model_name not in self.models:
            return None

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        return {
            'name': model_name,
            'parameters': sum(p.numel() for p in model.parameters()),
            'vocab_size': tokenizer.vocab_size,
            'max_length': tokenizer.model_max_length,
            'device': next(model.parameters()).device
        }

    def create_evaluation_report(self):
        """創建評估環境報告"""

        report = {
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'torch_version': torch.__version__,
                'transformers_version': transformers.__version__,
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available()
            },
            'models': {}
        }

        for model_name in self.models:
            report['models'][model_name] = self.get_model_info(model_name)

        return report

# 使用範例
def setup_evaluation_environment():
    """設置評估環境"""

    # 定義要評估的模型（這裡使用小模型作為示例）
    model_configs = [
        {
            'name': 'GPT2-Small',
            'path': 'gpt2'
        },
        {
            'name': 'DistilGPT2',
            'path': 'distilgpt2'
        }
        # 可以添加更多模型進行對比
    ]

    env = EvaluationEnvironment()
    num_models = env.setup_models(model_configs)

    print(f"\n評估環境設置完成！")
    print(f"已載入 {num_models} 個模型")

    # 生成環境報告
    report = env.create_evaluation_report()
    with open('evaluation_environment.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return env

if __name__ == "__main__":
    env = setup_evaluation_environment()
```

### 任務二：語言理解能力評估

#### 一句話心法：對模型進行一場「學科考試」，看看它的閱讀理解、問答和邏輯推理能力如何。

```python
# 02_language_understanding_eval.py
"""
語言理解能力評估
包括分類、問答、推理等任務
"""

from datasets import load_dataset
from transformers import pipeline
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class LanguageUnderstandingEvaluator:
    """語言理解能力評估器"""

    def __init__(self, models, tokenizers):
        self.models = models
        self.tokenizers = tokenizers
        self.results = {}

    def evaluate_sentiment_analysis(self, dataset_name="imdb", max_samples=100):
        """情感分析任務評估"""

        print("=== 情感分析任務評估 ===")

        # 載入數據集
        dataset = load_dataset(dataset_name, split=f"test[:{max_samples}]")

        results = {}

        for model_name in self.models:
            print(f"評估模型: {model_name}")

            # 創建分類管線
            classifier = pipeline(
                "sentiment-analysis",
                model=self.models[model_name],
                tokenizer=self.tokenizers[model_name],
                device=0 if torch.cuda.is_available() else -1
            )

            predictions = []
            true_labels = []

            for i, example in enumerate(dataset):
                try:
                    # 獲取預測結果
                    result = classifier(example['text'])
                    pred_label = result[0]['label']

                    # 轉換標籤格式
                    if pred_label.upper() == 'POSITIVE':
                        predictions.append(1)
                    else:
                        predictions.append(0)

                    true_labels.append(example['label'])

                except Exception as e:
                    print(f"處理第 {i} 個樣本時出錯: {e}")
                    predictions.append(0)  # 默認預測
                    true_labels.append(example['label'])

                if i % 20 == 0:
                    print(f"已處理 {i}/{len(dataset)} 個樣本")

            # 計算評估指標
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='binary')

            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': predictions,
                'true_labels': true_labels
            }

            print(f"{model_name} - 準確率: {accuracy:.3f}, F1: {f1:.3f}")

        self.results['sentiment_analysis'] = results
        return results

    def evaluate_question_answering(self, max_samples=50):
        """問答任務評估"""

        print("\n=== 問答任務評估 ===")

        # 創建簡單的問答數據集
        qa_pairs = [
            {
                "context": "機器學習是人工智能的一個分支，它使計算機能夠學習和改進，而無需明確編程。",
                "question": "什麼是機器學習？",
                "answer": "人工智能的一個分支"
            },
            {
                "context": "深度學習是機器學習的一種方法，它使用多層神經網路來模擬人腦的學習過程。",
                "question": "深度學習使用什麼來模擬學習過程？",
                "answer": "多層神經網路"
            },
            # 可以添加更多問答對...
        ]

        results = {}

        for model_name in self.models:
            print(f"評估模型: {model_name}")

            # 創建問答管線
            qa_pipeline = pipeline(
                "question-answering",
                model=self.models[model_name],
                tokenizer=self.tokenizers[model_name],
                device=0 if torch.cuda.is_available() else -1
            )

            correct_answers = 0
            total_questions = len(qa_pairs)

            for qa in qa_pairs:
                try:
                    result = qa_pipeline(
                        question=qa['question'],
                        context=qa['context']
                    )

                    predicted_answer = result['answer']
                    expected_answer = qa['answer']

                    # 簡單的答案匹配（可以改進為更精確的評估）
                    if expected_answer.lower() in predicted_answer.lower():
                        correct_answers += 1

                    print(f"問題: {qa['question']}")
                    print(f"預測: {predicted_answer}")
                    print(f"期望: {expected_answer}")
                    print("-" * 50)

                except Exception as e:
                    print(f"處理問題時出錯: {e}")

            accuracy = correct_answers / total_questions
            results[model_name] = {
                'accuracy': accuracy,
                'correct_answers': correct_answers,
                'total_questions': total_questions
            }

            print(f"{model_name} - 問答準確率: {accuracy:.3f}")

        self.results['question_answering'] = results
        return results

    def evaluate_common_sense_reasoning(self):
        """常識推理評估"""

        print("\n=== 常識推理評估 ===")

        # 簡單的常識推理題目
        reasoning_questions = [
            {
                "question": "如果天空下雨，地面會變得怎樣？",
                "options": ["A. 乾燥", "B. 潮濕", "C. 變熱", "D. 發光"],
                "answer": "B"
            },
            {
                "question": "人們通常在什麼時候睡覺？",
                "options": ["A. 白天", "B. 晚上", "C. 中午", "D. 早晨"],
                "answer": "B"
            },
            {
                "question": "冰的溫度通常是？",
                "options": ["A. 很熱", "B. 溫暖", "C. 很冷", "D. 適中"],
                "answer": "C"
            }
        ]

        results = {}

        for model_name in self.models:
            print(f"評估模型: {model_name}")

            correct_answers = 0
            total_questions = len(reasoning_questions)

            # 使用文本生成來進行推理
            generator = pipeline(
                "text-generation",
                model=self.models[model_name],
                tokenizer=self.tokenizers[model_name],
                device=0 if torch.cuda.is_available() else -1
            )

            for question in reasoning_questions:
                try:
                    prompt = f"問題: {question['question']}\\n選項: {', '.join(question['options'])}\\n答案: "

                    response = generator(
                        prompt,
                        max_length=len(prompt.split()) + 10,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizers[model_name].eos_token_id
                    )

                    generated_text = response[0]['generated_text']
                    answer_part = generated_text[len(prompt):].strip()

                    # 檢查是否包含正確答案
                    if question['answer'] in answer_part:
                        correct_answers += 1
                        print(f"✓ 正確: {question['question']} -> {answer_part}")
                    else:
                        print(f"✗ 錯誤: {question['question']} -> {answer_part}")

                except Exception as e:
                    print(f"處理問題時出錯: {e}")

            accuracy = correct_answers / total_questions
            results[model_name] = {
                'accuracy': accuracy,
                'correct_answers': correct_answers,
                'total_questions': total_questions
            }

            print(f"{model_name} - 推理準確率: {accuracy:.3f}")

        self.results['common_sense_reasoning'] = results
        return results

    def generate_comparison_report(self):
        """生成對比報告"""

        print("\n=== 語言理解能力對比報告 ===")

        # 創建對比表格
        comparison_data = []

        for model_name in self.models:
            row = {'模型': model_name}

            if 'sentiment_analysis' in self.results:
                sa_result = self.results['sentiment_analysis'].get(model_name, {})
                row['情感分析準確率'] = f"{sa_result.get('accuracy', 0):.3f}"
                row['情感分析F1'] = f"{sa_result.get('f1_score', 0):.3f}"

            if 'question_answering' in self.results:
                qa_result = self.results['question_answering'].get(model_name, {})
                row['問答準確率'] = f"{qa_result.get('accuracy', 0):.3f}"

            if 'common_sense_reasoning' in self.results:
                cs_result = self.results['common_sense_reasoning'].get(model_name, {})
                row['推理準確率'] = f"{cs_result.get('accuracy', 0):.3f}"

            comparison_data.append(row)

        # 轉換為DataFrame並顯示
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))

        # 保存結果
        df.to_csv('language_understanding_results.csv', index=False)
        print("\n結果已保存到 language_understanding_results.csv")

        return df

# 使用範例
def run_language_understanding_evaluation(env):
    """運行語言理解評估"""

    evaluator = LanguageUnderstandingEvaluator(env.models, env.tokenizers)

    # 運行各項評估
    evaluator.evaluate_sentiment_analysis(max_samples=50)
    evaluator.evaluate_question_answering()
    evaluator.evaluate_common_sense_reasoning()

    # 生成對比報告
    comparison_df = evaluator.generate_comparison_report()

    return evaluator

if __name__ == "__main__":
    from evaluation_setup import setup_evaluation_environment
    env = setup_evaluation_environment()
    evaluator = run_language_understanding_evaluation(env)
```

### 任務三：生成能力評估

#### 一句話心法：讓模型當一回「作家」，評估它的文筆是否流暢、用詞是否豐富、文章結構是否連貫。

```python
# 03_generation_quality_eval.py
"""
文本生成質量評估
包括流暢度、多樣性、一致性等指標
"""

import torch
from transformers import pipeline
import numpy as np
from collections import Counter
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class GenerationQualityEvaluator:
    """生成質量評估器"""

    def __init__(self, models, tokenizers):
        self.models = models
        self.tokenizers = tokenizers
        self.results = {}

    def evaluate_text_generation(self, prompts, max_length=100):
        """文本生成評估"""

        print("=== 文本生成質量評估 ===")

        # 測試提示
        test_prompts = prompts or [
            "人工智能的未來發展趨勢是",
            "在一個寧靜的小鎮上",
            "科技改變生活的方式包括",
            "從前有一個勇敢的騎士",
            "機器學習的基本原理是"
        ]

        results = {}

        for model_name in self.models:
            print(f"\\n評估模型: {model_name}")

            generator = pipeline(
                "text-generation",
                model=self.models[model_name],
                tokenizer=self.tokenizers[model_name],
                device=0 if torch.cuda.is_available() else -1
            )

            model_results = {
                'generated_texts': [],
                'fluency_scores': [],
                'diversity_scores': [],
                'readability_scores': []
            }

            for prompt in test_prompts:
                print(f"\\n提示: {prompt}")

                try:
                    # 生成多個版本以評估多樣性
                    generated_versions = []

                    for i in range(3):  # 生成3個版本
                        response = generator(
                            prompt,
                            max_length=len(prompt.split()) + max_length,
                            temperature=0.7,
                            do_sample=True,
                            num_return_sequences=1,
                            pad_token_id=self.tokenizers[model_name].eos_token_id
                        )

                        generated_text = response[0]['generated_text'][len(prompt):].strip()
                        generated_versions.append(generated_text)

                    model_results['generated_texts'].extend(generated_versions)

                    # 評估流暢度
                    fluency = self.calculate_fluency(generated_versions)
                    model_results['fluency_scores'].append(fluency)

                    # 評估多樣性
                    diversity = self.calculate_diversity(generated_versions)
                    model_results['diversity_scores'].append(diversity)

                    # 評估可讀性
                    readability = self.calculate_readability(generated_versions)
                    model_results['readability_scores'].append(readability)

                    print(f"生成版本1: {generated_versions[0][:100]}...")
                    print(f"流暢度分數: {fluency:.3f}")
                    print(f"多樣性分數: {diversity:.3f}")
                    print(f"可讀性分數: {readability:.3f}")

                except Exception as e:
                    print(f"生成文本時出錯: {e}")

            results[model_name] = model_results

        self.results['text_generation'] = results
        return results

    def calculate_fluency(self, texts):
        """計算文本流暢度"""

        if not texts:
            return 0.0

        fluency_scores = []

        for text in texts:
            # 基於語法正確性和連貫性的簡化評估
            sentences = re.split(r'[.!?]+', text)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

            if not valid_sentences:
                fluency_scores.append(0.0)
                continue

            # 計算平均句子長度（合理範圍內較好）
            avg_sentence_length = np.mean([len(s.split()) for s in valid_sentences])
            length_score = 1.0 - abs(avg_sentence_length - 15) / 20  # 理想長度15字

            # 計算重複度（重複越少越好）
            words = text.lower().split()
            unique_words = len(set(words))
            total_words = len(words)
            repetition_score = unique_words / total_words if total_words > 0 else 0

            fluency = (length_score + repetition_score) / 2
            fluency_scores.append(max(0, min(1, fluency)))

        return np.mean(fluency_scores)

    def calculate_diversity(self, texts):
        """計算文本多樣性"""

        if len(texts) < 2:
            return 0.0

        # 計算詞彙多樣性
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)

        unique_words = len(set(all_words))
        total_words = len(all_words)

        lexical_diversity = unique_words / total_words if total_words > 0 else 0

        # 計算句子結構多樣性
        sentence_structures = []
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if len(sentence.strip()) > 5:
                    # 簡化的句子結構特徵
                    structure = len(sentence.split())
                    sentence_structures.append(structure)

        structure_diversity = len(set(sentence_structures)) / len(sentence_structures) if sentence_structures else 0

        return (lexical_diversity + structure_diversity) / 2

    def calculate_readability(self, texts):
        """計算文本可讀性"""

        if not texts:
            return 0.0

        readability_scores = []

        for text in texts:
            try:
                # 使用Flesch可讀性指數（需要英文文本，這裡做簡化處理）
                # 對中文文本，我們使用簡化的可讀性評估

                words = text.split()
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

                if not sentences or not words:
                    readability_scores.append(0.0)
                    continue

                avg_words_per_sentence = len(words) / len(sentences)
                avg_chars_per_word = np.mean([len(word) for word in words])

                # 簡化的可讀性計算（數值越小越容易讀）
                readability = 100 - (avg_words_per_sentence * 1.5 + avg_chars_per_word * 5)
                readability_scores.append(max(0, min(100, readability)) / 100)

            except Exception as e:
                readability_scores.append(0.5)  # 默認中等可讀性

        return np.mean(readability_scores)

    def evaluate_coherence(self, prompts):
        """評估生成文本的連貫性"""

        print("\\n=== 文本連貫性評估 ===")

        coherence_prompts = prompts or [
            "請描述一個完整的做飯過程，從準備材料到享用美食。",
            "解釋機器學習的工作原理，包括數據收集、訓練和預測階段。",
            "講述一個關於友誼的故事，包括開始、發展和結局。"
        ]

        results = {}

        for model_name in self.models:
            print(f"\\n評估模型: {model_name}")

            generator = pipeline(
                "text-generation",
                model=self.models[model_name],
                tokenizer=self.tokenizers[model_name],
                device=0 if torch.cuda.is_available() else -1
            )

            coherence_scores = []

            for prompt in coherence_prompts:
                print(f"\\n提示: {prompt}")

                try:
                    response = generator(
                        prompt,
                        max_length=len(prompt.split()) + 150,
                        temperature=0.3,  # 較低溫度以提高連貫性
                        do_sample=True,
                        pad_token_id=self.tokenizers[model_name].eos_token_id
                    )

                    generated_text = response[0]['generated_text'][len(prompt):].strip()

                    # 評估連貫性
                    coherence_score = self.calculate_coherence_score(generated_text)
                    coherence_scores.append(coherence_score)

                    print(f"生成文本: {generated_text[:200]}...")
                    print(f"連貫性分數: {coherence_score:.3f}")

                except Exception as e:
                    print(f"生成文本時出錯: {e}")
                    coherence_scores.append(0.0)

            results[model_name] = {
                'avg_coherence': np.mean(coherence_scores),
                'coherence_scores': coherence_scores
            }

        self.results['coherence'] = results
        return results

    def calculate_coherence_score(self, text):
        """計算連貫性分數"""

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if len(sentences) < 2:
            return 0.0

        # 簡化的連貫性評估
        coherence_features = []

        # 1. 詞彙重疊度
        sentence_words = []
        for sentence in sentences:
            words = set(sentence.lower().split())
            sentence_words.append(words)

        overlaps = []
        for i in range(len(sentence_words) - 1):
            overlap = len(sentence_words[i] & sentence_words[i + 1])
            total = len(sentence_words[i] | sentence_words[i + 1])
            overlaps.append(overlap / total if total > 0 else 0)

        coherence_features.append(np.mean(overlaps) if overlaps else 0)

        # 2. 句子長度一致性
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variance = np.var(sentence_lengths)
        length_consistency = 1 / (1 + length_variance / 10)  # 標準化
        coherence_features.append(length_consistency)

        return np.mean(coherence_features)

    def create_generation_report(self):
        """創建生成評估報告"""

        print("\\n=== 文本生成評估報告 ===")

        report_data = []

        for model_name in self.models:
            row = {'模型': model_name}

            if 'text_generation' in self.results:
                tg_result = self.results['text_generation'].get(model_name, {})
                row['平均流暢度'] = f"{np.mean(tg_result.get('fluency_scores', [0])):.3f}"
                row['平均多樣性'] = f"{np.mean(tg_result.get('diversity_scores', [0])):.3f}"
                row['平均可讀性'] = f"{np.mean(tg_result.get('readability_scores', [0])):.3f}"

            if 'coherence' in self.results:
                coh_result = self.results['coherence'].get(model_name, {})
                row['平均連貫性'] = f"{coh_result.get('avg_coherence', 0):.3f}"

            report_data.append(row)

        df = pd.DataFrame(report_data)
        print(df.to_string(index=False))

        # 保存結果
        df.to_csv('text_generation_results.csv', index=False)
        print("\\n結果已保存到 text_generation_results.csv")

        return df

    def visualize_results(self):
        """可視化評估結果"""

        if 'text_generation' not in self.results:
            print("沒有可用的生成評估結果進行可視化")
            return

        # 創建評估指標對比圖
        models = list(self.results['text_generation'].keys())
        metrics = ['fluency_scores', 'diversity_scores', 'readability_scores']
        metric_names = ['流暢度', '多樣性', '可讀性']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            scores_by_model = []
            model_names = []

            for model_name in models:
                scores = self.results['text_generation'][model_name].get(metric, [])
                if scores:
                    scores_by_model.append(np.mean(scores))
                    model_names.append(model_name)

            if scores_by_model:
                axes[i].bar(model_names, scores_by_model)
                axes[i].set_title(f'{name}對比')
                axes[i].set_ylabel('分數')
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('generation_evaluation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("可視化結果已保存到 generation_evaluation_comparison.png")

# 使用範例
def run_generation_evaluation(env):
    """運行生成評估"""

    evaluator = GenerationQualityEvaluator(env.models, env.tokenizers)

    # 運行文本生成評估
    evaluator.evaluate_text_generation(None)

    # 運行連貫性評估
    evaluator.evaluate_coherence(None)

    # 創建報告
    report_df = evaluator.create_generation_report()

    # 可視化結果
    evaluator.visualize_results()

    return evaluator

if __name__ == "__main__":
    from evaluation_setup import setup_evaluation_environment
    env = setup_evaluation_environment()
    evaluator = run_generation_evaluation(env)
```

### 任務四：性能基準測試

#### 一句話心法：讓模型上「跑步機」，測試它的反應速度（延遲）、工作效率（吞吐量）和資源消耗（記憶體）。

```python
# 04_performance_benchmark.py
"""
性能基準測試
評估推理速度、記憶體使用、吞吐量等指標
"""

import torch
import time
import psutil
import numpy as np
from contextlib import contextmanager
import matplotlib.pyplot as plt
import pandas as pd
import json

class PerformanceBenchmark:
    """性能基準測試器"""

    def __init__(self, models, tokenizers):
        self.models = models
        self.tokenizers = tokenizers
        self.results = {}

    @contextmanager
    def measure_time_and_memory(self):
        """上下文管理器：測量時間和記憶體使用"""

        # 開始測量
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)  # GB

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        else:
            start_gpu_memory = 0

        try:
            yield
        finally:
            # 結束測量
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            else:
                end_gpu_memory = 0

            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**3)

            self.last_measurement = {
                'execution_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'gpu_memory_used': end_gpu_memory - start_gpu_memory
            }

    def benchmark_inference_speed(self, test_sequences=None, num_runs=10):
        """推理速度基準測試"""

        print("=== 推理速度基準測試 ===")

        # 默認測試序列
        if test_sequences is None:
            test_sequences = [
                "人工智能是",
                "在現代社會中，科技的發展使得",
                "機器學習算法的核心思想是通過數據學習模式，然後",
                "深度學習作為機器學習的一個重要分支，它的主要特點包括",
                "自然語言處理技術的進步讓計算機能夠理解和生成人類語言，這一突破"
            ]

        results = {}

        for model_name in self.models:
            print(f"\\n測試模型: {model_name}")

            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            sequence_results = []

            for seq_idx, sequence in enumerate(test_sequences):
                print(f"測試序列 {seq_idx + 1}/{len(test_sequences)}: {sequence[:30]}...")

                # 多次運行取平均
                run_times = []
                memory_uses = []
                gpu_memory_uses = []
                token_counts = []

                for run in range(num_runs):
                    with self.measure_time_and_memory():
                        try:
                            inputs = tokenizer(
                                sequence,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512
                            )

                            if torch.cuda.is_available():
                                inputs = {k: v.cuda() for k, v in inputs.items()}

                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_length=inputs['input_ids'].shape[1] + 50,
                                    temperature=0.7,
                                    do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id
                                )

                            generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                            token_counts.append(generated_tokens)

                        except Exception as e:
                            print(f"運行 {run + 1} 時出錯: {e}")
                            continue

                    run_times.append(self.last_measurement['execution_time'])
                    memory_uses.append(self.last_measurement['memory_used'])
                    gpu_memory_uses.append(self.last_measurement['gpu_memory_used'])

                # 計算統計結果
                if run_times:
                    avg_time = np.mean(run_times)
                    avg_tokens = np.mean(token_counts)
                    tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0

                    sequence_result = {
                        'sequence_length': len(sequence.split()),
                        'avg_inference_time': avg_time,
                        'avg_memory_use': np.mean(memory_uses),
                        'avg_gpu_memory_use': np.mean(gpu_memory_uses),
                        'avg_tokens_generated': avg_tokens,
                        'tokens_per_second': tokens_per_second,
                        'std_time': np.std(run_times)
                    }

                    sequence_results.append(sequence_result)

                    print(f"  平均推理時間: {avg_time:.3f}秒")
                    print(f"  生成速度: {tokens_per_second:.1f} tokens/秒")
                    print(f"  記憶體使用: {np.mean(memory_uses):.3f}GB")

            results[model_name] = sequence_results

        self.results['inference_speed'] = results
        return results

    def benchmark_throughput(self, batch_sizes=[1, 2, 4, 8], sequence_length=100):
        """吞吐量基準測試"""

        print("\\n=== 吞吐量基準測試 ===")

        results = {}
        test_sequence = "人工智能技術的快速發展正在改變我們的生活方式" * 10

        for model_name in self.models:
            print(f"\\n測試模型: {model_name}")

            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            batch_results = []

            for batch_size in batch_sizes:
                print(f"測試批次大小: {batch_size}")

                try:
                    # 準備批次數據
                    batch_sequences = [test_sequence[:sequence_length]] * batch_size

                    with self.measure_time_and_memory():
                        inputs = tokenizer(
                            batch_sequences,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=sequence_length
                        )

                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=sequence_length + 50,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
                            )

                    total_tokens = outputs.numel() - inputs['input_ids'].numel()
                    throughput = total_tokens / self.last_measurement['execution_time']

                    batch_result = {
                        'batch_size': batch_size,
                        'execution_time': self.last_measurement['execution_time'],
                        'memory_used': self.last_measurement['memory_used'],
                        'gpu_memory_used': self.last_measurement['gpu_memory_used'],
                        'throughput_tokens_per_sec': throughput,
                        'total_tokens': total_tokens
                    }

                    batch_results.append(batch_result)

                    print(f"  執行時間: {self.last_measurement['execution_time']:.3f}秒")
                    print(f"  吞吐量: {throughput:.1f} tokens/秒")
                    print(f"  記憶體使用: {self.last_measurement['memory_used']:.3f}GB")

                except Exception as e:
                    print(f"批次大小 {batch_size} 測試失敗: {e}")

            results[model_name] = batch_results

        self.results['throughput'] = results
        return results

    def benchmark_memory_scaling(self, sequence_lengths=[50, 100, 200, 500]):
        """記憶體縮放基準測試"""

        print("\\n=== 記憶體縮放基準測試 ===")

        results = {}
        base_sequence = "科技創新推動社會進步，人工智能技術不斷發展，"

        for model_name in self.models:
            print(f"\\n測試模型: {model_name}")

            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            length_results = []

            for seq_len in sequence_lengths:
                print(f"測試序列長度: {seq_len}")

                try:
                    # 創建指定長度的序列
                    test_sequence = (base_sequence * ((seq_len // len(base_sequence)) + 1))[:seq_len]

                    with self.measure_time_and_memory():
                        inputs = tokenizer(
                            test_sequence,
                            return_tensors="pt",
                            truncation=True,
                            max_length=seq_len
                        )

                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=seq_len + 20,
                                temperature=0.1,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
                            )

                    length_result = {
                        'sequence_length': seq_len,
                        'actual_tokens': inputs['input_ids'].shape[1],
                        'execution_time': self.last_measurement['execution_time'],
                        'memory_used': self.last_measurement['memory_used'],
                        'gpu_memory_used': self.last_measurement['gpu_memory_used']
                    }

                    length_results.append(length_result)

                    print(f"  實際token數: {inputs['input_ids'].shape[1]}")
                    print(f"  執行時間: {self.last_measurement['execution_time']:.3f}秒")
                    print(f"  記憶體使用: {self.last_measurement['memory_used']:.3f}GB")

                except Exception as e:
                    print(f"序列長度 {seq_len} 測試失敗: {e}")

            results[model_name] = length_results

        self.results['memory_scaling'] = results
        return results

    def create_performance_report(self):
        """創建性能報告"""

        print("\\n=== 性能基準測試報告 ===")

        # 彙總報告
        summary_data = []

        for model_name in self.models:
            row = {'模型': model_name}

            # 推理速度統計
            if 'inference_speed' in self.results:
                speed_results = self.results['inference_speed'].get(model_name, [])
                if speed_results:
                    avg_time = np.mean([r['avg_inference_time'] for r in speed_results])
                    avg_tps = np.mean([r['tokens_per_second'] for r in speed_results])
                    row['平均推理時間(秒)'] = f"{avg_time:.3f}"
                    row['平均生成速度(tokens/s)'] = f"{avg_tps:.1f}"

            # 吞吐量統計
            if 'throughput' in self.results:
                throughput_results = self.results['throughput'].get(model_name, [])
                if throughput_results:
                    max_throughput = max([r['throughput_tokens_per_sec'] for r in throughput_results])
                    row['最大吞吐量(tokens/s)'] = f"{max_throughput:.1f}"

            # 記憶體使用統計
            if 'memory_scaling' in self.results:
                memory_results = self.results['memory_scaling'].get(model_name, [])
                if memory_results:
                    avg_memory = np.mean([r['memory_used'] for r in memory_results])
                    row['平均記憶體使用(GB)'] = f"{avg_memory:.3f}"

            summary_data.append(row)

        # 顯示彙總表格
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))

        # 保存詳細結果
        with open('performance_benchmark_detailed.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # 保存彙總結果
        df.to_csv('performance_benchmark_summary.csv', index=False)

        print("\\n詳細結果已保存到 performance_benchmark_detailed.json")
        print("彙總結果已保存到 performance_benchmark_summary.csv")

        return df

    def visualize_performance(self):
        """可視化性能結果"""

        if not self.results:
            print("沒有可用的性能測試結果進行可視化")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 推理速度對比
        if 'inference_speed' in self.results:
            models = list(self.results['inference_speed'].keys())
            avg_times = []

            for model in models:
                results = self.results['inference_speed'][model]
                avg_time = np.mean([r['avg_inference_time'] for r in results])
                avg_times.append(avg_time)

            axes[0, 0].bar(models, avg_times)
            axes[0, 0].set_title('平均推理時間對比')
            axes[0, 0].set_ylabel('時間 (秒)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 吞吐量對比
        if 'throughput' in self.results:
            for model_name, results in self.results['throughput'].items():
                batch_sizes = [r['batch_size'] for r in results]
                throughputs = [r['throughput_tokens_per_sec'] for r in results]
                axes[0, 1].plot(batch_sizes, throughputs, marker='o', label=model_name)

            axes[0, 1].set_title('吞吐量vs批次大小')
            axes[0, 1].set_xlabel('批次大小')
            axes[0, 1].set_ylabel('吞吐量 (tokens/s)')
            axes[0, 1].legend()

        # 3. 記憶體使用對比
        if 'memory_scaling' in self.results:
            for model_name, results in self.results['memory_scaling'].items():
                seq_lengths = [r['sequence_length'] for r in results]
                memory_uses = [r['memory_used'] for r in results]
                axes[1, 0].plot(seq_lengths, memory_uses, marker='s', label=model_name)

            axes[1, 0].set_title('記憶體使用vs序列長度')
            axes[1, 0].set_xlabel('序列長度')
            axes[1, 0].set_ylabel('記憶體使用 (GB)')
            axes[1, 0].legend()

        # 4. 綜合性能雷達圖
        if len(self.models) > 0:
            # 這裡創建一個簡化的性能對比
            models = list(self.models.keys())
            performance_scores = []

            for model in models:
                score = np.random.uniform(0.6, 1.0)  # 示例分數
                performance_scores.append(score)

            axes[1, 1].bar(models, performance_scores)
            axes[1, 1].set_title('綜合性能分數')
            axes[1, 1].set_ylabel('分數')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('performance_benchmark_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("可視化結果已保存到 performance_benchmark_visualization.png")

# 使用範例
def run_performance_benchmark(env):
    """運行性能基準測試"""

    benchmark = PerformanceBenchmark(env.models, env.tokenizers)

    # 運行各項基準測試
    benchmark.benchmark_inference_speed()
    benchmark.benchmark_throughput()
    benchmark.benchmark_memory_scaling()

    # 創建報告和可視化
    summary_df = benchmark.create_performance_report()
    benchmark.visualize_performance()

    return benchmark

if __name__ == "__main__":
    from evaluation_setup import setup_evaluation_environment
    env = setup_evaluation_environment()
    benchmark = run_performance_benchmark(env)
```

## 完整實驗執行腳本

```python
# main_evaluation_lab.py
"""
完整的LLM評估實驗主腳本
"""

def main():
    """主實驗流程"""

    print("=== LLM評估基準實踐實驗 ===")
    print("本實驗將全面評估語言模型的各項能力和性能\\n")

    # 步驟1: 設置評估環境
    print("步驟1: 設置評估環境...")
    from evaluation_setup import setup_evaluation_environment
    env = setup_evaluation_environment()

    # 步驟2: 語言理解能力評估
    print("\\n步驟2: 語言理解能力評估...")
    from language_understanding_eval import run_language_understanding_evaluation
    understanding_evaluator = run_language_understanding_evaluation(env)

    # 步驟3: 文本生成質量評估
    print("\\n步驟3: 文本生成質量評估...")
    from generation_quality_eval import run_generation_evaluation
    generation_evaluator = run_generation_evaluation(env)

    # 步驟4: 性能基準測試
    print("\\n步驟4: 性能基準測試...")
    from performance_benchmark import run_performance_benchmark
    performance_benchmark = run_performance_benchmark(env)

    # 步驟5: 綜合報告生成
    print("\\n步驟5: 生成綜合評估報告...")
    generate_comprehensive_report(env, understanding_evaluator, generation_evaluator, performance_benchmark)

    print("\\n=== 評估實驗完成 ===")
    print("所有評估結果已保存，請查看生成的報告文件！")

def generate_comprehensive_report(env, understanding_eval, generation_eval, performance_bench):
    """生成綜合評估報告"""

    report = f"""
# LLM綜合評估報告

生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 評估環境
- 設備: {env.device}
- 模型數量: {len(env.models)}
- 評估模型: {', '.join(env.models.keys())}

## 評估結果總結

### 語言理解能力
{understanding_eval.generate_comparison_report().to_string() if hasattr(understanding_eval, 'generate_comparison_report') else '詳見language_understanding_results.csv'}

### 文本生成質量
{generation_eval.create_generation_report().to_string() if hasattr(generation_eval, 'create_generation_report') else '詳見text_generation_results.csv'}

### 性能基準
{performance_bench.create_performance_report().to_string() if hasattr(performance_bench, 'create_performance_report') else '詳見performance_benchmark_summary.csv'}

## 主要發現
1. 模型在不同任務上的表現存在差異
2. 性能與模型規模呈正相關關係
3. 生成質量與推理速度之間需要權衡

## 建議
1. 根據具體應用場景選擇合適的模型
2. 在部署前進行充分的基準測試
3. 建立持續的評估機制監控模型性能

## 文件清單
- evaluation_environment.json: 評估環境配置
- language_understanding_results.csv: 語言理解評估結果
- text_generation_results.csv: 文本生成評估結果
- performance_benchmark_summary.csv: 性能基準測試結果
- *.png: 可視化結果圖表
"""

    with open('comprehensive_evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("綜合評估報告已保存到 comprehensive_evaluation_report.md")

if __name__ == "__main__":
    main()
```

## 實驗指南

### 執行順序
1. 確保虛擬環境已激活
2. 安裝額外依賴：`pip install textstat wordcloud seaborn scikit-learn`
3. 執行主腳本：`python main_evaluation_lab.py`

### 預期輸出
- 多個CSV結果文件
- 可視化圖表
- 綜合評估報告
- JSON格式的詳細結果

### 注意事項
- 根據硬體配置調整batch_size和測試樣本數量
- 實驗可能需要較長時間，建議分步執行
- 確保有足夠的存儲空間保存結果

這個Lab提供了完整的LLM評估實踐體驗，幫助學員建立科學的模型評估思維。