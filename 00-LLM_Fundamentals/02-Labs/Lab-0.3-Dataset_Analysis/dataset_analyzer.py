#!/usr/bin/env python3
"""
數據集類型分析工具
分析不同類型LLM數據集的特性和質量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset
import re
from collections import Counter, defaultdict
import langdetect
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    """數據集分析器"""

    def __init__(self):
        self.analysis_results = {}

    def analyze_pretraining_dataset(self, dataset_name: str = "wikitext",
                                  config: str = "wikitext-2-raw-v1",
                                  max_samples: int = 1000) -> Dict:
        """
        分析預訓練數據集

        分析維度：
        - 基礎統計：長度分佈、詞彙統計
        - 內容質量：重複率、完整性、清潔度
        - 語言分佈：多語言檢測
        - 主題多樣性：內容主題分析
        """

        print(f"=== 分析預訓練數據集: {dataset_name} ===")

        try:
            # 載入數據集
            dataset = load_dataset(dataset_name, config, split=f"train[:{max_samples}]")
            print(f"成功載入 {len(dataset)} 個樣本")

        except Exception as e:
            print(f"數據集載入失敗，創建模擬數據: {e}")
            dataset = self._create_mock_pretraining_data(max_samples)

        # 執行各項分析
        analysis_result = {
            'dataset_info': {
                'name': dataset_name,
                'config': config,
                'total_samples': len(dataset),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

        # 基礎統計分析
        analysis_result['basic_statistics'] = self._analyze_basic_statistics(dataset)

        # 內容質量分析
        analysis_result['content_quality'] = self._analyze_content_quality(dataset)

        # 語言分佈分析
        analysis_result['language_distribution'] = self._analyze_language_distribution(dataset)

        # 詞彙分析
        analysis_result['vocabulary_analysis'] = self._analyze_vocabulary(dataset)

        # 主題多樣性分析
        analysis_result['topic_diversity'] = self._analyze_topic_diversity(dataset)

        self.analysis_results['pretraining_dataset'] = analysis_result

        return analysis_result

    def analyze_instruction_dataset(self, dataset_name: str = None, max_samples: int = 500) -> Dict:
        """
        分析指令微調數據集

        分析維度：
        - 指令類型分佈
        - 長度統計分析
        - 複雜度評估
        - 質量問題識別
        """

        print(f"=== 分析指令數據集 ===")

        try:
            if dataset_name:
                dataset = load_dataset(dataset_name, split=f"train[:{max_samples}]")
            else:
                dataset = self._create_mock_instruction_data(max_samples)

            print(f"載入 {len(dataset)} 個指令樣本")

        except Exception as e:
            print(f"使用模擬指令數據: {e}")
            dataset = self._create_mock_instruction_data(max_samples)

        analysis_result = {
            'dataset_info': {
                'name': dataset_name or 'mock_instruction_data',
                'total_samples': len(dataset),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

        # 指令類型分析
        analysis_result['instruction_types'] = self._analyze_instruction_types(dataset)

        # 長度分佈分析
        analysis_result['length_distribution'] = self._analyze_instruction_lengths(dataset)

        # 複雜度分析
        analysis_result['complexity_analysis'] = self._analyze_instruction_complexity(dataset)

        # 質量問題檢測
        analysis_result['quality_issues'] = self._detect_instruction_quality_issues(dataset)

        self.analysis_results['instruction_dataset'] = analysis_result

        return analysis_result

    def analyze_preference_dataset(self, max_samples: int = 200) -> Dict:
        """
        分析偏好對齊數據集

        分析維度：
        - 偏好標註一致性
        - 回答質量分佈
        - 安全性問題識別
        """

        print("=== 分析偏好對齊數據集 ===")

        # 創建模擬偏好數據
        dataset = self._create_mock_preference_data(max_samples)

        analysis_result = {
            'dataset_info': {
                'name': 'mock_preference_data',
                'total_samples': len(dataset),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

        # 偏好一致性分析
        analysis_result['preference_consistency'] = self._analyze_preference_consistency(dataset)

        # 回答質量分析
        analysis_result['response_quality'] = self._analyze_response_quality(dataset)

        # 安全性分析
        analysis_result['safety_analysis'] = self._analyze_safety_aspects(dataset)

        self.analysis_results['preference_dataset'] = analysis_result

        return analysis_result

    def _create_mock_pretraining_data(self, num_samples: int) -> Dataset:
        """創建模擬預訓練數據"""

        sample_texts = [
            "人工智能技術正在快速發展，深刻改變著我們的生活和工作方式。從智能手機到自動駕駛汽車，AI的應用無處不在。",
            "機器學習是AI的核心技術之一，通過算法讓計算機從數據中學習模式，從而做出預測和決策。",
            "深度學習使用多層神經網路來模擬人腦的信息處理方式，在圖像識別和自然語言處理領域取得了突破性進展。",
            "自然語言處理技術使計算機能夠理解和生成人類語言，這為人機交互開闢了新的可能性。",
            "計算機視覺技術讓機器能夠識別和理解圖像內容，廣泛應用於醫療診斷、安防監控等領域。",
            "量子計算利用量子力學原理進行計算，在某些特定問題上可能實現指數級的計算優勢。",
            "區塊鏈技術通過分散式記賬和密碼學保證數據安全，正在金融、供應鏈等行業得到應用。",
            "雲計算提供按需分配的計算資源，降低了企業的IT成本並提高了靈活性。"
        ]

        # 擴展數據集
        expanded_texts = []
        for i in range(num_samples):
            base_text = sample_texts[i % len(sample_texts)]

            # 添加一些變化
            if i % 3 == 0:
                expanded_texts.append(base_text)
            elif i % 3 == 1:
                expanded_texts.append(base_text + " 這項技術的發展前景值得期待。")
            else:
                expanded_texts.append(base_text[:len(base_text)//2])  # 創建一些短文本

        return Dataset.from_dict({'text': expanded_texts})

    def _create_mock_instruction_data(self, num_samples: int) -> Dataset:
        """創建模擬指令數據"""

        instruction_templates = [
            {
                "instruction": "解釋以下概念",
                "input": "深度學習",
                "output": "深度學習是機器學習的子領域，使用多層神經網路來學習數據表示。"
            },
            {
                "instruction": "翻譯以下文本",
                "input": "Hello world",
                "output": "你好世界"
            },
            {
                "instruction": "回答問題",
                "input": "Python有什麼優點？",
                "output": "Python語法簡潔、易學易用、生態豐富、跨平台性好。"
            },
            {
                "instruction": "總結要點",
                "input": "人工智能包括機器學習、深度學習、自然語言處理等技術。",
                "output": "主要要點：1. AI包含多個技術領域 2. 機器學習是核心技術 3. 應用範圍廣泛"
            },
            {
                "instruction": "編寫代碼",
                "input": "寫一個計算階乘的函數",
                "output": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)"
            }
        ]

        # 擴展數據集
        instructions = []
        inputs = []
        outputs = []

        for i in range(num_samples):
            template = instruction_templates[i % len(instruction_templates)]
            instructions.append(template['instruction'])
            inputs.append(template['input'])
            outputs.append(template['output'])

        return Dataset.from_dict({
            'instruction': instructions,
            'input': inputs,
            'output': outputs
        })

    def _create_mock_preference_data(self, num_samples: int) -> Dataset:
        """創建模擬偏好數據"""

        preference_examples = [
            {
                'prompt': '解釋量子計算',
                'response_a': '量子計算很複雜。',
                'response_b': '量子計算利用量子力學原理進行計算，在某些問題上具有指數級優勢。',
                'preference': 'B'
            },
            {
                'prompt': '如何學習編程？',
                'response_a': '多練習，從基礎開始，選擇合適的語言，堅持學習。',
                'response_b': '隨便學學就行。',
                'preference': 'A'
            }
        ] * (num_samples // 2)

        return Dataset.from_dict({
            'prompt': [item['prompt'] for item in preference_examples],
            'response_a': [item['response_a'] for item in preference_examples],
            'response_b': [item['response_b'] for item in preference_examples],
            'preference': [item['preference'] for item in preference_examples]
        })

    def _analyze_basic_statistics(self, dataset) -> Dict:
        """基礎統計分析"""

        texts = dataset['text']

        # 長度統計
        char_lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]

        # 非空文本統計
        non_empty_texts = [text for text in texts if text.strip()]

        stats = {
            'total_samples': len(texts),
            'non_empty_samples': len(non_empty_texts),
            'empty_samples': len(texts) - len(non_empty_texts),
            'avg_char_length': np.mean(char_lengths),
            'median_char_length': np.median(char_lengths),
            'max_char_length': np.max(char_lengths),
            'min_char_length': np.min(char_lengths),
            'std_char_length': np.std(char_lengths),
            'avg_word_count': np.mean(word_counts),
            'median_word_count': np.median(word_counts),
            'std_word_count': np.std(word_counts)
        }

        return stats

    def _analyze_content_quality(self, dataset) -> Dict:
        """內容質量分析"""

        texts = dataset['text']
        quality_issues = {
            'very_short_texts': 0,
            'very_long_texts': 0,
            'repetitive_texts': 0,
            'special_char_heavy_texts': 0,
            'potential_quality_issues': []
        }

        for i, text in enumerate(texts):
            text_len = len(text.strip())

            # 長度問題
            if text_len < 10:
                quality_issues['very_short_texts'] += 1
                if text_len > 0:
                    quality_issues['potential_quality_issues'].append(f"樣本{i}: 文本過短")
            elif text_len > 5000:
                quality_issues['very_long_texts'] += 1
                quality_issues['potential_quality_issues'].append(f"樣本{i}: 文本過長")

            # 重複性檢查
            words = text.lower().split()
            if len(words) > 5:
                unique_words = len(set(words))
                repetition_ratio = unique_words / len(words)
                if repetition_ratio < 0.5:
                    quality_issues['repetitive_texts'] += 1
                    quality_issues['potential_quality_issues'].append(f"樣本{i}: 重複內容過多")

            # 特殊字符檢查
            special_chars = re.findall(r'[^\w\s\u4e00-\u9fff]', text)
            if len(special_chars) > len(text) * 0.3:
                quality_issues['special_char_heavy_texts'] += 1
                quality_issues['potential_quality_issues'].append(f"樣本{i}: 特殊字符過多")

        # 計算質量分數
        total_samples = len(texts)
        quality_score = 1.0 - (
            quality_issues['very_short_texts'] +
            quality_issues['repetitive_texts'] +
            quality_issues['special_char_heavy_texts']
        ) / total_samples

        quality_issues['overall_quality_score'] = max(0, quality_score)
        quality_issues['quality_grade'] = self._grade_quality(quality_score)

        return quality_issues

    def _analyze_language_distribution(self, dataset) -> Dict:
        """語言分佈分析"""

        texts = dataset['text']
        language_counts = Counter()
        detection_failures = 0

        # 限制樣本數量以提高速度
        sample_size = min(200, len(texts))
        sample_texts = texts[:sample_size]

        for i, text in enumerate(sample_texts):
            if not text.strip() or len(text.strip()) < 20:
                continue

            try:
                detected_lang = langdetect.detect(text)
                language_counts[detected_lang] += 1
            except Exception:
                detection_failures += 1

        # 轉換為百分比
        total_detected = sum(language_counts.values())
        language_percentages = {
            lang: count / total_detected * 100
            for lang, count in language_counts.items()
        } if total_detected > 0 else {}

        return {
            'language_counts': dict(language_counts),
            'language_percentages': language_percentages,
            'detection_failures': detection_failures,
            'samples_analyzed': sample_size,
            'dominant_language': language_counts.most_common(1)[0] if language_counts else None
        }

    def _analyze_vocabulary(self, dataset) -> Dict:
        """詞彙分析"""

        # 合併所有文本
        all_text = " ".join([text for text in dataset['text'] if text.strip()])

        # 分詞（簡單基於空格）
        words = re.findall(r'\b\w+\b', all_text.lower())

        vocabulary_stats = {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'vocabulary_diversity': len(set(words)) / len(words) if words else 0
        }

        # 詞頻分析
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(20)

        # Zipf定律檢驗（詞頻分佈規律）
        ranks = list(range(1, len(most_common_words) + 1))
        frequencies = [freq for word, freq in most_common_words]

        zipf_correlation = np.corrcoef(np.log(ranks), np.log(frequencies))[0, 1] if len(ranks) > 1 else 0

        return {
            'vocabulary_stats': vocabulary_stats,
            'most_common_words': most_common_words,
            'zipf_correlation': zipf_correlation,
            'vocabulary_richness': len(set(words)) / len(word_freq) if word_freq else 0
        }

    def _analyze_topic_diversity(self, dataset) -> Dict:
        """主題多樣性分析"""

        texts = dataset['text']

        # 關鍵詞分類（簡化的主題識別）
        topic_keywords = {
            '科技': ['技術', '科技', '計算機', '人工智能', '機器學習', 'AI', '算法'],
            '教育': ['學習', '教育', '知識', '學校', '學生', '老師', '課程'],
            '健康': ['健康', '醫療', '疾病', '治療', '醫生', '藥物', '營養'],
            '經濟': ['經濟', '金融', '投資', '市場', '商業', '企業', '貿易'],
            '文化': ['文化', '藝術', '音樂', '電影', '文學', '歷史', '傳統'],
            '體育': ['運動', '體育', '足球', '籃球', '比賽', '訓練', '健身']
        }

        topic_counts = Counter()

        for text in texts:
            text_lower = text.lower()

            for topic, keywords in topic_keywords.items():
                keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
                if keyword_count > 0:
                    topic_counts[topic] += 1

        # 計算主題多樣性指標
        total_topic_mentions = sum(topic_counts.values())
        topic_distribution = {
            topic: count / total_topic_mentions * 100
            for topic, count in topic_counts.items()
        } if total_topic_mentions > 0 else {}

        # 計算多樣性指數（Shannon熵）
        if topic_distribution:
            probabilities = [p/100 for p in topic_distribution.values()]
            diversity_index = -sum(p * np.log2(p) for p in probabilities if p > 0)
        else:
            diversity_index = 0

        return {
            'topic_counts': dict(topic_counts),
            'topic_distribution': topic_distribution,
            'diversity_index': diversity_index,
            'num_topics_covered': len(topic_counts),
            'topic_balance_score': self._calculate_topic_balance(topic_distribution)
        }

    def _analyze_instruction_types(self, dataset) -> Dict:
        """分析指令類型"""

        instructions = dataset['instruction']

        # 指令類型關鍵詞
        instruction_keywords = {
            '解釋說明': ['解釋', '說明', '描述', '闡述', '定義'],
            '翻譯轉換': ['翻譯', 'translate', '轉換', 'convert'],
            '編程代碼': ['編寫', '代碼', '函數', 'python', '程序', '算法'],
            '總結歸納': ['總結', '歸納', '概括', '摘要', '要點'],
            '列舉枚舉': ['列舉', '枚舉', '列出', '給出', '提供'],
            '比較對比': ['比較', '對比', '差異', '區別', '異同'],
            '創作生成': ['寫', '創作', '生成', '製作', '設計'],
            '問答回答': ['什麼', '如何', '為什麼', '怎樣', '回答']
        }

        type_counts = Counter()

        for instruction in instructions:
            instruction_lower = instruction.lower()
            categorized = False

            for category, keywords in instruction_keywords.items():
                if any(keyword in instruction_lower for keyword in keywords):
                    type_counts[category] += 1
                    categorized = True
                    break

            if not categorized:
                type_counts['其他'] += 1

        # 計算分佈
        total = len(instructions)
        type_distribution = {
            category: count / total * 100
            for category, count in type_counts.items()
        }

        return {
            'type_counts': dict(type_counts),
            'type_distribution': type_distribution,
            'most_common_type': type_counts.most_common(1)[0] if type_counts else None,
            'type_diversity_score': len(type_counts) / len(instruction_keywords) * 100
        }

    def _analyze_instruction_lengths(self, dataset) -> Dict:
        """分析指令長度分佈"""

        instructions = dataset['instruction']
        inputs = dataset['input']
        outputs = dataset['output']

        length_stats = {
            'instruction_lengths': [len(inst) for inst in instructions],
            'input_lengths': [len(inp) for inp in inputs],
            'output_lengths': [len(out) for out in outputs]
        }

        # 計算統計量
        stats_summary = {}
        for component, lengths in length_stats.items():
            stats_summary[component] = {
                'avg_length': np.mean(lengths),
                'median_length': np.median(lengths),
                'max_length': np.max(lengths),
                'min_length': np.min(lengths),
                'std_length': np.std(lengths)
            }

        # 輸入非空比例
        non_empty_inputs = sum(1 for inp in inputs if inp.strip())
        input_coverage = non_empty_inputs / len(inputs) * 100

        return {
            'length_statistics': stats_summary,
            'input_coverage_percent': input_coverage,
            'avg_total_length': np.mean([
                len(inst) + len(inp) + len(out)
                for inst, inp, out in zip(instructions, inputs, outputs)
            ])
        }

    def _analyze_instruction_complexity(self, dataset) -> Dict:
        """分析指令複雜度"""

        instructions = dataset['instruction']
        
        complexity_keywords = {
            '高': ['分析', '評估', '比較', '設計', '創建', '論證', '推導'],
            '中': ['解釋', '總結', '轉換', '編寫', '分類', '應用'],
            '低': ['列舉', '定義', '找出', '什麼是', '誰是', '列出']
        }

        complexity_counts = Counter()

        for instruction in instructions:
            instruction_lower = instruction.lower()
            categorized = False
            for level, keywords in complexity_keywords.items():
                if any(keyword in instruction_lower for keyword in keywords):
                    complexity_counts[level] += 1
                    categorized = True
                    break
            if not categorized:
                complexity_counts['未知'] += 1

        total = len(instructions)
        complexity_distribution = {
            level: count / total * 100
            for level, count in complexity_counts.items()
        }

        # Calculate a weighted complexity score
        score_mapping = {'高': 3, '中': 2, '低': 1, '未知': 1.5}
        weighted_score = sum(complexity_counts[level] * score_mapping[level] for level in complexity_counts) / total if total > 0 else 0


        return {
            'complexity_counts': dict(complexity_counts),
            'complexity_distribution': complexity_distribution,
            'average_complexity_score': weighted_score # Scale could be 1-3
        }

    def _detect_instruction_quality_issues(self, dataset) -> Dict:
        """檢測指令質量問題"""

        instructions = dataset['instruction']
        outputs = dataset['output']

        issues = {
            'very_short_instructions': 0,
            'very_short_outputs': 0,
            'very_long_outputs': 0,
            'repetitive_outputs': 0,
            'incomplete_outputs': 0,
            'quality_issues_list': []
        }

        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            # 檢查指令長度
            if len(instruction.strip()) < 5:
                issues['very_short_instructions'] += 1
                issues['quality_issues_list'].append(f"樣本{i}: 指令過短")

            # 檢查輸出質量
            if len(output.strip()) < 10:
                issues['very_short_outputs'] += 1
                issues['quality_issues_list'].append(f"樣本{i}: 輸出過短")
            elif len(output.strip()) > 1000:
                issues['very_long_outputs'] += 1
                issues['quality_issues_list'].append(f"樣本{i}: 輸出過長")

            # 檢查重複性
            output_words = output.split()
            if len(output_words) > 5:
                unique_ratio = len(set(output_words)) / len(output_words)
                if unique_ratio < 0.4:
                    issues['repetitive_outputs'] += 1
                    issues['quality_issues_list'].append(f"樣本{i}: 輸出重複度高")

            # 檢查完整性
            if output.strip().endswith(('...', '。。。', '等')):
                issues['incomplete_outputs'] += 1
                issues['quality_issues_list'].append(f"樣本{i}: 輸出可能不完整")

        # 計算質量評分
        total_samples = len(instructions)
        total_issues = (
            issues['very_short_instructions'] +
            issues['very_short_outputs'] +
            issues['very_long_outputs'] +
            issues['repetitive_outputs'] +
            issues['incomplete_outputs']
        )

        quality_score = max(0, 1 - total_issues / (total_samples * 5))  # 5種問題類型
        issues['overall_quality_score'] = quality_score
        issues['quality_grade'] = self._grade_quality(quality_score)

        return issues

    def _grade_quality(self, score: float) -> str:
        """質量評分"""

        if score >= 0.9:
            return "優秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "中等"
        elif score >= 0.6:
            return "及格"
        else:
            return "需要改進"

    def _calculate_topic_balance(self, topic_distribution: Dict) -> float:
        """計算主題平衡度"""

        if not topic_distribution:
            return 0.0

        # 理想情況是各主題均勻分佈
        ideal_percentage = 100 / len(topic_distribution)
        deviations = [abs(percentage - ideal_percentage) for percentage in topic_distribution.values()]
        avg_deviation = np.mean(deviations)

        # 轉換為0-1分數（偏差越小分數越高）
        balance_score = max(0, 1 - avg_deviation / ideal_percentage)

        return balance_score

    def _analyze_preference_consistency(self, dataset) -> Dict:
        """分析偏好一致性 (模擬)"""
        # 在真實場景中，這需要多個標註者的數據來計算 Fleiss' Kappa 或 Krippendorff's Alpha
        # 這裡我們模擬一個高一致性分數
        
        # 檢查是否有明顯的矛盾 (例如，相同 prompt 和 responses 但偏好不同)
        seen = {}
        conflicts = 0
        for item in dataset:
            key = (item['prompt'], item['response_a'], item['response_b'])
            if key in seen and seen[key] != item['preference']:
                conflicts += 1
            seen[key] = item['preference']

        total_pairs = len(dataset)
        consistency_score = (total_pairs - conflicts) / total_pairs if total_pairs > 0 else 1.0
        
        return {
            'consistency_score': consistency_score, # 模擬 Fleiss' Kappa
            'conflicts_found': conflicts,
            'note': 'This is a simulated consistency score. Real-world analysis requires multi-annotator data.'
        }

    def _analyze_response_quality(self, dataset) -> Dict:
        """分析回答質量"""
        
        chosen_responses = []
        rejected_responses = []

        for item in dataset:
            if item['preference'] == 'A':
                chosen_responses.append(item['response_a'])
                rejected_responses.append(item['response_b'])
            else:
                chosen_responses.append(item['response_b'])
                rejected_responses.append(item['response_a'])

        def get_quality_stats(responses):
            if not responses:
                return {'avg_length': 0, 'avg_word_count': 0}
            lengths = [len(r) for r in responses]
            words = [len(r.split()) for r in responses]
            return {
                'avg_length': np.mean(lengths),
                'avg_word_count': np.mean(words)
            }

        chosen_stats = get_quality_stats(chosen_responses)
        rejected_stats = get_quality_stats(rejected_responses)

        return {
            'chosen_response_stats': chosen_stats,
            'rejected_response_stats': rejected_stats,
            'avg_length_delta': chosen_stats['avg_length'] - rejected_stats['avg_length'],
        }

    def _analyze_safety_aspects(self, dataset) -> Dict:
        """分析安全性問題"""

        safety_keywords = ['暴力', '色情', '歧視', '危險', '非法']
        
        issues_found = 0
        issue_details = []

        for i, item in enumerate(dataset):
            prompt = item['prompt']
            resp_a = item['response_a']
            resp_b = item['response_b']
            
            for keyword in safety_keywords:
                if keyword in prompt or keyword in resp_a or keyword in resp_b:
                    issues_found += 1
                    issue_details.append(f"樣本{i}: 發現潛在不安全關鍵詞 '{keyword}'")
                    break # Move to next item once an issue is found

        total_samples = len(dataset)
        safety_score = (total_samples - issues_found) / total_samples if total_samples > 0 else 1.0

        return {
            'potential_safety_issues_count': issues_found,
            'safety_score': safety_score, # 0-1, 1 is best
            'issue_details': issue_details[:10] # show first 10 issues
        }

    def compare_datasets(self, dataset_results: List[Dict]) -> Dict:
        """對比多個數據集"""

        print("=== 數據集對比分析 ===")

        comparison_data = []

        for result in dataset_results:
            dataset_name = result['dataset_info']['name']

            row = {
                '數據集': dataset_name,
                '樣本數': result['dataset_info']['total_samples']
            }

            # 基礎統計 (主要用於預訓練)
            stats = result.get('basic_statistics')
            if stats:
                row['平均長度'] = f"{stats.get('avg_char_length', 0):.0f}"
                row['平均詞數'] = f"{stats.get('avg_word_count', 0):.0f}"
            else:
                row['平均長度'] = 'N/A'
                row['平均詞數'] = 'N/A'

            # 質量評分 (用於預訓練和指令)
            quality = result.get('content_quality') or result.get('quality_issues')
            if quality:
                row['質量評分'] = f"{quality.get('overall_quality_score', 0):.3f}"
                row['質量等級'] = quality.get('quality_grade', 'N/A')
            else:
                row['質量評分'] = 'N/A'
                row['質量等級'] = 'N/A'

            # 語言分佈 (主要用於預訓練)
            lang_dist = result.get('language_distribution')
            if lang_dist and lang_dist.get('dominant_language'):
                dominant_lang = lang_dist.get('dominant_language')
                row['主要語言'] = f"{dominant_lang[0]} ({dominant_lang[1]}次)"
            else:
                row['主要語言'] = 'N/A'

            # 詞彙多樣性 (主要用於預訓練)
            vocab_analysis = result.get('vocabulary_analysis')
            if vocab_analysis and vocab_analysis.get('vocabulary_stats'):
                vocab_stats = vocab_analysis.get('vocabulary_stats')
                row['詞彙多樣性'] = f"{vocab_stats.get('vocabulary_diversity', 0):.3f}"
            else:
                row['詞彙多樣性'] = 'N/A'

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        return {
            'comparison_table': comparison_df,
            'summary_insights': self._generate_comparison_insights(dataset_results),
            'recommendations': self._generate_dataset_recommendations(dataset_results)
        }

    def _generate_comparison_insights(self, results: List[Dict]) -> List[str]:
        """生成對比洞察"""

        insights = []

        # 分析樣本規模差異
        sample_sizes = [r['dataset_info']['total_samples'] for r in results]
        if len(sample_sizes) > 1 and min(sample_sizes) > 0 and max(sample_sizes) / min(sample_sizes) > 10:
            insights.append("數據集規模差異巨大，需要考慮平衡性")

        # 分析質量差異
        quality_scores = []
        for r in results:
            quality = r.get('content_quality') or r.get('quality_issues')
            if quality and 'overall_quality_score' in quality:
                quality_scores.append(quality['overall_quality_score'])

        if len(quality_scores) > 1 and max(quality_scores) - min(quality_scores) > 0.3:
            insights.append("數據集質量差異顯著，建議優先使用高質量數據")

        # 分析多樣性
        diversity_scores = []
        for r in results:
            vocab_analysis = r.get('vocabulary_analysis')
            if vocab_analysis and 'vocabulary_stats' in vocab_analysis:
                diversity_scores.append(vocab_analysis['vocabulary_stats']['vocabulary_diversity'])

        if len(diversity_scores) > 1 and np.std(diversity_scores) > 0.1:
            insights.append("詞彙多樣性差異明顯，可能影響模型泛化能力")

        return insights

    def _generate_dataset_recommendations(self, results: List[Dict]) -> List[str]:
        """生成數據集建議"""

        recommendations = []

        for result in results:
            dataset_name = result['dataset_info']['name']

            # 基於質量評分的建議
            quality = result.get('content_quality') or result.get('quality_issues')
            if quality and 'overall_quality_score' in quality:
                quality_score = quality['overall_quality_score']
                if quality_score > 0.8:
                    recommendations.append(f"✅ {dataset_name}: 質量優秀，推薦使用")
                elif quality_score > 0.6:
                    recommendations.append(f"⚠️ {dataset_name}: 質量中等，建議清洗後使用")
                else:
                    recommendations.append(f"❌ {dataset_name}: 質量較差，需要大量清洗")

            # 基於多樣性的建議
            vocab_analysis = result.get('vocabulary_analysis')
            if vocab_analysis and 'vocabulary_stats' in vocab_analysis:
                diversity = vocab_analysis['vocabulary_stats']['vocabulary_diversity']
                if diversity > 0.7:
                    recommendations.append(f"✅ {dataset_name}: 詞彙多樣性良好")
                else:
                    recommendations.append(f"⚠️ {dataset_name}: 詞彙多樣性不足，建議補充")

        return recommendations

    def visualize_analysis_results(self, analysis_type: str = 'all'):
        """可視化分析結果"""

        if analysis_type == 'all' or analysis_type == 'pretraining':
            self._visualize_pretraining_analysis()

        if analysis_type == 'all' or analysis_type == 'instruction':
            self._visualize_instruction_analysis()

    def _visualize_pretraining_analysis(self):
        """可視化預訓練數據分析"""

        if 'pretraining_dataset' not in self.analysis_results:
            print("沒有預訓練數據分析結果")
            return

        result = self.analysis_results['pretraining_dataset']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 文本長度分佈
        if 'basic_statistics' in result:
            # 模擬長度分佈數據
            np.random.seed(42)
            lengths = np.random.lognormal(4, 1, 1000)  # 對數正態分佈
            axes[0, 0].hist(lengths, bins=50, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('文本長度分佈')
            axes[0, 0].set_xlabel('字符數')
            axes[0, 0].set_ylabel('頻率')

        # 2. 語言分佈
        if 'language_distribution' in result:
            lang_dist = result['language_distribution']['language_percentages']
            if lang_dist:
                languages = list(lang_dist.keys())[:6]
                percentages = [lang_dist[lang] for lang in languages]
                axes[0, 1].pie(percentages, labels=languages, autopct='%1.1f%%')
                axes[0, 1].set_title('語言分佈')

        # 3. 主題多樣性
        if 'topic_diversity' in result:
            topic_dist = result['topic_diversity']['topic_distribution']
            if topic_dist:
                topics = list(topic_dist.keys())
                counts = list(topic_dist.values())
                axes[1, 0].bar(topics, counts, color='lightgreen')
                axes[1, 0].set_title('主題分佈')
                axes[1, 0].set_ylabel('百分比')
                axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. 詞頻分佈
        if 'vocabulary_analysis' in result:
            most_common = result['vocabulary_analysis']['most_common_words'][:10]
            if most_common:
                words = [item[0] for item in most_common]
                freqs = [item[1] for item in most_common]
                axes[1, 1].barh(words, freqs, color='salmon')
                axes[1, 1].set_title('高頻詞彙')
                axes[1, 1].set_xlabel('頻率')

        plt.tight_layout()
        plt.savefig('pretraining_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _visualize_instruction_analysis(self):
        """可視化指令數據分析"""

        if 'instruction_dataset' not in self.analysis_results:
            print("沒有指令數據分析結果")
            return

        result = self.analysis_results['instruction_dataset']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Instruction Dataset Analysis', fontsize=16)

        # 1. 指令類型分佈
        if 'instruction_types' in result and result['instruction_types'].get('type_counts'):
            types = result['instruction_types']['type_counts']
            categories = list(types.keys())
            values = list(types.values())

            axes[0, 0].bar(categories, values, color='cornflowerblue')
            axes[0, 0].set_title('指令類型分佈')
            axes[0, 0].set_ylabel('數量')
            axes[0, 0].tick_params(axis='x', labelrotation=45)

        # 2. 長度分佈
        if 'length_distribution' in result and result['length_distribution'].get('length_statistics'):
            stats = result['length_distribution']['length_statistics']
            lengths = ['指令', '輸入', '輸出']
            avg_lengths = [
                stats.get('instruction_lengths', {}).get('avg_length', 0),
                stats.get('input_lengths', {}).get('avg_length', 0),
                stats.get('output_lengths', {}).get('avg_length', 0)
            ]

            axes[0, 1].bar(lengths, avg_lengths, color=['skyblue', 'lightgreen', 'salmon'])
            axes[0, 1].set_title('平均長度對比')
            axes[0, 1].set_ylabel('平均字符數')

        # 3. 質量問題分佈
        if 'quality_issues' in result:
            quality = result['quality_issues']
            issues = ['過短指令', '過短輸出', '過長輸出', '重複輸出', '不完整輸出']
            counts = [
                quality.get('very_short_instructions', 0),
                quality.get('very_short_outputs', 0),
                quality.get('very_long_outputs', 0),
                quality.get('repetitive_outputs', 0),
                quality.get('incomplete_outputs', 0)
            ]

            axes[1, 0].bar(issues, counts, color='orange', alpha=0.7)
            axes[1, 0].set_title('質量問題統計')
            axes[1, 0].set_ylabel('問題數量')
            axes[1, 0].tick_params(axis='x', labelrotation=45)

        # 4. 指令複雜度
        if 'complexity_analysis' in result and result['complexity_analysis'].get('complexity_counts'):
            complexity = result['complexity_analysis']['complexity_counts']
            comp_types = list(complexity.keys())
            comp_values = list(complexity.values())

            axes[1, 1].pie(comp_values, labels=comp_types, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('指令複雜度分佈')
            axes[1, 1].axis('equal') 


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('instruction_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("指令數據集分析圖表已保存: instruction_dataset_analysis.png")

    def generate_analysis_report(self) -> str:
        """生成綜合分析報告"""

        report = f"""# 數據集分析綜合報告

生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 分析概述

本報告對{len(self.analysis_results)}個數據集進行了全面分析，涵蓋數據質量、分佈特性、多樣性等關鍵維度。

"""

        # 為每個數據集生成詳細分析
        for dataset_type, result in self.analysis_results.items():
            dataset_name = result['dataset_info']['name']
            total_samples = result['dataset_info']['total_samples']

            report += f"""
## {dataset_type.upper()}數據集分析

**數據集**: {dataset_name}
**樣本數量**: {total_samples:,}

"""

            # 添加關鍵發現
            if 'content_quality' in result:
                quality = result['content_quality']
                report += f"""
### 質量評估
- 整體質量評分: {quality['overall_quality_score']:.3f}
- 質量等級: {quality['quality_grade']}
- 主要問題: {len(quality['potential_quality_issues'])} 項
"""

            if 'vocabulary_analysis' in result:
                vocab = result['vocabulary_analysis']['vocabulary_stats']
                report += f"""
### 詞彙統計
- 總詞數: {vocab['total_words']:,}
- 唯一詞數: {vocab['unique_words']:,}
- 詞彙多樣性: {vocab['vocabulary_diversity']:.3f}
"""

        # 添加總結和建議
        report += """
## 總結與建議

### 關鍵發現
1. 數據質量是影響模型性能的關鍵因素
2. 詞彙多樣性直接影響模型的表達能力
3. 主題平衡性影響模型的知識覆蓋面
4. 數據清洗和預處理對最終效果至關重要

### 改進建議
1. 建立數據質量評估標準和自動化檢測流程
2. 實施分層抽樣確保主題和類型的平衡性
3. 建立持續的數據質量監控機制
4. 考慮數據增強技術提升多樣性

### 最佳實踐
- 預訓練數據：注重規模和多樣性，建立嚴格的質量控制
- 指令數據：注重質量和平衡性，確保指令類型充分覆蓋
- 偏好數據：注重一致性和代表性，避免標註偏差

---
*此報告基於數據集樣本分析生成，實際應用時請根據完整數據集進行評估。*
"""

        return report

def main():
    """主函數演示"""

    print("數據集類型分析工具演示")
    print("=" * 50)

    # 初始化分析器
    analyzer = DatasetAnalyzer()

    # 1. 分析預訓練數據集
    print("\n1. 分析預訓練數據集...")
    pretraining_result = analyzer.analyze_pretraining_dataset(max_samples=500)

    # 2. 分析指令數據集
    print("\n2. 分析指令數據集...")
    instruction_result = analyzer.analyze_instruction_dataset(max_samples=300)

    # 3. 分析偏好數據集
    print("\n3. 分析偏好數據集...")
    preference_result = analyzer.analyze_preference_dataset(max_samples=100)

    # 4. 數據集對比
    print("\n4. 數據集對比分析...")
    comparison_result = analyzer.compare_datasets([
        pretraining_result,
        instruction_result,
        preference_result
    ])

    print("\n📊 對比結果:")
    print(comparison_result['comparison_table'].to_string(index=False))

    print("\n💡 洞察發現:")
    for insight in comparison_result['summary_insights']:
        print(f"  - {insight}")

    print("\n📋 數據集建議:")
    for recommendation in comparison_result['recommendations']:
        print(f"  {recommendation}")

    # 5. 可視化結果
    print("\n5. 生成可視化圖表...")
    analyzer.visualize_analysis_results()

    # 6. 生成報告
    print("\n6. 生成綜合報告...")
    report = analyzer.generate_analysis_report()

    # 保存報告
    with open('dataset_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n✅ 分析完成！")
    print("📁 結果文件:")
    print("   - dataset_analysis_report.md (綜合報告)")
    print("   - pretraining_dataset_analysis.png (可視化圖表)")
    print("   - instruction_dataset_analysis.png (可視化圖表)")

    print("\n🎓 學習要點:")
    print("1. 數據質量比數據量更重要")
    print("2. 多樣性是模型泛化能力的基礎")
    print("3. 不同訓練階段需要不同特性的數據")
    print("4. 建立數據質量控制是工程化的關鍵")

if __name__ == "__main__":
    main()