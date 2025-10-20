# Lab 0.3: LLM數據集分析實踐

## 核心內容架構

### 實驗學習路徑總結

本實驗將讓您化身為「數據偵探」，學習如何審查和評估用於訓練大型語言模型的「食材」——數據集。我們的調查流程如下：

*   **1. 審查「大米」——預訓練數據集 (`任務一`)**
    *   **目標**：分析大規模、通用的文本數據。
    *   **一句話心法**：檢查「米」裡面有沒有沙子、石頭或壞掉的米粒，確保基礎食材的純淨。

*   **2. 檢驗「菜譜」——指令微調數據集 (`任務二`)**
    *   **目標**：分析結構化的指令-回答數據對。
    *   **一句話心法**：檢查「菜譜」是否清晰、準確、多樣，能否教模型學會做各種不同的「菜」。

*   **3. 撰寫「食材檢驗報告」 (`完整腳本`)**
    *   **目標**：整合所有分析結果，並對比不同數據集的特點。
    *   **一句話心法**：總結「大米」和「菜譜」的質量，為如何做出「美味大餐」（好模型）提供數據層面的建議。

## 實驗環境要求

### 硬體要求
- RAM：16GB+系統記憶體
- 存儲：20GB可用空間
- 網路：用於下載數據集

### 軟體要求
- Python 3.8+
- 已激活的poetry虛擬環境

## 實驗程式碼說明

本Lab包含數據分析工具：
- **`dataset_analyzer.py`**: 數據集分析器，支援預訓練、指令、偏好等多種數據集類型分析

## 執行方式

```bash
# 運行完整數據集分析演示
python dataset_analyzer.py

# 或在代碼中使用
from dataset_analyzer import DatasetAnalyzer
```

## 實驗內容

### 任務一：預訓練數據集分析

#### 一句話心法：檢查「米」裡面有沒有沙子、石頭或壞掉的米粒，確保基礎食材的純淨。

```python
# 01_pretraining_data_analysis.py
"""
預訓練數據集分析
分析大規模文本語料的特性和質量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import re
from collections import Counter
import langdetect
from wordcloud import WordCloud
import jieba
import warnings
warnings.filterwarnings('ignore')

class PretrainingDataAnalyzer:
    """預訓練數據分析器"""

    def __init__(self):
        self.dataset = None
        self.analysis_results = {}

    def load_sample_dataset(self, dataset_name="wikitext", config="wikitext-2-raw-v1", split="train", max_samples=1000):
        """載入樣本數據集"""

        print(f"載入數據集: {dataset_name}-{config}")
        print(f"最大樣本數: {max_samples}")

        try:
            self.dataset = load_dataset(
                dataset_name,
                config,
                split=f"{split}[:{max_samples}]"
            )
            print(f"成功載入 {len(self.dataset)} 個樣本")
            return True
        except Exception as e:
            print(f"數據集載入失敗: {e}")
            return False

    def basic_statistics(self):
        """基礎統計分析"""

        print("\\n=== 基礎統計分析 ===")

        if not self.dataset:
            print("請先載入數據集")
            return None

        # 文本長度統計
        text_lengths = [len(text) for text in self.dataset['text']]
        word_counts = [len(text.split()) for text in self.dataset['text']]

        # 非空文本統計
        non_empty_texts = [text for text in self.dataset['text'] if text.strip()]

        stats = {
            'total_samples': len(self.dataset),
            'non_empty_samples': len(non_empty_texts),
            'empty_samples': len(self.dataset) - len(non_empty_texts),
            'avg_char_length': np.mean(text_lengths),
            'median_char_length': np.median(text_lengths),
            'max_char_length': np.max(text_lengths),
            'min_char_length': np.min(text_lengths),
            'avg_word_count': np.mean(word_counts),
            'median_word_count': np.median(word_counts),
            'std_word_count': np.std(word_counts)
        }

        # 打印統計結果
        print(f"總樣本數: {stats['total_samples']:,}")
        print(f"非空樣本數: {stats['non_empty_samples']:,}")
        print(f"空樣本數: {stats['empty_samples']:,}")
        print(f"平均字符長度: {stats['avg_char_length']:.1f}")
        print(f"中位數字符長度: {stats['median_char_length']:.1f}")
        print(f"最大字符長度: {stats['max_char_length']:,}")
        print(f"平均詞數: {stats['avg_word_count']:.1f}")
        print(f"詞數標準差: {stats['std_word_count']:.1f}")

        self.analysis_results['basic_stats'] = stats
        return stats

    def content_quality_analysis(self):
        """內容質量分析"""

        print("\\n=== 內容質量分析 ===")

        quality_metrics = {
            'very_short_texts': 0,    # <10 字符
            'short_texts': 0,         # 10-50 字符
            'medium_texts': 0,        # 50-200 字符
            'long_texts': 0,          # >200 字符
            'repetitive_texts': 0,    # 高重複內容
            'special_char_texts': 0,  # 特殊字符過多
            'potential_quality_issues': []
        }

        for i, text in enumerate(self.dataset['text']):
            text_len = len(text.strip())

            # 長度分類
            if text_len < 10:
                quality_metrics['very_short_texts'] += 1
                if text_len > 0:
                    quality_metrics['potential_quality_issues'].append(f"樣本{i}: 過短文本")
            elif text_len < 50:
                quality_metrics['short_texts'] += 1
            elif text_len < 200:
                quality_metrics['medium_texts'] += 1
            else:
                quality_metrics['long_texts'] += 1

            # 重複性檢查
            words = text.lower().split()
            if len(words) > 5:
                unique_words = len(set(words))
                repetition_ratio = unique_words / len(words)
                if repetition_ratio < 0.5:
                    quality_metrics['repetitive_texts'] += 1
                    quality_metrics['potential_quality_issues'].append(f"樣本{i}: 高重複內容")

            # 特殊字符檢查
            special_chars = re.findall(r'[^\\w\\s\\u4e00-\\u9fff]', text)
            if len(special_chars) > len(text) * 0.3:
                quality_metrics['special_char_texts'] += 1
                quality_metrics['potential_quality_issues'].append(f"樣本{i}: 特殊字符過多")

        # 打印質量分析結果
        total_samples = len(self.dataset)
        print(f"極短文本 (<10字符): {quality_metrics['very_short_texts']:,} ({quality_metrics['very_short_texts']/total_samples*100:.1f}%)")
        print(f"短文本 (10-50字符): {quality_metrics['short_texts']:,} ({quality_metrics['short_texts']/total_samples*100:.1f}%)")
        print(f"中等文本 (50-200字符): {quality_metrics['medium_texts']:,} ({quality_metrics['medium_texts']/total_samples*100:.1f}%)")
        print(f"長文本 (>200字符): {quality_metrics['long_texts']:,} ({quality_metrics['long_texts']/total_samples*100:.1f}%)")
        print(f"重複內容文本: {quality_metrics['repetitive_texts']:,} ({quality_metrics['repetitive_texts']/total_samples*100:.1f}%)")
        print(f"特殊字符過多文本: {quality_metrics['special_char_texts']:,} ({quality_metrics['special_char_texts']/total_samples*100:.1f}%)")

        # 顯示前10個質量問題
        if quality_metrics['potential_quality_issues']:
            print("\\n潛在質量問題（前10個）:")
            for issue in quality_metrics['potential_quality_issues'][:10]:
                print(f"  {issue}")

        self.analysis_results['quality_metrics'] = quality_metrics
        return quality_metrics

    def language_detection_analysis(self):
        """語言檢測分析"""

        print("\\n=== 語言檢測分析 ===")

        language_counts = {}
        detection_failures = 0

        sample_size = min(200, len(self.dataset))  # 限制樣本大小以提高速度
        print(f"對 {sample_size} 個樣本進行語言檢測...")

        for i, text in enumerate(self.dataset['text'][:sample_size]):
            if not text.strip():
                continue

            try:
                # 只對足夠長的文本進行語言檢測
                if len(text.strip()) > 20:
                    detected_lang = langdetect.detect(text)
                    language_counts[detected_lang] = language_counts.get(detected_lang, 0) + 1
            except:
                detection_failures += 1

        # 按頻率排序
        sorted_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)

        print("檢測到的語言分佈:")
        for lang, count in sorted_languages[:10]:
            percentage = count / sample_size * 100
            print(f"  {lang}: {count} ({percentage:.1f}%)")

        if detection_failures > 0:
            print(f"檢測失敗: {detection_failures} 個樣本")

        self.analysis_results['language_distribution'] = dict(sorted_languages)
        return dict(sorted_languages)

    def vocabulary_analysis(self):
        """詞彙分析"""

        print("\\n=== 詞彙分析 ===")

        # 合併所有文本
        all_text = " ".join([text for text in self.dataset['text'] if text.strip()])

        # 簡單的詞彙統計（基於空格分割）
        words = re.findall(r'\\b\\w+\\b', all_text.lower())

        vocabulary_stats = {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'vocabulary_diversity': len(set(words)) / len(words) if words else 0
        }

        # 最常見詞彙
        word_freq = Counter(words)
        most_common = word_freq.most_common(20)

        print(f"總詞數: {vocabulary_stats['total_words']:,}")
        print(f"唯一詞數: {vocabulary_stats['unique_words']:,}")
        print(f"詞彙多樣性: {vocabulary_stats['vocabulary_diversity']:.4f}")

        print("\\n最常見的20個詞:")
        for word, freq in most_common:
            print(f"  {word}: {freq}")

        self.analysis_results['vocabulary'] = {
            'stats': vocabulary_stats,
            'most_common': most_common
        }

        return vocabulary_stats, most_common

    def visualize_analysis(self):
        """可視化分析結果"""

        print("\\n=== 生成可視化圖表 ===")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 文本長度分佈
        if 'basic_stats' in self.analysis_results:
            text_lengths = [len(text) for text in self.dataset['text']]
            axes[0, 0].hist(text_lengths, bins=50, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('文本長度分佈')
            axes[0, 0].set_xlabel('字符數')
            axes[0, 0].set_ylabel('頻率')

        # 2. 文本質量分類
        if 'quality_metrics' in self.analysis_results:
            quality = self.analysis_results['quality_metrics']
            categories = ['極短', '短', '中等', '長']
            values = [
                quality['very_short_texts'],
                quality['short_texts'],
                quality['medium_texts'],
                quality['long_texts']
            ]
            axes[0, 1].pie(values, labels=categories, autopct='%1.1f%%')
            axes[0, 1].set_title('文本長度分類')

        # 3. 語言分佈
        if 'language_distribution' in self.analysis_results:
            lang_dist = self.analysis_results['language_distribution']
            if lang_dist:
                languages = list(lang_dist.keys())[:8]  # 前8種語言
                counts = [lang_dist[lang] for lang in languages]
                axes[1, 0].bar(languages, counts)
                axes[1, 0].set_title('語言分佈')
                axes[1, 0].set_xlabel('語言')
                axes[1, 0].set_ylabel('樣本數')
                axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. 詞頻分佈
        if 'vocabulary' in self.analysis_results:
            most_common = self.analysis_results['vocabulary']['most_common'][:15]
            if most_common:
                words = [item[0] for item in most_common]
                freqs = [item[1] for item in most_common]
                axes[1, 1].barh(words, freqs)
                axes[1, 1].set_title('最常見詞彙')
                axes[1, 1].set_xlabel('頻率')

        plt.tight_layout()
        plt.savefig('pretraining_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("分析結果已保存到 pretraining_data_analysis.png")

    def generate_report(self):
        """生成分析報告"""

        report = f"""# 預訓練數據集分析報告

## 數據集信息
- 數據集: {getattr(self, 'dataset_name', '未知')}
- 樣本數量: {len(self.dataset) if self.dataset else 0:,}

## 基礎統計
"""

        if 'basic_stats' in self.analysis_results:
            stats = self.analysis_results['basic_stats']
            report += f"""
- 非空樣本: {stats['non_empty_samples']:,}
- 平均字符長度: {stats['avg_char_length']:.1f}
- 平均詞數: {stats['avg_word_count']:.1f}
- 詞數標準差: {stats['std_word_count']:.1f}
"""

        if 'quality_metrics' in self.analysis_results:
            quality = self.analysis_results['quality_metrics']
            report += f"""
## 質量分析
- 極短文本: {quality['very_short_texts']:,}
- 短文本: {quality['short_texts']:,}
- 中等文本: {quality['medium_texts']:,}
- 長文本: {quality['long_texts']:,}
- 重複內容: {quality['repetitive_texts']:,}
- 特殊字符過多: {quality['special_char_texts']:,}
"""

        if 'language_distribution' in self.analysis_results:
            lang_dist = self.analysis_results['language_distribution']
            report += "\\n## 語言分佈\\n"
            for lang, count in list(lang_dist.items())[:5]:
                report += f"- {lang}: {count}\\n"

        if 'vocabulary' in self.analysis_results:
            vocab_stats = self.analysis_results['vocabulary']['stats']
            report += f"""
## 詞彙統計
- 總詞數: {vocab_stats['total_words']:,}
- 唯一詞數: {vocab_stats['unique_words']:,}
- 詞彙多樣性: {vocab_stats['vocabulary_diversity']:.4f}
"""

        report += """
## 建議
1. 對極短文本進行過濾或合併
2. 對重複內容進行去重處理
3. 根據應用需求調整語言分佈
4. 考慮詞彙平衡性，避免某些詞過於頻繁
"""

        with open('pretraining_data_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("分析報告已保存到 pretraining_data_report.md")
        return report

def run_pretraining_analysis():
    """運行預訓練數據分析"""

    analyzer = PretrainingDataAnalyzer()

    # 載入數據集
    success = analyzer.load_sample_dataset()
    if not success:
        return None

    # 運行各項分析
    analyzer.basic_statistics()
    analyzer.content_quality_analysis()
    analyzer.language_detection_analysis()
    analyzer.vocabulary_analysis()

    # 生成可視化和報告
    analyzer.visualize_analysis()
    analyzer.generate_report()

    return analyzer

if __name__ == "__main__":
    analyzer = run_pretraining_analysis()
```

### 任務二：指令微調數據集分析

#### 一句話心法：檢查「菜譜」是否清晰、準確、多樣，能否教模型學會做各種不同的「菜」。

```python
# 02_instruction_data_analysis.py
"""
指令微調數據集分析
分析指令-回答對的質量和特性
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import re
from collections import Counter, defaultdict

class InstructionDataAnalyzer:
    """指令數據分析器"""

    def __init__(self):
        self.dataset = None
        self.analysis_results = {}

    def load_instruction_dataset(self, dataset_name="tatsu-lab/alpaca", split="train", max_samples=1000):
        """載入指令數據集"""

        print(f"載入指令數據集: {dataset_name}")

        try:
            self.dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
            print(f"成功載入 {len(self.dataset)} 個指令樣本")
            return True
        except Exception as e:
            print(f"數據集載入失敗，使用模擬數據: {e}")
            # 創建模擬的指令數據
            self.create_mock_instruction_data()
            return True

    def create_mock_instruction_data(self):
        """創建模擬指令數據"""

        mock_data = {
            'instruction': [
                "解釋什麼是機器學習",
                "翻譯以下英文句子",
                "寫一個Python函數計算階乘",
                "總結以下文章的主要內容",
                "給出三個健康飲食的建議",
                "解釋量子計算的基本原理",
                "比較Python和Java的優缺點",
                "描述如何製作一杯咖啡",
                "列舉五種可再生能源",
                "解釋什麼是區塊鏈技術"
            ] * 10,  # 重複以增加樣本數
            'input': [
                "",
                "Hello world, how are you?",
                "",
                "人工智能技術正在快速發展...",
                "",
                "",
                "",
                "",
                "",
                ""
            ] * 10,
            'output': [
                "機器學習是人工智能的一個子領域，通過算法讓計算機從數據中學習模式。",
                "你好世界，你好嗎？",
                "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)",
                "文章主要討論了人工智能技術的發展趨勢和應用前景。",
                "1. 多吃蔬菜水果 2. 減少加工食品 3. 保持適量運動",
                "量子計算利用量子力學原理，使用量子比特進行計算，在某些問題上具有指數級優勢。",
                "Python語法簡潔易學，Java性能優秀企業級應用廣泛，各有優勢。",
                "1. 研磨咖啡豆 2. 加熱水至90-96度 3. 沖泡3-4分鐘 4. 享用",
                "太陽能、風能、水力發電、地熱能、生物質能",
                "區塊鏈是一種分佈式賬本技術，通過加密和共識機制確保數據安全和透明。"
            ] * 10
        }

        # 轉換為dataset格式
        from datasets import Dataset
        self.dataset = Dataset.from_dict(mock_data)
        print(f"創建了 {len(self.dataset)} 個模擬指令樣本")

    def analyze_instruction_types(self):
        """分析指令類型"""

        print("\\n=== 指令類型分析 ===")

        instruction_keywords = {
            '解釋/說明': ['解釋', '說明', '描述', '闡述'],
            '翻譯': ['翻譯', 'translate', '轉換'],
            '編程/代碼': ['函數', '代碼', 'python', 'java', '編程', '算法'],
            '總結/歸納': ['總結', '歸納', '概括', '摘要'],
            '列舉/枚舉': ['列舉', '枚舉', '列出', '給出'],
            '比較/對比': ['比較', '對比', '區別', '差異'],
            '創作/生成': ['寫', '創作', '生成', '製作'],
            '問答': ['什麼是', '如何', '為什麼', '怎樣'],
            '其他': []
        }

        type_counts = defaultdict(int)

        for instruction in self.dataset['instruction']:
            instruction_lower = instruction.lower()
            categorized = False

            for category, keywords in instruction_keywords.items():
                if category == '其他':
                    continue

                for keyword in keywords:
                    if keyword in instruction_lower:
                        type_counts[category] += 1
                        categorized = True
                        break

                if categorized:
                    break

            if not categorized:
                type_counts['其他'] += 1

        # 打印結果
        total_instructions = len(self.dataset)
        print("指令類型分佈:")
        for category, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_instructions * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        self.analysis_results['instruction_types'] = dict(type_counts)
        return dict(type_counts)

    def analyze_length_distribution(self):
        """分析長度分佈"""

        print("\\n=== 長度分佈分析 ===")

        instruction_lengths = [len(inst) for inst in self.dataset['instruction']]
        input_lengths = [len(inp) for inp in self.dataset['input']]
        output_lengths = [len(out) for out in self.dataset['output']]

        # 計算統計量
        stats = {
            'instruction': {
                'avg_length': np.mean(instruction_lengths),
                'median_length': np.median(instruction_lengths),
                'max_length': np.max(instruction_lengths),
                'min_length': np.min(instruction_lengths)
            },
            'input': {
                'avg_length': np.mean(input_lengths),
                'median_length': np.median(input_lengths),
                'max_length': np.max(input_lengths),
                'non_empty_ratio': sum(1 for x in input_lengths if x > 0) / len(input_lengths)
            },
            'output': {
                'avg_length': np.mean(output_lengths),
                'median_length': np.median(output_lengths),
                'max_length': np.max(output_lengths),
                'min_length': np.min(output_lengths)
            }
        }

        print("指令長度統計:")
        print(f"  平均長度: {stats['instruction']['avg_length']:.1f} 字符")
        print(f"  中位數長度: {stats['instruction']['median_length']:.1f} 字符")
        print(f"  最大長度: {stats['instruction']['max_length']} 字符")

        print("\\n輸入長度統計:")
        print(f"  平均長度: {stats['input']['avg_length']:.1f} 字符")
        print(f"  非空比例: {stats['input']['non_empty_ratio']:.1%}")

        print("\\n輸出長度統計:")
        print(f"  平均長度: {stats['output']['avg_length']:.1f} 字符")
        print(f"  中位數長度: {stats['output']['median_length']:.1f} 字符")
        print(f"  最大長度: {stats['output']['max_length']} 字符")

        self.analysis_results['length_stats'] = stats
        return stats

    def analyze_quality_indicators(self):
        """分析質量指標"""

        print("\\n=== 質量指標分析 ===")

        quality_issues = {
            'very_short_instructions': 0,  # 過短指令
            'very_short_outputs': 0,       # 過短回答
            'very_long_outputs': 0,        # 過長回答
            'repetitive_patterns': 0,       # 重複模式
            'incomplete_responses': 0,      # 不完整回答
            'potential_issues': []
        }

        for i, (instruction, input_text, output) in enumerate(zip(
            self.dataset['instruction'],
            self.dataset['input'],
            self.dataset['output']
        )):
            # 檢查指令長度
            if len(instruction.strip()) < 5:
                quality_issues['very_short_instructions'] += 1
                quality_issues['potential_issues'].append(f"樣本{i}: 指令過短")

            # 檢查輸出長度
            if len(output.strip()) < 10:
                quality_issues['very_short_outputs'] += 1
                quality_issues['potential_issues'].append(f"樣本{i}: 回答過短")
            elif len(output.strip()) > 1000:
                quality_issues['very_long_outputs'] += 1
                quality_issues['potential_issues'].append(f"樣本{i}: 回答過長")

            # 檢查重複模式
            if len(set(output.split())) < len(output.split()) * 0.3:
                quality_issues['repetitive_patterns'] += 1
                quality_issues['potential_issues'].append(f"樣本{i}: 回答重複度高")

            # 檢查不完整回答
            if output.strip().endswith(('...', '。。。', '等等')) or len(output.strip()) < 3:
                quality_issues['incomplete_responses'] += 1
                quality_issues['potential_issues'].append(f"樣本{i}: 回答可能不完整")

        # 打印質量分析結果
        total_samples = len(self.dataset)
        print("質量問題統計:")
        print(f"過短指令: {quality_issues['very_short_instructions']} ({quality_issues['very_short_instructions']/total_samples*100:.1f}%)")
        print(f"過短回答: {quality_issues['very_short_outputs']} ({quality_issues['very_short_outputs']/total_samples*100:.1f}%)")
        print(f"過長回答: {quality_issues['very_long_outputs']} ({quality_issues['very_long_outputs']/total_samples*100:.1f}%)")
        print(f"重複模式: {quality_issues['repetitive_patterns']} ({quality_issues['repetitive_patterns']/total_samples*100:.1f}%)")
        print(f"不完整回答: {quality_issues['incomplete_responses']} ({quality_issues['incomplete_responses']/total_samples*100:.1f}%)")

        if quality_issues['potential_issues'][:10]:
            print("\\n潛在問題樣本（前10個）:")
            for issue in quality_issues['potential_issues'][:10]:
                print(f"  {issue}")

        self.analysis_results['quality_issues'] = quality_issues
        return quality_issues

    def analyze_instruction_complexity(self):
        """分析指令複雜度"""

        print("\\n=== 指令複雜度分析 ===")

        complexity_indicators = {
            'single_step': 0,      # 單步指令
            'multi_step': 0,       # 多步指令
            'conditional': 0,      # 條件指令
            'creative': 0,         # 創作指令
            'analytical': 0        # 分析指令
        }

        # 複雜度關鍵詞
        multi_step_keywords = ['首先', '然後', '接下來', '最後', '步驟', '和']
        conditional_keywords = ['如果', '假如', '當', '若', 'if']
        creative_keywords = ['創作', '寫', '設計', '想像', '編寫']
        analytical_keywords = ['分析', '比較', '評估', '解釋', '討論']

        for instruction in self.dataset['instruction']:
            instruction_lower = instruction.lower()

            # 檢查多步指令
            multi_step_count = sum(1 for keyword in multi_step_keywords if keyword in instruction)
            if multi_step_count > 1:
                complexity_indicators['multi_step'] += 1
            else:
                complexity_indicators['single_step'] += 1

            # 檢查其他類型
            if any(keyword in instruction_lower for keyword in conditional_keywords):
                complexity_indicators['conditional'] += 1

            if any(keyword in instruction for keyword in creative_keywords):
                complexity_indicators['creative'] += 1

            if any(keyword in instruction for keyword in analytical_keywords):
                complexity_indicators['analytical'] += 1

        # 打印複雜度分析
        total = len(self.dataset)
        print("指令複雜度分佈:")
        for category, count in complexity_indicators.items():
            percentage = count / total * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        self.analysis_results['complexity'] = complexity_indicators
        return complexity_indicators

    def visualize_instruction_analysis(self):
        """可視化指令分析結果"""

        print("\\n=== 生成可視化圖表 ===")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 指令類型分佈
        if 'instruction_types' in self.analysis_results:
            types = self.analysis_results['instruction_types']
            categories = list(types.keys())
            values = list(types.values())

            axes[0, 0].bar(categories, values)
            axes[0, 0].set_title('指令類型分佈')
            axes[0, 0].set_xlabel('指令類型')
            axes[0, 0].set_ylabel('數量')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 長度分佈
        if 'length_stats' in self.analysis_results:
            lengths = ['指令', '輸入', '輸出']
            avg_lengths = [
                self.analysis_results['length_stats']['instruction']['avg_length'],
                self.analysis_results['length_stats']['input']['avg_length'],
                self.analysis_results['length_stats']['output']['avg_length']
            ]

            axes[0, 1].bar(lengths, avg_lengths, color=['skyblue', 'lightgreen', 'salmon'])
            axes[0, 1].set_title('平均長度對比')
            axes[0, 1].set_ylabel('平均字符數')

        # 3. 質量問題分佈
        if 'quality_issues' in self.analysis_results:
            quality = self.analysis_results['quality_issues']
            issues = ['過短指令', '過短回答', '過長回答', '重複模式', '不完整回答']
            counts = [
                quality['very_short_instructions'],
                quality['very_short_outputs'],
                quality['very_long_outputs'],
                quality['repetitive_patterns'],
                quality['incomplete_responses']
            ]

            axes[1, 0].bar(issues, counts, color='orange', alpha=0.7)
            axes[1, 0].set_title('質量問題統計')
            axes[1, 0].set_ylabel('問題數量')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. 指令複雜度
        if 'complexity' in self.analysis_results:
            complexity = self.analysis_results['complexity']
            comp_types = list(complexity.keys())
            comp_values = list(complexity.values())

            axes[1, 1].pie(comp_values, labels=comp_types, autopct='%1.1f%%')
            axes[1, 1].set_title('指令複雜度分佈')

        plt.tight_layout()
        plt.savefig('instruction_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("可視化結果已保存到 instruction_data_analysis.png")

    def generate_instruction_report(self):
        """生成指令數據分析報告"""

        report = f"""# 指令微調數據集分析報告

## 數據集信息
- 樣本數量: {len(self.dataset):,}
- 分析時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 指令類型分析
"""

        if 'instruction_types' in self.analysis_results:
            types = self.analysis_results['instruction_types']
            for category, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(self.dataset) * 100
                report += f"- {category}: {count} ({percentage:.1f}%)\\n"

        if 'length_stats' in self.analysis_results:
            stats = self.analysis_results['length_stats']
            report += f"""
## 長度統計
- 指令平均長度: {stats['instruction']['avg_length']:.1f} 字符
- 輸入平均長度: {stats['input']['avg_length']:.1f} 字符
- 輸出平均長度: {stats['output']['avg_length']:.1f} 字符
- 輸入非空比例: {stats['input']['non_empty_ratio']:.1%}
"""

        if 'quality_issues' in self.analysis_results:
            quality = self.analysis_results['quality_issues']
            total = len(self.dataset)
            report += f"""
## 質量問題統計
- 過短指令: {quality['very_short_instructions']} ({quality['very_short_instructions']/total*100:.1f}%)
- 過短回答: {quality['very_short_outputs']} ({quality['very_short_outputs']/total*100:.1f}%)
- 過長回答: {quality['very_long_outputs']} ({quality['very_long_outputs']/total*100:.1f}%)
- 重複模式: {quality['repetitive_patterns']} ({quality['repetitive_patterns']/total*100:.1f}%)
- 不完整回答: {quality['incomplete_responses']} ({quality['incomplete_responses']/total*100:.1f}%)
"""

        if 'complexity' in self.analysis_results:
            complexity = self.analysis_results['complexity']
            report += "\\n## 指令複雜度分析\\n"
            for category, count in complexity.items():
                percentage = count / len(self.dataset) * 100
                report += f"- {category}: {count} ({percentage:.1f}%)\\n"

        report += """
## 改進建議
1. 對過短的指令和回答進行人工審核或自動過濾
2. 增加複雜多步指令的比例以提升模型推理能力
3. 保持指令類型的平衡分佈，避免某類指令過度集中
4. 對重複內容進行去重和多樣化處理
5. 建立更完善的質量檢查機制

## 數據質量建議
- 建議保留高質量樣本比例 > 90%
- 指令長度建議控制在 10-200 字符
- 輸出長度建議控制在 20-500 字符
- 定期進行人工抽檢以確保質量
"""

        with open('instruction_data_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("指令數據分析報告已保存到 instruction_data_report.md")
        return report

def run_instruction_analysis():
    """運行指令數據分析"""

    analyzer = InstructionDataAnalyzer()

    # 載入數據集
    analyzer.load_instruction_dataset()

    # 運行各項分析
    analyzer.analyze_instruction_types()
    analyzer.analyze_length_distribution()
    analyzer.analyze_quality_indicators()
    analyzer.analyze_instruction_complexity()

    # 生成可視化和報告
    analyzer.visualize_instruction_analysis()
    analyzer.generate_instruction_report()

    return analyzer

if __name__ == "__main__":
    analyzer = run_instruction_analysis()
```

## 完整實驗執行腳本

#### 一句話心法：總結「大米」和「菜譜」的質量，為如何做出「美味大餐」（好模型）提供數據層面的建議。

```python
# main_dataset_analysis.py
"""
數據集分析實驗主腳本
"""

def main():
    """主實驗流程"""

    print("=== LLM數據集分析實踐實驗 ===")
    print("本實驗將深入分析不同類型的LLM訓練數據\\n")

    # 任務1: 預訓練數據分析
    print("任務1: 預訓練數據集分析...")
    from pretraining_data_analysis import run_pretraining_analysis
    pretraining_analyzer = run_pretraining_analysis()

    print("\\n" + "="*60 + "\\n")

    # 任務2: 指令數據分析
    print("任務2: 指令微調數據集分析...")
    from instruction_data_analysis import run_instruction_analysis
    instruction_analyzer = run_instruction_analysis()

    print("\\n" + "="*60 + "\\n")

    # 任務3: 數據對比分析
    print("任務3: 數據集對比分析...")
    generate_comparative_analysis(pretraining_analyzer, instruction_analyzer)

    print("\\n=== 數據集分析實驗完成 ===")
    print("所有分析結果已保存，請查看生成的報告和圖表！")

def generate_comparative_analysis(pretraining_analyzer, instruction_analyzer):
    """生成對比分析報告"""

    report = f"""# 數據集對比分析報告

## 預訓練數據 vs 指令微調數據

### 數據規模對比
- 預訓練數據: {len(pretraining_analyzer.dataset) if pretraining_analyzer.dataset else 0:,} 樣本
- 指令數據: {len(instruction_analyzer.dataset) if instruction_analyzer.dataset else 0:,} 樣本

### 內容特性對比

#### 預訓練數據特點
1. **多樣性高**: 涵蓋各種主題和領域的自然文本
2. **結構自由**: 沒有固定的輸入輸出格式
3. **規模龐大**: 通常包含數十億到數萬億token
4. **質量參差**: 需要大量清洗和過濾工作

#### 指令數據特點
1. **結構化**: 明確的指令-輸入-輸出三元組格式
2. **目標明確**: 每個樣本都有明確的任務目標
3. **質量要求高**: 需要人工設計或精心篩選
4. **規模相對小**: 通常數萬到數百萬樣本

### 處理策略對比

#### 預訓練數據處理
- 重點：數據清洗、去重、質量篩選
- 挑戰：處理大規模數據、多語言處理
- 工具：分散式處理、自動化篩選

#### 指令數據處理
- 重點：格式統一、質量保證、平衡性
- 挑戰：指令多樣性、回答質量評估
- 工具：人工審核、自動化質量檢查

### 對模型訓練的影響

#### 預訓練階段
- 提供基礎語言理解能力
- 建立世界知識基礎
- 形成文本生成能力

#### 指令微調階段
- 學習遵循指令的能力
- 提升任務理解準確性
- 增強與人類交互的自然性

## 建議

### 數據集構建建議
1. **預訓練數據**: 注重多樣性和規模，建立完善的質量控制流程
2. **指令數據**: 注重質量和平衡性，確保指令類型的充分覆蓋

### 質量控制建議
1. 建立分層的數據質量評估體系
2. 實施自動化與人工相結合的審核流程
3. 定期更新和擴充數據集以保持時效性

### 倫理合規建議
1. 確保數據來源的合法性和透明性
2. 保護個人隱私和敏感信息
3. 避免有害和偏見內容的包含
"""

    with open('dataset_comparative_analysis.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("對比分析報告已保存到 dataset_comparative_analysis.md")

if __name__ == "__main__":
    main()
```

## 實驗指南

### 環境準備
```bash
# 激活虛擬環境
source 00-Course_Setup/.venv/bin/activate

# 安裝依賴包
pip install langdetect wordcloud jieba seaborn

# 進入實驗目錄
cd 00-LLM_Fundamentals/02-Labs/Lab-0.3-Dataset_Analysis
```

### 執行步驟
1. 運行完整分析：`python main_dataset_analysis.py`
2. 單獨運行預訓練分析：`python 01_pretraining_data_analysis.py`
3. 單獨運行指令分析：`python 02_instruction_data_analysis.py`

### 預期產出
- 多個分析報告（Markdown格式）
- 可視化圖表（PNG格式）
- 數據統計結果（CSV/JSON格式）

## 實驗報告要求

### 必答問題
1. **數據質量評估**：如何定義和衡量數據質量？
2. **數據平衡性**：不同類型數據的平衡性如何影響模型性能？
3. **處理策略**：針對發現的問題，提出相應的數據處理策略
4. **倫理考量**：在數據收集和處理過程中需要注意哪些倫理問題？

### 延伸思考
1. 如何自動化數據質量檢查流程？
2. 不同語言和文化的數據集可能存在哪些差異？
3. 如何平衡數據集的多樣性和質量？

這個Lab提供了全面的數據集分析實踐，幫助學員建立數據工程思維和質量意識。