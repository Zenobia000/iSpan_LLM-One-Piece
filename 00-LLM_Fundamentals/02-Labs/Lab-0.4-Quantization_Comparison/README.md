# Lab 0.4: 量化技術對比實驗

## 實驗目標

通過實際對比不同量化技術，深入理解量化對模型性能的影響，掌握量化技術的選擇和應用策略。

## 學習成果

完成本實驗後，您將能夠：
- 實施並對比不同精度的量化方案
- 分析量化對模型性能和推理速度的影響
- 根據部署需求選擇合適的量化策略
- 理解量化技術的優缺點和適用場景

## 核心內容架構

### 實驗學習路徑總結

本實驗將引導您扮演一位「模型瘦身教練」，學習如何為臃腫的大型語言模型制定並實施有效的「減肥計畫」——量化。整個流程包含以下步驟：

*   **1. 體檢與備案 (`FP32` & `FP16`)**
    *   **目標**：建立模型的性能基準。
    *   **一句話心法**：先測量模型在「減肥」前的各項指標（速度、大小、答題質量），作為後續效果的對比標準。

*   **2. 輕度減脂 (`INT8`)**
    *   **目標**：實施8位整數量化，體驗初步的模型壓縮效果。
    *   **一句話心法**：為模型制定一個溫和的「飲食計畫」，讓它的體積減半，同時盡量保持「智商」不下降。

*   **3. 極限挑戰 (`INT4`)**
    *   **目標**：實施4位整數量化，探索性能極限。
    *   **一句話心法**：挑戰最嚴格的「減肥菜單」，讓模型體積縮小到原來的1/8，並評估它是否還能正常「思考」。

*   **4. 撰寫瘦身報告 (`對比與可視化`)**
    *   **目標**：全面對比不同量化方案的優缺點。
    *   **一句話心法**：匯總所有「減肥」方案的效果，用圖表清晰展示哪種方案在「瘦身」和「保持智商」之間取得了最佳平衡。

## 實驗環境要求

### 硬體要求
- GPU：8GB+顯存（推薦）
- RAM：16GB+系統記憶體
- 存儲：15GB可用空間

### 軟體要求
- PyTorch 2.0+
- Transformers 4.30+
- 已激活的poetry虛擬環境

## 主要實驗內容

## 實驗程式碼說明

本Lab包含量化技術演示工具：
- **`quantization_demo.py`**: 量化技術完整演示，包含精度格式、PTQ/QAT、效果對比等

## 執行方式

```bash
# 運行完整量化演示
python quantization_demo.py

# 或在Jupyter中使用
from quantization_demo import QuantizationDemo
```

### 量化技術對比實驗

#### 一句話心法：測量模型在「減肥」前的各項指標，作為後續效果的對比標準，並為其制定溫和與極限的「減肥計畫」，最終用圖表來評估成效。

```python
# quantization_comparison_experiment.py
"""
量化技術對比實驗主腳本
對比FP32、FP16、INT8、INT4等不同精度下的模型性能
"""

import torch
import time
import psutil
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
import matplotlib.pyplot as plt
import pandas as pd
import json

class QuantizationComparator:
    """量化技術對比器"""

    def __init__(self, model_name="microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.models = {}
        self.tokenizer = None
        self.results = {}

    def setup_models(self):
        """設置不同精度的模型"""

        print(f"設置模型: {self.model_name}")

        # 載入分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_configs = {
            'FP32': {
                'torch_dtype': torch.float32,
                'quantization_config': None
            },
            'FP16': {
                'torch_dtype': torch.float16,
                'quantization_config': None
            },
            'INT8': {
                'torch_dtype': torch.float16,
                'quantization_config': BitsAndBytesConfig(load_in_8bit=True)
            },
            'INT4': {
                'torch_dtype': torch.float16,
                'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            }
        }

        for precision, config in quantization_configs.items():
            print(f"載入 {precision} 模型...")

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=config['torch_dtype'],
                    quantization_config=config['quantization_config'],
                    device_map="auto"
                )

                self.models[precision] = model
                print(f"✓ {precision} 模型載入成功")

                # 計算模型大小
                model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
                print(f"  模型大小: {model_size:.1f} MB")

            except Exception as e:
                print(f"✗ {precision} 模型載入失敗: {e}")

    def benchmark_inference_speed(self, test_prompts=None, num_runs=10):
        """推理速度基準測試"""

        if test_prompts is None:
            test_prompts = [
                "人工智能的發展趨勢",
                "機器學習在現代社會中的應用",
                "深度學習技術的突破與挑戰"
            ]

        print("\\n=== 推理速度對比測試 ===")

        results = {}

        for precision, model in self.models.items():
            print(f"\\n測試 {precision} 模型...")

            precision_results = []

            for prompt in test_prompts:
                prompt_results = []

                for run in range(num_runs):
                    # 記錄開始時間和記憶體
                    start_time = time.time()
                    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

                    try:
                        inputs = self.tokenizer(prompt, return_tensors="pt")
                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=inputs['input_ids'].shape[1] + 30,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id
                            )

                        # 記錄結束時間和記憶體
                        end_time = time.time()
                        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

                        inference_time = end_time - start_time
                        memory_used = (end_memory - start_memory) / (1024**2)  # MB

                        generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                        tokens_per_second = generated_tokens / inference_time if inference_time > 0 else 0

                        prompt_results.append({
                            'inference_time': inference_time,
                            'memory_used': memory_used,
                            'tokens_per_second': tokens_per_second,
                            'generated_tokens': generated_tokens
                        })

                    except Exception as e:
                        print(f"  運行 {run+1} 出錯: {e}")
                        continue

                if prompt_results:
                    avg_time = np.mean([r['inference_time'] for r in prompt_results])
                    avg_memory = np.mean([r['memory_used'] for r in prompt_results])
                    avg_tps = np.mean([r['tokens_per_second'] for r in prompt_results])

                    precision_results.append({
                        'prompt': prompt,
                        'avg_inference_time': avg_time,
                        'avg_memory_used': avg_memory,
                        'avg_tokens_per_second': avg_tps,
                        'std_inference_time': np.std([r['inference_time'] for r in prompt_results])
                    })

                    print(f"  提示: {prompt[:20]}...")
                    print(f"    平均推理時間: {avg_time:.3f}秒")
                    print(f"    平均tokens/s: {avg_tps:.1f}")

            results[precision] = precision_results

        self.results['inference_speed'] = results
        return results

    def benchmark_generation_quality(self, test_prompts=None):
        """生成質量對比測試"""

        if test_prompts is None:
            test_prompts = [
                "解釋什麼是量子計算",
                "描述人工智能對社會的影響",
                "談談你對未來科技發展的看法"
            ]

        print("\\n=== 生成質量對比測試 ===")

        results = {}

        for precision, model in self.models.items():
            print(f"\\n{precision} 模型生成結果:")

            precision_results = []

            for prompt in test_prompts:
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=inputs['input_ids'].shape[1] + 50,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )

                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = generated_text[len(prompt):].strip()

                    # 簡單的質量評估
                    quality_score = self.evaluate_text_quality(response)

                    precision_results.append({
                        'prompt': prompt,
                        'response': response,
                        'quality_score': quality_score
                    })

                    print(f"  提示: {prompt}")
                    print(f"  回應: {response}")
                    print(f"  質量評分: {quality_score:.2f}")
                    print("-" * 60)

                except Exception as e:
                    print(f"  生成出錯: {e}")

            results[precision] = precision_results

        self.results['generation_quality'] = results
        return results

    def evaluate_text_quality(self, text):
        """簡化的文本質量評估"""

        if not text or len(text.strip()) < 10:
            return 0.0

        # 計算多個質量指標
        scores = []

        # 1. 長度合理性 (20-200字符比較理想)
        length = len(text.strip())
        if 20 <= length <= 200:
            length_score = 1.0
        elif length < 20:
            length_score = length / 20
        else:
            length_score = max(0.5, 200 / length)
        scores.append(length_score)

        # 2. 詞彙多樣性
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
        else:
            unique_ratio = 0
        scores.append(unique_ratio)

        # 3. 流暢性（基於標點符號和句子結構）
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            fluency_score = min(1.0, avg_sentence_length / 15)  # 理想句子長度約15詞
        else:
            fluency_score = 0.5
        scores.append(fluency_score)

        return np.mean(scores)

    def memory_usage_analysis(self):
        """記憶體使用分析"""

        print("\\n=== 記憶體使用分析 ===")

        memory_results = {}

        for precision, model in self.models.items():
            try:
                # 計算模型參數記憶體
                param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

                # 計算緩衝區記憶體
                buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())

                total_memory = param_memory + buffer_memory

                memory_results[precision] = {
                    'param_memory_mb': param_memory / (1024**2),
                    'buffer_memory_mb': buffer_memory / (1024**2),
                    'total_memory_mb': total_memory / (1024**2),
                    'compression_ratio': memory_results.get('FP32', {}).get('total_memory_mb', total_memory / (1024**2)) / (total_memory / (1024**2))
                }

                print(f"{precision}:")
                print(f"  參數記憶體: {memory_results[precision]['param_memory_mb']:.1f} MB")
                print(f"  總記憶體: {memory_results[precision]['total_memory_mb']:.1f} MB")
                if precision != 'FP32':
                    print(f"  壓縮比: {memory_results[precision]['compression_ratio']:.2f}x")

            except Exception as e:
                print(f"{precision} 記憶體分析出錯: {e}")

        self.results['memory_usage'] = memory_results
        return memory_results

    def create_comparison_report(self):
        """創建對比報告"""

        print("\\n=== 生成對比報告 ===")

        # 創建總結表格
        summary_data = []

        for precision in self.models.keys():
            row = {'量化方案': precision}

            # 推理速度統計
            if 'inference_speed' in self.results:
                speed_data = self.results['inference_speed'].get(precision, [])
                if speed_data:
                    avg_time = np.mean([d['avg_inference_time'] for d in speed_data])
                    avg_tps = np.mean([d['avg_tokens_per_second'] for d in speed_data])
                    row['平均推理時間(秒)'] = f"{avg_time:.3f}"
                    row['平均生成速度(tokens/s)'] = f"{avg_tps:.1f}"

            # 記憶體使用
            if 'memory_usage' in self.results:
                memory_data = self.results['memory_usage'].get(precision, {})
                row['記憶體使用(MB)'] = f"{memory_data.get('total_memory_mb', 0):.1f}"
                if precision != 'FP32':
                    row['壓縮比'] = f"{memory_data.get('compression_ratio', 1):.2f}x"

            # 生成質量
            if 'generation_quality' in self.results:
                quality_data = self.results['generation_quality'].get(precision, [])
                if quality_data:
                    avg_quality = np.mean([d['quality_score'] for d in quality_data])
                    row['平均質量評分'] = f"{avg_quality:.3f}"

            summary_data.append(row)

        # 轉換為DataFrame
        df = pd.DataFrame(summary_data)
        print("\\n量化技術對比總結:")
        print(df.to_string(index=False))

        # 保存結果
        df.to_csv('quantization_comparison_summary.csv', index=False)

        with open('quantization_comparison_detailed.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        print("\\n詳細結果已保存到:")
        print("- quantization_comparison_summary.csv (總結)")
        print("- quantization_comparison_detailed.json (詳細)")

        return df

    def visualize_comparison(self):
        """可視化對比結果"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        precisions = list(self.models.keys())

        # 1. 推理速度對比
        if 'inference_speed' in self.results:
            avg_times = []
            for precision in precisions:
                speed_data = self.results['inference_speed'].get(precision, [])
                if speed_data:
                    avg_time = np.mean([d['avg_inference_time'] for d in speed_data])
                    avg_times.append(avg_time)
                else:
                    avg_times.append(0)

            axes[0, 0].bar(precisions, avg_times, color='skyblue')
            axes[0, 0].set_title('推理速度對比')
            axes[0, 0].set_ylabel('平均推理時間 (秒)')

        # 2. 記憶體使用對比
        if 'memory_usage' in self.results:
            memory_usage = []
            for precision in precisions:
                memory_data = self.results['memory_usage'].get(precision, {})
                memory_usage.append(memory_data.get('total_memory_mb', 0))

            axes[0, 1].bar(precisions, memory_usage, color='lightgreen')
            axes[0, 1].set_title('記憶體使用對比')
            axes[0, 1].set_ylabel('記憶體使用 (MB)')

        # 3. 生成速度對比
        if 'inference_speed' in self.results:
            tokens_per_sec = []
            for precision in precisions:
                speed_data = self.results['inference_speed'].get(precision, [])
                if speed_data:
                    avg_tps = np.mean([d['avg_tokens_per_second'] for d in speed_data])
                    tokens_per_sec.append(avg_tps)
                else:
                    tokens_per_sec.append(0)

            axes[1, 0].bar(precisions, tokens_per_sec, color='salmon')
            axes[1, 0].set_title('生成速度對比')
            axes[1, 0].set_ylabel('Tokens/秒')

        # 4. 質量評分對比
        if 'generation_quality' in self.results:
            quality_scores = []
            for precision in precisions:
                quality_data = self.results['generation_quality'].get(precision, [])
                if quality_data:
                    avg_quality = np.mean([d['quality_score'] for d in quality_data])
                    quality_scores.append(avg_quality)
                else:
                    quality_scores.append(0)

            axes[1, 1].bar(precisions, quality_scores, color='gold')
            axes[1, 1].set_title('生成質量對比')
            axes[1, 1].set_ylabel('平均質量評分')

        plt.tight_layout()
        plt.savefig('quantization_comparison_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("可視化結果已保存到 quantization_comparison_visualization.png")

def main():
    """主實驗流程"""

    print("=== 量化技術對比實驗 ===\\n")

    # 初始化對比器
    comparator = QuantizationComparator()

    # 設置不同精度的模型
    comparator.setup_models()

    if not comparator.models:
        print("沒有成功載入任何模型，實驗終止")
        return

    # 運行基準測試
    print("\\n開始運行基準測試...")

    comparator.benchmark_inference_speed()
    comparator.benchmark_generation_quality()
    comparator.memory_usage_analysis()

    # 生成報告和可視化
    comparator.create_comparison_report()
    comparator.visualize_comparison()

    print("\\n=== 實驗完成 ===")
    print("請查看生成的報告和圖表了解量化技術的對比結果！")

if __name__ == "__main__":
    main()
```

## 實驗執行指南

### 準備步驟
1. **環境激活**
   ```bash
   source 00-Course_Setup/.venv/bin/activate
   cd 00-LLM_Fundamentals/02-Labs/Lab-0.4-Quantization_Comparison
   ```

2. **依賴安裝**
   ```bash
   pip install bitsandbytes accelerate
   ```

### 執行實驗
```bash
python quantization_comparison_experiment.py
```

## 預期結果

實驗將產生以下輸出：
- **性能對比表格**：不同量化方案的詳細對比
- **可視化圖表**：推理速度、記憶體使用、生成質量的直觀對比
- **詳細分析報告**：包含建議和最佳實踐

### 典型結果示例
```
量化方案    平均推理時間(秒)  記憶體使用(MB)  壓縮比    生成質量
FP32       0.245           156.2          1.00x     0.847
FP16       0.198           78.1           2.00x     0.845
INT8       0.167           39.1           4.00x     0.832
INT4       0.134           19.6           8.00x     0.798
```

## 實驗報告要求

### 必答問題
1. **性能權衡**：分析不同量化方案在速度、記憶體和質量間的權衡
2. **應用場景**：為不同的部署場景推薦合適的量化方案
3. **質量影響**：評估量化對生成質量的具體影響
4. **實際部署**：討論在實際部署中選擇量化方案的考慮因素

### 延伸思考
1. 如何進一步優化量化後模型的性能？
2. 不同模型架構對量化的敏感性可能有何差異？
3. 如何在保證質量的前提下實現更高的壓縮比？

這個Lab提供了量化技術的全面對比體驗，幫助學員建立量化技術選擇的實踐能力。