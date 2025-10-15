#!/usr/bin/env python3
"""
量化技術演示程式碼
展示不同量化方法的實現和效果對比
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
from typing import Dict, List, Tuple
import json
import pandas as pd
from datetime import datetime

class QuantizationDemo:
    """量化技術演示類"""

    def __init__(self):
        self.quantization_results = {}
        self.precision_bytes = {
            'fp32': 4, 'fp16': 2, 'bf16': 2,
            'int8': 1, 'int4': 0.5, 'nf4': 0.5
        }

    def demonstrate_precision_formats(self):
        """演示不同精度格式"""

        print("=== 數值精度格式演示 ===")

        # 創建測試數值
        test_values = [
            0.123456789,
            1000.123456,
            0.000001234,
            -456.789123,
            1e-8,
            1e8
        ]

        precision_demo = {}

        for precision in ['fp32', 'fp16', 'bf16']:
            precision_demo[precision] = []

            for value in test_values:
                if precision == 'fp32':
                    quantized = np.float32(value)
                elif precision == 'fp16':
                    # 模擬fp16精度限制
                    quantized = np.float16(value)
                elif precision == 'bf16':
                    # 模擬bf16（與fp32範圍相同但精度較低）
                    quantized = np.float32(np.float16(value))  # 簡化模擬

                error = abs(float(quantized) - value) / (abs(value) + 1e-10)

                precision_demo[precision].append({
                    'original': value,
                    'quantized': float(quantized),
                    'error': error
                })

        # 顯示結果
        for precision, results in precision_demo.items():
            avg_error = np.mean([r['error'] for r in results])
            print(f"{precision.upper()}: 平均相對誤差 = {avg_error:.6f}")

        return precision_demo

    def demonstrate_linear_quantization(self):
        """演示線性量化過程"""

        print("\n=== 線性量化演示 ===")

        # 創建模擬權重分佈
        np.random.seed(42)
        weights = np.random.normal(0, 0.5, 1000)  # 正態分佈權重

        quantization_demo = {}

        # INT8 對稱量化
        print("1. INT8 對稱量化:")
        int8_result = self._apply_symmetric_quantization(weights, 8)
        quantization_demo['int8_symmetric'] = int8_result

        # INT8 非對稱量化
        print("2. INT8 非對稱量化:")
        int8_asymm_result = self._apply_asymmetric_quantization(weights, 8)
        quantization_demo['int8_asymmetric'] = int8_asymm_result

        # INT4 量化
        print("3. INT4 量化:")
        int4_result = self._apply_symmetric_quantization(weights, 4)
        quantization_demo['int4_symmetric'] = int4_result

        return quantization_demo

    def _apply_symmetric_quantization(self, weights: np.ndarray, bits: int) -> Dict:
        """應用對稱量化"""

        # 對稱量化參數
        max_val = np.max(np.abs(weights))
        qmax = 2**(bits-1) - 1
        scale = max_val / qmax

        # 量化
        quantized_weights = np.round(weights / scale)
        quantized_weights = np.clip(quantized_weights, -qmax, qmax)

        # 反量化
        dequantized_weights = quantized_weights * scale

        # 計算量化誤差
        mse = np.mean((weights - dequantized_weights) ** 2)
        snr = 10 * np.log10(np.var(weights) / mse) if mse > 0 else float('inf')

        print(f"   Scale: {scale:.6f}")
        print(f"   MSE: {mse:.6f}")
        print(f"   SNR: {snr:.2f} dB")

        return {
            'type': 'symmetric',
            'bits': bits,
            'scale': scale,
            'zero_point': 0,
            'mse': mse,
            'snr_db': snr,
            'quantized_weights': quantized_weights,
            'dequantized_weights': dequantized_weights
        }

    def _apply_asymmetric_quantization(self, weights: np.ndarray, bits: int) -> Dict:
        """應用非對稱量化"""

        # 非對稱量化參數
        min_val = np.min(weights)
        max_val = np.max(weights)
        qmin = 0
        qmax = 2**bits - 1

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale

        # 量化
        quantized_weights = np.round(weights / scale + zero_point)
        quantized_weights = np.clip(quantized_weights, qmin, qmax)

        # 反量化
        dequantized_weights = (quantized_weights - zero_point) * scale

        # 計算量化誤差
        mse = np.mean((weights - dequantized_weights) ** 2)
        snr = 10 * np.log10(np.var(weights) / mse) if mse > 0 else float('inf')

        print(f"   Scale: {scale:.6f}")
        print(f"   Zero point: {zero_point:.2f}")
        print(f"   MSE: {mse:.6f}")
        print(f"   SNR: {snr:.2f} dB")

        return {
            'type': 'asymmetric',
            'bits': bits,
            'scale': scale,
            'zero_point': zero_point,
            'mse': mse,
            'snr_db': snr,
            'quantized_weights': quantized_weights,
            'dequantized_weights': dequantized_weights
        }

    def demonstrate_ptq_quantization(self, model_name: str = "microsoft/DialoGPT-small"):
        """演示PTQ量化"""

        print(f"\n=== PTQ量化演示: {model_name} ===")

        try:
            # 載入原始模型
            print("1. 載入原始FP16模型...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            original_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # 測試原始模型
            original_performance = self._benchmark_model(original_model, tokenizer, "原始FP16")

            # INT8 PTQ量化
            print("2. 執行INT8 PTQ量化...")
            int8_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto"
            )

            int8_performance = self._benchmark_model(int8_model, tokenizer, "INT8量化")

            # INT4 PTQ量化
            print("3. 執行INT4 NF4量化...")
            int4_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                ),
                device_map="auto"
            )

            int4_performance = self._benchmark_model(int4_model, tokenizer, "INT4量化")

            # 整理對比結果
            ptq_comparison = {
                'original_fp16': original_performance,
                'int8_ptq': int8_performance,
                'int4_nf4': int4_performance
            }

            self.quantization_results['ptq_demo'] = ptq_comparison

            return ptq_comparison

        except Exception as e:
            print(f"PTQ量化演示失敗: {e}")
            return None

    def _benchmark_model(self, model, tokenizer, model_type: str) -> Dict:
        """基準測試模型性能"""

        print(f"  測試 {model_type} 模型性能...")

        # 測試prompt
        test_prompts = [
            "人工智能的發展",
            "機器學習應用",
            "深度學習技術"
        ]

        # 計算模型大小
        model_size_mb = sum(
            param.numel() * param.element_size() for param in model.parameters()
        ) / (1024**2)

        # 性能測試
        inference_times = []
        generated_responses = []

        for prompt in test_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                start_time = time.time()

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 20,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()
                generated_responses.append(response)

            except Exception as e:
                print(f"    生成出錯: {e}")
                inference_times.append(float('inf'))
                generated_responses.append("ERROR")

        # 計算平均性能
        avg_inference_time = np.mean([t for t in inference_times if t != float('inf')])
        tokens_per_second = 20 / avg_inference_time if avg_inference_time > 0 else 0

        return {
            'model_type': model_type,
            'model_size_mb': model_size_mb,
            'avg_inference_time': avg_inference_time,
            'tokens_per_second': tokens_per_second,
            'generated_responses': generated_responses,
            'functional_test_passed': all('ERROR' not in resp for resp in generated_responses)
        }

    def analyze_quantization_effects(self):
        """分析量化效果"""

        if 'ptq_demo' not in self.quantization_results:
            print("請先運行PTQ量化演示")
            return None

        print("\n=== 量化效果分析 ===")

        ptq_results = self.quantization_results['ptq_demo']

        # 創建對比表格
        comparison_data = []

        for model_type, performance in ptq_results.items():
            if performance:
                row = {
                    '模型類型': performance['model_type'],
                    '模型大小(MB)': f"{performance['model_size_mb']:.1f}",
                    '推理時間(s)': f"{performance['avg_inference_time']:.3f}",
                    'Tokens/s': f"{performance['tokens_per_second']:.1f}",
                    '功能正常': "✅" if performance['functional_test_passed'] else "❌"
                }

                # 計算相對於原始模型的變化
                if model_type != 'original_fp16':
                    original = ptq_results['original_fp16']
                    if original:
                        size_ratio = performance['model_size_mb'] / original['model_size_mb']
                        speed_ratio = original['avg_inference_time'] / performance['avg_inference_time']

                        row['大小比例'] = f"{size_ratio:.2f}x"
                        row['速度提升'] = f"{speed_ratio:.2f}x"

                comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        print("量化效果對比:")
        print(comparison_df.to_string(index=False))

        # 分析關鍵洞察
        insights = self._generate_quantization_insights(ptq_results)

        return {
            'comparison_table': comparison_df,
            'insights': insights,
            'recommended_strategy': self._recommend_quantization_strategy(ptq_results)
        }

    def _generate_quantization_insights(self, results: Dict) -> List[str]:
        """生成量化洞察"""

        insights = []

        if 'original_fp16' in results and 'int8_ptq' in results:
            original = results['original_fp16']
            int8 = results['int8_ptq']

            if original and int8:
                compression_ratio = original['model_size_mb'] / int8['model_size_mb']
                speedup = original['avg_inference_time'] / int8['avg_inference_time']

                insights.append(f"INT8量化實現 {compression_ratio:.1f}x 模型壓縮")
                insights.append(f"推理速度提升 {speedup:.1f}x")

        if 'int4_nf4' in results:
            int4 = results['int4_nf4']
            if int4 and int4['functional_test_passed']:
                insights.append("INT4 NF4量化在保持功能性的同時實現極致壓縮")
            elif int4:
                insights.append("INT4量化可能導致功能性問題，需謹慎使用")

        insights.append("量化技術需要在壓縮比、性能和準確性間找平衡")

        return insights

    def _recommend_quantization_strategy(self, results: Dict) -> str:
        """推薦量化策略"""

        # 基於結果推薦最佳策略
        strategies = []

        for model_type, performance in results.items():
            if performance and performance['functional_test_passed']:
                # 計算綜合評分：壓縮效果 + 性能提升
                if model_type != 'original_fp16':
                    original = results['original_fp16']
                    if original:
                        compression_score = original['model_size_mb'] / performance['model_size_mb']
                        speed_score = original['avg_inference_time'] / performance['avg_inference_time']
                        overall_score = compression_score * 0.6 + speed_score * 0.4

                        strategies.append({
                            'strategy': model_type,
                            'score': overall_score
                        })

        if strategies:
            best_strategy = max(strategies, key=lambda x: x['score'])
            return f"推薦策略: {best_strategy['strategy']} (綜合評分: {best_strategy['score']:.2f})"
        else:
            return "無法確定最佳策略，建議進一步測試"

    def demonstrate_quantization_aware_training(self):
        """演示量化感知訓練概念"""

        print("\n=== 量化感知訓練(QAT)概念演示 ===")

        # 模擬QAT訓練過程
        qat_simulation = {
            'training_phases': [
                {
                    'phase': 'normal_training',
                    'description': '正常FP32訓練',
                    'precision': 'fp32',
                    'accuracy': 0.95
                },
                {
                    'phase': 'qat_insertion',
                    'description': '插入偽量化層',
                    'precision': 'fp32_with_fake_quantization',
                    'accuracy': 0.93
                },
                {
                    'phase': 'qat_fine_tuning',
                    'description': '量化感知微調',
                    'precision': 'fp32_with_fake_quantization',
                    'accuracy': 0.94
                },
                {
                    'phase': 'real_quantization',
                    'description': '轉換為真實量化',
                    'precision': 'int8',
                    'accuracy': 0.94
                }
            ],
            'key_concepts': {
                'fake_quantization': '訓練時模擬量化過程但保持FP32精度計算',
                'straight_through_estimator': '量化函數的梯度近似方法',
                'learnable_quantization_params': '量化參數作為可學習變量'
            }
        }

        # 顯示QAT過程
        for phase in qat_simulation['training_phases']:
            print(f"階段: {phase['phase']}")
            print(f"  描述: {phase['description']}")
            print(f"  精度: {phase['precision']}")
            print(f"  準確率: {phase['accuracy']:.3f}")

        print("\nQAT關鍵概念:")
        for concept, description in qat_simulation['key_concepts'].items():
            print(f"  {concept}: {description}")

        return qat_simulation

    def compare_quantization_methods(self):
        """對比不同量化方法"""

        print("\n=== 量化方法對比 ===")

        # 量化方法特點對比
        quantization_methods = {
            'PTQ': {
                '實施複雜度': '低',
                '準確性保持': '中等',
                '壓縮效果': '良好',
                '適用場景': '快速部署',
                '校準數據需求': '少量(100-1000樣本)',
                '訓練時間': '無需訓練'
            },
            'QAT': {
                '實施複雜度': '高',
                '準確性保持': '優秀',
                '壓縮效果': '良好',
                '適用場景': '高質量要求',
                '校準數據需求': '完整訓練集',
                '訓練時間': '額外訓練時間'
            },
            'GPTQ': {
                '實施複雜度': '中等',
                '準確性保持': '良好',
                '壓縮效果': '優秀',
                '適用場景': '大模型壓縮',
                '校準數據需求': '少量高質量樣本',
                '訓練時間': '一次性校準'
            },
            'AWQ': {
                '實施複雜度': '中等',
                '準確性保持': '優秀',
                '壓縮效果': '良好',
                '適用場景': '平衡性要求',
                '校準數據需求': '少量樣本',
                '訓練時間': '快速校準'
            }
        }

        # 轉換為DataFrame
        comparison_df = pd.DataFrame(quantization_methods).T

        print("量化方法對比:")
        print(comparison_df.to_string())

        # 生成選擇建議
        selection_guide = {
            '快速部署': 'PTQ - 實施簡單，效果可接受',
            '高質量要求': 'QAT - 準確性最佳，需要額外訓練',
            '大模型壓縮': 'GPTQ - 專門針對大模型優化',
            '平衡性要求': 'AWQ - 在各個方面都有良好表現'
        }

        print("\n量化方法選擇指南:")
        for scenario, recommendation in selection_guide.items():
            print(f"  {scenario}: {recommendation}")

        return {
            'comparison_table': comparison_df,
            'selection_guide': selection_guide
        }

    def visualize_quantization_effects(self):
        """可視化量化效果"""

        print("\n=== 生成量化效果可視化 ===")

        # 創建量化精度對比圖
        precisions = ['FP32', 'FP16', 'INT8', 'INT4']
        model_sizes = [100, 50, 25, 12.5]  # 相對大小
        inference_speeds = [1.0, 1.8, 2.5, 3.2]  # 相對速度
        accuracy_scores = [1.0, 0.998, 0.985, 0.97]  # 相對準確率

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 模型大小對比
        bars1 = ax1.bar(precisions, model_sizes, color='skyblue', alpha=0.7)
        ax1.set_title('模型大小對比')
        ax1.set_ylabel('相對大小')
        for bar, size in zip(bars1, model_sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{size}%', ha='center', va='bottom')

        # 2. 推理速度對比
        bars2 = ax2.bar(precisions, inference_speeds, color='lightgreen', alpha=0.7)
        ax2.set_title('推理速度對比')
        ax2.set_ylabel('相對速度')
        for bar, speed in zip(bars2, inference_speeds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{speed:.1f}x', ha='center', va='bottom')

        # 3. 準確率對比
        bars3 = ax3.bar(precisions, accuracy_scores, color='salmon', alpha=0.7)
        ax3.set_title('準確率對比')
        ax3.set_ylabel('相對準確率')
        ax3.set_ylim(0.95, 1.005)
        for bar, acc in zip(bars3, accuracy_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom')

        # 4. 綜合性能雷達圖
        categories = ['模型大小\n(越小越好)', '推理速度\n(越快越好)', '準確率\n(越高越好)', '實施難度\n(越簡單越好)']

        # 歸一化數據
        fp32_scores = [0.25, 1.0, 1.0, 1.0]  # FP32作為基線
        int8_scores = [1.0, 0.8, 0.985, 0.9]  # INT8
        int4_scores = [1.0, 0.9, 0.97, 0.7]   # INT4

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 閉合雷達圖

        for scores, label, color in [
            (fp32_scores + fp32_scores[:1], 'FP32', 'blue'),
            (int8_scores + int8_scores[:1], 'INT8', 'green'),
            (int4_scores + int4_scores[:1], 'INT4', 'red')
        ]:
            ax4.plot(angles, scores, 'o-', linewidth=2, label=label, color=color)
            ax4.fill(angles, scores, alpha=0.1, color=color)

        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1.1)
        ax4.set_title('綜合性能對比')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('quantization_effects_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("可視化圖表已保存: quantization_effects_comparison.png")

    def generate_quantization_report(self) -> str:
        """生成量化技術報告"""

        report = f"""# 量化技術演示分析報告

生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 量化技術概述

本報告展示了不同量化技術的實現原理和效果對比，為實際應用中的量化策略選擇提供參考。

## 核心發現

### 1. 精度格式特性
- **FP32**: 最高精度，最大存儲開銷
- **FP16**: 精度略微下降，存儲減半
- **INT8**: 較大精度損失，存儲進一步減少
- **INT4**: 極致壓縮，需要特殊技術保證可用性

### 2. 量化方法比較
"""

        # 添加實際測試結果
        if 'ptq_demo' in self.quantization_results:
            results = self.quantization_results['ptq_demo']

            report += "### PTQ量化實測效果\n"

            for model_type, performance in results.items():
                if performance:
                    report += f"""
**{performance['model_type']}**:
- 模型大小: {performance['model_size_mb']:.1f} MB
- 推理時間: {performance['avg_inference_time']:.3f} 秒
- 推理速度: {performance['tokens_per_second']:.1f} tokens/s
- 功能完整性: {'正常' if performance['functional_test_passed'] else '有問題'}
"""

        report += """
## 實際應用建議

### 量化策略選擇
1. **原型開發階段**: 使用FP16，平衡精度和效率
2. **生產部署階段**: 根據硬體條件選擇INT8或INT4
3. **邊緣設備部署**: 優先考慮INT4或更激進的量化
4. **高精度要求場景**: 使用QAT保證量化後性能

### 實施建議
1. **充分測試**: 量化後必須進行全面功能和性能測試
2. **漸進實施**: 從保守的量化策略開始，逐步優化
3. **監控部署**: 部署後持續監控量化模型的表現
4. **備選方案**: 準備回滾到原始模型的方案

### 技術選型
- **快速部署**: PTQ + BitsAndBytesConfig
- **高質量要求**: QAT + 完整訓練流程
- **大模型處理**: GPTQ + 專業校準數據
- **平衡方案**: AWQ + 自動精度搜索

## 未來發展趨勢

1. **更低精度**: FP8、FP6等更低精度格式的發展
2. **硬體協同**: 與專用硬體深度協同的量化技術
3. **自動化**: 自動量化參數搜索和優化
4. **多模態**: 多模態模型的量化技術

---
*本報告基於演示數據生成，實際應用時請根據具體模型和需求進行詳細測試。*
"""

        return report

    def run_complete_quantization_demo(self):
        """運行完整量化演示"""

        print("🔢 量化技術完整演示")
        print("=" * 60)

        try:
            # 1. 精度格式演示
            precision_demo = self.demonstrate_precision_formats()

            # 2. 線性量化演示
            linear_quant_demo = self.demonstrate_linear_quantization()

            # 3. PTQ量化演示
            ptq_demo = self.demonstrate_ptq_quantization()

            # 4. QAT概念演示
            qat_demo = self.demonstrate_quantization_aware_training()

            # 5. 量化方法對比
            methods_comparison = self.compare_quantization_methods()

            # 6. 效果分析
            if ptq_demo:
                effects_analysis = self.analyze_quantization_effects()

                # 7. 可視化
                self.visualize_quantization_effects()

            # 8. 生成報告
            report = self.generate_quantization_report()

            # 保存報告
            with open('quantization_demo_report.md', 'w', encoding='utf-8') as f:
                f.write(report)

            print("\n✅ 量化演示完成！")
            print("📁 生成文件:")
            print("   - quantization_demo_report.md")
            print("   - quantization_effects_comparison.png")

            print("\n🎓 關鍵學習要點:")
            print("1. 量化是精度和效率之間的權衡")
            print("2. 不同量化方法適用於不同場景")
            print("3. PTQ適合快速部署，QAT適合高質量要求")
            print("4. 量化後必須進行充分的測試驗證")

            return {
                'precision_demo': precision_demo,
                'linear_quant_demo': linear_quant_demo,
                'ptq_demo': ptq_demo,
                'qat_demo': qat_demo,
                'methods_comparison': methods_comparison
            }

        except Exception as e:
            print(f"❌ 量化演示失敗: {e}")
            return None

def main():
    """主函數"""

    print("量化技術演示程式")
    print("本程式將展示量化技術的原理和實際應用\n")

    # 創建演示實例
    demo = QuantizationDemo()

    # 運行完整演示
    results = demo.run_complete_quantization_demo()

    if results:
        print("\n🔬 演示總結:")
        print("- 理解了不同精度格式的特點")
        print("- 掌握了量化的數學原理")
        print("- 體驗了PTQ和QAT的區別")
        print("- 學會了量化方法的選擇策略")

        print("\n💡 實際應用提示:")
        print("- 在生產環境使用前務必充分測試")
        print("- 根據具體硬體選擇合適的量化方案")
        print("- 建立量化模型的監控和回滾機制")

if __name__ == "__main__":
    main()