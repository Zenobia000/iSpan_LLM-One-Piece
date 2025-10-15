#!/usr/bin/env python3
"""
Lab 0.2: LLM評估基準實踐主腳本
使用evaluation_toolkit進行完整的模型評估實驗
"""

from evaluation_toolkit import LLMEvaluationToolkit
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

class EvaluationBenchmarkLab:
    """評估基準實驗室"""

    def __init__(self):
        self.toolkit = LLMEvaluationToolkit()
        self.experiment_dir = Path(f"./results/eval_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def run_multi_model_comparison(self):
        """運行多模型對比評估"""

        print("=== 多模型評估對比實驗 ===")

        # 測試模型列表（使用小模型進行演示）
        test_models = [
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium",
            # 可以添加更多模型
        ]

        comparison_results = {}

        for model_name in test_models:
            print(f"\n📊 評估模型: {model_name}")

            try:
                # 運行完整評估
                result = self.toolkit.run_comprehensive_evaluation(model_name)
                comparison_results[model_name] = result

                # 顯示關鍵指標
                if 'overall_evaluation' in result:
                    overall = result['overall_evaluation']
                    print(f"   綜合評分: {overall['weighted_overall_score']:.3f}")
                    print(f"   性能等級: {overall['grade']}")

            except Exception as e:
                print(f"   ❌ 評估失敗: {e}")
                comparison_results[model_name] = {'error': str(e)}

        # 生成對比報告
        self._generate_comparison_report(comparison_results)

        return comparison_results

    def _generate_comparison_report(self, results: dict):
        """生成對比報告"""

        # 創建對比表格
        comparison_data = []

        for model_name, result in results.items():
            if 'error' in result:
                continue

            row = {'模型': model_name.split('/')[-1]}

            # 提取關鍵指標
            if 'perplexity' in result:
                row['困惑度'] = f"{result['perplexity']['perplexity']:.2f}"

            if 'language_understanding' in result:
                row['語言理解F1'] = f"{result['language_understanding']['f1_score']:.3f}"

            if 'generation_quality' in result:
                row['生成質量'] = f"{result['generation_quality']['average_quality']['overall_score']:.3f}"

            if 'inference_performance' in result:
                perf = result['inference_performance']['overall_performance']
                row['推理速度'] = f"{perf.get('avg_tokens_per_second', 0):.1f} TPS"

            if 'overall_evaluation' in result:
                row['綜合評分'] = f"{result['overall_evaluation']['weighted_overall_score']:.3f}"

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # 保存結果
        df.to_csv(self.experiment_dir / 'model_comparison.csv', index=False)

        print("\n📊 模型對比結果:")
        print(df.to_string(index=False))

def main():
    """主實驗函數"""

    print("Lab 0.2: LLM評估基準實踐")
    print("=" * 50)

    lab = EvaluationBenchmarkLab()
    results = lab.run_multi_model_comparison()

    print(f"\n✅ 實驗完成！結果保存在: {lab.experiment_dir}")

if __name__ == "__main__":
    main()