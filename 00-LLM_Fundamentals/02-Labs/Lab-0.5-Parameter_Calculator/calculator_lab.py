#!/usr/bin/env python3
"""
Lab 0.5: 參數計算器實踐主腳本
使用parameter_calculator進行完整的資源估算實驗
"""

from parameter_calculator import ParameterCalculator, ModelConfig, create_example_configs
import pandas as pd
from pathlib import Path
from datetime import datetime

class ParameterCalculatorLab:
    """參數計算器實驗室"""

    def __init__(self):
        self.calculator = ParameterCalculator()
        self.experiment_dir = Path(f"./results/parameter_calculator_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def run_complete_calculation_experiment(self):
        """運行完整計算實驗"""

        print("=== 參數計算器完整實驗 ===")

        # 1. 創建測試配置
        configs = create_example_configs()

        # 2. 參數量計算實驗
        print("1. 參數量計算實驗...")
        param_results = self._run_parameter_calculation(configs)

        # 3. 記憶體需求分析實驗
        print("\n2. 記憶體需求分析...")
        memory_results = self._run_memory_analysis(configs)

        # 4. 硬體匹配實驗
        print("\n3. 硬體匹配分析...")
        hardware_results = self._run_hardware_matching(configs)

        # 5. 縮放法則實驗
        print("\n4. 縮放法則應用...")
        scaling_results = self.calculator.demonstrate_scaling_laws()

        # 6. 可視化和報告
        self.calculator.visualize_parameter_analysis(configs)

        # 保存結果
        experiment_results = {
            'parameter_results': param_results,
            'memory_results': memory_results,
            'hardware_results': hardware_results,
            'scaling_results': scaling_results
        }

        self._save_experiment_results(experiment_results)

        return experiment_results

    def _run_parameter_calculation(self, configs):
        """參數量計算實驗"""

        results = []

        for config in configs:
            param_result = self.calculator.calculate_transformer_parameters(config)
            results.append({
                'model_name': config.name,
                'config': config,
                'parameter_analysis': param_result
            })

        return results

    def _run_memory_analysis(self, configs):
        """記憶體需求分析實驗"""

        results = []

        for config in configs[:2]:  # 分析前兩個配置
            # 推理記憶體
            inference_memory = self.calculator.calculate_memory_requirements(
                config, batch_size=8, sequence_length=2048, precision='fp16', mode='inference'
            )

            # 訓練記憶體
            training_memory = self.calculator.calculate_memory_requirements(
                config, batch_size=4, sequence_length=2048, precision='fp16', mode='training'
            )

            results.append({
                'model_name': config.name,
                'inference_memory': inference_memory,
                'training_memory': training_memory
            })

        return results

    def _run_hardware_matching(self, configs):
        """硬體匹配實驗"""

        # 定義目標GPU
        target_gpus = {
            'RTX_4090': {'memory_gb': 24, 'price_usd': 1600},
            'A100_40GB': {'memory_gb': 40, 'price_usd': 15000},
            'A100_80GB': {'memory_gb': 80, 'price_usd': 20000}
        }

        workload = {
            'batch_size': 8,
            'sequence_length': 2048,
            'precision': 'fp16',
            'mode': 'inference'
        }

        results = []

        for config in configs[:2]:  # 分析前兩個配置
            matching_result = self.calculator.hardware_matching_calculator(
                config, target_gpus, workload
            )
            results.append({
                'model_name': config.name,
                'matching_result': matching_result
            })

        return results

    def _save_experiment_results(self, results: dict):
        """保存實驗結果"""

        import json

        # 保存JSON結果
        with open(self.experiment_dir / 'calculation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # 生成CSV摘要
        self._generate_csv_summary(results)

        print(f"📁 實驗結果已保存: {self.experiment_dir}")

    def _generate_csv_summary(self, results: dict):
        """生成CSV摘要"""

        # 參數量摘要
        param_data = []
        for result in results['parameter_results']:
            config = result['config']
            analysis = result['parameter_analysis']

            param_data.append({
                '模型': config.name,
                '參數量(B)': f"{analysis['total_parameters'] / 1e9:.2f}",
                '注意力佔比': f"{analysis['parameter_percentages']['attention_ratio']:.1f}%",
                'FFN佔比': f"{analysis['parameter_percentages']['ffn_ratio']:.1f}%"
            })

        pd.DataFrame(param_data).to_csv(
            self.experiment_dir / 'parameter_summary.csv', index=False
        )

        print("📊 CSV摘要已生成")

def main():
    """主實驗函數"""

    print("Lab 0.5: 參數計算器實踐")
    print("=" * 50)

    lab = ParameterCalculatorLab()
    results = lab.run_complete_calculation_experiment()

    print("\n✅ 計算實驗完成！")
    print("\n🎓 學習收穫:")
    print("- 掌握了Transformer精確參數計算方法")
    print("- 理解了記憶體需求的組成和估算")
    print("- 學會了硬體匹配和成本分析")
    print("- 了解了縮放法則的實際應用")

if __name__ == "__main__":
    main()