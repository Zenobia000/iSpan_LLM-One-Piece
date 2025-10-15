#!/usr/bin/env python3
"""
Lab 0.3: 數據集分析實踐主腳本
使用dataset_analyzer進行完整的數據集分析實驗
"""

from dataset_analyzer import DatasetAnalyzer
import pandas as pd
from pathlib import Path
from datetime import datetime

class DatasetAnalysisLab:
    """數據集分析實驗室"""

    def __init__(self):
        self.analyzer = DatasetAnalyzer()
        self.experiment_dir = Path(f"./results/dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def run_comprehensive_dataset_analysis(self):
        """運行綜合數據集分析"""

        print("=== 綜合數據集分析實驗 ===")

        # 分析預訓練數據集
        print("1. 分析預訓練數據集...")
        pretraining_result = self.analyzer.analyze_pretraining_dataset(max_samples=500)

        # 分析指令數據集
        print("\n2. 分析指令數據集...")
        instruction_result = self.analyzer.analyze_instruction_dataset(max_samples=300)

        # 分析偏好數據集
        print("\n3. 分析偏好數據集...")
        preference_result = self.analyzer.analyze_preference_dataset(max_samples=100)

        # 數據集對比
        print("\n4. 數據集對比分析...")
        comparison = self.analyzer.compare_datasets([
            pretraining_result,
            instruction_result,
            preference_result
        ])

        # 生成可視化
        print("\n5. 生成可視化圖表...")
        self.analyzer.visualize_analysis_results('all')

        # 生成報告
        report = self.analyzer.generate_analysis_report()

        # 保存結果
        self._save_results({
            'pretraining': pretraining_result,
            'instruction': instruction_result,
            'preference': preference_result,
            'comparison': comparison
        }, report)

        return {
            'pretraining_result': pretraining_result,
            'instruction_result': instruction_result,
            'preference_result': preference_result,
            'comparison_result': comparison
        }

    def _save_results(self, results: dict, report: str):
        """保存實驗結果"""

        # 保存JSON結果
        with open(self.experiment_dir / 'analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # 保存報告
        with open(self.experiment_dir / 'analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📁 結果已保存到: {self.experiment_dir}")

def main():
    """主實驗函數"""

    print("Lab 0.3: 數據集分析實踐")
    print("=" * 50)

    lab = DatasetAnalysisLab()
    results = lab.run_comprehensive_dataset_analysis()

    print(f"\n✅ 實驗完成！")
    print("\n🎓 關鍵學習要點:")
    print("1. 數據質量比數據量更重要")
    print("2. 不同訓練階段需要不同特性的數據")
    print("3. 數據平衡性影響模型泛化能力")
    print("4. 建立數據質量控制體系是關鍵")

if __name__ == "__main__":
    main()