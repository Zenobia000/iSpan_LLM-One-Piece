#!/usr/bin/env python3
"""
Lab 0.4: 量化技術對比實踐主腳本
使用quantization_demo進行完整的量化技術對比實驗
"""

from quantization_demo import QuantizationDemo
import pandas as pd
from pathlib import Path
from datetime import datetime

class QuantizationComparisonLab:
    """量化對比實驗室"""

    def __init__(self):
        self.demo = QuantizationDemo()
        self.experiment_dir = Path(f"./results/quantization_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def run_quantization_comparison_experiment(self):
        """運行量化對比實驗"""

        print("=== 量化技術對比實驗 ===")

        # 運行完整量化演示
        results = self.demo.run_complete_quantization_demo()

        if results:
            # 保存實驗結果
            self._save_experiment_results(results)

            # 生成實驗總結
            self._generate_experiment_summary(results)

        return results

    def _save_experiment_results(self, results: dict):
        """保存實驗結果"""

        import json

        with open(self.experiment_dir / 'quantization_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def _generate_experiment_summary(self, results: dict):
        """生成實驗總結"""

        summary = f"""# 量化技術對比實驗總結

實驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 實驗結果

### 精度格式對比
{self._summarize_precision_demo(results.get('precision_demo', {}))}

### 量化方法對比
{self._summarize_methods_comparison(results.get('methods_comparison', {}))}

### 關鍵發現
1. 量化是精度和效率之間的重要權衡
2. PTQ適合快速部署，QAT適合高質量要求
3. 不同模型對量化的敏感性不同
4. 硬體支持是量化效果的重要因素

### 實際應用建議
- 生產部署前必須進行充分測試
- 根據硬體特性選擇量化策略
- 建立量化模型的監控機制
"""

        with open(self.experiment_dir / 'experiment_summary.md', 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"📊 實驗總結已保存: {self.experiment_dir / 'experiment_summary.md'}")

    def _summarize_precision_demo(self, precision_demo: dict) -> str:
        if not precision_demo:
            return "未執行精度演示"

        return "- FP32: 最高精度，最大存儲\n- FP16: 精度略降，存儲減半\n- INT8: 明顯壓縮，需要校準\n- INT4: 極致壓縮，需要特殊技術"

    def _summarize_methods_comparison(self, methods_comp: dict) -> str:
        if not methods_comp:
            return "未執行方法對比"

        return "- PTQ: 簡單快速，準確性中等\n- QAT: 複雜耗時，準確性最佳\n- GPTQ: 適合大模型\n- AWQ: 平衡性優秀"

def main():
    """主實驗函數"""

    print("Lab 0.4: 量化技術對比實踐")
    print("=" * 50)

    lab = QuantizationComparisonLab()
    results = lab.run_quantization_comparison_experiment()

    if results:
        print("\n✅ 量化對比實驗完成！")
        print("\n🎓 學習收穫:")
        print("- 掌握了量化技術的理論基礎")
        print("- 體驗了不同量化方法的實際效果")
        print("- 建立了量化策略的選擇思維")
        print("- 了解了量化在工程實踐中的考量")

if __name__ == "__main__":
    main()