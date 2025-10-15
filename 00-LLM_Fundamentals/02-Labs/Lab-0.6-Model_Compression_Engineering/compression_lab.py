#!/usr/bin/env python3
"""
Lab 0.6: 模型壓縮工程化實踐主腳本
完整的端到端模型壓縮實驗
"""

import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
from datetime import datetime
import json

class ModelCompressionEngineeringLab:
    """模型壓縮工程化實驗室"""

    def __init__(self, base_model="microsoft/DialoGPT-small"):
        self.base_model = base_model
        self.experiment_dir = Path(f"./results/compression_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.compression_results = {}

    def run_engineering_compression_pipeline(self):
        """運行工程化壓縮管線"""

        print("=== 模型壓縮工程化管線 ===")

        # 階段1: 模型分析與基線
        print("階段1: 模型分析與基線建立...")
        baseline_result = self._establish_baseline()

        # 階段2: 壓縮策略選擇
        print("\n階段2: 壓縮策略選擇...")
        strategy = self._select_compression_strategy(baseline_result)

        # 階段3: 壓縮實施
        print("\n階段3: 壓縮實施...")
        compression_result = self._implement_compression(strategy)

        # 階段4: 效果驗證與優化
        print("\n階段4: 效果驗證...")
        validation_result = self._validate_compression_effects(compression_result)

        # 整合結果
        pipeline_results = {
            'baseline': baseline_result,
            'strategy': strategy,
            'compression': compression_result,
            'validation': validation_result
        }

        # 保存結果並生成報告
        self._save_pipeline_results(pipeline_results)
        report = self._generate_engineering_report(pipeline_results)

        print(f"\n✅ 工程化壓縮管線完成！")
        print(f"📁 結果保存在: {self.experiment_dir}")

        return pipeline_results

    def _establish_baseline(self):
        """建立基線"""

        print("  載入原始模型並建立基線...")

        try:
            # 載入模型
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # 計算基線指標
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            param_count = sum(p.numel() for p in model.parameters())

            # 性能基線測試
            performance_baseline = self._benchmark_model_performance(model, tokenizer)

            baseline = {
                'model_name': self.base_model,
                'parameter_count': param_count,
                'model_size_mb': model_size_mb,
                'performance_baseline': performance_baseline,
                'baseline_established': True
            }

            print(f"    參數量: {param_count:,} ({param_count/1e9:.2f}B)")
            print(f"    模型大小: {model_size_mb:.1f} MB")

            return baseline

        except Exception as e:
            print(f"    基線建立失敗: {e}")
            return {'baseline_established': False, 'error': str(e)}

    def _select_compression_strategy(self, baseline):
        """選擇壓縮策略"""

        if not baseline.get('baseline_established'):
            return {'strategy_selected': False}

        model_size_mb = baseline['model_size_mb']
        target_memory_gb = 8  # 目標：適配8GB GPU

        # 基於模型大小選擇策略
        if model_size_mb > target_memory_gb * 1024 * 0.5:  # 模型佔用超過50%目標記憶體
            strategy = {
                'method': 'aggressive_quantization',
                'target_precision': 'int4',
                'additional_techniques': ['model_sharding'],
                'expected_compression_ratio': 8.0
            }
        elif model_size_mb > target_memory_gb * 1024 * 0.2:
            strategy = {
                'method': 'moderate_quantization',
                'target_precision': 'int8',
                'additional_techniques': ['inference_optimization'],
                'expected_compression_ratio': 2.0
            }
        else:
            strategy = {
                'method': 'conservative_optimization',
                'target_precision': 'fp16',
                'additional_techniques': ['inference_engine_optimization'],
                'expected_compression_ratio': 1.2
            }

        print(f"    選擇策略: {strategy['method']}")
        print(f"    目標精度: {strategy['target_precision']}")
        print(f"    預期壓縮比: {strategy['expected_compression_ratio']:.1f}x")

        return strategy

    def _implement_compression(self, strategy):
        """實施壓縮"""

        if not strategy.get('method'):
            return {'compression_successful': False}

        print(f"  實施{strategy['method']}...")

        try:
            # 根據策略執行壓縮
            if strategy['target_precision'] == 'int4':
                compressed_model = self._apply_int4_quantization()
            elif strategy['target_precision'] == 'int8':
                compressed_model = self._apply_int8_quantization()
            else:
                compressed_model = self._apply_fp16_optimization()

            # 測試壓縮後效果
            compression_test = self._test_compressed_model(compressed_model, strategy)

            return {
                'compression_successful': True,
                'strategy_applied': strategy,
                'compressed_model_info': compression_test,
                'implementation_time': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"    壓縮實施失敗: {e}")
            return {'compression_successful': False, 'error': str(e)}

    def _apply_int4_quantization(self):
        """應用INT4量化"""

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto"
        )

        return model

    def _apply_int8_quantization(self):
        """應用INT8量化"""

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto"
        )

        return model

    def _apply_fp16_optimization(self):
        """應用FP16優化"""

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        return model

    def _test_compressed_model(self, model, strategy):
        """測試壓縮後模型"""

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 計算壓縮後大小
        compressed_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

        # 功能測試
        test_prompt = "人工智能技術"
        try:
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 15,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            functional_test_passed = len(generated_text) > len(test_prompt)

        except Exception as e:
            functional_test_passed = False

        return {
            'compressed_size_mb': compressed_size_mb,
            'functional_test_passed': functional_test_passed,
            'compression_method': strategy['method'],
            'target_precision': strategy['target_precision']
        }

    def _benchmark_model_performance(self, model, tokenizer):
        """基準測試模型性能"""

        test_prompts = ["AI技術", "機器學習", "深度學習"]
        inference_times = []

        for prompt in test_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                start_time = time.time()

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

            except Exception:
                inference_times.append(float('inf'))

        avg_inference_time = np.mean([t for t in inference_times if t != float('inf')])

        return {
            'avg_inference_time': avg_inference_time,
            'inference_success_rate': sum(1 for t in inference_times if t != float('inf')) / len(inference_times)
        }

    def _validate_compression_effects(self, compression_result):
        """驗證壓縮效果"""

        if not compression_result.get('compression_successful'):
            return {'validation_passed': False}

        compressed_info = compression_result['compressed_model_info']

        # 驗證標準
        validation_checks = {
            'functional_test': compressed_info['functional_test_passed'],
            'size_reduction': compressed_info['compressed_size_mb'] < 1000,  # 簡化檢查
            'method_applied': compression_result['strategy_applied']['method'] != 'none'
        }

        all_passed = all(validation_checks.values())

        return {
            'validation_passed': all_passed,
            'validation_checks': validation_checks,
            'compression_summary': {
                'final_size_mb': compressed_info['compressed_size_mb'],
                'compression_method': compressed_info['compression_method'],
                'functional_integrity': compressed_info['functional_test_passed']
            }
        }

    def _save_pipeline_results(self, results: dict):
        """保存管線結果"""

        with open(self.experiment_dir / 'pipeline_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def _generate_engineering_report(self, results: dict) -> str:
        """生成工程化報告"""

        report = f"""# 模型壓縮工程化實驗報告

實驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
基礎模型: {self.base_model}

## 工程化流程執行結果

### 階段1: 基線建立
{self._format_baseline_results(results.get('baseline', {}))}

### 階段2: 策略選擇
{self._format_strategy_results(results.get('strategy', {}))}

### 階段3: 壓縮實施
{self._format_compression_results(results.get('compression', {}))}

### 階段4: 效果驗證
{self._format_validation_results(results.get('validation', {}))}

## 關鍵學習要點

1. **系統性思維**: 工程化壓縮需要系統性的分析和實施流程
2. **效果權衡**: 壓縮比、性能和準確性之間需要權衡
3. **充分測試**: 每個階段都需要充分的測試和驗證
4. **風險控制**: 建立回滾機制和監控體系

## 生產部署建議

- 在生產環境部署前進行更大規模的測試
- 建立A/B測試機制對比壓縮前後效果
- 設置監控告警，及時發現問題
- 準備快速回滾方案

---
*本報告基於演示實驗生成，生產環境請進行更完整的測試。*
"""

        report_path = self.experiment_dir / 'engineering_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📋 工程化報告已保存: {report_path}")

        return report

    def _format_baseline_results(self, baseline: dict) -> str:
        if baseline.get('baseline_established'):
            return f"""
✅ 基線建立成功
- 參數量: {baseline['parameter_count']:,}
- 模型大小: {baseline['model_size_mb']:.1f} MB
- 平均推理時間: {baseline['performance_baseline']['avg_inference_time']:.3f} 秒
"""
        else:
            return f"❌ 基線建立失敗: {baseline.get('error', '未知錯誤')}"

    def _format_strategy_results(self, strategy: dict) -> str:
        if strategy.get('method'):
            return f"""
✅ 策略選擇完成
- 壓縮方法: {strategy['method']}
- 目標精度: {strategy['target_precision']}
- 預期壓縮比: {strategy['expected_compression_ratio']:.1f}x
"""
        else:
            return "❌ 策略選擇失敗"

    def _format_compression_results(self, compression: dict) -> str:
        if compression.get('compression_successful'):
            info = compression['compressed_model_info']
            return f"""
✅ 壓縮實施成功
- 壓縮後大小: {info['compressed_size_mb']:.1f} MB
- 功能測試: {'通過' if info['functional_test_passed'] else '失敗'}
- 壓縮方法: {info['compression_method']}
"""
        else:
            return f"❌ 壓縮實施失敗: {compression.get('error', '未知錯誤')}"

    def _format_validation_results(self, validation: dict) -> str:
        if validation.get('validation_passed'):
            summary = validation['compression_summary']
            return f"""
✅ 驗證通過
- 最終大小: {summary['final_size_mb']:.1f} MB
- 功能完整性: {'保持' if summary['functional_integrity'] else '受損'}
- 壓縮方法: {summary['compression_method']}
"""
        else:
            return "❌ 驗證未通過"

def main():
    """主實驗函數"""

    print("Lab 0.6: 模型壓縮工程化實踐")
    print("=" * 60)

    # 創建實驗室實例
    lab = ModelCompressionEngineeringLab()

    # 運行完整工程化流程
    results = lab.run_engineering_compression_pipeline()

    if results:
        print("\n🎓 實驗學習總結:")
        print("✅ 體驗了完整的工程化壓縮流程")
        print("✅ 理解了各階段的技術要點和注意事項")
        print("✅ 掌握了壓縮效果的評估和驗證方法")
        print("✅ 建立了工程化實施的系統性思維")

        print("\n💡 實際應用建議:")
        print("- 在真實項目中使用更大的模型和數據集")
        print("- 建立自動化的壓縮和測試流程")
        print("- 與業務團隊緊密配合，確保壓縮效果符合需求")
        print("- 持續跟蹤最新的模型壓縮技術發展")

if __name__ == "__main__":
    main()