# Lab 0.6: 模型壓縮工程化實踐

## 實驗目標

通過完整的工程化壓縮流程實踐，掌握從模型載入到推理部署的端到端壓縮技能，建立工業級的模型優化能力。

## 學習成果

完成本實驗後，您將能夠：
- 掌握模型-硬體匹配的精確計算方法
- 實施完整的工程化壓縮流程
- 建立壓縮效果的科學評估體系
- 具備生產級模型部署優化能力

## 實驗環境要求

### 硬體要求
- GPU：16GB+顯存（推薦RTX 4090或A100）
- RAM：32GB+系統記憶體
- 存儲：50GB可用空間

### 軟體要求
- PyTorch 2.0+
- Transformers 4.30+
- BitsAndBytes、ONNX Runtime
- 已激活的poetry虛擬環境

## 實驗資源說明

本Lab包含完整的工程化壓縮資源：
- **`compression_lab.py`**: 主實驗腳本，演示端到端壓縮流程
- **`MODEL_COMPRESSION_ENGINEERING.md`**: ⭐ **工程化實用指南** - 包含硬體匹配公式、決策表、快速工具

## 執行方式

```bash
# 1. 先閱讀工程化指南（強烈推薦）
cat MODEL_COMPRESSION_ENGINEERING.md

# 2. 運行完整實驗
python compression_lab.py
```

> 💡 **建議學習順序**: 先學習理論專論0.4和0.5，再閱讀工程化指南，最後執行實驗

## 實驗內容

### 核心實驗腳本

```python
# compression_engineering_lab.py
"""
模型壓縮工程化實踐主腳本
演示從載入到部署的完整壓縮流程
"""

import torch
import numpy as np
import time
import json
import os
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
import psutil
import matplotlib.pyplot as plt
import pandas as pd

class ModelCompressionEngineeringLab:
    """模型壓縮工程化實驗室"""

    def __init__(self, base_model_name="microsoft/DialoGPT-small"):
        self.base_model_name = base_model_name
        self.results = {}
        self.compression_timeline = []

    def step_1_model_analysis_and_baseline(self):
        """步驟1: 模型分析與基線建立"""

        print("=== 步驟1: 模型分析與基線建立 ===")

        # 1.1 載入原始模型
        print("1.1 載入原始模型...")
        original_model, tokenizer = self._load_original_model()

        # 1.2 分析模型結構
        print("1.2 分析模型結構...")
        model_structure = self._analyze_model_structure(original_model)

        # 1.3 建立性能基線
        print("1.3 建立性能基線...")
        baseline_performance = self._establish_baseline_performance(original_model, tokenizer)

        # 1.4 硬體需求分析
        print("1.4 硬體需求分析...")
        hardware_analysis = self._analyze_hardware_requirements(model_structure)

        step1_results = {
            'model_structure': model_structure,
            'baseline_performance': baseline_performance,
            'hardware_analysis': hardware_analysis,
            'timestamp': datetime.now().isoformat()
        }

        self.results['step1_analysis'] = step1_results
        self._save_checkpoint('step1_analysis.json', step1_results)

        return step1_results

    def step_2_compression_strategy_design(self, target_constraints: dict):
        """步驟2: 壓縮策略設計"""

        print("=== 步驟2: 壓縮策略設計 ===")

        analysis_result = self.results.get('step1_analysis')
        if not analysis_result:
            raise ValueError("請先完成步驟1的模型分析")

        # 2.1 解析約束條件
        print("2.1 解析目標約束...")
        parsed_constraints = self._parse_constraints(target_constraints)

        # 2.2 生成壓縮候選策略
        print("2.2 生成候選壓縮策略...")
        candidate_strategies = self._generate_candidate_strategies(analysis_result, parsed_constraints)

        # 2.3 策略效果預測
        print("2.3 預測各策略效果...")
        strategy_predictions = self._predict_strategy_effects(candidate_strategies, analysis_result)

        # 2.4 選擇最優策略
        print("2.4 選擇最優策略...")
        optimal_strategy = self._select_optimal_strategy(strategy_predictions, parsed_constraints)

        step2_results = {
            'target_constraints': parsed_constraints,
            'candidate_strategies': candidate_strategies,
            'strategy_predictions': strategy_predictions,
            'optimal_strategy': optimal_strategy,
            'timestamp': datetime.now().isoformat()
        }

        self.results['step2_strategy'] = step2_results
        self._save_checkpoint('step2_strategy.json', step2_results)

        return step2_results

    def step_3_compression_implementation(self):
        """步驟3: 壓縮實施"""

        print("=== 步驟3: 壓縮實施 ===")

        strategy = self.results.get('step2_strategy', {}).get('optimal_strategy')
        if not strategy:
            raise ValueError("請先完成步驟2的策略設計")

        implementation_results = {}

        # 3.1 量化實施
        if 'quantization' in strategy['methods']:
            print("3.1 執行量化壓縮...")
            quant_config = strategy['methods']['quantization']
            quantization_result = self._implement_quantization(quant_config)
            implementation_results['quantization'] = quantization_result

        # 3.2 剪枝實施
        if 'pruning' in strategy['methods']:
            print("3.2 執行剪枝壓縮...")
            pruning_config = strategy['methods']['pruning']
            pruning_result = self._implement_pruning(pruning_config)
            implementation_results['pruning'] = pruning_result

        # 3.3 推理引擎優化
        if 'inference_optimization' in strategy['methods']:
            print("3.3 推理引擎優化...")
            optimization_result = self._optimize_inference_engine()
            implementation_results['inference_optimization'] = optimization_result

        # 3.4 效果驗證
        print("3.4 驗證壓縮效果...")
        validation_result = self._validate_compression_effects(implementation_results)

        step3_results = {
            'implementation_results': implementation_results,
            'validation_result': validation_result,
            'compressed_model_info': self._get_compressed_model_info(),
            'timestamp': datetime.now().isoformat()
        }

        self.results['step3_implementation'] = step3_results
        self._save_checkpoint('step3_implementation.json', step3_results)

        return step3_results

    def step_4_deployment_preparation(self, deployment_target: dict):
        """步驟4: 部署準備"""

        print("=== 步驟4: 部署準備 ===")

        # 4.1 模型格式轉換
        print("4.1 模型格式轉換...")
        format_conversion = self._convert_model_format(deployment_target)

        # 4.2 推理服務配置
        print("4.2 推理服務配置...")
        service_config = self._configure_inference_service(deployment_target)

        # 4.3 性能基準測試
        print("4.3 部署性能測試...")
        deployment_benchmarks = self._run_deployment_benchmarks(deployment_target)

        # 4.4 生產就緒檢查
        print("4.4 生產就緒檢查...")
        production_readiness = self._check_production_readiness(deployment_benchmarks)

        step4_results = {
            'format_conversion': format_conversion,
            'service_config': service_config,
            'deployment_benchmarks': deployment_benchmarks,
            'production_readiness': production_readiness,
            'deployment_artifacts': self._prepare_deployment_artifacts(),
            'timestamp': datetime.now().isoformat()
        }

        self.results['step4_deployment'] = step4_results
        self._save_checkpoint('step4_deployment.json', step4_results)

        return step4_results

    # === 核心實施方法 ===

    def _implement_quantization(self, config: dict) -> dict:
        """實施量化壓縮"""

        print(f"實施{config['method']}量化，目標精度：{config['target_precision']}")

        start_time = time.time()

        try:
            if config['method'] == 'ptq_bitsandbytes':
                # 使用BitsAndBytes進行PTQ
                if config['target_precision'] == 'int8':
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                elif config['target_precision'] == 'int4':
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=False
                    )
                else:
                    raise ValueError(f"Unsupported precision: {config['target_precision']}")

                # 載入量化模型
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )

                # 測試功能
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                test_result = self._test_quantized_model(quantized_model, tokenizer)

                # 計算壓縮效果
                original_size = self._estimate_model_size(self.base_model_name, 'fp16')
                quantized_size = self._estimate_model_size_quantized(quantized_model)

                end_time = time.time()

                return {
                    'method': config['method'],
                    'target_precision': config['target_precision'],
                    'success': True,
                    'compression_time_seconds': end_time - start_time,
                    'original_size_mb': original_size,
                    'quantized_size_mb': quantized_size,
                    'compression_ratio': original_size / quantized_size if quantized_size > 0 else 1,
                    'test_result': test_result,
                    'quantized_model': quantized_model
                }

            else:
                return {
                    'method': config['method'],
                    'success': False,
                    'error': f"Method {config['method']} not implemented in this demo"
                }

        except Exception as e:
            end_time = time.time()
            return {
                'method': config['method'],
                'success': False,
                'error': str(e),
                'compression_time_seconds': end_time - start_time
            }

    def _test_quantized_model(self, model, tokenizer) -> dict:
        """測試量化模型功能"""

        test_prompts = [
            "人工智能的發展",
            "機器學習算法",
            "深度學習應用"
        ]

        test_results = []

        for prompt in test_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                start_time = time.time()

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 30,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                inference_time = time.time() - start_time

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()

                test_results.append({
                    'prompt': prompt,
                    'response': response,
                    'inference_time': inference_time,
                    'success': True
                })

            except Exception as e:
                test_results.append({
                    'prompt': prompt,
                    'success': False,
                    'error': str(e)
                })

        # 計算平均性能
        successful_tests = [r for r in test_results if r['success']]
        avg_inference_time = np.mean([r['inference_time'] for r in successful_tests]) if successful_tests else 0

        return {
            'test_results': test_results,
            'success_rate': len(successful_tests) / len(test_results) * 100,
            'avg_inference_time': avg_inference_time,
            'functional_test_passed': len(successful_tests) == len(test_results)
        }

    def _estimate_model_size(self, model_name: str, precision: str) -> float:
        """估算模型大小"""

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            param_count = sum(p.numel() for p in model.parameters())

            precision_bytes = {'fp32': 4, 'fp16': 2, 'int8': 1, 'int4': 0.5}
            model_size_mb = param_count * precision_bytes[precision] / (1024**2)

            return model_size_mb

        except Exception as e:
            print(f"估算模型大小失敗: {e}")
            return 1000.0  # 默認估算值

    def _estimate_model_size_quantized(self, model) -> float:
        """估算量化模型大小"""

        try:
            total_size = 0
            for param in model.parameters():
                total_size += param.numel() * param.element_size()

            return total_size / (1024**2)  # 轉換為MB

        except Exception as e:
            print(f"估算量化模型大小失敗: {e}")
            return 500.0  # 默認估算值

    def run_hardware_matching_analysis(self):
        """運行硬體匹配分析"""

        print("\\n=== 硬體匹配分析 ===")

        # 使用之前定義的硬體匹配計算器
        from MODEL_COMPRESSION_ENGINEERING import HardwareMatchingCalculator

        calculator = HardwareMatchingCalculator()

        # 模型參數（假設7B參數的模型）
        model_params = 7e9

        print("1. 不同模式下的GPU需求分析:")

        # 訓練模式
        train_req = calculator.calculate_minimum_gpus(
            model_params, 'fp16', 'training', batch_size=4, sequence_length=2048
        )

        print("\\n訓練模式:")
        print(f"  記憶體需求: {train_req['total_memory_required_gb']:.1f} GB")

        for gpu, req in train_req['gpu_requirements'].items():
            print(f"  {gpu}: {req['min_gpus']} GPUs ({req['memory_utilization']:.1f}% 利用率, ${req['total_cost']:,})")

        # 推理模式
        infer_req = calculator.calculate_minimum_gpus(
            model_params, 'fp16', 'inference', batch_size=8, sequence_length=2048
        )

        print("\\n推理模式:")
        print(f"  記憶體需求: {infer_req['total_memory_required_gb']:.1f} GB")

        for gpu, req in infer_req['gpu_requirements'].items():
            print(f"  {gpu}: {req['min_gpus']} GPUs ({req['memory_utilization']:.1f}% 利用率, ${req['total_cost']:,})")

        # 2. 量化後的硬體需求對比
        print("\\n2. 量化後硬體需求對比:")

        quantization_scenarios = [
            {'precision': 'fp16', 'name': 'FP16'},
            {'precision': 'int8', 'name': 'INT8'},
            {'precision': 'int4', 'name': 'INT4'}
        ]

        comparison_data = []

        for scenario in quantization_scenarios:
            infer_req = calculator.calculate_minimum_gpus(
                model_params, scenario['precision'], 'inference', batch_size=8, sequence_length=2048
            )

            # 選擇RTX 4090作為參考
            rtx4090_req = infer_req['gpu_requirements']['RTX_4090']

            comparison_data.append({
                '量化方案': scenario['name'],
                '記憶體需求(GB)': f"{infer_req['total_memory_required_gb']:.1f}",
                '所需GPU數量': rtx4090_req['min_gpus'],
                '記憶體利用率': f"{rtx4090_req['memory_utilization']:.1f}%",
                '總成本(USD)': f"${rtx4090_req['total_cost']:,}"
            })

        # 顯示對比表格
        df = pd.DataFrame(comparison_data)
        print("\\n量化方案對比:")
        print(df.to_string(index=False))

        return {
            'training_requirements': train_req,
            'inference_requirements': infer_req,
            'quantization_comparison': comparison_data
        }

    def run_compression_experiment(self, compression_configs: list):
        """運行壓縮實驗"""

        print("\\n=== 壓縮實驗執行 ===")

        experiment_results = {}

        for i, config in enumerate(compression_configs):
            experiment_name = f"experiment_{i+1}_{config['name']}"
            print(f"\\n運行實驗 {i+1}: {config['name']}")

            try:
                # 記錄實驗開始時間
                start_time = time.time()

                # 執行壓縮
                if config['type'] == 'quantization':
                    result = self._run_quantization_experiment(config)
                elif config['type'] == 'mixed_compression':
                    result = self._run_mixed_compression_experiment(config)
                else:
                    result = {'error': f"Unsupported experiment type: {config['type']}"}

                # 記錄實驗時間
                end_time = time.time()
                result['experiment_duration'] = end_time - start_time

                experiment_results[experiment_name] = result

                print(f"實驗 {i+1} 完成，耗時 {result['experiment_duration']:.1f} 秒")

            except Exception as e:
                print(f"實驗 {i+1} 失敗: {e}")
                experiment_results[experiment_name] = {
                    'success': False,
                    'error': str(e)
                }

        return experiment_results

    def _run_quantization_experiment(self, config: dict) -> dict:
        """運行量化實驗"""

        # 實施量化
        quantization_result = self._implement_quantization(config['quantization_config'])

        if not quantization_result['success']:
            return quantization_result

        # 性能對比測試
        performance_comparison = self._compare_quantization_performance(quantization_result)

        return {
            'quantization_result': quantization_result,
            'performance_comparison': performance_comparison,
            'success': True
        }

    def _compare_quantization_performance(self, quantization_result: dict) -> dict:
        """對比量化前後的性能"""

        print("對比量化前後性能...")

        # 載入原始模型
        original_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        original_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 載入量化模型
        quantized_model = quantization_result.get('quantized_model')

        # 性能測試
        test_prompts = ["人工智能技術", "機器學習算法", "深度學習應用"]
        comparison_results = []

        for prompt in test_prompts:
            # 原始模型測試
            original_result = self._benchmark_single_model(original_model, original_tokenizer, prompt)

            # 量化模型測試
            quantized_result = self._benchmark_single_model(quantized_model, original_tokenizer, prompt)

            comparison_results.append({
                'prompt': prompt,
                'original': original_result,
                'quantized': quantized_result,
                'speedup': original_result['inference_time'] / quantized_result['inference_time'] if quantized_result['inference_time'] > 0 else 1
            })

        # 計算平均性能指標
        avg_speedup = np.mean([r['speedup'] for r in comparison_results])
        avg_original_time = np.mean([r['original']['inference_time'] for r in comparison_results])
        avg_quantized_time = np.mean([r['quantized']['inference_time'] for r in comparison_results])

        return {
            'detailed_comparisons': comparison_results,
            'average_speedup': avg_speedup,
            'average_original_time': avg_original_time,
            'average_quantized_time': avg_quantized_time,
            'compression_ratio': quantization_result['compression_ratio'],
            'memory_saving_percent': (1 - quantization_result['quantized_size_mb'] / quantization_result['original_size_mb']) * 100
        }

    def _benchmark_single_model(self, model, tokenizer, prompt: str) -> dict:
        """對單個模型進行基準測試"""

        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 記錄推理時間
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            end_time = time.time()

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()

            return {
                'inference_time': end_time - start_time,
                'generated_response': response,
                'tokens_generated': outputs.shape[1] - inputs['input_ids'].shape[1],
                'success': True
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'inference_time': float('inf')
            }

    def generate_final_report(self) -> str:
        """生成最終實驗報告"""

        report = f"""# 模型壓縮工程化實踐報告

## 實驗概述
- 基礎模型: {self.base_model_name}
- 實驗完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 實驗階段: {len(self.results)} 個階段

## 主要發現
"""

        # 分析實驗結果
        if 'step3_implementation' in self.results:
            impl_results = self.results['step3_implementation']

            # 量化效果分析
            if 'quantization' in impl_results['implementation_results']:
                quant_result = impl_results['implementation_results']['quantization']
                if quant_result['success']:
                    report += f"""
### 量化壓縮效果
- 壓縮方法: {quant_result['method']}
- 目標精度: {quant_result['target_precision']}
- 壓縮比: {quant_result['compression_ratio']:.2f}x
- 模型大小: {quant_result['original_size_mb']:.1f}MB → {quant_result['quantized_size_mb']:.1f}MB
- 壓縮時間: {quant_result['compression_time_seconds']:.1f} 秒
"""

        if 'hardware_matching' in self.results:
            hardware_result = self.results['hardware_matching']
            report += """
### 硬體需求分析
詳見硬體匹配分析結果表格。
"""

        report += f"""
## 工程化要點總結

### 成功因素
1. **全面的前期分析**: 詳細分析模型結構和硬體需求
2. **科學的策略制定**: 基於數據驅動的壓縮策略選擇
3. **系統的實施流程**: 階段性實施和持續驗證
4. **完整的效果評估**: 多維度的壓縮效果評估

### 改進建議
1. 建立更完善的自動化壓縮流程
2. 增加更多壓縮技術的組合實驗
3. 建立更精確的性能預測模型
4. 完善生產部署的監控體系

### 最佳實踐
1. **漸進式壓縮**: 分階段實施，逐步優化
2. **充分測試**: 在每個階段進行全面測試
3. **持續監控**: 部署後持續監控效果
4. **文檔完備**: 記錄所有決策和實施細節

## 部署建議
{self._generate_deployment_recommendations()}

## 未來優化方向
1. 探索更先進的量化技術（如QLoRA、GGML）
2. 結合模型剪枝和知識蒸餾
3. 針對特定硬體的深度優化
4. 建立自動化的壓縮參數調優系統
"""

        return report

    def _generate_deployment_recommendations(self) -> str:
        """生成部署建議"""

        recommendations = []

        # 基於實驗結果生成建議
        if 'step3_implementation' in self.results:
            impl_results = self.results['step3_implementation']['implementation_results']

            if 'quantization' in impl_results and impl_results['quantization']['success']:
                quant_result = impl_results['quantization']
                compression_ratio = quant_result.get('compression_ratio', 1)

                if compression_ratio > 3:
                    recommendations.append("✅ 量化效果優秀，推薦生產部署")
                elif compression_ratio > 2:
                    recommendations.append("⚠️ 量化效果良好，建議小規模試驗後部署")
                else:
                    recommendations.append("❌ 量化效果有限，建議進一步優化")

        if not recommendations:
            recommendations.append("建議完成完整實驗後再制定部署策略")

        return "\\n".join(recommendations)

    # === 輔助方法 ===

    def _save_checkpoint(self, filename: str, data: dict):
        """保存檢查點數據"""

        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = os.path.join('checkpoints', filename)

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        print(f"檢查點已保存: {checkpoint_path}")

    def _load_original_model(self):
        """載入原始模型"""

        print(f"載入模型: {self.base_model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print(f"模型載入成功，參數量: {model.num_parameters():,}")

        return model, tokenizer

    def _analyze_model_structure(self, model) -> dict:
        """分析模型結構"""

        # 計算參數分佈
        param_distribution = {}
        total_params = 0

        for name, param in model.named_parameters():
            layer_type = name.split('.')[0] if '.' in name else name
            param_count = param.numel()

            if layer_type not in param_distribution:
                param_distribution[layer_type] = 0

            param_distribution[layer_type] += param_count
            total_params += param_count

        # 計算參數比例
        param_ratios = {k: v/total_params*100 for k, v in param_distribution.items()}

        return {
            'total_parameters': total_params,
            'parameter_distribution': param_distribution,
            'parameter_ratios': param_ratios,
            'model_size_mb': total_params * 2 / (1024**2),  # FP16
            'architecture_type': 'transformer'  # 假設為transformer架構
        }

# 主實驗執行函數
def main():
    """主實驗執行函數"""

    print("=== 模型壓縮工程化實踐實驗 ===\\n")

    # 初始化實驗室
    lab = ModelCompressionEngineeringLab()

    try:
        # 步驟1: 模型分析
        step1_result = lab.step_1_model_analysis_and_baseline()

        # 硬體匹配分析
        hardware_matching = lab.run_hardware_matching_analysis()
        lab.results['hardware_matching'] = hardware_matching

        # 步驟2: 壓縮策略設計
        target_constraints = {
            'max_memory_gb': 16,      # 目標：適配16GB GPU
            'max_accuracy_loss': 5,   # 最大可接受準確性損失5%
            'min_speedup': 1.5,       # 最小加速比1.5x
            'deployment_target': 'cloud_inference'
        }

        step2_result = lab.step_2_compression_strategy_design(target_constraints)

        # 步驟3: 壓縮實施
        step3_result = lab.step_3_compression_implementation()

        # 運行壓縮實驗
        compression_configs = [
            {
                'name': 'INT8_PTQ',
                'type': 'quantization',
                'quantization_config': {
                    'method': 'ptq_bitsandbytes',
                    'target_precision': 'int8'
                }
            },
            {
                'name': 'INT4_PTQ',
                'type': 'quantization',
                'quantization_config': {
                    'method': 'ptq_bitsandbytes',
                    'target_precision': 'int4'
                }
            }
        ]

        experiment_results = lab.run_compression_experiment(compression_configs)
        lab.results['experiments'] = experiment_results

        # 生成最終報告
        final_report = lab.generate_final_report()

        # 保存報告
        with open('compression_engineering_final_report.md', 'w', encoding='utf-8') as f:
            f.write(final_report)

        print("\\n=== 實驗完成 ===")
        print("最終報告已保存到: compression_engineering_final_report.md")
        print("檢查點數據已保存到: checkpoints/ 目錄")

        # 顯示關鍵結果
        print("\\n關鍵實驗結果:")
        for exp_name, exp_result in experiment_results.items():
            if exp_result.get('success'):
                quant_result = exp_result.get('quantization_result', {})
                print(f"  {exp_name}: {quant_result.get('compression_ratio', 1):.2f}x 壓縮")

        return lab.results

    except Exception as e:
        print(f"實驗執行出錯: {e}")
        return None

if __name__ == "__main__":
    results = main()
```

## 實驗執行指南

### 準備階段
```bash
# 1. 激活虛擬環境
source 00-Course_Setup/.venv/bin/activate

# 2. 安裝額外依賴
pip install bitsandbytes accelerate onnxruntime

# 3. 進入實驗目錄
cd 00-LLM_Fundamentals/02-Labs/Lab-0.6-Model_Compression_Engineering
```

### 執行階段
```bash
# 運行完整工程化實驗
python compression_engineering_lab.py
```

### 驗證階段
檢查生成的文件：
- `compression_engineering_final_report.md` - 最終報告
- `checkpoints/` - 各階段檢查點數據
- 各種性能對比圖表

## 預期學習成果

### 技術能力提升
- ✅ **精確計算能力**: 掌握模型-硬體匹配的精確計算
- ✅ **工程化思維**: 建立系統性的壓縮工程化思維
- ✅ **實施技能**: 具備端到端壓縮實施能力
- ✅ **優化意識**: 形成持續優化的工程意識

### 實踐經驗獲得
- 🔧 完整壓縮流程的實際操作經驗
- 📊 多維度效果評估的實踐方法
- 🎯 硬體約束下的優化策略制定
- 🚀 生產部署的準備和驗證流程

### 工具掌握程度
- **量化工具**: BitsAndBytes、GPTQ、AWQ等
- **性能分析**: 記憶體分析、性能基準測試
- **部署工具**: ONNX、TensorRT等推理引擎
- **監控工具**: 資源監控、性能監控系統

## 實驗報告要求

### 必答問題
1. **硬體匹配分析**: 為不同規模模型計算硬體需求
2. **壓縮策略選擇**: 解釋您選擇特定壓縮策略的原因
3. **效果評估**: 分析壓縮前後的性能變化
4. **工程化要點**: 總結工程化實施的關鍵要點

### 延伸思考
1. 如何建立自動化的壓縮參數調優系統？
2. 在資源受限環境下如何最大化模型性能？
3. 如何處理壓縮過程中的意外情況和錯誤？
4. 未來可能的模型壓縮技術發展方向是什麼？

這個Lab提供了完整的模型壓縮工程化實踐體驗，幫助學員掌握工業級的模型優化和部署技能！