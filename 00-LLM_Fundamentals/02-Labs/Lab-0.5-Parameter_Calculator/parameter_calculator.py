#!/usr/bin/env python3
"""
LLM參數量與計算複雜度精確計算工具
實現Transformer架構的精確參數計算和資源估算
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

@dataclass
class ModelConfig:
    """模型配置數據類"""
    name: str
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: Optional[int] = None
    use_bias: bool = False
    tie_embeddings: bool = True
    attention_type: str = "mha"  # mha, mqa, gqa
    ffn_type: str = "standard"  # standard, gated, moe
    n_kv_heads: Optional[int] = None  # for GQA
    n_experts: Optional[int] = None   # for MoE

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.n_kv_heads is None and self.attention_type == 'gqa':
            self.n_kv_heads = max(1, self.n_heads // 4)
        if self.n_experts is None and self.ffn_type == 'moe':
            self.n_experts = 8

class ParameterCalculator:
    """LLM參數計算器"""

    def __init__(self):
        self.precision_bytes = {
            'fp32': 4, 'fp16': 2, 'bf16': 2,
            'int8': 1, 'int4': 0.5, 'nf4': 0.5
        }

    def calculate_transformer_parameters(self, config: ModelConfig) -> Dict:
        """
        精確計算Transformer模型參數量

        參數分解：
        1. Embedding層：Token Embedding + Position Embedding
        2. Transformer層：Attention + FFN + LayerNorm
        3. 輸出層：Language Modeling Head (可選擇與輸入共享)
        """

        print(f"計算模型參數量: {config.name}")

        # 1. Embedding層參數
        token_embedding = config.vocab_size * config.d_model
        position_embedding = config.max_seq_len * config.d_model

        if config.tie_embeddings:
            embedding_params = token_embedding + position_embedding
            output_params = 0  # 輸出層與輸入embedding共享
            print(f"  Token Embedding (共享): {token_embedding:,} 參數")
        else:
            embedding_params = token_embedding + position_embedding
            output_params = config.vocab_size * config.d_model
            print(f"  Token Embedding: {token_embedding:,} 參數")
            print(f"  Output Layer: {output_params:,} 參數")

        print(f"  Position Embedding: {position_embedding:,} 參數")

        # 2. Attention層參數
        attention_params = self._calculate_attention_params(config)
        print(f"  Attention (per layer): {attention_params:,} 參數")

        # 3. FFN層參數
        ffn_params = self._calculate_ffn_params(config)
        print(f"  FFN (per layer): {ffn_params:,} 參數")

        # 4. LayerNorm參數
        layernorm_params = 2 * 2 * config.d_model  # 每層2個LayerNorm，每個2個參數
        print(f"  LayerNorm (per layer): {layernorm_params:,} 參數")

        # 5. 偏置項
        bias_params = 0
        if config.use_bias:
            bias_params = self._calculate_bias_params(config)
            print(f"  Bias terms (per layer): {bias_params:,} 參數")

        # 6. 每層總參數
        per_layer_params = attention_params + ffn_params + layernorm_params + bias_params

        # 7. 總參數計算
        total_params = (
            embedding_params +                    # 輸入embedding
            config.n_layers * per_layer_params + # 所有Transformer層
            output_params +                       # 輸出層
            2 * config.d_model                   # 最終LayerNorm
        )

        print(f"\n總參數量: {total_params:,} ({total_params/1e9:.2f}B)")

        # 詳細分解
        parameter_breakdown = {
            'total_parameters': total_params,
            'embedding_parameters': embedding_params,
            'attention_parameters_per_layer': attention_params,
            'ffn_parameters_per_layer': ffn_params,
            'layernorm_parameters_per_layer': layernorm_params,
            'bias_parameters_per_layer': bias_params,
            'per_layer_parameters': per_layer_params,
            'all_layers_parameters': config.n_layers * per_layer_params,
            'output_parameters': output_params,
            'parameter_percentages': {
                'embedding_ratio': embedding_params / total_params * 100,
                'attention_ratio': (config.n_layers * attention_params) / total_params * 100,
                'ffn_ratio': (config.n_layers * ffn_params) / total_params * 100,
                'layernorm_ratio': (config.n_layers * layernorm_params + 2 * config.d_model) / total_params * 100,
                'output_ratio': output_params / total_params * 100
            }
        }

        return parameter_breakdown

    def _calculate_attention_params(self, config: ModelConfig) -> int:
        """計算Attention層參數"""

        if config.attention_type == "mha":
            # Multi-Head Attention: Q, K, V, O 四個變換矩陣
            return 4 * config.d_model * config.d_model

        elif config.attention_type == "mqa":
            # Multi-Query Attention: Q為多頭，K,V為單頭
            d_head = config.d_model // config.n_heads
            return (
                3 * config.d_model * config.d_model +  # Q, O變換 + K,V變換
                2 * config.d_model * d_head             # K,V單頭參數
            )

        elif config.attention_type == "gqa":
            # Grouped-Query Attention
            d_head = config.d_model // config.n_heads
            kv_heads = config.n_kv_heads
            return (
                3 * config.d_model * config.d_model +  # Q, O變換
                2 * config.d_model * kv_heads * d_head # K,V分組參數
            )

        else:
            raise ValueError(f"Unsupported attention type: {config.attention_type}")

    def _calculate_ffn_params(self, config: ModelConfig) -> int:
        """計算FFN層參數"""

        if config.ffn_type == "standard":
            # 標準FFN: d_model -> d_ff -> d_model
            return 2 * config.d_model * config.d_ff

        elif config.ffn_type == "gated":
            # 門控FFN (如SwiGLU): 需要額外的門控變換
            return 3 * config.d_model * config.d_ff

        elif config.ffn_type == "moe":
            # Mixture of Experts
            routing_params = config.d_model * config.n_experts
            expert_params = config.n_experts * 2 * config.d_model * config.d_ff
            return routing_params + expert_params

        else:
            raise ValueError(f"Unsupported FFN type: {config.ffn_type}")

    def _calculate_bias_params(self, config: ModelConfig) -> int:
        """計算偏置參數"""

        attention_bias = 4 * config.d_model  # Q,K,V,O的偏置
        ffn_bias = 2 * config.d_ff           # FFN兩層的偏置

        return attention_bias + ffn_bias

    def calculate_flops(self, config: ModelConfig, batch_size: int,
                       sequence_length: int, mode: str = 'inference') -> Dict:
        """
        計算FLOPs（浮點運算次數）

        計算公式：
        - Forward FLOPs ≈ 2 × P × N (P=參數數，N=token數)
        - Training FLOPs ≈ 6 × P × N (包含前向、反向、優化器)
        """

        print(f"計算FLOPs: {config.name} ({mode}模式)")

        param_result = self.calculate_transformer_parameters(config)
        total_params = param_result['total_parameters']

        if mode == 'training':
            # 訓練模式：前向 + 反向 + 優化器更新
            total_tokens = batch_size * sequence_length

            # 前向傳播
            forward_flops = 2 * total_params * total_tokens

            # 反向傳播（約為前向的2倍）
            backward_flops = 4 * total_params * total_tokens

            # 優化器更新（相對較小）
            optimizer_flops = total_params  # 簡化估算

            total_flops = forward_flops + backward_flops + optimizer_flops

            flops_breakdown = {
                'forward_flops': forward_flops,
                'backward_flops': backward_flops,
                'optimizer_flops': optimizer_flops,
                'total_flops': total_flops,
                'flops_per_token': total_flops / total_tokens
            }

        else:
            # 推理模式：僅前向傳播
            total_tokens = batch_size * sequence_length
            forward_flops = 2 * total_params * total_tokens

            flops_breakdown = {
                'forward_flops': forward_flops,
                'total_flops': forward_flops,
                'flops_per_token': forward_flops / total_tokens
            }

        print(f"  總FLOPs: {flops_breakdown['total_flops']:.2e}")
        print(f"  每Token FLOPs: {flops_breakdown['flops_per_token']:.2e}")

        return flops_breakdown

    def calculate_memory_requirements(self, config: ModelConfig, batch_size: int,
                                    sequence_length: int, precision: str = 'fp16',
                                    mode: str = 'inference') -> Dict:
        """
        計算記憶體需求

        記憶體組成：
        - 模型參數記憶體
        - 激活值記憶體
        - KV Cache記憶體（推理）
        - 優化器狀態記憶體（訓練）
        - 梯度記憶體（訓練）
        """

        print(f"計算記憶體需求: {config.name} ({mode}模式, {precision}精度)")

        param_result = self.calculate_transformer_parameters(config)
        total_params = param_result['total_parameters']
        bytes_per_param = self.precision_bytes[precision]

        # 模型參數記憶體
        model_memory_bytes = total_params * bytes_per_param

        if mode == 'training':
            # 訓練模式記憶體
            # 優化器狀態（Adam: momentum + variance）
            optimizer_memory_bytes = total_params * 8  # FP32精度存儲

            # 梯度記憶體
            gradient_memory_bytes = total_params * bytes_per_param

            # 激活值記憶體（需要保存所有層）
            activation_per_layer = batch_size * sequence_length * config.d_model * bytes_per_param
            total_activation_bytes = activation_per_layer * config.n_layers

            total_memory_bytes = (
                model_memory_bytes + optimizer_memory_bytes +
                gradient_memory_bytes + total_activation_bytes
            )

            memory_breakdown = {
                'model_memory_gb': model_memory_bytes / (1024**3),
                'optimizer_memory_gb': optimizer_memory_bytes / (1024**3),
                'gradient_memory_gb': gradient_memory_bytes / (1024**3),
                'activation_memory_gb': total_activation_bytes / (1024**3),
                'total_memory_gb': total_memory_bytes / (1024**3)
            }

        else:
            # 推理模式記憶體
            # KV Cache記憶體
            d_head = config.d_model // config.n_heads
            kv_cache_bytes = (
                2 * config.n_layers * batch_size * config.n_heads *
                sequence_length * d_head * bytes_per_param
            )

            # 當前激活值記憶體（只需要當前層）
            current_activation_bytes = batch_size * sequence_length * config.d_model * bytes_per_param

            total_memory_bytes = model_memory_bytes + kv_cache_bytes + current_activation_bytes

            memory_breakdown = {
                'model_memory_gb': model_memory_bytes / (1024**3),
                'kv_cache_memory_gb': kv_cache_bytes / (1024**3),
                'activation_memory_gb': current_activation_bytes / (1024**3),
                'total_memory_gb': total_memory_bytes / (1024**3)
            }

        for key, value in memory_breakdown.items():
            print(f"  {key}: {value:.2f} GB")

        return memory_breakdown

    def estimate_training_cost(self, config: ModelConfig, training_tokens: int,
                             gpu_type: str = 'A100', efficiency: float = 0.5,
                             electricity_cost_per_kwh: float = 0.1) -> Dict:
        """
        估算訓練成本

        成本組成：
        - GPU硬體成本
        - 電力成本
        - 時間成本
        """

        print(f"估算訓練成本: {config.name}")

        # GPU規格數據庫
        gpu_specs = {
            'V100': {'flops': 125e12, 'power_w': 300, 'price_usd': 8000},
            'A100': {'flops': 312e12, 'power_w': 400, 'price_usd': 15000},
            'H100': {'flops': 1000e12, 'power_w': 700, 'price_usd': 30000}
        }

        if gpu_type not in gpu_specs:
            raise ValueError(f"Unsupported GPU type: {gpu_type}")

        gpu_spec = gpu_specs[gpu_type]

        # 計算總FLOPs
        flop_result = self.calculate_flops(config, 1, training_tokens, 'training')
        total_flops = flop_result['total_flops']

        # 計算訓練時間
        effective_flops_per_second = gpu_spec['flops'] * efficiency
        training_time_seconds = total_flops / effective_flops_per_second
        training_time_hours = training_time_seconds / 3600
        training_time_days = training_time_hours / 24

        # 計算成本
        gpu_cost = gpu_spec['price_usd']
        power_cost = (gpu_spec['power_w'] / 1000) * training_time_hours * electricity_cost_per_kwh
        total_cost = gpu_cost + power_cost  # 簡化：不考慮折舊

        cost_breakdown = {
            'total_flops': total_flops,
            'training_time_hours': training_time_hours,
            'training_time_days': training_time_days,
            'gpu_hardware_cost': gpu_cost,
            'electricity_cost': power_cost,
            'total_cost_usd': total_cost,
            'cost_per_billion_params': total_cost / (config.d_model * config.n_layers / 1e9),
            'cost_per_billion_tokens': total_cost / (training_tokens / 1e9)
        }

        print(f"  訓練時間: {training_time_days:.2f} 天")
        print(f"  總成本: ${total_cost:,.0f}")
        print(f"  每十億參數成本: ${cost_breakdown['cost_per_billion_params']:,.0f}")

        return cost_breakdown

    def apply_scaling_laws(self, compute_budget: float, law_type: str = 'chinchilla') -> Dict:
        """
        應用縮放法則進行資源分配優化

        Chinchilla縮放法則：
        - 最優參數量 ∝ C^0.73
        - 最優數據量 ∝ C^0.27
        - 其中C為計算預算（FLOPs）
        """

        print(f"應用{law_type}縮放法則, 計算預算: {compute_budget:.2e} FLOPs")

        if law_type == 'chinchilla':
            # Chinchilla縮放法則參數
            alpha = 0.34  # 參數縮放指數
            beta = 0.28   # 數據縮放指數
            A = 406.4     # 參數係數
            B = 410.7     # 數據係數
            E = 1.69      # 基礎損失

            # 計算最優分配
            a = alpha / (alpha + beta)  # ≈ 0.55
            b = beta / (alpha + beta)   # ≈ 0.45

            # 最優參數量和數據量
            optimal_params = (compute_budget / 6) ** a
            optimal_tokens = (compute_budget / 6) ** b

            # 預測性能
            loss_param = A / (optimal_params ** alpha)
            loss_data = B / (optimal_tokens ** beta)
            predicted_loss = E + loss_param + loss_data

            scaling_result = {
                'law_type': law_type,
                'compute_budget': compute_budget,
                'optimal_parameters': optimal_params,
                'optimal_tokens': optimal_tokens,
                'predicted_loss': predicted_loss,
                'params_ratio': a,
                'tokens_ratio': b,
                'efficiency_score': (optimal_params * optimal_tokens) / compute_budget
            }

            print(f"  最優參數量: {optimal_params:.2e} ({optimal_params/1e9:.1f}B)")
            print(f"  最優數據量: {optimal_tokens:.2e} ({optimal_tokens/1e9:.0f}B tokens)")
            print(f"  預測Loss: {predicted_loss:.3f}")

        else:
            raise ValueError(f"Unsupported scaling law: {law_type}")

        return scaling_result

    def hardware_matching_calculator(self, model_config: ModelConfig,
                                   target_gpus: Dict, workload: Dict) -> Dict:
        """
        硬體匹配計算器

        根據模型配置和工作負載要求，計算最適合的GPU配置
        """

        print(f"=== 硬體匹配計算: {model_config.name} ===")

        # 計算模型記憶體需求
        memory_req = self.calculate_memory_requirements(
            model_config,
            workload.get('batch_size', 8),
            workload.get('sequence_length', 2048),
            workload.get('precision', 'fp16'),
            workload.get('mode', 'inference')
        )

        required_memory_gb = memory_req['total_memory_gb']
        print(f"記憶體需求: {required_memory_gb:.1f} GB")

        # 計算各GPU方案
        matching_results = {}

        for gpu_name, gpu_specs in target_gpus.items():
            # 考慮安全餘量
            safety_margin = 0.2  # 20%安全餘量
            available_memory = gpu_specs['memory_gb'] * (1 - safety_margin)

            # 計算最小GPU數量
            min_gpus = max(1, np.ceil(required_memory_gb / available_memory))

            # 計算實際記憶體利用率
            actual_utilization = required_memory_gb / (min_gpus * gpu_specs['memory_gb']) * 100

            # 計算成本
            total_cost = min_gpus * gpu_specs['price_usd']
            cost_per_gb = total_cost / required_memory_gb

            matching_results[gpu_name] = {
                'min_gpus': int(min_gpus),
                'memory_utilization_percent': min(100, actual_utilization),
                'total_cost_usd': total_cost,
                'cost_per_gb': cost_per_gb,
                'cost_efficiency': gpu_specs['memory_gb'] / gpu_specs['price_usd'],
                'recommended': actual_utilization >= 50 and actual_utilization <= 90  # 合理利用率
            }

        # 按成本效益排序
        sorted_options = sorted(
            matching_results.items(),
            key=lambda x: x[1]['cost_per_gb']
        )

        print("\n硬體匹配結果:")
        for gpu_name, specs in sorted_options:
            status = "✅ 推薦" if specs['recommended'] else "⚠️ 考慮"
            print(f"  {status} {gpu_name}: {specs['min_gpus']} GPUs, "
                  f"利用率 {specs['memory_utilization_percent']:.1f}%, "
                  f"成本 ${specs['total_cost_usd']:,}")

        return {
            'memory_requirements': memory_req,
            'gpu_matching_results': matching_results,
            'best_option': sorted_options[0] if sorted_options else None,
            'workload_config': workload
        }

    def compare_model_configurations(self, configs: List[ModelConfig]) -> pd.DataFrame:
        """對比多個模型配置"""

        print("\n=== 模型配置對比 ===")

        comparison_data = []

        for config in configs:
            param_result = self.calculate_transformer_parameters(config)

            row = {
                '模型名稱': config.name,
                '參數量(B)': f"{param_result['total_parameters'] / 1e9:.2f}",
                '詞表大小': f"{config.vocab_size:,}",
                '模型維度': config.d_model,
                '層數': config.n_layers,
                '注意力頭數': config.n_heads,
                'FFN倍數': f"{config.d_ff / config.d_model:.1f}",
                '注意力類型': config.attention_type.upper(),
                'Attention佔比': f"{param_result['parameter_percentages']['attention_ratio']:.1f}%",
                'FFN佔比': f"{param_result['parameter_percentages']['ffn_ratio']:.1f}%"
            }

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        return comparison_df

    def demonstrate_scaling_laws(self):
        """演示縮放法則應用"""

        print("\n=== 縮放法則演示 ===")

        # 不同計算預算的分析
        compute_budgets = [1e20, 1e21, 1e22, 1e23, 1e24]  # FLOPs

        scaling_results = []

        print("計算預算與最優配置:")
        print(f"{'計算預算':<12} {'最優參數(B)':<12} {'最優數據(B)':<12} {'預測Loss':<10}")
        print("-" * 55)

        for budget in compute_budgets:
            result = self.apply_scaling_laws(budget, 'chinchilla')

            params_b = result['optimal_parameters'] / 1e9
            tokens_b = result['optimal_tokens'] / 1e9
            loss = result['predicted_loss']

            print(f"{budget:.1e}     {params_b:8.1f}      {tokens_b:8.0f}        {loss:.3f}")

            scaling_results.append(result)

        # 現有模型效率分析
        print("\n現有模型vs最優配置對比:")
        existing_models = [
            {'name': 'GPT-3', 'params': 175e9, 'tokens': 300e9},
            {'name': 'LLaMA-7B', 'params': 7e9, 'tokens': 1000e9},
            {'name': 'LLaMA-65B', 'params': 65e9, 'tokens': 1400e9}
        ]

        for model in existing_models:
            compute_budget = 6 * model['params'] * model['tokens']
            optimal = self.apply_scaling_laws(compute_budget, 'chinchilla')
            efficiency = optimal['predicted_loss'] / self._predict_actual_loss(model['params'], model['tokens'])

            print(f"{model['name']:10} 效率係數: {efficiency:.3f}")

        return scaling_results

    def _predict_actual_loss(self, params: float, tokens: float) -> float:
        """基於Chinchilla公式預測實際Loss"""

        alpha, beta = 0.34, 0.28
        A, B, E = 406.4, 410.7, 1.69

        loss_param = A / (params ** alpha)
        loss_data = B / (tokens ** beta)

        return E + loss_param + loss_data

    def visualize_parameter_analysis(self, configs: List[ModelConfig]):
        """可視化參數分析"""

        print("\n=== 生成參數分析可視化 ===")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 參數量對比
        model_names = [config.name for config in configs]
        param_counts = []
        for config in configs:
            result = self.calculate_transformer_parameters(config)
            param_counts.append(result['total_parameters'] / 1e9)

        bars1 = ax1.bar(model_names, param_counts, color='skyblue', alpha=0.8)
        ax1.set_title('模型參數量對比')
        ax1.set_ylabel('參數量 (Billions)')
        ax1.tick_params(axis='x', rotation=45)

        for bar, count in zip(bars1, param_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count:.1f}B', ha='center', va='bottom')

        # 2. 參數分佈（以第一個模型為例）
        if configs:
            config = configs[0]
            result = self.calculate_transformer_parameters(config)
            percentages = result['parameter_percentages']

            labels = ['Embedding', 'Attention', 'FFN', 'LayerNorm', 'Output']
            sizes = [
                percentages['embedding_ratio'],
                percentages['attention_ratio'],
                percentages['ffn_ratio'],
                percentages['layernorm_ratio'],
                percentages['output_ratio']
            ]

            ax2.pie(sizes, labels=labels, autopct='%1.1f%%')
            ax2.set_title(f'{config.name} 參數分佈')

        # 3. 記憶體需求對比（推理模式）
        memory_requirements = []
        for config in configs:
            memory_req = self.calculate_memory_requirements(config, 8, 2048, 'fp16', 'inference')
            memory_requirements.append(memory_req['total_memory_gb'])

        bars3 = ax3.bar(model_names, memory_requirements, color='lightgreen', alpha=0.8)
        ax3.set_title('推理記憶體需求對比')
        ax3.set_ylabel('記憶體需求 (GB)')
        ax3.tick_params(axis='x', rotation=45)

        for bar, mem in zip(bars3, memory_requirements):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mem:.1f}GB', ha='center', va='bottom')

        # 4. FLOPs對比
        flops_requirements = []
        for config in configs:
            flop_result = self.calculate_flops(config, 1, 2048, 'inference')
            flops_requirements.append(flop_result['total_flops'])

        ax4.bar(model_names, flops_requirements, color='salmon', alpha=0.8)
        ax4.set_title('推理FLOPs需求對比')
        ax4.set_ylabel('FLOPs')
        ax4.set_yscale('log')  # 對數座標
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('parameter_analysis_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("可視化圖表已保存: parameter_analysis_comparison.png")

def create_example_configs() -> List[ModelConfig]:
    """創建示例模型配置"""

    configs = [
        ModelConfig("GPT-2 Small", 50257, 1024, 768, 12, 12, 3072),
        ModelConfig("LLaMA-7B", 32000, 2048, 4096, 32, 32, 11008),
        ModelConfig("LLaMA-13B", 32000, 2048, 5120, 40, 40, 13824),
        ModelConfig("Custom-MQA", 32000, 2048, 4096, 32, 32, 11008, attention_type="mqa"),
        ModelConfig("Custom-MoE", 32000, 2048, 4096, 32, 32, 11008, ffn_type="moe", n_experts=8)
    ]

    return configs

def main():
    """主函數演示"""

    print("LLM參數量與計算複雜度計算工具")
    print("=" * 60)

    # 初始化計算器
    calculator = ParameterCalculator()

    # 創建示例配置
    configs = create_example_configs()

    print(f"\n📊 將分析 {len(configs)} 個模型配置")

    # 1. 參數量計算演示
    print("\n1. 參數量詳細計算:")
    for config in configs[:2]:  # 演示前2個
        param_result = calculator.calculate_transformer_parameters(config)

    # 2. FLOPs計算演示
    print("\n2. FLOPs計算演示:")
    flop_result = calculator.calculate_flops(configs[1], 8, 2048, 'inference')

    # 3. 記憶體需求演示
    print("\n3. 記憶體需求分析:")
    memory_result = calculator.calculate_memory_requirements(configs[1], 8, 2048, 'fp16', 'inference')

    # 4. 訓練成本估算
    print("\n4. 訓練成本估算:")
    cost_result = calculator.estimate_training_cost(configs[1], 100e9, 'A100')

    # 5. 硬體匹配演示
    print("\n5. 硬體匹配分析:")
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

    hardware_result = calculator.hardware_matching_calculator(configs[1], target_gpus, workload)

    # 6. 模型對比
    print("\n6. 模型配置對比:")
    comparison_df = calculator.compare_model_configurations(configs)

    # 7. 縮放法則演示
    scaling_results = calculator.demonstrate_scaling_laws()

    # 8. 可視化
    calculator.visualize_parameter_analysis(configs)

    # 9. 生成報告
    report = generate_comprehensive_report(calculator, configs)

    print("\n✅ 分析完成！")
    print("📁 生成文件:")
    print("   - parameter_analysis_report.md")
    print("   - parameter_analysis_comparison.png")

    print("\n🎓 關鍵學習要點:")
    print("1. 參數量計算需要考慮架構變體的差異")
    print("2. 記憶體需求包含多個組件，KV Cache影響較大")
    print("3. 縮放法則可以指導資源配置優化")
    print("4. 硬體匹配需要綜合考慮成本和利用率")

def generate_comprehensive_report(calculator, configs) -> str:
    """生成綜合報告"""

    report = f"""# LLM參數量與計算複雜度分析報告

生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 分析概述

本報告對{len(configs)}個不同的LLM配置進行了全面的參數量、計算複雜度和資源需求分析。

## 主要發現

### 1. 參數量分佈規律
- FFN層通常佔總參數量的60-70%
- Attention層佔20-30%
- Embedding層和LayerNorm佔比較小

### 2. 架構變體影響
- MQA相比MHA可減少KV參數約75%
- MoE大幅增加參數量但計算量可控
- 不同架構的記憶體/計算特性差異明顯

### 3. 硬體需求特點
- 推理記憶體主要由模型參數和KV Cache組成
- 訓練記憶體需求是推理的3-5倍
- 批次大小和序列長度對記憶體影響巨大

### 4. 成本優化策略
- 根據縮放法則優化參數量和數據量分配
- 選擇合適的GPU配置平衡成本和性能
- 考慮量化技術降低部署成本

## 實際應用建議

1. **模型選型**: 根據應用需求和資源約束選擇合適規模
2. **硬體配置**: 使用硬體匹配計算器確定最優GPU配置
3. **成本控制**: 應用縮放法則優化訓練資源分配
4. **部署優化**: 結合量化技術降低推理成本

---
*本報告基於理論計算生成，實際部署時請結合具體測試結果。*
"""

    with open('parameter_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    return report

if __name__ == "__main__":
    main()