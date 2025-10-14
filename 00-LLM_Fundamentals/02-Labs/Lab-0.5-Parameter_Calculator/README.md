# Lab 0.5: LLM參數量與資源計算器

## 實驗目標

開發一個完整的LLM資源估算工具，實踐參數量計算、FLOPs估算和部署資源規劃，建立精確的資源評估能力。

## 學習成果

完成本實驗後，您將能夠：
- 精確計算Transformer模型的參數量
- 估算訓練和推理的計算複雜度
- 進行記憶體需求和成本分析
- 應用縮放法則進行性能預測

## 實驗環境要求

### 硬體要求
- RAM：8GB+系統記憶體
- 存儲：5GB可用空間

### 軟體要求
- Python 3.8+
- 已激活的poetry虛擬環境

## 主要實驗內容

### LLM資源計算器實現

```python
# llm_resource_calculator.py
"""
LLM資源計算器
提供參數量、FLOPs、記憶體需求等全面計算功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import streamlit as st
from datetime import datetime

@dataclass
class ModelConfig:
    """模型配置類"""
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

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

class LLMResourceCalculator:
    """LLM資源計算器"""

    def __init__(self):
        self.precision_bytes = {
            'fp32': 4, 'fp16': 2, 'bf16': 2,
            'int8': 1, 'int4': 0.5, 'nf4': 0.5
        }

    def calculate_parameters(self, config: ModelConfig) -> Dict:
        """計算模型參數量"""

        # 1. Embedding層參數
        token_embedding = config.vocab_size * config.d_model
        position_embedding = config.max_seq_len * config.d_model

        if config.tie_embeddings:
            embedding_params = token_embedding + position_embedding
            output_params = 0
        else:
            embedding_params = token_embedding + position_embedding
            output_params = config.vocab_size * config.d_model

        # 2. Attention層參數
        if config.attention_type == "mha":
            attention_params = 4 * config.d_model * config.d_model  # Q,K,V,O
        elif config.attention_type == "mqa":
            d_head = config.d_model // config.n_heads
            attention_params = 3 * config.d_model * config.d_model + 2 * config.d_model * d_head
        elif config.attention_type == "gqa":
            d_head = config.d_model // config.n_heads
            n_kv_heads = max(1, config.n_heads // 4)  # 假設KV頭數為Query頭數的1/4
            attention_params = 3 * config.d_model * config.d_model + 2 * config.d_model * n_kv_heads * d_head
        else:
            raise ValueError(f"Unsupported attention type: {config.attention_type}")

        # 3. FFN層參數
        if config.ffn_type == "standard":
            ffn_params = 2 * config.d_model * config.d_ff
        elif config.ffn_type == "gated":
            ffn_params = 3 * config.d_model * config.d_ff
        elif config.ffn_type == "moe":
            n_experts = 8  # 假設8個專家
            routing_params = config.d_model * n_experts
            expert_params = n_experts * 2 * config.d_model * config.d_ff
            ffn_params = routing_params + expert_params
        else:
            raise ValueError(f"Unsupported FFN type: {config.ffn_type}")

        # 4. LayerNorm參數
        layernorm_params = 2 * 2 * config.d_model  # 每層2個LayerNorm，每個2個參數

        # 5. 偏置項（如果使用）
        bias_params = 0
        if config.use_bias:
            bias_params = 4 * config.d_model + 2 * config.d_ff  # 注意力和FFN的偏置

        # 6. 每層總參數
        per_layer_params = attention_params + ffn_params + layernorm_params + bias_params

        # 7. 總參數
        total_params = (
            embedding_params +
            config.n_layers * per_layer_params +
            output_params +
            2 * config.d_model  # 最終LayerNorm
        )

        return {
            'total_parameters': total_params,
            'embedding_parameters': embedding_params,
            'attention_parameters_per_layer': attention_params,
            'ffn_parameters_per_layer': ffn_params,
            'layernorm_parameters_per_layer': layernorm_params,
            'per_layer_parameters': per_layer_params,
            'all_layers_parameters': config.n_layers * per_layer_params,
            'output_parameters': output_params,
            'parameter_breakdown': {
                'embedding_ratio': embedding_params / total_params * 100,
                'attention_ratio': (config.n_layers * attention_params) / total_params * 100,
                'ffn_ratio': (config.n_layers * ffn_params) / total_params * 100,
                'layernorm_ratio': (config.n_layers * layernorm_params + 2 * config.d_model) / total_params * 100,
                'output_ratio': output_params / total_params * 100
            }
        }

    def calculate_flops(self, config: ModelConfig, sequence_length: int,
                       batch_size: int = 1, generation_length: int = 0) -> Dict:
        """計算FLOPs"""

        # 前向傳播FLOPs計算
        def forward_flops(seq_len: int) -> int:
            # Attention FLOPs
            attention_flops = (
                4 * seq_len * config.d_model * config.d_model +  # QKV變換 + Output變換
                2 * config.n_heads * seq_len * seq_len * (config.d_model // config.n_heads)  # 注意力計算
            )

            # FFN FLOPs
            ffn_flops = 2 * seq_len * config.d_model * config.d_ff

            # 每層FLOPs
            layer_flops = attention_flops + ffn_flops

            # 所有層FLOPs
            model_flops = config.n_layers * layer_flops

            # 輸出層FLOPs（如果需要）
            if not config.tie_embeddings:
                output_flops = seq_len * config.d_model * config.vocab_size
                model_flops += output_flops

            return model_flops * batch_size

        # 推理階段計算
        if generation_length == 0:
            # 僅預填充
            total_flops = forward_flops(sequence_length)
            return {
                'total_flops': total_flops,
                'prefill_flops': total_flops,
                'decode_flops': 0
            }
        else:
            # 預填充 + 解碼
            prefill_flops = forward_flops(sequence_length)

            # 解碼階段（簡化計算）
            decode_flops = generation_length * forward_flops(1)  # 每步生成一個token

            total_flops = prefill_flops + decode_flops

            return {
                'total_flops': total_flops,
                'prefill_flops': prefill_flops,
                'decode_flops': decode_flops,
                'flops_breakdown': {
                    'prefill_ratio': prefill_flops / total_flops * 100,
                    'decode_ratio': decode_flops / total_flops * 100
                }
            }

    def calculate_memory(self, config: ModelConfig, batch_size: int,
                        sequence_length: int, precision: str = 'fp16',
                        training: bool = False) -> Dict:
        """計算記憶體需求"""

        param_result = self.calculate_parameters(config)
        total_params = param_result['total_parameters']
        bytes_per_param = self.precision_bytes[precision]

        # 1. 模型參數記憶體
        model_memory = total_params * bytes_per_param

        # 2. 激活值記憶體
        activation_memory = batch_size * sequence_length * config.d_model * bytes_per_param

        if training:
            # 訓練模式需要更多記憶體
            # 優化器狀態（Adam需要2倍參數記憶體）
            optimizer_memory = total_params * 4 * 2  # FP32精度存儲

            # 梯度記憶體
            gradient_memory = total_params * bytes_per_param

            # 激活值需要保存用於反向傳播
            activation_memory *= config.n_layers

            total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory

            return {
                'total_memory_gb': total_memory / (1024**3),
                'model_memory_gb': model_memory / (1024**3),
                'optimizer_memory_gb': optimizer_memory / (1024**3),
                'gradient_memory_gb': gradient_memory / (1024**3),
                'activation_memory_gb': activation_memory / (1024**3),
                'breakdown': {
                    'model_ratio': model_memory / total_memory * 100,
                    'optimizer_ratio': optimizer_memory / total_memory * 100,
                    'gradient_ratio': gradient_memory / total_memory * 100,
                    'activation_ratio': activation_memory / total_memory * 100
                }
            }
        else:
            # 推理模式
            # KV Cache記憶體
            kv_cache_memory = (
                2 * config.n_layers * batch_size * config.n_heads *
                sequence_length * (config.d_model // config.n_heads) * bytes_per_param
            )

            total_memory = model_memory + activation_memory + kv_cache_memory

            return {
                'total_memory_gb': total_memory / (1024**3),
                'model_memory_gb': model_memory / (1024**3),
                'activation_memory_gb': activation_memory / (1024**3),
                'kv_cache_memory_gb': kv_cache_memory / (1024**3),
                'breakdown': {
                    'model_ratio': model_memory / total_memory * 100,
                    'activation_ratio': activation_memory / total_memory * 100,
                    'kv_cache_ratio': kv_cache_memory / total_memory * 100
                }
            }

    def estimate_training_cost(self, config: ModelConfig, training_tokens: int,
                              gpu_type: str = 'A100', efficiency: float = 0.5) -> Dict:
        """估算訓練成本"""

        # GPU規格（簡化）
        gpu_specs = {
            'V100': {'flops_per_second': 125e12, 'cost_per_hour': 3.0},
            'A100': {'flops_per_second': 312e12, 'cost_per_hour': 4.0},
            'H100': {'flops_per_second': 1000e12, 'cost_per_hour': 8.0}
        }

        if gpu_type not in gpu_specs:
            raise ValueError(f"Unsupported GPU type: {gpu_type}")

        gpu_spec = gpu_specs[gpu_type]

        # 計算總FLOPs（簡化：6*N*D，N為參數數，D為數據量）
        param_result = self.calculate_parameters(config)
        total_params = param_result['total_parameters']
        total_flops = 6 * total_params * training_tokens  # 前向+反向

        # 計算訓練時間
        effective_flops_per_second = gpu_spec['flops_per_second'] * efficiency
        training_time_seconds = total_flops / effective_flops_per_second
        training_time_hours = training_time_seconds / 3600
        training_time_days = training_time_hours / 24

        # 計算成本
        total_cost = training_time_hours * gpu_spec['cost_per_hour']

        return {
            'total_flops': total_flops,
            'training_time_hours': training_time_hours,
            'training_time_days': training_time_days,
            'total_cost_usd': total_cost,
            'cost_per_billion_params': total_cost / (total_params / 1e9),
            'gpu_type': gpu_type,
            'efficiency': efficiency
        }

    def scaling_laws_prediction(self, compute_budget: float, law_type: str = 'chinchilla') -> Dict:
        """基於縮放法則的性能預測"""

        if law_type == 'chinchilla':
            # Chinchilla縮放法則參數
            alpha = 0.34
            beta = 0.28
            A = 406.4
            B = 410.7
            E = 1.69

            # 最優分配
            a = alpha / (alpha + beta)
            b = beta / (alpha + beta)

            optimal_params = (compute_budget / 6) ** a
            optimal_tokens = (compute_budget / 6) ** b

            # 預測損失
            loss_param = A / (optimal_params ** alpha)
            loss_data = B / (optimal_tokens ** beta)
            predicted_loss = E + loss_param + loss_data

            return {
                'optimal_parameters': optimal_params,
                'optimal_tokens': optimal_tokens,
                'predicted_loss': predicted_loss,
                'compute_budget': compute_budget,
                'params_ratio': a,
                'tokens_ratio': b
            }
        else:
            raise ValueError(f"Unsupported scaling law: {law_type}")

    def generate_comparison_table(self, configs: List[ModelConfig]) -> pd.DataFrame:
        """生成模型對比表格"""

        comparison_data = []

        for config in configs:
            param_result = self.calculate_parameters(config)

            row = {
                '模型名稱': config.name,
                '參數量(B)': f"{param_result['total_parameters'] / 1e9:.1f}",
                '詞表大小': f"{config.vocab_size:,}",
                '模型維度': config.d_model,
                '層數': config.n_layers,
                '注意力頭數': config.n_heads,
                'FFN維度': config.d_ff,
                '注意力比例(%)': f"{param_result['parameter_breakdown']['attention_ratio']:.1f}",
                'FFN比例(%)': f"{param_result['parameter_breakdown']['ffn_ratio']:.1f}"
            }

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

class LLMCalculatorUI:
    """計算器用戶界面"""

    def __init__(self):
        self.calculator = LLMResourceCalculator()

    def run_streamlit_app(self):
        """運行Streamlit應用"""

        st.title("🧮 LLM資源計算器")
        st.markdown("精確計算Transformer模型的參數量、FLOPs和記憶體需求")

        # 側邊欄配置
        st.sidebar.header("模型配置")

        # 預設配置選項
        preset_configs = {
            "自定義": None,
            "GPT-2 Small": ModelConfig("GPT-2 Small", 50257, 1024, 768, 12, 12, 3072),
            "LLaMA-7B": ModelConfig("LLaMA-7B", 32000, 2048, 4096, 32, 32, 11008),
            "LLaMA-13B": ModelConfig("LLaMA-13B", 32000, 2048, 5120, 40, 40, 13824)
        }

        selected_preset = st.sidebar.selectbox("選擇預設配置", list(preset_configs.keys()))

        if selected_preset != "自定義":
            config = preset_configs[selected_preset]
        else:
            # 自定義配置
            config = ModelConfig(
                name=st.sidebar.text_input("模型名稱", "Custom Model"),
                vocab_size=st.sidebar.number_input("詞表大小", 1000, 100000, 50000),
                max_seq_len=st.sidebar.number_input("最大序列長度", 512, 8192, 2048),
                d_model=st.sidebar.number_input("模型維度", 256, 8192, 512, step=64),
                n_layers=st.sidebar.number_input("層數", 1, 100, 6),
                n_heads=st.sidebar.number_input("注意力頭數", 1, 64, 8),
                d_ff=st.sidebar.number_input("FFN維度", 256, 32768, 2048, step=64),
            )

        # 主界面
        tab1, tab2, tab3, tab4 = st.tabs(["參數計算", "性能分析", "成本估算", "模型對比"])

        with tab1:
            st.header("📊 參數量計算")

            param_result = self.calculator.calculate_parameters(config)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("總參數量", f"{param_result['total_parameters']:,}")
                st.metric("Embedding參數", f"{param_result['embedding_parameters']:,}")
                st.metric("每層參數", f"{param_result['per_layer_parameters']:,}")

            with col2:
                st.metric("總參數(B)", f"{param_result['total_parameters'] / 1e9:.2f}")
                st.metric("Attention參數/層", f"{param_result['attention_parameters_per_layer']:,}")
                st.metric("FFN參數/層", f"{param_result['ffn_parameters_per_layer']:,}")

            # 參數分佈餅圖
            breakdown = param_result['parameter_breakdown']
            fig, ax = plt.subplots()
            sizes = [breakdown['embedding_ratio'], breakdown['attention_ratio'],
                    breakdown['ffn_ratio'], breakdown['layernorm_ratio']]
            labels = ['Embedding', 'Attention', 'FFN', 'LayerNorm']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%')
            ax.set_title('參數分佈')
            st.pyplot(fig)

        with tab2:
            st.header("⚡ 性能分析")

            col1, col2 = st.columns(2)

            with col1:
                batch_size = st.number_input("批次大小", 1, 32, 1)
                sequence_length = st.number_input("序列長度", 128, 4096, 512)

            with col2:
                precision = st.selectbox("精度", ["fp32", "fp16", "int8", "int4"])
                mode = st.selectbox("模式", ["推理", "訓練"])

            # FLOPs計算
            flops_result = self.calculator.calculate_flops(config, sequence_length, batch_size)

            st.subheader("計算複雜度 (FLOPs)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("總FLOPs", f"{flops_result['total_flops']:.2e}")
            with col2:
                if 'prefill_flops' in flops_result:
                    st.metric("預填充FLOPs", f"{flops_result['prefill_flops']:.2e}")
            with col3:
                if 'decode_flops' in flops_result:
                    st.metric("解碼FLOPs", f"{flops_result['decode_flops']:.2e}")

            # 記憶體計算
            memory_result = self.calculator.calculate_memory(
                config, batch_size, sequence_length, precision, mode == "訓練"
            )

            st.subheader("記憶體需求")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("總記憶體", f"{memory_result['total_memory_gb']:.2f} GB")
            with col2:
                st.metric("模型記憶體", f"{memory_result['model_memory_gb']:.2f} GB")
            with col3:
                if mode == "訓練":
                    st.metric("優化器記憶體", f"{memory_result['optimizer_memory_gb']:.2f} GB")
                else:
                    st.metric("KV Cache", f"{memory_result['kv_cache_memory_gb']:.2f} GB")

        with tab3:
            st.header("💰 成本估算")

            col1, col2 = st.columns(2)

            with col1:
                training_tokens = st.number_input("訓練Token數(B)", 1, 10000, 100) * 1e9
                gpu_type = st.selectbox("GPU類型", ["V100", "A100", "H100"])

            with col2:
                efficiency = st.slider("硬體效率", 0.1, 1.0, 0.5, 0.1)

            if st.button("計算訓練成本"):
                cost_result = self.calculator.estimate_training_cost(
                    config, training_tokens, gpu_type, efficiency
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("訓練時間(天)", f"{cost_result['training_time_days']:.1f}")
                with col2:
                    st.metric("總成本(USD)", f"${cost_result['total_cost_usd']:,.0f}")
                with col3:
                    st.metric("單位成本($/B參數)", f"${cost_result['cost_per_billion_params']:.0f}")

        with tab4:
            st.header("🔍 模型對比")

            # 創建對比模型列表
            comparison_configs = [
                ModelConfig("GPT-2 Small", 50257, 1024, 768, 12, 12, 3072),
                ModelConfig("GPT-2 Medium", 50257, 1024, 1024, 24, 16, 4096),
                ModelConfig("GPT-2 Large", 50257, 1024, 1280, 36, 20, 5120),
                ModelConfig("LLaMA-7B", 32000, 2048, 4096, 32, 32, 11008),
                ModelConfig("LLaMA-13B", 32000, 2048, 5120, 40, 40, 13824)
            ]

            comparison_df = self.calculator.generate_comparison_table(comparison_configs)
            st.dataframe(comparison_df)

            # 下載對比結果
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="下載對比結果",
                data=csv,
                file_name=f"llm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    """主函數"""

    print("=== LLM資源計算器 ===\\n")

    # 如果在Jupyter環境中，運行交互式計算
    calculator = LLMResourceCalculator()

    # 示例計算
    llama_7b = ModelConfig(
        name="LLaMA-7B",
        vocab_size=32000,
        max_seq_len=2048,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_ff=11008
    )

    print("1. 參數量計算:")
    param_result = calculator.calculate_parameters(llama_7b)
    print(f"總參數量: {param_result['total_parameters']:,} ({param_result['total_parameters']/1e9:.1f}B)")

    print("\\n2. FLOPs計算:")
    flops_result = calculator.calculate_flops(llama_7b, sequence_length=512, batch_size=1)
    print(f"推理FLOPs: {flops_result['total_flops']:.2e}")

    print("\\n3. 記憶體需求:")
    memory_result = calculator.calculate_memory(llama_7b, batch_size=1, sequence_length=512)
    print(f"推理記憶體: {memory_result['total_memory_gb']:.2f} GB")

    print("\\n4. 訓練成本估算:")
    cost_result = calculator.estimate_training_cost(llama_7b, training_tokens=1000e9)
    print(f"訓練成本: ${cost_result['total_cost_usd']:,.0f}")

    print("\\n=== 計算完成 ===")
    print("運行 'streamlit run llm_resource_calculator.py' 啟動Web界面")

if __name__ == "__main__":
    # 檢查是否在Streamlit環境
    try:
        import streamlit as st
        ui = LLMCalculatorUI()
        ui.run_streamlit_app()
    except ImportError:
        print("Streamlit未安裝，運行基礎計算模式")
        main()
```

## 使用指南

### 基礎模式運行
```bash
python llm_resource_calculator.py
```

### Web界面模式（推薦）
```bash
# 安裝Streamlit
pip install streamlit

# 啟動Web應用
streamlit run llm_resource_calculator.py
```

## 計算器功能特點

### 1. 精確參數計算
- 支持多種Transformer變體（MHA、MQA、GQA）
- 考慮不同FFN類型（標準、門控、MoE）
- 詳細的參數分解和比例分析

### 2. 全面性能分析
- FLOPs計算（訓練和推理）
- 記憶體需求估算（不同精度）
- KV Cache和激活值記憶體分析

### 3. 成本估算功能
- 基於實際GPU規格的訓練成本計算
- 考慮硬體效率和實際利用率
- 支持多種GPU類型對比

### 4. 縮放法則應用
- Chinchilla縮放法則實現
- 最優資源分配計算
- 性能預測功能

### 5. 直觀可視化界面
- 參數分佈餅圖
- 模型對比表格
- 交互式Web界面

## 實驗任務

### 任務1：模型規模分析
比較不同規模模型的資源需求差異

### 任務2：精度影響評估
分析不同量化精度對資源需求的影響

### 任務3：部署方案設計
為特定應用場景設計最優部署方案

### 任務4：成本效益分析
進行訓練和推理的完整成本分析

## 實驗報告要求

### 必答問題
1. **參數分佈**：分析不同規模模型的參數分佈特點
2. **資源縮放**：探討模型規模與資源需求的關係
3. **成本優化**：提出降低訓練和部署成本的策略
4. **實際應用**：為具體場景推薦合適的模型配置

### 延伸思考
1. 如何在資源約束下最大化模型性能？
2. 縮放法則在實際應用中的局限性是什麼？
3. 未來模型架構可能如何影響資源需求？

這個Lab提供了完整的LLM資源估算工具，幫助學員建立精確的資源評估和規劃能力。