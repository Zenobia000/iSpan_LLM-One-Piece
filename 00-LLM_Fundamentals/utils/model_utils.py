#!/usr/bin/env python3
"""
模型相關工具模組
提供模型載入、分析、操作的統一接口
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig
)
import time
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class ModelInfo:
    """模型信息數據類"""
    name: str
    parameters: int
    size_mb: float
    vocab_size: int
    max_length: int
    architecture: str
    precision: str

class ModelLoader:
    """統一模型載入器"""

    def __init__(self):
        self.loaded_models = {}
        self.tokenizers = {}

    def load_model(self, model_name: str, precision: str = 'fp16',
                   quantization: Optional[str] = None,
                   device_map: str = "auto") -> Tuple[object, object]:
        """
        統一模型載入接口

        Args:
            model_name: 模型名稱或路徑
            precision: 精度 ('fp32', 'fp16', 'bf16')
            quantization: 量化配置 ('int8', 'int4', 'nf4')
            device_map: 設備映射策略

        Returns:
            model, tokenizer: 載入的模型和分詞器
        """

        print(f"載入模型: {model_name} (精度: {precision}, 量化: {quantization})")

        try:
            # 設置精度
            torch_dtype_map = {
                'fp32': torch.float32,
                'fp16': torch.float16,
                'bf16': torch.bfloat16
            }
            torch_dtype = torch_dtype_map.get(precision, torch.float16)

            # 設置量化配置
            quantization_config = None
            if quantization == 'int8':
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == 'int4':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype
                )
            elif quantization == 'nf4':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True
                )

            # 載入分詞器
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 載入模型
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                device_map=device_map,
                low_cpu_mem_usage=True
            )

            # 緩存載入的模型
            cache_key = f"{model_name}_{precision}_{quantization}"
            self.loaded_models[cache_key] = model
            self.tokenizers[cache_key] = tokenizer

            print(f"✅ 模型載入成功")

            return model, tokenizer

        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise

    def get_model_info(self, model, tokenizer, model_name: str) -> ModelInfo:
        """獲取模型詳細信息"""

        try:
            # 計算參數量
            parameters = sum(p.numel() for p in model.parameters())

            # 估算模型大小
            size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            size_mb = size_bytes / (1024**2)

            # 獲取配置信息
            config = model.config
            vocab_size = getattr(config, 'vocab_size', len(tokenizer))
            max_length = getattr(config, 'max_position_embeddings', tokenizer.model_max_length)

            # 判斷架構類型
            architecture = self._identify_architecture(config)

            # 判斷精度
            precision = self._identify_precision(model)

            return ModelInfo(
                name=model_name,
                parameters=parameters,
                size_mb=size_mb,
                vocab_size=vocab_size,
                max_length=max_length,
                architecture=architecture,
                precision=precision
            )

        except Exception as e:
            print(f"獲取模型信息失敗: {e}")
            return None

    def _identify_architecture(self, config) -> str:
        """識別模型架構"""

        config_class = config.__class__.__name__.lower()

        if 'gpt' in config_class:
            return 'GPT'
        elif 'bert' in config_class:
            return 'BERT'
        elif 'llama' in config_class:
            return 'LLaMA'
        elif 't5' in config_class:
            return 'T5'
        else:
            return 'Transformer'

    def _identify_precision(self, model) -> str:
        """識別模型精度"""

        # 檢查第一個參數的數據類型
        first_param = next(model.parameters())
        dtype = first_param.dtype

        if dtype == torch.float32:
            return 'fp32'
        elif dtype == torch.float16:
            return 'fp16'
        elif dtype == torch.bfloat16:
            return 'bf16'
        else:
            return 'unknown'

class ModelAnalyzer:
    """模型分析器"""

    def __init__(self):
        self.analysis_cache = {}

    def analyze_parameter_distribution(self, model, model_name: str) -> Dict:
        """分析模型參數分佈"""

        print(f"分析模型參數分佈: {model_name}")

        param_distribution = {}
        total_params = 0

        # 按模組統計參數
        for name, param in model.named_parameters():
            # 提取模組類型
            module_type = self._extract_module_type(name)
            param_count = param.numel()

            if module_type not in param_distribution:
                param_distribution[module_type] = 0

            param_distribution[module_type] += param_count
            total_params += param_count

        # 計算百分比
        param_percentages = {
            module_type: count / total_params * 100
            for module_type, count in param_distribution.items()
        }

        analysis_result = {
            'total_parameters': total_params,
            'parameter_distribution': param_distribution,
            'parameter_percentages': param_percentages,
            'dominant_module': max(param_percentages, key=param_percentages.get),
            'model_complexity': self._assess_model_complexity(param_distribution, total_params)
        }

        print(f"  總參數量: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  主導模組: {analysis_result['dominant_module']}")

        return analysis_result

    def _extract_module_type(self, parameter_name: str) -> str:
        """從參數名提取模組類型"""

        name_parts = parameter_name.split('.')

        # 常見模組類型映射
        if 'embed' in parameter_name.lower():
            return 'embedding'
        elif any(attn_key in parameter_name.lower()
                for attn_key in ['attn', 'self_attn', 'attention']):
            return 'attention'
        elif any(ffn_key in parameter_name.lower()
                for ffn_key in ['mlp', 'ffn', 'feed_forward']):
            return 'ffn'
        elif 'norm' in parameter_name.lower():
            return 'normalization'
        elif any(out_key in parameter_name.lower()
                for out_key in ['lm_head', 'classifier', 'output']):
            return 'output'
        else:
            return 'other'

    def _assess_model_complexity(self, param_dist: Dict, total_params: int) -> str:
        """評估模型複雜度"""

        if total_params > 50e9:
            return 'very_large'
        elif total_params > 10e9:
            return 'large'
        elif total_params > 1e9:
            return 'medium'
        elif total_params > 100e6:
            return 'small'
        else:
            return 'tiny'

    def analyze_model_architecture(self, model) -> Dict:
        """分析模型架構特點"""

        config = model.config

        arch_info = {
            'model_type': getattr(config, 'model_type', 'unknown'),
            'num_layers': getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'unknown')),
            'hidden_size': getattr(config, 'hidden_size', getattr(config, 'n_embd', 'unknown')),
            'num_attention_heads': getattr(config, 'num_attention_heads', getattr(config, 'n_head', 'unknown')),
            'vocab_size': getattr(config, 'vocab_size', 'unknown'),
            'max_position_embeddings': getattr(config, 'max_position_embeddings', 'unknown')
        }

        # 計算架構效率指標
        if all(isinstance(v, int) for v in [arch_info['hidden_size'], arch_info['num_layers']]):
            arch_info['params_per_layer_millions'] = (
                4 * arch_info['hidden_size']**2 / 1e6
            )  # 簡化估算

        return arch_info

    def compare_models(self, models_info: List[ModelInfo]) -> Dict:
        """對比多個模型"""

        comparison_data = []

        for info in models_info:
            comparison_data.append({
                '模型名稱': info.name,
                '參數量': f"{info.parameters:,}",
                '參數量(B)': f"{info.parameters / 1e9:.2f}",
                '模型大小(MB)': f"{info.size_mb:.1f}",
                '詞表大小': f"{info.vocab_size:,}",
                '最大長度': info.max_length,
                '架構': info.architecture,
                '精度': info.precision
            })

        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)

        return {
            'comparison_table': comparison_df,
            'model_count': len(models_info),
            'size_range': f"{min(info.size_mb for info in models_info):.1f}-{max(info.size_mb for info in models_info):.1f} MB",
            'param_range': f"{min(info.parameters for info in models_info)/1e9:.2f}-{max(info.parameters for info in models_info)/1e9:.2f}B"
        }

def quick_model_check(model_name: str, precision: str = 'fp16') -> Dict:
    """快速模型檢查工具"""

    try:
        loader = ModelLoader()
        model, tokenizer = loader.load_model(model_name, precision)

        analyzer = ModelAnalyzer()
        model_info = loader.get_model_info(model, tokenizer, model_name)
        param_dist = analyzer.analyze_parameter_distribution(model, model_name)
        arch_info = analyzer.analyze_model_architecture(model)

        return {
            'model_info': model_info,
            'parameter_distribution': param_dist,
            'architecture_info': arch_info,
            'check_status': 'success'
        }

    except Exception as e:
        return {
            'check_status': 'failed',
            'error': str(e)
        }

if __name__ == "__main__":
    # 測試工具模組
    print("測試模型工具模組...")

    result = quick_model_check("microsoft/DialoGPT-small")

    if result['check_status'] == 'success':
        info = result['model_info']
        print(f"✅ 模型檢查成功")
        print(f"   參數量: {info.parameters:,}")
        print(f"   大小: {info.size_mb:.1f} MB")
        print(f"   架構: {info.architecture}")
    else:
        print(f"❌ 模型檢查失敗: {result['error']}")