# LLM模型壓縮工程化快速指南

## 📋 總覽

本指南提供從模型載入到推理部署的完整工程化壓縮流程實用方案。

> **💡 詳細計算公式參考**: `01-Theory/Parameter_Estimation/README.md` 第0.5.6-0.5.7節

## 🎯 快速決策表

### 模型規模與硬體匹配

| 模型規模 | 推薦GPU | 量化策略 | 預期壓縮比 | 適用場景 |
|----------|---------|----------|------------|----------|
| <1B | RTX 4070 | INT8/INT4 | 4-8x | 邊緣設備 |
| 1B-7B | RTX 4090 | FP16→INT8 | 2-4x | 個人工作站 |
| 7B-30B | A100 40GB | AWQ/GPTQ | 2-4x | 小型服務 |
| 30B-70B | A100 80GB ×2 | GPTQ+分片 | 4x+ | 企業服務 |
| >70B | H100 ×4+ | 混合壓縮 | 自定義 | 大規模服務 |

### 壓縮方法選擇

```python
def choose_compression_method(model_size_gb: float, target_device: str) -> str:
    """壓縮方法選擇決策函數"""

    if target_device == "edge":
        return "INT4量化 + 結構化剪枝"
    elif model_size_gb < 4:
        return "INT8 PTQ"
    elif model_size_gb < 16:
        return "AWQ量化"
    else:
        return "GPTQ + 模型分片"
```

## 🔧 4階段實施流程

### 階段1: 快速分析（10分鐘）
```bash
# 1. 檢查模型基本信息
python -c "from transformers import AutoModel; model=AutoModel.from_pretrained('model_name'); print(f'參數量: {model.num_parameters():,}')"

# 2. 估算記憶體需求
python quick_memory_check.py --model model_name --precision fp16
```

### 階段2: 策略選擇（5分鐘）
```python
# 快速策略選擇
strategy = {
    "小模型(<3GB)": "INT8_PTQ",
    "中模型(3-15GB)": "AWQ_INT8",
    "大模型(>15GB)": "GPTQ_INT4"
}
```

### 階段3: 一鍵壓縮（30分鐘）
```python
# 統一壓縮接口
from compression_toolkit import AutoCompress

compressor = AutoCompress(model_name, target_memory_gb=16)
compressed_model = compressor.compress()  # 自動選擇最佳策略
```

### 階段4: 部署驗證（15分鐘）
```python
# 快速部署測試
from deployment_tester import DeploymentValidator

validator = DeploymentValidator(compressed_model)
results = validator.run_quick_test()  # 功能、性能、準確性測試
```

## 🛠️ 實用工具腳本

### 記憶體需求快速檢查
```python
# quick_memory_check.py
import sys
from transformers import AutoConfig

def quick_memory_estimate(model_name, precision='fp16'):
    config = AutoConfig.from_pretrained(model_name)

    # 基本參數量估算
    if hasattr(config, 'n_parameters'):
        params = config.n_parameters
    else:
        # 估算公式
        params = config.n_layer * config.n_embd * config.n_embd * 6

    # 記憶體估算
    precision_bytes = {'fp32': 4, 'fp16': 2, 'int8': 1, 'int4': 0.5}
    model_memory_gb = params * precision_bytes[precision] / (1024**3)

    # KV Cache估算（假設批次=8，序列=2048）
    kv_cache_gb = (2 * config.n_layer * 8 * config.n_head *
                   2048 * (config.n_embd // config.n_head) *
                   precision_bytes[precision]) / (1024**3)

    total_memory = model_memory_gb + kv_cache_gb

    print(f"模型: {model_name}")
    print(f"參數量: {params:,} ({params/1e9:.1f}B)")
    print(f"模型記憶體: {model_memory_gb:.1f} GB")
    print(f"KV Cache: {kv_cache_gb:.1f} GB")
    print(f"總記憶體需求: {total_memory:.1f} GB")

    # GPU推薦
    if total_memory <= 16:
        print("推薦GPU: RTX 4090 (24GB)")
    elif total_memory <= 24:
        print("推薦GPU: RTX A6000 (48GB)")
    elif total_memory <= 80:
        print("推薦GPU: A100 80GB")
    else:
        print("推薦: 多GPU配置或雲端服務")

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "microsoft/DialoGPT-small"
    quick_memory_estimate(model_name)
```

### 一鍵壓縮工具
```python
# auto_compress.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class AutoCompress:
    """自動壓縮工具"""

    def __init__(self, model_name, target_memory_gb=16):
        self.model_name = model_name
        self.target_memory = target_memory_gb

    def compress(self):
        """自動選擇最佳壓縮策略並執行"""

        # 1. 分析模型大小
        original_size = self._estimate_model_size()

        # 2. 選擇壓縮策略
        if original_size > self.target_memory * 4:
            strategy = "int4_nf4"
        elif original_size > self.target_memory * 2:
            strategy = "int8"
        else:
            strategy = "fp16"

        print(f"選擇策略: {strategy} (原始大小: {original_size:.1f}GB)")

        # 3. 執行壓縮
        return self._execute_compression(strategy)

    def _execute_compression(self, strategy):
        """執行壓縮"""

        if strategy == "int4_nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif strategy == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        return model, strategy
```

## ⚡ 快速上手流程

### 5分鐘快速評估
```bash
# 1. 檢查模型大小
python quick_memory_check.py "your_model_name"

# 2. 選擇壓縮策略 (基於上述輸出)

# 3. 一鍵壓縮
python auto_compress.py "your_model_name" --target-memory 16

# 4. 驗證效果
python validate_compression.py --model compressed_model_path
```

### 30分鐘完整流程
1. **分析階段(5min)**: 運行模型分析腳本
2. **策略選擇(5min)**: 基於分析結果選擇策略
3. **執行壓縮(15min)**: 運行壓縮腳本
4. **效果驗證(5min)**: 驗證壓縮效果

## 🚨 常見問題解決

### 記憶體不足
```python
# 解決方案1: 降低精度
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)

# 解決方案2: 分片載入
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True
)
```

### 精度下降過大
```python
# 解決方案1: 混合精度
# 保留重要層為FP16，其他層INT8

# 解決方案2: 校準數據優化
# 使用更多高質量校準數據

# 解決方案3: QAT微調
# 量化後進行少量微調恢復性能
```

## 📊 效果評估檢查清單

- [ ] **功能正常**: 基本推理功能無異常
- [ ] **性能提升**: 推理速度提升1.5x+
- [ ] **記憶體節省**: 記憶體使用減少50%+
- [ ] **精度可接受**: 準確率下降<5%
- [ ] **穩定性良好**: 長時間運行無異常

## 🔗 相關資源

- **理論基礎**: `01-Theory/` 目錄下的5個專論
- **實踐教程**: `02-Labs/Lab-0.6-Model_Compression_Engineering/`
- **計算工具**: `02-Labs/Lab-0.5-Parameter_Calculator/`
- **評估工具**: `02-Labs/Lab-0.2-Evaluation_Benchmark/`

---

**⚡ 最重要的建議**: 先運行Lab 0.6獲得完整實踐經驗，再使用本快速指南進行生產環境應用！