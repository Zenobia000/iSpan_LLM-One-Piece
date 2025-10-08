# Lab-1.3: 使用 DeepSpeed 微調 Alpaca 模型
## Finetune Alpaca with DeepSpeed

---

## ⚠️ 環境限制聲明

本實驗室的**完整實驗**需要**多GPU環境**才能展示 DeepSpeed 的全部功能，目前專案開發環境為單GPU，因此**部分進階內容暫未開發**。

### 為何需要多GPU?
- **DeepSpeed ZeRO-3** 的核心優勢在於跨GPU的記憶體優化
- **多GPU性能對比**需要真實的分散式環境
- **3D平行化** (數據+張量+流水線) 需要多GPU協作

### 單GPU可用功能 ✅
- DeepSpeed ZeRO-2 (優化器狀態 + 梯度分片)
- 混合精度訓練
- 梯度累積與檢查點
- 基礎配置與調優

---

## 📚 實驗室概述

### 學習目標
通過本實驗室，您將學習如何使用 **DeepSpeed** 框架高效微調大型語言模型。即使在單GPU環境，您也能掌握 DeepSpeed 的核心概念與配置技巧。

### 技術棧
- **DeepSpeed**: 微軟開源的分散式訓練框架
- **ZeRO (Zero Redundancy Optimizer)**: 記憶體優化核心技術
- **Alpaca 數據集**: Stanford 的指令微調數據集
- **Llama-2-7B**: Meta 的開源基礎模型

---

## 🎯 DeepSpeed 核心技術

### ZeRO 優化器 (Zero Redundancy Optimizer)

DeepSpeed 的核心創新是 **ZeRO**，通過消除記憶體冗餘來實現大模型訓練。

#### ZeRO 三個階段對比

| Stage | 優化內容 | 記憶體節省 | 通訊開銷 | 單GPU可用 |
|-------|---------|-----------|---------|----------|
| **ZeRO-1** | 優化器狀態分片 | 4x | 1.5x | ✅ 是 |
| **ZeRO-2** | + 梯度分片 | 8x | 1.5x | ✅ 是 |
| **ZeRO-3** | + 參數分片 | N倍 (N=GPU數) | 1.5x | ⚠️ 多GPU更優 |

#### 記憶體占用分析
標準訓練 (FP16 + Adam):
```
總記憶體 = 模型參數 + 梯度 + 優化器狀態
         = 2P + 2P + 12P = 16P

其中 P = 參數量
```

ZeRO-2 優化後 (N個GPU):
```
每GPU記憶體 = 2P + (2P + 12P) / N
            ≈ 2P (當N較大時)
```

---

## 🔧 DeepSpeed 配置詳解

### 配置文件結構

DeepSpeed 使用 JSON 配置文件控制訓練行為:

#### 基礎配置 (ds_config_zero2.json)
```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 32,

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-5,
      "warmup_num_steps": 100,
      "total_num_steps": 1000
    }
  },

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },

  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },

  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
```

### 配置參數詳解

#### 1. 批次大小配置
```json
"train_batch_size": 32,              // 全局有效批次
"train_micro_batch_size_per_gpu": 1, // 每GPU每次處理
"gradient_accumulation_steps": 32    // 梯度累積次數
```
**關係**: `train_batch_size = micro_batch_size × accumulation × GPU數量`

#### 2. ZeRO Stage 2 配置
```json
"zero_optimization": {
  "stage": 2,                    // ZeRO-2: 優化器 + 梯度分片
  "offload_optimizer": {
    "device": "cpu",             // 將優化器狀態卸載到CPU
    "pin_memory": true           // 使用固定記憶體加速傳輸
  },
  "overlap_comm": true,          // 通訊與計算重疊
  "contiguous_gradients": true   // 梯度連續儲存
}
```

#### 3. 混合精度配置
```json
"fp16": {
  "enabled": true,               // 啟用FP16訓練
  "loss_scale": 0,               // 動態損失縮放 (0=自動)
  "loss_scale_window": 1000,     // 損失縮放調整窗口
  "min_loss_scale": 1            // 最小損失縮放值
}
```

---

## 📖 理論學習 (完整可用) ✅

### 相關理論文件
請參閱以下完整的理論教學:

1. **`01-Theory/1.2-Distributed_Training.md`** 第3章: DeepSpeed 框架
   - DeepSpeed 架構與設計哲學
   - ZeRO-1/2/3 詳細原理
   - 3D 平行化策略
   - 配置最佳實踐

2. **`01-Theory/1.3-Optimization_and_Alignment.md`** 相關章節
   - 混合精度訓練
   - 梯度累積與檢查點
   - 記憶體優化技術

### 核心代碼範例 (來自理論文件)

#### DeepSpeed 初始化
```python
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# DeepSpeed 初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config="ds_config_zero2.json"
)

# 訓練循環
for batch in dataloader:
    inputs = tokenizer(batch["text"], return_tensors="pt",
                      padding=True, truncation=True)
    inputs = {k: v.to(model_engine.device) for k, v in inputs.items()}

    outputs = model_engine(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    model_engine.backward(loss)
    model_engine.step()
```

#### 單GPU DeepSpeed 訓練
```bash
# 單GPU訓練 (使用 ZeRO-2)
deepspeed train.py --deepspeed ds_config_zero2.json

# 多GPU訓練 (使用 ZeRO-3)
deepspeed --num_gpus=4 train.py --deepspeed ds_config_zero3.json
```

---

## 🎓 學習路徑

### 階段 1: 理論理解 ✅ (單GPU可完成)
- [x] 閱讀 `1.2-Distributed_Training.md` 第3章
- [x] 理解 ZeRO 的三個階段原理
- [x] 掌握 DeepSpeed 配置文件結構
- [x] 理解記憶體優化策略

### 階段 2: 配置實踐 ✅ (單GPU可完成)
- [x] 分析不同 ZeRO Stage 配置文件
- [x] 理解批次大小與梯度累積關係
- [x] 學習混合精度配置參數
- [x] 掌握優化器卸載策略

### 階段 3: 單GPU實驗 🟡 (部分可完成)
- [x] 在單GPU上使用 DeepSpeed ZeRO-2
- [x] 對比有無 DeepSpeed 的記憶體占用
- [x] 測試梯度累積與批次大小配置
- [ ] ⚠️ 無法測試 ZeRO-3 的完整效果 (需多GPU)

### 階段 4: 多GPU實驗 ⏸️ (需多GPU環境)
- [ ] 對比 ZeRO-1/2/3 的性能差異
- [ ] 測試 3D 平行化配置
- [ ] 測量通訊開銷與擴展性
- [ ] 優化大規模訓練配置

---

## 🔬 單GPU環境實驗建議

雖然無法展示多GPU的完整功能，您仍可在單GPU上進行有價值的實驗:

### 實驗 1: DeepSpeed vs 標準訓練
**目標**: 對比 DeepSpeed ZeRO-2 與標準訓練的記憶體效率

```python
# 標準訓練
# 預期GPU記憶體占用: ~12-14GB (Llama-2-7B)

# DeepSpeed ZeRO-2 + CPU Offload
# 預期GPU記憶體占用: ~8-10GB (節省 20-30%)
```

### 實驗 2: 配置參數調優
**目標**: 測試不同配置對訓練的影響

測試參數:
- `train_micro_batch_size_per_gpu`: 1, 2, 4
- `gradient_accumulation_steps`: 8, 16, 32
- `offload_optimizer`: true/false
- `overlap_comm`: true/false

### 實驗 3: 記憶體分析
**目標**: 深入理解 DeepSpeed 的記憶體優化

```python
import torch

# 訓練前記憶體
torch.cuda.empty_cache()
mem_before = torch.cuda.memory_allocated()

# 訓練過程
model_engine.train()
for step, batch in enumerate(dataloader):
    # ... 訓練代碼
    if step % 10 == 0:
        mem_current = torch.cuda.memory_allocated()
        print(f"Step {step}: {mem_current / 1e9:.2f} GB")

# 訓練後記憶體
mem_after = torch.cuda.memory_allocated()
print(f"記憶體增長: {(mem_after - mem_before) / 1e9:.2f} GB")
```

---

## 📊 配置文件範例庫

### ZeRO-1 配置 (單GPU友好)
```json
{
  "zero_optimization": {
    "stage": 1,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

### ZeRO-2 配置 (推薦單GPU)
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### ZeRO-3 配置 (多GPU推薦)
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  }
}
```

---

## 🚀 多GPU環境執行指南

當您有多GPU環境時，可以執行完整的 DeepSpeed 訓練:

### 前置準備
```bash
# 安裝 DeepSpeed
pip install deepspeed

# 檢查安裝
ds_report

# 驗證多GPU可見性
python -c "import torch; print(torch.cuda.device_count())"
```

### 執行命令

#### 單節點多GPU
```bash
# 4 GPU訓練 (ZeRO-2)
deepspeed --num_gpus=4 train.py \
  --deepspeed ds_config_zero2.json \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset alpaca

# 4 GPU訓練 (ZeRO-3)
deepspeed --num_gpus=4 train.py \
  --deepspeed ds_config_zero3.json
```

#### 多節點多GPU
```bash
# Master 節點
deepspeed --num_nodes=2 --num_gpus=4 \
  --master_addr=192.168.1.1 --master_port=29500 \
  --node_rank=0 train.py --deepspeed ds_config_zero3.json

# Worker 節點
deepspeed --num_nodes=2 --num_gpus=4 \
  --master_addr=192.168.1.1 --master_port=29500 \
  --node_rank=1 train.py --deepspeed ds_config_zero3.json
```

---

## 📈 性能預期

### 記憶體節省 (Llama-2-7B 為例)

| 配置 | GPU記憶體 | 支援batch size | 相對節省 |
|------|----------|---------------|---------|
| 標準訓練 (FP32) | ~28GB | 1 | - |
| 標準訓練 (FP16) | ~14GB | 1 | 50% |
| DeepSpeed ZeRO-1 | ~12GB | 2 | 57% |
| DeepSpeed ZeRO-2 | ~10GB | 4 | 64% |
| DeepSpeed ZeRO-3 (4 GPU) | ~8GB/GPU | 8 | 71% |

### 訓練速度 (相對於單GPU)

| 配置 | 相對速度 | 備註 |
|------|---------|------|
| 1 GPU (標準) | 1.0x | 基準 |
| 1 GPU (DeepSpeed ZeRO-2) | 0.95x | 輕微開銷 |
| 4 GPU (DeepSpeed ZeRO-2) | 3.6x | 近線性擴展 |
| 4 GPU (DeepSpeed ZeRO-3) | 3.4x | 通訊開銷稍大 |

---

## 🛠️ 故障排除

### 常見問題

#### 1. 記憶體不足 (OOM)
```bash
# 解決方案:
# 1. 降低 micro_batch_size
# 2. 增加 gradient_accumulation_steps
# 3. 啟用 optimizer offload
# 4. 使用 ZeRO-3
```

#### 2. DeepSpeed 找不到配置文件
```bash
# 確保路徑正確
deepspeed train.py --deepspeed ./configs/ds_config.json
```

#### 3. NCCL 初始化失敗
```bash
# 設置環境變數
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand (如果不支援)
```

---

## 📚 延伸學習資源

### 官方資源
- [DeepSpeed 官方文檔](https://www.deepspeed.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)

### 理論學習
- **`01-Theory/1.2-Distributed_Training.md`** 第3章: DeepSpeed 完整講解
- **`01-Theory/1.3-Optimization_and_Alignment.md`**: 優化技術

### HuggingFace 整合
- [Transformers + DeepSpeed](https://huggingface.co/docs/transformers/main_classes/deepspeed)
- [Accelerate + DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)

---

## 🎯 未來實驗室計劃

當專案獲得多GPU環境後，將補充完整實驗內容:

### 計劃中的實驗內容
- [ ] **01-Setup.ipynb**: 環境驗證與 DeepSpeed 配置
- [ ] **02-Train_ZeRO2.ipynb**: ZeRO-2 單GPU/多GPU訓練
- [ ] **03-Train_ZeRO3.ipynb**: ZeRO-3 大規模訓練
- [ ] **04-Performance_Analysis.ipynb**: 性能對比與優化

### 實驗目標
- 展示 ZeRO-1/2/3 的記憶體節省效果
- 對比不同配置的訓練速度
- 演示 3D 平行化配置
- 提供生產級訓練範例

---

## 💡 給學習者的建議

### 單GPU學習者
1. ✅ **理論先行**: 徹底理解 ZeRO 的工作原理
2. ✅ **配置熟悉**: 研讀各種配置參數的含義
3. ✅ **單GPU實驗**: 在單GPU上測試 ZeRO-2 + CPU Offload
4. ✅ **記憶體分析**: 對比有無 DeepSpeed 的記憶體使用

### 多GPU學習者
1. ✅ **完整實驗**: 測試 ZeRO-1/2/3 的完整功能
2. ✅ **性能對比**: 測量不同配置的訓練速度與記憶體
3. ✅ **擴展性**: 測試從2GPU到8GPU的擴展效果
4. ✅ **優化調參**: 根據硬體調整配置參數

---

**狀態**: 🟡 部分可用 (單GPU: ZeRO-2, 多GPU: 完整功能)
**最後更新**: 2025-10-08
**相關文件**:
- 理論: `01-Theory/1.2-Distributed_Training.md` (第3章)
- 理論: `01-Theory/1.3-Optimization_and_Alignment.md`
- 實驗室: `Lab-1.2-PyTorch_DDP_Basics` (多GPU)
