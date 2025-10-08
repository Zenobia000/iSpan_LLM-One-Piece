# Lab-1.2: PyTorch DDP 分散式訓練基礎
## PyTorch Distributed Data Parallel (DDP) Basics

---

## ⚠️ 環境限制聲明

本實驗室需要**多GPU環境**才能完整執行，目前專案開發環境為單GPU，因此**實驗室內容暫未開發**。

### 為何需要多GPU?
- **DDP (Distributed Data Parallel)** 是 PyTorch 的多GPU分散式訓練框架
- 需要至少 **2個GPU** 才能驗證分散式訓練邏輯
- 單GPU環境無法展示真實的梯度同步、通訊優化等核心功能

---

## 📚 學習替代方案

雖然無法在單GPU環境執行實驗，您仍可通過以下方式學習 DDP 技術:

### 1. 理論學習 (完整可用) ✅
請參閱完整的理論文件:
- **`01-Theory/1.2-Distributed_Training.md`** (526行, 13.6KB)
  - 第2章: PyTorch DDP 完整講解
  - 涵蓋: 基本原理、通訊機制、最佳實踐
  - 包含: 代碼範例、配置說明、性能優化

### 2. 代碼範例 (理論文件中) ✅
理論文件包含完整的 DDP 代碼範例:

```python
# 完整的 DDP 訓練範例 (來自 1.2-Distributed_Training.md)
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size):
    setup(rank, world_size)

    # 創建模型並包裝為 DDP
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 訓練循環
    for data, labels in dataloader:
        data, labels = data.to(rank), labels.to(rank)

        optimizer.zero_grad()
        outputs = ddp_model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    dist.destroy_process_group()
```

### 3. 配置模板參考

#### 多GPU啟動腳本
```bash
# 單節點多GPU訓練
torchrun --nproc_per_node=4 train.py

# 多節點多GPU訓練
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=0 --master_addr="192.168.1.1" \
         --master_port=29500 train.py
```

#### DDP 配置最佳實踐
```python
# DDP 配置參數
ddp_config = {
    'backend': 'nccl',           # GPU 使用 nccl
    'init_method': 'env://',     # 環境變數初始化
    'world_size': 4,             # 總GPU數量
    'rank': 0,                   # 當前進程編號
    'find_unused_parameters': False,  # 性能優化
    'broadcast_buffers': True,   # 同步buffer
    'gradient_as_bucket_view': True,  # 梯度桶視圖優化
}
```

---

## 🎯 學習目標

即使無法執行實際的多GPU訓練，您仍應掌握以下核心概念:

### 基礎概念
- [x] 理解 DDP 的基本原理與工作流程
- [x] 掌握分散式訓練的初始化與進程管理
- [x] 了解梯度同步機制 (All-Reduce)

### 配置技能
- [x] 理解 DDP 的配置參數與含義
- [x] 掌握多GPU訓練的啟動方式
- [x] 了解通訊後端的選擇 (NCCL vs Gloo)

### 優化策略
- [x] 理解梯度累積在分散式訓練中的應用
- [x] 掌握混合精度訓練的分散式配置
- [x] 了解通訊優化的最佳實踐

---

## 🔧 如何在多GPU環境中執行

當您有多GPU環境時，可以按以下步驟執行 DDP 訓練:

### 前置準備
```bash
# 檢查GPU數量
nvidia-smi

# 驗證 PyTorch 分散式支援
python -c "import torch; print(torch.distributed.is_available())"
python -c "import torch; print(torch.distributed.is_nccl_available())"
```

### 執行步驟

#### 1. 準備訓練腳本 (參考理論文件)
創建 `train_ddp.py`, 實現:
- 進程初始化 (`dist.init_process_group`)
- 模型包裝 (`DDP(model)`)
- 數據集分片 (`DistributedSampler`)
- 訓練循環

#### 2. 啟動分散式訓練
```bash
# 單機4GPU訓練
torchrun --nproc_per_node=4 train_ddp.py

# 或使用 torch.distributed.launch (舊版)
python -m torch.distributed.launch --nproc_per_node=4 train_ddp.py
```

#### 3. 監控訓練
```bash
# 監控GPU使用
watch -n 0.5 nvidia-smi

# 查看訓練日誌
tail -f train.log
```

---

## 📖 延伸學習資源

### 官方文檔
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Getting Started with DDP](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [DDP API Reference](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

### 理論深入
- **理論文件**: `01-Theory/1.2-Distributed_Training.md`
  - 第2.1節: PyTorch DDP 完整講解
  - 第2.2節: 通訊優化與梯度同步
  - 第2.3節: 分散式訓練最佳實踐

### 實戰範例
- [PyTorch Examples - ImageNet](https://github.com/pytorch/examples/tree/main/imagenet)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate)

---

## 🚀 未來計劃

當專案獲得多GPU環境後，將補充完整的實驗內容:

### 計劃中的實驗內容
- [ ] **01-Setup.ipynb**: 環境驗證與進程初始化
- [ ] **02-Train.ipynb**: DDP 訓練完整流程
- [ ] **03-Optimization.ipynb**: 通訊優化與性能調優
- [ ] **04-Advanced.ipynb**: 梯度累積、混合精度、checkpointing

### 實驗目標
- 展示真實的多GPU訓練加速
- 對比單GPU vs 多GPU性能
- 演示通訊瓶頸與優化策略
- 提供完整的生產級配置範例

---

## 💡 給學習者的建議

1. **理論先行**: 先徹底理解 DDP 的工作原理
2. **代碼閱讀**: 仔細研讀理論文件中的代碼範例
3. **配置熟悉**: 掌握各種配置參數的含義
4. **雲端實踐**: 考慮使用雲端多GPU資源進行實驗 (AWS, GCP, Azure)
5. **耐心等待**: 專案未來將補充完整實驗內容

---

**狀態**: ⏸️ 待開發 (等待多GPU環境)
**最後更新**: 2025-10-08
**相關文件**:
- 理論: `01-Theory/1.2-Distributed_Training.md`
- 實驗室: `Lab-1.3-Finetune_Alpaca_with_DeepSpeed` (同樣待開發)
