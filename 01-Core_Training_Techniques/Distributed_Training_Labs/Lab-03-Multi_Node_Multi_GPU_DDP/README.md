# Lab 3: PyTorch 分散式訓練 - 多機多 GPU (DDP)

## 概述

本實驗是 Lab 1 的延伸，將分散式訓練從單機多 GPU 擴展到**多機多 GPU** 的場景。我們將深入探討如何配置和使用 `DistributedDataParallel (DDP)` 實現跨越多台伺服器的大規模模型訓練，這是工業界進行大型語言模型 (LLM) 訓練的標準實踐。

![多機多 GPU 架構](https://pic2.zhimg.com/v2-57124b89f899e1d848039c2772591694_1440w.jpg)

---

## 1. 技術背景與動機

### 1.1 單機訓練的極限

即使擁有多張 GPU，單台伺服器的資源仍然有限：
- **GPU 數量上限**：一台伺服器通常最多容納 8-16 張 GPU。
- **記憶體與 CPU 瓶頸**：單機的總記憶體、CPU 核心數和網路頻寬成為新的瓶頸。
- **無法擴展至超大規模**：訓練千億甚至兆級參數的模型，單機資源遠遠不足。

### 1.2 多機訓練的必要性

為了訓練最先進的大型模型，必須將計算能力擴展到多台機器：
1. **突破硬體限制**：聚合多台伺服器的 GPU 和記憶體資源。
2. **極致加速訓練**：通過增加節點數量，線性地提升數據處理能力，縮短訓練週期。
3. **實現超大規模模型訓練**：結合模型平行與數據平行，使訓練超大模型成為可能。

---

## 2. 多機 DDP 核心原理

### 2.1 架構演進

多機 DDP 的核心思想與單機 DDP 相同，但引入了**跨節點通信**的複雜性。

![多機 DDP 通信](https://pic3.zhimg.com/v2-9452097e33501705e468e7c1d1a660a1_1440w.jpg)

**關鍵流程**：
1. **進程組初始化**：所有參與訓練的進程（跨越多台機器）需要加入同一個通信組。
2. **節點間通信**：使用 `NCCL` 後端，通過高速網路（如 InfiniBand）在不同機器的 GPU 之間高效地同步梯度。
3. **全局梯度同步**：`Ring All-Reduce` 算法擴展到多機環境，所有 GPU（無論在哪台機器上）共同計算和同步平均梯度。
4. **獨立權重更新**：每個 GPU 獨立更新其模型副本的權重，保持全局一致。

### 2.2 關鍵環境變數

為了協調多個節點，DDP 依賴於一組關鍵的環境變數：
- `MASTER_ADDR`：主節點的 IP 地址。所有其他節點將連接到此地址以加入進程組。
- `MASTER_PORT`：主節點上用於通信的開放端口。
- `WORLD_SIZE`：參與訓練的**總進程數**（通常等於總 GPU 數量）。
- `RANK`：當前進程的全局唯一排名（從 0 到 `WORLD_SIZE - 1`）。

---

## 3. 實驗設計與實作

### 3.1 實驗環境

- **框架**: PyTorch
- **硬體**: 至少兩台帶有多張 GPU 的伺服器，並確保它們之間網路互通。
- **通信後端**: `NCCL` (NVIDIA Collective Communications Library)，專為 NVIDIA GPU 優化。

### 3.2 實驗流程

1. **環境配置** (`01-Setup.ipynb`)
   - 確保所有節點都安裝了相同版本的 PyTorch、CUDA 和 NCCL。
   - 設置防火牆規則，開放 `MASTER_PORT`。
   - 準備共享儲存（如 NFS），確保所有節點都能訪問到相同的程式碼和數據集。
2. **多機 DDP 腳本撰寫** (`02-Multi_Node_DDP.py`)
   - 編寫一個能夠接收 `RANK` 和 `WORLD_SIZE` 等參數的訓練腳本。
   - 在腳本中正確初始化 `torch.distributed.init_process_group`。
3. **啟動多機訓練**
   - 在每個節點上運行啟動命令，並正確設置環境變數。

### 3.3 啟動方式

#### 方法一：手動設置環境變數

**節點 0 (主節點, IP: 192.168.1.1)**
```bash
# 終端 1 (GPU 0)
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=12355
export WORLD_SIZE=4 # 假設共 2 台機器，每台 2 張 GPU
export RANK=0
python train.py

# 終端 2 (GPU 1)
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=12355
export WORLD_SIZE=4
export RANK=1
python train.py
```

**節點 1 (IP: 192.168.1.2)**
```bash
# 終端 1 (GPU 0)
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=12355
export WORLD_SIZE=4
export RANK=2
python train.py

# 終端 2 (GPU 1)
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=12355
export WORLD_SIZE=4
export RANK=3
python train.py
```

#### 方法二：使用 `torchrun` (推薦)

`torchrun` (以前的 `torch.distributed.launch`) 大大簡化了啟動過程。

**節點 0 (主節點)**
```bash
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12355 train.py
```

**節點 1**
```bash
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=12355 train.py
```
`torchrun` 會自動為每個進程設置 `WORLD_SIZE` 和 `RANK`。

### 3.4 程式碼關鍵修改

```python
import torch.distributed as dist
import os

def setup(rank, world_size):
    # 從環境變數初始化
    os.environ['MASTER_ADDR'] = 'localhost' # torchrun 會覆蓋這個
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化進程組
    # rank 和 world_size 由 torchrun 自動傳入或手動設置
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main(rank, world_size):
    # 1. 設置進程組
    setup(rank, world_size)
    
    # 2. 準備模型和數據
    model = MyModel().to(rank) # 將模型移動到對應的 GPU
    ddp_model = DDP(model, device_ids=[rank])
    
    # ... 訓練 ...
    
    # 3. 清理
    dist.destroy_process_group()
```

---

## 4. 結論與最佳實踐

- **網路是關鍵**：多機訓練的性能瓶頸通常在於節點間的網路頻寬和延遲。推薦使用高速網路如 InfiniBand。
- **使用 `torchrun`**：它極大地簡化了多機訓練的啟動和管理。
- **確保環境一致**：所有節點的軟體版本（PyTorch, CUDA）必須一致。
- **共享數據**：使用共享文件系統（如 NFS）來存儲數據集和程式碼，避免數據拷貝和不一致。
- **監控所有節點**：使用 `nvidia-smi` 等工具監控所有節點的 GPU 使用率，確保負載均衡。
- **模型保存**：只在 `rank == 0` 的進程中保存模型，避免多個進程同時寫入文件造成衝突。

掌握多機 DDP 是進行大規模深度學習研究和工程實踐的必備技能。
