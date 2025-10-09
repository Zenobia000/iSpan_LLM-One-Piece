# Lab 1: PyTorch 分散式訓練 - 單機多 GPU (DP/DDP)

## 概述

本實驗深入探討 PyTorch 在單機多 GPU 環境下的兩種核心分散式訓練技術：`DataParallel (DP)` 和 `DistributedDataParallel (DDP)`。通過本實驗，您將理解這兩種方法的原理、優劣勢，並掌握在實際項目中應用的工程實踐。

![DP vs DDP 架構](https://pic4.zhimg.com/v2-9452097e33501705e468e7c1d1a660a1_1440w.jpg)

---

## 1. 技術背景與動機

### 1.1 傳統單 GPU 訓練的瓶頸

隨著模型規模和數據集的急劇增長，單 GPU 的記憶體和計算能力已成為訓練的瓶頸：
- **記憶體不足 (Out of Memory)**：大型模型（如 LLMs）無法在單張 GPU 上完整載入。
- **訓練時間過長**：巨大的數據集使得單 GPU 訓練耗時數週甚至數月。
- **硬體利用率低**：無法充分利用伺服器上的多張 GPU 卡，造成資源浪費。

### 1.2 分散式訓練的解決方案

分散式訓練通過將計算任務分發到多個設備（GPUs）上，協同完成模型訓練，旨在解決上述挑戰：
1. **擴展模型規模**：通過模型平行 (Model Parallelism) 將模型切分到多張卡上。
2. **加速數據處理**：通過數據平行 (Data Parallelism) 讓每張卡處理一部分數據，加快訓練速度。

本實驗聚焦於最常用、最直接的**數據平行**技術。

---

## 2. `DataParallel (DP)` 核心原理

### 2.1 基本概念與架構

**`DataParallel`** 是 PyTorch 中實現數據平行最簡單的方式。其核心思想是：
1. **模型複製**：將主 GPU (通常是 `cuda:0`) 上的模型複製到所有其他 GPU。
2. **數據分發**：將一個批次 (batch) 的數據切分成多個微批次 (mini-batches)，分發給每個 GPU。
3. **並行計算**：每個 GPU 獨立進行前向傳播，計算損失。
4. **梯度聚合**：將所有 GPU 上的梯度統一收集到**主 GPU**。
5. **權重更新**：在主 GPU 上更新模型權重。
6. **權重廣播**：將更新後的權重廣播回所有 GPU。

![DataParallel 工作流程](https://pic3.zhimg.com/v2-b7e671231f28682a39223788a1077759_1440w.jpg)

### 2.2 技術限制與瓶頸

儘管 DP 實現簡單，但存在明顯的性能瓶頸：
- **主 GPU 負載不均**：所有梯度聚合和權重更新都在主 GPU 上完成，導致其負載遠高於其他 GPU。
- **GIL 限制**：受 Python 的全局解釋器鎖 (Global Interpreter Lock, GIL) 影響，多線程並行效率不高。
- **單進程多線程**：所有 GPU 都在同一個進程中，容錯性差，且不適用於多機訓練。

---

## 3. `DistributedDataParallel (DDP)` 核心原理

### 3.1 基本概念與架構

**`DistributedDataParallel`** 是 PyTorch 官方推薦的、更高效的數據平行方案。其核心思想是：
1. **多進程架構**：為每個 GPU 創建一個獨立的進程，避免了 GIL 的限制。
2. **模型獨立**：每個進程獨立擁有模型的副本。
3. **數據分片**：使用 `DistributedSampler` 確保每個進程處理數據集的一個不重複子集。
4. **梯度同步 (All-Reduce)**：在反向傳播過程中，各個 GPU 非同步地計算梯度，並通過 `Ring All-Reduce` 算法高效地在所有 GPU 之間同步梯度，使得每個 GPU 在計算結束後都擁有完整的平均梯度。
5. **獨立更新**：每個 GPU 獨立更新自己的模型權重，由於梯度同步，保證了權重的一致性。

![DistributedDataParallel 工作流程](https://pic1.zhimg.com/v2-53a5c2d7667f0237587822989c93f0b2_1440w.jpg)

### 3.2 `Ring All-Reduce` 算法

DDP 的高效性很大程度上歸功於 `Ring All-Reduce` 算法，它避免了 DP 的單點瓶頸：
- **環形通信**：GPU 形成一個環，梯度數據在環中分塊傳輸。
- **計算與通信重疊**：在反向傳播計算梯度的同時，就可以開始同步已計算好的梯度，提高了訓練效率。

---

## 4. DP vs DDP 深度對比

| 特性維度 | `DataParallel (DP)` | `DistributedDataParallel (DDP)` | 推薦選擇 |
|:---|:---|:---|:---|
| **實現方式** | 單進程、多線程 | **多進程** | DDP |
| **並行效率** | 受 GIL 限制，效率較低 | 無 GIL 限制，**效率高** | DDP |
| **GPU 負載** | 主 GPU 負載高，易成瓶頸 | **負載均衡** | DDP |
| **通信算法** | 梯度聚合到主 GPU (Parameter Server) | **Ring All-Reduce** | DDP |
| **擴展性** | 僅限單機多卡 | **支持單機和多機** | DDP |
| **使用簡潔性** | **極簡**，一行代碼包裹模型 | 稍複雜，需設置進程組 | DP (僅快速原型) |
| **推薦場景** | 快速原型驗證、教學 | **所有生產環境、嚴肅訓練** | DDP |

---

## 5. 實驗設計與實作

### 5.1 實驗環境

- **框架**: PyTorch
- **硬體**: 單機多 GPU (e.g., 2x NVIDIA RTX 3090)
- **核心技術**: `torch.nn.DataParallel`, `torch.nn.parallel.DistributedDataParallel`

### 5.2 實驗流程

1. **環境準備** (`01-Setup.ipynb`)
   - 驗證 PyTorch 與 CUDA 環境。
   - 準備數據集和基礎模型。
2. **`DataParallel` 實作** (`02-DataParallel.ipynb`)
   - 使用 `nn.DataParallel` 包裹模型。
   - 運行訓練並觀察 GPU 負載情況。
3. **`DistributedDataParallel` 實作** (`03-DistributedDataParallel.ipynb`)
   - 使用 `torch.distributed.init_process_group` 初始化進程組。
   - 使用 `DistributedSampler` 處理數據。
   - 使用 `DDP` 包裹模型。
   - 透過 `torch.multiprocessing.spawn` 或 `torchrun` 啟動多進程訓練。

### 5.3 關鍵實現邏輯

#### `DataParallel` 實現
```python
import torch.nn as nn

model = MyModel()
if torch.cuda.device_count() > 1:
  print(f"Using {torch.cuda.device_count()} GPUs!")
  model = nn.DataParallel(model)

model.to(device)
```

#### `DistributedDataParallel` 實現
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 1. 初始化進程組
dist.init_process_group(backend="nccl")

# 2. 創建模型並移至對應 GPU
local_rank = int(os.environ["LOCAL_RANK"])
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 3. 創建 DistributedSampler
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler)

# 4. 執行訓練...

# 5. 清理
dist.destroy_process_group()
```

---

## 6. 結論與最佳實踐

- **優先選擇 DDP**：在任何需要使用多於一張 GPU 的場景，都應優先考慮使用 `DistributedDataParallel`。
- **僅在原型階段使用 DP**：`DataParallel` 因其簡單性，僅適用於快速驗證想法，不應用於正式的訓練任務。
- **注意 Batch Size**：在使用 DDP 時，`DataLoader` 中的 `batch_size` 指的是**每個進程**的批次大小。總批次大小為 `batch_size * world_size`。
- **使用 `torchrun`**：這是 PyTorch 官方推薦的啟動 DDP 訓練的工具。

通過本實驗，您將能夠在自己的項目中高效地利用單機多 GPU 資源，顯著加速模型訓練過程。
