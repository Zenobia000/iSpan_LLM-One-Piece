# Lab-01: 單節點多 GPU 分散式訓練 - DataParallel vs DistributedDataParallel

## 🎯 實驗目標

本實驗將深入探索在單節點多 GPU 環境下的兩種主要數據並行訓練方法：
- **DataParallel (DP)**: PyTorch 的基礎多 GPU 訓練方法
- **DistributedDataParallel (DDP)**: 現代推薦的分散式訓練方法

通過實際實驗對比兩者的性能差異、記憶體使用情況和擴展性，理解為什麼 DDP 已成為業界標準。

## 📚 理論背景

### DataParallel (DP) 特點
- ✅ 使用簡單，只需一行代碼包裝模型
- ❌ 主 GPU 負載不均，存在通信瓶頸
- ❌ 受 Python GIL 限制，無法充分利用多核 CPU
- ❌ 難以擴展到多節點

### DistributedDataParallel (DDP) 特點
- ✅ 無主 GPU 概念，負載完全均衡
- ✅ 多進程架構，避免 GIL 限制
- ✅ 高效的 All-Reduce 通信，支援多種後端
- ✅ 原生支援多節點擴展
- ✅ 與自動混合精度 (AMP) 更好整合

## 🛠️ 環境準備

### 硬體需求
- **GPU**: 至少 2 張 GPU（建議 4 張以上以觀察明顯差異）
- **記憶體**: 每張 GPU 至少 8GB VRAM
- **網路**: 單節點內 GPUs 間高速互連（PCIe/NVLink）

### 軟體依賴
```bash
# 激活 Poetry 環境
cd 00-Course_Setup
source .venv/bin/activate

# 檢查必要套件
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

## 🔬 實驗流程

### 步驟 1: 環境檢查與基準測試
```bash
python 01_environment_check.py
python 02_baseline_single_gpu.py --model resnet50 --batch-size 64 --epochs 5
```

### 步驟 2: DataParallel 實驗
```bash
python 03_dataparallel_training.py --model resnet50 --batch-size 256 --epochs 5 --gpus 0,1,2,3
```

### 步驟 3: DistributedDataParallel 實驗
```bash
torchrun --nproc_per_node=4 04_ddp_training.py --model resnet50 --batch-size 64 --epochs 5
```

### 步驟 4: 性能對比分析
```bash
python 05_performance_comparison.py --results-dir ./results --generate-plots --output-report comparison_report.html
```

## 📊 預期實驗結果 (4 x RTX 4090)

| 訓練方法 | 總批次大小 | 訓練時間/Epoch | GPU 記憶體 | 吞吐量 | 擴展效率 |
|---------|------------|---------------|----------|--------|---------|
| 單 GPU | 64 | 120.5s | 3.2GB | 425 samples/s | - |
| DP (4 GPU) | 256 | 85.2s | 7.8GB/4.1GB | 1,205 samples/s | 71% |
| DDP (4 GPU) | 256 | 32.1s | 3.4GB | 1,598 samples/s | 94% |

## ⚡ DDP 最佳實踐

```python
# 優化的 DDP 初始化
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,    # 性能優化
    gradient_as_bucket_view=True,    # 減少記憶體複製
    bucket_cap_mb=25,               # 調整通信桶大小
    broadcast_buffers=False,        # 不同步 buffer
)

# DDP + 混合精度
from torch.cuda.amp import autocast, GradScaler
model = DDP(model, device_ids=[local_rank])
scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 🐛 常見問題解決

### NCCL 通信超時
```bash
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
```

### 端口衝突
```bash
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
```

### GPU 記憶體不足
```python
# 梯度累積技術
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🎯 學習檢核點

完成本實驗後，您應該能夠：
- [ ] 理解 DataParallel 和 DistributedDataParallel 的根本差異
- [ ] 獨立設置和運行多 GPU 分散式訓練
- [ ] 識別和解決常見的分散式訓練問題
- [ ] 針對特定硬體配置選擇合適的並行策略
- [ ] 優化分散式訓練的性能和資源使用

## 🔗 參考資料

- [PyTorch DistributedDataParallel 官方文檔](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch DDP 教學](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NVIDIA 深度學習性能指南](https://docs.nvidia.com/deeplearning/performance/index.html)

---

**恭喜！您已經掌握了單節點多 GPU 分散式訓練的核心技術。接下來可以進入 Lab-02 探索更複雜的流水線並行技術。**