# Lab-1.4: 訓練優化基礎技術
## Training Optimization Basics

**實驗室類型**: 單GPU優化技術
**難度等級**: ⭐⭐⭐ (中級)
**預估時間**: 4-6小時
**適用GPU**: 8GB+ VRAM

---

## 📚 實驗室概述

本實驗室專注於**單GPU環境下的訓練優化技術**，涵蓋業界最常用的三大優化策略：混合精度訓練、梯度累積和梯度檢查點。這些技術是訓練大型語言模型的基石，能夠顯著降低記憶體占用、提升訓練速度。

### 學習目標

完成本實驗室後，您將能夠：
- ✅ 掌握混合精度訓練 (FP16/BF16) 的原理與實作
- ✅ 理解梯度累積如何突破記憶體限制
- ✅ 應用梯度檢查點實現時間換空間優化
- ✅ 使用記憶體分析工具進行性能調優
- ✅ 在實際項目中選擇合適的優化策略組合

---

## 🎯 核心技術概覽

### 1. 混合精度訓練 (Mixed Precision Training)
**問題**: FP32 訓練記憶體占用大，速度慢
**解決方案**: 使用 FP16/BF16 進行計算，關鍵部分保持 FP32
**效果**: 記憶體減少 ~50%，速度提升 2-3x

#### 技術要點
- **自動混合精度 (AMP)**: PyTorch 提供的 `torch.cuda.amp`
- **動態損失縮放**: 防止 FP16 梯度下溢
- **BF16 vs FP16**: 不同精度格式的選擇

### 2. 梯度累積 (Gradient Accumulation)
**問題**: GPU記憶體不足以支持大批次訓練
**解決方案**: 分多次前向/反向傳播，累積梯度後更新
**效果**: 實現大批次訓練效果，無需額外記憶體

#### 技術要點
- **有效批次大小**: `batch_size × accumulation_steps`
- **梯度同步策略**: 何時清零梯度
- **學習率調整**: 大批次訓練的學習率縮放

### 3. 梯度檢查點 (Gradient Checkpointing)
**問題**: 反向傳播需要儲存所有中間激活值
**解決方案**: 只儲存部分檢查點，需要時重新計算
**效果**: 記憶體從 O(L) 降至 O(√L)，代價是 20-30% 計算時間

#### 技術要點
- **檢查點策略**: 選擇哪些層設置檢查點
- **記憶體 vs 速度**: 權衡取捨分析
- **PyTorch 實現**: `torch.utils.checkpoint`

---

## 📂 實驗室結構

```
Lab-1.4-Training_Optimization_Basics/
├── README.md                        # 本文檔
├── 01-Mixed_Precision.ipynb        # 混合精度訓練實驗
├── 02-Gradient_Accumulation.ipynb  # 梯度累積實驗
├── 03-Gradient_Checkpointing.ipynb # 梯度檢查點實驗
├── 04-Memory_Profiling.ipynb       # 記憶體分析工具
└── configs/                         # 配置文件目錄
    ├── fp32_config.yaml
    ├── fp16_config.yaml
    └── optimized_config.yaml
```

---

## 🔧 環境準備

### 前置要求
```bash
# 確認 PyTorch 版本 (需要 2.0+)
python -c "import torch; print(torch.__version__)"

# 確認 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"

# 確認 GPU 記憶體
nvidia-smi
```

### 安裝依賴
```bash
# 已在 Poetry 環境中包含
cd /path/to/iSpan_LLM-One-Piece/00-Course_Setup
source .venv/bin/activate

# 驗證關鍵套件
python -c "from torch.cuda.amp import autocast, GradScaler"
python -c "import transformers"
```

---

## 📊 實驗內容詳解

### Notebook 1: 混合精度訓練 (01-Mixed_Precision.ipynb)
**時間**: 60-90分鐘

#### 實驗目標
- 對比 FP32、FP16、BF16 三種精度的訓練效果
- 實作動態損失縮放防止梯度下溢
- 測量記憶體占用與訓練速度差異

#### 實驗內容
1. **基準測試**: FP32 標準訓練
   - 載入 GPT-2 small 模型 (124M 參數)
   - 訓練 1000 steps
   - 記錄: 記憶體峰值、訓練時間、最終 loss

2. **FP16 混合精度**
   - 使用 `torch.cuda.amp.autocast()`
   - 實現 `GradScaler` 動態損失縮放
   - 對比性能提升

3. **BF16 混合精度** (GPU 支持的情況下)
   - BF16 vs FP16 精度對比
   - 數值穩定性分析

#### 預期結果
| 精度 | 記憶體峰值 | 訓練時間 | 相對速度 |
|------|----------|---------|---------|
| FP32 | ~8GB | 100% | 1.0x |
| FP16 | ~4.5GB | 45% | 2.2x |
| BF16 | ~4.5GB | 48% | 2.1x |

---

### Notebook 2: 梯度累積 (02-Gradient_Accumulation.ipynb)
**時間**: 45-60分鐘

#### 實驗目標
- 理解梯度累積如何實現大批次訓練
- 對比不同累積步數的效果
- 掌握有效批次大小計算

#### 實驗內容
1. **小批次基準**
   - micro_batch_size = 2
   - 無梯度累積
   - 記錄訓練效果

2. **梯度累積實驗**
   - accumulation_steps = [4, 8, 16]
   - 對比等效批次大小: [8, 16, 32]
   - 分析收斂速度與穩定性

3. **記憶體分析**
   - 梯度累積對記憶體占用的影響
   - 最佳配置選擇

#### 關鍵代碼
```python
# 梯度累積實現
accumulation_steps = 8
model.zero_grad()

for i, batch in enumerate(dataloader):
    # 前向傳播
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps

    # 反向傳播 (累積梯度)
    loss.backward()

    # 每 accumulation_steps 更新一次
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        model.zero_grad()
```

---

### Notebook 3: 梯度檢查點 (03-Gradient_Checkpointing.ipynb)
**時間**: 60-75分鐘

#### 實驗目標
- 理解梯度檢查點的時間換空間策略
- 測量記憶體節省 vs 計算開銷
- 掌握 HuggingFace Transformers 的檢查點使用

#### 實驗內容
1. **標準訓練基準**
   - GPT-2 medium (355M 參數)
   - 記錄記憶體峰值與訓練時間

2. **啟用梯度檢查點**
   - 使用 `model.gradient_checkpointing_enable()`
   - 對比記憶體節省效果

3. **性能權衡分析**
   - 記憶體節省百分比
   - 訓練時間增加百分比
   - 最佳使用場景

#### 預期結果
| 配置 | 記憶體峰值 | 訓練時間 | 記憶體節省 |
|------|----------|---------|-----------|
| 無檢查點 | ~12GB | 100% | - |
| 檢查點 | ~7GB | 125% | 42% |

---

### Notebook 4: 記憶體分析工具 (04-Memory_Profiling.ipynb)
**時間**: 30-45分鐘

#### 實驗目標
- 掌握 PyTorch 記憶體分析工具
- 識別記憶體瓶頸
- 優化記憶體使用策略

#### 實驗內容
1. **基礎記憶體監控**
   ```python
   import torch

   # 當前記憶體使用
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
   ```

2. **詳細記憶體追蹤**
   ```python
   # 記憶體快照
   torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

   # 使用 memory_profiler
   from torch.profiler import profile, ProfilerActivity
   ```

3. **優化策略識別**
   - 找出記憶體峰值時刻
   - 分析各層記憶體占用
   - 提出優化建議

---

## 🚀 實驗流程建議

### 推薦學習路徑
```
Day 1:
├── 理論學習: 閱讀 1.3-Optimization_and_Alignment.md (1-2小時)
└── 實驗1: 01-Mixed_Precision.ipynb (1.5小時)

Day 2:
├── 實驗2: 02-Gradient_Accumulation.ipynb (1小時)
└── 實驗3: 03-Gradient_Checkpointing.ipynb (1.5小時)

Day 3:
├── 實驗4: 04-Memory_Profiling.ipynb (0.75小時)
└── 綜合實踐: 組合多種優化技術 (1小時)
```

### 實驗順序重要性
1. **先學混合精度** - 最基礎的優化，為其他技術打基礎
2. **再學梯度累積** - 在混合精度基礎上進一步優化
3. **最後學檢查點** - 理解更複雜的記憶體優化策略
4. **記憶體分析** - 工具性實驗，隨時可穿插

---

## 📈 性能對比總覽

### 單一技術效果

| 優化技術 | 記憶體節省 | 速度提升 | 實現難度 | 推薦指數 |
|---------|----------|---------|---------|---------|
| 混合精度 | ~50% | 2-3x | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 梯度累積 | 0% | -5% | ⭐ | ⭐⭐⭐⭐⭐ |
| 梯度檢查點 | 30-50% | -20~-30% | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 組合優化效果

**最佳實踐組合**: 混合精度 + 梯度累積 + 梯度檢查點

```python
# 完整優化配置範例
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2LMHeadModel

# 1. 載入模型並啟用梯度檢查點
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.gradient_checkpointing_enable()

# 2. 設置混合精度訓練
scaler = GradScaler()

# 3. 梯度累積配置
accumulation_steps = 8
micro_batch_size = 2
effective_batch_size = micro_batch_size * accumulation_steps  # = 16

# 訓練循環
model.zero_grad()
for i, batch in enumerate(dataloader):
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad()
```

**優化效果**:
- 記憶體節省: ~65-70%
- 速度影響: 整體持平或略慢 10-15%
- **關鍵優勢**: 可在小GPU上訓練大模型

---

## 💡 最佳實踐建議

### 何時使用混合精度?
✅ **推薦場景**:
- 所有 GPU 訓練 (Volta 架構以上)
- 需要加速訓練的場景
- 記憶體不足的情況

❌ **不推薦場景**:
- CPU 訓練
- 對數值精度極度敏感的任務
- GPU 不支持 FP16 (GTX 10系列以下)

### 何時使用梯度累積?
✅ **推薦場景**:
- GPU記憶體無法容納所需批次大小
- 需要大批次訓練以穩定訓練
- 分散式訓練前的原型開發

❌ **不推薦場景**:
- 記憶體充足且追求最快速度
- 小批次訓練效果足夠好的任務

### 何時使用梯度檢查點?
✅ **推薦場景**:
- 訓練大模型 (數百M到數B參數)
- 記憶體極度緊張
- 訓練速度不是主要瓶頸

❌ **不推薦場景**:
- 小模型訓練
- 對訓練速度要求極高
- 記憶體充足的情況

---

## 🛠️ 故障排除

### 常見問題

#### 1. RuntimeError: CUDA out of memory
**原因**: 記憶體不足
**解決方案**:
```python
# 1. 減小批次大小
batch_size = 1  # 從小開始嘗試

# 2. 啟用梯度檢查點
model.gradient_checkpointing_enable()

# 3. 清空緩存
torch.cuda.empty_cache()

# 4. 增加梯度累積
accumulation_steps = 16
```

#### 2. Loss 出現 NaN 或 Inf
**原因**: FP16 數值下溢或上溢
**解決方案**:
```python
# 1. 檢查損失縮放
scaler = GradScaler(init_scale=2**10)

# 2. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 使用 BF16 替代 FP16 (如果GPU支持)
with autocast(dtype=torch.bfloat16):
    loss = model(**batch).loss
```

#### 3. 訓練速度沒有提升
**原因**: 可能的瓶頸
**排查步驟**:
```python
# 1. 確認使用了 GPU
print(next(model.parameters()).device)

# 2. 確認 AMP 正確啟用
print(f"AMP enabled: {torch.is_autocast_enabled()}")

# 3. 使用 profiler 分析
from torch.profiler import profile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # 訓練代碼
    pass
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 📚 延伸學習資源

### 理論基礎
- **必讀**: `01-Theory/1.3-Optimization_and_Alignment.md`
  - 1.2.2 節: 混合精度訓練
  - 梯度累積與檢查點原理

### 官方文檔
- [PyTorch AMP 文檔](https://pytorch.org/docs/stable/amp.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [HuggingFace Performance](https://huggingface.co/docs/transformers/performance)

### 論文閱讀
- **Mixed Precision Training** (ICLR 2018)
- **Gradient Checkpointing** (Chen et al., 2016)
- **Training Large Neural Networks with Constant Memory** (Kumar et al., 2019)

---

## 🎓 學習檢查清單

完成本實驗室後，您應該能夠:

### 混合精度訓練
- [ ] 理解 FP32/FP16/BF16 的精度差異
- [ ] 使用 `torch.cuda.amp` 實現混合精度
- [ ] 配置動態損失縮放 `GradScaler`
- [ ] 分析混合精度的性能提升
- [ ] 處理數值穩定性問題

### 梯度累積
- [ ] 理解有效批次大小概念
- [ ] 實現梯度累積訓練循環
- [ ] 計算不同配置的記憶體占用
- [ ] 調整學習率以適配大批次

### 梯度檢查點
- [ ] 理解時間換空間的權衡
- [ ] 使用 HuggingFace 的梯度檢查點
- [ ] 分析記憶體節省 vs 速度損失
- [ ] 選擇合適的檢查點策略

### 記憶體優化
- [ ] 使用 PyTorch 記憶體分析工具
- [ ] 識別記憶體瓶頸
- [ ] 組合多種優化技術
- [ ] 為不同場景選擇最佳配置

---

## 🚀 下一步學習

完成本實驗室後，建議繼續學習:

1. **Lab-1.5: FlashAttention Deep Dive**
   - 注意力機制的記憶體優化
   - 長序列訓練技術

2. **Lab-1.6: Efficient Attention (MQA/GQA)**
   - KV Cache 優化
   - 推理加速技術

3. **PEFT Labs 應用**
   - 將優化技術應用於 LoRA 等 PEFT 方法
   - 訓練更大的模型

---

**實驗室狀態**: ✅ 準備完成
**最後更新**: 2025-10-08
**維護者**: LLM 教學專案團隊

**相關文件**:
- 理論: `01-Theory/1.3-Optimization_and_Alignment.md`
- 下一個實驗室: `Lab-1.5-FlashAttention`
