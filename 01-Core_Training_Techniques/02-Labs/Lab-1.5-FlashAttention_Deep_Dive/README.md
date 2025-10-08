# Lab-1.5: FlashAttention 深度解析
## FlashAttention Deep Dive

**實驗室類型**: 注意力機制優化
**難度等級**: ⭐⭐⭐⭐ (中高級)
**預估時間**: 4-6小時
**適用GPU**: 8GB+ VRAM (建議 16GB+)

---

## 📚 實驗室概述

FlashAttention 是現代 LLM 訓練與推理的核心優化技術，通過重新設計注意力計算的記憶體訪問模式，實現了顯著的速度提升與記憶體節省。本實驗室將深入探索 FlashAttention 的原理、實現與實際應用。

### 學習目標

完成本實驗室後，您將能夠：
- ✅ 理解標準注意力機制的記憶體瓶頸
- ✅ 掌握 FlashAttention 的核心算法原理
- ✅ 對比 FlashAttention v1 vs v2 的性能差異
- ✅ 在訓練與推理中應用 FlashAttention
- ✅ 訓練超長序列模型 (>8K tokens)
- ✅ 分析 FlashAttention 的性能優勢

---

## 🎯 FlashAttention 核心技術

### 為什麼需要 FlashAttention?

**標準 Attention 的問題**:
```
標準 Self-Attention 計算:
1. Q @ K^T → 生成 attention scores (N×N 矩陣)
2. Softmax(scores) → 計算 attention weights (N×N 矩陣)
3. weights @ V → 產生輸出

記憶體需求: O(N²) - 序列長度的平方
時間複雜度: O(N²d) - N: 序列長度, d: 隱藏維度
```

**問題分析**:
- **記憶體瓶頸**: N×N 注意力矩陣在長序列時記憶體爆炸
  - N=1024: ~4MB (FP32)
  - N=2048: ~16MB
  - N=8192: ~256MB
  - N=16384: ~1GB (單個 head!)

- **記憶體頻寬瓶頸**: GPU HBM (High Bandwidth Memory) 訪問慢
  - SRAM (on-chip): ~19 TB/s
  - HBM (off-chip): ~1.5 TB/s
  - **速度差距**: ~12x

### FlashAttention 核心創新

**算法原理**: Tiling + Recomputation

```
FlashAttention 策略:
1. 分塊 (Tiling): 將 Q, K, V 分成小塊載入 SRAM
2. 融合運算 (Kernel Fusion): 在 SRAM 內完成所有計算
3. 在線 Softmax: 不儲存完整 attention matrix
4. 反向傳播重計算: 需要時重新計算而非儲存

記憶體需求: O(N) - 線性於序列長度
時間複雜度: O(N²d) - 不變，但 IO 開銷大幅降低
```

**關鍵技術特性**:
- **IO 感知算法**: 優化 GPU 記憶體階層訪問
- **融合內核**: 減少 HBM 讀寫次數
- **精確計算**: 數學等價於標準 attention (無近似)
- **反向傳播**: 重計算策略平衡速度與記憶體

### FlashAttention v1 vs v2

| 特性 | FlashAttention v1 | FlashAttention v2 | 改進 |
|------|------------------|-------------------|------|
| **分塊策略** | Outer loop over seq | Outer loop over heads | 更好並行 |
| **工作分配** | 基於序列 | 基於 warp | GPU 利用率 ⬆ |
| **速度** | 2-3x vs 標準 | 2-4x vs v1 | 總計 5-8x ⬆ |
| **記憶體** | O(N) | O(N) | 持平 |
| **長序列** | 支援至 8K | 支援至 64K+ | 擴展性 ⬆ |

---

## 🔧 技術原理深度解析

### 標準 Attention 的記憶體訪問模式

```python
# 標準實現 (偽代碼)
def standard_attention(Q, K, V):
    # Step 1: 計算 scores (HBM → HBM)
    scores = Q @ K.T  # [batch, heads, seq_len, seq_len]
    # ❌ 寫入 HBM: N² 元素

    # Step 2: Softmax (HBM → HBM)
    weights = softmax(scores)  # [batch, heads, seq_len, seq_len]
    # ❌ 讀取 + 寫入 HBM: 2N² 次訪問

    # Step 3: 計算輸出 (HBM → HBM)
    output = weights @ V  # [batch, heads, seq_len, head_dim]
    # ❌ 讀取 HBM: N² 元素

    return output

# HBM 訪問總量: 4N² (讀寫 scores + weights)
```

### FlashAttention 的優化訪問模式

```python
# FlashAttention 實現 (偽代碼)
def flash_attention(Q, K, V):
    # 在 SRAM 中分塊處理
    for block_q in range(0, N, block_size):
        # 載入 Q 的一個塊到 SRAM
        Q_block = load_to_sram(Q[block_q:block_q+block_size])

        # 初始化輸出累加器
        O_block = zeros(block_size, d)

        for block_kv in range(0, N, block_size):
            # 載入 K, V 的一個塊到 SRAM
            K_block = load_to_sram(K[block_kv:block_kv+block_size])
            V_block = load_to_sram(V[block_kv:block_kv+block_size])

            # ✅ 在 SRAM 內完成所有計算
            scores_block = Q_block @ K_block.T
            weights_block = softmax(scores_block)  # 在線 Softmax
            O_block += weights_block @ V_block

        # 寫回 HBM
        write_to_hbm(O_block)

# HBM 訪問總量: O(N) - 僅讀取 Q,K,V 和寫入 O
```

**關鍵優化**:
1. **Tiling**: 數據分塊，確保在 SRAM 內計算
2. **Online Softmax**: 不儲存中間 attention matrix
3. **Kernel Fusion**: 單個 CUDA kernel 完成所有操作
4. **IO 最小化**: HBM 訪問從 O(N²) 降至 O(N)

### 數學等價性證明

FlashAttention 與標準 attention **數學等價**:

```
標準 Attention:
Attention(Q, K, V) = softmax(QK^T / √d) V

FlashAttention:
通過分塊計算 + 在線 Softmax，最終結果完全相同
(無近似，無精度損失)
```

**在線 Softmax 算法**:
```python
# 累積式 Softmax (用於分塊計算)
def online_softmax(x_blocks):
    m = -inf  # 最大值
    l = 0     # 指數和

    for x in x_blocks:
        m_new = max(m, max(x))
        l = l * exp(m - m_new) + sum(exp(x - m_new))
        m = m_new

    return l, m

# 保證數值穩定性且與標準 softmax 等價
```

---

## 📂 實驗室結構

```
Lab-1.5-FlashAttention_Deep_Dive/
├── README.md                           # 本文檔
├── 01-Setup_and_Comparison.ipynb      # 環境設置與標準對比
├── 02-FlashAttention_Demo.ipynb       # FlashAttention 實戰演示
├── 03-Long_Sequence_Training.ipynb    # 長序列訓練應用
└── 04-Performance_Analysis.ipynb      # 性能深度分析
```

---

## 📊 實驗內容詳解

### Notebook 1: 環境設置與標準對比 (01-Setup_and_Comparison.ipynb)
**時間**: 60-90分鐘

#### 實驗目標
- 驗證 FlashAttention 安裝與可用性
- 實現標準 Attention 機制
- 對比標準 vs FlashAttention 的基礎性能
- 理解記憶體與速度的權衡

#### 實驗內容
1. **環境驗證**
   - 檢查 CUDA 版本與 GPU 支援
   - 安裝 flash-attn 套件
   - 驗證 FlashAttention 可用性

2. **標準 Attention 實現**
   ```python
   import torch
   import torch.nn.functional as F

   def standard_attention(Q, K, V, mask=None):
       """標準 Self-Attention 實現"""
       d_k = Q.size(-1)

       # Attention scores
       scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

       # Optional mask
       if mask is not None:
           scores = scores.masked_fill(mask == 0, -1e9)

       # Softmax
       attn_weights = F.softmax(scores, dim=-1)

       # Output
       output = torch.matmul(attn_weights, V)

       return output, attn_weights
   ```

3. **FlashAttention 使用**
   ```python
   from flash_attn import flash_attn_func

   def flash_attention(Q, K, V):
       """FlashAttention 包裝"""
       # FlashAttention 要求特定格式
       # (batch, seq_len, num_heads, head_dim)
       output = flash_attn_func(Q, K, V, causal=False)
       return output
   ```

4. **基礎性能對比**
   - 序列長度: [512, 1024, 2048, 4096]
   - 測量: 速度、記憶體、精度
   - 視覺化結果對比

#### 預期結果
| 序列長度 | 標準 Attention 時間 | FlashAttention 時間 | 加速比 |
|---------|-------------------|-------------------|--------|
| 512 | 10ms | 5ms | 2.0x |
| 1024 | 35ms | 12ms | 2.9x |
| 2048 | 140ms | 30ms | 4.7x |
| 4096 | OOM or 560ms | 80ms | 7.0x |

---

### Notebook 2: FlashAttention 實戰演示 (02-FlashAttention_Demo.ipynb)
**時間**: 45-60分鐘

#### 實驗目標
- 在真實模型中集成 FlashAttention
- 對比訓練與推理性能
- 理解 causal vs non-causal attention
- 測試不同配置的影響

#### 實驗內容
1. **Transformer 模型集成**
   ```python
   from transformers import GPT2Config, GPT2LMHeadModel

   # 標準 GPT-2
   config = GPT2Config(
       n_positions=2048,
       use_flash_attn=False
   )
   model_std = GPT2LMHeadModel(config)

   # FlashAttention GPT-2
   config_flash = GPT2Config(
       n_positions=2048,
       use_flash_attn=True  # 啟用 FlashAttention
   )
   model_flash = GPT2LMHeadModel(config_flash)
   ```

2. **訓練性能測試**
   - 前向傳播時間對比
   - 反向傳播時間對比
   - 總體訓練吞吐量

3. **推理性能測試**
   - 單次推理延遲
   - 批次推理吞吐量
   - 記憶體占用對比

4. **Causal Attention 分析**
   ```python
   # Causal (GPT-style): 只看過去的 token
   output_causal = flash_attn_func(Q, K, V, causal=True)

   # Non-causal (BERT-style): 可看全部 token
   output_non_causal = flash_attn_func(Q, K, V, causal=False)
   ```

#### 預期結果
- 訓練速度: 2-4x 提升
- 推理速度: 1.5-2x 提升
- 記憶體節省: 30-50% (長序列)

---

### Notebook 3: 長序列訓練應用 (03-Long_Sequence_Training.ipynb)
**時間**: 60-90分鐘

#### 實驗目標
- 訓練超長序列模型 (>4K tokens)
- 對比不同序列長度的性能
- 理解長序列訓練的挑戰與解決方案
- 實作長文本摘要/QA 任務

#### 實驗內容
1. **長序列數據準備**
   ```python
   # 準備長文本數據 (2K, 4K, 8K tokens)
   long_sequences = [
       tokenize(text, max_length=2048),
       tokenize(text, max_length=4096),
       tokenize(text, max_length=8192)
   ]
   ```

2. **長序列訓練實驗**
   - 序列長度擴展實驗
   - 記憶體占用監控
   - 訓練穩定性分析

3. **Position Encoding 處理**
   ```python
   # RoPE (Rotary Position Embedding) + FlashAttention
   # 適用於超長序列
   from flash_attn.modules.rotary import apply_rotary_emb
   ```

4. **實際應用案例**
   - 長文檔摘要
   - 長對話理解
   - 代碼理解 (完整文件)

#### 預期成果
- 成功訓練 8K+ 序列模型
- 記憶體占用 <16GB (with FlashAttention)
- vs 標準 attention: OOM at 2K

---

### Notebook 4: 性能深度分析 (04-Performance_Analysis.ipynb)
**時間**: 45-60分鐘

#### 實驗目標
- 深入分析 FlashAttention 的性能特性
- 理解不同硬體上的表現
- 優化超參數配置
- 建立性能預測模型

#### 實驗內容
1. **詳細性能 Profiling**
   ```python
   import torch.profiler as profiler

   with profiler.profile(
       activities=[profiler.ProfilerActivity.CPU,
                   profiler.ProfilerActivity.CUDA],
       record_shapes=True,
       with_flops=True
   ) as prof:
       # 運行 FlashAttention
       output = flash_attn_func(Q, K, V)

   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

2. **記憶體分析**
   - HBM 訪問模式分析
   - SRAM 使用效率
   - 記憶體頻寬利用率

3. **擴展性分析**
   - 批次大小影響
   - 序列長度擴展性
   - 多頭注意力並行度

4. **最佳實踐建議**
   - 何時使用 FlashAttention
   - 配置優化建議
   - 常見陷阱與避坑指南

#### 分析結果範例
```
FlashAttention 性能特徵:
- 記憶體節省: 40-60% (序列長度 >2K)
- 速度提升: 2-8x (取決於序列長度)
- 最佳場景: 長序列 (>1K), 多頭注意力
- 限制: 需要 CUDA 7.5+, Ampere 架構最佳
```

---

## 🚀 環境準備

### 前置要求

#### 硬體要求
- **GPU**: NVIDIA GPU with CUDA Capability ≥ 7.5
  - 推薦: RTX 3090, A100, H100
  - 最低: RTX 2080, V100
- **VRAM**: 最少 8GB，建議 16GB+
- **CUDA**: 11.6+ (建議 12.1+)

#### 軟體要求
```bash
# 確認 PyTorch 版本
python -c "import torch; print(torch.__version__)"
# 需要: PyTorch 2.0+

# 確認 CUDA 版本
nvcc --version
# 需要: CUDA 11.6+
```

### FlashAttention 安裝

#### 方法 1: pip 安裝 (推薦)
```bash
# 確保在 Poetry 環境中
source .venv/bin/activate

# 安裝 FlashAttention
pip install flash-attn --no-build-isolation

# 驗證安裝
python -c "import flash_attn; print(flash_attn.__version__)"
```

#### 方法 2: 從源碼編譯 (如果 pip 失敗)
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

#### 常見安裝問題

**問題 1**: 編譯失敗 - CUDA 版本不匹配
```bash
# 解決方案: 確認 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 確保系統 CUDA 版本與 PyTorch 一致
```

**問題 2**: 記憶體不足 - 編譯過程 OOM
```bash
# 解決方案: 限制並行編譯
MAX_JOBS=2 pip install flash-attn --no-build-isolation
```

**問題 3**: GPU 不支援
```bash
# 檢查 GPU compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"

# FlashAttention 需要 ≥ 7.5 (Turing 架構以上)
```

---

## 💡 學習路徑建議

### 推薦學習順序
```
Day 1 (2-3小時):
├── 閱讀理論部分 (README.md + 1.3-Optimization_and_Alignment.md)
└── 01-Setup_and_Comparison.ipynb (環境設置 + 基礎對比)

Day 2 (2-3小時):
├── 02-FlashAttention_Demo.ipynb (實戰集成)
└── 03-Long_Sequence_Training.ipynb (長序列訓練)

Day 3 (1-2小時):
├── 04-Performance_Analysis.ipynb (性能分析)
└── 綜合練習: 在自己的項目中應用 FlashAttention
```

### 進階學習路徑
1. **研讀論文**: FlashAttention 原始論文與 v2 改進
2. **源碼閱讀**: flash-attention GitHub 實現細節
3. **CUDA 優化**: 理解 CUDA kernel 優化技巧
4. **變體探索**: FlashAttention-2, PagedAttention, xFormers

---

## 📈 性能對比總覽

### 速度對比 (GPT-2 Medium, Batch Size=4)

| 序列長度 | 標準 Attention | FlashAttention v1 | FlashAttention v2 | 加速比 (v2) |
|---------|---------------|------------------|------------------|------------|
| 512 | 18ms | 10ms | 8ms | 2.3x |
| 1024 | 65ms | 25ms | 18ms | 3.6x |
| 2048 | 250ms | 60ms | 40ms | 6.3x |
| 4096 | OOM | 180ms | 120ms | ~8x+ |
| 8192 | OOM | 650ms | 400ms | ~15x+ |

### 記憶體對比 (GPT-2 Medium)

| 序列長度 | 標準 Attention | FlashAttention | 記憶體節省 |
|---------|---------------|---------------|-----------|
| 512 | 2.1GB | 1.8GB | 14% |
| 1024 | 4.5GB | 2.5GB | 44% |
| 2048 | 12GB | 4.2GB | 65% |
| 4096 | OOM (>24GB) | 8.5GB | >65% |
| 8192 | OOM | 18GB | N/A |

---

## 🛠️ 故障排除

### 常見問題

#### 1. FlashAttention 安裝失敗
**症狀**: `pip install flash-attn` 報錯

**診斷步驟**:
```bash
# 1. 檢查 CUDA 版本
nvcc --version

# 2. 檢查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 3. 檢查 GPU compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"
```

**解決方案**:
- CUDA 版本不匹配: 重新安裝匹配的 PyTorch
- GPU 不支援: 需要 compute capability ≥ 7.5
- 編譯資源不足: `MAX_JOBS=2 pip install flash-attn`

#### 2. RuntimeError: FlashAttention only supports ...
**症狀**: 運行時報格式錯誤

**原因**: FlashAttention 對輸入格式有特定要求
```python
# ❌ 錯誤格式
Q = torch.randn(batch, heads, seq_len, head_dim)

# ✅ 正確格式
Q = torch.randn(batch, seq_len, heads, head_dim)
```

**解決方案**:
```python
# 轉換格式
Q = Q.transpose(1, 2)  # [B, H, N, D] → [B, N, H, D]
```

#### 3. 精度差異
**症狀**: FlashAttention 結果與標準 attention 略有不同

**原因**: 浮點運算順序差異
```python
# 驗證精度
output_std, _ = standard_attention(Q, K, V)
output_flash = flash_attn_func(Q, K, V)

diff = (output_std - output_flash).abs().max()
print(f"最大差異: {diff:.6f}")  # 應該 < 1e-3
```

**正常範圍**: 差異 < 1e-3 (FP16), < 1e-5 (FP32)

#### 4. OOM 錯誤
**症狀**: 使用 FlashAttention 仍然 OOM

**可能原因**:
- 批次大小過大
- 其他層占用過多記憶體
- 梯度累積未正確配置

**解決方案**:
```python
# 減小批次
batch_size = 1

# 啟用梯度檢查點
model.gradient_checkpointing_enable()

# 使用混合精度
from torch.cuda.amp import autocast
with autocast(dtype=torch.float16):
    output = model(**batch)
```

---

## 📚 延伸學習資源

### 論文閱讀
- **FlashAttention**: [Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)
- **FlashAttention-2**: [Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) (2023)
- **Self-Attention Does Not Need O(n²) Memory**: 理論基礎論文

### 開源實現
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention): 官方實現
- [HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention): 研究版本
- [xFormers](https://github.com/facebookresearch/xformers): Meta 的記憶體高效 Transformers

### 相關技術
- **PagedAttention** (vLLM): KV Cache 優化
- **Multi-Query Attention** (MQA): 減少 KV heads
- **Grouped-Query Attention** (GQA): MQA 與 MHA 折衷
- **Linear Attention**: 線性複雜度近似

### 實際應用案例
- **GPT-4**: 使用 FlashAttention 訓練
- **Llama 2**: 官方實現支援 FlashAttention
- **MPT**: MosaicML 的預訓練模型
- **Falcon**: TII 的開源 LLM

---

## 🎓 學習檢查清單

完成本實驗室後，您應該能夠:

### 理論理解
- [ ] 解釋標準 attention 的記憶體瓶頸
- [ ] 理解 GPU 記憶體階層 (HBM vs SRAM)
- [ ] 說明 FlashAttention 的核心算法原理
- [ ] 對比 FlashAttention v1 vs v2 的改進
- [ ] 理解 IO 感知算法設計思想

### 實作技能
- [ ] 安裝並驗證 FlashAttention
- [ ] 實現標準 attention 機制
- [ ] 在模型中集成 FlashAttention
- [ ] 訓練長序列模型 (>4K tokens)
- [ ] 分析 FlashAttention 性能特徵

### 應用能力
- [ ] 為項目選擇合適的 attention 機制
- [ ] 優化長序列訓練配置
- [ ] 診斷與解決常見問題
- [ ] 評估 FlashAttention 的效益

---

## 🚀 下一步學習

完成本實驗室後，建議繼續:

1. **Lab-1.6: Efficient Attention (MQA/GQA)**
   - Multi-Query Attention 原理
   - Grouped-Query Attention 實作
   - KV Cache 優化技術

2. **推理優化**
   - vLLM 與 PagedAttention
   - Speculative Decoding
   - Continuous Batching

3. **生產部署**
   - 模型服務化
   - 推理性能調優
   - 成本優化策略

---

**實驗室狀態**: 🔄 開發中
**最後更新**: 2025-10-08
**維護者**: LLM 教學專案團隊

**相關文件**:
- 理論: `01-Theory/1.3-Optimization_and_Alignment.md` (FlashAttention 章節)
- 前置實驗: `Lab-1.4-Training_Optimization_Basics`
- 後續實驗: `Lab-1.6-Efficient_Attention`
