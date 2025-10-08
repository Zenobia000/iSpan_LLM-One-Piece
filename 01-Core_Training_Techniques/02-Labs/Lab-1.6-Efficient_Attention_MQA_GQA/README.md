# Lab-1.6: 高效注意力機制 (MQA/GQA)
## Efficient Attention - Multi-Query and Grouped-Query Attention

**實驗室類型**: 注意力架構優化
**難度等級**: ⭐⭐⭐⭐ (中高級)
**預估時間**: 3-5小時
**適用GPU**: 8GB+ VRAM

---

## 📚 實驗室概述

Multi-Query Attention (MQA) 和 Grouped-Query Attention (GQA) 是針對 LLM 推理優化的重要技術，通過減少 KV Cache 大小來顯著提升推理速度與降低記憶體占用。本實驗室將深入探索這些技術的原理、實現與實際應用。

### 學習目標

完成本實驗室後，您將能夠：
- ✅ 理解標準 MHA (Multi-Head Attention) 的推理瓶頸
- ✅ 掌握 MQA 的核心原理與實現
- ✅ 理解 GQA 如何平衡 MHA 與 MQA
- ✅ 實現並對比三種注意力機制
- ✅ 優化 LLM 推理性能
- ✅ 分析 KV Cache 的記憶體占用

---

## 🎯 核心技術概覽

### 為什麼需要 MQA/GQA?

**推理瓶頸分析**:
```
LLM 自回歸推理過程:
1. Prefill 階段: 處理整個 prompt (並行)
2. Decode 階段: 逐個生成 token (串行)

記憶體占用:
- 模型參數: 固定
- KV Cache: 隨生成長度線性增長
- Decode 階段: 記憶體頻寬受限 (Memory-bound)

問題: KV Cache 成為推理瓶頸!
```

**KV Cache 記憶體計算**:
```
標準 MHA (Multi-Head Attention):
  每層 KV Cache = 2 × seq_len × num_heads × head_dim × bytes

  例: Llama-2-7B (32層, 32heads, 128dim, FP16)
  單個序列 2K tokens:
  = 2 × 2048 × 32 × 128 × 2 bytes
  = 33.6 MB/層
  = 1.07 GB (32層)

  批次推理 (batch=16):
  = 1.07 GB × 16 = 17 GB (僅 KV Cache!)
```

### 三種注意力機制對比

#### 1. Multi-Head Attention (MHA) - 標準方法

```
MHA 結構:
  Q: [batch, seq_len, num_heads, head_dim]  # 32 heads
  K: [batch, seq_len, num_heads, head_dim]  # 32 heads
  V: [batch, seq_len, num_heads, head_dim]  # 32 heads

特點:
  ✅ 每個 head 有獨立的 K, V
  ✅ 表達能力最強
  ❌ KV Cache 最大
  ❌ 推理速度較慢
```

#### 2. Multi-Query Attention (MQA) - 激進優化

```
MQA 結構:
  Q: [batch, seq_len, num_heads, head_dim]  # 32 heads
  K: [batch, seq_len, 1, head_dim]          # 1 head (共享)
  V: [batch, seq_len, 1, head_dim]          # 1 head (共享)

特點:
  ✅ 所有 Query heads 共享同一組 K, V
  ✅ KV Cache 減少 32x
  ✅ 推理速度提升 1.5-2x
  ❌ 表達能力略有下降
  ❌ 訓練質量可能受影響
```

#### 3. Grouped-Query Attention (GQA) - 平衡方案

```
GQA 結構:
  Q: [batch, seq_len, num_heads, head_dim]  # 32 heads
  K: [batch, seq_len, num_groups, head_dim] # 4-8 groups
  V: [batch, seq_len, num_groups, head_dim] # 4-8 groups

特點:
  ✅ Query heads 分組共享 K, V
  ✅ KV Cache 減少 4-8x
  ✅ 推理速度提升 1.3-1.5x
  ✅ 表達能力接近 MHA
  ✅ Llama-2, Mistral 等模型採用
```

### 架構對比圖

| 架構 | Query Heads | KV Heads | KV Cache 大小 | 推理速度 | 模型質量 | 代表模型 |
|------|------------|----------|--------------|---------|---------|----------|
| **MHA** | 32 | 32 | 100% (基準) | 1.0x | ⭐⭐⭐⭐⭐ | GPT-3, GPT-4 |
| **GQA** | 32 | 4-8 | 12-25% | 1.3-1.5x | ⭐⭐⭐⭐⭐ | Llama-2, Mistral |
| **MQA** | 32 | 1 | 3% | 1.5-2x | ⭐⭐⭐⭐ | PaLM, Falcon |

---

## 🔧 技術原理深度解析

### MHA (標準) 實現

```python
def multi_head_attention(Q, K, V):
    """
    Q: [B, N, H, D]  - H 個獨立的 Q heads
    K: [B, N, H, D]  - H 個獨立的 K heads
    V: [B, N, H, D]  - H 個獨立的 V heads
    """
    # 每個 head 獨立計算
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(D)
    # scores: [B, H, N, N]

    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    # output: [B, N, H, D]

    return output
```

### MQA 實現

```python
def multi_query_attention(Q, K, V):
    """
    Q: [B, N, H, D]  - H 個獨立的 Q heads
    K: [B, N, 1, D]  - 1 個共享的 K head (廣播)
    V: [B, N, 1, D]  - 1 個共享的 V head (廣播)
    """
    # K, V 會自動廣播到所有 Q heads
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(D)
    # K 自動廣播: [B, N, 1, D] → [B, N, H, D]

    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    # V 自動廣播: [B, N, 1, D] → [B, N, H, D]

    return output

# KV Cache 節省: H → 1 (32x 減少!)
```

### GQA 實現

```python
def grouped_query_attention(Q, K, V, num_groups=4):
    """
    Q: [B, N, H, D]      - H=32 個獨立的 Q heads
    K: [B, N, G, D]      - G=4 個 KV groups
    V: [B, N, G, D]      - G=4 個 KV groups

    分組策略: 每 (H/G) 個 Q heads 共享一組 K, V
    例: 32 Q heads, 4 KV groups → 每 8 個 Q 共享 1 組 KV
    """
    B, N, H, D = Q.shape
    G = K.size(2)

    # 重塑 Q 為 [B, N, G, H/G, D]
    Q = Q.view(B, N, G, H // G, D)

    # K, V 擴展維度: [B, N, G, 1, D]
    K = K.unsqueeze(3)
    V = V.unsqueeze(3)

    # 計算 attention (每組內獨立)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(D)
    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    # 重塑回 [B, N, H, D]
    output = output.view(B, N, H, D)

    return output

# KV Cache 節省: H → G (32/4 = 8x 減少!)
```

---

## 📂 實驗室結構

```
Lab-1.6-Efficient_Attention_MQA_GQA/
├── README.md                          # 本文檔
├── 01-Setup.ipynb                    # 標準 MHA 基準測試
├── 02-MQA_Implementation.ipynb       # Multi-Query Attention 實現
├── 03-GQA_Implementation.ipynb       # Grouped-Query Attention 實現
└── 04-Inference_Optimization.ipynb   # 推理加速與 KV Cache 優化
```

---

## 📊 實驗內容詳解

### Notebook 1: 標準 MHA 基準測試 (01-Setup.ipynb)
**時間**: 30-45分鐘

#### 實驗目標
- 實現標準 Multi-Head Attention
- 測量訓練與推理性能基準
- 分析 KV Cache 記憶體占用
- 理解推理過程的記憶體瓶頸

#### 實驗內容
1. **MHA 完整實現**
   ```python
   class MultiHeadAttention(nn.Module):
       def __init__(self, hidden_dim, num_heads):
           self.num_heads = num_heads
           self.head_dim = hidden_dim // num_heads

           self.q_proj = nn.Linear(hidden_dim, hidden_dim)
           self.k_proj = nn.Linear(hidden_dim, hidden_dim)  # num_heads 組 K
           self.v_proj = nn.Linear(hidden_dim, hidden_dim)  # num_heads 組 V
           self.out_proj = nn.Linear(hidden_dim, hidden_dim)
   ```

2. **KV Cache 機制實現**
   ```python
   class KVCache:
       """KV Cache 管理"""
       def __init__(self, max_batch_size, max_seq_len, num_heads, head_dim):
           self.cache_k = torch.zeros(
               max_batch_size, max_seq_len, num_heads, head_dim
           )
           self.cache_v = torch.zeros(
               max_batch_size, max_seq_len, num_heads, head_dim
           )

       def update(self, k, v, start_pos):
           # 更新 cache
           seq_len = k.size(1)
           self.cache_k[:, start_pos:start_pos+seq_len] = k
           self.cache_v[:, start_pos:start_pos+seq_len] = v
   ```

3. **推理性能基準**
   - Prefill 階段性能
   - Decode 階段性能
   - KV Cache 記憶體占用
   - 端到端推理延遲

#### 預期結果
- 理解 KV Cache 的作用
- 識別推理瓶頸
- 建立性能基準

---

### Notebook 2: Multi-Query Attention 實現 (02-MQA_Implementation.ipynb)
**時間**: 45-60分鐘

#### 實驗目標
- 實現 MQA 機制
- 對比 MQA vs MHA 的性能
- 分析 KV Cache 記憶體節省
- 評估模型質量影響

#### 實驗內容
1. **MQA 實現**
   ```python
   class MultiQueryAttention(nn.Module):
       def __init__(self, hidden_dim, num_heads):
           self.num_heads = num_heads
           self.head_dim = hidden_dim // num_heads

           self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # num_heads 組 Q
           self.k_proj = nn.Linear(hidden_dim, self.head_dim)  # 1 組 K (共享)
           self.v_proj = nn.Linear(hidden_dim, self.head_dim)  # 1 組 V (共享)
           self.out_proj = nn.Linear(hidden_dim, hidden_dim)

       def forward(self, x):
           Q = self.q_proj(x)  # [B, N, H*D]
           K = self.k_proj(x)  # [B, N, D] - 單個 K
           V = self.v_proj(x)  # [B, N, D] - 單個 V

           # Q 重塑為多頭
           Q = Q.view(B, N, num_heads, head_dim)

           # K, V 重塑並擴展到所有 heads
           K = K.view(B, N, 1, head_dim).expand(B, N, num_heads, head_dim)
           V = V.view(B, N, 1, head_dim).expand(B, N, num_heads, head_dim)
   ```

2. **KV Cache 對比**
   - MHA Cache 大小: `2 × L × H × D`
   - MQA Cache 大小: `2 × L × 1 × D`
   - 節省比例: `(H-1)/H` (32 heads → 96.9%)

3. **性能測試**
   - 訓練速度對比
   - 推理速度對比 (關鍵!)
   - 記憶體占用對比

4. **質量評估**
   - 訓練 Loss 對比
   - 生成質量評估
   - Perplexity 測試

#### 預期結果
- KV Cache 減少 **30-32x**
- 推理速度提升 **1.5-2x**
- 模型質量輕微下降 (**<5%**)

---

### Notebook 3: Grouped-Query Attention 實現 (03-GQA_Implementation.ipynb)
**時間**: 45-60分鐘

#### 實驗目標
- 實現 GQA 機制
- 測試不同分組數的影響
- 對比 MHA/MQA/GQA 三者性能
- 理解質量與效率的權衡

#### 實驗內容
1. **GQA 實現**
   ```python
   class GroupedQueryAttention(nn.Module):
       def __init__(self, hidden_dim, num_heads, num_kv_groups=4):
           self.num_heads = num_heads  # 32 query heads
           self.num_kv_groups = num_kv_groups  # 4 KV groups
           self.head_dim = hidden_dim // num_heads

           assert num_heads % num_kv_groups == 0
           self.heads_per_group = num_heads // num_kv_groups  # 8

           self.q_proj = nn.Linear(hidden_dim, hidden_dim)
           self.k_proj = nn.Linear(hidden_dim, num_kv_groups * self.head_dim)
           self.v_proj = nn.Linear(hidden_dim, num_kv_groups * self.head_dim)
           self.out_proj = nn.Linear(hidden_dim, hidden_dim)

       def forward(self, x):
           Q = self.q_proj(x).view(B, N, num_heads, head_dim)
           K = self.k_proj(x).view(B, N, num_kv_groups, head_dim)
           V = self.v_proj(x).view(B, N, num_kv_groups, head_dim)

           # 每組 K, V 服務多個 Q heads
           # K: [B, N, 4, D] → repeat → [B, N, 32, D]
           K = K.repeat_interleave(self.heads_per_group, dim=2)
           V = V.repeat_interleave(self.heads_per_group, dim=2)
   ```

2. **分組數實驗**
   - num_kv_groups = [1, 2, 4, 8, 16, 32]
   - 測試每種配置的性能與質量
   - 找出最佳平衡點

3. **三方對比**
   - MHA (32 KV heads)
   - GQA-8 (8 KV groups) - Llama-2 配置
   - GQA-4 (4 KV groups)
   - MQA (1 KV head)

#### 預期結果
| 配置 | KV Heads | Cache 大小 | 推理速度 | 質量 | 推薦 |
|------|---------|-----------|---------|------|------|
| MHA | 32 | 100% | 1.0x | 100% | 訓練 |
| GQA-8 | 8 | 25% | 1.3x | 99% | ⭐推薦 |
| GQA-4 | 4 | 12.5% | 1.4x | 98% | 平衡 |
| MQA | 1 | 3% | 1.8x | 95% | 激進 |

---

### Notebook 4: 推理優化實戰 (04-Inference_Optimization.ipynb)
**時間**: 45-60分鐘

#### 實驗目標
- 實現完整的 KV Cache 推理優化
- 對比長文本生成性能
- 測試批次推理加速效果
- 分析實際部署場景

#### 實驗內容
1. **KV Cache 推理實現**
   ```python
   def generate_with_kv_cache(model, input_ids, max_new_tokens=100):
       """使用 KV Cache 的高效生成"""
       batch_size, seq_len = input_ids.size()

       # 初始化 KV Cache
       kv_cache = KVCache(batch_size, max_seq_len, num_kv_heads, head_dim)

       # Prefill 階段: 處理整個 prompt
       with torch.no_grad():
           outputs = model(input_ids, past_key_values=kv_cache)
           next_token = outputs.logits[:, -1, :].argmax(dim=-1)

       # Decode 階段: 逐個生成 token
       generated = [next_token]
       for _ in range(max_new_tokens - 1):
           outputs = model(
               next_token.unsqueeze(1),
               past_key_values=kv_cache,
               use_cache=True
           )
           next_token = outputs.logits[:, -1, :].argmax(dim=-1)
           generated.append(next_token)

       return torch.stack(generated, dim=1)
   ```

2. **長文本生成測試**
   - 生成長度: [100, 500, 1000, 2000] tokens
   - 測量每種架構的速度與記憶體
   - 對比 tokens/sec

3. **批次推理優化**
   ```python
   # 批次推理 (batch_size = 16)
   # MHA: KV Cache = 1.07 GB × 16 = 17 GB
   # GQA-8: KV Cache = 0.27 GB × 16 = 4.3 GB
   # MQA: KV Cache = 0.03 GB × 16 = 0.5 GB
   ```

4. **實際場景模擬**
   - Chatbot 多輪對話
   - 長文檔摘要
   - 代碼生成
   - 批次翻譯

#### 預期成果
- 理解 KV Cache 優化的重要性
- 掌握不同架構的適用場景
- 能夠為實際應用選擇最佳配置

---

## 🚀 環境準備

### 前置要求

#### 硬體要求
- **GPU**: 8GB+ VRAM (推薦 16GB+)
- **CUDA**: 11.6+
- **推理測試**: 記憶體越大, 可測試的批次/長度越大

#### 軟體依賴
```bash
# 已在 Poetry 環境中包含
source .venv/bin/activate

# 驗證關鍵套件
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

---

## 💡 理論深度解析

### KV Cache 原理

**自回歸生成過程**:
```
生成 "The cat is"

Step 1: Input = "The"
  → Q₁, K₁, V₁
  → Attention(Q₁, K₁, V₁) → output₁ → "cat"
  → Cache: K₁, V₁

Step 2: Input = "cat" (不需要重新計算 "The")
  → Q₂, K₂, V₂
  → Attention(Q₂, [K₁,K₂], [V₁,V₂]) → output₂ → "is"
  → Cache: K₁, V₁, K₂, V₂

Step 3: Input = "is"
  → Q₃, K₃, V₃
  → Attention(Q₃, [K₁,K₂,K₃], [V₁,V₂,V₃]) → output₃
  → Cache: K₁, V₁, K₂, V₂, K₃, V₃

每步只計算新 token 的 Q, K, V
歷史 K, V 從 cache 讀取 (避免重複計算)
```

**記憶體增長**:
```
每生成一個 token:
  KV Cache += 2 × num_kv_heads × head_dim × bytes

生成 N 個 tokens:
  Total KV Cache = 2 × N × num_kv_heads × head_dim × bytes

MQA 優勢: num_kv_heads = 1 (最小)
GQA 平衡: num_kv_heads = 4-8 (中等)
MHA 標準: num_kv_heads = 32 (最大)
```

### 為什麼 MQA/GQA 有效?

**關鍵洞察**:
1. **Query 需要多樣性** - 不同 heads 關注不同模式
2. **Key/Value 可以共享** - 資訊表示不需要那麼多樣化
3. **推理是記憶體受限** - 減少 KV Cache 是關鍵

**理論支持**:
- **實驗證據**: PaLM, Llama-2 等大模型證明 MQA/GQA 可行
- **質量保持**: GQA 可保持接近 MHA 的模型質量
- **速度提升**: KV Cache 減少直接轉化為速度提升

---

## 📈 性能預期

### 訓練性能對比 (GPT-2, seq_len=1024)

| 架構 | 前向時間 | 反向時間 | 記憶體 | 相對MHA |
|------|---------|---------|--------|---------|
| MHA | 45ms | 90ms | 8.5GB | 1.0x |
| GQA-8 | 42ms | 85ms | 8.2GB | 1.05x ⬆ |
| GQA-4 | 40ms | 82ms | 8.0GB | 1.08x ⬆ |
| MQA | 38ms | 78ms | 7.8GB | 1.12x ⬆ |

*訓練階段差異較小*

### 推理性能對比 (生成 1000 tokens)

| 架構 | KV Cache | Decode 時間 | 總時間 | 吞吐量 | 相對MHA |
|------|---------|------------|--------|--------|---------|
| MHA | 1.07GB | 15.2s | 16.0s | 62 tok/s | 1.0x |
| GQA-8 | 0.27GB | 11.8s | 12.5s | 80 tok/s | 1.29x ⬆ |
| GQA-4 | 0.13GB | 10.5s | 11.2s | 89 tok/s | 1.43x ⬆ |
| MQA | 0.03GB | 8.9s | 9.6s | 104 tok/s | 1.67x ⬆ |

*推理階段差異顯著*

### 批次推理對比 (batch_size=16, 生成 500 tokens)

| 架構 | 總 KV Cache | 是否 OOM (24GB GPU) | 吞吐量 |
|------|------------|-------------------|--------|
| MHA | 17.1GB | ⚠️  接近極限 | 880 tok/s |
| GQA-8 | 4.3GB | ✅ 充足 | 1150 tok/s |
| GQA-4 | 2.1GB | ✅ 充足 | 1280 tok/s |
| MQA | 0.5GB | ✅ 充足 | 1500 tok/s |

---

## 🛠️ 實際應用案例

### 1. Llama-2 的 GQA 配置

```python
# Llama-2-7B 使用 GQA
config = {
    'hidden_size': 4096,
    'num_attention_heads': 32,    # Query heads
    'num_key_value_heads': 8,     # KV groups (GQA-8)
    'max_position_embeddings': 4096
}

# KV Cache 節省: 32 → 8 (4x 減少)
```

### 2. Mistral-7B 的 GQA 配置

```python
# Mistral-7B 使用 GQA + Sliding Window
config = {
    'hidden_size': 4096,
    'num_attention_heads': 32,
    'num_key_value_heads': 8,     # GQA-8
    'sliding_window': 4096,       # 滑動窗口注意力
    'max_position_embeddings': 32768  # 支援超長序列
}
```

### 3. 自定義配置指南

**選擇 KV groups 數量**:
```python
# 推薦比例
num_query_heads = 32

# 保守 (質量優先)
num_kv_groups = 16  # GQA-16, 2x 減少

# 平衡 (Llama-2 方案)
num_kv_groups = 8   # GQA-8, 4x 減少

# 激進 (速度優先)
num_kv_groups = 4   # GQA-4, 8x 減少

# 極致 (MQA)
num_kv_groups = 1   # MQA, 32x 減少
```

---

## 🎓 學習檢查清單

完成本實驗室後，您應該能夠:

### 理論理解
- [ ] 解釋 KV Cache 的作用與重要性
- [ ] 理解 MHA/MQA/GQA 的架構差異
- [ ] 說明為什麼推理是記憶體受限的
- [ ] 分析不同架構的質量與效率權衡
- [ ] 理解分組注意力的廣播機制

### 實作技能
- [ ] 實現 Multi-Head Attention (MHA)
- [ ] 實現 Multi-Query Attention (MQA)
- [ ] 實現 Grouped-Query Attention (GQA)
- [ ] 實現 KV Cache 管理
- [ ] 優化推理性能

### 應用能力
- [ ] 為項目選擇合適的注意力架構
- [ ] 配置 KV Cache 參數
- [ ] 優化批次推理吞吐量
- [ ] 評估質量與速度的權衡

---

## 🚀 下一步學習

完成本實驗室後，建議繼續:

1. **Lab-1.7: DPO Alignment**
   - 直接偏好優化
   - 模型對齊技術

2. **推理部署優化**
   - vLLM 與 PagedAttention
   - Continuous Batching
   - Speculative Decoding

3. **生產環境實踐**
   - 模型服務化
   - 延遲優化
   - 成本控制

---

**實驗室狀態**: 🔄 開發中
**最後更新**: 2025-10-08
**維護者**: LLM 教學專案團隊

**相關文件**:
- 理論: `01-Theory/1.3-Optimization_and_Alignment.md` (MQA/GQA 章節)
- 前置實驗: `Lab-1.5-FlashAttention_Deep_Dive`
- 後續實驗: `Lab-1.7-DPO_Alignment`

**相關論文**:
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
