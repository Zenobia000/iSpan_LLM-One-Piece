# LLM 推理性能指標完整指南

## 核心指標

### 1. TTFT (Time To First Token)

**定義**：從發送請求到收到第一個 token 的時間。

#### 重要性
- **用戶體驗**：感知速度快慢
- **系統響應**：衡量「立即反饋」能力

#### 計算方式
```
TTFT = 處理時間 + 預填充時間 + KV Cache 分配時間

具體包括：
1. 輸入處理（tokenization）
2. 預填充（prefill）階段的所有計算
3. KV Cache 初始化
4. 第一個 output token 生成
```

#### 為什麼 TTFT 很重要？

**用戶心理學**：
- 前 100ms：用戶感覺系統「即時」
- 100-300ms：感覺「快速」
- 300-1000ms：感覺「慢」
- >1000ms：感覺「卡頓」

**常見數值範圍**：
```
優秀 (< 100ms):  聊天場景、即時反饋
良好 (100-300ms): 生產環境可接受
一般 (300-1000ms): 需要優化
差勁 (> 1000ms):  用戶體驗差
```

#### 優化策略
1. **預填充優化**：
   - FlashAttention-2 加速 attention 計算
   - 量化模型降低計算量
   
2. **KV Cache 優化**：
   - 提前分配記憶體
   - 使用連續記憶體避免碎片
   
3. **並行處理**：
   - Continuous batching
   - PagedAttention（vLLM 的核心技術）

---

### 2. 吞吐量 (Throughput)

**定義**：每秒生成的 token 數量 (tokens/s)。

#### 計算方式
```
吞吐量 = 總生成 tokens / 總時間

或考慮 batch 大小：
吞吐量 = batch_size × (平均序列長度 / 平均時間)
```

#### 常見數值（A100, 7B 模型）
```
HuggingFace:      250 tokens/s  （基準）
vLLM:            2,800 tokens/s  （11.2x 加速）
TensorRT-LLM:    2,500 tokens/s  （10x 加速）
FastAPI 簡單:    200 tokens/s   （慢）
```

#### 影響因素
- **GPU 利用率**：記憶體利用率越高，吞吐量越好
- **Batch 大小**：適當的 batching 提升效率
- **上下文長度**：長上下文會降低吞吐量

---

### 3. 記憶體利用率 (Memory Utilization)

**定義**：GPU 記憶體的有效使用率。

#### 計算方式
```
利用率 = (實際使用記憶體 / 總可用記憶體) × 100%

關鍵記憶體組成：
1. Model weights:      ~14GB (7B model)
2. KV Cache:          變動（與 seq len 相關）
3. Activation:         ~2-4GB
4. System overhead:   ~1GB
```

#### 為什麼重要？
```
低利用率 (40-60%):
- 浪費資源
- 無法充分利用硬體
- 成本效益差

高利用率 (80-95%):
- 充分利用硬體
- 最大化吞吐量
- 經濟效益好
```

#### 優化策略
1. **量化**：FP16 → INT8 → INT4
2. **PagedAttention**：動態分配 KV Cache
3. **Continuous Batching**：保持滿載

---

## 三個指標的關係

### 通常的 Trade-off

```
高吞吐量 ⟷ 低 TTFT  ⟷ 高利用率
    ✓          ✗          ✓
```

**挑戰**：很難同時優化三者

**實際策略**：
1. **聊天場景**：優先降低 TTFT（用戶體驗）
2. **批量推理**：優先提升吞吐量（成本效益）
3. **記憶體受限**：優先提升利用率（資源效率）

---

## 性能對比分析

### 你的數據解讀

```yaml
引擎對比 (Llama-2-7B, A100):

HuggingFace:
  吞吐量: 250 tokens/s
  TTFT: 450ms
  記憶體: 60%
  → 特點: 簡單但效率低

vLLM:
  吞吐量: 2,800 tokens/s (11.2x)
  TTFT: 120ms (3.8x faster)
  記憶體: 95%
  → 特點: 生產級優化
```

**vLLM 如何做到？**
1. **PagedAttention**：動態 KV Cache 管理
2. **Continuous Batching**：始終保持 GPU 滿載
3. **FlashAttention**：高效的 attention 計算
4. **記憶體優化**：避免不必要的拷貝

---

## 完整性能指標體系

### 1. 延遲指標 (Latency Metrics)

```python
class LatencyMetrics:
    def __init__(self):
        self.ttft = None        # Time To First Token
        self.ttl = None         # Time To Last Token (總時間)
        self.tpt = None         # Time Per Token (平均每個 token 時間)
        
    def calculate_metrics(self, request_time, first_token_time, last_token_time, num_tokens):
        """計算延遲指標"""
        self.ttft = first_token_time - request_time
        self.ttl = last_token_time - request_time
        self.tpt = (self.ttl - self.ttft) / (num_tokens - 1)
        
        return {
            "TTFT (ms)": self.ttft * 1000,
            "TTLT (ms)": self.ttl * 1000,
            "平均時間/token (ms)": self.tpt * 1000
        }
```

---

### 2. 吞吐量指標 (Throughput Metrics)

```python
class ThroughputMetrics:
    def __init__(self):
        self.overall_throughput = None   # 整體吞吐量
        self.concurrent_throughput = None # 並發吞吐量
        self.p95_latency = None          # P95 延遲
        self.p99_latency = None          # P99 延遲
        
    def benchmark(self, num_requests, total_tokens, total_time):
        """基準測試"""
        self.overall_throughput = total_tokens / total_time
        
        return {
            "吞吐量 (tokens/s)": self.overall_throughput,
            "每秒請求數 (req/s)": num_requests / total_time,
            "平均批次大小": total_tokens / num_requests
        }
```

---

### 3. 資源利用率 (Resource Utilization)

```python
class ResourceMetrics:
    def __init__(self):
        self.gpu_memory = None
        self.gpu_utilization = None
        self.tensor_cores = None
        
    def monitor_resources(self):
        """監控資源使用"""
        import torch
        
        return {
            "GPU 記憶體 (GB)": torch.cuda.memory_allocated() / 1e9,
            "GPU 利用率 (%)": get_gpu_util(),
            "Tensor Core 使用率 (%)": get_tensor_core_usage()
        }
```

---

## 實際測試範例

### 基準測試腳本

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMInferenceBenchmark:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def benchmark_single_request(self, prompt, max_tokens=100):
        """單個請求的性能測試"""
        # 測量 TTFT
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        first_token_time = time.time()
        ttft = first_token_time - start_time
        
        total_tokens = len(outputs.sequences[0]) - len(inputs['input_ids'][0])
        tpt = (time.time() - first_token_time) / (total_tokens - 1) if total_tokens > 1 else 0
        
        return {
            "TTFT (ms)": ttft * 1000,
            "總時延 (ms)": (time.time() - start_time) * 1000,
            "生成 tokens": total_tokens,
            "平均 TPT (ms)": tpt * 1000
        }
    
    def benchmark_throughput(self, prompts, num_requests=10):
        """吞吐量測試"""
        start_time = time.time()
        total_tokens = 0
        
        for i, prompt in enumerate(prompts * (num_requests // len(prompts) + 1))[:num_requests]:
            metrics = self.benchmark_single_request(prompt)
            total_tokens += metrics["生成 tokens"]
            
        total_time = time.time() - start_time
        
        return {
            "總時間 (s)": total_time,
            "總 tokens": total_tokens,
            "吞吐量 (tokens/s)": total_tokens / total_time,
            "平均請求延遲 (ms)": total_time / num_requests * 1000
        }

# 使用範例
benchmark = LLMInferenceBenchmark()
results = benchmark.benchmark_single_request("解釋什麼是機器學習")
print(results)
```

---

## 生產環境最佳實踐

### 1. 指標監控

```python
import prometheus_client
from prometheus_client import Counter, Histogram

# 定義指標
ttft_histogram = Histogram('llm_ttft_seconds', 'Time to first token')
throughput_counter = Counter('llm_tokens_total', 'Total generated tokens')
memory_gauge = Gauge('llm_gpu_memory_gb', 'GPU memory usage')

def monitored_generate(prompt):
    """帶監控的生成"""
    start = time.time()
    
    # 生成邏輯
    output = model.generate(prompt)
    
    ttft = # ... 計算
    ttft_histogram.observe(ttft)
    throughput_counter.inc(len(output))
    
    return output
```

### 2. SLA 設定

```yaml
性能目標（生產環境）:

聊天應用:
  TTFT: < 200ms
  吞吐量: > 1,000 tokens/s
  記憶體利用率: > 70%

批量處理:
  TTFT: < 1000ms (可接受)
  吞吐量: > 5,000 tokens/s
  記憶體利用率: > 90%

實時應用:
  TTFT: < 50ms (極致優化)
  吞吐量: > 500 tokens/s
  記憶體利用率: > 80%
```

### 3. 優化策略選擇

```
情況 1: TTFT 太慢
解決: 
  - 使用 FlashAttention
  - 量化模型（INT8/INT4）
  - 預加載模型

情況 2: 吞吐量太低
解決:
  - Continuous batching
  - 增加 batch size
  - 使用 vLLM/TensorRT-LLM

情況 3: 記憶體不夠
解決:
  - 量化模型
  - 使用 PagedAttention
  - 多卡推理
```

---

## 總結

### 指標選擇指南

| 使用場景 | 優先指標 | 次要指標 | 可忽略 |
|---------|---------|---------|--------|
| **聊天應用** | TTFT | 吞吐量 | 記憶體 |
| **批量推理** | 吞吐量 | 記憶體 | TTFT |
| **API 服務** | TTFT + 吞吐量 | 記憶體 | - |
| **研究開發** | 靈活性 | - | - |

### 關鍵記住

1. **TTFT** = 用戶感知的速度（最重要）
2. **吞吐量** = 系統處理能力（成本效益）
3. **記憶體利用率** = 資源效率（硬體利用）

三者通常是 trade-off 關係！

---

**最後更新**: 2025-01-13
**版本**: 1.0

