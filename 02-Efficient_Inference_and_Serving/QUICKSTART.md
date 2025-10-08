# Chapter 2: Efficient Inference & Serving - Quick Start Guide

## 快速開始

### 1. 環境準備

```bash
# 激活 Poetry 環境
cd 00-Course_Setup
source .venv/bin/activate

# 安裝第二章依賴
poetry install --all-extras

# 可選: Flash Attention
pip install flash-attn --no-build-isolation
```

### 2. 學習路徑

#### 第一週: 理論基礎 + vLLM 部署
```bash
# 1. 閱讀理論文件
cat 01-Theory/2.1-Inference_Engines.md
cat 01-Theory/2.2-Serving_and_Optimization.md

# 2. 開始 Lab-2.1
cd 02-Labs/Lab-2.1-vLLM_Deployment
jupyter lab

# 執行順序:
# - 01-Setup_and_Installation.ipynb
# - 02-Basic_Inference.ipynb  
# - 03-Advanced_Features.ipynb
# - 04-Production_Deployment.ipynb
```

#### 第二週: 推理優化
```bash
cd 02-Labs/Lab-2.2-Inference_Optimization
jupyter lab

# 執行順序:
# - 01-KV_Cache_Optimization.ipynb
# - 02-Speculative_Decoding.ipynb
# - 03-Quantization_Inference.ipynb
# - 04-Comprehensive_Optimization.ipynb
```

#### 第三週: FastAPI 服務
```bash
cd 02-Labs/Lab-2.3-FastAPI_Service
jupyter lab

# 執行順序:
# - 01-Basic_API.ipynb
# - 02-Async_Processing.ipynb
# - 03-Integration_with_vLLM.ipynb
# - 04-Monitoring_and_Deploy.ipynb
```

### 3. 核心技能檢查表

完成本章後，你應該能夠:
- ✅ 部署並優化 vLLM 推理引擎
- ✅ 理解 PagedAttention 原理
- ✅ 實現 Speculative Decoding (1.5-3x 加速)
- ✅ 應用量化技術 (INT8/INT4)
- ✅ 構建 FastAPI 服務
- ✅ 集成 Prometheus 監控
- ✅ 使用 Docker 部署

### 4. 常見問題

**Q: GPU 記憶體不足怎麼辦?**
A: 使用更小的模型 (如 `gpt2`, `opt-125m`) 或降低 `gpu_memory_utilization`

**Q: vLLM 安裝失敗?**
A: 確認 CUDA 版本匹配，可能需要: `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121`

**Q: Notebooks 無法執行?**
A: 確認已激活正確的虛擬環境，並安裝所有依賴

### 5. 快速驗證

```python
# 驗證 vLLM 安裝
from vllm import LLM, SamplingParams
llm = LLM(model="gpt2")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
print(outputs[0].outputs[0].text)

# 驗證 FastAPI
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def root():
    return {"status": "ok"}
```

---

**最後更新**: 2025-10-09
**適用版本**: Chapter 2 v0.3.0
