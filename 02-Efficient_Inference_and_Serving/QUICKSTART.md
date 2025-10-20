# Chapter 2: Efficient Inference & Serving - Quick Start Guide
## 雙軌道架構快速開始

### 1. 環境準備

```bash
# 激活 Poetry 環境
cd 00-Course_Setup
source .venv/bin/activate

# 安裝第二章依賴
poetry install --all-extras

# 可選: Flash Attention
pip install flash-attn --no-build-isolation

# Triton Server (需要 Docker)
docker pull nvcr.io/nvidia/tritonserver:24.08-py3
```

### 2. 選擇學習路徑

#### 🎯 基礎路徑 (vLLM Track) - 推薦新手
適合快速原型開發和個人項目

```bash
# 1. 理論基礎
cat 01-Theory/2.1-Inference_Engines.md
cat 01-Theory/2.2-Serving_and_Optimization.md

# 2. vLLM 軌道 (依序執行)
cd 02-Labs/vLLM_Track/Lab-2.1-vLLM_Deployment
cd 02-Labs/vLLM_Track/Lab-2.2-Inference_Optimization
cd 02-Labs/vLLM_Track/Lab-2.3-FastAPI_Service
cd 02-Labs/vLLM_Track/Lab-2.4-Production_Deployment
cd 02-Labs/vLLM_Track/Lab-2.5-Performance_Monitoring
```

#### 🏢 企業路徑 (Triton Track) - 推薦有經驗者
適合企業級平台開發和 MLOps 工程師

```bash
# 1. 理論基礎 (重點企業級架構)
cat 01-Theory/2.2-Serving_and_Optimization.md

# 2. Triton 軌道 (依序執行)
cd 02-Labs/Triton_Track/Lab-2.1-Triton_Server_Basics
cd 02-Labs/Triton_Track/Lab-2.2-Multi_Model_Management
cd 02-Labs/Triton_Track/Lab-2.3-Backend_Integration
cd 02-Labs/Triton_Track/Lab-2.4-Enterprise_Features
cd 02-Labs/Triton_Track/Lab-2.5-Production_Operations
```

#### 🚀 完整路徑 (Both Tracks) - 推薦專業進階
適合 AI Infrastructure Engineer 和技術主管

```bash
# Phase 1: vLLM 基礎 (4週)
# Phase 2: Triton 企業級 (4週)
# Phase 3: 整合與對比 (2週)
```

### 3. 核心技能檢查表

#### vLLM Track 技能
- ✅ 部署並優化 vLLM 推理引擎
- ✅ 理解 PagedAttention 原理
- ✅ 實現 Speculative Decoding (1.5-3x 加速)
- ✅ 應用量化技術 (INT8/INT4)
- ✅ 構建 FastAPI 服務
- ✅ 集成 Prometheus 監控
- ✅ 使用 Docker 部署

#### Triton Track 技能
- ✅ 部署和配置 Triton Inference Server
- ✅ 設計多模型倉庫架構
- ✅ 實現 A/B 測試和版本控制
- ✅ 整合多種 Backend (PyTorch/TensorRT/vLLM)
- ✅ 開發模型組合 (Ensemble)
- ✅ 實施企業級監控和運維
- ✅ 使用 Kubernetes 進行生產部署

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

### 6. 軌道選擇建議

| 背景 | 推薦軌道 | 理由 |
|------|----------|------|
| 初學者/個人項目 | vLLM Track | 學習曲線平緩，快速上手 |
| 有ML經驗 | 先 vLLM 後 Triton | 循序漸進，技能互補 |
| 企業環境/MLOps | Triton Track | 直接對應工作需求 |
| 技術主管/架構師 | Both Tracks | 完整技術視野 |

---

**最後更新**: 2025-10-16
**適用版本**: Chapter 2 v4.0 (雙軌道架構)
