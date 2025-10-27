# Lab-2.1: vLLM Deployment Track

## 🎯 實驗目標
掌握 vLLM 推理引擎的部署與優化，從基礎概念到生產級應用。

## 📚 學習路徑

### **階段式學習設計**
```
📚 環境與概念 → 🚀 實戰與優化 → ⚡ 進階功能
     (基礎)         (實用)         (精通)
```

## 📋 實驗結構

### **01-Setup_and_Installation.ipynb** (重新設計)
**專注**: 環境設置 + 核心概念理解
- ✅ 環境驗證與 vLLM 安裝
- 🧠 PagedAttention 原理深度解析
- 📊 記憶體效率分析與比較
- 🔧 基本 API 使用與配置理解
- 💡 vLLM 與傳統方法的本質差異

**⏱️ 時間**: 45-75 分鐘

### **02-Basic_Inference.ipynb** (重新設計)
**專注**: 生產級應用 + 性能優化
- 🚀 批次推理與動態調度技術
- 📊 系統性性能基準測試
- 💾 GPU 記憶體使用分析與監控
- ⚡ vLLM vs HuggingFace 詳細對比
- 🎛️ 進階 Sampling 參數調優
- 🔧 生產環境配置策略

**⏱️ 時間**: 75-120 分鐘

### 03-Advanced_Features.ipynb
- Continuous Batching
- Sampling strategies
- Long context handling
- Multi-model management

### 04-Production_Deployment.ipynb
- OpenAI-compatible API server
- Performance tuning
- Monitoring and logging
- Deployment best practices

## Prerequisites

- GPU with 16GB+ VRAM (recommended)
- CUDA 12.1+
- Python 3.9+
- Poetry environment activated

## Estimated Time

- Setup: 30-60 min
- Each notebook: 60-90 min
- Total: 4-6 hours

## Key Technologies

- **vLLM**: High-performance inference engine
- **PagedAttention**: Memory-efficient KV cache management
- **Continuous Batching**: Dynamic request scheduling

## Getting Started

```bash
# Activate environment
cd /path/to/00-Course_Setup
source .venv/bin/activate

# Navigate to lab
cd ../02-Efficient_Inference_and_Serving/02-Labs/Lab-2.1-vLLM_Deployment

# Start Jupyter Lab
jupyter lab
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)

---

**Version**: v1.0
**Last Updated**: 2025-10-09
**Difficulty**: ⭐⭐⭐ (Intermediate)
