# Lab-2.1: Triton Inference Server 基礎與模型部署

## 🎯 實驗室概述

本實驗室介紹 **NVIDIA Triton Inference Server**，業界領先的企業級推理服務器。您將學習從安裝配置到模型部署的完整流程，掌握多模型統一管理的核心技能。

**為什麼選擇 Triton？**
- 🏢 **企業級標準**: NVIDIA 官方推理解決方案，被 Netflix、PayPal、VISA 等企業廣泛使用
- 🔧 **多 Backend 支援**: PyTorch、TensorRT、ONNX、Python、vLLM 統一管理
- ⚡ **高性能**: 動態批次處理、GPU 優化、並發推理
- 📊 **完善監控**: 內建豐富指標，企業級可觀測性

## 🎓 學習目標

完成本實驗室後，您將能夠：
- ✅ 安裝和配置 Triton Inference Server
- ✅ 設計和管理模型倉庫 (Model Repository)
- ✅ 部署 PyTorch 模型到 Triton
- ✅ 配置動態批次處理優化性能
- ✅ 實施基礎監控和健康檢查
- ✅ 進行性能基準測試和調優

## 📚 實驗室結構 (4個階段)

### 📖 01-Triton_Setup_and_Installation.ipynb
**重點**: Triton 基礎環境建置
**時長**: 60-90 分鐘

**學習內容**:
- Triton Server 安裝與驗證
- Docker 容器化部署
- 基礎配置與服務啟動
- REST 和 gRPC API 介紹
- 健康檢查端點測試

**技術棧**:
- NVIDIA Triton Inference Server
- Docker 容器化
- curl/httpx API 測試
- 基礎監控設置

### 🏗️ 02-Model_Repository_Design.ipynb
**重點**: 模型倉庫設計與配置
**時長**: 60-90 分鐘

**學習內容**:
- Model Repository 架構設計
- 模型配置文件 (`config.pbtxt`) 編寫
- 版本管理策略
- 模型載入與卸載
- 動態批次處理配置

**技術重點**:
- Model Repository 最佳實踐
- 配置文件語法與參數調優
- 版本控制策略
- 記憶體管理

### ⚡ 03-PyTorch_Backend_Deployment.ipynb
**重點**: PyTorch 模型部署與優化
**時長**: 90-120 分鐘

**學習內容**:
- PyTorch 模型轉換與部署
- TorchScript 優化
- 動態批次配置調優
- 推理性能測試
- 與 HuggingFace 性能對比

**實踐項目**:
- 部署 Llama-2-7B 模型
- 配置最佳批次處理參數
- 測量延遲和吞吐量指標
- 記憶體使用分析

### 📊 04-Monitoring_and_Performance.ipynb
**重點**: 監控系統與性能分析
**時長**: 60-90 分鐘

**學習內容**:
- Triton 內建指標系統
- Prometheus 整合配置
- 性能瓶頸分析
- 資源利用率優化
- 基礎告警設置

**監控指標**:
- 推理延遲 (TTFT, ITL)
- 吞吐量 (RPS, TPS)
- 資源利用 (GPU, Memory)
- 錯誤率與可用性

## 🛠️ 技術要求

### 硬體需求
- **GPU**: 16GB+ VRAM (RTX 4080, A10G 或更高)
- **RAM**: 32GB+ 系統記憶體
- **存儲**: 50GB+ 可用空間
- **網路**: 穩定網路連線 (模型下載)

### 軟體環境
```bash
# NVIDIA Container Toolkit
nvidia-container-toolkit

# Docker (with GPU support)
docker >= 20.10

# Python 環境
python >= 3.8
tritonclient[all] >= 2.40

# 可選工具
curl, httpx, locust (負載測試)
```

### 預備知識
- ✅ **完成第一章**: PEFT 訓練與優化技術
- ✅ **Docker 基礎**: 容器化概念與操作
- ✅ **REST API 概念**: HTTP 協議與 API 設計
- ✅ **PyTorch 基礎**: 模型載入與推理

## 🚀 快速開始

```bash
# 1. 激活環境
cd /path/to/00-Course_Setup
source .venv/bin/activate

# 2. 安裝 Triton 客戶端
pip install tritonclient[all]

# 3. 拉取 Triton Server 映像
docker pull nvcr.io/nvidia/tritonserver:24.07-py3

# 4. 進入實驗室
cd ../02-Efficient_Inference_and_Serving/02-Labs/Lab-2.1-Triton_Server_Basics

# 5. 啟動 Jupyter Lab
jupyter lab
```

## 📈 預期學習成果

### 技術技能
- **Triton 部署**: 掌握企業級推理服務器部署
- **模型管理**: 理解多模型統一管理架構
- **性能調優**: 優化動態批次處理參數
- **監控運維**: 建立完整的監控體系

### 業界應用
- **企業 AI 平台**: 構建統一的模型服務平台
- **MLOps 工程師**: 推理服務的運維管理
- **性能優化**: 企業級推理性能調優
- **系統架構**: 大規模 AI 服務設計

## 🌟 實驗室特色

### 1. 企業級實踐導向
- 使用 Netflix、PayPal 等企業的真實架構模式
- 涵蓋從開發到生產的完整流程
- 包含故障排除與運維最佳實踐

### 2. 完整技術棧覆蓋
- **推理引擎**: Triton Server 深度解析
- **容器化**: Docker 優化與部署
- **API 設計**: REST/gRPC 雙協議支援
- **監控運維**: Prometheus 企業級監控

### 3. 漸進式學習設計
- 從簡單模型開始，逐步增加複雜度
- 每個階段都有明確的學習檢查點
- 包含豐富的練習與延伸思考

## 🔗 與其他實驗室的銜接

### 前置依賴
- **Lab-1.1 到 Lab-1.8**: PEFT 訓練的模型將用於 Triton 部署
- **第一章理論**: 分散式訓練、優化技術的理論基礎

### 後續實驗室預覽
- **Lab-2.2**: 多模型管理 (A/B 測試、版本控制)
- **Lab-2.3**: Backend 整合 (TensorRT、vLLM、自定義)
- **Lab-2.4**: 企業級功能 (模型組合、高級調優)
- **Lab-2.5**: 生產環境 (Kubernetes、CI/CD、運維)

## 📚 參考資源

### 官方文檔
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Model Repository Guide](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md)

### 企業案例
- [Netflix 的 Triton 部署經驗](https://netflixtechblog.com/how-netflix-uses-triton-inference-server-to-serve-ml-models-at-scale-4aaa0c7bfce)
- [PayPal ML Platform](https://medium.com/paypal-tech/paypal-machine-learning-platform-9b8dd4b75c1)

### 技術論文
- "Triton Inference Server: Optimizing Deep Learning Inference" (NVIDIA Technical Paper)
- "Dynamic Batching for Production Machine Learning" (MLSys 2022)

## 🎯 成功評估標準

### 基礎要求 (必須達成)
- [ ] 成功安裝並啟動 Triton Server
- [ ] 部署至少一個 PyTorch 模型
- [ ] 完成基礎推理API調用測試
- [ ] 配置基本的健康檢查

### 進階目標 (推薦達成)
- [ ] 實現動態批次處理優化
- [ ] 設置 Prometheus 監控整合
- [ ] 完成性能基準測試
- [ ] 實施基礎故障排除

### 專家級挑戰 (選擇性)
- [ ] 部署多個模型版本
- [ ] 自定義配置文件優化
- [ ] 實現負載測試腳本
- [ ] 設計監控儀表板

---

**版本**: v1.0 (Triton-focused redesign)
**創建日期**: 2025-10-09
**預估教學時間**: 4-6 小時
**難度等級**: ⭐⭐⭐ (Intermediate)
**優先級**: 🔴 P0 (最高)

**注意**: 本實驗室代表第二章的重大架構轉變，從 vLLM 單一引擎轉向 Triton 企業級多模型管理平台。