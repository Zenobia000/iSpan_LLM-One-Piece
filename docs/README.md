# 項目文檔導航

## 📚 主要文檔位置

### 總體課程介紹
- **主README**: `/README.md` - 完整課程介紹和結構
- **CLAUDE.md**: `/CLAUDE.md` - Claude Code使用指導

### 各章節文檔

#### 第0章：LLM基礎知識體系 ⭐ **[新增]**
- **章節README**: `/00-LLM_Fundamentals/README.md`
- **學習路徑**: `/00-LLM_Fundamentals/LEARNING_PATH.md`
- **理論文檔**: `/00-LLM_Fundamentals/01-Theory/*/README.md`
- **實踐Lab**: `/00-LLM_Fundamentals/02-Labs/Lab-*/README.md`

#### 第1章：核心訓練技術
- **章節狀態**: `/01-Core_Training_Techniques/README.md`
- **理論文檔**: `/01-Core_Training_Techniques/01-Theory/*/README.md`
- **PEFT Labs**: `/01-Core_Training_Techniques/02-Labs/PEFT_Labs/Lab-*/README.md`

#### 第2章：高效推理部署
- **章節狀態**: `/02-Efficient_Inference_and_Serving/CHAPTER_02_STATUS.md`
- **Triton狀態**: `/02-Efficient_Inference_and_Serving/CHAPTER_02_TRITON_STATUS.md`
- **快速開始**: `/02-Efficient_Inference_and_Serving/QUICKSTART.md`

#### 第3章：模型壓縮 & 第4章：評估工程
- 詳細文檔請查看對應章節目錄

## 🛠️ 開發文檔

### 環境配置
- **Poetry環境**: `/00-Course_Setup/pyproject.toml` - 依賴管理
- **GPU檢查**: `/01-PyTorch_Basics/check_gpu.py` - 環境驗證

### 共享工具
- **第1章工具**: `/common_utils/` - PEFT訓練工具
- **第0章工具**: `/00-LLM_Fundamentals/utils/` - 基礎分析工具

## 📖 如何使用本項目

### 新手入門
1. 閱讀主`/README.md`了解完整課程結構
2. 從第0章開始：`/00-LLM_Fundamentals/LEARNING_PATH.md`
3. 按章節順序學習：0→1→2→3→4

### 查找特定內容
- **理論學習**: 查看各章節`01-Theory/`目錄
- **實踐操作**: 查看各章節`02-Labs/`目錄
- **環境問題**: 查看`/00-Course_Setup/`和`/01-PyTorch_Basics/`
- **工具使用**: 查看`/common_utils/`或章節utils目錄

### 貢獻改進
- 按照各README中的說明進行環境設置
- 遵循現有代碼風格和文檔格式
- 提交前測試所有代碼功能

---

**最後更新**: 2025-10-15
**維護者**: LLM工程化課程團隊