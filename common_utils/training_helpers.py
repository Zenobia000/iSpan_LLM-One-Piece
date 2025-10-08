"""
Training Helper Functions with Error Handling
訓練輔助函數 - 提供統一的錯誤處理與資源管理

包含所有 PEFT 實驗室共用的訓練輔助功能。
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import gc

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# GPU 與資源管理
# =============================================================================

def check_gpu_availability(verbose: bool = True) -> Dict[str, Any]:
    """
    統一的 GPU 檢查函數

    Args:
        verbose: 是否打印詳細信息

    Returns:
        GPU 信息字典
    """
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device_count': 0,
        'device_name': None,
        'total_memory_gb': 0,
        'cuda_version': None
    }

    if torch.cuda.is_available():
        gpu_info['device_count'] = torch.cuda.device_count()
        gpu_info['device_name'] = torch.cuda.get_device_name(0)
        gpu_info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_info['cuda_version'] = torch.version.cuda

        if verbose:
            print("=" * 60)
            print("GPU 環境檢查")
            print("=" * 60)
            print(f"✅ CUDA 可用")
            print(f"GPU 數量: {gpu_info['device_count']}")
            print(f"GPU 型號: {gpu_info['device_name']}")
            print(f"總記憶體: {gpu_info['total_memory_gb']:.2f} GB")
            print(f"CUDA 版本: {gpu_info['cuda_version']}")
            print("=" * 60)
    else:
        if verbose:
            print("⚠️  CUDA 不可用，將使用 CPU 訓練")
            print("建議: 使用 GPU 可大幅加速訓練")

    return gpu_info


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    獲取推薦的訓練設備

    Args:
        prefer_cuda: 優先使用 CUDA

    Returns:
        torch.device 對象
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 使用設備: {device} ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ 使用設備: {device} (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"⚠️  使用設備: {device}")

    return device


def clear_gpu_cache():
    """清空 GPU 緩存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU 緩存已清空")


def print_gpu_memory_usage(prefix: str = ""):
    """
    打印當前 GPU 記憶體使用情況

    Args:
        prefix: 打印前綴
    """
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    print(f"{prefix}GPU 記憶體: 已分配 {allocated:.2f}GB / "
          f"已保留 {reserved:.2f}GB / 峰值 {max_allocated:.2f}GB")


# =============================================================================
# 檢查點管理
# =============================================================================

def load_latest_checkpoint(
    output_dir: str,
    prefix: str = "checkpoint-",
    raise_on_missing: bool = False
) -> Optional[str]:
    """
    安全地載入最新檢查點

    Args:
        output_dir: 輸出目錄
        prefix: 檢查點前綴
        raise_on_missing: 找不到時是否拋出異常

    Returns:
        檢查點路徑，如果不存在則返回 None

    Raises:
        ValueError: 如果 raise_on_missing=True 且目錄不存在
        FileNotFoundError: 如果 raise_on_missing=True 且找不到檢查點
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        msg = f"輸出目錄不存在: {output_dir}"
        if raise_on_missing:
            raise ValueError(msg)
        logger.warning(msg)
        return None

    try:
        checkpoints = [
            d for d in output_path.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]

        if not checkpoints:
            msg = f"在 {output_dir} 中找不到檢查點 (prefix: {prefix})"
            if raise_on_missing:
                raise FileNotFoundError(msg)
            logger.warning(f"⚠️  {msg}")
            return None

        # 按修改時間排序，取最新
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        latest_path = str(latest)

        logger.info(f"✅ 載入檢查點: {latest_path}")
        return latest_path

    except Exception as e:
        logger.error(f"載入檢查點時發生錯誤: {e}")
        if raise_on_missing:
            raise
        return None


def save_checkpoint_safe(
    model: Any,
    output_dir: str,
    checkpoint_name: str = "best_model"
) -> bool:
    """
    安全地保存模型檢查點

    Args:
        model: 模型對象
        output_dir: 輸出目錄
        checkpoint_name: 檢查點名稱

    Returns:
        是否成功保存
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_path / checkpoint_name

        # 根據模型類型選擇保存方法
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path / "pytorch_model.bin")

        logger.info(f"✅ 模型已保存至: {checkpoint_path}")
        return True

    except Exception as e:
        logger.error(f"❌ 保存模型失敗: {e}")
        return False


# =============================================================================
# 訓練配置驗證
# =============================================================================

def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    驗證訓練配置的合理性

    Args:
        config: 訓練配置字典

    Returns:
        (是否有效, 警告訊息列表)
    """
    warnings = []
    is_valid = True

    # 檢查必要參數
    required_params = ['learning_rate', 'num_train_epochs', 'per_device_train_batch_size']
    for param in required_params:
        if param not in config:
            warnings.append(f"❌ 缺少必要參數: {param}")
            is_valid = False

    # 檢查學習率範圍
    if 'learning_rate' in config:
        lr = config['learning_rate']
        if lr <= 0 or lr > 1e-2:
            warnings.append(f"⚠️  學習率可能不合理: {lr} (建議範圍: 1e-5 ~ 1e-3)")

    # 檢查批次大小
    if 'per_device_train_batch_size' in config:
        bs = config['per_device_train_batch_size']
        if bs < 1 or bs > 128:
            warnings.append(f"⚠️  批次大小可能不合理: {bs}")

    # 檢查梯度累積
    if 'gradient_accumulation_steps' in config:
        gas = config['gradient_accumulation_steps']
        bs = config.get('per_device_train_batch_size', 1)
        effective_bs = bs * gas
        if effective_bs < 8:
            warnings.append(f"⚠️  有效批次大小過小: {effective_bs} (建議 ≥ 8)")

    # 檢查記憶體相關設置
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_memory < 16 and config.get('per_device_train_batch_size', 1) > 4:
            warnings.append(f"⚠️  GPU 記憶體 {total_memory:.1f}GB 可能不足，建議減小批次大小")

    return is_valid, warnings


# =============================================================================
# 評估指標計算
# =============================================================================

def compute_metrics_safe(eval_pred, tokenizer=None):
    """
    安全的評估指標計算函數（帶異常處理）

    Args:
        eval_pred: (predictions, labels) 元組
        tokenizer: Tokenizer（可選，用於解碼）

    Returns:
        指標字典
    """
    try:
        predictions, labels = eval_pred

        # 型態安全轉換
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.from_numpy(predictions).float()
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels).long()

        # 計算損失
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1)

        # 過濾 padding tokens (假設 -100)
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]

        # 計算交叉熵損失
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions, labels)

        # 計算困惑度
        perplexity = torch.exp(loss)

        metrics = {
            "eval_loss": loss.item(),
            "perplexity": perplexity.item()
        }

        # 計算準確率（Top-1）
        pred_labels = torch.argmax(predictions, dim=-1)
        accuracy = (pred_labels == labels).float().mean()
        metrics["accuracy"] = accuracy.item()

        return metrics

    except Exception as e:
        logger.error(f"計算評估指標時發生錯誤: {e}")
        return {
            "eval_loss": float('inf'),
            "perplexity": float('inf'),
            "accuracy": 0.0,
            "error": str(e)
        }


# =============================================================================
# 數據處理輔助
# =============================================================================

def check_dataset_size(dataset, min_samples: int = 100) -> bool:
    """
    檢查數據集大小是否足夠

    Args:
        dataset: 數據集對象
        min_samples: 最小樣本數

    Returns:
        是否足夠
    """
    try:
        dataset_size = len(dataset)

        if dataset_size < min_samples:
            logger.warning(f"⚠️  數據集樣本數過少: {dataset_size} < {min_samples}")
            logger.warning(f"建議: 使用至少 {min_samples} 個樣本以確保訓練效果")
            return False

        logger.info(f"✅ 數據集大小: {dataset_size} 樣本")
        return True

    except Exception as e:
        logger.error(f"檢查數據集時發生錯誤: {e}")
        return False


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    seconds_per_batch: float = 1.0
) -> Dict[str, float]:
    """
    估算訓練時間

    Args:
        num_samples: 樣本總數
        batch_size: 批次大小
        num_epochs: 訓練輪數
        seconds_per_batch: 每批次秒數（經驗值）

    Returns:
        時間估算字典
    """
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * num_epochs
    estimated_seconds = total_steps * seconds_per_batch

    return {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'estimated_minutes': estimated_seconds / 60,
        'estimated_hours': estimated_seconds / 3600
    }


# =============================================================================
# 模型統計
# =============================================================================

def print_trainable_parameters(model, verbose: bool = True) -> Dict[str, int]:
    """
    打印可訓練參數統計（統一格式）

    Args:
        model: 模型對象
        verbose: 是否打印詳細信息

    Returns:
        參數統計字典
    """
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / all_param if all_param > 0 else 0

    stats = {
        'total_params': all_param,
        'trainable_params': trainable_params,
        'frozen_params': all_param - trainable_params,
        'trainable_percent': trainable_percent
    }

    if verbose:
        print("=" * 60)
        print("模型參數統計")
        print("=" * 60)
        print(f"總參數: {all_param:,} ({all_param/1e6:.2f}M)")
        print(f"可訓練參數: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"凍結參數: {stats['frozen_params']:,}")
        print(f"可訓練比例: {trainable_percent:.4f}%")
        print("=" * 60)

    return stats


def get_layer_wise_parameters(model) -> Dict[str, int]:
    """
    獲取各層的參數統計

    Args:
        model: 模型對象

    Returns:
        各層參數字典
    """
    layer_params = {}

    for name, param in model.named_parameters():
        # 提取層名稱（取第一個點之前的部分）
        layer_name = name.split('.')[0]

        if layer_name not in layer_params:
            layer_params[layer_name] = 0

        layer_params[layer_name] += param.numel()

    return layer_params


# =============================================================================
# 訓練狀態監控
# =============================================================================

class TrainingMonitor:
    """訓練過程監控器"""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step = 0
        self.losses = []
        self.metrics = []
        self.gpu_memory = []

    def log_step(self, loss: float, metrics: Optional[Dict] = None):
        """記錄訓練步驟"""
        self.step += 1
        self.losses.append(loss)

        if metrics:
            self.metrics.append(metrics)

        # 記錄 GPU 記憶體
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1e9
            self.gpu_memory.append((self.step, memory_gb))

        # 定期打印
        if self.step % self.log_interval == 0:
            print(f"Step {self.step}: Loss = {loss:.4f}", end="")
            if torch.cuda.is_available():
                print(f" | GPU Memory = {memory_gb:.2f}GB", end="")
            print()

    def get_history(self) -> Dict[str, Any]:
        """獲取訓練歷史"""
        return {
            'losses': self.losses,
            'metrics': self.metrics,
            'gpu_memory': self.gpu_memory,
            'total_steps': self.step
        }

    def reset(self):
        """重置監控器"""
        self.step = 0
        self.losses = []
        self.metrics = []
        self.gpu_memory = []


# =============================================================================
# 錯誤處理包裝器
# =============================================================================

def safe_load_model(model_name: str, model_class, **kwargs):
    """
    安全載入模型（帶錯誤處理）

    Args:
        model_name: 模型名稱或路徑
        model_class: 模型類別
        **kwargs: 其他參數

    Returns:
        模型對象，失敗則返回 None
    """
    try:
        logger.info(f"正在載入模型: {model_name}")
        model = model_class.from_pretrained(model_name, **kwargs)
        logger.info(f"✅ 模型載入成功")
        return model

    except Exception as e:
        logger.error(f"❌ 模型載入失敗: {e}")
        logger.error(f"請檢查:")
        logger.error(f"  1. 模型名稱是否正確")
        logger.error(f"  2. 網絡連接是否正常")
        logger.error(f"  3. Hugging Face token 是否設置 (私有模型)")
        return None


def safe_load_dataset(dataset_name: str, dataset_class, split: str = "train", **kwargs):
    """
    安全載入數據集（帶錯誤處理）

    Args:
        dataset_name: 數據集名稱
        dataset_class: load_dataset 函數
        split: 數據集分割
        **kwargs: 其他參數

    Returns:
        數據集對象，失敗則返回 None
    """
    try:
        logger.info(f"正在載入數據集: {dataset_name} (split: {split})")
        dataset = dataset_class(dataset_name, split=split, **kwargs)
        logger.info(f"✅ 數據集載入成功: {len(dataset)} 樣本")
        return dataset

    except Exception as e:
        logger.error(f"❌ 數據集載入失敗: {e}")
        logger.error(f"請檢查:")
        logger.error(f"  1. 數據集名稱是否正確")
        logger.error(f"  2. 網絡連接是否正常")
        logger.error(f"  3. 是否需要認證")
        return None


# =============================================================================
# 訓練前檢查
# =============================================================================

def pre_training_checklist(
    model,
    train_dataset,
    output_dir: str,
    min_disk_space_gb: float = 10.0
) -> Tuple[bool, List[str]]:
    """
    訓練前檢查清單

    Args:
        model: 模型對象
        train_dataset: 訓練數據集
        output_dir: 輸出目錄
        min_disk_space_gb: 最小磁碟空間（GB）

    Returns:
        (是否通過, 問題列表)
    """
    issues = []

    print("=" * 60)
    print("訓練前檢查")
    print("=" * 60)

    # 1. 檢查模型
    try:
        _ = print_trainable_parameters(model, verbose=False)
        print("✅ 模型檢查通過")
    except Exception as e:
        issues.append(f"模型檢查失敗: {e}")
        print(f"❌ 模型檢查失敗")

    # 2. 檢查數據集
    try:
        dataset_size = len(train_dataset)
        if dataset_size < 100:
            issues.append(f"數據集過小: {dataset_size} < 100")
            print(f"⚠️  數據集樣本數: {dataset_size} (建議 ≥ 100)")
        else:
            print(f"✅ 數據集樣本數: {dataset_size}")
    except Exception as e:
        issues.append(f"數據集檢查失敗: {e}")
        print(f"❌ 數據集檢查失敗")

    # 3. 檢查輸出目錄
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 輸出目錄: {output_dir}")
    except Exception as e:
        issues.append(f"無法創建輸出目錄: {e}")
        print(f"❌ 輸出目錄創建失敗")

    # 4. 檢查磁碟空間
    try:
        import shutil
        stat = shutil.disk_usage(output_path.parent)
        free_gb = stat.free / 1e9

        if free_gb < min_disk_space_gb:
            issues.append(f"磁碟空間不足: {free_gb:.1f}GB < {min_disk_space_gb}GB")
            print(f"⚠️  可用磁碟空間: {free_gb:.1f}GB (建議 ≥ {min_disk_space_gb}GB)")
        else:
            print(f"✅ 可用磁碟空間: {free_gb:.1f}GB")
    except Exception as e:
        logger.warning(f"無法檢查磁碟空間: {e}")

    # 5. 檢查 GPU（如果可用）
    if torch.cuda.is_available():
        try:
            print_gpu_memory_usage("✅ ")
        except Exception as e:
            issues.append(f"GPU 狀態檢查失敗: {e}")

    print("=" * 60)

    is_passed = len(issues) == 0
    if is_passed:
        print("✅ 所有檢查通過，可以開始訓練")
    else:
        print(f"⚠️  發現 {len(issues)} 個問題，請檢查後再訓練")

    return is_passed, issues


# =============================================================================
# 訓練完成後分析
# =============================================================================

def analyze_training_results(
    trainer,
    output_dir: str,
    save_visualizations: bool = True
) -> Dict[str, Any]:
    """
    分析訓練結果並生成報告

    Args:
        trainer: Hugging Face Trainer 對象
        output_dir: 輸出目錄
        save_visualizations: 是否保存視覺化

    Returns:
        分析結果字典
    """
    results = {}

    try:
        # 1. 提取訓練歷史
        log_history = trainer.state.log_history

        train_loss = [entry.get('loss') for entry in log_history if 'loss' in entry]
        eval_loss = [entry.get('eval_loss') for entry in log_history if 'eval_loss' in entry]

        results['train_loss'] = train_loss
        results['eval_loss'] = eval_loss
        results['best_eval_loss'] = min(eval_loss) if eval_loss else None

        # 2. 訓練時間
        if hasattr(trainer.state, 'log_history') and len(log_history) > 0:
            if 'train_runtime' in log_history[-1]:
                results['total_time_seconds'] = log_history[-1]['train_runtime']
                results['total_time_minutes'] = results['total_time_seconds'] / 60

        # 3. 生成視覺化
        if save_visualizations:
            from .visualization import plot_training_curves

            viz_dir = Path(output_dir) / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            history = {'train_loss': train_loss, 'eval_loss': eval_loss}
            plot_training_curves(history,
                               save_path=str(viz_dir / "training_curves.png"))

        # 4. 生成摘要
        print("\n" + "=" * 60)
        print("訓練結果摘要")
        print("=" * 60)
        print(f"訓練輪數: {len(train_loss)}")
        if results.get('best_eval_loss'):
            print(f"最佳驗證損失: {results['best_eval_loss']:.4f}")
        if results.get('total_time_minutes'):
            print(f"總訓練時間: {results['total_time_minutes']:.1f} 分鐘")
        print("=" * 60)

        return results

    except Exception as e:
        logger.error(f"分析訓練結果時發生錯誤: {e}")
        return results


# =============================================================================
# 工具函數
# =============================================================================

def create_output_structure(base_dir: str) -> Dict[str, Path]:
    """
    創建標準化的輸出目錄結構

    Args:
        base_dir: 基礎目錄

    Returns:
        目錄路徑字典
    """
    base_path = Path(base_dir)

    dirs = {
        'checkpoints': base_path / "checkpoints",
        'logs': base_path / "logs",
        'visualizations': base_path / "visualizations",
        'metrics': base_path / "metrics"
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ 創建目錄: {path}")

    return dirs


if __name__ == "__main__":
    # 測試
    print("=" * 60)
    print("Training Helpers 模組測試")
    print("=" * 60)

    # 測試 GPU 檢查
    gpu_info = check_gpu_availability()

    # 測試配置驗證
    test_config = {
        'learning_rate': 5e-5,
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 4
    }

    is_valid, warnings = validate_training_config(test_config)
    print(f"\n配置驗證: {'✅ 通過' if is_valid else '❌ 失敗'}")
    if warnings:
        for w in warnings:
            print(f"  {w}")

    print("\n✅ Training helpers 模組測試完成")
