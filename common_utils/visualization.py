"""
Visualization Utilities for PEFT Training Experiments
視覺化工具模組 - 用於 PEFT 實驗室訓練過程與結果展示

提供統一的繪圖函數，確保所有實驗室的視覺化風格一致。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path

# =============================================================================
# 全局配置
# =============================================================================

# 設定中文字體（支援繁體中文）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 設定圖表風格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# 統一配色方案
PEFT_COLORS = {
    'LoRA': '#FF6B6B',
    'QLoRA': '#FF8E8E',
    'Adapter': '#4ECDC4',
    'IA3': '#45B7D1',
    'Prefix': '#FFA07A',
    'Prompt': '#98D8C8',
    'BitFit': '#F7DC6F',
    'P-Tuning': '#BB8FCE',
    'P-Tuning-v2': '#9B59B6',
    'train': '#3498db',
    'eval': '#e74c3c',
    'test': '#2ecc71'
}


# =============================================================================
# 訓練過程視覺化
# =============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "訓練過程"
) -> None:
    """
    繪製訓練與評估曲線

    Args:
        history: 訓練歷史字典，例如:
            {
                'train_loss': [3.2, 2.8, 2.5, ...],
                'eval_loss': [3.5, 3.0, 2.7, ...],
                'eval_perplexity': [24.5, 20.1, 17.3, ...]
            }
        metrics: 要繪製的指標列表，None 表示自動檢測
        save_path: 圖片保存路徑（可選）
        title: 圖表標題

    Example:
        >>> history = trainer.state.log_history
        >>> plot_training_curves(history)
    """
    if metrics is None:
        # 自動檢測可用指標
        metrics = [k for k in history.keys() if 'loss' in k or 'perplexity' in k.lower()]

    num_metrics = len(metrics)
    if num_metrics == 0:
        print("⚠️ 沒有可繪製的指標")
        return

    # 決定子圖布局
    if num_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    elif num_metrics == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        ncols = 2
        nrows = (num_metrics + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        axes = axes.flatten()

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 繪製每個指標
    for idx, metric in enumerate(metrics):
        if metric not in history:
            continue

        values = history[metric]
        epochs = range(1, len(values) + 1)

        # 選擇顏色
        color = PEFT_COLORS.get('train' if 'train' in metric else 'eval', '#3498db')

        axes[idx].plot(epochs, values, marker='o', linewidth=2,
                      markersize=6, color=color, label=metric)

        # 標題和標籤
        axes[idx].set_title(metric.replace('_', ' ').title(),
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epoch', fontsize=11)
        axes[idx].set_ylabel(metric.split('_')[-1].title(), fontsize=11)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

        # 添加數值標註（僅在數據點少於20時）
        if len(values) <= 20:
            for x, y in zip(epochs, values):
                if idx == 0 or 'loss' in metric:  # 只在 loss 圖上標註
                    axes[idx].text(x, y, f'{y:.3f}', fontsize=8,
                                  ha='center', va='bottom')

    # 隱藏多餘的子圖
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 圖表已保存至: {save_path}")

    plt.show()


def plot_loss_comparison(
    losses_dict: Dict[str, List[float]],
    labels: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    對比多個模型/方法的損失曲線

    Args:
        losses_dict: 損失字典，例如:
            {
                'baseline': [3.2, 2.8, 2.5],
                'lora_r8': [3.1, 2.6, 2.3],
                'lora_r16': [3.0, 2.5, 2.2]
            }
        labels: 標籤字典（可選）
        save_path: 保存路徑

    Example:
        >>> losses = {
        ...     'LoRA r=8': history1['train_loss'],
        ...     'LoRA r=16': history2['train_loss']
        ... }
        >>> plot_loss_comparison(losses)
    """
    plt.figure(figsize=(12, 6))

    for idx, (name, losses) in enumerate(losses_dict.items()):
        epochs = range(1, len(losses) + 1)
        label = labels.get(name, name) if labels else name

        plt.plot(epochs, losses, marker='o', linewidth=2.5,
                markersize=7, label=label, alpha=0.8)

    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('訓練損失對比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# =============================================================================
# PEFT 方法對比視覺化
# =============================================================================

def plot_peft_comparison(
    results: List[Dict[str, Any]],
    metrics: List[str] = ['trainable_params_%', 'performance', 'training_time'],
    save_path: Optional[str] = None
) -> None:
    """
    繪製不同 PEFT 方法的對比圖（條形圖或雷達圖）

    Args:
        results: 結果列表，例如:
            [
                {'method': 'LoRA', 'trainable_params_%': 0.5,
                 'performance': 85.2, 'training_time': 120},
                {'method': 'Adapter', 'trainable_params_%': 2.0,
                 'performance': 86.1, 'training_time': 150},
                ...
            ]
        metrics: 要比較的指標
        save_path: 保存路徑
    """
    df = pd.DataFrame(results)
    methods = df['method'].tolist()

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    fig.suptitle('PEFT 方法對比', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        if metric not in df.columns:
            continue

        values = df[metric].tolist()
        colors = [PEFT_COLORS.get(method, '#95a5a6') for method in methods]

        bars = axes[idx].bar(range(len(methods)), values, color=colors, alpha=0.8)
        axes[idx].set_xticks(range(len(methods)))
        axes[idx].set_xticklabels(methods, rotation=45, ha='right')
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        axes[idx].set_title(metric.replace('_', ' ').title(),
                           fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

        # 添加數值標籤
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.2f}',
                          ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_parameter_efficiency(
    methods_data: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    繪製參數效率對比圖（可訓練參數 vs 性能）

    Args:
        methods_data: 方法數據，例如:
            [
                {'method': 'LoRA', 'params_%': 0.5, 'accuracy': 85.2},
                {'method': 'Full FT', 'params_%': 100.0, 'accuracy': 87.1},
                ...
            ]
        save_path: 保存路徑
    """
    df = pd.DataFrame(methods_data)

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, row in df.iterrows():
        method = row['method']
        color = PEFT_COLORS.get(method, '#95a5a6')

        ax.scatter(row['params_%'], row.get('accuracy', row.get('performance', 0)),
                  s=300, color=color, alpha=0.7, edgecolors='black', linewidth=1.5,
                  label=method)

        # 添加方法標籤
        ax.text(row['params_%'], row.get('accuracy', row.get('performance', 0)),
               f"  {method}", fontsize=10, va='center')

    ax.set_xlabel('可訓練參數比例 (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('性能指標', fontsize=12, fontweight='bold')
    ax.set_title('參數效率 vs 性能', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# =============================================================================
# 模型分析視覺化
# =============================================================================

def plot_parameter_distribution(
    model_stats: Dict[str, int],
    save_path: Optional[str] = None
) -> None:
    """
    視覺化模型參數的可訓練/凍結分佈

    Args:
        model_stats: 參數統計，例如:
            {
                'total_params': 7000000000,
                'trainable_params': 4194304,
                'frozen_params': 6995805696
            }
        save_path: 保存路徑
    """
    labels = ['可訓練參數', '凍結參數']
    sizes = [model_stats.get('trainable_params', 0),
             model_stats.get('frozen_params', 0)]
    colors = ['#2ecc71', '#95a5a6']
    explode = (0.1, 0)  # 突出可訓練參數

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 圓餅圖
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('參數分佈', fontsize=14, fontweight='bold')

    # 條形圖（百萬為單位）
    sizes_m = [s / 1e6 for s in sizes]
    bars = ax2.bar(labels, sizes_m, color=colors, alpha=0.8)
    ax2.set_ylabel('參數數量 (百萬)', fontsize=12, fontweight='bold')
    ax2.set_title('參數數量對比', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 添加數值標籤
    for bar, size in zip(bars, sizes_m):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.1f}M',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加總參數信息
    total_params = model_stats.get('total_params', sum(sizes))
    trainable_pct = (sizes[0] / total_params * 100) if total_params > 0 else 0

    fig.text(0.5, 0.02,
            f'總參數: {total_params/1e6:.1f}M | 可訓練: {trainable_pct:.3f}%',
            ha='center', fontsize=11, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_layer_wise_parameters(
    layer_params: Dict[str, int],
    save_path: Optional[str] = None
) -> None:
    """
    視覺化各層的參數分佈（堆疊條形圖）

    Args:
        layer_params: 各層參數字典，例如:
            {
                'embeddings': 50000,
                'layer_0': 12000,
                'layer_1': 12000,
                ...
            }
        save_path: 保存路徑
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = list(layer_params.keys())
    params = [layer_params[layer] / 1e6 for layer in layers]  # 轉為百萬

    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    bars = ax.barh(layers, params, color=colors, alpha=0.8)

    ax.set_xlabel('參數數量 (百萬)', fontsize=12, fontweight='bold')
    ax.set_title('各層參數分佈', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 添加數值標籤
    for bar, param in zip(bars, params):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{param:.2f}M',
               ha='left', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# =============================================================================
# 推理性能視覺化
# =============================================================================

def plot_inference_benchmark(
    benchmark_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """
    繪製推理性能基準測試結果

    Args:
        benchmark_results: 基準測試結果，例如:
            {
                'Base Model': {'latency_ms': 45.2, 'throughput': 22.1, 'memory_gb': 14.2},
                'LoRA': {'latency_ms': 46.1, 'throughput': 21.7, 'memory_gb': 14.5},
                'IA3': {'latency_ms': 45.5, 'throughput': 22.0, 'memory_gb': 14.3}
            }
        save_path: 保存路徑
    """
    methods = list(benchmark_results.keys())
    metrics = ['latency_ms', 'throughput', 'memory_gb']
    metric_names = ['延遲 (ms)', '吞吐量 (tokens/s)', '記憶體 (GB)']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('推理性能基準測試', fontsize=16, fontweight='bold')

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [benchmark_results[method].get(metric, 0) for method in methods]
        colors = [PEFT_COLORS.get(method, '#95a5a6') for method in methods]

        bars = axes[idx].bar(range(len(methods)), values, color=colors, alpha=0.8)
        axes[idx].set_xticks(range(len(methods)))
        axes[idx].set_xticklabels(methods, rotation=45, ha='right')
        axes[idx].set_ylabel(name, fontsize=11)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

        # 數值標籤
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.1f}',
                          ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# =============================================================================
# 學習曲線與收斂分析
# =============================================================================

def plot_learning_curves(
    train_sizes: List[int],
    train_scores: List[float],
    val_scores: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    繪製學習曲線（訓練集大小 vs 性能）

    Args:
        train_sizes: 訓練集大小列表
        train_scores: 訓練集性能
        val_scores: 驗證集性能
        save_path: 保存路徑
    """
    plt.figure(figsize=(10, 6))

    plt.plot(train_sizes, train_scores, marker='o', linewidth=2.5,
            markersize=8, label='訓練集', color=PEFT_COLORS['train'])
    plt.plot(train_sizes, val_scores, marker='s', linewidth=2.5,
            markersize=8, label='驗證集', color=PEFT_COLORS['eval'])

    plt.fill_between(train_sizes, train_scores, val_scores,
                     alpha=0.2, color='gray')

    plt.xlabel('訓練樣本數', fontsize=12, fontweight='bold')
    plt.ylabel('性能指標', fontsize=12, fontweight='bold')
    plt.title('學習曲線', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_convergence_analysis(
    losses: List[float],
    window_size: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    繪製收斂分析圖（原始 loss + 移動平均）

    Args:
        losses: 損失值列表
        window_size: 移動平均窗口大小
        save_path: 保存路徑
    """
    steps = range(1, len(losses) + 1)

    # 計算移動平均
    if len(losses) >= window_size:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        moving_avg_steps = range(window_size, len(losses) + 1)
    else:
        moving_avg = losses
        moving_avg_steps = steps

    fig, ax = plt.subplots(figsize=(12, 6))

    # 原始損失（半透明）
    ax.plot(steps, losses, alpha=0.3, color='#3498db', label='原始 Loss')

    # 移動平均（醒目）
    ax.plot(moving_avg_steps, moving_avg, linewidth=2.5,
           color='#e74c3c', label=f'移動平均 (window={window_size})')

    ax.set_xlabel('訓練步數', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('訓練收斂分析', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 添加收斂指標
    final_avg = np.mean(losses[-window_size:]) if len(losses) >= window_size else np.mean(losses)
    initial_avg = np.mean(losses[:window_size]) if len(losses) >= window_size else losses[0]
    improvement = ((initial_avg - final_avg) / initial_avg) * 100

    ax.text(0.98, 0.98,
           f'初始 Loss: {initial_avg:.4f}\n最終 Loss: {final_avg:.4f}\n改善: {improvement:.1f}%',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# =============================================================================
# 記憶體與資源監控視覺化
# =============================================================================

def plot_memory_usage(
    memory_timeline: List[Tuple[int, float]],
    events: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    繪製記憶體使用時間線

    Args:
        memory_timeline: (step, memory_gb) 列表
        events: 特殊事件標記，例如 {100: 'Checkpoint 保存'}
        save_path: 保存路徑
    """
    steps, memory = zip(*memory_timeline)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(steps, memory, linewidth=2, color='#9b59b6', marker='o', markersize=4)
    ax.fill_between(steps, 0, memory, alpha=0.3, color='#9b59b6')

    # 標記特殊事件
    if events:
        for step, event in events.items():
            if step in steps:
                idx = steps.index(step)
                ax.axvline(x=step, color='red', linestyle='--', alpha=0.7)
                ax.text(step, memory[idx], f'  {event}',
                       rotation=90, va='bottom', fontsize=9)

    ax.set_xlabel('訓練步數', fontsize=12, fontweight='bold')
    ax.set_ylabel('GPU 記憶體 (GB)', fontsize=12, fontweight='bold')
    ax.set_title('GPU 記憶體使用時間線', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # 添加統計信息
    peak_memory = max(memory)
    avg_memory = np.mean(memory)
    ax.text(0.98, 0.98,
           f'峰值: {peak_memory:.2f} GB\n平均: {avg_memory:.2f} GB',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# =============================================================================
# 工具函數
# =============================================================================

def save_all_figures(output_dir: str) -> None:
    """
    保存所有已開啟的圖表到指定目錄

    Args:
        output_dir: 輸出目錄路徑
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figs = [plt.figure(n) for n in plt.get_fignums()]

    for idx, fig in enumerate(figs, 1):
        save_path = output_path / f'figure_{idx}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 保存圖表 {idx}: {save_path}")


def create_comparison_table(
    results: List[Dict[str, Any]],
    columns: List[str],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    創建對比表格並可視化（熱力圖）

    Args:
        results: 結果列表
        columns: 要顯示的欄位
        save_path: 保存路徑

    Returns:
        DataFrame 對象
    """
    df = pd.DataFrame(results)[columns]

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 2))

    # 創建熱力圖
    sns.heatmap(df.select_dtypes(include=[np.number]),
               annot=True, fmt='.2f', cmap='YlGnBu',
               cbar_kws={'label': '數值'},
               ax=ax)

    ax.set_title('PEFT 方法性能對比熱力圖', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return df


# =============================================================================
# 實用示例函數
# =============================================================================

def quick_plot_trainer_history(trainer, save_dir: Optional[str] = None):
    """
    快速繪製 Hugging Face Trainer 的訓練歷史

    Args:
        trainer: Hugging Face Trainer 對象
        save_dir: 保存目錄（可選）

    Example:
        >>> from transformers import Trainer
        >>> trainer = Trainer(...)
        >>> trainer.train()
        >>> quick_plot_trainer_history(trainer)
    """
    log_history = trainer.state.log_history

    # 提取損失值
    train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
    eval_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]

    # 提取其他指標
    eval_metrics = {}
    for entry in log_history:
        for key in entry:
            if key.startswith('eval_') and key != 'eval_loss':
                if key not in eval_metrics:
                    eval_metrics[key] = []
                eval_metrics[key].append(entry[key])

    # 繪製
    history = {'train_loss': train_loss, 'eval_loss': eval_loss}
    history.update(eval_metrics)

    plot_training_curves(history, save_path=save_dir)


if __name__ == "__main__":
    # 測試示例
    print("✅ Visualization utilities loaded successfully")
    print(f"可用函數: {', '.join([name for name in dir() if not name.startswith('_')])}")

    # 測試繪圖
    test_history = {
        'train_loss': [3.2, 2.8, 2.5, 2.3, 2.1, 2.0],
        'eval_loss': [3.5, 3.0, 2.7, 2.6, 2.5, 2.4],
        'eval_perplexity': [33.1, 20.1, 14.9, 13.5, 12.2, 11.0]
    }

    print("\n測試繪圖功能...")
    # plot_training_curves(test_history, title="測試訓練曲線")
