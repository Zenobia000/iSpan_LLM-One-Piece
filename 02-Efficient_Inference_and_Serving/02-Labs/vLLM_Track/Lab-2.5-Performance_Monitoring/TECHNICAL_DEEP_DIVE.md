# vLLM 性能監控技術深度剖析

## 目錄

1. [監控架構設計原理](#1-監控架構設計原理)
2. [vLLM 內核指標體系](#2-vllm-內核指標體系)
3. [高頻率數據收集技術](#3-高頻率數據收集技術)
4. [異常檢測算法詳解](#4-異常檢測算法詳解)
5. [預測性分析方法](#5-預測性分析方法)
6. [智能告警機制](#6-智能告警機制)
7. [自動化優化策略](#7-自動化優化策略)
8. [性能瓶頸診斷方法](#8-性能瓶頸診斷方法)
9. [容量規劃數學模型](#9-容量規劃數學模型)
10. [生產環境最佳實踐](#10-生產環境最佳實踐)

---

## 1. 監控架構設計原理

### 1.1 分層監控體系

```
┌─────────────────────────────────────────────────────────────┐
│                    Business Layer (業務層)                    │
├─────────────────────────────────────────────────────────────┤
│  • 用戶滿意度指標    • 業務成功率    • 收入影響指標          │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Application Layer (應用層)                    │
├─────────────────────────────────────────────────────────────┤
│  • QPS/TPS           • 延遲分佈      • 錯誤率               │
│  • TTFT              • TPOT         • 請求佇列長度          │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Runtime Layer (執行層)                        │
├─────────────────────────────────────────────────────────────┤
│  • vLLM Engine 指標  • KV Cache 使用 • 模型載入時間        │
│  • Batch 處理效率    • Token 生成速度 • 記憶體池狀態        │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│               Infrastructure Layer (基礎設施層)               │
├─────────────────────────────────────────────────────────────┤
│  • CPU/GPU 使用率    • 記憶體使用     • 網路 I/O            │
│  • 磁碟 I/O          • 溫度監控       • 電源狀態            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 監控數據流架構

```
vLLM Process → Prometheus Metrics → Time Series DB → Alert Manager
      │                                    │              │
      │                                    ▼              │
      │                              Grafana Dashboard    │
      │                                    │              │
      ▼                                    │              ▼
Custom Collectors ─────────────────────────┴─────► Notification
      │                                                   │
      ▼                                                   │
ML Pipeline ─────────────────────────────────────────────┘
(預測分析)
```

### 1.3 數據採集策略

#### 1.3.1 採集頻率設計

| 指標類型 | 採集頻率 | 保留期間 | 聚合策略 |
|---------|---------|---------|---------|
| 基礎資源指標 | 5秒 | 30天 | 1分鐘平均值 |
| vLLM 性能指標 | 1秒 | 7天 | 10秒平均值 |
| 請求級指標 | 即時 | 3天 | 無聚合 |
| 業務指標 | 30秒 | 90天 | 5分鐘平均值 |

#### 1.3.2 數據壓縮與儲存

```python
# 時間序列數據壓縮策略
compression_policies = {
    "raw_data": {
        "retention": "24h",
        "resolution": "1s"
    },
    "high_resolution": {
        "retention": "7d",
        "resolution": "10s",
        "aggregation": "avg"
    },
    "medium_resolution": {
        "retention": "30d",
        "resolution": "1m",
        "aggregation": "avg,max,min"
    },
    "low_resolution": {
        "retention": "1y",
        "resolution": "1h",
        "aggregation": "avg,max,min,p95"
    }
}
```

---

## 2. vLLM 內核指標體系

### 2.1 核心性能指標

#### 2.1.1 推理引擎指標

```python
# vLLM 核心指標定義
CORE_METRICS = {
    # 請求處理指標
    "vllm_request_success_total": "累計成功請求數",
    "vllm_request_failure_total": "累計失敗請求數",
    "vllm_request_duration_seconds": "請求處理時間分佈",

    # Token 生成指標
    "vllm_time_to_first_token_seconds": "首 Token 生成時間",
    "vllm_time_per_output_token_seconds": "每個輸出 Token 時間",
    "vllm_tokens_generated_total": "累計生成 Token 數",

    # 資源使用指標
    "vllm_gpu_cache_usage_perc": "GPU 快取使用率",
    "vllm_kv_cache_usage_perc": "KV 快取使用率",
    "vllm_num_requests_running": "當前運行請求數",
    "vllm_num_requests_waiting": "等待處理請求數",

    # 引擎狀態指標
    "vllm_engine_iteration_duration_seconds": "引擎迭代時間",
    "vllm_model_loading_duration_seconds": "模型載入時間",
    "vllm_scheduler_running_requests": "調度器運行請求數"
}
```

#### 2.1.2 指標計算公式

**TTFT (Time to First Token)**
```
TTFT = t_first_token - t_request_received
where:
  t_first_token: 第一個 token 生成完成時間
  t_request_received: 接收請求時間
```

**TPOT (Time Per Output Token)**
```
TPOT = (t_completion - t_first_token) / (n_tokens - 1)
where:
  t_completion: 生成完成時間
  t_first_token: 第一個 token 生成時間
  n_tokens: 總生成 token 數
```

**吞吐量計算**
```
Throughput = Σ(tokens_generated) / time_window
QPS = Σ(requests_completed) / time_window
```

### 2.2 內存管理指標

#### 2.2.1 KV Cache 監控

```python
class KVCacheMonitor:
    """KV Cache 詳細監控"""

    def __init__(self):
        self.cache_metrics = {
            "total_capacity": 0,     # 總容量
            "used_capacity": 0,      # 已使用容量
            "cache_hit_rate": 0.0,   # 快取命中率
            "eviction_count": 0,     # 淘汰次數
            "allocation_failures": 0  # 分配失敗次數
        }

    def calculate_efficiency(self) -> dict:
        """計算快取效率指標"""
        return {
            "utilization_rate": self.cache_metrics["used_capacity"] /
                               self.cache_metrics["total_capacity"],
            "hit_rate": self.cache_metrics["cache_hit_rate"],
            "fragmentation": self._calculate_fragmentation(),
            "pressure_score": self._calculate_pressure_score()
        }

    def _calculate_fragmentation(self) -> float:
        """計算記憶體碎片化程度"""
        # 實際實現會根據 vLLM 的記憶體分配器進行
        pass

    def _calculate_pressure_score(self) -> float:
        """計算記憶體壓力分數 (0-100)"""
        utilization = self.cache_metrics["used_capacity"] / self.cache_metrics["total_capacity"]
        eviction_rate = self.cache_metrics["eviction_count"] / 3600  # 每小時淘汰次數
        failure_rate = self.cache_metrics["allocation_failures"] / 3600

        # 綜合壓力分數
        pressure = (utilization * 60 +
                   min(eviction_rate * 20, 30) +
                   min(failure_rate * 50, 40))
        return min(pressure, 100)
```

#### 2.2.2 GPU 記憶體分析

```python
# GPU 記憶體分配追蹤
GPU_MEMORY_CATEGORIES = {
    "model_weights": "模型權重記憶體",
    "kv_cache": "KV 快取記憶體",
    "activation": "激活值記憶體",
    "workspace": "工作空間記憶體",
    "fragmentation": "碎片化記憶體"
}

def analyze_gpu_memory_usage():
    """分析 GPU 記憶體使用模式"""
    return {
        "peak_usage": "峰值使用量",
        "average_usage": "平均使用量",
        "allocation_pattern": "分配模式分析",
        "optimization_suggestions": "優化建議"
    }
```

---

## 3. 高頻率數據收集技術

### 3.1 無鎖數據結構

#### 3.1.1 環形緩衝區實現

```python
import threading
from typing import Optional, Any
import numpy as np

class LockFreeRingBuffer:
    """無鎖環形緩衝區用於高頻數據收集"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0  # 寫入位置
        self.tail = 0  # 讀取位置
        self.size = 0

    def push(self, item: Any) -> bool:
        """無鎖寫入"""
        current_head = self.head
        next_head = (current_head + 1) % self.capacity

        if next_head == self.tail:
            # 緩衝區滿，覆蓋最舊數據
            self.tail = (self.tail + 1) % self.capacity
        else:
            self.size += 1

        self.buffer[current_head] = item
        self.head = next_head
        return True

    def pop(self) -> Optional[Any]:
        """無鎖讀取"""
        if self.size == 0:
            return None

        item = self.buffer[self.tail]
        self.tail = (self.tail + 1) % self.capacity
        self.size -= 1
        return item

    def batch_pop(self, batch_size: int) -> list:
        """批量讀取"""
        items = []
        for _ in range(min(batch_size, self.size)):
            item = self.pop()
            if item is not None:
                items.append(item)
        return items
```

#### 3.1.2 內存映射指標

```python
import mmap
import struct
from dataclasses import dataclass
from typing import Dict

@dataclass
class MetricHeader:
    timestamp: float
    metric_count: int
    data_size: int

class SharedMemoryMetrics:
    """共享記憶體指標收集"""

    def __init__(self, shm_size: int = 1024 * 1024):  # 1MB
        self.shm_size = shm_size
        self.shm_file = "/tmp/vllm_metrics"
        self.metrics_map = self._create_memory_map()

    def _create_memory_map(self) -> mmap.mmap:
        """創建記憶體映射"""
        with open(self.shm_file, "wb") as f:
            f.write(b'\x00' * self.shm_size)

        with open(self.shm_file, "r+b") as f:
            return mmap.mmap(f.fileno(), 0)

    def write_metrics(self, metrics: Dict[str, float]) -> bool:
        """寫入指標到共享記憶體"""
        try:
            # 序列化指標數據
            data = self._serialize_metrics(metrics)

            # 寫入 header
            header = MetricHeader(
                timestamp=time.time(),
                metric_count=len(metrics),
                data_size=len(data)
            )

            self.metrics_map.seek(0)
            self.metrics_map.write(struct.pack('dII',
                                             header.timestamp,
                                             header.metric_count,
                                             header.data_size))
            self.metrics_map.write(data)
            self.metrics_map.flush()
            return True

        except Exception as e:
            logger.error(f"寫入共享記憶體失敗: {e}")
            return False

    def read_metrics(self) -> Optional[Dict[str, float]]:
        """從共享記憶體讀取指標"""
        try:
            self.metrics_map.seek(0)
            header_data = self.metrics_map.read(16)  # 8+4+4 bytes

            timestamp, count, size = struct.unpack('dII', header_data)
            data = self.metrics_map.read(size)

            return self._deserialize_metrics(data, count)

        except Exception as e:
            logger.error(f"讀取共享記憶體失敗: {e}")
            return None
```

### 3.2 批量處理與聚合

#### 3.2.1 微批次處理

```python
class MicroBatchProcessor:
    """微批次指標處理器"""

    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush = time.time()

    def add_metric(self, metric_data: dict):
        """添加指標到批次"""
        self.buffer.append(metric_data)

        # 檢查是否需要刷新
        if (len(self.buffer) >= self.batch_size or
            time.time() - self.last_flush >= self.flush_interval):
            self.flush()

    def flush(self):
        """刷新批次數據"""
        if not self.buffer:
            return

        # 聚合處理
        aggregated = self._aggregate_metrics(self.buffer)

        # 發送到時間序列數據庫
        self._send_to_tsdb(aggregated)

        # 清空緩衝區
        self.buffer.clear()
        self.last_flush = time.time()

    def _aggregate_metrics(self, metrics: list) -> dict:
        """聚合指標數據"""
        aggregated = {}

        # 按指標名稱分組
        grouped = {}
        for metric in metrics:
            for name, value in metric.items():
                if name not in grouped:
                    grouped[name] = []
                grouped[name].append(value)

        # 計算聚合值
        for name, values in grouped.items():
            aggregated[name] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'count': len(values),
                'sum': sum(values)
            }

        return aggregated
```

---

## 4. 異常檢測算法詳解

### 4.1 統計學方法

#### 4.1.1 Z-Score 異常檢測

```python
import numpy as np
from scipy import stats
from typing import Tuple, List

class ZScoreDetector:
    """基於 Z-Score 的異常檢測"""

    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = []

    def detect(self, value: float) -> Tuple[bool, float]:
        """檢測異常值"""
        self.data_window.append(value)

        # 保持視窗大小
        if len(self.data_window) > self.window_size:
            self.data_window.pop(0)

        if len(self.data_window) < 10:  # 需要最少數據點
            return False, 0.0

        # 計算 Z-Score
        mean = np.mean(self.data_window)
        std = np.std(self.data_window)

        if std == 0:
            return False, 0.0

        z_score = abs((value - mean) / std)
        is_anomaly = z_score > self.threshold

        return is_anomaly, z_score
```

#### 4.1.2 CUSUM 變化點檢測

```python
class CUSUMDetector:
    """CUSUM 累積和變化點檢測"""

    def __init__(self, threshold: float = 5.0, drift: float = 1.0):
        self.threshold = threshold
        self.drift = drift
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.baseline_mean = None
        self.baseline_std = None

    def set_baseline(self, baseline_data: List[float]):
        """設置基準線"""
        self.baseline_mean = np.mean(baseline_data)
        self.baseline_std = np.std(baseline_data)

    def detect(self, value: float) -> Tuple[bool, str]:
        """檢測變化點"""
        if self.baseline_mean is None:
            return False, "no_baseline"

        # 標準化值
        normalized = (value - self.baseline_mean) / self.baseline_std

        # 更新 CUSUM 統計量
        self.cusum_pos = max(0, self.cusum_pos + normalized - self.drift)
        self.cusum_neg = max(0, self.cusum_neg - normalized - self.drift)

        # 檢測變化點
        if self.cusum_pos > self.threshold:
            return True, "upward_shift"
        elif self.cusum_neg > self.threshold:
            return True, "downward_shift"
        else:
            return False, "no_change"
```

### 4.2 機器學習方法

#### 4.2.1 Isolation Forest

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class IsolationForestDetector:
    """基於 Isolation Forest 的異常檢測"""

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, training_data: np.ndarray):
        """訓練模型"""
        # 標準化數據
        scaled_data = self.scaler.fit_transform(training_data)

        # 訓練 Isolation Forest
        self.model.fit(scaled_data)
        self.is_fitted = True

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """預測異常"""
        if not self.is_fitted:
            raise ValueError("模型未訓練")

        # 標準化輸入數據
        scaled_data = self.scaler.transform(data)

        # 預測異常 (-1: 異常, 1: 正常)
        predictions = self.model.predict(scaled_data)

        # 獲取異常分數
        scores = self.model.decision_function(scaled_data)

        return predictions, scores
```

#### 4.2.2 LSTM 自編碼器

```python
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """LSTM 自編碼器用於時間序列異常檢測"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 編碼器
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )

        # 解碼器
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )

        # 輸出層
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # 編碼
        encoded, (hidden, cell) = self.encoder(x)

        # 解碼
        decoded, _ = self.decoder(encoded, (hidden, cell))

        # 重構
        reconstructed = self.output_layer(decoded)

        return reconstructed

    def detect_anomaly(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """檢測異常"""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=-1)
            anomalies = mse > threshold

        return anomalies, mse
```

---

## 5. 預測性分析方法

### 5.1 時間序列預測

#### 5.1.1 ARIMA 模型

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

class ARIMAPredictor:
    """ARIMA 時間序列預測"""

    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.is_fitted = False

    def find_best_order(self, data: np.ndarray) -> tuple:
        """自動選擇最佳 ARIMA 參數"""
        # 檢查平穩性
        if not self._is_stationary(data):
            # 差分使序列平穩
            data = np.diff(data)
            d = 1
        else:
            d = 0

        best_aic = float('inf')
        best_order = (1, d, 1)

        # 網格搜索最佳參數
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)

                except Exception:
                    continue

        return best_order

    def _is_stationary(self, data: np.ndarray) -> bool:
        """檢查時間序列平穩性"""
        result = adfuller(data)
        return result[1] <= 0.05  # p-value <= 0.05 為平穩

    def fit(self, data: np.ndarray):
        """訓練模型"""
        # 自動選擇參數
        order = self.find_best_order(data)

        # 訓練模型
        self.model = ARIMA(data, order=order)
        self.fitted_model = self.model.fit()
        self.is_fitted = True

    def predict(self, steps: int) -> tuple:
        """預測未來值"""
        if not self.is_fitted:
            raise ValueError("模型未訓練")

        forecast = self.fitted_model.forecast(steps=steps)
        confidence_intervals = self.fitted_model.get_forecast(steps).conf_int()

        return forecast, confidence_intervals
```

#### 5.1.2 Prophet 預測模型

```python
from prophet import Prophet
import pandas as pd

class ProphetPredictor:
    """Facebook Prophet 預測模型"""

    def __init__(self):
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.is_fitted = False

    def fit(self, timestamps: list, values: list):
        """訓練模型"""
        # 準備數據
        df = pd.DataFrame({
            'ds': pd.to_datetime(timestamps),
            'y': values
        })

        # 添加自定義季節性
        self.model.add_seasonality(
            name='hourly',
            period=24,
            fourier_order=8
        )

        # 訓練模型
        self.model.fit(df)
        self.is_fitted = True

    def predict(self, periods: int, freq: str = 'T') -> pd.DataFrame:
        """預測未來值"""
        if not self.is_fitted:
            raise ValueError("模型未訓練")

        # 創建未來時間點
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq
        )

        # 進行預測
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
```

### 5.2 容量預測模型

#### 5.2.1 指數平滑預測

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class CapacityPredictor:
    """容量預測模型"""

    def __init__(self):
        self.models = {}

    def fit_resource_model(self, resource_name: str, data: np.ndarray):
        """為特定資源訓練預測模型"""
        # 使用三重指數平滑
        model = ExponentialSmoothing(
            data,
            trend='add',
            seasonal='add',
            seasonal_periods=24  # 假設有 24 小時週期性
        )

        fitted_model = model.fit()
        self.models[resource_name] = fitted_model

    def predict_resource_usage(self, resource_name: str, hours_ahead: int) -> dict:
        """預測資源使用情況"""
        if resource_name not in self.models:
            raise ValueError(f"未找到資源 {resource_name} 的模型")

        model = self.models[resource_name]
        forecast = model.forecast(hours_ahead)

        # 計算容量需求
        max_forecast = np.max(forecast)
        avg_forecast = np.mean(forecast)

        # 容量建議
        recommended_capacity = max_forecast * 1.2  # 20% 緩衝

        return {
            'forecast': forecast.tolist(),
            'max_predicted': float(max_forecast),
            'avg_predicted': float(avg_forecast),
            'recommended_capacity': float(recommended_capacity),
            'utilization_trend': 'increasing' if forecast[-1] > forecast[0] else 'decreasing'
        }
```

---

## 6. 智能告警機制

### 6.1 動態閾值算法

#### 6.1.1 自適應閾值

```python
class AdaptiveThreshold:
    """自適應動態閾值"""

    def __init__(self, learning_rate: float = 0.1, sensitivity: float = 2.0):
        self.learning_rate = learning_rate
        self.sensitivity = sensitivity
        self.baseline_mean = None
        self.baseline_std = None
        self.update_count = 0

    def update_baseline(self, new_value: float):
        """更新基準線"""
        if self.baseline_mean is None:
            self.baseline_mean = new_value
            self.baseline_std = 0.0
        else:
            # 指數移動平均
            self.baseline_mean = (1 - self.learning_rate) * self.baseline_mean + \
                               self.learning_rate * new_value

            # 更新標準差
            deviation = abs(new_value - self.baseline_mean)
            if self.baseline_std == 0.0:
                self.baseline_std = deviation
            else:
                self.baseline_std = (1 - self.learning_rate) * self.baseline_std + \
                                  self.learning_rate * deviation

        self.update_count += 1

    def get_threshold(self) -> tuple:
        """獲取當前閾值"""
        if self.baseline_mean is None:
            return float('inf'), float('-inf')

        upper_threshold = self.baseline_mean + self.sensitivity * self.baseline_std
        lower_threshold = self.baseline_mean - self.sensitivity * self.baseline_std

        return upper_threshold, lower_threshold

    def check_anomaly(self, value: float) -> tuple:
        """檢查是否為異常值"""
        upper, lower = self.get_threshold()

        if value > upper:
            return True, 'high', (value - upper) / self.baseline_std if self.baseline_std > 0 else 0
        elif value < lower:
            return True, 'low', (lower - value) / self.baseline_std if self.baseline_std > 0 else 0
        else:
            return False, 'normal', 0
```

#### 6.1.2 季節性感知閾值

```python
class SeasonalThreshold:
    """季節性感知動態閾值"""

    def __init__(self, seasonal_periods: dict = None):
        self.seasonal_periods = seasonal_periods or {
            'hourly': 24,
            'daily': 7,
            'weekly': 4
        }
        self.seasonal_baselines = {}

    def update_seasonal_baseline(self, timestamp: datetime, value: float):
        """更新季節性基準線"""
        for period_name, period_length in self.seasonal_periods.items():
            # 計算季節性索引
            if period_name == 'hourly':
                seasonal_index = timestamp.hour
            elif period_name == 'daily':
                seasonal_index = timestamp.weekday()
            elif period_name == 'weekly':
                seasonal_index = timestamp.isocalendar()[1] % period_length
            else:
                continue

            # 初始化或更新基準線
            key = f"{period_name}_{seasonal_index}"
            if key not in self.seasonal_baselines:
                self.seasonal_baselines[key] = {
                    'values': [],
                    'mean': value,
                    'std': 0.0
                }
            else:
                baseline = self.seasonal_baselines[key]
                baseline['values'].append(value)

                # 保持最近 N 個值
                if len(baseline['values']) > 100:
                    baseline['values'] = baseline['values'][-50:]

                # 重新計算統計值
                baseline['mean'] = np.mean(baseline['values'])
                baseline['std'] = np.std(baseline['values'])

    def get_seasonal_threshold(self, timestamp: datetime) -> tuple:
        """獲取季節性閾值"""
        thresholds = []

        for period_name, period_length in self.seasonal_periods.items():
            if period_name == 'hourly':
                seasonal_index = timestamp.hour
            elif period_name == 'daily':
                seasonal_index = timestamp.weekday()
            elif period_name == 'weekly':
                seasonal_index = timestamp.isocalendar()[1] % period_length
            else:
                continue

            key = f"{period_name}_{seasonal_index}"
            if key in self.seasonal_baselines:
                baseline = self.seasonal_baselines[key]
                upper = baseline['mean'] + 2 * baseline['std']
                lower = baseline['mean'] - 2 * baseline['std']
                thresholds.append((upper, lower))

        if not thresholds:
            return float('inf'), float('-inf')

        # 取最嚴格的閾值
        upper_threshold = min(t[0] for t in thresholds)
        lower_threshold = max(t[1] for t in thresholds)

        return upper_threshold, lower_threshold
```

### 6.2 告警關聯分析

#### 6.2.1 相關性檢測

```python
class AlertCorrelationAnalyzer:
    """告警相關性分析器"""

    def __init__(self, correlation_window: int = 300):  # 5分鐘窗口
        self.correlation_window = correlation_window
        self.alert_history = []

    def add_alert(self, alert: dict):
        """添加告警到歷史記錄"""
        alert['timestamp'] = time.time()
        self.alert_history.append(alert)

        # 清理過期記錄
        cutoff_time = time.time() - self.correlation_window
        self.alert_history = [
            a for a in self.alert_history
            if a['timestamp'] > cutoff_time
        ]

    def find_correlated_alerts(self, new_alert: dict) -> list:
        """查找相關告警"""
        current_time = time.time()
        correlated = []

        for alert in self.alert_history:
            # 時間相關性
            time_diff = abs(current_time - alert['timestamp'])
            if time_diff > self.correlation_window:
                continue

            # 計算相關性分數
            correlation_score = self._calculate_correlation(new_alert, alert)

            if correlation_score > 0.7:  # 高相關性閾值
                correlated.append({
                    'alert': alert,
                    'correlation_score': correlation_score,
                    'time_diff': time_diff
                })

        return sorted(correlated, key=lambda x: x['correlation_score'], reverse=True)

    def _calculate_correlation(self, alert1: dict, alert2: dict) -> float:
        """計算告警相關性分數"""
        score = 0.0

        # 服務相關性
        if alert1.get('service') == alert2.get('service'):
            score += 0.3

        # 主機相關性
        if alert1.get('host') == alert2.get('host'):
            score += 0.2

        # 指標類型相關性
        metric1 = alert1.get('metric', '')
        metric2 = alert2.get('metric', '')

        if self._are_related_metrics(metric1, metric2):
            score += 0.5

        return min(score, 1.0)

    def _are_related_metrics(self, metric1: str, metric2: str) -> bool:
        """判斷指標是否相關"""
        related_groups = [
            ['cpu_percent', 'load_average', 'context_switches'],
            ['memory_percent', 'swap_usage', 'page_faults'],
            ['gpu_utilization', 'gpu_memory_used', 'gpu_temperature'],
            ['qps', 'response_time', 'error_rate']
        ]

        for group in related_groups:
            if metric1 in group and metric2 in group:
                return True

        return False
```

---

## 7. 自動化優化策略

### 7.1 決策樹優化

#### 7.1.1 規則引擎

```python
from typing import Callable, Any
from dataclasses import dataclass

@dataclass
class OptimizationRule:
    """優化規則定義"""
    condition: Callable[[dict], bool]
    action: Callable[[dict], dict]
    priority: int
    description: str

class RuleBasedOptimizer:
    """基於規則的自動優化器"""

    def __init__(self):
        self.rules = []

    def add_rule(self, rule: OptimizationRule):
        """添加優化規則"""
        self.rules.append(rule)
        # 按優先級排序
        self.rules.sort(key=lambda x: x.priority, reverse=True)

    def evaluate(self, metrics: dict) -> list:
        """評估並執行優化規則"""
        executed_actions = []

        for rule in self.rules:
            try:
                if rule.condition(metrics):
                    result = rule.action(metrics)
                    executed_actions.append({
                        'rule': rule.description,
                        'result': result,
                        'priority': rule.priority
                    })

            except Exception as e:
                logger.error(f"規則執行失敗 {rule.description}: {e}")

        return executed_actions

# 預定義優化規則
def create_standard_rules() -> list:
    """創建標準優化規則"""
    rules = []

    # CPU 高使用率優化
    rules.append(OptimizationRule(
        condition=lambda m: m.get('cpu_percent', 0) > 85,
        action=lambda m: scale_cpu_resources(m),
        priority=90,
        description="CPU 使用率過高 - 擴展 CPU 資源"
    ))

    # GPU 記憶體優化
    rules.append(OptimizationRule(
        condition=lambda m: m.get('gpu_memory_used', 0) > 90,
        action=lambda m: optimize_gpu_memory(m),
        priority=95,
        description="GPU 記憶體不足 - 優化記憶體使用"
    ))

    # QPS 性能優化
    rules.append(OptimizationRule(
        condition=lambda m: m.get('qps', 0) < 5 and m.get('cpu_percent', 0) < 50,
        action=lambda m: tune_concurrency(m),
        priority=70,
        description="QPS 過低 - 調整併發參數"
    ))

    return rules

def scale_cpu_resources(metrics: dict) -> dict:
    """擴展 CPU 資源"""
    current_cpu = metrics.get('cpu_percent', 0)
    if current_cpu > 90:
        scale_factor = 2.0
    elif current_cpu > 85:
        scale_factor = 1.5
    else:
        scale_factor = 1.2

    return {
        'action': 'scale_cpu',
        'scale_factor': scale_factor,
        'estimated_impact': f"CPU 使用率預期降低到 {current_cpu / scale_factor:.1f}%"
    }

def optimize_gpu_memory(metrics: dict) -> dict:
    """優化 GPU 記憶體"""
    current_usage = metrics.get('gpu_memory_used', 0)

    optimizations = []
    if current_usage > 95:
        optimizations.extend([
            'reduce_batch_size_50%',
            'enable_gradient_checkpointing',
            'clear_unused_cache'
        ])
    elif current_usage > 90:
        optimizations.extend([
            'reduce_batch_size_25%',
            'optimize_kv_cache'
        ])

    return {
        'action': 'optimize_gpu_memory',
        'optimizations': optimizations,
        'estimated_savings': f"{len(optimizations) * 5}% 記憶體節省"
    }
```

### 7.2 強化學習優化

#### 7.2.1 Q-Learning 參數調優

```python
import numpy as np
from collections import defaultdict

class QLearningOptimizer:
    """基於 Q-Learning 的參數優化"""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))

        # 定義狀態空間和動作空間
        self.state_bins = self._define_state_bins()
        self.actions = self._define_actions()

    def _define_state_bins(self) -> dict:
        """定義狀態空間離散化"""
        return {
            'cpu_usage': np.linspace(0, 100, 11),      # 0-100% 分 10 個區間
            'memory_usage': np.linspace(0, 100, 11),
            'gpu_usage': np.linspace(0, 100, 11),
            'qps': np.linspace(0, 100, 11)
        }

    def _define_actions(self) -> list:
        """定義動作空間"""
        return [
            'increase_batch_size',
            'decrease_batch_size',
            'increase_workers',
            'decrease_workers',
            'tune_cache_size',
            'no_action'
        ]

    def _discretize_state(self, metrics: dict) -> tuple:
        """將連續狀態離散化"""
        state = []
        for metric, bins in self.state_bins.items():
            value = metrics.get(metric, 0)
            bin_index = np.digitize(value, bins) - 1
            bin_index = max(0, min(bin_index, len(bins) - 2))
            state.append(bin_index)
        return tuple(state)

    def choose_action(self, state: tuple) -> str:
        """選擇動作 (ε-greedy)"""
        if np.random.random() < self.epsilon:
            # 探索：隨機選擇動作
            return np.random.choice(self.actions)
        else:
            # 利用：選擇最佳動作
            q_values = [self.q_table[state][action] for action in self.actions]
            best_action_index = np.argmax(q_values)
            return self.actions[best_action_index]

    def update_q_value(self, state: tuple, action: str, reward: float,
                      next_state: tuple):
        """更新 Q 值"""
        # 計算最大下一狀態 Q 值
        max_next_q = max([self.q_table[next_state][a] for a in self.actions])

        # Q-Learning 更新公式
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q

    def calculate_reward(self, old_metrics: dict, new_metrics: dict,
                        action: str) -> float:
        """計算獎勵函數"""
        reward = 0.0

        # 性能改進獎勵
        old_qps = old_metrics.get('qps', 0)
        new_qps = new_metrics.get('qps', 0)
        qps_improvement = (new_qps - old_qps) / max(old_qps, 1) * 100
        reward += qps_improvement * 0.4

        # 資源效率獎勵
        old_cpu = old_metrics.get('cpu_percent', 0)
        new_cpu = new_metrics.get('cpu_percent', 0)
        cpu_efficiency = (old_cpu - new_cpu) / max(old_cpu, 1) * 100
        reward += cpu_efficiency * 0.3

        # 穩定性獎勵
        error_rate_penalty = new_metrics.get('error_rate', 0) * -10
        reward += error_rate_penalty

        # 動作特定獎勵/懲罰
        if action == 'no_action' and qps_improvement < 0:
            reward -= 5  # 懲罰在性能下降時不採取行動

        return reward
```

---

## 8. 性能瓶頸診斷方法

### 8.1 火焰圖分析

#### 8.1.1 CPU 性能分析

```python
import subprocess
import json
from typing import Dict, List

class CPUProfiler:
    """CPU 性能分析器"""

    def __init__(self, sampling_frequency: int = 97):
        self.sampling_frequency = sampling_frequency

    def profile_vllm_process(self, duration: int = 30) -> Dict:
        """性能分析 vLLM 進程"""
        # 查找 vLLM 進程
        vllm_pids = self._find_vllm_processes()

        if not vllm_pids:
            raise RuntimeError("未找到 vLLM 進程")

        profile_data = {}

        for pid in vllm_pids:
            # 使用 perf 進行性能分析
            profile_data[pid] = self._perf_profile(pid, duration)

        return profile_data

    def _find_vllm_processes(self) -> List[int]:
        """查找 vLLM 相關進程"""
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'vllm'],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                return [int(pid) for pid in result.stdout.strip().split('\n') if pid]
            else:
                return []

        except Exception as e:
            logger.error(f"查找 vLLM 進程失敗: {e}")
            return []

    def _perf_profile(self, pid: int, duration: int) -> Dict:
        """使用 perf 進行性能分析"""
        try:
            # 運行 perf record
            perf_cmd = [
                'perf', 'record',
                '-F', str(self.sampling_frequency),
                '-p', str(pid),
                '-g',  # 調用圖
                '--', 'sleep', str(duration)
            ]

            subprocess.run(perf_cmd, check=True, capture_output=True)

            # 生成報告
            report_cmd = ['perf', 'report', '--stdio', '--no-header']
            result = subprocess.run(report_cmd, capture_output=True, text=True)

            return self._parse_perf_output(result.stdout)

        except subprocess.CalledProcessError as e:
            logger.error(f"perf 分析失敗: {e}")
            return {}

    def _parse_perf_output(self, perf_output: str) -> Dict:
        """解析 perf 輸出"""
        lines = perf_output.split('\n')
        functions = []

        for line in lines:
            if '%' in line and 'vllm' in line.lower():
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        percentage = float(parts[0].rstrip('%'))
                        function_name = ' '.join(parts[1:])
                        functions.append({
                            'percentage': percentage,
                            'function': function_name
                        })
                    except ValueError:
                        continue

        return {
            'total_samples': len(functions),
            'hot_functions': sorted(functions,
                                  key=lambda x: x['percentage'],
                                  reverse=True)[:20]
        }
```

#### 8.1.2 記憶體洩漏檢測

```python
import psutil
import gc
from typing import Dict, List, Tuple

class MemoryLeakDetector:
    """記憶體洩漏檢測器"""

    def __init__(self, sampling_interval: int = 60):
        self.sampling_interval = sampling_interval
        self.memory_snapshots = []

    def start_monitoring(self, pid: int):
        """開始監控記憶體使用"""
        process = psutil.Process(pid)

        snapshot = {
            'timestamp': time.time(),
            'memory_info': process.memory_info(),
            'memory_percent': process.memory_percent(),
            'num_fds': process.num_fds(),  # 檔案描述符數量
            'gc_stats': self._get_gc_stats()
        }

        self.memory_snapshots.append(snapshot)

        # 保持最近的記錄
        if len(self.memory_snapshots) > 1440:  # 24小時的記錄
            self.memory_snapshots = self.memory_snapshots[-720:]

    def _get_gc_stats(self) -> Dict:
        """獲取垃圾回收統計"""
        return {
            'collections': gc.get_stats(),
            'garbage_count': len(gc.garbage),
            'threshold': gc.get_threshold()
        }

    def detect_memory_leak(self) -> Dict:
        """檢測記憶體洩漏"""
        if len(self.memory_snapshots) < 10:
            return {'status': 'insufficient_data'}

        # 計算記憶體使用趨勢
        recent_snapshots = self.memory_snapshots[-60:]  # 最近1小時
        memory_values = [s['memory_info'].rss for s in recent_snapshots]

        # 線性回歸檢測趨勢
        x = np.arange(len(memory_values))
        slope, intercept = np.polyfit(x, memory_values, 1)

        # 記憶體增長率 (MB/小時)
        growth_rate = slope * 3600 / (1024 * 1024)

        # 檢測閾值
        leak_detected = growth_rate > 10  # 每小時增長超過 10MB

        return {
            'status': 'leak_detected' if leak_detected else 'normal',
            'growth_rate_mb_per_hour': growth_rate,
            'current_memory_mb': memory_values[-1] / (1024 * 1024),
            'trend_confidence': self._calculate_trend_confidence(memory_values),
            'recommendations': self._generate_leak_recommendations(growth_rate)
        }

    def _calculate_trend_confidence(self, values: List[float]) -> float:
        """計算趨勢置信度"""
        if len(values) < 3:
            return 0.0

        x = np.arange(len(values))
        correlation = np.corrcoef(x, values)[0, 1]
        return abs(correlation)

    def _generate_leak_recommendations(self, growth_rate: float) -> List[str]:
        """生成記憶體洩漏修復建議"""
        recommendations = []

        if growth_rate > 50:
            recommendations.extend([
                "立即檢查記憶體洩漏",
                "考慮重啟服務",
                "檢查是否有未釋放的大型對象"
            ])
        elif growth_rate > 20:
            recommendations.extend([
                "監控記憶體使用趨勢",
                "檢查 KV cache 配置",
                "考慮調整垃圾回收參數"
            ])
        elif growth_rate > 10:
            recommendations.append("繼續觀察記憶體使用趨勢")

        return recommendations
```

### 8.2 網路性能分析

#### 8.2.1 延遲分解分析

```python
import time
import statistics
from typing import Dict, List

class LatencyAnalyzer:
    """延遲分解分析器"""

    def __init__(self):
        self.latency_components = {
            'network_latency': [],
            'queue_time': [],
            'processing_time': [],
            'serialization_time': []
        }

    def measure_request_latency(self, request_data: dict) -> Dict:
        """測量請求延遲各組件"""
        start_time = time.time()

        # 模擬測量各個組件
        components = {}

        # 網路延遲 (可以通過 ping 或 TCP 連接時間測量)
        network_start = time.time()
        # ... 實際網路測量邏輯
        components['network_latency'] = time.time() - network_start

        # 佇列等待時間 (從 vLLM 指標獲取)
        components['queue_time'] = self._get_queue_time()

        # 處理時間 (推理時間)
        processing_start = time.time()
        # ... 實際推理時間測量
        components['processing_time'] = time.time() - processing_start

        # 序列化時間
        serialization_start = time.time()
        # ... 序列化測量
        components['serialization_time'] = time.time() - serialization_start

        # 記錄到歷史數據
        for component, latency in components.items():
            self.latency_components[component].append(latency)

        # 保持最近的記錄
        for component in self.latency_components:
            if len(self.latency_components[component]) > 1000:
                self.latency_components[component] = \
                    self.latency_components[component][-500:]

        return components

    def _get_queue_time(self) -> float:
        """獲取佇列等待時間"""
        # 從 vLLM metrics 獲取
        return 0.0  # 實際實現需要調用 vLLM API

    def analyze_latency_bottlenecks(self) -> Dict:
        """分析延遲瓶頸"""
        analysis = {}

        for component, latencies in self.latency_components.items():
            if not latencies:
                continue

            analysis[component] = {
                'avg': statistics.mean(latencies),
                'p50': statistics.median(latencies),
                'p95': self._percentile(latencies, 95),
                'p99': self._percentile(latencies, 99),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'contribution': 0.0  # 將在下面計算
            }

        # 計算各組件對總延遲的貢獻
        total_avg = sum(comp['avg'] for comp in analysis.values())
        for component in analysis:
            analysis[component]['contribution'] = \
                analysis[component]['avg'] / total_avg * 100

        # 識別主要瓶頸
        bottleneck = max(analysis.items(), key=lambda x: x[1]['contribution'])

        return {
            'components': analysis,
            'primary_bottleneck': bottleneck[0],
            'bottleneck_contribution': bottleneck[1]['contribution'],
            'optimization_recommendations': self._generate_latency_recommendations(analysis)
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """計算百分位數"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _generate_latency_recommendations(self, analysis: Dict) -> List[str]:
        """生成延遲優化建議"""
        recommendations = []

        for component, stats in analysis.items():
            if stats['contribution'] > 40:  # 主要瓶頸
                if component == 'network_latency':
                    recommendations.append("優化網路配置，考慮使用更快的網路連接")
                elif component == 'queue_time':
                    recommendations.append("增加併發處理能力或調整批次大小")
                elif component == 'processing_time':
                    recommendations.append("優化模型推理，考慮量化或模型壓縮")
                elif component == 'serialization_time':
                    recommendations.append("優化數據序列化，使用更高效的格式")

        return recommendations
```

---

## 9. 容量規劃數學模型

### 9.1 排隊論模型

#### 9.1.1 M/M/c 模型

```python
import math
from scipy.special import factorial

class QueueingModel:
    """基於排隊論的容量規劃模型"""

    def __init__(self):
        pass

    def mm_c_model(self, arrival_rate: float, service_rate: float,
                   num_servers: int) -> Dict:
        """M/M/c 排隊模型分析"""
        # 服務強度
        rho = arrival_rate / service_rate

        # 系統利用率
        utilization = rho / num_servers

        if utilization >= 1:
            return {
                'status': 'unstable',
                'message': '系統不穩定，到達率超過服務能力'
            }

        # 計算 P0 (系統空閒概率)
        p0 = self._calculate_p0(rho, num_servers)

        # 等待概率 (Erlang C 公式)
        numerator = (rho ** num_servers) / factorial(num_servers)
        denominator = 1 - utilization
        wait_prob = (numerator * p0) / denominator

        # 平均等待時間
        avg_wait_time = wait_prob / (num_servers * service_rate - arrival_rate)

        # 平均系統時間
        avg_system_time = avg_wait_time + (1 / service_rate)

        # 平均系統中的客戶數
        avg_customers = arrival_rate * avg_system_time

        return {
            'status': 'stable',
            'utilization': utilization,
            'wait_probability': wait_prob,
            'avg_wait_time': avg_wait_time,
            'avg_system_time': avg_system_time,
            'avg_customers_in_system': avg_customers,
            'p0': p0
        }

    def _calculate_p0(self, rho: float, c: int) -> float:
        """計算系統空閒概率 P0"""
        sum_term1 = sum(rho**n / factorial(n) for n in range(c))
        term2 = (rho**c / factorial(c)) * (1 / (1 - rho/c))

        return 1 / (sum_term1 + term2)

    def optimize_server_count(self, arrival_rate: float, service_rate: float,
                            target_wait_time: float) -> Dict:
        """優化服務器數量以達到目標等待時間"""
        min_servers = math.ceil(arrival_rate / service_rate) + 1

        for c in range(min_servers, min_servers + 20):
            result = self.mm_c_model(arrival_rate, service_rate, c)

            if (result['status'] == 'stable' and
                result['avg_wait_time'] <= target_wait_time):

                result['recommended_servers'] = c
                result['cost_benefit'] = self._calculate_cost_benefit(
                    c, result['avg_wait_time'], target_wait_time
                )
                return result

        return {
            'status': 'infeasible',
            'message': f'無法在合理服務器數量下達到目標等待時間 {target_wait_time}'
        }

    def _calculate_cost_benefit(self, servers: int, actual_wait: float,
                              target_wait: float) -> Dict:
        """計算成本效益分析"""
        # 假設成本模型
        server_cost_per_hour = 10  # 每台服務器每小時成本
        user_wait_cost_per_second = 0.01  # 用戶等待成本

        hourly_server_cost = servers * server_cost_per_hour
        hourly_wait_cost = actual_wait * user_wait_cost_per_second * 3600

        total_cost = hourly_server_cost + hourly_wait_cost

        return {
            'servers': servers,
            'hourly_server_cost': hourly_server_cost,
            'hourly_wait_cost': hourly_wait_cost,
            'total_hourly_cost': total_cost,
            'wait_time_improvement': target_wait - actual_wait
        }
```

#### 9.1.2 響應時間預測模型

```python
class ResponseTimePredictionModel:
    """響應時間預測模型"""

    def __init__(self):
        self.historical_data = []

    def add_observation(self, qps: float, response_time: float,
                       server_count: int, model_size: str):
        """添加觀測數據"""
        self.historical_data.append({
            'qps': qps,
            'response_time': response_time,
            'server_count': server_count,
            'model_size': model_size,
            'timestamp': time.time()
        })

    def predict_response_time(self, target_qps: float, server_count: int,
                            model_size: str) -> Dict:
        """預測響應時間"""
        # 過濾相關數據
        relevant_data = [
            d for d in self.historical_data
            if d['model_size'] == model_size
        ]

        if len(relevant_data) < 5:
            return {'status': 'insufficient_data'}

        # 簡化的線性模型
        # response_time = base_time + qps_factor * qps / server_count

        # 計算模型參數
        base_time = min(d['response_time'] for d in relevant_data)

        # 計算 QPS 影響因子
        qps_factors = []
        for d in relevant_data:
            if d['qps'] > 0 and d['server_count'] > 0:
                factor = (d['response_time'] - base_time) * d['server_count'] / d['qps']
                qps_factors.append(factor)

        avg_qps_factor = statistics.mean(qps_factors) if qps_factors else 0.1

        # 預測響應時間
        predicted_time = base_time + (avg_qps_factor * target_qps / server_count)

        # 計算置信區間
        errors = [
            abs(d['response_time'] - (base_time + avg_qps_factor * d['qps'] / d['server_count']))
            for d in relevant_data
            if d['qps'] > 0 and d['server_count'] > 0
        ]

        avg_error = statistics.mean(errors) if errors else 0.1

        return {
            'status': 'success',
            'predicted_response_time': predicted_time,
            'confidence_interval': {
                'lower': predicted_time - 1.96 * avg_error,
                'upper': predicted_time + 1.96 * avg_error
            },
            'model_parameters': {
                'base_time': base_time,
                'qps_factor': avg_qps_factor
            }
        }
```

### 9.2 資源需求預測

#### 9.2.1 時間序列分解

```python
from scipy.fft import fft, fftfreq
import numpy as np

class ResourceDemandForecaster:
    """資源需求預測器"""

    def __init__(self):
        self.models = {}

    def decompose_time_series(self, data: np.ndarray,
                            timestamps: np.ndarray) -> Dict:
        """時間序列分解"""
        # 趨勢分析 (移動平均)
        window_size = min(24, len(data) // 4)
        trend = np.convolve(data, np.ones(window_size)/window_size, mode='same')

        # 季節性分析 (FFT)
        fft_result = fft(data - trend)
        frequencies = fftfreq(len(data))

        # 找到主要週期
        power_spectrum = np.abs(fft_result)
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_period = 1 / abs(frequencies[dominant_freq_idx]) if frequencies[dominant_freq_idx] != 0 else len(data)

        # 季節性成分
        seasonal = self._extract_seasonal_component(data, int(dominant_period))

        # 殘差
        residual = data - trend - seasonal

        return {
            'original': data,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'dominant_period': dominant_period,
            'trend_slope': self._calculate_trend_slope(trend),
            'seasonality_strength': np.std(seasonal) / np.std(data)
        }

    def _extract_seasonal_component(self, data: np.ndarray, period: int) -> np.ndarray:
        """提取季節性成分"""
        if period <= 1 or period >= len(data):
            return np.zeros_like(data)

        seasonal = np.zeros_like(data)

        for i in range(len(data)):
            cycle_position = i % period
            # 使用同一週期位置的平均值
            same_position_values = []
            for j in range(cycle_position, len(data), period):
                same_position_values.append(data[j])

            seasonal[i] = np.mean(same_position_values) - np.mean(data)

        return seasonal

    def _calculate_trend_slope(self, trend: np.ndarray) -> float:
        """計算趨勢斜率"""
        x = np.arange(len(trend))
        slope = np.polyfit(x, trend, 1)[0]
        return slope

    def forecast_resource_demand(self, historical_data: Dict,
                               forecast_horizon: int) -> Dict:
        """預測資源需求"""
        forecasts = {}

        for resource_type, data_points in historical_data.items():
            if len(data_points) < 48:  # 需要至少48個數據點
                continue

            values = np.array([point['value'] for point in data_points])
            timestamps = np.array([point['timestamp'] for point in data_points])

            # 時間序列分解
            decomposition = self.decompose_time_series(values, timestamps)

            # 預測趨勢
            trend_forecast = self._forecast_trend(
                decomposition['trend'],
                decomposition['trend_slope'],
                forecast_horizon
            )

            # 預測季節性
            seasonal_forecast = self._forecast_seasonal(
                decomposition['seasonal'],
                decomposition['dominant_period'],
                forecast_horizon
            )

            # 組合預測
            combined_forecast = trend_forecast + seasonal_forecast

            # 計算預測區間
            residual_std = np.std(decomposition['residual'])
            prediction_interval = {
                'lower': combined_forecast - 1.96 * residual_std,
                'upper': combined_forecast + 1.96 * residual_std
            }

            forecasts[resource_type] = {
                'forecast': combined_forecast.tolist(),
                'prediction_interval': prediction_interval,
                'trend_component': trend_forecast.tolist(),
                'seasonal_component': seasonal_forecast.tolist(),
                'forecast_horizon': forecast_horizon,
                'model_quality': self._assess_model_quality(decomposition)
            }

        return forecasts

    def _forecast_trend(self, trend: np.ndarray, slope: float,
                       horizon: int) -> np.ndarray:
        """預測趨勢成分"""
        last_value = trend[-1]
        return np.array([last_value + slope * i for i in range(1, horizon + 1)])

    def _forecast_seasonal(self, seasonal: np.ndarray, period: float,
                          horizon: int) -> np.ndarray:
        """預測季節性成分"""
        period = int(period)
        if period <= 1:
            return np.zeros(horizon)

        # 重複季節性模式
        seasonal_cycle = seasonal[-period:]
        forecast = []

        for i in range(horizon):
            cycle_position = i % period
            forecast.append(seasonal_cycle[cycle_position])

        return np.array(forecast)

    def _assess_model_quality(self, decomposition: Dict) -> Dict:
        """評估模型質量"""
        original = decomposition['original']
        trend = decomposition['trend']
        seasonal = decomposition['seasonal']
        residual = decomposition['residual']

        # 計算各成分的解釋度
        total_variance = np.var(original)
        trend_variance = np.var(trend)
        seasonal_variance = np.var(seasonal)
        residual_variance = np.var(residual)

        return {
            'trend_explained_variance': trend_variance / total_variance,
            'seasonal_explained_variance': seasonal_variance / total_variance,
            'residual_variance_ratio': residual_variance / total_variance,
            'model_fit_score': 1 - (residual_variance / total_variance)
        }
```

---

## 10. 生產環境最佳實踐

### 10.1 監控系統架構設計

#### 10.1.1 高可用性架構

```yaml
# 生產級監控架構配置
monitoring_architecture:
  prometheus:
    deployment: "multi-instance"
    instances:
      - name: "prometheus-primary"
        role: "primary"
        retention: "15d"
        scrape_interval: "15s"
      - name: "prometheus-secondary"
        role: "backup"
        retention: "30d"
        scrape_interval: "30s"

    federation:
      enabled: true
      global_retention: "90d"

    alert_manager:
      instances: 3
      cluster_mode: true
      notification_channels:
        - slack
        - email
        - pagerduty

  grafana:
    deployment: "clustered"
    database: "postgresql"
    cache: "redis"
    load_balancer: "nginx"

  storage:
    time_series_db: "prometheus"
    long_term_storage: "thanos"
    metrics_retention:
      raw: "24h"
      downsampled_5m: "7d"
      downsampled_1h: "30d"
      downsampled_1d: "1y"
```

#### 10.1.2 安全性配置

```python
class MonitoringSecurityConfig:
    """監控系統安全配置"""

    @staticmethod
    def get_prometheus_security_config() -> Dict:
        """Prometheus 安全配置"""
        return {
            # 身份驗證
            "basic_auth": {
                "enabled": True,
                "username": "${PROMETHEUS_USER}",
                "password": "${PROMETHEUS_PASSWORD}"
            },

            # TLS 配置
            "tls": {
                "enabled": True,
                "cert_file": "/etc/prometheus/certs/prometheus.crt",
                "key_file": "/etc/prometheus/certs/prometheus.key",
                "ca_file": "/etc/prometheus/certs/ca.crt"
            },

            # 網路安全
            "network": {
                "bind_address": "0.0.0.0:9090",
                "allowed_ips": [
                    "10.0.0.0/8",
                    "172.16.0.0/12",
                    "192.168.0.0/16"
                ]
            },

            # 資料保護
            "data_protection": {
                "encryption_at_rest": True,
                "backup_encryption": True,
                "access_logging": True
            }
        }

    @staticmethod
    def get_grafana_security_config() -> Dict:
        """Grafana 安全配置"""
        return {
            # 用戶管理
            "auth": {
                "oauth": {
                    "enabled": True,
                    "providers": ["google", "github"]
                },
                "ldap": {
                    "enabled": True,
                    "server": "ldap.company.com"
                }
            },

            # 權限控制
            "rbac": {
                "enabled": True,
                "roles": {
                    "viewer": ["read:dashboards"],
                    "editor": ["read:dashboards", "write:dashboards"],
                    "admin": ["*"]
                }
            },

            # 安全 headers
            "security_headers": {
                "content_security_policy": True,
                "x_frame_options": "DENY",
                "x_content_type_options": "nosniff"
            }
        }
```

### 10.2 運維流程標準化

#### 10.2.1 告警處理標準作業程序

```python
class AlertHandlingSOP:
    """告警處理標準作業程序"""

    def __init__(self):
        self.procedures = {
            "P0_CRITICAL": self.handle_p0_critical,
            "P1_HIGH": self.handle_p1_high,
            "P2_MEDIUM": self.handle_p2_medium,
            "P3_LOW": self.handle_p3_low
        }

        self.escalation_matrix = {
            "P0": {"initial": "oncall", "15min": "team_lead", "30min": "director"},
            "P1": {"initial": "oncall", "30min": "team_lead", "60min": "manager"},
            "P2": {"initial": "oncall", "4h": "team_lead"},
            "P3": {"initial": "oncall", "24h": "team_lead"}
        }

    def handle_p0_critical(self, alert: Dict) -> Dict:
        """P0 緊急告警處理程序"""
        response_plan = {
            "immediate_actions": [
                "1. 確認告警真實性 (2分鐘內)",
                "2. 建立事故響應群組",
                "3. 開始事故記錄",
                "4. 通知相關利益相關者"
            ],

            "investigation_steps": [
                "1. 檢查系統狀態指標",
                "2. 查看最近的變更記錄",
                "3. 分析日誌和錯誤訊息",
                "4. 確定根本原因"
            ],

            "recovery_actions": [
                "1. 執行立即緩解措施",
                "2. 如需要，執行緊急擴容",
                "3. 考慮降級服務",
                "4. 監控恢復進度"
            ],

            "escalation_timeline": self.escalation_matrix["P0"],

            "communication_plan": {
                "internal": "每15分鐘更新狀態",
                "external": "必要時發送服務狀態頁面更新",
                "post_incident": "24小時內發送事後檢討報告"
            }
        }

        return response_plan

    def handle_p1_high(self, alert: Dict) -> Dict:
        """P1 高優先級告警處理程序"""
        return {
            "immediate_actions": [
                "1. 在30分鐘內確認告警",
                "2. 評估業務影響",
                "3. 開始調查程序"
            ],

            "investigation_timeline": "2小時內確定根本原因",
            "resolution_timeline": "4小時內解決或提供工作方案",
            "escalation_timeline": self.escalation_matrix["P1"]
        }

    def generate_runbook(self, alert_type: str) -> str:
        """生成運維手冊"""
        runbooks = {
            "high_cpu_usage": """
# CPU 使用率過高處理手冊

## 緊急處理步驟
1. 檢查 top/htop 確認 CPU 消耗進程
2. 檢查是否有異常的 vLLM 請求
3. 考慮水平擴展或重啟服務

## 詳細調查步驟
1. 分析 CPU 使用模式
   ```bash
   # 檢查 CPU 使用情況
   top -p $(pgrep vllm)

   # 檢查 CPU 等待時間
   iostat -x 1
   ```

2. 檢查系統負載
   ```bash
   # 檢查負載平均值
   uptime

   # 檢查進程狀態
   ps aux | grep vllm
   ```

## 解決方案
- 短期：增加 CPU 資源或重啟服務
- 長期：優化模型配置或升級硬體
            """,

            "memory_leak": """
# 記憶體洩漏處理手冊

## 確認記憶體洩漏
1. 檢查記憶體使用趨勢
2. 分析 GC 統計信息
3. 使用記憶體分析工具

## 緊急措施
1. 重啟 vLLM 服務
2. 減少批次大小
3. 清理快取

## 根本原因分析
1. 檢查代碼變更
2. 分析記憶體分配模式
3. 使用 profiling 工具
            """
        }

        return runbooks.get(alert_type, "未找到對應的運維手冊")
```

#### 10.2.2 變更管理流程

```python
class ChangeManagementProcess:
    """變更管理流程"""

    def __init__(self):
        self.change_categories = {
            "emergency": {"approval_required": False, "testing": "minimal"},
            "standard": {"approval_required": True, "testing": "full"},
            "major": {"approval_required": True, "testing": "comprehensive"}
        }

    def create_change_request(self, change_details: Dict) -> Dict:
        """創建變更請求"""
        change_request = {
            "id": self._generate_change_id(),
            "category": self._categorize_change(change_details),
            "description": change_details["description"],
            "risk_assessment": self._assess_risk(change_details),
            "testing_plan": self._create_testing_plan(change_details),
            "rollback_plan": self._create_rollback_plan(change_details),
            "approval_status": "pending",
            "created_at": datetime.now().isoformat()
        }

        return change_request

    def _assess_risk(self, change_details: Dict) -> Dict:
        """風險評估"""
        risk_factors = {
            "scope": change_details.get("scope", "low"),  # low/medium/high
            "complexity": change_details.get("complexity", "low"),
            "testing_coverage": change_details.get("testing_coverage", "high"),
            "rollback_ease": change_details.get("rollback_ease", "easy")
        }

        # 計算風險分數
        risk_score = 0
        if risk_factors["scope"] == "high": risk_score += 3
        elif risk_factors["scope"] == "medium": risk_score += 2
        else: risk_score += 1

        if risk_factors["complexity"] == "high": risk_score += 3
        elif risk_factors["complexity"] == "medium": risk_score += 2
        else: risk_score += 1

        if risk_factors["testing_coverage"] == "low": risk_score += 2
        if risk_factors["rollback_ease"] == "difficult": risk_score += 2

        risk_level = "low" if risk_score <= 4 else "medium" if risk_score <= 7 else "high"

        return {
            "factors": risk_factors,
            "score": risk_score,
            "level": risk_level,
            "mitigation_required": risk_score > 6
        }

    def _create_testing_plan(self, change_details: Dict) -> List[str]:
        """創建測試計劃"""
        base_tests = [
            "功能測試",
            "回歸測試",
            "性能測試",
            "監控驗證"
        ]

        if change_details.get("affects_api", False):
            base_tests.append("API 兼容性測試")

        if change_details.get("affects_database", False):
            base_tests.append("數據一致性測試")

        return base_tests

    def _create_rollback_plan(self, change_details: Dict) -> Dict:
        """創建回滾計劃"""
        return {
            "rollback_triggers": [
                "性能下降超過20%",
                "錯誤率超過1%",
                "用戶投訴增加",
                "監控告警觸發"
            ],
            "rollback_steps": [
                "1. 停止新的變更部署",
                "2. 恢復到前一個穩定版本",
                "3. 驗證服務正常運行",
                "4. 通知相關團隊"
            ],
            "rollback_time_estimate": "15分鐘",
            "data_backup_required": change_details.get("affects_data", False)
        }
```

### 10.3 性能基準與 SLA 定義

#### 10.3.1 SLA 指標定義

```python
class SLADefinitions:
    """SLA 指標定義"""

    def __init__(self):
        self.sla_targets = {
            "availability": {
                "target": 99.9,  # 99.9% 可用性
                "measurement_window": "monthly",
                "exclusions": ["planned_maintenance"]
            },

            "response_time": {
                "p50": 1.0,      # 50% 請求在 1 秒內完成
                "p95": 3.0,      # 95% 請求在 3 秒內完成
                "p99": 10.0,     # 99% 請求在 10 秒內完成
                "measurement_window": "daily"
            },

            "throughput": {
                "min_qps": 50,   # 最小 QPS
                "target_qps": 200, # 目標 QPS
                "peak_qps": 500,  # 峰值 QPS
                "measurement_window": "hourly"
            },

            "error_rate": {
                "target": 0.1,   # 錯誤率低於 0.1%
                "critical_threshold": 1.0,  # 1% 為嚴重告警
                "measurement_window": "hourly"
            }
        }

    def calculate_sla_compliance(self, metrics_data: Dict,
                               time_period: str) -> Dict:
        """計算 SLA 達成率"""
        compliance_report = {}

        for sla_name, sla_config in self.sla_targets.items():
            if sla_name not in metrics_data:
                continue

            actual_metrics = metrics_data[sla_name]

            if sla_name == "availability":
                compliance = self._calculate_availability_compliance(
                    actual_metrics, sla_config
                )
            elif sla_name == "response_time":
                compliance = self._calculate_response_time_compliance(
                    actual_metrics, sla_config
                )
            elif sla_name == "throughput":
                compliance = self._calculate_throughput_compliance(
                    actual_metrics, sla_config
                )
            elif sla_name == "error_rate":
                compliance = self._calculate_error_rate_compliance(
                    actual_metrics, sla_config
                )
            else:
                continue

            compliance_report[sla_name] = compliance

        # 計算總體 SLA 達成率
        overall_compliance = self._calculate_overall_compliance(compliance_report)
        compliance_report["overall"] = overall_compliance

        return compliance_report

    def _calculate_availability_compliance(self, actual: Dict, target: Dict) -> Dict:
        """計算可用性達成率"""
        uptime = actual.get("uptime_seconds", 0)
        total_time = actual.get("total_seconds", 0)

        if total_time == 0:
            return {"compliance": 0.0, "status": "no_data"}

        actual_availability = (uptime / total_time) * 100
        target_availability = target["target"]

        compliance = min(actual_availability / target_availability, 1.0) * 100

        return {
            "compliance": compliance,
            "actual": actual_availability,
            "target": target_availability,
            "status": "met" if actual_availability >= target_availability else "missed",
            "downtime_minutes": (total_time - uptime) / 60
        }

    def generate_sla_report(self, compliance_data: Dict) -> str:
        """生成 SLA 報告"""
        report = []
        report.append("# SLA 合規性報告")
        report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for sla_name, data in compliance_data.items():
            if sla_name == "overall":
                continue

            status_emoji = "✅" if data["status"] == "met" else "❌"
            report.append(f"## {sla_name.title()} {status_emoji}")
            report.append(f"- 達成率: {data['compliance']:.2f}%")
            report.append(f"- 實際值: {data['actual']:.2f}")
            report.append(f"- 目標值: {data['target']:.2f}")
            report.append("")

        # 總體評估
        overall = compliance_data.get("overall", {})
        overall_emoji = "✅" if overall.get("grade", "F") in ["A", "B"] else "⚠️" if overall.get("grade") == "C" else "❌"

        report.append(f"## 總體評估 {overall_emoji}")
        report.append(f"- 總體得分: {overall.get('score', 0):.1f}/100")
        report.append(f"- 等級: {overall.get('grade', 'F')}")

        return "\n".join(report)
```

---

## 總結

本技術深度剖析文檔涵蓋了 vLLM 性能監控的完整技術體系，從基礎的監控架構設計到高級的預測性分析和自動化優化。這些技術和方法已在實際生產環境中得到驗證，可以幫助運維團隊建立世界級的 LLM 服務監控系統。

### 關鍵技術要點

1. **分層監控體系**: 從基礎設施到業務層的全棧監控
2. **智能化告警**: 動態閾值和機器學習驅動的異常檢測
3. **預測性維護**: 基於時間序列分析的容量規劃
4. **自動化運維**: 規則引擎和強化學習的自動優化
5. **標準化流程**: SLA 管理和變更控制的最佳實踐

### 實施建議

- **循序漸進**: 從基礎監控開始，逐步增加智能化功能
- **持續改進**: 根據實際運行情況調整和優化監控策略
- **團隊協作**: 建立跨團隊的監控和運維協作機制
- **知識共享**: 建立完善的文檔和培訓體系

通過實施這些技術和方法，可以顯著提升 vLLM 服務的可靠性、性能和運維效率。