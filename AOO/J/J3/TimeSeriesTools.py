"""
J3时间序列工具模块

该模块提供完整的时间序列分析、预测、变换和特征提取功能。
包含异步处理和批处理支持，适用于大规模时间序列数据分析。

主要功能：
1. 时间序列分析：趋势分析、季节性分解、周期性检测
2. 时间序列预测：ARIMA、指数平滑、状态空间模型
3. 时间序列变换：差分、对数变换、移动平均
4. 时间序列特征提取：技术指标、统计特征、频域特征
5. 多时间尺度处理
6. 异步处理和批处理支持

Author: AI Assistant
Date: 2025-11-06
Version: 1.0.0
"""

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, 
    Iterator, Generator, TypeVar, Generic
)
from collections import defaultdict, deque
import json
import pickle
import hashlib

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 类型定义
T = TypeVar('T')
TimeSeriesData = Union[pd.Series, np.ndarray, List[float]]
TimeSeriesIndex = Union[pd.DatetimeIndex, pd.RangeIndex, List[datetime]]


class TimeSeriesError(Exception):
    """时间序列处理基础异常类"""
    pass


class TimeSeriesValidationError(TimeSeriesError):
    """时间序列数据验证异常"""
    pass


class TimeSeriesProcessingError(TimeSeriesError):
    """时间序列处理异常"""
    pass


def validate_time_series(func):
    """时间序列数据验证装饰器"""
    @wraps(func)
    def wrapper(self, data: TimeSeriesData, *args, **kwargs):
        try:
            # 转换为pandas Series
            if isinstance(data, (list, np.ndarray)):
                data = pd.Series(data)
            elif not isinstance(data, pd.Series):
                raise TimeSeriesValidationError(f"不支持的数据类型: {type(data)}")
            
            # 检查数据有效性
            if len(data) == 0:
                raise TimeSeriesValidationError("时间序列数据不能为空")
            
            if data.isna().all():
                raise TimeSeriesValidationError("时间序列数据全部为NaN")
            
            # 检查数值有效性
            if not np.isfinite(data.values).all():
                raise TimeSeriesValidationError("时间序列包含无效数值（inf/nan）")
            
            return func(self, data, *args, **kwargs)
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            raise TimeSeriesValidationError(f"数据验证失败: {e}")
    return wrapper


def async_processing(func):
    """异步处理装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(func(self, *args, **kwargs))
            finally:
                loop.close()
        else:
            return func(self, *args, **kwargs)
    return wrapper



@dataclass
class TimeSeriesResult:
    """时间序列处理结果基类"""
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    def save(self, filepath: str) -> None:
        """保存结果到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)



@dataclass
class TrendResult(TimeSeriesResult):
    """趋势分析结果"""
    data: Any = field(default=None)  # 继承自TimeSeriesResult
    trend: np.ndarray = field(default=None)
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    p_value: float = 1.0



@dataclass
class SeasonalResult(TimeSeriesResult):
    """季节性分解结果"""
    data: Any = field(default=None)  # 继承自TimeSeriesResult
    trend: np.ndarray = field(default=None)
    seasonal: np.ndarray = field(default=None)
    residual: np.ndarray = field(default=None)
    seasonal_strength: float = 0.0



@dataclass
class ForecastResult(TimeSeriesResult):
    """预测结果"""
    data: Any = field(default=None)  # 继承自TimeSeriesResult
    forecast: np.ndarray = field(default=None)
    confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None
    forecast_error: float = 0.0
    model_params: Dict[str, Any] = field(default_factory=dict)



@dataclass
class FeatureResult(TimeSeriesResult):
    """特征提取结果"""
    data: Any = field(default=None)  # 继承自TimeSeriesResult
    features: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)


class TimeSeriesAnalyzer:
    """
    时间序列分析器
    
    提供趋势分析、季节性分解、周期性检测等功能
    """
    
    def __init__(self, executor: Optional[ThreadPoolExecutor] = None):
        """
        初始化时间序列分析器
        
        Args:
            executor: 线程池执行器，用于异步处理
        """
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @validate_time_series
    def analyze_trend(
        self, 
        data: TimeSeriesData,
        method: str = 'linear',
        window: Optional[int] = None
    ) -> TrendResult:
        """
        趋势分析
        
        Args:
            data: 时间序列数据
            method: 趋势分析方法 ('linear', 'polynomial', 'moving_average')
            window: 移动平均窗口大小
            
        Returns:
            TrendResult: 趋势分析结果
            
        Examples:
            >>> analyzer = TimeSeriesAnalyzer()
            >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> result = analyzer.analyze_trend(data, method='linear')
            >>> print(f"趋势斜率: {result.slope:.3f}")
        """
        try:
            data = pd.Series(data)
            
            if method == 'linear':
                return self._linear_trend(data)
            elif method == 'polynomial':
                return self._polynomial_trend(data, degree=2)
            elif method == 'moving_average':
                return self._moving_average_trend(data, window or len(data)//4)
            else:
                raise TimeSeriesValidationError(f"不支持的趋势分析方法: {method}")
                
        except Exception as e:
            self.logger.error(f"趋势分析失败: {e}")
            raise TimeSeriesProcessingError(f"趋势分析失败: {e}")
    
    def _linear_trend(self, data: pd.Series) -> TrendResult:
        """线性趋势分析"""
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data.values)
        
        trend = slope * x + intercept
        
        return TrendResult(
            trend=trend,
            slope=slope,
            intercept=intercept,
            r_squared=r_value**2,
            p_value=p_value
        )
    
    def _polynomial_trend(self, data: pd.Series, degree: int = 2) -> TrendResult:
        """多项式趋势分析"""
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data.values, degree)
        trend = np.polyval(coeffs, x)
        
        # 计算R²
        ss_res = np.sum((data.values - trend) ** 2)
        ss_tot = np.sum((data.values - np.mean(data.values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return TrendResult(
            trend=trend,
            slope=coeffs[-2] if len(coeffs) >= 2 else 0,
            intercept=coeffs[-1],
            r_squared=r_squared,
            p_value=1.0
        )
    
    def _moving_average_trend(self, data: pd.Series, window: int) -> TrendResult:
        """移动平均趋势分析"""
        if window >= len(data):
            window = len(data) // 2
            
        trend = data.rolling(window=window, center=True).mean().bfill().ffill()
        
        # 计算趋势强度
        trend_strength = 1 - np.var(data.values - trend.values) / np.var(data.values)
        
        return TrendResult(
            data=trend.values,
            trend=trend.values,
            slope=0,  # 移动平均无法直接计算斜率
            intercept=np.mean(trend.values),
            r_squared=max(0, trend_strength),
            p_value=1.0
        )
    
    @validate_time_series
    def decompose_seasonal(
        self,
        data: TimeSeriesData,
        period: Optional[int] = None,
        method: str = 'additive'
    ) -> SeasonalResult:
        """
        季节性分解
        
        Args:
            data: 时间序列数据
            period: 季节周期，如果为None则自动检测
            method: 分解方法 ('additive', 'multiplicative')
            
        Returns:
            SeasonalResult: 季节性分解结果
            
        Examples:
            >>> analyzer = TimeSeriesAnalyzer()
            >>> data = pd.Series([1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2] * 3)
            >>> result = analyzer.decompose_seasonal(data, period=12)
            >>> print(f"季节性强度: {result.seasonal_strength:.3f}")
        """
        try:
            data = pd.Series(data)
            
            # 自动检测周期
            if period is None:
                period = self._detect_seasonal_period(data)
            
            if period is None or period < 2:
                raise TimeSeriesValidationError("无法检测到有效的季节周期")
            
            if method == 'additive':
                return self._additive_decomposition(data, period)
            elif method == 'multiplicative':
                return self._multiplicative_decomposition(data, period)
            else:
                raise TimeSeriesValidationError(f"不支持的分解方法: {method}")
                
        except Exception as e:
            self.logger.error(f"季节性分解失败: {e}")
            raise TimeSeriesProcessingError(f"季节性分解失败: {e}")
    
    def _detect_seasonal_period(self, data: pd.Series) -> Optional[int]:
        """自动检测季节周期"""
        try:
            # 使用FFT检测主要频率
            fft_vals = np.abs(fft(data.values))
            freqs = fftfreq(len(data))
            
            # 找到最大幅度的频率（排除直流分量）
            max_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
            period = int(1 / freqs[max_idx]) if freqs[max_idx] != 0 else None
            
            # 验证周期的合理性
            if period and 2 <= period <= len(data) // 2:
                return period
            else:
                return None
                
        except Exception:
            return None
    
    def _additive_decomposition(self, data: pd.Series, period: int) -> SeasonalResult:
        """加法季节性分解"""
        # 计算趋势
        trend = data.rolling(window=period, center=True).mean().bfill().ffill()
        
        # 去趋势
        detrended = data - trend
        
        # 计算季节性
        seasonal = np.zeros(len(data))
        for i in range(period):
            seasonal_values = detrended[i::period]
            seasonal[i::period] = np.mean(seasonal_values)
        
        # 计算残差
        residual = data - trend - seasonal
        
        # 计算季节性强度
        seasonal_strength = np.var(seasonal) / np.var(data)
        
        return SeasonalResult(trend=trend.values,
            seasonal=seasonal,
            residual=residual.values,
            seasonal_strength=seasonal_strength
        )
    
    def _multiplicative_decomposition(self, data: pd.Series, period: int) -> SeasonalResult:
        """乘法季节性分解"""
        # 计算趋势
        trend = data.rolling(window=period, center=True).mean().bfill().ffill()
        
        # 去趋势
        detrended = data / trend
        
        # 计算季节性
        seasonal = np.zeros(len(data))
        for i in range(period):
            seasonal_values = detrended[i::period]
            seasonal[i::period] = np.mean(seasonal_values)
        
        # 计算残差
        residual = data / (trend * seasonal)
        
        # 计算季节性强度
        seasonal_strength = np.var(seasonal) / np.var(data)
        
        return SeasonalResult(trend=trend.values,
            seasonal=seasonal,
            residual=residual.values,
            seasonal_strength=seasonal_strength
        )
    
    @validate_time_series
    def detect_cycles(
        self,
        data: TimeSeriesData,
        min_period: int = 2,
        max_period: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        周期性检测
        
        Args:
            data: 时间序列数据
            min_period: 最小周期长度
            max_period: 最大周期长度
            
        Returns:
            List[Dict]: 检测到的周期列表
            
        Examples:
            >>> analyzer = TimeSeriesAnalyzer()
            >>> data = pd.Series(np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100))
            >>> cycles = analyzer.detect_cycles(data)
            >>> for cycle in cycles:
            ...     print(f"周期: {cycle['period']:.2f}, 强度: {cycle['strength']:.3f}")
        """
        try:
            data = pd.Series(data)
            max_period = max_period or len(data) // 2
            
            # 使用自相关检测周期
            cycles = []
            
            for period in range(min_period, min(max_period, len(data) // 2)):
                # 计算自相关
                autocorr = self._autocorrelation(data.values, lag=period)
                
                if abs(autocorr) > 0.3:  # 阈值可调
                    cycles.append({
                        'period': period,
                        'strength': abs(autocorr),
                        'phase': np.angle(np.fft.fft(data.values)[period]) if period < len(data) else 0
                    })
            
            # 按强度排序
            cycles.sort(key=lambda x: x['strength'], reverse=True)
            
            return cycles
            
        except Exception as e:
            self.logger.error(f"周期性检测失败: {e}")
            raise TimeSeriesProcessingError(f"周期性检测失败: {e}")
    
    def _autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """计算自相关"""
        if lag >= len(data):
            return 0.0
        
        mean_val = np.mean(data)
        numerator = np.sum((data[:-lag] - mean_val) * (data[lag:] - mean_val))
        denominator = np.sum((data - mean_val) ** 2)
        
        return numerator / denominator if denominator != 0 else 0.0


class TimeSeriesForecaster:
    """
    时间序列预测器
    
    提供ARIMA、指数平滑、状态空间模型等预测功能
    """
    
    def __init__(self, executor: Optional[ThreadPoolExecutor] = None):
        """
        初始化时间序列预测器
        
        Args:
            executor: 线程池执行器，用于异步处理
        """
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model_cache = {}
    
    @validate_time_series
    def forecast_arima(
        self,
        data: TimeSeriesData,
        steps: int,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ) -> ForecastResult:
        """
        ARIMA模型预测
        
        Args:
            data: 时间序列数据
            steps: 预测步数
            order: ARIMA参数 (p, d, q)
            seasonal_order: 季节性ARIMA参数 (P, D, Q, s)
            
        Returns:
            ForecastResult: 预测结果
            
        Examples:
            >>> forecaster = TimeSeriesForecaster()
            >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> result = forecaster.forecast_arima(data, steps=5, order=(1, 1, 1))
            >>> print(f"预测值: {result.forecast}")
        """
        try:
            data = pd.Series(data)
            
            # 自动选择ARIMA参数
            if order is None:
                order = self._auto_arima_order(data)
            
            # 简化版ARIMA实现（实际应用中可使用statsmodels）
            forecast, confidence_interval = self._simple_arima_forecast(
                data.values, order, steps
            )
            
            return ForecastResult(
                forecast=forecast,
                confidence_interval=confidence_interval,
                forecast_error=0.0,
                model_params={'order': order, 'seasonal_order': seasonal_order}
            )
            
        except Exception as e:
            self.logger.error(f"ARIMA预测失败: {e}")
            raise TimeSeriesProcessingError(f"ARIMA预测失败: {e}")
    
    def _auto_arima_order(self, data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """自动选择ARIMA参数"""
        # 简化版实现，实际应用中应使用更复杂的模型选择方法
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        # 简化的AIC计算
                        model = self._fit_arima_simple(data.values, (p, d, q))
                        if model is not None:
                            aic = model.get('aic', float('inf'))
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def _fit_arima_simple(self, data: np.ndarray, order: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
        """简化的ARIMA拟合"""
        try:
            p, d, q = order
            
            # 差分
            if d > 0:
                diff_data = np.diff(data, n=d)
            else:
                diff_data = data
            
            # 简化实现：使用线性回归近似
            n = len(diff_data)
            if n <= max(p, q):
                return None
            
            # 构建ARMA模型的简化版本
            # 这里使用一个非常简化的实现
            X = np.column_stack([
                np.ones(n),
                np.arange(n)
            ])
            
            # 最小二乘拟合
            try:
                coeffs = np.linalg.lstsq(X, diff_data, rcond=None)[0]
                residuals = diff_data - X @ coeffs
                residuals_var = np.var(residuals)
                if residuals_var <= 0 or not np.isfinite(residuals_var):
                    return None
                aic = n * np.log(residuals_var) + 2 * 3  # 简化AIC
                
                return {
                    'params': coeffs,
                    'residuals': residuals,
                    'aic': aic
                }
            except Exception:
                return None
                
        except Exception:
            return None
    
    def _simple_arima_forecast(self, data: np.ndarray, order: Tuple[int, int, int], steps: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """简化的ARIMA预测"""
        p, d, q = order
        
        # 差分
        if d > 0:
            diff_data = np.diff(data, n=d)
        else:
            diff_data = data
        
        # 简化预测：使用线性趋势
        trend = np.polyfit(range(len(diff_data)), diff_data, 1)
        
        # 预测未来值
        future_indices = np.arange(len(diff_data), len(diff_data) + steps)
        diff_forecast = np.polyval(trend, future_indices)
        
        # 反差分
        if d > 0:
            # 累积和还原
            last_original = data[-d:] if d <= len(data) else data
            forecast = np.cumsum(np.concatenate([last_original, diff_forecast]))[d:]
        else:
            forecast = diff_forecast
        
        # 置信区间（简化）
        std_res = np.std(diff_data[-100:] if len(diff_data) > 100 else diff_data)
        if not np.isfinite(std_res) or std_res <= 0:
            std_res = 1.0  # 默认值
        
        forecast_errors = np.sqrt(np.arange(1, steps + 1))
        confidence_interval = (
            forecast - 1.96 * std_res * forecast_errors,
            forecast + 1.96 * std_res * forecast_errors
        )
        
        return forecast, confidence_interval
    
    @validate_time_series
    def forecast_exponential_smoothing(
        self,
        data: TimeSeriesData,
        steps: int,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        seasonal: Optional[int] = None,
        method: str = 'simple'
    ) -> ForecastResult:
        """
        指数平滑预测
        
        Args:
            data: 时间序列数据
            steps: 预测步数
            alpha: 水平参数
            beta: 趋势参数
            gamma: 季节参数
            seasonal: 季节周期
            method: 平滑方法 ('simple', 'double', 'triple')
            
        Returns:
            ForecastResult: 预测结果
            
        Examples:
            >>> forecaster = TimeSeriesForecaster()
            >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> result = forecaster.forecast_exponential_smoothing(data, steps=5, method='double')
            >>> print(f"预测值: {result.forecast}")
        """
        try:
            data = pd.Series(data)
            
            if method == 'simple':
                return self._simple_exponential_smoothing(data.values, alpha or 0.3, steps)
            elif method == 'double':
                return self._double_exponential_smoothing(data.values, alpha or 0.3, beta or 0.3, steps)
            elif method == 'triple':
                return self._triple_exponential_smoothing(
                    data.values, alpha or 0.3, beta or 0.3, gamma or 0.3, seasonal or 12, steps
                )
            else:
                raise TimeSeriesValidationError(f"不支持的指数平滑方法: {method}")
                
        except Exception as e:
            self.logger.error(f"指数平滑预测失败: {e}")
            raise TimeSeriesProcessingError(f"指数平滑预测失败: {e}")
    
    def _simple_exponential_smoothing(self, data: np.ndarray, alpha: float, steps: int) -> ForecastResult:
        """简单指数平滑"""
        # 初始化
        level = data[0]
        
        # 平滑
        for i in range(1, len(data)):
            level = alpha * data[i] + (1 - alpha) * level
        
        # 预测
        forecast = np.full(steps, level)
        
        # 置信区间
        residuals = data - level
        std_res = np.std(residuals)
        if not np.isfinite(std_res) or std_res <= 0:
            std_res = 1.0  # 默认值
        confidence_interval = (
            forecast - 1.96 * std_res,
            forecast + 1.96 * std_res
        )
        
        return ForecastResult(
            forecast=forecast,
            confidence_interval=confidence_interval,
            forecast_error=0.0,
            model_params={'alpha': alpha, 'method': 'simple'}
        )
    
    def _double_exponential_smoothing(self, data: np.ndarray, alpha: float, beta: float, steps: int) -> ForecastResult:
        """双重指数平滑（Holt方法）"""
        # 初始化
        level = data[0]
        trend = data[1] - data[0] if len(data) > 1 else 0
        
        # 平滑
        for i in range(1, len(data)):
            prev_level = level
            level = alpha * data[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        # 预测
        forecast = level + trend * np.arange(1, steps + 1)
        
        # 置信区间
        residuals = data[1:] - (level + trend * np.arange(len(data) - 1))
        std_res = np.std(residuals)
        if not np.isfinite(std_res) or std_res <= 0:
            std_res = 1.0  # 默认值
        
        forecast_errors = np.sqrt(np.arange(1, steps + 1))
        confidence_interval = (
            forecast - 1.96 * std_res * forecast_errors,
            forecast + 1.96 * std_res * forecast_errors
        )
        
        return ForecastResult(
            forecast=forecast,
            confidence_interval=confidence_interval,
            forecast_error=0.0,
            model_params={'alpha': alpha, 'beta': beta, 'method': 'double'}
        )
    
    def _triple_exponential_smoothing(
        self, 
        data: np.ndarray, 
        alpha: float, 
        beta: float, 
        gamma: float, 
        seasonal: int, 
        steps: int
    ) -> ForecastResult:
        """三重指数平滑（Holt-Winters方法）"""
        n = len(data)
        
        # 初始化
        level = np.mean(data[:seasonal])
        trend = (np.mean(data[seasonal:2*seasonal]) - np.mean(data[:seasonal])) / seasonal
        
        # 季节性初始化
        seasonal_pattern = np.zeros(seasonal)
        for i in range(seasonal):
            seasonal_pattern[i] = np.mean(data[i::seasonal]) - level
        
        # 平滑
        for i in range(n):
            prev_level = level
            level = alpha * (data[i] - seasonal_pattern[i % seasonal]) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasonal_pattern[i % seasonal] = gamma * (data[i] - level) + (1 - gamma) * seasonal_pattern[i % seasonal]
        
        # 预测
        forecast = np.zeros(steps)
        for h in range(1, steps + 1):
            forecast[h-1] = (level + h * trend) + seasonal_pattern[(n + h - 1) % seasonal]
        
        # 置信区间
        residuals = data - ((level + trend * np.arange(n)) + seasonal_pattern[np.arange(n) % seasonal])
        std_res = np.std(residuals)
        if not np.isfinite(std_res) or std_res <= 0:
            std_res = 1.0  # 默认值
        confidence_interval = (
            forecast - 1.96 * std_res,
            forecast + 1.96 * std_res
        )
        
        return ForecastResult(
            forecast=forecast,
            confidence_interval=confidence_interval,
            forecast_error=0.0,
            model_params={'alpha': alpha, 'beta': beta, 'gamma': gamma, 'seasonal': seasonal, 'method': 'triple'}
        )
    
    @validate_time_series
    def forecast_state_space(
        self,
        data: TimeSeriesData,
        steps: int,
        model_type: str = 'local_level'
    ) -> ForecastResult:
        """
        状态空间模型预测
        
        Args:
            data: 时间序列数据
            steps: 预测步数
            model_type: 模型类型 ('local_level', 'local_linear', 'seasonal')
            
        Returns:
            ForecastResult: 预测结果
        """
        try:
            data = pd.Series(data)
            
            if model_type == 'local_level':
                return self._local_level_model(data.values, steps)
            elif model_type == 'local_linear':
                return self._local_linear_model(data.values, steps)
            elif model_type == 'seasonal':
                return self._seasonal_model(data.values, steps)
            else:
                raise TimeSeriesValidationError(f"不支持的状态空间模型: {model_type}")
                
        except Exception as e:
            self.logger.error(f"状态空间模型预测失败: {e}")
            raise TimeSeriesProcessingError(f"状态空间模型预测失败: {e}")
    
    def _local_level_model(self, data: np.ndarray, steps: int) -> ForecastResult:
        """局部水平状态空间模型"""
        n = len(data)
        
        # 简化的卡尔曼滤波实现
        # 状态方程: x_t = x_{t-1} + w_t
        # 观测方程: y_t = x_t + v_t
        
        # 初始化
        x_pred = np.zeros(n + steps)  # 状态预测
        P_pred = np.zeros(n + steps)  # 预测误差协方差
        
        x_filt = np.zeros(n)  # 状态滤波
        P_filt = np.zeros(n)  # 滤波误差协方差
        
        # 参数估计（简化）
        Q = np.var(np.diff(data))  # 过程噪声方差
        R = np.var(data) * 0.1     # 观测噪声方差
        
        # 初始值
        x_pred[0] = data[0]
        P_pred[0] = 1.0
        
        # 前向滤波
        for t in range(n):
            # 预测步骤
            if t > 0:
                x_pred[t] = x_filt[t-1]
                P_pred[t] = P_filt[t-1] + Q
            
            # 更新步骤
            K = P_pred[t] / (P_pred[t] + R)  # 卡尔曼增益
            x_filt[t] = x_pred[t] + K * (data[t] - x_pred[t])
            P_filt[t] = (1 - K) * P_pred[t]
        
        # 预测
        for t in range(n, n + steps):
            x_pred[t] = x_filt[n-1]
            P_pred[t] = P_filt[n-1] + Q * (t - n + 1)
        
        forecast = x_pred[n:]
        
        # 置信区间
        confidence_interval = (
            forecast - 1.96 * np.sqrt(P_pred[n:]),
            forecast + 1.96 * np.sqrt(P_pred[n:])
        )
        
        return ForecastResult(
            forecast=forecast,
            confidence_interval=confidence_interval,
            forecast_error=0.0,
            model_params={'model_type': 'local_level', 'Q': Q, 'R': R}
        )
    
    def _local_linear_model(self, data: np.ndarray, steps: int) -> ForecastResult:
        """局部线性状态空间模型"""
        n = len(data)
        
        # 状态向量: [level, trend]^T
        # 状态方程: x_t = F * x_{t-1} + w_t
        # 观测方程: y_t = H * x_t + v_t
        
        F = np.array([[1, 1], [0, 1]])  # 状态转移矩阵
        H = np.array([[1, 0]])          # 观测矩阵
        
        # 初始化
        x_pred = np.zeros((n + steps, 2))  # 状态预测
        P_pred = np.zeros((n + steps, 2, 2))  # 预测误差协方差
        
        x_filt = np.zeros((n, 2))  # 状态滤波
        P_filt = np.zeros((n, 2, 2))  # 滤波误差协方差
        
        # 参数
        Q = np.eye(2) * np.var(np.diff(data, 2))  # 过程噪声协方差
        R = np.var(data) * 0.1                     # 观测噪声方差
        
        # 初始值
        x_pred[0] = [data[0], data[1] - data[0] if n > 1 else 0]
        P_pred[0] = np.eye(2)
        
        # 前向滤波
        for t in range(n):
            # 预测步骤
            if t > 0:
                x_pred[t] = F @ x_filt[t-1]
                P_pred[t] = F @ P_filt[t-1] @ F.T + Q
            
            # 更新步骤
            S = H @ P_pred[t] @ H.T + R
            K = P_pred[t] @ H.T / S  # 卡尔曼增益
            x_filt[t] = x_pred[t] + K * (data[t] - H @ x_pred[t])
            P_filt[t] = (np.eye(2) - K @ H) @ P_pred[t]
        
        # 预测
        for t in range(n, n + steps):
            x_pred[t] = F @ x_filt[n-1]
            P_pred[t] = F @ P_filt[n-1] @ F.T + Q
        
        forecast = (H @ x_pred[n:].T).flatten()
        
        # 置信区间
        forecast_var = np.array([H @ P_pred[t] @ H.T for t in range(n, n + steps)])
        confidence_interval = (
            forecast - 1.96 * np.sqrt(forecast_var),
            forecast + 1.96 * np.sqrt(forecast_var)
        )
        
        return ForecastResult(
            forecast=forecast,
            confidence_interval=confidence_interval,
            forecast_error=0.0,
            model_params={'model_type': 'local_linear', 'Q': Q.tolist(), 'R': R}
        )
    
    def _seasonal_model(self, data: np.ndarray, steps: int) -> ForecastResult:
        """季节性状态空间模型"""
        # 简化实现：使用季节性分解后的趋势进行预测
        n = len(data)
        
        # 季节性周期检测
        seasonal_period = self._detect_simple_seasonal(data)
        if seasonal_period is None:
            seasonal_period = 12  # 默认季节周期
        
        # 计算季节性模式
        seasonal_pattern = np.zeros(seasonal_period)
        for i in range(seasonal_period):
            seasonal_values = data[i::seasonal_period]
            if len(seasonal_values) > 0:
                seasonal_pattern[i] = np.mean(seasonal_values)
        
        # 去季节性
        deseasonalized = np.zeros(n)
        for i in range(n):
            deseasonalized[i] = data[i] - seasonal_pattern[i % seasonal_period]
        
        # 对去季节性数据应用局部线性模型
        linear_result = self._local_linear_model(deseasonalized, steps)
        
        # 重新加入季节性
        forecast = linear_result.forecast.copy()
        for i in range(steps):
            forecast[i] += seasonal_pattern[(n + i) % seasonal_period]
        
        # 调整置信区间
        if linear_result.confidence_interval:
            ci_lower, ci_upper = linear_result.confidence_interval
            for i in range(steps):
                ci_lower[i] += seasonal_pattern[(n + i) % seasonal_period]
                ci_upper[i] += seasonal_pattern[(n + i) % seasonal_period]
        
        return ForecastResult(
            forecast=forecast,
            confidence_interval=linear_result.confidence_interval,
            model_params={'model_type': 'seasonal', 'seasonal_period': seasonal_period}
        )
    
    def _detect_simple_seasonal(self, data: np.ndarray) -> Optional[int]:
        """简单的季节周期检测"""
        # 使用自相关检测
        max_corr = 0
        best_period = None
        
        for period in range(2, min(len(data) // 2, 50)):
            corr = self._autocorrelation(data, period)
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                best_period = period
        
        return best_period if max_corr > 0.3 else None


class TimeSeriesTransformer:
    """
    时间序列变换器
    
    提供差分、对数变换、移动平均等变换功能
    """
    
    def __init__(self, executor: Optional[ThreadPoolExecutor] = None):
        """
        初始化时间序列变换器
        
        Args:
            executor: 线程池执行器，用于异步处理
        """
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @validate_time_series
    def difference(
        self,
        data: TimeSeriesData,
        order: int = 1,
        seasonal: Optional[int] = None
    ) -> TimeSeriesResult:
        """
        差分变换
        
        Args:
            data: 时间序列数据
            order: 差分阶数
            seasonal: 季节性差分周期
            
        Returns:
            TimeSeriesResult: 变换结果
            
        Examples:
            >>> transformer = TimeSeriesTransformer()
            >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> result = transformer.difference(data, order=1)
            >>> print(f"一阶差分: {result.data}")
        """
        try:
            data = pd.Series(data)
            
            if seasonal is not None:
                # 季节性差分
                diff_data = data.diff(seasonal)
            else:
                # 普通差分
                diff_data = data
                for _ in range(order):
                    diff_data = diff_data.diff()
            
            # 去除NaN值
            diff_data = diff_data.dropna()
            
            return TimeSeriesResult(
                data=diff_data.values,
                metadata={
                    'order': order,
                    'seasonal': seasonal,
                    'original_length': len(data),
                    'transformed_length': len(diff_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"差分变换失败: {e}")
            raise TimeSeriesProcessingError(f"差分变换失败: {e}")
    
    @validate_time_series
    def log_transform(
        self,
        data: TimeSeriesData,
        base: Optional[float] = None,
        offset: float = 1.0
    ) -> TimeSeriesResult:
        """
        对数变换
        
        Args:
            data: 时间序列数据
            base: 对数底数，默认为自然对数
            offset: 平移量，确保数据为正
            
        Returns:
            TimeSeriesResult: 变换结果
        """
        try:
            data = pd.Series(data)
            
            # 确保数据为正
            min_val = data.min()
            if min_val <= 0:
                data = data - min_val + offset
            
            # 对数变换
            if base is None:
                log_data = np.log(data)
            else:
                log_data = np.log(data) / np.log(base)
            
            return TimeSeriesResult(
                data=log_data.values,
                metadata={
                    'base': base,
                    'offset': offset,
                    'original_min': min_val,
                    'transformed_min': log_data.min()
                }
            )
            
        except Exception as e:
            self.logger.error(f"对数变换失败: {e}")
            raise TimeSeriesProcessingError(f"对数变换失败: {e}")
    
    @validate_time_series
    def box_cox_transform(
        self,
        data: TimeSeriesData,
        lambda_param: Optional[float] = None
    ) -> TimeSeriesResult:
        """
        Box-Cox变换
        
        Args:
            data: 时间序列数据
            lambda_param: Box-Cox参数，如果为None则自动选择
            
        Returns:
            TimeSeriesResult: 变换结果
        """
        try:
            data = pd.Series(data)
            
            # 确保数据为正
            min_val = data.min()
            if min_val <= 0:
                data = data - min_val + 1
            
            # 自动选择lambda参数
            if lambda_param is None:
                lambda_param = self._optimize_box_cox_lambda(data.values)
            
            # Box-Cox变换
            if lambda_param == 0:
                transformed_data = np.log(data)
            else:
                transformed_data = (data ** lambda_param - 1) / lambda_param
            
            return TimeSeriesResult(
                data=transformed_data.values,
                metadata={
                    'lambda_param': lambda_param,
                    'original_min': min_val,
                    'transformed_min': transformed_data.min()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Box-Cox变换失败: {e}")
            raise TimeSeriesProcessingError(f"Box-Cox变换失败: {e}")
    
    def _optimize_box_cox_lambda(self, data: np.ndarray) -> float:
        """优化Box-Cox变换参数"""
        def box_cox_log_likelihood(lambda_val):
            try:
                lambda_val = lambda_val[0]  # 提取标量值
                if lambda_val == 0:
                    transformed = np.log(data)
                else:
                    # 确保数据为正，避免数值错误
                    transformed = np.power(data, lambda_val) - 1
                    if lambda_val < 0:
                        transformed = -transformed / abs(lambda_val)
                    else:
                        transformed = transformed / lambda_val
                
                # 计算对数似然
                mean_val = np.mean(transformed)
                var_val = np.var(transformed)
                if var_val <= 0 or not np.isfinite(var_val):
                    return -np.inf
                
                # 添加数值稳定性检查
                if not np.all(np.isfinite(transformed)):
                    return -np.inf
                
                log_likelihood = -len(transformed) / 2 * np.log(var_val) - len(transformed) / 2
                return log_likelihood if np.isfinite(log_likelihood) else -np.inf
            except Exception:
                return -np.inf
        
        # 优化lambda参数
        try:
            result = minimize(
                box_cox_log_likelihood,
                x0=[0.0],
                bounds=[(-2, 2)],
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            
            if result.success and np.isfinite(result.fun):
                return float(result.x[0])
            else:
                return 0.0
        except Exception:
            return 0.0
    
    @validate_time_series
    def moving_average(
        self,
        data: TimeSeriesData,
        window: int,
        method: str = 'simple',
        center: bool = False
    ) -> TimeSeriesResult:
        """
        移动平均变换
        
        Args:
            data: 时间序列数据
            window: 窗口大小
            method: 平均方法 ('simple', 'weighted', 'exponential')
            center: 是否居中
            
        Returns:
            TimeSeriesResult: 变换结果
        """
        try:
            data = pd.Series(data)
            
            if method == 'simple':
                ma_data = data.rolling(window=window, center=center).mean()
            elif method == 'weighted':
                weights = np.arange(1, window + 1)
                ma_data = data.rolling(window=window, center=center).apply(
                    lambda x: np.sum(weights * x) / np.sum(weights)
                )
            elif method == 'exponential':
                alpha = 2 / (window + 1)
                ma_data = data.ewm(alpha=alpha).mean()
            else:
                raise TimeSeriesValidationError(f"不支持的移动平均方法: {method}")
            
            # 去除NaN值
            ma_data = ma_data.dropna()
            
            return TimeSeriesResult(
                data=ma_data.values,
                metadata={
                    'window': window,
                    'method': method,
                    'center': center,
                    'original_length': len(data),
                    'transformed_length': len(ma_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"移动平均变换失败: {e}")
            raise TimeSeriesProcessingError(f"移动平均变换失败: {e}")
    
    @validate_time_series
    def standardize(
        self,
        data: TimeSeriesData,
        method: str = 'zscore'
    ) -> TimeSeriesResult:
        """
        标准化变换
        
        Args:
            data: 时间序列数据
            method: 标准化方法 ('zscore', 'minmax', 'robust')
            
        Returns:
            TimeSeriesResult: 变换结果
        """
        try:
            data = pd.Series(data)
            
            if method == 'zscore':
                # Z-score标准化
                mean_val = data.mean()
                std_val = data.std()
                standardized_data = (data - mean_val) / std_val
                metadata = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
                
            elif method == 'minmax':
                # Min-Max标准化
                min_val = data.min()
                max_val = data.max()
                standardized_data = (data - min_val) / (max_val - min_val)
                metadata = {'method': 'minmax', 'min': min_val, 'max': max_val}
                
            elif method == 'robust':
                # 鲁棒标准化
                from scipy.stats import median_abs_deviation
                median_val = data.median()
                mad_val = median_abs_deviation(data.values)  # 平均绝对偏差
                mad_val = mad_val if mad_val > 0 else 1.0  # 防止除零
                standardized_data = (data - median_val) / mad_val
                metadata = {'method': 'robust', 'median': median_val, 'mad': mad_val}
                
            else:
                raise TimeSeriesValidationError(f"不支持的标准化方法: {method}")
            
            return TimeSeriesResult(
                data=standardized_data.values,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"标准化变换失败: {e}")
            raise TimeSeriesProcessingError(f"标准化变换失败: {e}")
    
    @validate_time_series
    def wavelet_transform(
        self,
        data: TimeSeriesData,
        wavelet: str = 'haar',
        levels: Optional[int] = None
    ) -> TimeSeriesResult:
        """
        小波变换
        
        Args:
            data: 时间序列数据
            wavelet: 小波类型
            levels: 分解层数
            
        Returns:
            TimeSeriesResult: 变换结果
        """
        try:
            try:
                from pywt import wavedec, waverec
            except ImportError:
                raise ImportError("需要安装PyWavelets库: pip install PyWavelets")
            
            data = pd.Series(data).values
            
            # 确定分解层数
            if levels is None:
                levels = int(np.log2(len(data)))
            
            # 小波分解
            coeffs = wavedec(data, wavelet, level=levels)
            
            # 重构信号
            reconstructed_data = waverec(coeffs, wavelet)
            
            return TimeSeriesResult(
                data=reconstructed_data,
                metadata={
                    'wavelet': wavelet,
                    'levels': levels,
                    'coefficients': [len(c) for c in coeffs]
                }
            )
            
        except ImportError:
            raise TimeSeriesError("需要安装PyWavelets库: pip install PyWavelets")
        except Exception as e:
            self.logger.error(f"小波变换失败: {e}")
            raise TimeSeriesProcessingError(f"小波变换失败: {e}")


class TimeSeriesFeatureExtractor:
    """
    时间序列特征提取器
    
    提供技术指标、统计特征、频域特征提取功能
    """
    
    def __init__(self, executor: Optional[ThreadPoolExecutor] = None):
        """
        初始化时间序列特征提取器
        
        Args:
            executor: 线程池执行器，用于异步处理
        """
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @validate_time_series
    def extract_technical_indicators(
        self,
        data: TimeSeriesData,
        indicators: Optional[List[str]] = None
    ) -> FeatureResult:
        """
        提取技术指标
        
        Args:
            data: 时间序列数据
            indicators: 要提取的指标列表
            
        Returns:
            FeatureResult: 特征提取结果
            
        Examples:
            >>> extractor = TimeSeriesFeatureExtractor()
            >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> result = extractor.extract_technical_indicators(data, ['sma', 'rsi', 'macd'])
            >>> print(f"SMA: {result.features['sma']:.3f}")
        """
        try:
            data = pd.Series(data)
            
            if indicators is None:
                indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic']
            
            features = {}
            
            for indicator in indicators:
                if indicator == 'sma':
                    features.update(self._simple_moving_average(data))
                elif indicator == 'ema':
                    features.update(self._exponential_moving_average(data))
                elif indicator == 'rsi':
                    features.update(self._relative_strength_index(data))
                elif indicator == 'macd':
                    features.update(self._macd(data))
                elif indicator == 'bollinger':
                    features.update(self._bollinger_bands(data))
                elif indicator == 'stochastic':
                    features.update(self._stochastic_oscillator(data))
                else:
                    self.logger.warning(f"未知的技术指标: {indicator}")
            
            return FeatureResult(
                data=features,
                features=features,
                feature_names=list(features.keys())
            )
            
        except Exception as e:
            self.logger.error(f"技术指标提取失败: {e}")
            raise TimeSeriesProcessingError(f"技术指标提取失败: {e}")
    
    def _simple_moving_average(self, data: pd.Series, windows: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """简单移动平均"""
        features = {}
        for window in windows:
            if len(data) >= window:
                sma = data.rolling(window=window).mean().iloc[-1]
                features[f'sma_{window}'] = sma if not np.isnan(sma) else 0.0
        return features
    
    def _exponential_moving_average(self, data: pd.Series, spans: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """指数移动平均"""
        features = {}
        for span in spans:
            if len(data) >= span:
                ema = data.ewm(span=span).mean().iloc[-1]
                features[f'ema_{span}'] = ema if not np.isnan(ema) else 0.0
        return features
    
    def _relative_strength_index(self, data: pd.Series, period: int = 14) -> Dict[str, float]:
        """相对强弱指数"""
        if len(data) < period + 1:
            return {'rsi': 0.0}
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # 避免除零错误
        rs = gain / (loss + 1e-10)  # 添加小量避免除零
        rsi = 100 - (100 / (1 + rs))
        
        return {'rsi': rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0}
    
    def _macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """MACD指标"""
        if len(data) < slow:
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
        
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1] if not np.isnan(macd_line.iloc[-1]) else 0.0,
            'macd_signal': signal_line.iloc[-1] if not np.isnan(signal_line.iloc[-1]) else 0.0,
            'macd_histogram': histogram.iloc[-1] if not np.isnan(histogram.iloc[-1]) else 0.0
        }
    
    def _bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, float]:
        """布林带"""
        if len(data) < period:
            return {'bb_upper': 0.0, 'bb_middle': 0.0, 'bb_lower': 0.0, 'bb_width': 0.0, 'bb_position': 0.0}
        
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = data.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_middle = sma.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        bb_width = (current_upper - current_lower) / (current_middle + 1e-10) if current_middle != 0 else 0
        bb_position = (current_price - current_lower) / (current_upper - current_lower + 1e-10) if current_upper != current_lower else 0.5
        
        return {
            'bb_upper': current_upper if not np.isnan(current_upper) else 0.0,
            'bb_middle': current_middle if not np.isnan(current_middle) else 0.0,
            'bb_lower': current_lower if not np.isnan(current_lower) else 0.0,
            'bb_width': bb_width if not np.isnan(bb_width) else 0.0,
            'bb_position': bb_position if not np.isnan(bb_position) else 0.5
        }
    
    def _stochastic_oscillator(self, data: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """随机振荡器"""
        if len(data) < k_period:
            return {'stoch_k': 0.0, 'stoch_d': 0.0}
        
        # 简化实现，假设数据为价格序列
        low_min = data.rolling(window=k_period).min()
        high_max = data.rolling(window=k_period).max()
        
        k_percent = 100 * (data - low_min) / (high_max - low_min + 1e-10)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'stoch_k': k_percent.iloc[-1] if not np.isnan(k_percent.iloc[-1]) else 50.0,
            'stoch_d': d_percent.iloc[-1] if not np.isnan(d_percent.iloc[-1]) else 50.0
        }
    
    @validate_time_series
    def extract_statistical_features(
        self,
        data: TimeSeriesData,
        features: Optional[List[str]] = None
    ) -> FeatureResult:
        """
        提取统计特征
        
        Args:
            data: 时间序列数据
            features: 要提取的特征列表
            
        Returns:
            FeatureResult: 特征提取结果
        """
        try:
            data = pd.Series(data)
            
            if features is None:
                features = ['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'range', 'iqr']
            
            result_features = {}
            
            for feature in features:
                if feature == 'mean':
                    result_features['mean'] = float(data.mean())
                elif feature == 'std':
                    result_features['std'] = float(data.std())
                elif feature == 'skew':
                    result_features['skew'] = float(data.skew())
                elif feature == 'kurtosis':
                    result_features['kurtosis'] = float(data.kurtosis())
                elif feature == 'min':
                    result_features['min'] = float(data.min())
                elif feature == 'max':
                    result_features['max'] = float(data.max())
                elif feature == 'range':
                    result_features['range'] = float(data.max() - data.min())
                elif feature == 'iqr':
                    result_features['iqr'] = float(data.quantile(0.75) - data.quantile(0.25))
                elif feature == 'median':
                    result_features['median'] = float(data.median())
                elif feature == 'variance':
                    result_features['variance'] = float(data.var())
                elif feature == 'cv':  # 变异系数
                    mean_val = data.mean()
                    result_features['cv'] = float(data.std() / abs(mean_val)) if abs(mean_val) > 1e-10 else 0.0
                else:
                    self.logger.warning(f"未知的统计特征: {feature}")
            
            return FeatureResult(
                data=result_features,
                features=result_features,
                feature_names=list(result_features.keys())
            )
            
        except Exception as e:
            self.logger.error(f"统计特征提取失败: {e}")
            raise TimeSeriesProcessingError(f"统计特征提取失败: {e}")
    
    @validate_time_series
    def extract_frequency_features(
        self,
        data: TimeSeriesData,
        max_freq: Optional[int] = None
    ) -> FeatureResult:
        """
        提取频域特征
        
        Args:
            data: 时间序列数据
            max_freq: 最大频率分量数
            
        Returns:
            FeatureResult: 特征提取结果
        """
        try:
            data = pd.Series(data).values
            
            # FFT变换
            fft_vals = fft(data)
            freqs = fftfreq(len(data))
            
            # 幅值谱
            magnitude = np.abs(fft_vals)
            
            # 功率谱
            power_spectrum = magnitude ** 2
            
            # 计算主要特征
            half_magnitude = magnitude[1:len(magnitude)//2]
            half_freqs = freqs[1:len(freqs)//2]
            half_power = power_spectrum[1:len(power_spectrum)//2]
            
            if len(half_magnitude) > 0:
                dominant_idx = np.argmax(half_magnitude)
                dominant_frequency = float(half_freqs[dominant_idx])
                dominant_magnitude = float(half_magnitude[dominant_idx])
            else:
                dominant_frequency = 0.0
                dominant_magnitude = 0.0
            
            magnitude_sum = np.sum(magnitude[:len(magnitude)//2])
            spectral_centroid = float(np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / 
                                    magnitude_sum) if magnitude_sum > 0 else 0.0
            
            features = {
                'dominant_frequency': dominant_frequency,
                'dominant_magnitude': dominant_magnitude,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': float(self._spectral_rolloff(magnitude, freqs)),
                'spectral_bandwidth': float(self._spectral_bandwidth(magnitude, freqs)),
                'total_energy': float(np.sum(power_spectrum)),
                'energy_entropy': float(self._energy_entropy(power_spectrum))
            }
            
            # 添加主要频率分量
            if max_freq is None:
                max_freq = min(10, len(magnitude) // 4)
            
            sorted_indices = np.argsort(magnitude[1:len(magnitude)//2])[::-1][:max_freq]
            for i, idx in enumerate(sorted_indices):
                features[f'freq_{i+1}_magnitude'] = float(magnitude[idx + 1])
                features[f'freq_{i+1}_frequency'] = float(freqs[idx + 1])
            
            return FeatureResult(
                data=features,
                features=features,
                feature_names=list(features.keys())
            )
            
        except Exception as e:
            self.logger.error(f"频域特征提取失败: {e}")
            raise TimeSeriesProcessingError(f"频域特征提取失败: {e}")
    
    def _spectral_rolloff(self, magnitude: np.ndarray, freqs: np.ndarray) -> float:
        """频谱滚降点"""
        total_energy = np.sum(magnitude)
        threshold = 0.85 * total_energy
        
        cumulative_energy = np.cumsum(magnitude)
        rolloff_idx = np.where(cumulative_energy >= threshold)[0]
        
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[len(freqs)//2]
    
    def _spectral_bandwidth(self, magnitude: np.ndarray, freqs: np.ndarray) -> float:
        """频谱带宽"""
        half_magnitude = magnitude[:len(magnitude)//2]
        half_freqs = freqs[:len(freqs)//2]
        
        magnitude_sum = np.sum(half_magnitude)
        if magnitude_sum == 0:
            return 0.0
            
        centroid = np.sum(half_freqs * half_magnitude) / magnitude_sum
        bandwidth = np.sqrt(np.sum(((half_freqs - centroid) ** 2) * half_magnitude) / magnitude_sum)
        return bandwidth
    
    def _energy_entropy(self, power_spectrum: np.ndarray) -> float:
        """能量熵"""
        power_spectrum = power_spectrum / np.sum(power_spectrum)  # 归一化
        power_spectrum = power_spectrum[power_spectrum > 0]  # 去除零值
        
        if len(power_spectrum) == 0:
            return 0.0
        
        entropy = -np.sum(power_spectrum * np.log2(power_spectrum))
        return entropy
    
    @validate_time_series
    def extract_all_features(
        self,
        data: TimeSeriesData,
        include_technical: bool = True,
        include_statistical: bool = True,
        include_frequency: bool = True
    ) -> FeatureResult:
        """
        提取所有类型特征
        
        Args:
            data: 时间序列数据
            include_technical: 是否包含技术指标
            include_statistical: 是否包含统计特征
            include_frequency: 是否包含频域特征
            
        Returns:
            FeatureResult: 特征提取结果
        """
        try:
            all_features = {}
            
            if include_technical:
                tech_result = self.extract_technical_indicators(data)
                all_features.update(tech_result.features)
            
            if include_statistical:
                stat_result = self.extract_statistical_features(data)
                all_features.update(stat_result.features)
            
            if include_frequency:
                freq_result = self.extract_frequency_features(data)
                all_features.update(freq_result.features)
            
            return FeatureResult(
                data=all_features,
                features=all_features,
                feature_names=list(all_features.keys())
            )
            
        except Exception as e:
            self.logger.error(f"全特征提取失败: {e}")
            raise TimeSeriesProcessingError(f"全特征提取失败: {e}")


class MultiTimeScaleProcessor:
    """
    多时间尺度处理器
    
    提供不同时间尺度的数据处理和同步功能
    """
    
    def __init__(self, executor: Optional[ThreadPoolExecutor] = None):
        """
        初始化多时间尺度处理器
        
        Args:
            executor: 线程池执行器，用于异步处理
        """
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @validate_time_series
    def resample_time_series(
        self,
        data: TimeSeriesData,
        target_freq: str,
        method: str = 'mean'
    ) -> TimeSeriesResult:
        """
        时间序列重采样
        
        Args:
            data: 时间序列数据
            target_freq: 目标频率 ('D', 'H', 'M', 'W'等)
            method: 重采样方法 ('mean', 'sum', 'max', 'min', 'last', 'first')
            
        Returns:
            TimeSeriesResult: 重采样结果
            
        Examples:
            >>> processor = MultiTimeScaleProcessor()
            >>> data = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('2023-01-01', periods=6, freq='H'))
            >>> result = processor.resample_time_series(data, 'D', 'sum')
            >>> print(f"重采样结果: {result.data}")
        """
        try:
            if isinstance(data, (list, np.ndarray)):
                # 如果没有时间索引，创建默认索引
                data = pd.Series(data, index=pd.date_range('2023-01-01', periods=len(data), freq='H'))
            else:
                data = pd.Series(data)
            
            # 确保索引为datetime类型
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # 重采样
            if method == 'mean':
                resampled = data.resample(target_freq).mean()
            elif method == 'sum':
                resampled = data.resample(target_freq).sum()
            elif method == 'max':
                resampled = data.resample(target_freq).max()
            elif method == 'min':
                resampled = data.resample(target_freq).min()
            elif method == 'last':
                resampled = data.resample(target_freq).last()
            elif method == 'first':
                resampled = data.resample(target_freq).first()
            else:
                raise TimeSeriesValidationError(f"不支持的重采样方法: {method}")
            
            # 去除NaN值
            resampled = resampled.dropna()
            
            return TimeSeriesResult(
                data=resampled.values,
                metadata={
                    'target_freq': target_freq,
                    'method': method,
                    'original_length': len(data),
                    'resampled_length': len(resampled),
                    'index': resampled.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                }
            )
            
        except Exception as e:
            self.logger.error(f"时间序列重采样失败: {e}")
            raise TimeSeriesProcessingError(f"时间序列重采样失败: {e}")
    
    @validate_time_series
    def aggregate_multiple_scales(
        self,
        data: TimeSeriesData,
        scales: List[str],
        methods: Optional[Dict[str, str]] = None
    ) -> Dict[str, TimeSeriesResult]:
        """
        多尺度聚合
        
        Args:
            data: 时间序列数据
            scales: 时间尺度列表
            methods: 各尺度的聚合方法
            
        Returns:
            Dict[str, TimeSeriesResult]: 各尺度的聚合结果
        """
        try:
            if methods is None:
                methods = {scale: 'mean' for scale in scales}
            
            results = {}
            
            for scale in scales:
                method = methods.get(scale, 'mean')
                result = self.resample_time_series(data, scale, method)
                results[scale] = result
            
            return results
            
        except Exception as e:
            self.logger.error(f"多尺度聚合失败: {e}")
            raise TimeSeriesProcessingError(f"多尺度聚合失败: {e}")
    
    @validate_time_series
    def synchronize_time_series(
        self,
        series_list: List[TimeSeriesData],
        target_freq: str,
        method: str = 'mean'
    ) -> List[TimeSeriesResult]:
        """
        同步多个时间序列
        
        Args:
            series_list: 时间序列列表
            target_freq: 目标频率
            method: 同步方法
            
        Returns:
            List[TimeSeriesResult]: 同步后的时间序列列表
        """
        try:
            synchronized_series = []
            
            for i, series in enumerate(series_list):
                result = self.resample_time_series(series, target_freq, method)
                synchronized_series.append(result)
            
            return synchronized_series
            
        except Exception as e:
            self.logger.error(f"时间序列同步失败: {e}")
            raise TimeSeriesProcessingError(f"时间序列同步失败: {e}")
    
    @validate_time_series
    def sliding_window_analysis(
        self,
        data: TimeSeriesData,
        window_size: int,
        step_size: int,
        analyzer_func: Callable
    ) -> List[TimeSeriesResult]:
        """
        滑动窗口分析
        
        Args:
            data: 时间序列数据
            window_size: 窗口大小
            step_size: 步长
            analyzer_func: 分析函数
            
        Returns:
            List[TimeSeriesResult]: 滑动窗口分析结果
        """
        try:
            data = pd.Series(data)
            results = []
            
            for start_idx in range(0, len(data) - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window_data = data.iloc[start_idx:end_idx]
                
                try:
                    result = analyzer_func(window_data)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"窗口分析失败 (索引 {start_idx}-{end_idx}): {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"滑动窗口分析失败: {e}")
            raise TimeSeriesProcessingError(f"滑动窗口分析失败: {e}")


class AsyncTimeSeriesProcessor:
    """
    异步时间序列处理器
    
    提供异步处理功能，支持并发处理多个时间序列
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化异步时间序列处理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def process_multiple_series(
        self,
        series_list: List[TimeSeriesData],
        processor_func: Callable,
        **kwargs
    ) -> List[Any]:
        """
        异步处理多个时间序列
        
        Args:
            series_list: 时间序列列表
            processor_func: 处理函数
            **kwargs: 处理函数参数
            
        Returns:
            List[Any]: 处理结果列表
            
        Examples:
            >>> async_processor = AsyncTimeSeriesProcessor()
            >>> series_list = [pd.Series([1, 2, 3]), pd.Series([4, 5, 6])]
            >>> analyzer = TimeSeriesAnalyzer()
            >>> results = await async_processor.process_multiple_series(
            ...     series_list, analyzer.analyze_trend, method='linear'
            ... )
            >>> print(f"处理结果数量: {len(results)}")
        """
        try:
            tasks = []
            
            for series in series_list:
                task = asyncio.create_task(
                    self._process_single_series(series, processor_func, **kwargs)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"时间序列 {i} 处理失败: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"异步处理失败: {e}")
            raise TimeSeriesProcessingError(f"异步处理失败: {e}")
    
    async def _process_single_series(
        self,
        series: TimeSeriesData,
        processor_func: Callable,
        **kwargs
    ) -> Any:
        """处理单个时间序列"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            lambda: processor_func(series, **kwargs)
        )
    
    async def batch_forecast(
        self,
        series_list: List[TimeSeriesData],
        forecaster: TimeSeriesForecaster,
        steps: int,
        methods: Optional[List[str]] = None
    ) -> Dict[str, List[Any]]:
        """
        批量预测
        
        Args:
            series_list: 时间序列列表
            forecaster: 预测器实例
            steps: 预测步数
            methods: 预测方法列表
            
        Returns:
            Dict[str, List[Any]]: 各方法的预测结果
        """
        try:
            if methods is None:
                methods = ['arima', 'exponential_smoothing', 'state_space']
            
            results = {}
            
            for method in methods:
                if method == 'arima':
                    processor_func = lambda series: forecaster.forecast_arima(series, steps)
                elif method == 'exponential_smoothing':
                    processor_func = lambda series: forecaster.forecast_exponential_smoothing(series, steps)
                elif method == 'state_space':
                    processor_func = lambda series: forecaster.forecast_state_space(series, steps)
                else:
                    self.logger.warning(f"未知的预测方法: {method}")
                    continue
                
                method_results = await self.process_multiple_series(series_list, processor_func)
                results[method] = method_results
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量预测失败: {e}")
            raise TimeSeriesProcessingError(f"批量预测失败: {e}")
    
    async def batch_feature_extraction(
        self,
        series_list: List[TimeSeriesData],
        extractor: TimeSeriesFeatureExtractor,
        feature_types: Optional[List[str]] = None
    ) -> List[FeatureResult]:
        """
        批量特征提取
        
        Args:
            series_list: 时间序列列表
            extractor: 特征提取器实例
            feature_types: 特征类型列表
            
        Returns:
            List[FeatureResult]: 特征提取结果列表
        """
        try:
            if feature_types is None:
                feature_types = ['technical', 'statistical', 'frequency']
            
            def extract_features(series):
                if 'technical' in feature_types and 'statistical' in feature_types and 'frequency' in feature_types:
                    return extractor.extract_all_features(series)
                elif 'technical' in feature_types:
                    return extractor.extract_technical_indicators(series)
                elif 'statistical' in feature_types:
                    return extractor.extract_statistical_features(series)
                elif 'frequency' in feature_types:
                    return extractor.extract_frequency_features(series)
                else:
                    return extractor.extract_statistical_features(series)
            
            results = await self.process_multiple_series(series_list, extract_features)
            
            return [r for r in results if r is not None]
            
        except Exception as e:
            self.logger.error(f"批量特征提取失败: {e}")
            raise TimeSeriesProcessingError(f"批量特征提取失败: {e}")
    
    def close(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)


class BatchTimeSeriesProcessor:
    """
    批处理时间序列处理器
    
    提供高效的批处理功能，支持大规模时间序列数据处理
    """
    
    def __init__(self, max_workers: int = 4, batch_size: int = 100):
        """
        初始化批处理器
        
        Args:
            max_workers: 最大工作线程数
            batch_size: 批处理大小
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.async_processor = AsyncTimeSeriesProcessor(max_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_large_dataset(
        self,
        dataset: Dict[str, TimeSeriesData],
        processor_func: Callable,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理大规模数据集
        
        Args:
            dataset: 数据集字典 {name: series}
            processor_func: 处理函数
            progress_callback: 进度回调函数
            **kwargs: 处理函数参数
            
        Returns:
            Dict[str, Any]: 处理结果字典
            
        Examples:
            >>> batch_processor = BatchTimeSeriesProcessor()
            >>> dataset = {
            ...     'series1': pd.Series([1, 2, 3, 4, 5]),
            ...     'series2': pd.Series([6, 7, 8, 9, 10])
            ... }
            >>> analyzer = TimeSeriesAnalyzer()
            >>> results = batch_processor.process_large_dataset(
            ...     dataset, analyzer.analyze_trend, method='linear'
            ... )
        """
        try:
            results = {}
            series_items = list(dataset.items())
            total_series = len(series_items)
            
            # 分批处理
            for batch_start in range(0, total_series, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_series)
                batch_items = series_items[batch_start:batch_end]
                
                # 异步处理当前批次
                batch_data = [series for _, series in batch_items]
                batch_names = [name for name, _ in batch_items]
                
                batch_results = asyncio.run(
                    self.async_processor.process_multiple_series(
                        batch_data, processor_func, **kwargs
                    )
                )
                
                # 合并结果
                for name, result in zip(batch_names, batch_results):
                    results[name] = result
                
                # 进度回调
                if progress_callback:
                    progress = (batch_end / total_series) * 100
                    progress_callback(progress, batch_end, total_series)
            
            return results
            
        except Exception as e:
            self.logger.error(f"大规模数据集处理失败: {e}")
            raise TimeSeriesProcessingError(f"大规模数据集处理失败: {e}")
    
    def parallel_analysis_pipeline(
        self,
        series_list: List[TimeSeriesData],
        pipeline_steps: List[Tuple[str, Callable, Dict]],
        results_cache: Optional[Dict] = None
    ) -> Dict[str, List[Any]]:
        """
        并行分析流水线
        
        Args:
            series_list: 时间序列列表
            pipeline_steps: 流水线步骤 [(step_name, function, params), ...]
            results_cache: 结果缓存
            
        Returns:
            Dict[str, List[Any]]: 各步骤的结果
        """
        try:
            if results_cache is None:
                results_cache = {'input': series_list}
            
            pipeline_results = {}
            
            for step_name, step_func, step_params in pipeline_steps:
                self.logger.info(f"执行流水线步骤: {step_name}")
                
                # 获取输入数据
                if step_name in results_cache:
                    input_data = results_cache[step_name]
                else:
                    input_data = results_cache.get('input', series_list)
                
                # 执行步骤
                step_results = asyncio.run(
                    self.async_processor.process_multiple_series(
                        input_data, step_func, **step_params
                    )
                )
                
                pipeline_results[step_name] = step_results
                results_cache[step_name] = step_results
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"并行分析流水线失败: {e}")
            raise TimeSeriesProcessingError(f"并行分析流水线失败: {e}")
    
    def save_processing_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        format: str = 'json'
    ) -> None:
        """
        保存处理结果
        
        Args:
            results: 处理结果
            output_dir: 输出目录
            format: 保存格式 ('json', 'pickle', 'csv')
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            if format == 'json':
                # 保存为JSON格式
                for name, result in results.items():
                    if hasattr(result, 'to_dict'):
                        result_dict = result.to_dict()
                    else:
                        result_dict = {'data': result}
                    
                    filepath = os.path.join(output_dir, f"{name}.json")
                    with open(filepath, 'w') as f:
                        json.dump(result_dict, f, indent=2, default=str)
            
            elif format == 'pickle':
                # 保存为Pickle格式
                filepath = os.path.join(output_dir, "results.pkl")
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f)
            
            elif format == 'csv':
                # 保存为CSV格式（适用于数值结果）
                import pandas as pd
                
                for name, result in results.items():
                    if hasattr(result, 'data') and isinstance(result.data, (list, np.ndarray)):
                        df = pd.DataFrame({name: result.data})
                        filepath = os.path.join(output_dir, f"{name}.csv")
                        df.to_csv(filepath, index=False)
            
            self.logger.info(f"处理结果已保存到: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"保存处理结果失败: {e}")
            raise TimeSeriesProcessingError(f"保存处理结果失败: {e}")
    
    def close(self):
        """关闭批处理器"""
        self.async_processor.close()
        self.executor.shutdown(wait=True)


# 使用示例和测试代码
if __name__ == "__main__":
    # 基本使用示例
    print("=== J3时间序列工具模块使用示例 ===\n")
    
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    trend = np.linspace(10, 20, 100)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(100) / 30)
    noise = np.random.normal(0, 1, 100)
    data = pd.Series(trend + seasonal + noise, index=dates)
    
    print(f"示例数据: {len(data)} 个数据点")
    print(f"数据范围: {data.min():.2f} - {data.max():.2f}\n")
    
    # 1. 时间序列分析
    print("1. 趋势分析")
    analyzer = TimeSeriesAnalyzer()
    trend_result = analyzer.analyze_trend(data, method='linear')
    print(f"   趋势斜率: {trend_result.slope:.3f}")
    print(f"   R²: {trend_result.r_squared:.3f}\n")
    
    # 2. 季节性分解
    print("2. 季节性分解")
    seasonal_result = analyzer.decompose_seasonal(data, period=30)
    print(f"   季节性强度: {seasonal_result.seasonal_strength:.3f}\n")
    
    # 3. 时间序列预测
    print("3. 时间序列预测")
    forecaster = TimeSeriesForecaster()
    
    # ARIMA预测
    arima_result = forecaster.forecast_arima(data, steps=10, order=(1, 1, 1))
    print(f"   ARIMA预测均值: {np.mean(arima_result.forecast):.2f}")
    
    # 指数平滑预测
    exp_result = forecaster.forecast_exponential_smoothing(data, steps=10, method='double')
    print(f"   指数平滑预测均值: {np.mean(exp_result.forecast):.2f}\n")
    
    # 4. 时间序列变换
    print("4. 时间序列变换")
    transformer = TimeSeriesTransformer()
    
    # 差分变换
    diff_result = transformer.difference(data, order=1)
    print(f"   差分变换后长度: {len(diff_result.data)}")
    
    # 标准化
    std_result = transformer.standardize(data, method='zscore')
    print(f"   标准化后均值: {np.mean(std_result.data):.3f}")
    print(f"   标准化后标准差: {np.std(std_result.data):.3f}\n")
    
    # 5. 特征提取
    print("5. 特征提取")
    extractor = TimeSeriesFeatureExtractor()
    
    # 技术指标
    tech_features = extractor.extract_technical_indicators(data, ['sma', 'rsi', 'macd'])
    print(f"   SMA(20): {tech_features.features.get('sma_20', 'N/A')}")
    print(f"   RSI: {tech_features.features.get('rsi', 'N/A'):.2f}")
    
    # 统计特征
    stat_features = extractor.extract_statistical_features(data)
    print(f"   均值: {stat_features.features['mean']:.2f}")
    print(f"   标准差: {stat_features.features['std']:.2f}")
    print(f"   偏度: {stat_features.features['skew']:.3f}\n")
    
    # 6. 多时间尺度处理
    print("6. 多时间尺度处理")
    multi_processor = MultiTimeScaleProcessor()
    
    # 重采样到周频率
    weekly_result = multi_processor.resample_time_series(data, 'W', 'mean')
    print(f"   重采样后长度: {len(weekly_result.data)}")
    
    # 多尺度聚合
    scales = ['D', 'W', 'M']
    multi_results = multi_processor.aggregate_multiple_scales(data, scales)
    for scale, result in multi_results.items():
        print(f"   {scale}频率数据长度: {len(result.data)}\n")
    
    # 7. 异步处理示例
    print("7. 异步处理示例")
    async_processor = AsyncTimeSeriesProcessor()
    
    # 创建多个时间序列
    series_list = [data[:50], data[25:75], data[50:]]
    
    # 批量预测
    async def run_async_example():
        batch_results = await async_processor.batch_forecast(
            series_list, forecaster, steps=5, methods=['arima', 'exponential_smoothing']
        )
        
        for method, results in batch_results.items():
            print(f"   {method}方法预测结果数: {len([r for r in results if r is not None])}")
    
    asyncio.run(run_async_example())
    async_processor.close()
    
    print("\n=== 示例运行完成 ===")
    
    # 8. 批处理示例
    print("\n8. 批处理示例")
    batch_processor = BatchTimeSeriesProcessor(batch_size=2)
    
    # 创建测试数据集
    test_dataset = {
        f'series_{i}': pd.Series(np.random.randn(50) + i, 
                               index=pd.date_range('2023-01-01', periods=50, freq='D'))
        for i in range(5)
    }
    
    # 批量分析
    def progress_callback(progress, current, total):
        print(f"   进度: {progress:.1f}% ({current}/{total})")
    
    batch_results = batch_processor.process_large_dataset(
        test_dataset, 
        analyzer.analyze_trend, 
        method='linear',
        progress_callback=progress_callback
    )
    
    print(f"   批处理完成，处理了 {len(batch_results)} 个时间序列")
    batch_processor.close()
    
    print("\n=== J3时间序列工具模块测试完成 ===")
