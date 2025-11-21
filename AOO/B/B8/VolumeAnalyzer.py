#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B8成交量分析器
Volume Analyzer for B8 Module

功能包括：
1. 成交量模式识别和分析
2. 成交量异常检测
3. 成交量价格关系分析（OBV、VWAP等）
4. 成交量趋势分析
5. 成交量背离分析
6. 机构资金流向分析
7. 成交量交易信号生成


创建时间: 2025-11-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime, timedelta
# 尝试导入talib库，如果没有则使用fallback
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    talib = None
    TALIB_AVAILABLE = False

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolumePattern(Enum):
    """成交量模式枚举"""
    NORMAL = "正常"
    HIGH_VOLUME = "高成交量"
    LOW_VOLUME = "低成交量"
    SPIKE = "成交量激增"
    ACCUMULATION = "吸筹"
    DISTRIBUTION = "派发"
    BREAKOUT = "突破"
    REVERSAL = "反转"

class VolumeAnomaly(Enum):
    """成交量异常枚举"""
    NONE = "无异常"
    UNUSUAL_HIGH = "异常高成交量"
    UNUSUAL_LOW = "异常低成交量"
    VOLUME_SPIKE = "成交量激增"
    VOLUME_DRY_UP = "成交量枯竭"

class DivergenceType(Enum):
    """背离类型枚举"""
    NONE = "无背离"
    BULLISH_DIVERGENCE = "看涨背离"
    BEARISH_DIVERGENCE = "看跌背离"
    HIDDEN_BULLISH = "隐藏看涨背离"
    HIDDEN_BEARISH = "隐藏看跌背离"

@dataclass
class VolumeSignal:
    """成交量信号数据结构"""
    timestamp: datetime
    signal_type: str
    strength: float  # 信号强度 (0-1)
    description: str
    volume_ratio: float
    price_change: float

@dataclass
class VolumeAnalysisResult:
    """成交量分析结果"""
    timestamp: datetime
    volume_pattern: VolumePattern
    anomaly: VolumeAnomaly
    divergence: DivergenceType
    signals: List[VolumeSignal]
    metrics: Dict[str, float]
    confidence: float

class VolumeAnalyzer:
    """B8成交量分析器"""
    
    def __init__(self, lookback_periods: int = 20, anomaly_threshold: float = 2.0):
        """
        初始化成交量分析器
        
        Args:
            lookback_periods: 回看周期数
            anomaly_threshold: 异常检测阈值
        """
        self.lookback_periods = lookback_periods
        self.anomaly_threshold = anomaly_threshold
        self.volume_history = []
        self.price_history = []
        
        # 成交量指标参数
        self.obv_values = []
        self.vwap_values = []
        self.volume_ma_short = []
        self.volume_ma_long = []
        
        # 异常检测参数
        self.volume_percentiles = {}
        
        # 预警设置
        self.alert_callbacks = []
        
    def calculate_basic_volume_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        计算基础成交量指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            基础成交量指标字典
        """
        volume = data['volume']
        close = data['close']
        
        # 基础统计指标
        metrics = {
            'volume_mean': volume.mean(),
            'volume_std': volume.std(),
            'volume_median': volume.median(),
            'volume_max': volume.max(),
            'volume_min': volume.min(),
            'volume_current': volume.iloc[-1],
            'volume_ratio': volume.iloc[-1] / volume.mean(),  # 当前成交量与平均成交量比值
        }
        
        # 成交量移动平均
        metrics['volume_ma5'] = volume.rolling(5).mean().iloc[-1]
        metrics['volume_ma10'] = volume.rolling(10).mean().iloc[-1]
        metrics['volume_ma20'] = volume.rolling(20).mean().iloc[-1]
        
        # 成交量相对强度
        if metrics['volume_ma20'] > 0:
            metrics['volume_rsi'] = (volume.iloc[-1] / metrics['volume_ma20'] - 1) * 100
        
        return metrics
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """
        计算能量潮指标(OBV)
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            OBV指标序列
        """
        close = data['close']
        volume = data['volume']
        
        obv = np.zeros(len(close))
        obv[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv[i] = obv[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv[i] = obv[i-1] - volume.iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=close.index)
    
    def calculate_vwap(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        计算成交量加权平均价格(VWAP)
        
        Args:
            data: 包含OHLCV数据的DataFrame
            window: 计算窗口
            
        Returns:
            VWAP指标序列
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window).sum() / data['volume'].rolling(window).sum()
        
        return vwap
    
    def calculate_volume_profile(self, data: pd.DataFrame, bins: int = 50) -> Dict[str, pd.Series]:
        """
        计算成交量分布
        
        Args:
            data: 包含OHLCV数据的DataFrame
            bins: 价格分箱数量
            
        Returns:
            成交量分布数据
        """
        high = data['high'].max()
        low = data['low'].min()
        price_range = np.linspace(low, high, bins)
        
        volume_profile = {}
        
        for i, price_level in enumerate(price_range[:-1]):
            # 计算每个价格区间的成交量
            mask = (data['low'] <= price_level) & (data['high'] >= price_level[i+1])
            volume_at_level = data.loc[mask, 'volume'].sum()
            volume_profile[f'price_{i}'] = volume_at_level
        
        return volume_profile
    
    def detect_volume_patterns(self, data: pd.DataFrame) -> VolumePattern:
        """
        检测成交量模式
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            成交量模式
        """
        current_volume = data['volume'].iloc[-1]
        volume_mean = data['volume'].rolling(self.lookback_periods).mean().iloc[-1]
        volume_std = data['volume'].rolling(self.lookback_periods).std().iloc[-1]
        
        # 计算价格变化
        price_change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
        
        # 成交量激增检测
        if current_volume > volume_mean + self.anomaly_threshold * volume_std:
            if abs(price_change) > 0.02:  # 价格大幅变化
                return VolumePattern.SPIKE
            else:
                return VolumePattern.HIGH_VOLUME
        
        # 低成交量检测
        elif current_volume < volume_mean * 0.5:
            return VolumePattern.LOW_VOLUME
        
        # 吸筹派发模式检测
        elif len(data) >= 10:
            recent_volume = data['volume'].rolling(5).mean().iloc[-1]
            recent_price_change = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
            
            if recent_volume > volume_mean and recent_price_change < 0:
                return VolumePattern.ACCUMULATION  # 吸筹
            elif recent_volume > volume_mean and recent_price_change > 0:
                return VolumePattern.DISTRIBUTION  # 派发
        
        # 突破模式检测
        if abs(price_change) > 0.015 and current_volume > volume_mean * 1.2:
            return VolumePattern.BREAKOUT
        
        # 反转模式检测
        if len(data) >= 3:
            price_trend = np.polyfit(range(3), data['close'].iloc[-3:].values, 1)[0]
            volume_trend = np.polyfit(range(3), data['volume'].iloc[-3:].values, 1)[0]
            
            if price_trend * volume_trend < 0:  # 价格和成交量趋势相反
                return VolumePattern.REVERSAL
        
        return VolumePattern.NORMAL
    
    def detect_volume_anomalies(self, data: pd.DataFrame) -> VolumeAnomaly:
        """
        检测成交量异常
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            成交量异常类型
        """
        current_volume = data['volume'].iloc[-1]
        volume_mean = data['volume'].rolling(self.lookback_periods).mean().iloc[-1]
        volume_std = data['volume'].rolling(self.lookback_periods).std().iloc[-1]
        
        # 计算Z-score
        z_score = (current_volume - volume_mean) / volume_std if volume_std > 0 else 0
        
        if z_score > self.anomaly_threshold:
            return VolumeAnomaly.UNUSUAL_HIGH
        elif z_score < -self.anomaly_threshold:
            return VolumeAnomaly.UNUSUAL_LOW
        elif current_volume > volume_mean * 3:
            return VolumeAnomaly.VOLUME_SPIKE
        elif current_volume < volume_mean * 0.1:
            return VolumeAnomaly.VOLUME_DRY_UP
        else:
            return VolumeAnomaly.NONE
    
    def analyze_price_volume_relationship(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        分析价格与成交量关系
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            价格成交量关系指标
        """
        close = data['close']
        volume = data['volume']
        
        # 计算相关系数
        price_volume_corr = close.corr(volume)
        
        # 计算价格变化与成交量变化的关系
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        
        relationship_metrics = {
            'price_volume_correlation': price_volume_corr,
            'price_volume_change_correlation': price_change.corr(volume_change),
            'volume_price_trend_alignment': self._calculate_trend_alignment(close, volume),
        }
        
        # OBV分析
        obv = self.calculate_obv(data)
        obv_trend = self._calculate_trend(obv)
        price_trend = self._calculate_trend(close)
        
        relationship_metrics['obv_price_trend_match'] = 1 if obv_trend * price_trend > 0 else -1
        
        return relationship_metrics
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """计算趋势方向"""
        if len(series) < 2:
            return 0
        
        recent_values = series.tail(5).values
        if len(recent_values) < 2:
            return 0
        
        slope, _, _, _, _ = stats.linregress(range(len(recent_values)), recent_values)
        return slope
    
    def _calculate_trend_alignment(self, price: pd.Series, volume: pd.Series) -> float:
        """计算价格和成交量趋势对齐度"""
        price_trend = self._calculate_trend(price)
        volume_trend = self._calculate_trend(volume)
        
        if price_trend == 0 or volume_trend == 0:
            return 0
        
        # 计算趋势对齐度 (-1到1)
        alignment = np.sign(price_trend) * np.sign(volume_trend)
        return alignment
    
    def analyze_volume_trends(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        分析成交量趋势
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            成交量趋势指标
        """
        volume = data['volume']
        
        # 成交量趋势分析
        short_ma = volume.rolling(5).mean()
        long_ma = volume.rolling(20).mean()
        
        trend_metrics = {
            'volume_ma5': short_ma.iloc[-1],
            'volume_ma20': long_ma.iloc[-1],
            'volume_trend_strength': self._calculate_trend(short_ma),
            'volume_acceleration': self._calculate_acceleration(volume),
            'volume_volatility': volume.rolling(10).std().iloc[-1],
        }
        
        # 成交量方向性指标
        if trend_metrics['volume_ma20'] > 0:
            trend_metrics['volume_trend_direction'] = (trend_metrics['volume_ma5'] / trend_metrics['volume_ma20'] - 1) * 100
        
        # 成交量动量
        volume_momentum = volume.pct_change(5).iloc[-1] * 100
        trend_metrics['volume_momentum'] = volume_momentum
        
        return trend_metrics
    
    def _calculate_acceleration(self, series: pd.Series) -> float:
        """计算加速度"""
        if len(series) < 3:
            return 0
        
        recent_values = series.tail(3).values
        if len(recent_values) < 3:
            return 0
        
        # 计算二阶差分
        first_diff = np.diff(recent_values)
        acceleration = first_diff[-1] - first_diff[-2] if len(first_diff) >= 2 else 0
        
        return acceleration
    
    def detect_volume_divergence(self, data: pd.DataFrame, lookback: int = 10) -> DivergenceType:
        """
        检测成交量背离
        
        Args:
            data: 包含OHLCV数据的DataFrame
            lookback: 回看周期
            
        Returns:
            背离类型
        """
        if len(data) < lookback:
            return DivergenceType.NONE
        
        close = data['close']
        volume = data['volume']
        
        # 获取最近的价格和成交量数据
        recent_close = close.tail(lookback)
        recent_volume = volume.tail(lookback)
        
        # 计算价格和成交量的趋势
        price_trend = self._calculate_trend(recent_close)
        volume_trend = self._calculate_trend(recent_volume)
        
        # 检测看涨背离（价格创新低但成交量萎缩）
        price_lows = self._find_local_lows(recent_close)
        volume_at_lows = [recent_volume.iloc[i] for i in price_lows]
        
        if len(price_lows) >= 2 and len(volume_at_lows) >= 2:
            if (recent_close.iloc[price_lows[-1]] < recent_close.iloc[price_lows[-2]] and 
                volume_at_lows[-1] > volume_at_lows[-2]):
                return DivergenceType.BULLISH_DIVERGENCE
        
        # 检测看跌背离（价格创新高但成交量萎缩）
        price_highs = self._find_local_highs(recent_close)
        volume_at_highs = [recent_volume.iloc[i] for i in price_highs]
        
        if len(price_highs) >= 2 and len(volume_at_highs) >= 2:
            if (recent_close.iloc[price_highs[-1]] > recent_close.iloc[price_highs[-2]] and 
                volume_at_highs[-1] < volume_at_highs[-2]):
                return DivergenceType.BEARISH_DIVERGENCE
        
        # 隐藏背离检测
        if price_trend > 0 and volume_trend < 0:
            return DivergenceType.HIDDEN_BEARISH
        elif price_trend < 0 and volume_trend > 0:
            return DivergenceType.HIDDEN_BULLISH
        
        return DivergenceType.NONE
    
    def _find_local_lows(self, series: pd.Series) -> List[int]:
        """找到局部低点"""
        lows = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] < series.iloc[i-1] and series.iloc[i] < series.iloc[i+1]:
                lows.append(i)
        return lows
    
    def _find_local_highs(self, series: pd.Series) -> List[int]:
        """找到局部高点"""
        highs = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                highs.append(i)
        return highs
    
    def analyze_institutional_flow(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        分析机构资金流向
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            机构资金流向指标
        """
        # 简化版机构资金流向分析
        # 实际应用中需要更复杂的数据源
        
        close = data['close']
        volume = data['volume']
        high = data['high']
        low = data['low']
        
        # 计算资金流向指标
        money_flow = close * volume
        
        # 机构活动强度
        institutional_activity = {
            'net_volume_flow': money_flow.tail(5).sum(),
            'volume_concentration': self._calculate_volume_concentration(volume),
            'price_volume_efficiency': self._calculate_price_volume_efficiency(close, volume),
            'large_trade_ratio': self._estimate_large_trade_ratio(volume),
        }
        
        # 资金流向方向
        recent_flow = money_flow.tail(10)
        institutional_activity['flow_direction'] = 1 if recent_flow.iloc[-1] > recent_flow.mean() else -1
        
        # 资金流向强度
        flow_std = recent_flow.std()
        institutional_activity['flow_strength'] = abs(recent_flow.iloc[-1] - recent_flow.mean()) / flow_std if flow_std > 0 else 0
        
        return institutional_activity
    
    def _calculate_volume_concentration(self, volume: pd.Series) -> float:
        """计算成交量集中度"""
        if len(volume) < 5:
            return 0
        
        recent_volume = volume.tail(5)
        total_volume = recent_volume.sum()
        
        if total_volume == 0:
            return 0
        
        # 计算赫芬达尔指数
        hhi = sum((v / total_volume) ** 2 for v in recent_volume)
        return hhi
    
    def _calculate_price_volume_efficiency(self, price: pd.Series, volume: pd.Series) -> float:
        """计算价格成交量效率"""
        if len(price) < 2 or len(volume) < 2:
            return 0
        
        price_change = abs(price.iloc[-1] - price.iloc[-2])
        volume_change = abs(volume.iloc[-1] - volume.iloc[-2])
        
        if volume_change == 0:
            return 0
        
        efficiency = price_change / volume_change
        return efficiency
    
    def _estimate_large_trade_ratio(self, volume: pd.Series) -> float:
        """估算大单占比"""
        if len(volume) < 20:
            return 0
        
        recent_volume = volume.tail(20)
        volume_mean = recent_volume.mean()
        volume_std = recent_volume.std()
        
        # 假设超过1.5倍标准差的成交量为大单
        large_trade_threshold = volume_mean + 1.5 * volume_std
        large_trades = recent_volume[recent_volume > large_trade_threshold]
        
        ratio = len(large_trades) / len(recent_volume)
        return ratio
    
    def generate_volume_signals(self, data: pd.DataFrame) -> List[VolumeSignal]:
        """
        生成成交量交易信号
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            成交量信号列表
        """
        signals = []
        current_time = datetime.now()
        
        # 基础指标
        metrics = self.calculate_basic_volume_metrics(data)
        volume_ratio = metrics['volume_ratio']
        price_change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
        
        # 1. 成交量激增信号
        if volume_ratio > 2.0:
            strength = min(volume_ratio / 5.0, 1.0)
            signals.append(VolumeSignal(
                timestamp=current_time,
                signal_type="VOLUME_SPIKE",
                strength=strength,
                description=f"成交量激增，当前成交量是平均的{volume_ratio:.1f}倍",
                volume_ratio=volume_ratio,
                price_change=price_change
            ))
        
        # 2. 突破信号
        if volume_ratio > 1.5 and abs(price_change) > 0.02:
            strength = min(volume_ratio * abs(price_change) * 10, 1.0)
            direction = "上涨" if price_change > 0 else "下跌"
            signals.append(VolumeSignal(
                timestamp=current_time,
                signal_type="BREAKOUT",
                strength=strength,
                description=f"成交量配合{direction}突破，成交量放大{volume_ratio:.1f}倍",
                volume_ratio=volume_ratio,
                price_change=price_change
            ))
        
        # 3. 吸筹信号
        if price_change < -0.01 and volume_ratio > 1.2:
            strength = min(volume_ratio * abs(price_change) * 15, 1.0)
            signals.append(VolumeSignal(
                timestamp=current_time,
                signal_type="ACCUMULATION",
                strength=strength,
                description="疑似机构吸筹，价格下跌但成交量放大",
                volume_ratio=volume_ratio,
                price_change=price_change
            ))
        
        # 4. 派发信号
        if price_change > 0.01 and volume_ratio > 1.2:
            strength = min(volume_ratio * price_change * 15, 1.0)
            signals.append(VolumeSignal(
                timestamp=current_time,
                signal_type="DISTRIBUTION",
                strength=strength,
                description="疑似机构派发，价格上涨但成交量放大",
                volume_ratio=volume_ratio,
                price_change=price_change
            ))
        
        # 5. 背离信号
        divergence = self.detect_volume_divergence(data)
        if divergence != DivergenceType.NONE:
            strength = 0.7
            description_map = {
                DivergenceType.BULLISH_DIVERGENCE: "看涨背离信号，价格创新低但成交量萎缩",
                DivergenceType.BEARISH_DIVERGENCE: "看跌背离信号，价格创新高但成交量萎缩",
                DivergenceType.HIDDEN_BULLISH: "隐藏看涨背离信号",
                DivergenceType.HIDDEN_BEARISH: "隐藏看跌背离信号"
            }
            signals.append(VolumeSignal(
                timestamp=current_time,
                signal_type="DIVERGENCE",
                strength=strength,
                description=description_map[divergence],
                volume_ratio=volume_ratio,
                price_change=price_change
            ))
        
        # 6. 异常信号
        anomaly = self.detect_volume_anomalies(data)
        if anomaly != VolumeAnomaly.NONE:
            strength = 0.8
            description_map = {
                VolumeAnomaly.UNUSUAL_HIGH: "异常高成交量，需要关注",
                VolumeAnomaly.UNUSUAL_LOW: "异常低成交量，市场关注度下降",
                VolumeAnomaly.VOLUME_SPIKE: "成交量急剧放大，可能有重大消息",
                VolumeAnomaly.VOLUME_DRY_UP: "成交量极度萎缩，可能面临变盘"
            }
            signals.append(VolumeSignal(
                timestamp=current_time,
                signal_type="ANOMALY",
                strength=strength,
                description=description_map[anomaly],
                volume_ratio=volume_ratio,
                price_change=price_change
            ))
        
        return signals
    
    def analyze_multi_timeframe(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, VolumeAnalysisResult]:
        """
        多时间周期成交量分析
        
        Args:
            data_dict: 不同时间周期的数据字典 {'1m': df_1m, '5m': df_5m, ...}
            
        Returns:
            多时间周期分析结果
        """
        results = {}
        
        for timeframe, data in data_dict.items():
            if len(data) < self.lookback_periods:
                continue
            
            # 执行完整分析
            analysis_result = self._perform_complete_analysis(data)
            results[timeframe] = analysis_result
        
        return results
    
    def _perform_complete_analysis(self, data: pd.DataFrame) -> VolumeAnalysisResult:
        """执行完整的成交量分析"""
        current_time = datetime.now()
        
        # 检测各种模式
        volume_pattern = self.detect_volume_patterns(data)
        anomaly = self.detect_volume_anomalies(data)
        divergence = self.detect_volume_divergence(data)
        
        # 计算各种指标
        basic_metrics = self.calculate_basic_volume_metrics(data)
        price_volume_metrics = self.analyze_price_volume_relationship(data)
        trend_metrics = self.analyze_volume_trends(data)
        institutional_metrics = self.analyze_institutional_flow(data)
        
        # 合并所有指标
        all_metrics = {**basic_metrics, **price_volume_metrics, **trend_metrics, **institutional_metrics}
        
        # 生成信号
        signals = self.generate_volume_signals(data)
        
        # 计算置信度
        confidence = self._calculate_confidence(all_metrics, signals)
        
        return VolumeAnalysisResult(
            timestamp=current_time,
            volume_pattern=volume_pattern,
            anomaly=anomaly,
            divergence=divergence,
            signals=signals,
            metrics=all_metrics,
            confidence=confidence
        )
    
    def _calculate_confidence(self, metrics: Dict[str, float], signals: List[VolumeSignal]) -> float:
        """计算分析置信度"""
        confidence_factors = []
        
        # 基于信号强度
        if signals:
            avg_signal_strength = np.mean([signal.strength for signal in signals])
            confidence_factors.append(avg_signal_strength)
        
        # 基于数据质量
        if 'volume_std' in metrics and metrics['volume_std'] > 0:
            volume_consistency = 1 - (metrics['volume_std'] / metrics['volume_mean'])
            confidence_factors.append(max(0, volume_consistency))
        
        # 基于指标一致性
        if 'price_volume_correlation' in metrics:
            correlation_strength = abs(metrics['price_volume_correlation'])
            confidence_factors.append(correlation_strength)
        
        # 计算最终置信度
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5  # 默认置信度
    
    def monitor_real_time_volume(self, data: pd.DataFrame, alert_callback=None):
        """
        实时成交量监控
        
        Args:
            data: 实时数据
            alert_callback: 预警回调函数
        """
        if alert_callback:
            self.alert_callbacks.append(alert_callback)
        
        # 执行分析
        result = self._perform_complete_analysis(data)
        
        # 检查是否需要预警
        for signal in result.signals:
            if signal.strength > 0.7:  # 高强度信号
                self._trigger_alert(signal, result)
        
        return result
    
    def _trigger_alert(self, signal: VolumeSignal, result: VolumeAnalysisResult):
        """触发预警"""
        alert_data = {
            'timestamp': signal.timestamp,
            'signal_type': signal.signal_type,
            'strength': signal.strength,
            'description': signal.description,
            'volume_pattern': result.volume_pattern.value,
            'anomaly': result.anomaly.value,
            'confidence': result.confidence
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"预警回调执行失败: {e}")
    
    def visualize_volume_analysis(self, data: pd.DataFrame, result: VolumeAnalysisResult, 
                                 save_path: str = None):
        """
        成交量分析可视化
        
        Args:
            data: OHLCV数据
            result: 分析结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle(f'成交量分析报告 - {result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16)
        
        # 1. 价格和成交量
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(data.index, data['close'], label='收盘价', color='blue')
        ax1_twin.bar(data.index, data['volume'], alpha=0.3, label='成交量', color='gray')
        
        ax1.set_title('价格与成交量')
        ax1.set_ylabel('价格', color='blue')
        ax1_twin.set_ylabel('成交量', color='gray')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. OBV指标
        ax2 = axes[0, 1]
        obv = self.calculate_obv(data)
        ax2.plot(obv.index, obv.values, label='OBV', color='purple')
        ax2.set_title('能量潮指标(OBV)')
        ax2.set_ylabel('OBV值')
        ax2.legend()
        
        # 3. VWAP指标
        ax3 = axes[1, 0]
        vwap = self.calculate_vwap(data)
        ax3.plot(data.index, data['close'], label='收盘价', color='blue')
        ax3.plot(data.index, vwap.values, label='VWAP', color='orange')
        ax3.set_title('成交量加权平均价格(VWAP)')
        ax3.set_ylabel('价格')
        ax3.legend()
        
        # 4. 成交量移动平均
        ax4 = axes[1, 1]
        volume_ma5 = data['volume'].rolling(5).mean()
        volume_ma20 = data['volume'].rolling(20).mean()
        
        ax4.plot(data.index, data['volume'], alpha=0.3, label='成交量', color='gray')
        ax4.plot(data.index, volume_ma5, label='MA5', color='green')
        ax4.plot(data.index, volume_ma20, label='MA20', color='red')
        ax4.set_title('成交量移动平均')
        ax4.set_ylabel('成交量')
        ax4.legend()
        
        # 5. 成交量分布
        ax5 = axes[2, 0]
        ax5.hist(data['volume'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(data['volume'].mean(), color='red', linestyle='--', label='平均值')
        ax5.axvline(data['volume'].median(), color='green', linestyle='--', label='中位数')
        ax5.set_title('成交量分布')
        ax5.set_xlabel('成交量')
        ax5.set_ylabel('频次')
        ax5.legend()
        
        # 6. 成交量异常检测
        ax6 = axes[2, 1]
        volume_mean = data['volume'].rolling(self.lookback_periods).mean()
        volume_std = data['volume'].rolling(self.lookback_periods).std()
        
        upper_bound = volume_mean + self.anomaly_threshold * volume_std
        lower_bound = volume_mean - self.anomaly_threshold * volume_std
        
        ax6.plot(data.index, data['volume'], label='成交量', color='blue')
        ax6.fill_between(data.index, upper_bound, lower_bound, alpha=0.2, color='yellow', label='正常范围')
        ax6.plot(data.index, upper_bound, 'r--', alpha=0.5, label='上界')
        ax6.plot(data.index, lower_bound, 'r--', alpha=0.5, label='下界')
        ax6.set_title('成交量异常检测')
        ax6.set_ylabel('成交量')
        ax6.legend()
        
        # 7. 信号强度
        ax7 = axes[3, 0]
        if result.signals:
            signal_types = [signal.signal_type for signal in result.signals]
            signal_strengths = [signal.strength for signal in result.signals]
            
            colors = ['red' if s > 0.7 else 'orange' if s > 0.5 else 'green' for s in signal_strengths]
            ax7.bar(signal_types, signal_strengths, color=colors)
            ax7.set_title('成交量信号强度')
            ax7.set_ylabel('信号强度')
            ax7.tick_params(axis='x', rotation=45)
        else:
            ax7.text(0.5, 0.5, '无信号', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('成交量信号强度')
        
        # 8. 分析结果摘要
        ax8 = axes[3, 1]
        ax8.axis('off')
        
        summary_text = f"""
        分析摘要
        
        成交量模式: {result.volume_pattern.value}
        异常检测: {result.anomaly.value}
        背离分析: {result.divergence.value}
        置信度: {result.confidence:.2f}
        
        当前成交量: {result.metrics.get('volume_current', 0):.0f}
        成交量比值: {result.metrics.get('volume_ratio', 0):.2f}
        价格成交量相关性: {result.metrics.get('price_volume_correlation', 0):.3f}
        
        信号数量: {len(result.signals)}
        """
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"成交量分析图表已保存到: {save_path}")
        
        return fig
    
    def generate_report(self, data: pd.DataFrame, result: VolumeAnalysisResult) -> str:
        """
        生成成交量分析报告
        
        Args:
            data: OHLCV数据
            result: 分析结果
            
        Returns:
            分析报告文本
        """
        report = f"""
        # B8成交量分析报告
        
        ## 分析时间
        {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        
        ## 成交量模式分析
        - 当前模式: {result.volume_pattern.value}
        - 异常检测: {result.anomaly.value}
        - 背离分析: {result.divergence.value}
        - 分析置信度: {result.confidence:.2%}
        
        ## 关键指标
        - 当前成交量: {result.metrics.get('volume_current', 0):,.0f}
        - 成交量均值: {result.metrics.get('volume_mean', 0):,.0f}
        - 成交量比值: {result.metrics.get('volume_ratio', 0):.2f}
        - 价格成交量相关性: {result.metrics.get('price_volume_correlation', 0):.3f}
        
        ## OBV分析
        - OBV趋势匹配度: {result.metrics.get('obv_price_trend_match', 0)}
        
        ## 成交量趋势
        - 成交量动量: {result.metrics.get('volume_momentum', 0):.2f}%
        - 成交量波动率: {result.metrics.get('volume_volatility', 0):,.0f}
        
        ## 机构资金流向
        - 资金流向方向: {'净流入' if result.metrics.get('flow_direction', 0) > 0 else '净流出'}
        - 资金流向强度: {result.metrics.get('flow_strength', 0):.2f}
        - 大单占比: {result.metrics.get('large_trade_ratio', 0):.2%}
        
        ## 交易信号
        """
        
        if result.signals:
            for i, signal in enumerate(result.signals, 1):
                report += f"""
        ### 信号 {i}
        - 类型: {signal.signal_type}
        - 强度: {signal.strength:.2f}
        - 描述: {signal.description}
        - 成交量比值: {signal.volume_ratio:.2f}
        - 价格变化: {signal.price_change:.2%}
        """
        else:
            report += "\n暂无交易信号"
        
        report += f"""
        
        ## 投资建议
        """
        
        # 基于分析结果生成投资建议
        if result.volume_pattern == VolumePattern.ACCUMULATION:
            report += "- 检测到吸筹模式，可关注后续上涨机会\n"
        elif result.volume_pattern == VolumePattern.DISTRIBUTION:
            report += "- 检测到派发模式，注意风险控制\n"
        elif result.volume_pattern == VolumePattern.BREAKOUT:
            report += "- 检测到突破信号，可考虑顺势操作\n"
        
        if result.divergence == DivergenceType.BULLISH_DIVERGENCE:
            report += "- 出现看涨背离，可能存在反弹机会\n"
        elif result.divergence == DivergenceType.BEARISH_DIVERGENCE:
            report += "- 出现看跌背离，需要谨慎操作\n"
        
        if result.anomaly == VolumeAnomaly.VOLUME_SPIKE:
            report += "- 成交量异常放大，建议关注相关消息面\n"
        
        report += f"""
        ## 风险提示
        - 本分析仅供参考，不构成投资建议
        - 成交量分析需要结合其他技术指标综合判断
        - 建议设置止损，控制投资风险
        
        ---
        报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report
    
    def backtest_volume_signals(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, float]:
        """
        成交量信号回测
        
        Args:
            data: 历史OHLCV数据
            initial_capital: 初始资金
            
        Returns:
            回测结果
        """
        if len(data) < self.lookback_periods:
            return {}
        
        capital = initial_capital
        position = 0
        trades = []
        portfolio_value = []
        
        for i in range(self.lookback_periods, len(data)):
            current_data = data.iloc[:i+1]
            signals = self.generate_volume_signals(current_data)
            
            current_price = data['close'].iloc[i]
            
            # 执行交易逻辑
            for signal in signals:
                if signal.signal_type == "BREAKOUT" and signal.strength > 0.6:
                    if position == 0 and signal.price_change > 0:  # 买入信号
                        position = capital / current_price
                        capital = 0
                        trades.append({
                            'type': 'BUY',
                            'price': current_price,
                            'timestamp': data.index[i],
                            'signal_strength': signal.strength
                        })
                    elif position > 0 and signal.price_change < 0:  # 卖出信号
                        capital = position * current_price
                        position = 0
                        trades.append({
                            'type': 'SELL',
                            'price': current_price,
                            'timestamp': data.index[i],
                            'signal_strength': signal.strength
                        })
            
            # 计算当前组合价值
            current_value = capital + position * current_price
            portfolio_value.append(current_value)
        
        # 计算回测指标
        if len(portfolio_value) > 1:
            total_return = (portfolio_value[-1] - initial_capital) / initial_capital
            
            # 计算最大回撤
            peak = np.maximum.accumulate(portfolio_value)
            drawdown = (peak - portfolio_value) / peak
            max_drawdown = np.max(drawdown)
            
            # 计算夏普比率
            returns = np.diff(portfolio_value) / portfolio_value[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # 胜率
            winning_trades = [t for t in trades if t['type'] == 'SELL']
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            return {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'final_value': portfolio_value[-1] if portfolio_value else initial_capital
            }
        
        return {}


def create_sample_data(days: int = 100) -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # 生成价格数据
    price_base = 100
    returns = np.random.normal(0.001, 0.02, days)
    prices = price_base * np.exp(np.cumsum(returns))
    
    # 生成成交量数据（与价格有一定相关性）
    base_volume = 1000000
    volume_volatility = 0.3
    volume_trend = np.random.normal(0, 0.001, days)
    volume_noise = np.random.normal(0, volume_volatility, days)
    
    # 价格上涨时成交量增加
    price_volume_corr = 0.3
    volume = base_volume * (1 + price_volume_corr * returns + volume_trend + volume_noise)
    volume = np.maximum(volume, base_volume * 0.1)  # 确保成交量不为负
    
    # 生成OHLC数据
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume.astype(int)
    }, index=dates)
    
    return data


def example_usage():
    """使用示例"""
    # 创建分析器
    analyzer = VolumeAnalyzer(lookback_periods=20, anomaly_threshold=2.0)
    
    # 创建示例数据
    data = create_sample_data(100)
    
    print("=== B8成交量分析器示例 ===\n")
    
    # 基础分析
    print("1. 基础成交量指标:")
    basic_metrics = analyzer.calculate_basic_volume_metrics(data)
    for key, value in basic_metrics.items():
        print(f"   {key}: {value:.2f}")
    
    print("\n2. 成交量模式检测:")
    pattern = analyzer.detect_volume_patterns(data)
    print(f"   当前模式: {pattern.value}")
    
    print("\n3. 成交量异常检测:")
    anomaly = analyzer.detect_volume_anomalies(data)
    print(f"   异常类型: {anomaly.value}")
    
    print("\n4. 成交量背离分析:")
    divergence = analyzer.detect_volume_divergence(data)
    print(f"   背离类型: {divergence.value}")
    
    print("\n5. 价格成交量关系:")
    relationship = analyzer.analyze_price_volume_relationship(data)
    for key, value in relationship.items():
        print(f"   {key}: {value:.3f}")
    
    print("\n6. 成交量趋势分析:")
    trends = analyzer.analyze_volume_trends(data)
    for key, value in trends.items():
        print(f"   {key}: {value:.3f}")
    
    print("\n7. 机构资金流向:")
    institutional = analyzer.analyze_institutional_flow(data)
    for key, value in institutional.items():
        print(f"   {key}: {value:.3f}")
    
    print("\n8. 交易信号:")
    signals = analyzer.generate_volume_signals(data)
    for signal in signals:
        print(f"   - {signal.signal_type}: {signal.description} (强度: {signal.strength:.2f})")
    
    print("\n9. 完整分析:")
    result = analyzer._perform_complete_analysis(data)
    print(f"   置信度: {result.confidence:.2f}")
    print(f"   信号数量: {len(result.signals)}")
    
    print("\n10. 生成报告:")
    report = analyzer.generate_report(data, result)
    print(report[:500] + "..." if len(report) > 500 else report)
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    example_usage()