#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A7技术指标计算器
================

高级技术指标计算引擎，支持基础指标、高级指标、自定义指标开发、
指标组合优化、实时计算和有效性验证。

Author: A7 Technical Indicator Calculator
Date: 2025-11-05
Version: 1.0.0
"""

import numpy as np
import pandas as pd
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
import threading
import time
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class IndicatorSignal:
    """技术指标信号数据类"""
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float   # 信号强度 0-1
    timestamp: datetime
    price: float
    indicator_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorConfig:
    """指标配置数据类"""
    name: str
    parameters: Dict[str, Any]
    timeframes: List[str]
    enabled: bool = True
    priority: int = 1
    cache_ttl: int = 300  # 缓存时间（秒）


class IndicatorCache:
    """指标计算结果缓存管理器"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.RLock()
    
    def _generate_key(self, indicator_name: str, data_hash: str, params: Dict) -> str:
        """生成缓存键"""
        param_str = str(sorted(params.items()))
        return f"{indicator_name}:{data_hash}:{hashlib.md5(param_str.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < 300:  # 5分钟TTL
                    return value
                else:
                    del self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # 删除最旧的条目
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()


class BaseIndicator(ABC):
    """技术指标基类"""
    
    def __init__(self, name: str, config: IndicatorConfig):
        self.name = name
        self.config = config
        self.cache = IndicatorCache()
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算指标值"""
        pass
    
    @abstractmethod
    def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
        """生成交易信号"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)
    
    def optimize_parameters(self, data: pd.DataFrame, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """参数优化（基类方法，子类可重写）"""
        logger.info(f"优化指标 {self.name} 的参数...")
        # 默认返回当前参数
        return self.config.parameters


class BasicIndicators:
    """基础技术指标集合"""
    
    @staticmethod
    def moving_average(data: pd.Series, period: int = 20, ma_type: str = 'SMA') -> pd.Series:
        """移动平均线"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单移动平均
            if ma_type.upper() == 'SMA':
                return data.rolling(window=period).mean()
            elif ma_type.upper() == 'EMA':
                return data.ewm(span=period).mean()
            elif ma_type.upper() == 'WMA':
                # 简单线性加权移动平均
                weights = list(range(1, period + 1))
                return data.rolling(period).apply(lambda x: sum(w * v for w, v in zip(weights, x)) / sum(weights))
            else:
                raise ValueError(f"不支持的移动平均类型: {ma_type}")
        
        if ma_type.upper() == 'SMA':
            return talib.SMA(data, timeperiod=period)
        elif ma_type.upper() == 'EMA':
            return talib.EMA(data, timeperiod=period)
        elif ma_type.upper() == 'WMA':
            return talib.WMA(data, timeperiod=period)
        else:
            raise ValueError(f"不支持的移动平均类型: {ma_type}")
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指数"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单RSI
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        return talib.RSI(data, timeperiod=period)
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD指标"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单MACD
            ema_fast = data.ewm(span=fast).mean()
            ema_slow = data.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        
        macd_line, signal_line, histogram = talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return macd_line, signal_line, histogram
    
    @staticmethod
    def kdj(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 9, d_period: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """KDJ指标"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单KDJ
            lowest_low = lowest(low, k_period)
            highest_high = highest(high, k_period)
            
            rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
            rsv = rsv.fillna(0)
            
            k = rsv.ewm(com=d_period-1).mean()
            d = k.ewm(com=d_period-1).mean()
            j = 3 * k - 2 * d
            
            return k, d, j
        
        lowest_low = lowest(low, k_period)
        highest_high = highest(high, k_period)
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(0)
        
        k = rsv.ewm(com=d_period-1).mean()
        d = k.ewm(com=d_period-1).mean()
        j = 3 * k - 2 * d
        
        return k, d, j
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """布林带"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单布林带
            middle = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper, middle, lower
        
        upper, middle, lower = talib.BBANDS(data, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        return upper, middle, lower
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """随机指标"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单随机指标
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return k_percent, d_percent
        
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
        return slowk, slowd


def lowest(data: pd.Series, period: int) -> pd.Series:
    """计算最低值"""
    return data.rolling(window=period).min()


def highest(data: pd.Series, period: int) -> pd.Series:
    """计算最高值"""
    return data.rolling(window=period).max()


class AdvancedIndicators:
    """高级技术指标集合"""
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """斐波那契回撤位"""
        diff = high - low
        levels = {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low
        }
        return levels
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """一目均衡表"""
        tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """威廉指标"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单威廉指标
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            wr = -100 * (highest_high - close) / (highest_high - lowest_low)
            return wr
        
        return talib.WILLR(high, low, close, timeperiod=period)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """商品通道指数"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单CCI
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            return cci
        
        return talib.CCI(high, low, close, timeperiod=period)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均真实范围"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单ATR
            tr1 = high - low
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        
        return talib.ATR(high, low, close, timeperiod=period)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均方向指数"""
        if not TALIB_AVAILABLE:
            # 使用pandas实现简单ADX（简化版本）
            diff = close.diff()
            up = diff.where(diff > 0, 0).rolling(window=period).mean()
            down = (-diff.where(diff < 0, 0)).rolling(window=period).mean()
            dx = 100 * np.abs(up - down) / (up + down)
            adx = dx.rolling(window=period).mean()
            return adx
        
        return talib.ADX(high, low, close, timeperiod=period)


class CustomIndicator(BaseIndicator):
    """自定义指标基类"""
    
    def __init__(self, name: str, config: IndicatorConfig, calculation_func: Callable, signal_func: Optional[Callable] = None):
        super().__init__(name, config)
        self.calculation_func = calculation_func
        self.signal_func = signal_func
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """使用自定义函数计算指标"""
        if not self.validate_data(data):
            raise ValueError("输入数据格式不正确")
        
        return self.calculation_func(data, **kwargs)
    
    def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
        """生成交易信号"""
        if self.signal_func:
            return self.signal_func(indicator_values, price_data)
        
        # 默认信号生成逻辑
        signals = []
        for i in range(len(indicator_values)):
            if pd.isna(indicator_values.iloc[i]):
                continue
            
            value = indicator_values.iloc[i]
            # 基于数值变化生成简单信号
            if i > 0:
                prev_value = indicator_values.iloc[i-1]
                if not pd.isna(prev_value):
                    change = (value - prev_value) / prev_value
                    if change > 0.02:  # 2%以上上涨
                        signals.append(IndicatorSignal('BUY', min(0.8, abs(change) * 10), datetime.now(), price_data.iloc[i], value))
                    elif change < -0.02:  # 2%以上下跌
                        signals.append(IndicatorSignal('SELL', min(0.8, abs(change) * 10), datetime.now(), price_data.iloc[i], value))
        
        return signals


class IndicatorCombination:
    """指标组合器"""
    
    def __init__(self, indicators: List[BaseIndicator], combination_method: str = 'weighted'):
        self.indicators = indicators
        self.combination_method = combination_method
        self.weights = {indicator.name: 1.0 for indicator in indicators}
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """设置指标权重"""
        self.weights.update(weights)
    
    def combine_signals(self, signals: List[IndicatorSignal]) -> IndicatorSignal:
        """组合多个指标信号"""
        if not signals:
            return IndicatorSignal('HOLD', 0.0, datetime.now(), 0.0, 0.0)
        
        # 计算加权平均信号强度
        total_weight = sum(self.weights.get(signal.indicator_value, 1.0) for signal in signals)
        if total_weight == 0:
            return IndicatorSignal('HOLD', 0.0, datetime.now(), 0.0, 0.0)
        
        weighted_strength = sum(
            signal.strength * self.weights.get(signal.indicator_value, 1.0) 
            for signal in signals
        ) / total_weight
        
        # 决定最终信号类型
        if weighted_strength > 0.6:
            signal_type = 'BUY'
        elif weighted_strength < 0.4:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        return IndicatorSignal(
            signal_type=signal_type,
            strength=weighted_strength,
            timestamp=signals[0].timestamp,
            price=signals[0].price,
            indicator_value=weighted_strength
        )


class TechnicalIndicatorCalculator:
    """技术指标计算器主类"""
    
    def __init__(self, cache_size: int = 1000, max_workers: int = 4):
        self.cache = IndicatorCache(cache_size)
        self.indicators: Dict[str, BaseIndicator] = {}
        self.combinations: Dict[str, IndicatorCombination] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.data_cache = {}
        
        # 注册内置指标
        self._register_builtin_indicators()
    
    def _register_builtin_indicators(self):
        """注册内置技术指标"""
        # 基础指标
        self.register_indicator('SMA', self._create_sma_indicator())
        self.register_indicator('EMA', self._create_ema_indicator())
        self.register_indicator('RSI', self._create_rsi_indicator())
        self.register_indicator('MACD', self._create_macd_indicator())
        self.register_indicator('KDJ', self._create_kdj_indicator())
        self.register_indicator('BOLL', self._create_bollinger_indicator())
        
        # 高级指标
        self.register_indicator('WR', self._create_williams_r_indicator())
        self.register_indicator('CCI', self._create_cci_indicator())
        self.register_indicator('ATR', self._create_atr_indicator())
        self.register_indicator('ADX', self._create_adx_indicator())
    
    def _create_sma_indicator(self) -> BaseIndicator:
        """创建SMA指标"""
        config = IndicatorConfig(
            name='SMA',
            parameters={'period': 20},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class SMAIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                period = kwargs.get('period', self.config.parameters['period'])
                return BasicIndicators.moving_average(data['close'], period, 'SMA')
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                for i in range(1, len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]) or pd.isna(indicator_values.iloc[i-1]):
                        continue
                    
                    # 金叉死叉信号
                    if indicator_values.iloc[i-1] < price_data.iloc[i-1] and indicator_values.iloc[i] > price_data.iloc[i]:
                        signals.append(IndicatorSignal('BUY', 0.7, datetime.now(), price_data.iloc[i], indicator_values.iloc[i]))
                    elif indicator_values.iloc[i-1] > price_data.iloc[i-1] and indicator_values.iloc[i] < price_data.iloc[i]:
                        signals.append(IndicatorSignal('SELL', 0.7, datetime.now(), price_data.iloc[i], indicator_values.iloc[i]))
                
                return signals
        
        return SMAIndicator('SMA', config)
    
    def _create_ema_indicator(self) -> BaseIndicator:
        """创建EMA指标"""
        config = IndicatorConfig(
            name='EMA',
            parameters={'period': 12},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class EMAIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                period = kwargs.get('period', self.config.parameters['period'])
                return BasicIndicators.moving_average(data['close'], period, 'EMA')
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                for i in range(1, len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]) or pd.isna(indicator_values.iloc[i-1]):
                        continue
                    
                    # EMA趋势信号
                    if indicator_values.iloc[i] > indicator_values.iloc[i-1]:
                        strength = min(0.8, (indicator_values.iloc[i] - indicator_values.iloc[i-1]) / indicator_values.iloc[i-1] * 10)
                        signals.append(IndicatorSignal('BUY', strength, datetime.now(), price_data.iloc[i], indicator_values.iloc[i]))
                    else:
                        strength = min(0.8, (indicator_values.iloc[i-1] - indicator_values.iloc[i]) / indicator_values.iloc[i-1] * 10)
                        signals.append(IndicatorSignal('SELL', strength, datetime.now(), price_data.iloc[i], indicator_values.iloc[i]))
                
                return signals
        
        return EMAIndicator('EMA', config)
    
    def _create_rsi_indicator(self) -> BaseIndicator:
        """创建RSI指标"""
        config = IndicatorConfig(
            name='RSI',
            parameters={'period': 14},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class RSIIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                period = kwargs.get('period', self.config.parameters['period'])
                return BasicIndicators.rsi(data['close'], period)
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                for i in range(len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]):
                        continue
                    
                    rsi_value = indicator_values.iloc[i]
                    
                    # 超买超卖信号
                    if rsi_value > 70:
                        signals.append(IndicatorSignal('SELL', min(0.9, (rsi_value - 70) / 30), datetime.now(), price_data.iloc[i], rsi_value))
                    elif rsi_value < 30:
                        signals.append(IndicatorSignal('BUY', min(0.9, (30 - rsi_value) / 30), datetime.now(), price_data.iloc[i], rsi_value))
                
                return signals
        
        return RSIIndicator('RSI', config)
    
    def _create_macd_indicator(self) -> BaseIndicator:
        """创建MACD指标"""
        config = IndicatorConfig(
            name='MACD',
            parameters={'fast': 12, 'slow': 26, 'signal': 9},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class MACDIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                fast = kwargs.get('fast', self.config.parameters['fast'])
                slow = kwargs.get('slow', self.config.parameters['slow'])
                signal = kwargs.get('signal', self.config.parameters['signal'])
                
                macd_line, signal_line, histogram = BasicIndicators.macd(data['close'], fast, slow, signal)
                return histogram  # 使用MACD柱状图作为主要信号
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                for i in range(1, len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]) or pd.isna(indicator_values.iloc[i-1]):
                        continue
                    
                    # MACD金叉死叉
                    if indicator_values.iloc[i-1] < 0 and indicator_values.iloc[i] > 0:
                        signals.append(IndicatorSignal('BUY', 0.8, datetime.now(), price_data.iloc[i], indicator_values.iloc[i]))
                    elif indicator_values.iloc[i-1] > 0 and indicator_values.iloc[i] < 0:
                        signals.append(IndicatorSignal('SELL', 0.8, datetime.now(), price_data.iloc[i], indicator_values.iloc[i]))
                
                return signals
        
        return MACDIndicator('MACD', config)
    
    def _create_kdj_indicator(self) -> BaseIndicator:
        """创建KDJ指标"""
        config = IndicatorConfig(
            name='KDJ',
            parameters={'k_period': 9, 'd_period': 3},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class KDJIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                k_period = kwargs.get('k_period', self.config.parameters['k_period'])
                d_period = kwargs.get('d_period', self.config.parameters['d_period'])
                
                k, d, j = BasicIndicators.kdj(data['high'], data['low'], data['close'], k_period, d_period)
                return j  # 使用J线作为主要信号
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                for i in range(len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]):
                        continue
                    
                    j_value = indicator_values.iloc[i]
                    
                    # KDJ超买超卖信号
                    if j_value > 100:
                        signals.append(IndicatorSignal('SELL', min(0.9, (j_value - 100) / 100), datetime.now(), price_data.iloc[i], j_value))
                    elif j_value < 0:
                        signals.append(IndicatorSignal('BUY', min(0.9, abs(j_value) / 100), datetime.now(), price_data.iloc[i], j_value))
                
                return signals
        
        return KDJIndicator('KDJ', config)
    
    def _create_bollinger_indicator(self) -> BaseIndicator:
        """创建布林带指标"""
        config = IndicatorConfig(
            name='BOLL',
            parameters={'period': 20, 'std_dev': 2.0},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class BollingerIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                period = kwargs.get('period', self.config.parameters['period'])
                std_dev = kwargs.get('std_dev', self.config.parameters['std_dev'])
                
                upper, middle, lower = BasicIndicators.bollinger_bands(data['close'], period, std_dev)
                # 返回价格相对于布林带的位置
                position = (data['close'] - lower) / (upper - lower)
                return position
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                for i in range(len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]):
                        continue
                    
                    position = indicator_values.iloc[i]
                    
                    # 布林带位置信号
                    if position > 0.8:  # 接近上轨
                        signals.append(IndicatorSignal('SELL', position - 0.8, datetime.now(), price_data.iloc[i], position))
                    elif position < 0.2:  # 接近下轨
                        signals.append(IndicatorSignal('BUY', 0.2 - position, datetime.now(), price_data.iloc[i], position))
                
                return signals
        
        return BollingerIndicator('BOLL', config)
    
    def _create_williams_r_indicator(self) -> BaseIndicator:
        """创建威廉指标"""
        config = IndicatorConfig(
            name='WR',
            parameters={'period': 14},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class WilliamsRIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                period = kwargs.get('period', self.config.parameters['period'])
                return AdvancedIndicators.williams_r(data['high'], data['low'], data['close'], period)
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                for i in range(len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]):
                        continue
                    
                    wr_value = indicator_values.iloc[i]
                    
                    # 威廉指标超买超卖
                    if wr_value < -80:
                        signals.append(IndicatorSignal('BUY', abs(wr_value + 80) / 20, datetime.now(), price_data.iloc[i], wr_value))
                    elif wr_value > -20:
                        signals.append(IndicatorSignal('SELL', (wr_value + 20) / 20, datetime.now(), price_data.iloc[i], wr_value))
                
                return signals
        
        return WilliamsRIndicator('WR', config)
    
    def _create_cci_indicator(self) -> BaseIndicator:
        """创建CCI指标"""
        config = IndicatorConfig(
            name='CCI',
            parameters={'period': 20},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class CCIIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                period = kwargs.get('period', self.config.parameters['period'])
                return AdvancedIndicators.cci(data['high'], data['low'], data['close'], period)
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                for i in range(len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]):
                        continue
                    
                    cci_value = indicator_values.iloc[i]
                    
                    # CCI超买超卖
                    if cci_value > 100:
                        signals.append(IndicatorSignal('SELL', min(0.9, (cci_value - 100) / 200), datetime.now(), price_data.iloc[i], cci_value))
                    elif cci_value < -100:
                        signals.append(IndicatorSignal('BUY', min(0.9, abs(cci_value + 100) / 200), datetime.now(), price_data.iloc[i], cci_value))
                
                return signals
        
        return CCIIndicator('CCI', config)
    
    def _create_atr_indicator(self) -> BaseIndicator:
        """创建ATR指标"""
        config = IndicatorConfig(
            name='ATR',
            parameters={'period': 14},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class ATRIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                period = kwargs.get('period', self.config.parameters['period'])
                return AdvancedIndicators.atr(data['high'], data['low'], data['close'], period)
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                # ATR主要用于衡量波动性，通常不直接产生买卖信号
                # 这里可以基于ATR变化率产生信号
                for i in range(1, len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]) or pd.isna(indicator_values.iloc[i-1]):
                        continue
                    
                    atr_change = (indicator_values.iloc[i] - indicator_values.iloc[i-1]) / indicator_values.iloc[i-1]
                    
                    if atr_change > 0.1:  # ATR快速上升，可能有重大波动
                        signals.append(IndicatorSignal('HOLD', min(0.5, atr_change), datetime.now(), price_data.iloc[i], indicator_values.iloc[i]))
                
                return signals
        
        return ATRIndicator('ATR', config)
    
    def _create_adx_indicator(self) -> BaseIndicator:
        """创建ADX指标"""
        config = IndicatorConfig(
            name='ADX',
            parameters={'period': 14},
            timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
        )
        
        class ADXIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                period = kwargs.get('period', self.config.parameters['period'])
                return AdvancedIndicators.adx(data['high'], data['low'], data['close'], period)
            
            def generate_signal(self, indicator_values: pd.Series, price_data: pd.Series) -> List[IndicatorSignal]:
                signals = []
                for i in range(len(indicator_values)):
                    if pd.isna(indicator_values.iloc[i]):
                        continue
                    
                    adx_value = indicator_values.iloc[i]
                    
                    # ADX趋势强度信号
                    if adx_value > 25:  # 强趋势
                        signals.append(IndicatorSignal('HOLD', min(0.7, (adx_value - 25) / 25), datetime.now(), price_data.iloc[i], adx_value))
                
                return signals
        
        return ADXIndicator('ADX', config)
    
    def register_indicator(self, name: str, indicator: BaseIndicator) -> None:
        """注册技术指标"""
        self.indicators[name] = indicator
        logger.info(f"注册技术指标: {name}")
    
    def register_custom_indicator(self, name: str, calculation_func: Callable, config: IndicatorConfig, signal_func: Optional[Callable] = None) -> None:
        """注册自定义指标"""
        indicator = CustomIndicator(name, config, calculation_func, signal_func)
        self.register_indicator(name, indicator)
    
    def calculate_indicator(self, indicator_name: str, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算单个技术指标"""
        if indicator_name not in self.indicators:
            raise ValueError(f"未注册的技术指标: {indicator_name}")
        
        indicator = self.indicators[indicator_name]
        
        # 生成缓存键
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values.tobytes()).hexdigest()
        cache_key = indicator.cache._generate_key(indicator_name, data_hash, kwargs)
        
        # 检查缓存
        cached_result = indicator.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"从缓存获取指标 {indicator_name}")
            return cached_result
        
        # 计算指标
        logger.debug(f"计算指标 {indicator_name}")
        result = indicator.calculate(data, **kwargs)
        
        # 缓存结果
        indicator.cache.set(cache_key, result)
        
        return result
    
    def calculate_multiple_indicators(self, indicator_names: List[str], data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """并行计算多个技术指标"""
        futures = {}
        results = {}
        
        # 提交计算任务
        for indicator_name in indicator_names:
            future = self.executor.submit(self.calculate_indicator, indicator_name, data, **kwargs)
            futures[indicator_name] = future
        
        # 收集结果
        for indicator_name, future in futures.items():
            try:
                results[indicator_name] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"计算指标 {indicator_name} 失败: {e}")
                results[indicator_name] = pd.Series(dtype=float)
        
        return results
    
    def generate_signals(self, indicator_name: str, data: pd.DataFrame, **kwargs) -> List[IndicatorSignal]:
        """生成交易信号"""
        if indicator_name not in self.indicators:
            raise ValueError(f"未注册的技术指标: {indicator_name}")
        
        indicator = self.indicators[indicator_name]
        indicator_values = self.calculate_indicator(indicator_name, data, **kwargs)
        
        return indicator.generate_signal(indicator_values, data['close'])
    
    def combine_indicators(self, combination_name: str, indicator_names: List[str], 
                          weights: Optional[Dict[str, float]] = None) -> None:
        """组合多个指标"""
        indicators = [self.indicators[name] for name in indicator_names if name in self.indicators]
        
        if not indicators:
            raise ValueError("没有有效的指标用于组合")
        
        combination = IndicatorCombination(indicators)
        if weights:
            combination.set_weights(weights)
        
        self.combinations[combination_name] = combination
        logger.info(f"创建指标组合: {combination_name}")
    
    def optimize_parameters(self, indicator_name: str, data: pd.DataFrame, 
                           param_ranges: Dict[str, List], objective: str = 'sharpe_ratio') -> Dict[str, Any]:
        """优化指标参数"""
        if indicator_name not in self.indicators:
            raise ValueError(f"未注册的技术指标: {indicator_name}")
        
        indicator = self.indicators[indicator_name]
        return indicator.optimize_parameters(data, param_ranges)
    
    def validate_indicator_effectiveness(self, indicator_name: str, data: pd.DataFrame, 
                                       backtest_period: int = 252) -> Dict[str, float]:
        """验证指标有效性"""
        if indicator_name not in self.indicators:
            raise ValueError(f"未注册的技术指标: {indicator_name}")
        
        # 计算指标值
        indicator_values = self.calculate_indicator(indicator_name, data)
        
        # 生成信号
        signals = self.generate_signals(indicator_name, data)
        
        # 计算有效性指标
        if len(signals) < 2:
            return {'win_rate': 0.0, 'avg_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0}
        
        # 简化的有效性计算
        buy_signals = [s for s in signals if s.signal_type == 'BUY']
        sell_signals = [s for s in signals if s.signal_type == 'SELL']
        
        win_rate = len(buy_signals) / len(signals) if signals else 0.0
        avg_strength = np.mean([s.strength for s in signals])
        
        return {
            'win_rate': win_rate,
            'avg_strength': avg_strength,
            'signal_count': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals)
        }
    
    def get_realtime_indicators(self, data: pd.DataFrame, timeframe: str = '1m') -> Dict[str, Any]:
        """获取实时技术指标"""
        # 选择适合当前时间周期的指标
        available_indicators = []
        for name, indicator in self.indicators.items():
            if timeframe in indicator.config.timeframes:
                available_indicators.append(name)
        
        # 计算所有可用指标
        results = self.calculate_multiple_indicators(available_indicators, data)
        
        # 生成所有信号
        all_signals = {}
        for indicator_name in available_indicators:
            try:
                signals = self.generate_signals(indicator_name, data)
                all_signals[indicator_name] = signals[-1] if signals else None
            except Exception as e:
                logger.error(f"生成信号失败 {indicator_name}: {e}")
                all_signals[indicator_name] = None
        
        return {
            'indicators': results,
            'signals': all_signals,
            'timestamp': datetime.now(),
            'timeframe': timeframe
        }
    
    def create_custom_strategy(self, name: str, indicator_combination: List[str], 
                              signal_rules: Callable[[Dict], str]) -> None:
        """创建自定义交易策略"""
        def custom_strategy(data: pd.DataFrame) -> Dict[str, Any]:
            # 计算所需指标
            indicators = self.calculate_multiple_indicators(indicator_combination, data)
            
            # 应用自定义信号规则
            signal = signal_rules(indicators)
            
            return {
                'signal': signal,
                'indicators': indicators,
                'timestamp': datetime.now()
            }
        
        self.strategies[name] = custom_strategy
        logger.info(f"创建自定义策略: {name}")
    
    def backtest_strategy(self, strategy_name: str, data: pd.DataFrame, 
                         initial_capital: float = 10000) -> Dict[str, Any]:
        """回测策略"""
        if strategy_name not in getattr(self, 'strategies', {}):
            raise ValueError(f"未找到策略: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        
        portfolio_value = initial_capital
        position = 0
        trades = []
        
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            if len(current_data) < 50:  # 需要足够的历史数据
                continue
            
            try:
                result = strategy(current_data)
                signal = result.get('signal', 'HOLD')
                price = current_data['close'].iloc[-1]
                
                if signal == 'BUY' and position == 0:
                    # 开多头仓位
                    position = portfolio_value / price
                    portfolio_value = 0
                    trades.append({'type': 'BUY', 'price': price, 'timestamp': current_data.index[-1]})
                
                elif signal == 'SELL' and position > 0:
                    # 平多头仓位
                    portfolio_value = position * price
                    position = 0
                    trades.append({'type': 'SELL', 'price': price, 'timestamp': current_data.index[-1]})
            
            except Exception as e:
                logger.error(f"策略执行错误: {e}")
                continue
        
        # 计算最终收益
        final_value = portfolio_value + position * data['close'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'trades': trades
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("清空指标计算缓存")
    
    def get_indicator_info(self) -> Dict[str, Dict]:
        """获取所有指标信息"""
        info = {}
        for name, indicator in self.indicators.items():
            info[name] = {
                'name': indicator.name,
                'parameters': indicator.config.parameters,
                'timeframes': indicator.config.timeframes,
                'enabled': indicator.config.enabled
            }
        return info


# 使用示例和测试函数
def create_sample_data() -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    
    # 生成模拟价格数据
    price = 100
    prices = [price]
    for i in range(999):
        change = np.random.normal(0, 0.02)
        price *= (1 + change)
        prices.append(price)
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'timestamp': dates,
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    data.set_index('timestamp', inplace=True)
    return data


def demo_technical_indicators():
    """演示技术指标计算器功能"""
    print("=== A7技术指标计算器演示 ===\n")
    
    # 创建计算器实例
    calculator = TechnicalIndicatorCalculator()
    
    # 创建示例数据
    data = create_sample_data()
    print(f"创建示例数据: {len(data)} 条记录")
    print(f"数据时间范围: {data.index[0]} 到 {data.index[-1]}\n")
    
    # 显示可用指标
    print("可用技术指标:")
    indicator_info = calculator.get_indicator_info()
    for name, info in indicator_info.items():
        print(f"  - {name}: 参数 {info['parameters']}, 时间周期 {info['timeframes']}")
    print()
    
    # 计算单个指标
    print("计算RSI指标:")
    rsi_values = calculator.calculate_indicator('RSI', data, period=14)
    print(f"  最新RSI值: {rsi_values.iloc[-1]:.2f}")
    
    # 生成信号
    signals = calculator.generate_signals('RSI', data, period=14)
    print(f"  生成信号数量: {len(signals)}")
    if signals:
        latest_signal = signals[-1]
        print(f"  最新信号: {latest_signal.signal_type}, 强度: {latest_signal.strength:.2f}")
    print()
    
    # 计算多个指标
    print("并行计算多个指标:")
    indicator_names = ['SMA', 'EMA', 'MACD', 'KDJ', 'BOLL']
    results = calculator.calculate_multiple_indicators(indicator_names, data)
    
    for name, values in results.items():
        if not values.empty:
            print(f"  {name}: 最新值 {values.iloc[-1]:.2f}")
    print()
    
    # 指标组合
    print("创建指标组合:")
    calculator.combine_indicators('combo1', ['RSI', 'MACD', 'KDJ'], 
                                 weights={'RSI': 0.4, 'MACD': 0.3, 'KDJ': 0.3})
    print("  组合创建成功")
    print()
    
    # 验证指标有效性
    print("验证指标有效性:")
    for indicator_name in ['RSI', 'MACD', 'SMA']:
        effectiveness = calculator.validate_indicator_effectiveness(indicator_name, data)
        print(f"  {indicator_name}:")
        for metric, value in effectiveness.items():
            print(f"    {metric}: {value:.3f}")
    print()
    
    # 实时指标
    print("获取实时指标:")
    realtime_data = calculator.get_realtime_indicators(data, timeframe='1h')
    print(f"  时间周期: {realtime_data['timeframe']}")
    print(f"  计算指标数量: {len(realtime_data['indicators'])}")
    
    active_signals = {k: v for k, v in realtime_data['signals'].items() if v is not None}
    print(f"  活跃信号数量: {len(active_signals)}")
    
    for indicator, signal in list(active_signals.items())[:3]:  # 显示前3个信号
        print(f"    {indicator}: {signal.signal_type} (强度: {signal.strength:.2f})")
    print()
    
    # 自定义指标示例
    print("创建自定义指标:")
    
    def custom_momentum(data: pd.DataFrame, period: int = 10) -> pd.Series:
        """自定义动量指标"""
        return data['close'].pct_change(period) * 100
    
    custom_config = IndicatorConfig(
        name='CustomMomentum',
        parameters={'period': 10},
        timeframes=['1h', '4h', '1d']
    )
    
    calculator.register_custom_indicator('MOMENTUM', custom_momentum, custom_config)
    
    momentum_values = calculator.calculate_indicator('MOMENTUM', data, period=10)
    print(f"  自定义动量指标最新值: {momentum_values.iloc[-1]:.2f}%")
    print()
    
    print("=== 演示完成 ===")


if __name__ == "__main__":
    # 运行演示
    demo_technical_indicators()