import pandas as pd
import numpy as np
from typing import Dict, List

class RSICalculator:
    """RSI相对强弱指标计算器 - 核心技术指标"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        # 计算价格变化
        delta = data.diff()
        
        # 分离上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # 计算RS
        rs = gain / loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_stochastic_rsi(rsi: pd.Series, window: int = 14) -> pd.Series:
        """计算随机RSI"""
        # 计算RSI的最小值和最大值
        min_rsi = rsi.rolling(window=window).min()
        max_rsi = rsi.rolling(window=window).max()
        
        # 计算随机RSI
        stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
        return stoch_rsi * 100
    
    @staticmethod
    def generate_rsi_signals(data: pd.DataFrame, 
                           price_column: str = 'close',
                           rsi_window: int = 14,
                           overbought: int = 70,
                           oversold: int = 30) -> pd.DataFrame:
        """生成RSI交易信号"""
        df = data.copy()
        
        # 计算RSI
        df['RSI'] = RSICalculator.calculate_rsi(df[price_column], rsi_window)
        
        # 生成信号
        df['rsi_signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
        
        # 超卖区域买入信号
        oversold_signal = (df['RSI'] < oversold) & (df['RSI'].shift(1) >= oversold)
        df.loc[oversold_signal, 'rsi_signal'] = 1
        
        # 超买区域卖出信号
        overbought_signal = (df['RSI'] > overbought) & (df['RSI'].shift(1) <= overbought)
        df.loc[overbought_signal, 'rsi_signal'] = -1
        
        # RSI背离检测
        df = RSICalculator._detect_divergence(df, price_column)
        
        return df
    
    @staticmethod
    def _detect_divergence(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """检测RSI背离"""
        df = df.copy()
        df['price_high'] = df[price_column].rolling(window=5).max()
        df['price_low'] = df[price_column].rolling(window=5).min()
        df['rsi_high'] = df['RSI'].rolling(window=5).max()
        df['rsi_low'] = df['RSI'].rolling(window=5).min()
        
        # 顶背离 (价格创新高，RSI未创新高)
        top_divergence = (df[price_column] == df['price_high']) & (df['RSI'] < df['rsi_high'].shift(5))
        df.loc[top_divergence, 'rsi_signal'] = -1
        
        # 底背离 (价格创新低，RSI未创新低)
        bottom_divergence = (df[price_column] == df['price_low']) & (df['RSI'] > df['rsi_low'].shift(5))
        df.loc[bottom_divergence, 'rsi_signal'] = 1
        
        return df
    
    @staticmethod
    def calculate_rsi_momentum(data: pd.DataFrame, 
                             short_window: int = 6,
                             long_window: int = 14) -> pd.DataFrame:
        """计算RSI动量"""
        df = data.copy()
        
        # 计算短期和长期RSI
        df['RSI_short'] = RSICalculator.calculate_rsi(df['close'], short_window)
        df['RSI_long'] = RSICalculator.calculate_rsi(df['close'], long_window)
        
        # 计算RSI动量
        df['RSI_momentum'] = df['RSI_short'] - df['RSI_long']
        
        return df
    
    @staticmethod
    def get_rsi_strength(rsi_value: float) -> str:
        """获取RSI强度描述"""
        if rsi_value >= 80:
            return "极度超买"
        elif rsi_value >= 70:
            return "超买"
        elif rsi_value >= 60:
            return "强势"
        elif rsi_value >= 40:
            return "中性"
        elif rsi_value >= 30:
            return "弱势"
        elif rsi_value >= 20:
            return "超卖"
        else:
            return "极度超卖"