import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class MovingAverage:
    """移动平均线计算器 - 核心技术指标"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """计算简单移动平均线"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """计算指数移动平均线"""
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_wma(data: pd.Series, window: int) -> pd.Series:
        """计算加权移动平均线"""
        weights = np.arange(1, window + 1)
        return data.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def calculate_multiple_ma(data: pd.DataFrame, 
                            price_column: str = 'close',
                            windows: List[int] = None) -> pd.DataFrame:
        """计算多种移动平均线"""
        if windows is None:
            windows = [5, 10, 20, 50, 200]
            
        result = data.copy()
        
        for window in windows:
            # SMA
            result[f'SMA_{window}'] = MovingAverage.calculate_sma(
                data[price_column], window
            )
            # EMA
            result[f'EMA_{window}'] = MovingAverage.calculate_ema(
                data[price_column], window
            )
            
        return result
    
    @staticmethod
    def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
        """生成移动平均线交易信号"""
        df = data.copy()
        
        # 确保有必要的列
        if 'SMA_20' not in df.columns or 'SMA_50' not in df.columns:
            df = MovingAverage.calculate_multiple_ma(df)
            
        # 生成信号
        df['signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
        
        # 金叉信号 (短期均线上穿长期均线)
        golden_cross = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
        df.loc[golden_cross, 'signal'] = 1
        
        # 死叉信号 (短期均线下穿长期均线)
        death_cross = (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))
        df.loc[death_cross, 'signal'] = -1
        
        return df
    
    @staticmethod
    def calculate_ma_bands(data: pd.DataFrame, 
                         ma_window: int = 20,
                         std_window: int = 20,
                         std_multiplier: float = 2.0) -> pd.DataFrame:
        """计算移动平均线通道"""
        df = data.copy()
        
        # 计算移动平均线
        df['MA'] = MovingAverage.calculate_sma(df['close'], ma_window)
        
        # 计算标准差
        df['STD'] = df['close'].rolling(window=std_window).std()
        
        # 计算通道上下轨
        df['MA_Upper'] = df['MA'] + (df['STD'] * std_multiplier)
        df['MA_Lower'] = df['MA'] - (df['STD'] * std_multiplier)
        
        return df