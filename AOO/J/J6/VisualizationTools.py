#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J6可视化工具模块

这是一个完整的金融和统计数据可视化工具集，支持多种图表类型、
交互式功能、实时数据展示和异步处理。

主要功能：
1. 金融数据可视化（K线图、成交量图、技术指标图）
2. 统计图表工具（直方图、散点图、箱线图、热力图）
3. 交互式图表工具（Plotly、Bokeh集成）
4. 仪表板工具（实时数据展示、多图表布局）
5. 图表导出工具（PNG、SVG、PDF格式）
6. 异步图表生成和缓存
7. 完整的错误处理和日志记录

作者: J6开发团队
版本: 1.0.0
日期: 2025-11-06
"""

import asyncio
import logging
import os
import random
import sys
import time
import warnings
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, date2num
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

# 尝试导入可选依赖
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # 创建空的go模块以避免NameError
    class _DummyGo:
        Figure = object
    go = _DummyGo()

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.io import show as bokeh_show
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    # 创建空的bokeh对象以避免NameError
    class _DummyBokeh:
        figure = object
        output_file = object
        save = object
        ColumnDataSource = object
        HoverTool = object
        bokeh_show = object
    figure = _DummyBokeh.figure
    output_file = _DummyBokeh.output_file
    save = _DummyBokeh.save
    ColumnDataSource = _DummyBokeh.ColumnDataSource
    HoverTool = _DummyBokeh.HoverTool
    bokeh_show = _DummyBokeh.bokeh_show

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

# 设置中文字体 - 改进配置
import matplotlib.font_manager as fm
import platform

# 检测系统并设置合适的字体
system = platform.system()
if system == "Windows":
    # Windows系统
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system == "Darwin":
    # macOS系统
    chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS']
else:
    # Linux系统
    chinese_fonts = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']

# 查找系统中可用的中文字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_font = None
for font in chinese_fonts:
    if font in available_fonts:
        chinese_font = font
        break

# 设置字体
if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    # 如果没有找到中文字体，使用英文
    import warnings
    warnings.warn("未找到中文字体，图表中的中文可能无法正确显示")

plt.rcParams['axes.unicode_minus'] = False

# 配置日志
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, 'visualization.log')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """可视化工具基础异常类"""
    pass


class DataValidationError(VisualizationError):
    """数据验证错误"""
    pass


class ChartExportError(VisualizationError):
    """图表导出错误"""
    pass


class CacheError(VisualizationError):
    """缓存操作错误"""
    pass


def validate_data(func):
    """数据验证装饰器"""
    @wraps(func)
    def wrapper(self, data, *args, **kwargs):
        try:
            if data is None:
                raise DataValidationError(f"数据不能为空")
            
            # 检查DataFrame是否为空
            if isinstance(data, pd.DataFrame) and data.empty:
                raise DataValidationError(f"数据不能为空")
            
            # 检查Series是否为空
            if isinstance(data, pd.Series) and data.empty:
                raise DataValidationError(f"数据不能为空")
            
            # 检查数据类型
            if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
                raise DataValidationError(f"不支持的数据类型: {type(data)}")
            
            # 检查数值数据
            if isinstance(data, pd.DataFrame):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise DataValidationError("数据中没有数值列")
            elif isinstance(data, pd.Series):
                if not pd.api.types.is_numeric_dtype(data):
                    raise DataValidationError("Series数据不是数值类型")
            elif isinstance(data, np.ndarray):
                if not np.issubdtype(data.dtype, np.number):
                    raise DataValidationError("numpy数组不是数值类型")
            
            return func(self, data, *args, **kwargs)
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            raise DataValidationError(f"数据验证失败: {e}")
    return wrapper


def handle_exceptions(func):
    """异常处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
            raise VisualizationError(f"函数执行失败: {e}")
    return wrapper


class ChartCache:
    """图表缓存管理类"""
    
    def __init__(self, cache_dir: str = None):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        if cache_dir is None:
            self.cache_dir = CACHE_DIR
        else:
            self.cache_dir = cache_dir
        self.cache_dict = {}
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_key(self, func_name: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [func_name]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, pd.DataFrame):
                key_parts.append(f"{k}_{hash(v.to_string())}")
            elif isinstance(v, np.ndarray):
                key_parts.append(f"{k}_{hash(v.tobytes())}")
            else:
                key_parts.append(f"{k}_{v}")
        return "_".join(key_parts)
    
    def get(self, cache_key: str) -> Optional[Any]:
        """获取缓存"""
        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]
        return None
    
    def set(self, cache_key: str, value: Any):
        """设置缓存"""
        self.cache_dict[cache_key] = value
    
    def clear(self):
        """清空缓存"""
        self.cache_dict.clear()
    
    def cleanup_expired(self, max_age: int = 3600):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        for key, (timestamp, _) in self.cache_dict.items():
            if current_time - timestamp > max_age:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache_dict[key]


class BaseChart(ABC):
    """图表基类"""
    
    def __init__(self, width: int = 12, height: int = 8, dpi: int = 100):
        """
        初始化基础图表
        
        Args:
            width: 图表宽度（英寸）
            height: 图表高度（英寸）
            dpi: 分辨率
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.figure = None
        self.axes = None
    
    def create_chart(self, data: Any, **kwargs) -> Figure:
        """
        创建图表的默认实现
        
        Args:
            data: 图表数据
            **kwargs: 其他参数
            
        Returns:
            Figure对象
        """
        # 默认实现由子类重写
        raise NotImplementedError("子类必须实现create_chart方法")
    
    def setup_figure(self, subplot_count: int = 1) -> Figure:
        """设置图形"""
        self.figure = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
        
        if subplot_count == 1:
            self.axes = [self.figure.add_subplot(111)]
        else:
            self.axes = []
            for i in range(subplot_count):
                self.axes.append(self.figure.add_subplot(subplot_count, 1, i + 1))
        
        return self.figure
    
    def save_chart(self, filepath: str, format: str = 'png', **kwargs) -> None:
        """
        保存图表
        
        Args:
            filepath: 保存路径
            format: 文件格式 (png, svg, pdf)
            **kwargs: 其他保存参数
        """
        try:
            if self.figure is None:
                raise VisualizationError("图表未创建")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存图表
            if format.lower() == 'pdf':
                with PdfPages(filepath) as pdf:
                    pdf.savefig(self.figure, **kwargs)
            else:
                self.figure.savefig(filepath, format=format, **kwargs)
            
            logger.info(f"图表已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存图表失败: {e}")
            raise ChartExportError(f"保存图表失败: {e}")


class FinancialChart(BaseChart):
    """金融数据可视化类"""
    
    def __init__(self, width: int = 15, height: int = 10, dpi: int = 100):
        """
        初始化金融图表
        
        Args:
            width: 图表宽度
            height: 图表高度
            dpi: 分辨率
        """
        super().__init__(width, height, dpi)
        self.candlestick_data = None
        self.volume_data = None
        self.indicators = {}
    
    @validate_data
    def create_candlestick_chart(self, data: pd.DataFrame, 
                                open_col: str = 'open', high_col: str = 'high',
                                low_col: str = 'low', close_col: str = 'close',
                                volume_col: str = 'volume', title: str = "K线图") -> Figure:
        """
        创建K线图
        
        Args:
            data: 包含OHLCV数据的DataFrame
            open_col: 开盘价列名
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名
            title: 图表标题
            
        Returns:
            Figure对象
            
        Raises:
            DataValidationError: 数据验证失败
        """
        try:
            logger.info("开始创建K线图")
            
            # 设置图形布局
            self.setup_figure(3)
            gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
            
            # 主图 - K线
            ax1 = self.figure.add_subplot(gs[0])
            
            # 转换日期格式
            if 'date' in data.columns:
                dates = pd.to_datetime(data['date'])
            elif data.index.dtype == 'datetime64[ns]':
                dates = data.index
            else:
                dates = pd.to_datetime(data.index)
            
            # 绘制K线
            for i in range(len(data)):
                date = dates.iloc[i]
                open_price = data[open_col].iloc[i]
                high_price = data[high_col].iloc[i]
                low_price = data[low_col].iloc[i]
                close_price = data[close_col].iloc[i]
                
                # 确定颜色
                color = 'red' if close_price >= open_price else 'green'
                
                # 绘制影线
                ax1.plot([date, date], [low_price, high_price], color='black', linewidth=1)
                
                # 绘制实体
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                rect = patches.Rectangle((date - timedelta(days=0.3), body_bottom),
                                       timedelta(days=0.6), body_height,
                                       facecolor=color, edgecolor='black', alpha=0.8)
                ax1.add_patch(rect)
            
            # 设置主图
            ax1.set_title(title, fontsize=16, fontweight='bold')
            ax1.set_ylabel('价格', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            # 成交量图
            ax2 = self.figure.add_subplot(gs[1], sharex=ax1)
            
            colors = ['red' if data[close_col].iloc[i] >= data[open_col].iloc[i] 
                     else 'green' for i in range(len(data))]
            
            ax2.bar(dates, data[volume_col], color=colors, alpha=0.7, width=0.8)
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 隐藏主图的x轴标签
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            # 技术指标图（预留）
            ax3 = self.figure.add_subplot(gs[2], sharex=ax1)
            ax3.set_ylabel('指标', fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # 隐藏其他图的x轴标签
            plt.setp(ax2.get_xticklabels(), visible=False)
            
            # 调整布局
            plt.tight_layout()
            
            logger.info("K线图创建完成")
            return self.figure
            
        except Exception as e:
            logger.error(f"创建K线图失败: {e}")
            raise VisualizationError(f"创建K线图失败: {e}")
    
    @validate_data
    def add_technical_indicators(self, data: pd.DataFrame, 
                                indicators: List[str] = None,
                                **kwargs) -> None:
        """
        添加技术指标
        
        Args:
            data: 价格数据
            indicators: 指标列表 ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'kdj', 'williams', 'cci', 'stochastic', 'roc', 'atr']
            **kwargs: 指标参数
        """
        try:
            if indicators is None:
                indicators = ['sma', 'ema']
            
            ax3 = self.axes[2]  # 技术指标轴
            
            # 默认列名
            close_col = kwargs.get('close_col', 'close')
            high_col = kwargs.get('high_col', 'high')
            low_col = kwargs.get('low_col', 'low')
            
            for indicator in indicators:
                if indicator.lower() == 'sma':
                    self._add_sma(data, ax3, close_col=close_col, **kwargs)
                elif indicator.lower() == 'ema':
                    self._add_ema(data, ax3, close_col=close_col, **kwargs)
                elif indicator.lower() == 'rsi':
                    self._add_rsi(data, ax3, close_col=close_col, **kwargs)
                elif indicator.lower() == 'macd':
                    self._add_macd(data, ax3, close_col=close_col, **kwargs)
                elif indicator.lower() == 'bollinger':
                    self._add_bollinger_bands(data, ax3, close_col=close_col, **kwargs)
                elif indicator.lower() == 'kdj':
                    self._add_kdj(data, ax3, close_col=close_col, high_col=high_col, low_col=low_col, **kwargs)
                elif indicator.lower() == 'williams':
                    self._add_williams_r(data, ax3, close_col=close_col, high_col=high_col, low_col=low_col, **kwargs)
                elif indicator.lower() == 'cci':
                    self._add_cci(data, ax3, close_col=close_col, high_col=high_col, low_col=low_col, **kwargs)
                elif indicator.lower() == 'stochastic':
                    self._add_stochastic_oscillator(data, ax3, close_col=close_col, high_col=high_col, low_col=low_col, **kwargs)
                elif indicator.lower() == 'roc':
                    self._add_roc(data, ax3, close_col=close_col, **kwargs)
                elif indicator.lower() == 'atr':
                    self._add_atr(data, ax3, close_col=close_col, high_col=high_col, low_col=low_col, **kwargs)
            
            logger.info(f"已添加技术指标: {indicators}")
            
        except Exception as e:
            logger.error(f"添加技术指标失败: {e}")
            raise VisualizationError(f"添加技术指标失败: {e}")
    
    def _add_sma(self, data: pd.DataFrame, ax, period: int = 20, 
                close_col: str = 'close', color: str = 'blue') -> None:
        """添加简单移动平均线"""
        if close_col in data.columns:
            sma = data[close_col].rolling(window=period).mean()
            ax.plot(data.index, sma, label=f'SMA({period})', color=color, linewidth=2)
            ax.legend()
    
    def _add_ema(self, data: pd.DataFrame, ax, period: int = 12,
                close_col: str = 'close', color: str = 'orange') -> None:
        """添加指数移动平均线"""
        if close_col in data.columns:
            ema = data[close_col].ewm(span=period).mean()
            ax.plot(data.index, ema, label=f'EMA({period})', color=color, linewidth=2)
            ax.legend()
    
    def _add_rsi(self, data: pd.DataFrame, ax, period: int = 14,
                close_col: str = 'close') -> None:
        """添加RSI指标"""
        if close_col in data.columns:
            delta = data[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            ax.plot(data.index, rsi, label=f'RSI({period})', color='purple', linewidth=2)
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=30, color='g', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 100)
            ax.legend()
    
    def _add_macd(self, data: pd.DataFrame, ax, 
                 fast: int = 12, slow: int = 26, signal: int = 9,
                 close_col: str = 'close') -> None:
        """添加MACD指标"""
        if close_col in data.columns:
            ema_fast = data[close_col].ewm(span=fast).mean()
            ema_slow = data[close_col].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            ax.plot(data.index, macd_line, label='MACD', color='blue')
            ax.plot(data.index, signal_line, label='Signal', color='red')
            ax.bar(data.index, histogram, label='Histogram', alpha=0.3, color='gray')
            ax.legend()
    
    def _add_bollinger_bands(self, data: pd.DataFrame, ax, 
                           period: int = 20, std_dev: float = 2,
                           close_col: str = 'close') -> None:
        """添加布林带"""
        if close_col in data.columns:
            sma = data[close_col].rolling(window=period).mean()
            std = data[close_col].rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            ax.plot(data.index, sma, label=f'BB({period})', color='blue')
            ax.plot(data.index, upper_band, label='Upper', color='red', linestyle='--')
            ax.plot(data.index, lower_band, label='Lower', color='green', linestyle='--')
            ax.fill_between(data.index, upper_band, lower_band, alpha=0.1)
            ax.legend()
    
    def _add_kdj(self, data: pd.DataFrame, ax, 
                period: int = 9, close_col: str = 'close',
                high_col: str = 'high', low_col: str = 'low') -> None:
        """添加KDJ指标"""
        if close_col in data.columns:
            low_min = data[low_col].rolling(window=period).min()
            high_max = data[high_col].rolling(window=period).max()
            rsv = (data[close_col] - low_min) / (high_max - low_min) * 100
            
            k = rsv.ewm(com=2).mean()
            d = k.ewm(com=2).mean()
            j = 3 * k - 2 * d
            
            ax.plot(data.index, k, label='K', color='blue', linewidth=2)
            ax.plot(data.index, d, label='D', color='red', linewidth=2)
            ax.plot(data.index, j, label='J', color='orange', linewidth=1, alpha=0.7)
            ax.axhline(y=80, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=20, color='g', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 100)
            ax.legend()
    
    def _add_williams_r(self, data: pd.DataFrame, ax, 
                       period: int = 14, close_col: str = 'close',
                       high_col: str = 'high', low_col: str = 'low') -> None:
        """添加威廉指标%R"""
        if close_col in data.columns and low_col in data.columns and high_col in data.columns:
            high_max = data[high_col].rolling(window=period).max()
            low_min = data[low_col].rolling(window=period).min()
            
            wr = -100 * (high_max - data[close_col]) / (high_max - low_min)
            
            ax.plot(data.index, wr, label=f'Williams %R({period})', color='purple', linewidth=2)
            ax.axhline(y=-20, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=-80, color='g', linestyle='--', alpha=0.7)
            ax.set_ylim(-100, 0)
            ax.legend()
    
    def _add_cci(self, data: pd.DataFrame, ax, 
                period: int = 20, close_col: str = 'close',
                high_col: str = 'high', low_col: str = 'low') -> None:
        """添加商品通道指数CCI"""
        if close_col in data.columns and high_col in data.columns and low_col in data.columns:
            typical_price = (data[high_col] + data[low_col] + data[close_col]) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            
            ax.plot(data.index, cci, label=f'CCI({period})', color='brown', linewidth=2)
            ax.axhline(y=100, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=-100, color='g', linestyle='--', alpha=0.7)
            ax.legend()
    
    def _add_stochastic_oscillator(self, data: pd.DataFrame, ax, 
                                  period: int = 14, close_col: str = 'close',
                                  high_col: str = 'high', low_col: str = 'low') -> None:
        """添加随机振荡器"""
        if close_col in data.columns and low_col in data.columns and high_col in data.columns:
            low_min = data[low_col].rolling(window=period).min()
            high_max = data[high_col].rolling(window=period).max()
            
            k_percent = 100 * (data[close_col] - low_min) / (high_max - low_min)
            d_percent = k_percent.rolling(window=3).mean()
            
            ax.plot(data.index, k_percent, label=f'%K({period})', color='blue', linewidth=2)
            ax.plot(data.index, d_percent, label=f'%D(3)', color='red', linewidth=2)
            ax.axhline(y=80, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=20, color='g', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 100)
            ax.legend()
    
    def _add_roc(self, data: pd.DataFrame, ax, 
                period: int = 12, close_col: str = 'close') -> None:
        """添加变动率指标ROC"""
        if close_col in data.columns:
            roc = ((data[close_col] - data[close_col].shift(period)) / 
                   data[close_col].shift(period)) * 100
            
            ax.plot(data.index, roc, label=f'ROC({period})', color='teal', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.legend()
    
    def _add_atr(self, data: pd.DataFrame, ax, 
                period: int = 14, close_col: str = 'close',
                high_col: str = 'high', low_col: str = 'low') -> None:
        """添加平均真实范围ATR"""
        if close_col in data.columns and low_col in data.columns and high_col in data.columns:
            high_low = data[high_col] - data[low_col]
            high_close = np.abs(data[high_col] - data[close_col].shift())
            low_close = np.abs(data[low_col] - data[close_col].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            
            ax.plot(data.index, atr, label=f'ATR({period})', color='navy', linewidth=2)
            ax.legend()
    
    @validate_data
    def create_volume_chart(self, data: pd.DataFrame, 
                           volume_col: str = 'volume',
                           price_col: str = 'close',
                           title: str = "成交量分析图") -> Figure:
        """
        创建成交量分析图
        
        Args:
            data: 包含成交量和价格数据的DataFrame
            volume_col: 成交量列名
            price_col: 价格列名
            title: 图表标题
            
        Returns:
            Figure对象
        """
        try:
            logger.info("开始创建成交量分析图")
            
            self.setup_figure(2)
            
            # 成交量柱状图
            ax1 = self.axes[0]
            colors = ['red' if data[price_col].iloc[i] >= data[price_col].iloc[i-1] 
                     else 'green' for i in range(1, len(data))]
            colors.insert(0, 'blue')  # 第一个柱子的颜色
            
            ax1.bar(data.index, data[volume_col], color=colors, alpha=0.7)
            ax1.set_title(title, fontsize=16, fontweight='bold')
            ax1.set_ylabel('成交量', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # 成交量移动平均
            volume_sma = data[volume_col].rolling(window=20).mean()
            ax1.plot(data.index, volume_sma, color='orange', linewidth=2, label='成交量MA20')
            ax1.legend()
            
            # 价格与成交量关系散点图
            ax2 = self.axes[1]
            scatter = ax2.scatter(data[volume_col], data[price_col], 
                                c=range(len(data)), cmap='viridis', alpha=0.6)
            ax2.set_xlabel('成交量', fontsize=12)
            ax2.set_ylabel('价格', fontsize=12)
            ax2.set_title('价格与成交量关系', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax2, label='时间顺序')
            
            plt.tight_layout()
            logger.info("成交量分析图创建完成")
            return self.figure
            
        except Exception as e:
            logger.error(f"创建成交量分析图失败: {e}")
            raise VisualizationError(f"创建成交量分析图失败: {e}")


class StatisticalChart(BaseChart):
    """统计图表工具类"""
    
    def __init__(self, width: int = 12, height: int = 8, dpi: int = 100):
        """初始化统计图表工具"""
        super().__init__(width, height, dpi)
        self.style_config = {
            'hist': {'bins': 30, 'alpha': 0.7, 'density': True},
            'scatter': {'alpha': 0.6, 's': 50},
            'box': {'patch_artist': True},
            'heatmap': {'cmap': 'RdYlBu_r', 'center': 0}
        }
    
    @validate_data
    def create_histogram(self, data: Union[pd.Series, np.ndarray], 
                        bins: int = 30, title: str = "直方图",
                        xlabel: str = "值", ylabel: str = "频率",
                        density: bool = True, **kwargs) -> Figure:
        """
        创建直方图
        
        Args:
            data: 数据
            bins: 分箱数量
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
            density: 是否显示密度
            **kwargs: 其他参数
            
        Returns:
            Figure对象
        """
        try:
            logger.info("开始创建直方图")
            
            self.setup_figure()
            ax = self.axes[0]
            
            # 创建直方图
            n, bins_edges, patches = ax.hist(data, bins=bins, density=density, 
                                           alpha=0.7, color='skyblue', 
                                           edgecolor='black', linewidth=0.5)
            
            # 添加统计线
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.2f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1σ')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1σ')
            
            # 添加核密度估计
            if len(data) > 1:
                kde = gaussian_kde(data)
                x_range = np.linspace(min(data), max(data), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            logger.info("直方图创建完成")
            return self.figure
            
        except Exception as e:
            logger.error(f"创建直方图失败: {e}")
            raise VisualizationError(f"创建直方图失败: {e}")
    
    @validate_data
    def create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                          color_col: Optional[str] = None, size_col: Optional[str] = None,
                          title: str = "散点图", xlabel: Optional[str] = None,
                          ylabel: Optional[str] = None, **kwargs) -> Figure:
        """
        创建散点图
        
        Args:
            data: 数据DataFrame
            x_col: x轴列名
            y_col: y轴列名
            color_col: 颜色列名
            size_col: 大小列名
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
            **kwargs: 其他参数
            
        Returns:
            Figure对象
        """
        try:
            logger.info("开始创建散点图")
            
            self.setup_figure()
            ax = self.axes[0]
            
            # 设置点的属性
            scatter_kwargs = {'alpha': 0.6, 's': 50}
            scatter_kwargs.update(kwargs)
            
            # 基础散点图
            if color_col and size_col:
                scatter = ax.scatter(data[x_col], data[y_col], 
                                   c=data[color_col], s=data[size_col],
                                   **scatter_kwargs)
                plt.colorbar(scatter, ax=ax, label=color_col)
            elif color_col:
                scatter = ax.scatter(data[x_col], data[y_col], 
                                   c=data[color_col], **scatter_kwargs)
                plt.colorbar(scatter, ax=ax, label=color_col)
            elif size_col:
                scatter = ax.scatter(data[x_col], data[y_col], 
                                   s=data[size_col], **scatter_kwargs)
            else:
                scatter = ax.scatter(data[x_col], data[y_col], **scatter_kwargs)
            
            # 添加回归线
            if len(data) > 2:
                z = np.polyfit(data[x_col], data[y_col], 1)
                p = np.poly1d(z)
                ax.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8, linewidth=2)
                
                # 计算相关系数
                corr = np.corrcoef(data[x_col], data[y_col])[0, 1]
                ax.text(0.05, 0.95, f'相关系数: {corr:.3f}', 
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel(xlabel or x_col, fontsize=12)
            ax.set_ylabel(ylabel or y_col, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            logger.info("散点图创建完成")
            return self.figure
            
        except Exception as e:
            logger.error(f"创建散点图失败: {e}")
            raise VisualizationError(f"创建散点图失败: {e}")
    
    @validate_data
    def create_box_plot(self, data: pd.DataFrame, columns: List[str],
                       title: str = "箱线图", **kwargs) -> Figure:
        """
        创建箱线图
        
        Args:
            data: 数据DataFrame
            columns: 要绘制的列名列表
            title: 标题
            **kwargs: 其他参数
            
        Returns:
            Figure对象
        """
        try:
            logger.info("开始创建箱线图")
            
            self.setup_figure()
            ax = self.axes[0]
            
            # 准备数据
            plot_data = [data[col].dropna() for col in columns]
            
            # 创建箱线图
            box_plot = ax.boxplot(plot_data, labels=columns, patch_artist=True, **kwargs)
            
            # 设置颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(columns)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 添加统计信息
            for i, col in enumerate(columns):
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                median = data[col].median()
                mean = data[col].mean()
                
                ax.text(i+1, max(data[col]) * 1.05, f'Median: {median:.2f}', 
                       ha='center', fontsize=10)
                ax.text(i+1, max(data[col]) * 1.1, f'Mean: {mean:.2f}', 
                       ha='center', fontsize=10)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_ylabel('值', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            logger.info("箱线图创建完成")
            return self.figure
            
        except Exception as e:
            logger.error(f"创建箱线图失败: {e}")
            raise VisualizationError(f"创建箱线图失败: {e}")
    
    @validate_data
    def create_heatmap(self, data: pd.DataFrame, title: str = "热力图",
                      cmap: str = 'RdYlBu_r', center: float = 0,
                      annot: bool = True, fmt: str = '.2f', **kwargs) -> Figure:
        """
        创建热力图
        
        Args:
            data: 数据DataFrame
            title: 标题
            cmap: 颜色映射
            center: 中心值
            annot: 是否显示数值
            fmt: 数值格式
            **kwargs: 其他参数
            
        Returns:
            Figure对象
        """
        try:
            logger.info("开始创建热力图")
            
            self.setup_figure()
            ax = self.axes[0]
            
            # 创建热力图
            sns.heatmap(data, ax=ax, cmap=cmap, center=center, annot=annot, 
                       fmt=fmt, **kwargs)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('列', fontsize=12)
            ax.set_ylabel('行', fontsize=12)
            
            plt.tight_layout()
            logger.info("热力图创建完成")
            return self.figure
            
        except Exception as e:
            logger.error(f"创建热力图失败: {e}")
            raise VisualizationError(f"创建热力图失败: {e}")
    
    @validate_data
    def create_correlation_matrix(self, data: pd.DataFrame, 
                                 title: str = "相关性矩阵") -> Figure:
        """
        创建相关性矩阵热力图
        
        Args:
            data: 数据DataFrame
            title: 标题
            
        Returns:
            Figure对象
        """
        try:
            logger.info("开始创建相关性矩阵")
            
            # 计算相关性矩阵
            numeric_data = data.select_dtypes(include=[np.number])
            correlation_matrix = numeric_data.corr()
            
            return self.create_heatmap(correlation_matrix, title, 
                                     cmap='RdBu_r', annot=True, fmt='.2f')
            
        except Exception as e:
            logger.error(f"创建相关性矩阵失败: {e}")
            raise VisualizationError(f"创建相关性矩阵失败: {e}")
    
    @validate_data
    def create_violin_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                          title: str = "小提琴图", **kwargs) -> Figure:
        """
        创建小提琴图
        
        Args:
            data: 数据DataFrame
            x_col: x轴列名
            y_col: y轴列名
            title: 标题
            **kwargs: 其他参数
            
        Returns:
            Figure对象
        """
        try:
            logger.info("开始创建小提琴图")
            
            self.setup_figure()
            ax = self.axes[0]
            
            # 创建小提琴图
            sns.violinplot(data=data, x=x_col, y=y_col, ax=ax, **kwargs)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            logger.info("小提琴图创建完成")
            return self.figure
            
        except Exception as e:
            logger.error(f"创建小提琴图失败: {e}")
            raise VisualizationError(f"创建小提琴图失败: {e}")


class InteractiveChart:
    """交互式图表工具类"""
    
    def __init__(self):
        """初始化交互式图表工具"""
        self.plotly_available = PLOTLY_AVAILABLE
        self.bokeh_available = BOKEH_AVAILABLE
        
        if not self.plotly_available:
            logger.warning("Plotly未安装，交互式功能受限")
        if not self.bokeh_available:
            logger.warning("Bokeh未安装，交互式功能受限")
    
    @validate_data
    def create_plotly_candlestick(self, data: pd.DataFrame,
                                open_col: str = 'open', high_col: str = 'high',
                                low_col: str = 'low', close_col: str = 'close',
                                volume_col: str = 'volume', title: str = "交互式K线图"):
        """
        创建Plotly交互式K线图
        
        Args:
            data: OHLCV数据
            open_col: 开盘价列名
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名
            title: 标题
            
        Returns:
            Plotly Figure对象
        """
        if not self.plotly_available:
            raise VisualizationError("Plotly未安装，无法创建交互式图表")
        
        try:
            logger.info("开始创建Plotly交互式K线图")
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=('价格', '成交量')
            )
            
            # 添加K线图
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data[open_col],
                    high=data[high_col],
                    low=data[low_col],
                    close=data[close_col],
                    name='K线'
                ),
                row=1, col=1
            )
            
            # 添加成交量
            colors = ['red' if data[close_col].iloc[i] >= data[open_col].iloc[i] 
                     else 'green' for i in range(len(data))]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data[volume_col],
                    marker_color=colors,
                    name='成交量'
                ),
                row=2, col=1
            )
            
            # 更新布局
            fig.update_layout(
                title=title,
                xaxis_title='日期',
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            fig.update_yaxes(title_text="价格", row=1, col=1)
            fig.update_yaxes(title_text="成交量", row=2, col=1)
            
            logger.info("Plotly交互式K线图创建完成")
            return fig
            
        except Exception as e:
            logger.error(f"创建Plotly交互式K线图失败: {e}")
            raise VisualizationError(f"创建Plotly交互式K线图失败: {e}")
    
    @validate_data
    def create_plotly_scatter(self, data: pd.DataFrame, x_col: str, y_col: str,
                            color_col: Optional[str] = None, size_col: Optional[str] = None,
                            title: str = "交互式散点图"):
        """
        创建Plotly交互式散点图
        
        Args:
            data: 数据DataFrame
            x_col: x轴列名
            y_col: y轴列名
            color_col: 颜色列名
            size_col: 大小列名
            title: 标题
            
        Returns:
            Plotly Figure对象
        """
        if not self.plotly_available:
            raise VisualizationError("Plotly未安装，无法创建交互式图表")
        
        try:
            logger.info("开始创建Plotly交互式散点图")
            
            # 准备数据
            scatter_data = {
                'x': data[x_col],
                'y': data[y_col],
                'text': [f'{x_col}: {x}<br>{y_col}: {y}' for x, y in zip(data[x_col], data[y_col])]
            }
            
            if color_col:
                scatter_data['color'] = data[color_col]
            if size_col:
                scatter_data['size'] = data[size_col]
            
            # 创建散点图
            fig = px.scatter(
                data, 
                x=x_col, 
                y=y_col,
                color=color_col,
                size=size_col,
                title=title,
                hover_data=[x_col, y_col] + ([color_col] if color_col else []) + ([size_col] if size_col else [])
            )
            
            # 更新布局
            fig.update_layout(
                height=500,
                showlegend=True
            )
            
            logger.info("Plotly交互式散点图创建完成")
            return fig
            
        except Exception as e:
            logger.error(f"创建Plotly交互式散点图失败: {e}")
            raise VisualizationError(f"创建Plotly交互式散点图失败: {e}")
    
    @validate_data
    def create_bokeh_scatter(self, data: pd.DataFrame, x_col: str, y_col: str,
                           title: str = "Bokeh交互式散点图", 
                           output_file_path: str = None) -> Any:
        """
        创建Bokeh交互式散点图
        
        Args:
            data: 数据DataFrame
            x_col: x轴列名
            y_col: y轴列名
            title: 标题
            output_file_path: 输出文件路径
            
        Returns:
            Bokeh Figure对象
        """
        if not self.bokeh_available:
            raise VisualizationError("Bokeh未安装，无法创建交互式图表")
        
        try:
            logger.info("开始创建Bokeh交互式散点图")
            
            # 设置输出文件
            if output_file_path is None:
                output_file_path = os.path.join(BASE_DIR, 'bokeh_plot.html')
            output_file(output_file_path)
            
            # 创建数据源
            source = ColumnDataSource(data)
            
            # 创建图形
            p = figure(title=title, x_axis_label=x_col, y_axis_label=y_col,
                      tools="pan,wheel_zoom,box_zoom,reset,save")
            
            # 添加散点
            scatter = p.circle(x_col, y_col, source=source, size=8, alpha=0.6)
            
            # 添加悬停工具
            hover = HoverTool(
                tooltips=[
                    (x_col, f"@{x_col}"),
                    (y_col, f"@{y_col}")
                ]
            )
            p.add_tools(hover)
            
            # 显示图形
            bokeh_show(p)
            
            logger.info(f"Bokeh交互式散点图创建完成，保存到: {output_file_path}")
            return p
            
        except Exception as e:
            logger.error(f"创建Bokeh交互式散点图失败: {e}")
            raise VisualizationError(f"创建Bokeh交互式散点图失败: {e}")
    
    def save_plotly_figure(self, fig, filepath: str, 
                          format: str = 'html', **kwargs) -> None:
        """
        保存Plotly图表
        
        Args:
            fig: Plotly Figure对象
            filepath: 保存路径
            format: 文件格式 (html, png, svg, pdf)
            **kwargs: 其他参数
        """
        if not self.plotly_available:
            raise VisualizationError("Plotly未安装，无法保存图表")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if format.lower() == 'html':
                fig.write_html(filepath, **kwargs)
            elif format.lower() in ['png', 'svg', 'pdf']:
                if not KALEIDO_AVAILABLE:
                    raise ChartExportError("Kaleido未安装，无法导出静态图像")
                fig.write_image(filepath, format=format, **kwargs)
            else:
                raise ChartExportError(f"不支持的格式: {format}")
            
            logger.info(f"Plotly图表已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存Plotly图表失败: {e}")
            raise ChartExportError(f"保存Plotly图表失败: {e}")


class Dashboard:
    """仪表板工具类"""
    
    def __init__(self, width: int = 20, height: int = 12, dpi: int = 100):
        """
        初始化仪表板
        
        Args:
            width: 仪表板宽度
            height: 仪表板高度
            dpi: 分辨率
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.charts = {}
        self.layout = None
        self.real_time_data = {}
        self.update_interval = 5  # 秒
    
    def add_chart(self, chart_id: str, chart_type: str, data: Any, 
                  position: Tuple[int, int, int, int], **kwargs) -> None:
        """
        添加图表到仪表板
        
        Args:
            chart_id: 图表ID
            chart_type: 图表类型
            data: 图表数据
            position: 位置 (row, col, rowspan, colspan)
            **kwargs: 图表参数
        """
        try:
            self.charts[chart_id] = {
                'type': chart_type,
                'data': data,
                'position': position,
                'kwargs': kwargs,
                'created_at': datetime.now()
            }
            logger.info(f"图表 {chart_id} 已添加到仪表板")
        except Exception as e:
            logger.error(f"添加图表失败: {e}")
            raise VisualizationError(f"添加图表失败: {e}")
    
    def create_dashboard_layout(self, rows: int = 4, cols: int = 4) -> None:
        """
        创建仪表板布局
        
        Args:
            rows: 行数
            cols: 列数
        """
        try:
            self.layout = GridSpec(rows, cols, figure=plt.figure(figsize=(self.width, self.height)))
            logger.info(f"仪表板布局已创建: {rows}x{cols}")
        except Exception as e:
            logger.error(f"创建仪表板布局失败: {e}")
            raise VisualizationError(f"创建仪表板布局失败: {e}")
    
    def render_dashboard(self) -> Figure:
        """
        渲染仪表板
        
        Returns:
            渲染后的Figure对象
        """
        try:
            if not self.layout:
                self.create_dashboard_layout()
            
            fig = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
            fig.suptitle('金融数据仪表板', fontsize=20, fontweight='bold')
            
            chart_objects = {}
            
            for chart_id, chart_info in self.charts.items():
                try:
                    row, col, rowspan, colspan = chart_info['position']
                    
                    # 创建子图
                    ax = fig.add_subplot(self.layout[row:row+rowspan, col:col+colspan])
                    
                    # 根据图表类型创建相应的图表
                    if chart_info['type'] == 'candlestick':
                        chart_objects[chart_id] = self._create_candlestick_subplot(
                            chart_info['data'], ax, **chart_info['kwargs']
                        )
                    elif chart_info['type'] == 'volume':
                        chart_objects[chart_id] = self._create_volume_subplot(
                            chart_info['data'], ax, **chart_info['kwargs']
                        )
                    elif chart_info['type'] == 'line':
                        chart_objects[chart_id] = self._create_line_subplot(
                            chart_info['data'], ax, **chart_info['kwargs']
                        )
                    elif chart_info['type'] == 'scatter':
                        chart_objects[chart_id] = self._create_scatter_subplot(
                            chart_info['data'], ax, **chart_info['kwargs']
                        )
                    elif chart_info['type'] == 'heatmap':
                        chart_objects[chart_id] = self._create_heatmap_subplot(
                            chart_info['data'], ax, **chart_info['kwargs']
                        )
                    
                except Exception as e:
                    logger.error(f"创建图表 {chart_id} 失败: {e}")
                    continue
            
            plt.tight_layout()
            logger.info("仪表板渲染完成")
            return fig
            
        except Exception as e:
            logger.error(f"渲染仪表板失败: {e}")
            raise VisualizationError(f"渲染仪表板失败: {e}")
    
    def _create_candlestick_subplot(self, data: pd.DataFrame, ax, 
                                  open_col: str = 'open', high_col: str = 'high',
                                  low_col: str = 'low', close_col: str = 'close') -> None:
        """创建K线子图"""
        try:
            # 简化的K线图实现
            for i in range(len(data)):
                open_price = data[open_col].iloc[i]
                high_price = data[high_col].iloc[i]
                low_price = data[low_col].iloc[i]
                close_price = data[close_col].iloc[i]
                
                color = 'red' if close_price >= open_price else 'green'
                ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)
                
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                rect = patches.Rectangle((i-0.3, body_bottom), 0.6, body_height,
                                       facecolor=color, edgecolor='black', alpha=0.8)
                ax.add_patch(rect)
            
            ax.set_title('K线图', fontsize=12)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"创建K线子图失败: {e}")
    
    def _create_volume_subplot(self, data: pd.DataFrame, ax, 
                             volume_col: str = 'volume', price_col: str = 'close') -> None:
        """创建成交量子图"""
        try:
            colors = ['red' if data[price_col].iloc[i] >= data[price_col].iloc[i-1] 
                     else 'green' for i in range(1, len(data))]
            colors.insert(0, 'blue')
            
            ax.bar(data.index, data[volume_col], color=colors, alpha=0.7)
            ax.set_title('成交量', fontsize=12)
            ax.set_ylabel('Volume')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"创建成交量子图失败: {e}")
    
    def _create_line_subplot(self, data: pd.Series, ax, title: str = "趋势图") -> None:
        """创建线图子图"""
        try:
            ax.plot(data.index, data.values, linewidth=2)
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"创建线图子图失败: {e}")
    
    def _create_scatter_subplot(self, data: pd.DataFrame, ax, x_col: str, y_col: str) -> None:
        """创建散点图子图"""
        try:
            ax.scatter(data[x_col], data[y_col], alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title('散点图', fontsize=12)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"创建散点图子图失败: {e}")
    
    def _create_heatmap_subplot(self, data: pd.DataFrame, ax) -> None:
        """创建热力图子图"""
        try:
            sns.heatmap(data, ax=ax, cmap='RdYlBu_r', center=0, cbar=False)
            ax.set_title('热力图', fontsize=12)
            
        except Exception as e:
            logger.error(f"创建热力图子图失败: {e}")
    
    def set_real_time_data(self, chart_id: str, data_source: Callable) -> None:
        """
        设置实时数据源
        
        Args:
            chart_id: 图表ID
            data_source: 数据源函数
        """
        self.real_time_data[chart_id] = data_source
        logger.info(f"已设置图表 {chart_id} 的实时数据源")
    
    async def update_real_time_charts(self) -> None:
        """更新实时图表"""
        try:
            for chart_id, data_source in self.real_time_data.items():
                if chart_id in self.charts:
                    # 获取新数据
                    new_data = data_source()
                    self.charts[chart_id]['data'] = new_data
                    self.charts[chart_id]['updated_at'] = datetime.now()
            
            logger.info("实时图表更新完成")
        except Exception as e:
            logger.error(f"更新实时图表失败: {e}")
    
    def start_real_time_updates(self, duration: int = 300) -> None:
        """
        启动实时更新
        
        Args:
            duration: 持续时间（秒）
        """
        logger.info(f"启动实时更新，持续时间: {duration}秒")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                asyncio.run(self.update_real_time_charts())
                time.sleep(self.update_interval)
            except KeyboardInterrupt:
                logger.info("实时更新被用户中断")
                break
            except Exception as e:
                logger.error(f"实时更新错误: {e}")
                time.sleep(self.update_interval)


class ChartExporter:
    """图表导出工具类"""
    
    def __init__(self):
        """初始化导出工具"""
        self.supported_formats = ['png', 'jpg', 'svg', 'pdf', 'eps']
        self.default_quality = 300  # DPI
    
    def export_matplotlib_chart(self, figure: Figure, filepath: str, 
                              format: str = 'png', quality: int = None,
                              **kwargs) -> None:
        """
        导出Matplotlib图表
        
        Args:
            figure: Matplotlib Figure对象
            filepath: 保存路径
            format: 文件格式
            quality: 质量（DPI）
            **kwargs: 其他参数
        """
        try:
            if format.lower() not in self.supported_formats:
                raise ChartExportError(f"不支持的格式: {format}")
            
            # 设置默认参数
            if quality is None:
                quality = self.default_quality
            
            dpi = quality
            bbox_inches = kwargs.get('bbox_inches', 'tight')
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存图表
            if format.lower() == 'pdf':
                with PdfPages(filepath) as pdf:
                    pdf.savefig(figure, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
            else:
                figure.savefig(filepath, format=format, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
            
            logger.info(f"Matplotlib图表已导出到: {filepath} (格式: {format}, DPI: {dpi})")
            
        except Exception as e:
            logger.error(f"导出Matplotlib图表失败: {e}")
            raise ChartExportError(f"导出Matplotlib图表失败: {e}")
    
    def export_plotly_chart(self, figure, filepath: str, 
                          format: str = 'html', **kwargs) -> None:
        """
        导出Plotly图表
        
        Args:
            figure: Plotly Figure对象
            filepath: 保存路径
            format: 文件格式
            **kwargs: 其他参数
        """
        if not PLOTLY_AVAILABLE:
            raise ChartExportError("Plotly未安装，无法导出图表")
        
        try:
            if format.lower() == 'html':
                figure.write_html(filepath, **kwargs)
            elif format.lower() in ['png', 'svg', 'pdf']:
                if not KALEIDO_AVAILABLE:
                    raise ChartExportError("Kaleido未安装，无法导出静态图像")
                figure.write_image(filepath, format=format, **kwargs)
            else:
                raise ChartExportError(f"不支持的格式: {format}")
            
            logger.info(f"Plotly图表已导出到: {filepath} (格式: {format})")
            
        except Exception as e:
            logger.error(f"导出Plotly图表失败: {e}")
            raise ChartExportError(f"导出Plotly图表失败: {e}")
    
    def batch_export(self, charts: Dict[str, Tuple[Any, str]], 
                    output_dir: str, format: str = 'png') -> None:
        """
        批量导出图表
        
        Args:
            charts: 图表字典 {chart_id: (figure, format)}
            output_dir: 输出目录
            format: 默认格式
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for chart_id, (figure, chart_format) in charts.items():
                filepath = os.path.join(output_dir, f"{chart_id}.{chart_format}")
                
                if hasattr(figure, 'savefig'):  # Matplotlib figure
                    self.export_matplotlib_chart(figure, filepath, chart_format)
                elif hasattr(figure, 'write_html'):  # Plotly figure
                    self.export_plotly_chart(figure, filepath, chart_format)
                else:
                    logger.warning(f"不支持的图表类型: {type(figure)}")
            
            logger.info(f"批量导出完成，输出目录: {output_dir}")
            
        except Exception as e:
            logger.error(f"批量导出失败: {e}")
            raise ChartExportError(f"批量导出失败: {e}")
    
    def create_report(self, charts: List[Tuple[Any, str, str]], 
                     output_path: str, title: str = "数据分析报告") -> None:
        """
        创建PDF报告
        
        Args:
            charts: 图表列表 [(figure, format, description)]
            output_path: 输出路径
            title: 报告标题
        """
        try:
            with PdfPages(output_path) as pdf:
                # 封面页
                fig_cover = plt.figure(figsize=(8.5, 11))
                fig_cover.text(0.5, 0.7, title, fontsize=24, ha='center', va='center')
                fig_cover.text(0.5, 0.3, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                              fontsize=12, ha='center', va='center')
                pdf.savefig(fig_cover)
                plt.close(fig_cover)
                
                # 图表页
                for figure, format_type, description in charts:
                    fig = plt.figure(figsize=(8.5, 11))
                    fig.text(0.5, 0.95, description, fontsize=16, ha='center', va='top')
                    
                    # 这里需要将figure的内容复制到新figure中
                    # 简化实现，直接保存原figure
                    pdf.savefig(figure, bbox_inches='tight')
                    plt.close(fig)
            
            logger.info(f"报告已创建: {output_path}")
            
        except Exception as e:
            logger.error(f"创建报告失败: {e}")
            raise ChartExportError(f"创建报告失败: {e}")


class AsyncVisualizationTools:
    """异步可视化工具类"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化异步可视化工具
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = ChartCache()
        self.chart_creators = {
            'financial': FinancialChart,
            'statistical': StatisticalChart,
            'interactive': InteractiveChart
        }
    
    async def create_chart_async(self, chart_type: str, data: Any, 
                               cache_key: str = None, **kwargs) -> Any:
        """
        异步创建图表
        
        Args:
            chart_type: 图表类型
            data: 图表数据
            cache_key: 缓存键
            **kwargs: 图表参数
            
        Returns:
            创建的图表对象
        """
        try:
            # 检查缓存
            if cache_key:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.info(f"从缓存获取图表: {cache_key}")
                    return cached_result
            
            # 在线程池中执行图表创建
            loop = asyncio.get_event_loop()
            chart_creator = self.chart_creators.get(chart_type)
            
            if not chart_creator:
                raise VisualizationError(f"不支持的图表类型: {chart_type}")
            
            # 创建图表
            chart = chart_creator(**kwargs)
            figure = await loop.run_in_executor(
                self.executor, 
                self._create_chart_sync, 
                chart, data, kwargs
            )
            
            # 缓存结果
            if cache_key:
                self.cache.set(cache_key, figure)
            
            logger.info(f"异步创建图表完成: {chart_type}")
            return figure
            
        except Exception as e:
            logger.error(f"异步创建图表失败: {e}")
            raise VisualizationError(f"异步创建图表失败: {e}")
    
    def _create_chart_sync(self, chart: Any, data: Any, kwargs: Dict) -> Any:
        """同步创建图表"""
        try:
            if isinstance(chart, FinancialChart):
                if 'candlestick' in kwargs.get('chart_type', ''):
                    return chart.create_candlestick_chart(data, **kwargs)
                elif 'volume' in kwargs.get('chart_type', ''):
                    return chart.create_volume_chart(data, **kwargs)
            elif isinstance(chart, StatisticalChart):
                chart_type = kwargs.get('chart_type', 'histogram')
                if chart_type == 'histogram':
                    return chart.create_histogram(data, **kwargs)
                elif chart_type == 'scatter':
                    return chart.create_scatter_plot(data, **kwargs)
                elif chart_type == 'box':
                    return chart.create_box_plot(data, **kwargs)
                elif chart_type == 'heatmap':
                    return chart.create_heatmap(data, **kwargs)
            
            return chart.create_chart(data, **kwargs)
            
        except Exception as e:
            logger.error(f"同步创建图表失败: {e}")
            raise VisualizationError(f"同步创建图表失败: {e}")
    
    async def create_dashboard_async(self, charts_config: List[Dict], 
                                   output_path: str) -> None:
        """
        异步创建仪表板
        
        Args:
            charts_config: 图表配置列表
            output_path: 输出路径
        """
        try:
            logger.info("开始异步创建仪表板")
            
            # 创建仪表板
            dashboard = Dashboard()
            
            # 并行创建图表
            tasks = []
            for config in charts_config:
                task = self.create_chart_async(
                    config['type'], 
                    config['data'], 
                    cache_key=config.get('cache_key'),
                    **config.get('kwargs', {})
                )
                tasks.append(task)
            
            # 等待所有图表创建完成
            chart_results = await asyncio.gather(*tasks)
            
            # 添加图表到仪表板
            for i, (config, figure) in enumerate(zip(charts_config, chart_results)):
                chart_id = config.get('id', f'chart_{i}')
                position = config.get('position', (i, 0, 1, 1))
                dashboard.add_chart(chart_id, config['type'], config['data'], position)
            
            # 渲染仪表板
            dashboard_fig = dashboard.render_dashboard()
            
            # 保存仪表板
            exporter = ChartExporter()
            exporter.export_matplotlib_chart(dashboard_fig, output_path)
            
            logger.info(f"异步仪表板创建完成: {output_path}")
            
        except Exception as e:
            logger.error(f"异步创建仪表板失败: {e}")
            raise VisualizationError(f"异步创建仪表板失败: {e}")
    
    async def batch_process_charts(self, chart_requests: List[Dict]) -> List[Any]:
        """
        批量处理图表请求
        
        Args:
            chart_requests: 图表请求列表
            
        Returns:
            图表结果列表
        """
        try:
            logger.info(f"开始批量处理 {len(chart_requests)} 个图表请求")
            
            # 并行处理所有请求
            tasks = []
            for request in chart_requests:
                task = self.create_chart_async(
                    request['type'],
                    request['data'],
                    cache_key=request.get('cache_key'),
                    **request.get('kwargs', {})
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"图表请求 {i} 失败: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            logger.info(f"批量处理完成，成功: {len([r for r in processed_results if r is not None])}")
            return processed_results
            
        except Exception as e:
            logger.error(f"批量处理图表失败: {e}")
            raise VisualizationError(f"批量处理图表失败: {e}")
    
    def cleanup_cache(self, max_age: int = 3600) -> None:
        """
        清理缓存
        
        Args:
            max_age: 最大缓存时间（秒）
        """
        try:
            self.cache.cleanup_expired(max_age)
            logger.info("缓存清理完成")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            raise CacheError(f"清理缓存失败: {e}")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class VisualizationTools:
    """可视化工具主类"""
    
    def __init__(self, cache_enabled: bool = True, async_enabled: bool = True):
        """
        初始化可视化工具
        
        Args:
            cache_enabled: 是否启用缓存
            async_enabled: 是否启用异步功能
        """
        self.cache_enabled = cache_enabled
        self.async_enabled = async_enabled
        
        # 初始化各个组件
        self.financial_chart = FinancialChart()
        self.statistical_chart = StatisticalChart()
        self.interactive_chart = InteractiveChart()
        self.dashboard = Dashboard()
        self.exporter = ChartExporter()
        
        if async_enabled:
            self.async_tools = AsyncVisualizationTools()
        else:
            self.async_tools = None
        
        if cache_enabled:
            self.cache = ChartCache()
        else:
            self.cache = None
        
        logger.info("可视化工具初始化完成")
    
    @handle_exceptions
    def create_financial_chart(self, data: pd.DataFrame, chart_type: str = 'candlestick',
                             **kwargs) -> Figure:
        """
        创建金融图表
        
        Args:
            data: 金融数据
            chart_type: 图表类型 ('candlestick', 'volume')
            **kwargs: 其他参数
            
        Returns:
            Matplotlib Figure对象
        """
        try:
            logger.info(f"创建金融图表: {chart_type}")
            
            if chart_type == 'candlestick':
                return self.financial_chart.create_candlestick_chart(data, **kwargs)
            elif chart_type == 'volume':
                return self.financial_chart.create_volume_chart(data, **kwargs)
            else:
                raise ValueError(f"不支持的金融图表类型: {chart_type}")
                
        except Exception as e:
            logger.error(f"创建金融图表失败: {e}")
            raise VisualizationError(f"创建金融图表失败: {e}")
    
    @handle_exceptions
    def create_statistical_chart(self, data: Union[pd.DataFrame, pd.Series], 
                                chart_type: str = 'histogram', **kwargs) -> Figure:
        """
        创建统计图表
        
        Args:
            data: 统计数据
            chart_type: 图表类型 ('histogram', 'scatter', 'box', 'heatmap', 'violin')
            **kwargs: 其他参数
            
        Returns:
            Matplotlib Figure对象
        """
        try:
            logger.info(f"创建统计图表: {chart_type}")
            
            if chart_type == 'histogram':
                return self.statistical_chart.create_histogram(data, **kwargs)
            elif chart_type == 'scatter':
                return self.statistical_chart.create_scatter_plot(data, **kwargs)
            elif chart_type == 'box':
                return self.statistical_chart.create_box_plot(data, **kwargs)
            elif chart_type == 'heatmap':
                return self.statistical_chart.create_heatmap(data, **kwargs)
            elif chart_type == 'violin':
                return self.statistical_chart.create_violin_plot(data, **kwargs)
            elif chart_type == 'correlation':
                return self.statistical_chart.create_correlation_matrix(data)
            else:
                raise ValueError(f"不支持的统计图表类型: {chart_type}")
                
        except Exception as e:
            logger.error(f"创建统计图表失败: {e}")
            raise VisualizationError(f"创建统计图表失败: {e}")
    
    @handle_exceptions
    def create_interactive_chart(self, data: pd.DataFrame, chart_type: str = 'scatter',
                               library: str = 'plotly', **kwargs) -> Any:
        """
        创建交互式图表
        
        Args:
            data: 数据
            chart_type: 图表类型
            library: 图表库 ('plotly', 'bokeh')
            **kwargs: 其他参数
            
        Returns:
            交互式图表对象
        """
        try:
            logger.info(f"创建交互式图表: {chart_type} ({library})")
            
            if library == 'plotly':
                if chart_type == 'candlestick':
                    return self.interactive_chart.create_plotly_candlestick(data, **kwargs)
                elif chart_type == 'scatter':
                    return self.interactive_chart.create_plotly_scatter(data, **kwargs)
                else:
                    raise ValueError(f"Plotly不支持的图表类型: {chart_type}")
            elif library == 'bokeh':
                if chart_type == 'scatter':
                    return self.interactive_chart.create_bokeh_scatter(data, **kwargs)
                else:
                    raise ValueError(f"Bokeh不支持的图表类型: {chart_type}")
            else:
                raise ValueError(f"不支持的图表库: {library}")
                
        except Exception as e:
            logger.error(f"创建交互式图表失败: {e}")
            raise VisualizationError(f"创建交互式图表失败: {e}")
    
    @handle_exceptions
    def create_dashboard(self, charts_config: List[Dict], output_path: str) -> Figure:
        """
        创建仪表板
        
        Args:
            charts_config: 图表配置列表
            output_path: 输出路径
            
        Returns:
            仪表板Figure对象
        """
        try:
            logger.info(f"创建仪表板，包含 {len(charts_config)} 个图表")
            
            # 清空现有图表
            self.dashboard.charts.clear()
            
            # 添加图表
            for config in charts_config:
                self.dashboard.add_chart(
                    config['id'],
                    config['type'],
                    config['data'],
                    config['position']
                )
            
            # 渲染仪表板
            dashboard_fig = self.dashboard.render_dashboard()
            
            # 保存仪表板
            self.exporter.export_matplotlib_chart(dashboard_fig, output_path)
            
            logger.info(f"仪表板创建完成: {output_path}")
            return dashboard_fig
            
        except Exception as e:
            logger.error(f"创建仪表板失败: {e}")
            raise VisualizationError(f"创建仪表板失败: {e}")
    
    @handle_exceptions
    async def create_dashboard_async(self, charts_config: List[Dict], 
                                   output_path: str) -> None:
        """
        异步创建仪表板
        
        Args:
            charts_config: 图表配置列表
            output_path: 输出路径
        """
        if not self.async_enabled:
            raise VisualizationError("异步功能未启用")
        
        await self.async_tools.create_dashboard_async(charts_config, output_path)
    
    @handle_exceptions
    def export_chart(self, figure: Any, filepath: str, format: str = 'png', 
                    library: str = 'matplotlib', **kwargs) -> None:
        """
        导出图表
        
        Args:
            figure: 图表对象
            filepath: 保存路径
            format: 文件格式
            library: 图表库
            **kwargs: 其他参数
        """
        try:
            logger.info(f"导出图表: {filepath} (格式: {format})")
            
            if library == 'matplotlib':
                self.exporter.export_matplotlib_chart(figure, filepath, format, **kwargs)
            elif library == 'plotly':
                self.exporter.export_plotly_chart(figure, filepath, format, **kwargs)
            else:
                raise ValueError(f"不支持的图表库: {library}")
                
        except Exception as e:
            logger.error(f"导出图表失败: {e}")
            raise ChartExportError(f"导出图表失败: {e}")
    
    @handle_exceptions
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Args:
            data: 数据DataFrame
            
        Returns:
            统计信息字典
        """
        try:
            logger.info("计算数据统计信息")
            
            stats = {}
            numeric_data = data.select_dtypes(include=[np.number])
            
            for col in numeric_data.columns:
                stats[col] = {
                    'mean': float(numeric_data[col].mean()),
                    'std': float(numeric_data[col].std()),
                    'min': float(numeric_data[col].min()),
                    'max': float(numeric_data[col].max()),
                    'median': float(numeric_data[col].median()),
                    'q25': float(numeric_data[col].quantile(0.25)),
                    'q75': float(numeric_data[col].quantile(0.75)),
                    'skewness': float(numeric_data[col].skew()),
                    'kurtosis': float(numeric_data[col].kurtosis())
                }
            
            logger.info("数据统计信息计算完成")
            return stats
            
        except Exception as e:
            logger.error(f"计算数据统计信息失败: {e}")
            raise VisualizationError(f"计算数据统计信息失败: {e}")
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.async_tools:
                self.async_tools.cleanup_cache()
            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


# 使用示例和测试函数
def create_sample_financial_data(days: int = 100) -> pd.DataFrame:
    """创建示例金融数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # 生成模拟股价数据
    price_base = 100
    returns = np.random.normal(0.001, 0.02, days)
    prices = [price_base]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def create_sample_statistical_data(n: int = 1000) -> pd.DataFrame:
    """创建示例统计数据"""
    np.random.seed(42)
    
    # 生成相关数据
    x = np.random.normal(0, 1, n)
    y = x * 0.5 + np.random.normal(0, 0.5, n)
    z = np.random.exponential(2, n)
    w = np.random.uniform(-2, 2, n)
    
    return pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'w': w,
        'category': np.random.choice(['A', 'B', 'C'], n)
    })


def demo_financial_charts():
    """演示金融图表功能"""
    print("=== 金融图表演示 ===")
    
    # 创建可视化工具
    viz_tools = VisualizationTools()
    
    # 创建示例数据
    financial_data = create_sample_financial_data(50)
    
    # 创建K线图
    print("创建K线图...")
    candlestick_fig = viz_tools.create_financial_chart(
        financial_data, 
        chart_type='candlestick',
        title='示例K线图'
    )
    viz_tools.export_chart(candlestick_fig, os.path.join(BASE_DIR, 'candlestick_demo.png'))
    
    # 创建成交量图
    print("创建成交量图...")
    volume_fig = viz_tools.create_financial_chart(
        financial_data,
        chart_type='volume',
        title='示例成交量图'
    )
    viz_tools.export_chart(volume_fig, os.path.join(BASE_DIR, 'volume_demo.png'))
    
    print("金融图表演示完成")


def demo_statistical_charts():
    """演示统计图表功能"""
    print("=== 统计图表演示 ===")
    
    viz_tools = VisualizationTools()
    stat_data = create_sample_statistical_data(500)
    
    # 创建直方图
    print("创建直方图...")
    hist_fig = viz_tools.create_statistical_chart(
        stat_data['x'],
        chart_type='histogram',
        title='X变量分布'
    )
    viz_tools.export_chart(hist_fig, os.path.join(BASE_DIR, 'histogram_demo.png'))
    
    # 创建散点图
    print("创建散点图...")
    scatter_fig = viz_tools.create_statistical_chart(
        stat_data,
        chart_type='scatter',
        x_col='x',
        y_col='y',
        title='X-Y关系散点图'
    )
    viz_tools.export_chart(scatter_fig, os.path.join(BASE_DIR, 'scatter_demo.png'))
    
    # 创建箱线图
    print("创建箱线图...")
    box_fig = viz_tools.create_statistical_chart(
        stat_data,
        chart_type='box',
        columns=['x', 'y', 'z'],
        title='多变量箱线图'
    )
    viz_tools.export_chart(box_fig, os.path.join(BASE_DIR, 'box_demo.png'))
    
    # 创建热力图
    print("创建热力图...")
    correlation_data = stat_data[['x', 'y', 'z', 'w']].corr()
    heatmap_fig = viz_tools.create_statistical_chart(
        correlation_data,
        chart_type='heatmap',
        title='相关性矩阵'
    )
    viz_tools.export_chart(heatmap_fig, os.path.join(BASE_DIR, 'heatmap_demo.png'))
    
    print("统计图表演示完成")


def demo_interactive_charts():
    """演示交互式图表功能"""
    print("=== 交互式图表演示 ===")
    
    viz_tools = VisualizationTools()
    stat_data = create_sample_statistical_data(200)
    
    if PLOTLY_AVAILABLE:
        # 创建Plotly散点图
        print("创建Plotly交互式散点图...")
        plotly_fig = viz_tools.create_interactive_chart(
            stat_data,
            chart_type='scatter',
            library='plotly',
            x_col='x',
            y_col='y',
            color_col='category',
            title='Plotly交互式散点图'
        )
        viz_tools.export_chart(plotly_fig, os.path.join(BASE_DIR, 'plotly_scatter.html'), 
                              format='html', library='plotly')
        
        # 创建Plotly K线图
        print("创建Plotly交互式K线图...")
        financial_data = create_sample_financial_data(30)
        plotly_candlestick = viz_tools.create_interactive_chart(
            financial_data,
            chart_type='candlestick',
            library='plotly',
            title='Plotly交互式K线图'
        )
        viz_tools.export_chart(plotly_candlestick, os.path.join(BASE_DIR, 'plotly_candlestick.html'),
                              format='html', library='plotly')
    
    if BOKEH_AVAILABLE:
        # 创建Bokeh散点图
        print("创建Bokeh交互式散点图...")
        bokeh_fig = viz_tools.create_interactive_chart(
            stat_data,
            chart_type='scatter',
            library='bokeh',
            x_col='x',
            y_col='y',
            title='Bokeh交互式散点图',
            output_file_path=os.path.join(BASE_DIR, 'bokeh_scatter.html')
        )
    
    print("交互式图表演示完成")


def demo_dashboard():
    """演示仪表板功能"""
    print("=== 仪表板演示 ===")
    
    viz_tools = VisualizationTools()
    
    # 创建示例数据
    financial_data = create_sample_financial_data(30)
    stat_data = create_sample_statistical_data(200)
    
    # 配置图表
    charts_config = [
        {
            'id': 'kline',
            'type': 'candlestick',
            'data': financial_data,
            'position': (0, 0, 2, 2)
        },
        {
            'id': 'volume',
            'type': 'volume',
            'data': financial_data,
            'position': (2, 0, 1, 2)
        },
        {
            'id': 'scatter',
            'type': 'scatter',
            'data': stat_data[['x', 'y']],
            'position': (0, 2, 2, 2)
        },
        {
            'id': 'correlation',
            'type': 'heatmap',
            'data': stat_data[['x', 'y', 'z']].corr(),
            'position': (2, 2, 1, 2)
        }
    ]
    
    # 创建仪表板
    print("创建仪表板...")
    dashboard_fig = viz_tools.create_dashboard(
        charts_config,
        os.path.join(BASE_DIR, 'dashboard_demo.png')
    )
    
    print("仪表板演示完成")


async def demo_async_features():
    """演示异步功能"""
    print("=== 异步功能演示 ===")
    
    viz_tools = VisualizationTools()
    
    # 创建示例数据
    financial_data = create_sample_financial_data(20)
    stat_data = create_sample_statistical_data(100)
    
    # 异步创建多个图表
    chart_requests = [
        {
            'type': 'financial',
            'data': financial_data,
            'cache_key': 'financial_1',
            'chart_type': 'candlestick',
            'kwargs': {'title': '异步K线图'}
        },
        {
            'type': 'statistical',
            'data': stat_data['x'],
            'cache_key': 'stat_1',
            'chart_type': 'histogram',
            'kwargs': {'title': '异步直方图'}
        },
        {
            'type': 'statistical',
            'data': stat_data[['x', 'y']],
            'cache_key': 'stat_2',
            'chart_type': 'scatter',
            'kwargs': {'x_col': 'x', 'y_col': 'y', 'title': '异步散点图'}
        }
    ]
    
    print("异步批量创建图表...")
    results = await viz_tools.async_tools.batch_process_charts(chart_requests)
    
    # 保存图表
    for i, (result, request) in enumerate(zip(results, chart_requests)):
        if result is not None:
            filename = os.path.join(BASE_DIR, f'async_chart_{i}.png')
            viz_tools.export_chart(result, filename)
            print(f"图表 {i} 已保存: {filename}")
    
    print("异步功能演示完成")


def demo_data_analysis():
    """演示数据分析功能"""
    print("=== 数据分析演示 ===")
    
    viz_tools = VisualizationTools()
    stat_data = create_sample_statistical_data(1000)
    
    # 获取数据统计信息
    print("计算数据统计信息...")
    stats = viz_tools.get_data_statistics(stat_data)
    
    # 打印统计信息
    for col, col_stats in stats.items():
        print(f"\n{col} 统计信息:")
        for stat_name, value in col_stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    print("数据分析演示完成")


def demo_advanced_technical_indicators():
    """演示高级技术指标"""
    print("=== 高级技术指标演示 ===")
    
    viz_tools = VisualizationTools()
    financial_data = create_sample_financial_data(50)
    
    # 创建带有高级指标的K线图
    print("创建带有高级技术指标的K线图...")
    fig = viz_tools.financial_chart.create_candlestick_chart(
        financial_data, 
        title='高级技术指标K线图'
    )
    
    # 添加更多技术指标
    viz_tools.financial_chart.add_technical_indicators(
        financial_data, 
        indicators=['sma', 'ema', 'rsi', 'macd', 'bollinger', 'kdj', 'williams', 'cci']
    )
    
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'advanced_indicators.png'))
    print("高级技术指标演示完成")


def demo_advanced_statistical_charts():
    """演示高级统计图表"""
    print("=== 高级统计图表演示 ===")
    
    viz_tools = VisualizationTools()
    stat_data = create_sample_statistical_data(500)
    
    # 创建QQ图
    print("创建QQ图...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    stats.probplot(stat_data['x'], dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normal Distribution)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'qq_plot.png'))
    plt.close(fig)
    
    # 创建概率密度图
    print("创建概率密度图...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    for col in ['x', 'y', 'z']:
        data = stat_data[col].dropna()
        ax.hist(data, bins=30, alpha=0.5, label=f'{col} Distribution', density=True)
        
        # 添加核密度估计
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, kde(x_range), linewidth=2, label=f'{col} KDE')
    
    ax.set_title('Probability Density Functions')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'density_plot.png'))
    plt.close(fig)
    
    # 创建雷达图
    print("创建雷达图...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    categories = ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Q25', 'Q75']
    stats_data = viz_tools.get_data_statistics(stat_data)
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    for col, color in zip(['x', 'y', 'z'], ['red', 'blue', 'green']):
        values = [
            stats_data[col]['mean'],
            stats_data[col]['std'],
            stats_data[col]['skewness'],
            stats_data[col]['kurtosis'],
            stats_data[col]['q25'],
            stats_data[col]['q75']
        ]
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=col, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Radar Chart - Statistical Comparison')
    ax.legend()
    plt.tight_layout()
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'radar_chart.png'))
    plt.close(fig)
    
    print("高级统计图表演示完成")


def demo_custom_chart_styling():
    """演示图表自定义样式"""
    print("=== 图表自定义样式演示 ===")
    
    viz_tools = VisualizationTools()
    stat_data = create_sample_statistical_data(200)
    
    # 创建自定义样式的散点图
    print("创建自定义样式散点图...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # 自定义颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(stat_data)))
    
    scatter = ax.scatter(
        stat_data['x'], 
        stat_data['y'], 
        c=stat_data['z'], 
        cmap='plasma',
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, ax=ax, label='Z Value')
    ax.set_title('Custom Styled Scatter Plot', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X Variable', fontsize=12)
    ax.set_ylabel('Y Variable', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'custom_styled_chart.png'))
    plt.close(fig)
    
    print("图表自定义样式演示完成")


def demo_multiple_export_formats():
    """演示多种导出格式"""
    print("=== 多种导出格式演示 ===")
    
    viz_tools = VisualizationTools()
    financial_data = create_sample_financial_data(30)
    
    # 创建K线图
    fig = viz_tools.financial_chart.create_candlestick_chart(
        financial_data, 
        title='多格式导出演示'
    )
    
    # 导出多种格式
    print("导出PNG格式...")
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'export_demo.png'), format='png')
    
    print("导出SVG格式...")
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'export_demo.svg'), format='svg')
    
    print("导出PDF格式...")
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'export_demo.pdf'), format='pdf')
    
    print("多种导出格式演示完成")


def demo_data_processing_features():
    """演示数据处理功能"""
    print("=== 数据处理功能演示 ===")
    
    viz_tools = VisualizationTools()
    stat_data = create_sample_statistical_data(1000)
    
    # 数据清洗和预处理
    print("数据预处理...")
    
    # 移除异常值
    Q1 = stat_data['x'].quantile(0.25)
    Q3 = stat_data['x'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned_data = stat_data[(stat_data['x'] >= lower_bound) & (stat_data['x'] <= upper_bound)]
    
    print(f"原始数据点数: {len(stat_data)}")
    print(f"清洗后数据点数: {len(cleaned_data)}")
    print(f"移除异常值数量: {len(stat_data) - len(cleaned_data)}")
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始数据
    ax1.hist(stat_data['x'], bins=30, alpha=0.7, color='red', label='Original Data')
    ax1.set_title('Original Data Distribution')
    ax1.set_xlabel('X Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 清洗后数据
    ax2.hist(cleaned_data['x'], bins=30, alpha=0.7, color='blue', label='Cleaned Data')
    ax2.set_title('Cleaned Data Distribution')
    ax2.set_xlabel('X Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'data_processing_demo.png'))
    plt.close(fig)
    
    print("数据处理功能演示完成")


def demo_real_time_data_simulation():
    """演示实时数据模拟"""
    print("=== 实时数据模拟演示 ===")
    
    viz_tools = VisualizationTools()
    
    # 模拟实时数据源
    def generate_realtime_data():
        """生成模拟实时数据"""
        import random
        base_price = 100
        change = random.uniform(-2, 2)
        return base_price + change
    
    # 设置仪表板
    dashboard = viz_tools.dashboard
    
    # 添加实时图表
    for i in range(3):
        data = pd.DataFrame({
            'price': [generate_realtime_data() for _ in range(20)],
            'volume': [random.randint(1000, 10000) for _ in range(20)]
        })
        dashboard.add_chart(
            f'realtime_chart_{i}',
            'line',
            data,
            (i, 0, 1, 2)
        )
    
    # 创建实时仪表板
    dashboard_fig = dashboard.render_dashboard()
    viz_tools.export_chart(dashboard_fig, os.path.join(BASE_DIR, 'realtime_dashboard.png'))
    
    print("实时数据模拟演示完成")


def demo_performance_optimization():
    """演示性能优化功能"""
    print("=== 性能优化演示 ===")
    
    viz_tools = VisualizationTools()
    
    # 创建大数据集
    print("创建大数据集...")
    large_data = pd.DataFrame({
        'x': np.random.normal(0, 1, 10000),
        'y': np.random.normal(0, 1, 10000),
        'z': np.random.normal(0, 1, 10000)
    })
    
    # 创建优化的散点图（采样显示）
    print("创建性能优化散点图...")
    sample_data = large_data.sample(n=1000)  # 采样显示
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    scatter = ax.scatter(
        sample_data['x'], 
        sample_data['y'], 
        c=sample_data['z'], 
        cmap='viridis',
        alpha=0.6,
        s=20  # 减小点的大小以提高性能
    )
    
    plt.colorbar(scatter, ax=ax, label='Z Value')
    ax.set_title('Performance Optimized Scatter Plot (Sampled Data)')
    ax.set_xlabel('X Variable')
    ax.set_ylabel('Y Variable')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_tools.export_chart(fig, os.path.join(BASE_DIR, 'performance_optimized.png'))
    plt.close(fig)
    
    print("性能优化演示完成")


def demo_batch_processing():
    """演示批量处理功能"""
    print("=== 批量处理演示 ===")
    
    viz_tools = VisualizationTools()
    
    # 创建多个数据集
    datasets = []
    for i in range(5):
        data = pd.DataFrame({
            'x': np.random.normal(i, 1, 200),
            'y': np.random.normal(i*2, 1, 200),
            'category': [f'Group_{i}'] * 200
        })
        datasets.append(data)
    
    # 批量创建图表
    charts = {}
    for i, data in enumerate(datasets):
        print(f"处理数据集 {i+1}/5...")
        
        # 创建散点图
        fig = viz_tools.statistical_chart.create_scatter_plot(
            data, 
            'x', 'y',
            title=f'Batch Chart {i+1}'
        )
        charts[f'batch_chart_{i}'] = fig
    
    # 批量导出
    print("批量导出图表...")
    viz_tools.exporter.batch_export(
        charts, 
        BASE_DIR, 
        'png'
    )
    
    print("批量处理演示完成")


def demo_error_handling():
    """演示错误处理功能"""
    print("=== 错误处理演示 ===")
    
    viz_tools = VisualizationTools()
    
    # 测试各种错误情况
    error_cases = [
        ("空数据", pd.DataFrame()),
        ("无效数据类型", "not_a_dataframe"),
        ("缺失列", pd.DataFrame({'invalid': [1, 2, 3]})),
        ("非数值数据", pd.DataFrame({'text': ['a', 'b', 'c']}))
    ]
    
    for case_name, test_data in error_cases:
        try:
            print(f"测试: {case_name}")
            viz_tools.create_statistical_chart(test_data, 'histogram')
            print(f"  {case_name}: 未检测到错误")
        except DataValidationError as e:
            print(f"  {case_name}: 正确捕获错误 - {e}")
        except Exception as e:
            print(f"  {case_name}: 其他错误 - {e}")
    
    print("错误处理演示完成")


def main():
    """主函数 - 运行所有演示"""
    print("J6可视化工具演示开始")
    print("=" * 50)
    
    try:
        # 基础图表演示
        demo_financial_charts()
        demo_statistical_charts()
        
        # 交互式图表演示
        demo_interactive_charts()
        
        # 仪表板演示
        demo_dashboard()
        
        # 数据分析演示
        demo_data_analysis()
        
        # 高级功能演示
        demo_advanced_technical_indicators()
        demo_advanced_statistical_charts()
        demo_custom_chart_styling()
        demo_multiple_export_formats()
        demo_data_processing_features()
        demo_real_time_data_simulation()
        demo_performance_optimization()
        demo_batch_processing()
        demo_error_handling()
        
        # 异步功能演示
        print("\n开始异步功能演示...")
        asyncio.run(demo_async_features())
        
        print("\n" + "=" * 50)
        print("所有演示完成！")
        print("生成的文件:")
        print("- 金融图表: candlestick_demo.png, volume_demo.png, advanced_indicators.png")
        print("- 统计图表: histogram_demo.png, scatter_demo.png, box_demo.png, heatmap_demo.png")
        print("- 高级图表: qq_plot.png, density_plot.png, radar_chart.png")
        print("- 自定义图表: custom_styled_chart.png, performance_optimized.png")
        print("- 导出格式: export_demo.png/svg/pdf")
        print("- 数据处理: data_processing_demo.png")
        print("- 实时数据: realtime_dashboard.png")
        print("- 批量处理: batch_chart_*.png")
        print("- 交互式图表: plotly_*.html, bokeh_*.html")
        print("- 仪表板: dashboard_demo.png")
        print(f"- 日志文件: {LOG_FILE}")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        print(f"演示过程中出现错误: {e}")


if __name__ == "__main__":
    main()