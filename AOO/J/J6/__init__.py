#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J6可视化工具模块

这个模块提供了一个完整的金融和统计数据可视化工具集，
支持多种图表类型、交互式功能、实时数据展示和异步处理。

主要功能：
1. 金融数据可视化（K线图、成交量图、技术指标图）
2. 统计图表工具（直方图、散点图、箱线图、热力图）
3. 交互式图表工具（Plotly、Bokeh集成）
4. 仪表板工具（实时数据展示、多图表布局）
5. 图表导出工具（PNG、SVG、PDF格式）
6. 异步图表生成和缓存

作者: J6开发团队
版本: 1.0.0
日期: 2025-11-13
"""

# 导入所有主要的类和函数
from .VisualizationTools import (
    # 异常类
    VisualizationError,
    DataValidationError,
    ChartExportError,
    CacheError,
    
    # 核心图表类
    BaseChart,
    FinancialChart,
    StatisticalChart,
    InteractiveChart,
    
    # 仪表板和导出
    Dashboard,
    ChartExporter,
    
    # 异步工具
    AsyncVisualizationTools,
    
    # 主工具类
    VisualizationTools,
    
    # 工具函数
    validate_data,
    handle_exceptions,
    ChartCache,
    
    # 示例数据生成函数
    create_sample_financial_data,
    create_sample_statistical_data,
    
    # 演示函数
    demo_financial_charts,
    demo_statistical_charts,
    demo_interactive_charts,
    demo_dashboard,
    demo_async_features,
    demo_data_analysis,
    demo_advanced_technical_indicators,
    demo_advanced_statistical_charts,
    demo_custom_chart_styling,
    demo_multiple_export_formats,
    demo_data_processing_features,
    demo_real_time_data_simulation,
    demo_performance_optimization,
    demo_batch_processing,
    demo_error_handling
)

# 定义模块版本
__version__ = "1.0.0"
__author__ = "J6开发团队"

# 定义模块导出的公共接口
__all__ = [
    # 异常类
    'VisualizationError',
    'DataValidationError', 
    'ChartExportError',
    'CacheError',
    
    # 核心图表类
    'BaseChart',
    'FinancialChart',
    'StatisticalChart',
    'InteractiveChart',
    
    # 仪表板和导出
    'Dashboard',
    'ChartExporter',
    
    # 异步工具
    'AsyncVisualizationTools',
    
    # 主工具类
    'VisualizationTools',
    
    # 工具函数
    'validate_data',
    'handle_exceptions',
    'ChartCache',
    
    # 示例数据生成函数
    'create_sample_financial_data',
    'create_sample_statistical_data',
    
    # 演示函数
    'demo_financial_charts',
    'demo_statistical_charts',
    'demo_interactive_charts',
    'demo_dashboard',
    'demo_async_features',
    'demo_data_analysis',
    'demo_advanced_technical_indicators',
    'demo_advanced_statistical_charts',
    'demo_custom_chart_styling',
    'demo_multiple_export_formats',
    'demo_data_processing_features',
    'demo_real_time_data_simulation',
    'demo_performance_optimization',
    'demo_batch_processing',
    'demo_error_handling',
    
    # 模块信息
    '__version__',
    '__author__'
]

# 模块级配置
DEFAULT_CHART_WIDTH = 12
DEFAULT_CHART_HEIGHT = 8
DEFAULT_DPI = 100

# 支持的导出格式
SUPPORTED_FORMATS = ['png', 'jpg', 'svg', 'pdf', 'eps', 'html']

# 检查可选依赖的可用性
try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import bokeh
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

# 辅助函数
def get_version():
    """获取模块版本"""
    return __version__

def check_dependencies():
    """检查可选依赖的可用性"""
    deps = {
        'plotly': PLOTLY_AVAILABLE,
        'bokeh': BOKEH_AVAILABLE,
        'kaleido': KALEIDO_AVAILABLE
    }
    return deps

def is_interactive_available():
    """检查交互式功能是否可用"""
    return PLOTLY_AVAILABLE or BOKEH_AVAILABLE

def get_supported_export_formats():
    """获取支持的导出格式列表"""
    return SUPPORTED_FORMATS.copy()

# 模块初始化检查
def _check_module_health():
    """检查模块健康状态"""
    issues = []
    
    # 检查必要的依赖
    try:
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import scipy
    except ImportError as e:
        issues.append(f"缺少必要依赖: {e}")
    
    # 检查可选依赖状态
    missing_optional = []
    if not PLOTLY_AVAILABLE:
        missing_optional.append("plotly")
    if not BOKEH_AVAILABLE:
        missing_optional.append("bokeh")
    if not KALEIDO_AVAILABLE:
        missing_optional.append("kaleido")
    
    if missing_optional:
        issues.append(f"可选依赖未安装: {', '.join(missing_optional)}")
    
    return issues

# 在模块导入时进行健康检查
_module_issues = _check_module_health()
if _module_issues:
    import warnings
    for issue in _module_issues:
        warnings.warn(f"J6模块警告: {issue}", UserWarning)