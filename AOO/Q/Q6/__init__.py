"""
Q6报告生成器包

一个功能完整的报告生成系统，支持多种数据源、多种输出格式和丰富的可视化功能。
"""

try:
    from .ReportGenerator import ReportGenerator
    from .ReportGenerator import DataProcessor
    from .ReportGenerator import TemplateManager as ReportTemplateManager
    from .ReportGenerator import ChartGenerator
    from .ReportGenerator import ReportScheduler
    
    __all__ = [
        "ReportGenerator",
        "DataProcessor", 
        "TemplateManager",
        "ChartGenerator",
        "ReportScheduler"
    ]
except ImportError as e:
    # 提供基本的导出以避免完全失败
    print(f"警告: 无法导入Q6完整功能: {e}")
    __all__ = []
    
    # 提供占位符类
    class ReportGenerator:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Q6模块未完全加载")

__version__ = "1.0.0"
__author__ = "Q6开发团队"