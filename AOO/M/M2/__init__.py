"""
M2性能监控器模块

该模块提供全面的系统性能监控功能，包括：
- 应用性能监控
- 数据库性能监控
- 缓存性能监控
- API响应时间监控
- 吞吐量监控
- 错误率监控
- 性能基线管理
- 性能优化建议
- 性能报告生成

主要类:
    PerformanceMonitor: 主要的性能监控器类
    PerformanceMetrics: 性能指标数据类
    PerformanceBaseline: 性能基线数据类
    PerformanceAlert: 性能告警数据类
    PerformanceMonitorTest: 测试类

使用示例:
    from D.AO.AOO.M.M2 import PerformanceMonitor
    
    # 创建监控器
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # 记录指标
    monitor._record_metric('test', 'response_time', 1.5, 's', {})
    
    # 生成报告
    report = monitor.generate_performance_report()
    
    monitor.stop_monitoring()

版本: 1.0.0

创建时间: 2025-11-05
"""

from .PerformanceMonitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    PerformanceBaseline,
    PerformanceAlert,
    PerformanceMonitorTest
)

__version__ = "1.0.0"
__author__ = "AI系统"
__email__ = "ai@example.com"
__license__ = "MIT"

__all__ = [
    'PerformanceMonitor',
    'PerformanceMetrics', 
    'PerformanceBaseline',
    'PerformanceAlert',
    'PerformanceMonitorTest'
]

# 模块级别的便捷函数
def create_monitor(config=None):
    """
    创建性能监控器实例的便捷函数
    
    Args:
        config (dict, optional): 配置字典
        
    Returns:
        PerformanceMonitor: 监控器实例
    """
    return PerformanceMonitor(config)


def quick_monitor(duration=60):
    """
    快速启动性能监控的便捷函数
    
    Args:
        duration (int): 监控持续时间（秒）
        
    Returns:
        dict: 性能报告
    """
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        import time
        time.sleep(duration)
        
        report = monitor.generate_performance_report(hours=duration/3600)
        return report
        
    finally:
        monitor.stop_monitoring()


# 模块文档
__doc__ = """
M2性能监控器 - 全面的系统性能监控解决方案

该模块提供企业级的性能监控功能，支持实时指标收集、智能分析、
自动告警和详细报告生成。适用于Web应用、数据库系统、缓存系统
等各种场景的性能监控需求。

主要功能:
    ✓ 实时性能指标收集
    ✓ 多维度性能分析
    ✓ 智能告警系统
    ✓ 性能基线管理
    ✓ 优化建议生成
    ✓ 详细报告导出

快速开始:
    >>> from D.AO.AOO.M.M2 import create_monitor
    >>> monitor = create_monitor()
    >>> monitor.start_monitoring()
    >>> # ... 运行你的应用 ...
    >>> report = monitor.generate_performance_report()
    >>> monitor.stop_monitoring()

详细文档请参考 README.md 文件。
"""