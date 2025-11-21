"""
M5数据监控器模块

该模块提供全面的数据监控功能，包括：
- 数据质量监控（完整性、准确性、一致性、及时性、有效性、唯一性）
- 数据完整性监控
- 数据一致性监控
- 数据及时性监控
- 数据量监控
- 数据安全监控
- 数据访问监控
- 数据异常检测
- 监控报告生成

主要类:
    DataMonitor: 主要的数据监控器类
    AlertLevel: 告警级别枚举
    MonitorResult: 监控结果数据类
    DataQualityMetrics: 数据质量指标数据类
    AnomalyDetectionResult: 异常检测结果数据类

使用示例:
    from D.AO.AOO.M.M5 import DataMonitor
    
    # 创建监控器
    monitor = DataMonitor()
    
    # 数据质量监控
    result = monitor.monitor_data_quality(data, "data_id")
    print(f"质量评分: {result.score}")
    
    # 异常检测
    anomaly = monitor.detect_data_anomalies(data, "data_id")
    if anomaly.is_anomaly:
        print(f"检测到异常: {anomaly.anomaly_type}")
    
    # 生成监控报告
    report = monitor.generate_monitoring_report()

版本: 1.0.0

创建时间: 2025-11-05
"""

from .DataMonitor import (
    DataMonitor,
    AlertLevel,
    DataQualityScore,
    MonitorResult,
    DataQualityMetrics,
    AnomalyDetectionResult
)

__version__ = "1.0.0"
__author__ = "AI系统"
__email__ = "ai@example.com"
__license__ = "MIT"

__all__ = [
    'DataMonitor',
    'AlertLevel',
    'DataQualityScore',
    'MonitorResult',
    'DataQualityMetrics',
    'AnomalyDetectionResult'
]

# 模块级别的便捷函数
def create_monitor(config=None):
    """
    创建数据监控器实例的便捷函数
    
    Args:
        config (dict, optional): 配置字典
        
    Returns:
        DataMonitor: 监控器实例
    """
    return DataMonitor(config)


def quick_monitor(data, data_id="quick_monitor"):
    """
    快速数据监控的便捷函数
    
    Args:
        data: 待监控的数据
        data_id (str): 数据标识符
        
    Returns:
        dict: 监控结果汇总
    """
    monitor = DataMonitor()
    
    try:
        # 执行多种监控
        quality_result = monitor.monitor_data_quality(data, data_id)
        volume_result = monitor.monitor_data_volume(data, data_id)
        anomaly_result = monitor.detect_data_anomalies(data, data_id)
        
        return {
            'quality': {
                'score': quality_result.score,
                'status': quality_result.status
            },
            'volume': {
                'score': volume_result.score,
                'status': volume_result.status
            },
            'anomaly': {
                'is_anomaly': anomaly_result.is_anomaly,
                'anomaly_score': anomaly_result.anomaly_score,
                'anomaly_type': anomaly_result.anomaly_type
            }
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'monitor_type': 'quick_monitor_failed'
        }


# 模块文档
__doc__ = """
M5数据监控器 - 全面的数据质量与安全监控解决方案

该模块提供企业级的数据监控功能，支持实时数据质量检查、
智能异常检测、安全监控和详细报告生成。适用于数据仓库、
数据湖、大数据平台等各种场景的监控需求。

主要功能:
    ✓ 实时数据质量监控
    ✓ 多维度数据分析
    ✓ 智能异常检测
    ✓ 数据安全监控
    ✓ 访问模式分析
    ✓ 详细报告导出

快速开始:
    >>> from D.AO.AOO.M.M5 import create_monitor
    >>> monitor = create_monitor()
    >>> result = monitor.monitor_data_quality(your_data, "data_id")
    >>> print(f"数据质量评分: {result.score}")

详细文档请参考 README.md 文件。
"""