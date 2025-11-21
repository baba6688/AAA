"""
P9测试状态聚合器包
==================

这是一个功能完整的测试状态聚合器，用于收集、分析和报告测试状态信息。

主要功能：
- 状态收集器：从各个测试模块收集状态信息
- 数据聚合：聚合多个测试模块的结果
- 状态分析：分析测试状态和趋势
- 报告生成：生成综合测试状态报告
- 状态监控：实时监控测试状态
- 预警机制：测试失败或异常时预警
- 历史记录：保存历史测试状态
- 仪表板：提供可视化的测试状态仪表板

版本：1.0.0
作者：P9开发团队
"""

from .TestStatusAggregator import (
    # 核心枚举
    TestStatus as AggregatedTestStatus,
    AlertLevel,
    
    # 核心数据类
    TestResult as AggregatedTestResult,
    ModuleStatus,
    Alert,
    SystemHealth,
    
    # 核心类
    TestStatusAggregator,
    
    # 便利函数
    create_status_aggregator
)

__version__ = "1.0.0"
__author__ = "P9开发团队"

__all__ = [
    'AggregatedTestStatus',
    'AlertLevel',
    'AggregatedTestResult',
    'ModuleStatus',
    'Alert',
    'SystemHealth',
    'TestStatusAggregator',
    'create_status_aggregator'
]

# 便利函数
def create_status_aggregator():
    """
    创建测试状态聚合器实例
    
    Returns:
        TestStatusAggregator: 测试状态聚合器实例
    
    Examples:
        from P9 import create_status_aggregator
        
        # 创建状态聚合器
        aggregator = create_status_aggregator()
        
        # 收集测试状态
        status = aggregator.collect_test_status()
        
        # 生成综合报告
        report = aggregator.generate_comprehensive_report()
        print(report)
    """
    return TestStatusAggregator()

# 测试状态别名
TestStatusTypes = AggregatedTestStatus
AlertLevels = AlertLevel

# 聚合器模式
class AggregationMode:
    """聚合模式枚举"""
    REAL_TIME = "real_time"  # 实时聚合
    BATCH = "batch"  # 批量聚合
    SCHEDULED = "scheduled"  # 定时聚合

# 报告格式
class ReportFormat:
    """报告格式枚举"""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    PDF = "pdf"

# 监控级别
class MonitorLevel:
    """监控级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# 状态统计
class StatusStats:
    """状态统计工具类"""
    
    @staticmethod
    def calculate_success_rate(results: List[AggregatedTestResult]) -> float:
        """计算成功率"""
        if not results:
            return 0.0
        
        passed = sum(1 for r in results if r.status == AggregatedTestStatus.PASSED)
        return (passed / len(results)) * 100
    
    @staticmethod
    def get_status_distribution(results: List[AggregatedTestResult]) -> dict:
        """获取状态分布"""
        distribution = {}
        for result in results:
            status = result.status.value
            distribution[status] = distribution.get(status, 0) + 1
        return distribution
    
    @staticmethod
    def get_module_performance(results: List[AggregatedTestResult]) -> dict:
        """获取模块性能"""
        performance = {}
        for result in results:
            module = result.module_name
            if module not in performance:
                performance[module] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'avg_duration': 0
                }
            
            performance[module]['total'] += 1
            if result.status == AggregatedTestStatus.PASSED:
                performance[module]['passed'] += 1
            else:
                performance[module]['failed'] += 1
            
            performance[module]['avg_duration'] += result.duration
        
        # 计算平均值
        for module in performance:
            if performance[module]['total'] > 0:
                performance[module]['avg_duration'] /= performance[module]['total']
                performance[module]['success_rate'] = (
                    performance[module]['passed'] / performance[module]['total'] * 100
                )
        
        return performance

# 快速开始指南
QUICK_START = """
P9测试状态聚合器快速开始：

1. 创建状态聚合器：
   from P9 import create_status_aggregator
   aggregator = create_status_aggregator()

2. 收集测试状态：
   status = aggregator.collect_test_status()
   
3. 生成聚合报告：
   report = aggregator.generate_comprehensive_report()
   print(report)
   
4. 设置监控：
   aggregator.set_monitoring_config({
       'alert_threshold': 80,
       'check_interval': 300
   })
   
5. 启动实时监控：
   aggregator.start_real_time_monitoring()

6. 获取历史趋势：
   trends = aggregator.get_historical_trends(days=7)
"""

# 状态监控装饰器
def monitor_test_status(module_name: str):
    """测试状态监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # 记录成功的测试结果
                test_result = AggregatedTestResult(
                    test_id=f"{module_name}.{func.__name__}",
                    test_name=func.__name__,
                    module_name=module_name,
                    status=AggregatedTestStatus.PASSED,
                    duration=end_time - start_time,
                    timestamp=time.time(),
                    message=f"测试 {func.__name__} 执行成功"
                )
                
                # 这里可以保存到聚合器
                print(f"监控: {module_name}.{func.__name__} - 通过 ({end_time - start_time:.3f}s)")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                
                # 记录失败的测试结果
                test_result = AggregatedTestResult(
                    test_id=f"{module_name}.{func.__name__}",
                    test_name=func.__name__,
                    module_name=module_name,
                    status=AggregatedTestStatus.FAILED,
                    duration=end_time - start_time,
                    timestamp=time.time(),
                    message=f"测试失败: {str(e)}"
                )
                
                print(f"监控: {module_name}.{func.__name__} - 失败 ({end_time - start_time:.3f}s)")
                raise
        
        return wrapper
    return decorator

# 导入时间模块
import time
from typing import List

print("P9测试状态聚合器已加载")