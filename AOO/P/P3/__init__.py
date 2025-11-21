"""
P3性能测试器包
提供全面的性能测试功能，包括负载测试、压力测试、基准测试等
"""

from .PerformanceTester import (
    # 核心类和枚举
    TestResult,
    PerformanceMetrics,
    PerformanceTester,
    
    # 便利函数
    create_performance_tester
)

__version__ = "1.0.0"
__author__ = "P3 Performance Testing Team"

__all__ = [
    'TestResult',
    'PerformanceMetrics',
    'PerformanceTester',
    'create_performance_tester'
]

# 便利函数
def create_performance_tester(base_url: str = "http://localhost:8000", db_path: str = "performance_data.db"):
    """
    创建性能测试器实例
    
    Args:
        base_url: 被测试的API基础URL
        db_path: SQLite数据库文件路径
    
    Returns:
        PerformanceTester: 性能测试器实例
    
    Examples:
        from P3 import create_performance_tester
        
        # 创建基本性能测试器
        tester = create_performance_tester()
        
        # 创建带自定义配置的测试器
        tester = create_performance_tester(
            base_url="https://api.example.com",
            db_path="my_performance.db"
        )
    """
    return PerformanceTester(base_url=base_url, db_path=db_path)

# 使用示例
QUICK_START = """
P3性能测试器快速开始：

1. 创建性能测试器：
   from P3 import create_performance_tester
   tester = create_performance_tester('https://api.example.com')

2. 运行性能测试：
   # 运行100个并发请求，每个持续10秒
   result = tester.run_load_test(
       endpoint='/api/test',
       duration=10,
       concurrent_users=100
   )
   
3. 生成性能报告：
   report = tester.generate_report()
   print(report)
"""

print("P3性能测试器已加载")