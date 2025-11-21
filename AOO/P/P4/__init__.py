"""
P4压力测试器包
提供高性能压力测试框架，包括极限负载测试、峰值负载测试、长时间稳定性测试等
"""

from .StressTester import (
    # 核心类和枚举
    TestResult as StressTestResult,
    SystemMetrics,
    LoadGenerator,
    StressTester,
    StressTestManager,
    
    # 便利函数
    create_stress_tester
)

__version__ = "1.0.0"
__author__ = "P4压力测试团队"

__all__ = [
    'StressTestResult',
    'SystemMetrics',
    'LoadGenerator', 
    'StressTester',
    'StressTestManager',
    'create_stress_tester'
]

# 便利函数
def create_stress_tester():
    """
    创建压力测试器实例
    
    Returns:
        StressTester: 压力测试器实例
    
    Examples:
        from P4 import create_stress_tester
        
        # 创建压力测试器
        tester = create_stress_tester()
        
        # 运行极限负载测试
        result = tester.run_extreme_load_test(
            endpoint='https://api.example.com/stress',
            max_users=10000,
            duration=300
        )
    """
    return StressTester()

# 测试结果枚举别名
class StressTestStatus:
    """压力测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

# 系统资源类型
class ResourceType:
    """系统资源类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    THREADS = "threads"
    PROCESSES = "processes"

# 使用示例
QUICK_START = """
P4压力测试器快速开始：

1. 创建压力测试器：
   from P4 import create_stress_tester
   tester = create_stress_tester()

2. 运行压力测试：
   # 运行极限负载测试
   result = tester.run_extreme_load_test(
       endpoint='https://api.example.com',
       max_users=10000,
       duration=300
   )
   
   # 运行长时间稳定性测试
   result = tester.run_long_term_stability_test(
       endpoint='https://api.example.com',
       users=1000,
       duration=3600  # 1小时
   )

3. 查看测试结果：
   print(f"成功请求: {result.success_count}")
   print(f"失败请求: {result.failure_count}")
   print(f"平均响应时间: {result.avg_response_time:.3f}秒")
"""

# 系统监控装饰器
def monitor_resource_usage(func):
    """系统资源监控装饰器"""
    def wrapper(*args, **kwargs):
        import psutil
        import time
        
        # 获取初始系统状态
        initial_cpu = psutil.cpu_percent()
        initial_memory = psutil.virtual_memory().percent
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # 获取最终系统状态
        final_cpu = psutil.cpu_percent()
        final_memory = psutil.virtual_memory().percent
        
        print(f"函数 {func.__name__} 执行统计:")
        print(f"  执行时间: {end_time - start_time:.3f}秒")
        print(f"  CPU使用率: {initial_cpu}% → {final_cpu}%")
        print(f"  内存使用率: {initial_memory}% → {final_memory}%")
        
        return result
    
    return wrapper

print("P4压力测试器已加载")