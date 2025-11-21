"""
V9模块 - 模型状态聚合器模块

该模块实现了一个全面的模型状态聚合器，用于收集、监控、分析和报告模型状态信息。
支持多种模型类型的状态收集、性能指标聚合、健康度评估、资源监控等功能。

主要功能:
- 模型状态收集和聚合
- 性能指标监控和分析
- 模型使用统计
- 健康度评估
- 资源消耗监控
- 版本状态管理
- 部署状态监控
- 状态报告生成
- 告警系统
"""

from .ModelStateAggregator import (
    # 核心枚举和类
    ModelStatus,
    AlertLevel,
    HealthStatus,
    ModelInfo,
    ModelStatusAggregator,
    
    # 便利函数
    create_status_aggregator
)

__all__ = [
    'ModelStatus',
    'AlertLevel',
    'HealthStatus',
    'ModelInfo',
    'ModelStatusAggregator',
    'create_status_aggregator'
]

__version__ = '9.0.0'

# 便利函数
def create_status_aggregator(**kwargs):
    """
    创建模型状态聚合器实例
    
    Args:
        **kwargs: 聚合器参数
    
    Returns:
        ModelStatusAggregator: 模型状态聚合器实例
    
    Examples:
        from V9 import create_status_aggregator
        
        # 创建基本聚合器
        aggregator = create_status_aggregator()
        
        # 创建带数据库配置的聚合器
        aggregator = create_status_aggregator(
            db_path="model_status.db",
            update_interval=60
        )
    """
    return ModelStatusAggregator(**kwargs)

# 聚合模式
class AggregationModes:
    """聚合模式常量"""
    REAL_TIME = "real_time"  # 实时聚合
    BATCH = "batch"  # 批量聚合
    SCHEDULED = "scheduled"  # 定时聚合
    ON_DEMAND = "on_demand"  # 按需聚合

# 报告格式
class ReportFormats:
    """报告格式常量"""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    DASHBOARD = "dashboard"

# 监控级别
class MonitoringLevels:
    """监控级别常量"""
    BASIC = "basic"  # 基本监控
    STANDARD = "standard"  # 标准监控
    ADVANCED = "advanced"  # 高级监控
    ENTERPRISE = "enterprise"  # 企业级监控

# 快速开始指南
QUICK_START = """
V9模型状态聚合器快速开始：

1. 创建聚合器：
   from V9 import create_status_aggregator
   aggregator = create_status_aggregator()

2. 注册模型：
   # 注册要监控的模型
   model_info = ModelInfo(
       model_id="my_model_1",
       model_name="MyClassificationModel",
       model_type="classification",
       version="1.0.0"
   )
   aggregator.register_model(model_info)

3. 收集状态：
   # 手动收集状态
   status = aggregator.collect_model_status("my_model_1")
   
   # 启动自动收集
   aggregator.start_auto_collection()

4. 获取聚合状态：
   # 获取所有模型状态
   all_status = aggregator.get_all_model_status()
   
   # 获取特定模型状态
   model_status = aggregator.get_model_status("my_model_1")

5. 健康度评估：
   health = aggregator.assess_model_health("my_model_1")
   print(f"模型健康状态: {health.status}")

6. 生成综合报告：
   report = aggregator.generate_comprehensive_report(
       format="html",
       include_charts=True
   )
   aggregator.save_report(report, "model_status_report.html")

7. 设置告警：
   aggregator.set_alert_config(
       model_id="my_model_1",
       alert_level=AlertLevel.WARNING,
       conditions={
           "accuracy_threshold": 0.8,
           "response_time_threshold": 1.0
       }
   )

8. 获取使用统计：
   stats = aggregator.get_model_usage_stats("my_model_1")
   print(f"模型使用次数: {stats.total_requests}")
"""

# 状态聚合装饰器
def aggregate_model_status(func):
    """模型状态聚合装饰器"""
    def wrapper(*args, **kwargs):
        import time
        import numpy as np
        
        # 获取模型ID
        model_id = kwargs.get('model_id', 'default_model')
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 记录执行统计
        try:
            if hasattr(wrapper, '_aggregator'):
                aggregator = wrapper._aggregator
                aggregator.record_execution_time(model_id, execution_time)
                
                # 记录成功/失败
                success = not isinstance(result, Exception)
                aggregator.record_execution_result(model_id, success)
            else:
                print(f"模型 {model_id} 执行统计:")
                print(f"  执行时间: {execution_time:.3f}秒")
                print(f"  执行状态: {'成功' if success else '失败'}")
        except:
            print(f"模型 {model_id} 执行完成，耗时: {execution_time:.3f}秒")
        
        return result
    
    return wrapper

# 健康检查装饰器
def health_check(func):
    """健康检查装饰器"""
    def wrapper(*args, **kwargs):
        import warnings
        
        # 获取模型ID
        model_id = kwargs.get('model_id', 'default_model')
        
        try:
            result = func(*args, **kwargs)
            
            # 简单的健康检查
            if hasattr(result, '__len__'):
                if len(result) == 0:
                    warnings.warn(f"模型 {model_id} 返回空结果")
                elif hasattr(result, 'shape'):
                    if result.shape[0] == 0:
                        warnings.warn(f"模型 {model_id} 返回零行数据")
            
            return result
            
        except Exception as e:
            print(f"模型 {model_id} 执行失败: {e}")
            # 可以在这里触发告警
            raise
    
    return wrapper

# 批量状态收集装饰器
def batch_aggregate_status(batch_size: int = 50):
    """批量状态收集装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取模型列表
            models = kwargs.get('model_ids', [])
            
            if not models or len(models) <= batch_size:
                return func(*args, **kwargs)
            
            # 分批处理
            results = {}
            for i in range(0, len(models), batch_size):
                batch = models[i:i+batch_size]
                batch_kwargs = {**kwargs, 'model_ids': batch}
                batch_results = func(*args, **batch_kwargs)
                results.update(batch_results)
                print(f"处理批次 {i//batch_size + 1}: {len(batch)} 个模型")
            
            return results
        
        return wrapper
    return decorator

# 告警检查装饰器
def check_alerts(threshold_violations: dict):
    """告警检查装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # 检查是否触发告警
            for metric, threshold in threshold_violations.items():
                if hasattr(result, metric):
                    value = getattr(result, metric)
                    if value < threshold:  # 假设越低越差
                        print(f"告警: {metric} = {value} 低于阈值 {threshold}")
                        # 这里可以触发实际的告警机制
            
            return result
        return wrapper
    return decorator

# 设置聚合器的类装饰器
def with_status_aggregator(aggregator: ModelStatusAggregator):
    """添加状态聚合功能的类装饰器"""
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # 为实例添加聚合方法
            self._aggregator = aggregator
        
        cls.__init__ = new_init
        return cls
    
    return decorator

print("V9模型状态聚合器已加载")