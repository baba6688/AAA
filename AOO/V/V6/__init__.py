"""
V6模块 - 模型监控器模块

实现模型性能监控、数据漂移检测、模型衰减监控、预测质量监控等功能。

主要功能：
1. 性能监控 - 实时监控模型性能指标
2. 数据漂移检测 - 检测输入数据分布变化
3. 模型衰减监控 - 监控模型性能衰减
4. 预测质量监控 - 监控预测结果质量
5. 异常检测 - 检测异常数据和预测
6. 告警系统 - 自动告警和通知
7. 监控仪表板 - 可视化监控界面
8. 历史数据分析 - 分析监控历史数据
"""

from .ModelMonitor import (
    # 核心类和枚举
    MonitoringConfig,
    MonitoringResult,
    ModelMonitor,
    
    # 便利函数
    create_model_monitor
)

__all__ = [
    'MonitoringConfig',
    'MonitoringResult',
    'ModelMonitor',
    'create_model_monitor'
]

__version__ = '1.0.0'

# 便利函数
def create_model_monitor(config: MonitoringConfig = None, **kwargs):
    """
    创建模型监控器实例
    
    Args:
        config: 监控配置对象
        **kwargs: 监控器参数
    
    Returns:
        ModelMonitor: 模型监控器实例
    
    Examples:
        from V6 import create_model_monitor, MonitoringConfig
        
        # 使用默认配置创建监控器
        monitor = create_model_monitor()
        
        # 使用自定义配置创建监控器
        config = MonitoringConfig(
            performance_threshold=0.8,
            drift_threshold=0.05,
            alert_enabled=True
        )
        monitor = create_model_monitor(config=config)
    """
    return ModelMonitor(config=config, **kwargs)

# 监控类型
class MonitoringTypes:
    """监控类型常量"""
    PERFORMANCE = "performance"
    DATA_DRIFT = "data_drift"
    MODEL_DEGRADATION = "model_degradation"
    PREDICTION_QUALITY = "prediction_quality"
    ANOMALY_DETECTION = "anomaly_detection"
    RESOURCE_USAGE = "resource_usage"

# 告警级别
class AlertLevels:
    """告警级别常量"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# 监控频率
class MonitoringFrequency:
    """监控频率常量"""
    REAL_TIME = "real_time"  # 实时监控
    HOURLY = "hourly"  # 每小时
    DAILY = "daily"  # 每天
    WEEKLY = "weekly"  # 每周

# 快速开始指南
QUICK_START = """
V6模型监控器快速开始：

1. 创建监控器：
   from V6 import create_model_monitor
   monitor = create_model_monitor()

2. 配置监控：
   # 设置监控配置
   config = MonitoringConfig(
       performance_threshold=0.8,
       drift_threshold=0.05,
       alert_email="admin@example.com"
   )
   monitor.set_config(config)

3. 开始监控：
   # 开始实时监控
   monitor.start_monitoring(model_id="my_model")

4. 监控数据：
   # 监控预测数据
   X_new, y_true = get_new_data()
   monitor.log_prediction(model_id="my_model", X=X_new, y_true=y_true)

5. 获取监控状态：
   status = monitor.get_monitoring_status(model_id="my_model")
   print(f"模型状态: {status.overall_health}")

6. 检测数据漂移：
   drift_result = monitor.detect_data_drift(model_id="my_model")
   print(f"漂移检测: {drift_result.has_drift}")

7. 生成监控报告：
   report = monitor.generate_monitoring_report(model_id="my_model")
"""

# 监控装饰器
def monitor_model_predictions(func):
    """模型预测监控装饰器"""
    def wrapper(*args, **kwargs):
        import time
        import numpy as np
        
        # 尝试获取模型ID（从kwargs或闭包中）
        model_id = kwargs.get('model_id', 'default_model')
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # 记录预测时间
        prediction_time = end_time - start_time
        
        # 如果有预测结果，记录到监控器
        try:
            if hasattr(wrapper, '_monitor'):
                monitor = wrapper._monitor
                monitor.log_prediction_time(model_id, prediction_time)
            else:
                print(f"预测完成: {model_id}, 耗时: {prediction_time:.3f}秒")
        except:
            print(f"预测完成: {model_id}, 耗时: {prediction_time:.3f}秒")
        
        return result
    
    return wrapper

# 数据质量检查装饰器
def validate_input_data(func):
    """输入数据验证装饰器"""
    def wrapper(*args, **kwargs):
        import numpy as np
        
        # 检查输入参数中的数据
        for arg in args:
            if isinstance(arg, np.ndarray):
                if arg.size == 0:
                    raise ValueError("输入数据不能为空")
                if np.any(np.isnan(arg)):
                    print(f"警告: 输入数据包含 {np.isnan(arg).sum()} 个NaN值")
                if np.any(np.isinf(arg)):
                    print(f"警告: 输入数据包含 {np.isinf(arg).sum()} 个无穷值")
        
        return func(*args, **kwargs)
    
    return wrapper

# 设置监控器的类装饰器
def with_monitoring(monitor: ModelMonitor):
    """添加监控功能的类装饰器"""
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # 为实例添加监控方法
            self._monitor = monitor
        
        cls.__init__ = new_init
        return cls
    
    return decorator

print("V6模型监控器已加载")