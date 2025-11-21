"""
P7 A/B测试器
一个完整的A/B测试解决方案，包含实验设计、分组管理、统计分析等功能。

主要模块:
- ABTester: 核心A/B测试器类
- ExperimentConfig: 实验配置类
- UserData: 用户数据类
- ExperimentResult: 实验结果类
- ABTestManager: A/B测试管理器
"""

from typing import List

from .ABTester import (
    # 核心数据类
    ExperimentConfig,
    UserData,
    ExperimentResult,
    
    # 核心类
    ABTestManager,
    
    # 便利函数
    create_ab_tester
)

__version__ = "1.0.0"
__author__ = "P7 Team"

__all__ = [
    'ExperimentConfig',
    'UserData',
    'ExperimentResult',
    'ABTestManager',
    'create_ab_tester'
]

# 便利函数
def create_ab_tester():
    """
    创建A/B测试器实例
    
    Returns:
        ABTestManager: A/B测试管理器实例
    
    Examples:
        from P7 import create_ab_tester
        
        # 创建A/B测试器
        ab_tester = create_ab_tester()
        
        # 创建实验配置
        config = ExperimentConfig(
            name="按钮颜色测试",
            description="测试红色和蓝色按钮的点击率",
            variants={
                "control": {"color": "blue"},
                "treatment": {"color": "red"}
            },
            target_metrics=["click_rate"],
            traffic_split={"control": 0.5, "treatment": 0.5},
            start_time=datetime.now()
        )
        
        # 运行实验
        result = ab_tester.run_experiment(config)
    """
    return ABTestManager()

# 实验状态枚举
class ExperimentStatus:
    """实验状态枚举"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"

# 流量分配类型
class TrafficAllocationType:
    """流量分配类型枚举"""
    EQUAL = "equal"  # 平均分配
    WEIGHTED = "weighted"  # 加权分配
    CUSTOM = "custom"  # 自定义分配

# 统计显著性级别
class SignificanceLevel:
    """统计显著性级别"""
    P90 = 0.10
    P95 = 0.05
    P99 = 0.01

# 用户分组便利函数
def create_user_groups(num_groups: int = 2, group_names: List[str] = None):
    """
    创建用户分组
    
    Args:
        num_groups: 分组数量
        group_names: 分组名称列表
    
    Returns:
        dict: 分组配置
    
    Examples:
        # 创建2组平均分配
        groups = create_user_groups(2)
        
        # 创建3组自定义分配
        groups = create_user_groups(3, ["对照组", "实验组1", "实验组2"])
    """
    if group_names is None:
        group_names = [f"Group_{i+1}" for i in range(num_groups)]
    
    if len(group_names) != num_groups:
        raise ValueError("分组名称数量必须等于分组数量")
    
    # 平均分配流量
    traffic_split = {name: 1.0 / num_groups for name in group_names}
    
    return {
        "num_groups": num_groups,
        "group_names": group_names,
        "traffic_split": traffic_split
    }

# 实验配置模板
EXPERIMENT_TEMPLATES = {
    "button_color": {
        "name": "按钮颜色测试",
        "variants": {
            "control": {"color": "blue", "text": "提交"},
            "treatment": {"color": "red", "text": "提交"}
        },
        "target_metrics": ["click_rate", "conversion_rate"],
        "traffic_split": {"control": 0.5, "treatment": 0.5}
    },
    "pricing": {
        "name": "定价策略测试",
        "variants": {
            "control": {"price": 9.99, "currency": "USD"},
            "treatment": {"price": 12.99, "currency": "USD"}
        },
        "target_metrics": ["revenue", "conversion_rate"],
        "traffic_split": {"control": 0.5, "treatment": 0.5}
    },
    "layout": {
        "name": "页面布局测试",
        "variants": {
            "control": {"layout": "vertical"},
            "treatment": {"layout": "horizontal"}
        },
        "target_metrics": ["engagement_rate", "time_spent"],
        "traffic_split": {"control": 0.5, "treatment": 0.5}
    }
}

def get_experiment_template(template_name: str):
    """
    获取实验配置模板
    
    Args:
        template_name: 模板名称
    
    Returns:
        dict: 实验模板配置
    """
    if template_name not in EXPERIMENT_TEMPLATES:
        raise ValueError(f"未知的实验模板: {template_name}")
    
    return EXPERIMENT_TEMPLATES[template_name].copy()

# 快速开始指南
QUICK_START = """
P7 A/B测试器快速开始：

1. 创建A/B测试器：
   from P7 import create_ab_tester
   ab_tester = create_ab_tester()

2. 创建实验配置：
   config = ExperimentConfig(
       name="按钮颜色测试",
       description="测试按钮颜色对点击率的影响",
       variants={
           "control": {"color": "blue"},
           "treatment": {"color": "red"}
       },
       target_metrics=["click_rate"],
       traffic_split={"control": 0.5, "treatment": 0.5},
       start_time=datetime.now()
   )

3. 运行实验：
   result = ab_tester.run_experiment(config)

4. 分析结果：
   analysis = result.get_analysis()
   print(f"统计显著性: {analysis['significance']}")

5. 获取报告：
   report = ab_tester.generate_report(result)
   print(report)
"""

# A/B测试装饰器
def ab_test(test_name: str, variants: dict):
    """A/B测试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 模拟A/B测试逻辑
            import random
            variant = random.choice(list(variants.keys()))
            
            print(f"运行A/B测试: {test_name}")
            print(f"分配到变体: {variant}")
            
            result = func(variant=variant, *args, **kwargs)
            
            print(f"测试完成: {test_name}")
            return result
        
        return wrapper
    return decorator

print("P7 A/B测试器已加载")