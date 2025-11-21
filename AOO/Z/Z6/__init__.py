"""
Z6未来功能预留系统
================

该包提供完整的功能预留解决方案，支持功能预留、占位符管理、未来规划等功能。

主要模块:
- FutureFeatureReservation: 未来功能预留系统主类
- 各种数据类: FeatureReservation, Placeholder, VersionReservation 等

作者: Z6开发团队
版本: 1.0.0
创建时间: 2025-11-06
"""

from .FutureFeatureReservation import (
    FutureFeatureReservation,
    FeatureReservation,
    Placeholder,
    VersionReservation,
    DocumentReservation,
    TestReservation,
    ConfigReservation,
    MonitoringReservation,
    FeatureStatus,
    Priority,
    create_basic_reservation_system,
    quick_create_feature
)

__version__ = "1.0.0"
__author__ = "Z6开发团队"

__all__ = [
    "FutureFeatureReservation",
    "FeatureReservation", 
    "Placeholder",
    "VersionReservation",
    "DocumentReservation",
    "TestReservation",
    "ConfigReservation",
    "MonitoringReservation",
    "FeatureStatus",
    "Priority",
    "create_basic_reservation_system",
    "quick_create_feature"
]

# 包级便利函数
def create_reservation_system(storage_path=None):
    """
    创建未来功能预留系统的便利函数
    
    Args:
        storage_path: 数据存储路径，默认为当前目录下的 z6_reservations
        
    Returns:
        FutureFeatureReservation: 预留系统实例
    """
    return create_basic_reservation_system(storage_path)


def quick_reserve_feature(name, description, version="1.0.0", priority=Priority.MEDIUM):
    """
    快速创建功能预留的便利函数
    
    Args:
        name: 功能名称
        description: 功能描述
        version: 版本号
        priority: 优先级
        
    Returns:
        str: 功能ID
    """
    return quick_create_feature(name, description, version, priority)


# 包初始化信息
def get_package_info():
    """获取包信息"""
    return {
        "name": "Z6未来功能预留系统",
        "version": __version__,
        "author": __author__,
        "description": "完整的功能预留解决方案，支持功能预留、占位符管理、未来规划等功能",
        "features": [
            "功能预留（未来功能占位符和预留）",
            "占位符管理（占位符的创建和管理）",
            "未来规划（功能路线图和规划）",
            "版本预留（版本号和API预留）",
            "文档预留（功能文档和说明预留）",
            "测试预留（测试框架和用例预留）",
            "配置预留（配置参数和选项预留）",
            "监控预留（监控指标和告警预留）"
        ]
    }


# 示例用法
def demo():
    """演示系统基本用法"""
    print("Z6未来功能预留系统演示")
    print("-" * 40)
    
    # 创建系统
    system = create_reservation_system()
    
    # 创建功能预留
    feature_id = system.create_feature_reservation(
        name="智能推荐系统",
        description="基于AI的个性化推荐功能",
        version="2.0.0",
        priority=Priority.HIGH
    )
    print(f"创建功能预留: {feature_id}")
    
    # 创建占位符
    placeholder_id = system.create_placeholder(
        name="推荐算法参数",
        placeholder_type="config",
        description="推荐算法的配置参数",
        default_value=0.8
    )
    print(f"创建占位符: {placeholder_id}")
    
    # 获取统计信息
    stats = system.get_reservation_statistics()
    print(f"系统统计: {stats}")
    
    print("演示完成！")


if __name__ == "__main__":
    demo()