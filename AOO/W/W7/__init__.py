#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W7带宽管理器包
提供全面的网络带宽控制、监控和管理功能
"""

from .BandwidthManager import (
    BandwidthManager,
    BandwidthConfig,
    NetworkStats,
    QoSPolicy,
    create_default_manager,
    create_custom_manager
)

__version__ = "1.0.0"
__author__ = "W7 Team"
__description__ = "W7带宽管理器 - 提供全面的网络带宽控制、监控和管理功能"

__all__ = [
    'BandwidthManager',
    'BandwidthConfig', 
    'NetworkStats',
    'QoSPolicy',
    'create_default_manager',
    'create_custom_manager'
]

# 包级别的便利函数
def create_manager(config=None):
    """
    创建带宽管理器实例的便利函数
    
    Args:
        config: 可选的配置对象，如果不提供则使用默认配置
        
    Returns:
        BandwidthManager: 带宽管理器实例
    """
    if config is None:
        return create_default_manager()
    return BandwidthManager(config)


# 版本信息
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'features': [
        '带宽控制（上传、下载带宽限制）',
        '流量管理（流量统计、流量限制）',
        '网络优化（网络性能优化）',
        'QoS管理（服务质量管理）',
        '带宽监控（实时带宽监控）',
        '带宽统计（带宽使用统计）',
        '带宽报告（带宽使用报告）',
        '带宽配置（带宽策略配置）'
    ]
}


def get_version_info():
    """获取版本信息"""
    return VERSION_INFO


def print_version_info():
    """打印版本信息"""
    info = VERSION_INFO
    print(f"W7带宽管理器 v{info['version']}")
    print(f"作者: {info['author']}")
    print(f"描述: {info['description']}")
    print("\n主要功能:")
    for feature in info['features']:
        print(f"  - {feature}")


if __name__ == "__main__":
    print_version_info()