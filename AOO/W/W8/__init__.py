#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W8网络诊断器包

一个功能全面的网络诊断工具，提供网络连通性检测、故障排查、性能分析等功能。

主要功能:
- 网络诊断（网络连接诊断、路径诊断）
- 故障排查（网络故障定位和分析）
- 性能分析（网络性能分析和评估）
- 网络测试（网络连通性测试）
- 路由分析（网络路由分析）
- DNS诊断（DNS解析诊断）
- 诊断报告（网络诊断报告）
- 诊断工具（网络诊断工具集合）

版本: 1.0.0
作者: W8开发团队
日期: 2024-11-06
"""

__version__ = "1.0.0"
__author__ = "W8开发团队"
__email__ = "support@w8-diagnostics.com"
__license__ = "MIT"

# 导入主要类和函数
from .NetworkDiagnostics import (
    NetworkDiagnostics,
    DiagnosticStatus,
    NetworkTestResult,
    DiagnosticReport
)

# 包级别的公共接口
__all__ = [
    'NetworkDiagnostics',
    'DiagnosticStatus', 
    'NetworkTestResult',
    'DiagnosticReport'
]

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """获取版本字符串"""
    return __version__

def get_version_info():
    """获取详细版本信息"""
    return VERSION_INFO.copy()

# 包初始化完成标识
_PACKAGE_INITIALIZED = True

# 使用示例
"""
基本使用示例:

from NetworkDiagnostics import NetworkDiagnostics, DiagnosticStatus

# 创建诊断器实例
diagnostics = NetworkDiagnostics()

# 执行Ping测试
result = diagnostics.ping_host("8.8.8.8")
print(f"状态: {result.status.value}")
print(f"消息: {result.message}")

# 执行综合诊断
report = diagnostics.comprehensive_diagnosis("github.com", [80, 443])
diagnostics.print_report(report)

# 导出报告
filename = diagnostics.export_report(report, 'json')
print(f"报告已保存到: {filename}")
"""

print(f"W8网络诊断器 v{__version__} 已加载")
print("访问 https://github.com/w8-diagnostics 获取更多信息")