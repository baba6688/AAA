"""
W6网络安全器包
提供网络安全检测、威胁防护、数据加密等功能

主要组件:
- SecurityEvent: 安全事件数据类
- SecurityReport: 安全报告数据类
- NetworkSecurity: W6网络安全器主类

功能特性:
- 端口扫描和漏洞检测
- 恶意软件检测和入侵防护
- 数据加密和安全通信
- 访问控制和安全审计
- 实时安全监控和报告生成
"""

from NetworkSecurity import (
    SecurityEvent,
    SecurityReport, 
    NetworkSecurity
)

__version__ = "1.0.0"
__author__ = "W6网络安全团队"
__email__ = "security@w6.network"
__license__ = "MIT"

# 包级别的文档
__doc__ = """
W6网络安全器是一个综合性的网络安全解决方案，提供：
1. 网络安全检测 - 端口扫描、漏洞扫描、网络发现
2. 威胁防护 - 恶意软件检测、入侵检测、访问控制
3. 数据加密 - 文件加密、安全通信、数据保护
4. 安全审计 - 审计日志、安全监控、事件记录
5. 报告生成 - 安全报告、风险评估、建议生成

使用示例:
    from W.W6 import NetworkSecurity, SecurityEvent, SecurityReport
    
    # 初始化网络安全器
    security = NetworkSecurity()
    
    # 执行端口扫描
    scan_result = security.port_scan("192.168.1.1")
    
    # 生成安全报告
    report = security.generate_security_report()
    
    # 导出报告
    security.export_security_report(report, "security_report.html", "html")
"""

__all__ = [
    "SecurityEvent",
    "SecurityReport",
    "NetworkSecurity"
]