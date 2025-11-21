"""
W5防火墙管理器包

这个包提供了完整的防火墙管理功能，包括：
- 防火墙规则管理（FirewallRule类）
- 访问控制
- 端口管理
- 协议控制
- 安全防护
- 防火墙日志（SecurityEvent类）
- 防火墙统计
- 防火墙配置管理

主要组件：
- FirewallRule: 防火墙规则数据类
- SecurityEvent: 安全事件数据类
- FirewallManager: W5防火墙管理器主类

作者: W5防火墙开发团队
版本: 1.0.0
"""

# 导入核心类
from .FirewallManager import FirewallRule, SecurityEvent, FirewallManager

# 版本信息
__version__ = "1.0.0"
__author__ = "W5防火墙开发团队"

# 导出的公共接口
__all__ = [
    "FirewallRule",      # 防火墙规则数据类
    "SecurityEvent",     # 安全事件数据类
    "FirewallManager"    # W5防火墙管理器主类
]