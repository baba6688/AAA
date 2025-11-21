"""
M7风险监控器模块

该模块实现了一个全面的风险监控系统，包括：
1. 市场风险监控 - 监控价格波动、波动率、相关性等市场指标
2. 信用风险监控 - 监控交易对手信用状况、违约概率等
3. 操作风险监控 - 监控系统故障、人为错误、流程风险等
4. 流动性风险监控 - 监控市场流动性、持仓流动性等
5. 技术风险监控 - 监控系统性能、安全威胁等
6. 合规风险监控 - 监控监管合规性、交易限制等
7. 风险指标计算 - 计算各类风险指标和度量
8. 风险预警机制 - 基于阈值的风险预警系统
9. 风险监控报告 - 生成详细的风险监控报告

主要组件：
- RiskLevel: 风险等级枚举
- RiskType: 风险类型枚举  
- RiskMetrics: 风险指标数据类
- RiskAlert: 风险预警数据类
- RiskReport: 风险监控报告数据类
- RiskMonitor: 风险监控器主类

Author: AI量化交易系统
Date: 2025-11-13
Version: 1.0.0
License: MIT
"""

# 模块信息
__version__ = "1.0.0"
__author__ = "AI量化交易系统"
__email__ = "ai-quant@minimax.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 AI量化交易系统"

# 导入核心组件
from .RiskMonitor import (
    # 枚举类
    RiskLevel,
    RiskType,
    
    # 数据类
    RiskMetrics,
    RiskAlert,
    RiskReport,
    
    # 主类
    RiskMonitor
)

# 定义公共接口
__all__ = [
    # 枚举类
    'RiskLevel',
    'RiskType',
    
    # 数据类
    'RiskMetrics',
    'RiskAlert',
    'RiskReport',
    
    # 主类
    'RiskMonitor',
    
    # 便捷函数
    'create_risk_monitor',
    'quick_risk_assessment',
    'get_risk_level_display'
]

# ==================== 便捷函数 ====================

def create_risk_monitor(config: dict = None) -> RiskMonitor:
    """
    创建风险监控器实例的便捷函数
    
    Args:
        config: 可选的风险监控配置字典
        
    Returns:
        RiskMonitor: 风险监控器实例
        
    Example:
        >>> monitor = create_risk_monitor()
        >>> monitor = create_risk_monitor({'thresholds': {...}})
    """
    return RiskMonitor(config=config)


def quick_risk_assessment(market_data: dict, 
                         positions: dict = None,
                         counterparty_data: dict = None) -> dict:
    """
    快速风险评估功能
    
    Args:
        market_data: 市场数据字典，包含价格时间序列等
        positions: 可选，持仓数据字典
        counterparty_data: 可选，交易对手数据字典
        
    Returns:
        dict: 快速风险评估结果
        
    Example:
        >>> result = quick_risk_assessment(price_data, positions)
        >>> print(f"整体风险等级: {result['overall_risk_level']}")
    """
    monitor = RiskMonitor()
    
    # 执行市场风险监控
    if market_data and positions:
        monitor.monitor_market_risk(market_data, positions)
    
    # 执行信用风险监控
    if counterparty_data:
        exposure_data = counterparty_data.get('exposures', {})
        counterparty_info = counterparty_data.get('info', {})
        monitor.monitor_credit_risk(counterparty_info, exposure_data)
    
    # 生成评估报告
    report = monitor.generate_risk_report()
    
    return {
        'overall_risk_level': report.overall_risk_level.value,
        'risk_summary': {
            risk_type.value: summary 
            for risk_type, summary in report.risk_summary.items()
        },
        'key_metrics': [
            {
                'name': metric.metric_name,
                'value': metric.current_value,
                'risk_level': metric.risk_level.value,
                'risk_type': metric.risk_type.value
            }
            for metric in report.key_metrics
        ],
        'alerts_count': len(report.alerts),
        'recommendations': report.recommendations[:5]  # 只返回前5条建议
    }


def get_risk_level_display(risk_level: RiskLevel) -> dict:
    """
    获取风险等级的显示信息
    
    Args:
        risk_level: 风险等级枚举值
        
    Returns:
        dict: 包含颜色、描述等显示信息的字典
        
    Example:
        >>> info = get_risk_level_display(RiskLevel.HIGH)
        >>> print(f"颜色: {info['color']}")
    """
    level_info = {
        RiskLevel.LOW: {
            'color': 'green',
            'description': '低风险 - 正常运行',
            'icon': '✓',
            'priority': 1
        },
        RiskLevel.MEDIUM: {
            'color': 'orange', 
            'description': '中风险 - 需要关注',
            'icon': '⚠',
            'priority': 2
        },
        RiskLevel.HIGH: {
            'color': 'red',
            'description': '高风险 - 需要立即处理',
            'icon': '⚡',
            'priority': 3
        },
        RiskLevel.CRITICAL: {
            'color': 'darkred',
            'description': '严重风险 - 紧急处理',
            'icon': '🚨',
            'priority': 4
        }
    }
    
    return level_info.get(risk_level, level_info[RiskLevel.LOW])


def get_risk_type_display(risk_type: RiskType) -> dict:
    """
    获取风险类型的显示信息
    
    Args:
        risk_type: 风险类型枚举值
        
    Returns:
        dict: 包含颜色、描述等显示信息的字典
        
    Example:
        >>> info = get_risk_type_display(RiskType.MARKET)
        >>> print(f"描述: {info['description']}")
    """
    type_info = {
        RiskType.MARKET: {
            'color': 'blue',
            'description': '市场风险 - 价格波动、波动率风险',
            'icon': '📈',
            'category': '市场'
        },
        RiskType.CREDIT: {
            'color': 'purple',
            'description': '信用风险 - 交易对手违约风险',
            'icon': '💳',
            'category': '信用'
        },
        RiskType.OPERATIONAL: {
            'color': 'brown',
            'description': '操作风险 - 系统故障、人为错误',
            'icon': '⚙',
            'category': '操作'
        },
        RiskType.LIQUIDITY: {
            'color': 'teal',
            'description': '流动性风险 - 资金流动性不足',
            'icon': '💧',
            'category': '流动性'
        },
        RiskType.TECHNICAL: {
            'color': 'gray',
            'description': '技术风险 - 系统性能、安全威胁',
            'icon': '🔧',
            'category': '技术'
        },
        RiskType.COMPLIANCE: {
            'color': 'indigo',
            'description': '合规风险 - 监管合规性问题',
            'icon': '⚖',
            'category': '合规'
        }
    }
    
    return type_info.get(risk_type, type_info[RiskType.MARKET])


def create_default_config() -> dict:
    """
    创建默认的风险监控配置
    
    Returns:
        dict: 默认配置字典
        
    Example:
        >>> config = create_default_config()
        >>> monitor = RiskMonitor(config=config)
    """
    return {
        'thresholds': {
            'volatility': {'low': 0.1, 'medium': 0.2, 'high': 0.3},
            'var_95': {'low': 0.02, 'medium': 0.05, 'high': 0.1},
            'sharpe_ratio': {'low': 0.5, 'medium': 1.0, 'high': 2.0},
            'max_drawdown': {'low': 0.05, 'medium': 0.1, 'high': 0.2},
            'credit_score': {'low': 700, 'medium': 600, 'high': 500},
            'liquidity_ratio': {'low': 0.1, 'medium': 0.05, 'high': 0.02},
            'system_uptime': {'low': 0.999, 'medium': 0.995, 'high': 0.99},
            'compliance_score': {'low': 0.95, 'medium': 0.9, 'high': 0.8}
        },
        'alert_cooldown': 300,  # 5分钟预警冷却时间
        'max_alerts': 1000,     # 最大预警数量
        'data_retention_days': 30,  # 数据保留天数
        'auto_monitoring': True,    # 自动监控开关
        'report_generation': True,  # 自动报告生成
        'log_level': 'INFO'         # 日志级别
    }


def validate_config(config: dict) -> dict:
    """
    验证并补全风险监控配置
    
    Args:
        config: 配置字典
        
    Returns:
        dict: 验证后的完整配置
        
    Example:
        >>> config = validate_config({'thresholds': {...}})
    """
    default_config = create_default_config()
    
    # 合并配置
    if not isinstance(config, dict):
        config = {}
    
    validated_config = default_config.copy()
    validated_config.update(config)
    
    # 验证阈值配置
    if 'thresholds' in config and isinstance(config['thresholds'], dict):
        for key, value in config['thresholds'].items():
            if key in default_config['thresholds'] and isinstance(value, dict):
                validated_config['thresholds'][key].update(value)
    
    return validated_config


# ==================== 模块初始化日志 ====================

def _initialize_module():
    """模块初始化函数"""
    import logging
    
    logger = logging.getLogger('M7')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.info(f"M7风险监控器模块已加载 (版本: {__version__})")
    logger.info(f"可用组件: {', '.join(__all__[:7])}")  # 显示前7个主要组件


# ==================== 模块级属性 ====================

# 模块常量
MODULE_NAME = "M7风险监控器"
SUPPORTED_FORMATS = ['json', 'csv', 'html']
SUPPORTED_RISK_TYPES = list(RiskType)
SUPPORTED_RISK_LEVELS = list(RiskLevel)

# 配置信息
DEFAULT_CONFIG = create_default_config()

# 便捷别名
RiskMonitorClass = RiskMonitor  # 便于区分类和函数
RiskConfig = dict  # 风险配置类型别名

# ==================== 执行模块初始化 ====================

# 自动执行模块初始化
_initialize_module()

# 模块级别元数据
__all__.extend([
    'MODULE_NAME',
    'SUPPORTED_FORMATS', 
    'SUPPORTED_RISK_TYPES',
    'SUPPORTED_RISK_LEVELS',
    'DEFAULT_CONFIG',
    'create_default_config',
    'validate_config',
    'get_risk_type_display'
])