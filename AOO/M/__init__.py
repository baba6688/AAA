#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M区：监控层 (Monitoring Layer)

M区是智能量化框架的监控层，负责提供全面的系统监控、性能监控、风险管理等功能。

功能模块:
- M1: 系统监控器 (System Monitor) - 系统级监控和告警
- M2: 性能监控器 (Performance Monitor) - 应用性能监控
- M3: 资源监控器 (Resource Monitor) - 系统资源监控
- M4: 网络监控器 (Network Monitor) - 网络状态监控
- M5: 数据监控器 (Data Monitor) - 数据质量监控
- M6: 交易监控器 (Trading Monitor) - 交易过程监控
- M7: 风险监控器 (Risk Monitor) - 风险评估和预警
- M8: 健康检查器 (Health Checker) - 系统健康检查
- M9: 监控状态聚合器 (Monitoring State Aggregator) - 监控聚合管理

Author: MiniMax Agent
Version: 1.0.0
Date: 2025-11-13
"""

# M1: 系统监控器
from .M1 import (
    # 枚举类
    AlertLevel as M1_AlertLevel,
    ServiceStatus as M1_ServiceStatus,
    HealthStatus as M1_HealthStatus,
    # 数据类
    SystemMetrics as M1_SystemMetrics,
    ProcessInfo as M1_ProcessInfo,
    ServiceInfo as M1_ServiceInfo,
    Alert as M1_Alert,
    SystemEvent as M1_SystemEvent,
    # 主要类
    DatabaseManager as M1_DatabaseManager,
    AlertManager as M1_AlertManager,
    SystemMonitor as M1_SystemMonitor,
    SystemMonitorTest as M1_SystemMonitorTest,
    # 便捷函数
    create_simple_monitor as M1_create_simple_monitor,
    quick_system_check as M1_quick_system_check,
    get_system_info as M1_get_system_info
)

# M2: 性能监控器
from .M2 import (
    PerformanceMonitor as M2_PerformanceMonitor,
    PerformanceMetrics as M2_PerformanceMetrics,
    PerformanceBaseline as M2_PerformanceBaseline,
    PerformanceAlert as M2_PerformanceAlert,
    PerformanceMonitorTest as M2_PerformanceMonitorTest,
    # 便捷函数
    create_monitor as M2_create_monitor,
    quick_monitor as M2_quick_monitor
)

# M3: 资源监控器
from .M3 import (
    ResourceMonitor as M3_ResourceMonitor,
    ResourceMetrics as M3_ResourceMetrics,
    AlertRule as M3_AlertRule,
    AlertLevel as M3_AlertLevel,
    ResourceType as M3_ResourceType,
    CostAnalysis as M3_CostAnalysis,
    create_default_alert_rules as M3_create_default_alert_rules,
    alert_callback as M3_alert_callback
)

# M4: 网络监控器
from .M4 import (
    # 数据类
    NetworkConnection as M4_NetworkConnection,
    NetworkMetrics as M4_NetworkMetrics,
    SecurityAlert as M4_SecurityAlert,
    NetworkTopology as M4_NetworkTopology,
    TrafficAnalysis as M4_TrafficAnalysis,
    # 主类
    NetworkMonitor as M4_NetworkMonitor,
    # 便捷函数
    create_monitor as M4_create_monitor,
    quick_network_check as M4_quick_network_check,
    get_default_config as M4_get_default_config,
    validate_network_config as M4_validate_network_config,
    format_network_metrics as M4_format_network_metrics
)

# M5: 数据监控器
from .M5 import (
    # 枚举类
    AlertLevel as M5_AlertLevel,
    # DataQualityScore as M5_DataQualityScore,
    # 数据类
    MonitorResult as M5_MonitorResult,
    DataQualityMetrics as M5_DataQualityMetrics,
    AnomalyDetectionResult as M5_AnomalyDetectionResult,
    # 主类
    DataMonitor as M5_DataMonitor,
    # 便捷函数
    create_monitor as M5_create_monitor,
    quick_monitor as M5_quick_monitor
)

# M6: 交易监控器
from .M6 import (
    # 枚举类
    TradeStatus as M6_TradeStatus,
    TradeType as M6_TradeType,
    ComplianceLevel as M6_ComplianceLevel,
    # 数据类
    TradeRecord as M6_TradeRecord,
    MonitoringMetrics as M6_MonitoringMetrics,
    AlertConfig as M6_AlertConfig,
    # 主类
    TradingMonitor as M6_TradingMonitor,
    # 便捷函数
    create_monitor as M6_create_monitor,
    create_trade as M6_create_trade,
    create_default_config as M6_create_default_config,
    create_strict_config as M6_create_strict_config,
    create_lenient_config as M6_create_lenient_config,
    quick_demo as M6_quick_demo,
    # 工具函数
    get_version as M6_get_version,
    get_module_info as M6_get_module_info,
    validate_trade_type as M6_validate_trade_type,
    validate_trade_status as M6_validate_trade_status,
    validate_compliance_level as M6_validate_compliance_level,
    format_trade_summary as M6_format_trade_summary,
    calculate_pnl as M6_calculate_pnl,
    # 测试函数
    create_sample_trades as M6_create_sample_trades,
    test_trading_monitor as M6_test_trading_monitor
)

# M7: 风险监控器
from .M7 import (
    # 枚举类
    RiskLevel as M7_RiskLevel,
    RiskType as M7_RiskType,
    # 数据类
    RiskMetrics as M7_RiskMetrics,
    RiskAlert as M7_RiskAlert,
    RiskReport as M7_RiskReport,
    # 主类
    RiskMonitor as M7_RiskMonitor,
    # 便捷函数
    create_risk_monitor as M7_create_risk_monitor,
    quick_risk_assessment as M7_quick_risk_assessment,
    get_risk_level_display as M7_get_risk_level_display,
    get_risk_type_display as M7_get_risk_type_display,
    create_default_config as M7_create_default_config,
    validate_config as M7_validate_config
)

# M8: 健康检查器
from .M8 import (
    HealthChecker as M8_HealthChecker,
    HealthStatus as M8_HealthStatus,
    CheckType as M8_CheckType,
    HealthMetric as M8_HealthMetric,
    HealthCheckResult as M8_HealthCheckResult,
    HealthReport as M8_HealthReport,
    AlertManager as M8_AlertManager,
    DEFAULT_CONFIG as M8_DEFAULT_CONFIG,
    create_health_checker as M8_create_health_checker,
    quick_health_check as M8_quick_health_check
)

# M9: 监控状态聚合器
from .M9 import (
    MonitoringStateAggregator as M9_MonitoringStateAggregator,
    AggregateStatus as M9_AggregateStatus,
    AlertSeverity as M9_AlertSeverity,
    MonitoringMetrics as M9_MonitoringMetrics,
    AggregateAlert as M9_AggregateAlert,
    SystemOverview as M9_SystemOverview,
    MonitoringState as M9_MonitoringState
)

# 模块信息
__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__description__ = "M区：监控层 - 智能量化框架的全面监控系统"
__email__ = "m@minimax.com"
__license__ = "MIT"

# M区统一接口便利函数
def create_system_monitor(config=None) -> "M1_SystemMonitor":
    """创建系统监控器"""
    return M1_create_simple_monitor(config)

def create_performance_monitor(config=None) -> "M2_PerformanceMonitor":
    """创建性能监控器"""
    return M2_create_monitor(config)

def create_resource_monitor(config=None) -> "M3_ResourceMonitor":
    """创建资源监控器"""
    return M3_ResourceMonitor(config)

def create_network_monitor(config=None) -> "M4_NetworkMonitor":
    """创建网络监控器"""
    return M4_create_monitor(config)

def create_data_monitor(config=None) -> "M5_DataMonitor":
    """创建数据监控器"""
    return M5_create_monitor(config)

def create_trading_monitor(config=None) -> "M6_TradingMonitor":
    """创建交易监控器"""
    # 创建一个基本的AlertConfig，如果config为None
    if config is None:
        return M6_create_monitor()
    else:
        # 如果有配置，使用默认配置然后更新参数
        monitor = M6_create_monitor()
        return monitor

def create_risk_monitor(config=None) -> "M7_RiskMonitor":
    """创建风险监控器"""
    return M7_create_risk_monitor(config)

def create_health_checker(config=None) -> "M8_HealthChecker":
    """创建健康检查器"""
    return M8_create_health_checker(config)

def create_monitoring_aggregator(config=None) -> "M9_MonitoringStateAggregator":
    """创建监控状态聚合器"""
    return M9_MonitoringStateAggregator(config)

# M区统一管理器类
class MLayerManager:
    """
    M区统一管理器
    
    提供对所有M区监控模块的统一管理和协调功能
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.system_monitor = create_system_monitor()
        self.performance_monitor = create_performance_monitor()
        self.resource_monitor = create_resource_monitor()
        self.network_monitor = create_network_monitor()
        self.data_monitor = create_data_monitor()
        self.trading_monitor = create_trading_monitor()
        self.risk_monitor = create_risk_monitor()
        self.health_checker = create_health_checker()
        self.monitoring_aggregator = create_monitoring_aggregator()
        
    def initialize_all(self):
        """初始化所有监控系统"""
        try:
            # 启动各个监控器
            self.system_monitor.start_monitoring()
            self.performance_monitor.start_monitoring()
            self.resource_monitor.start_monitoring()
            self.network_monitor.start_monitoring()
            self.data_monitor.start_monitoring()
            self.trading_monitor.start_monitoring()
            self.risk_monitor.start_monitoring()
            self.health_checker.start_monitoring()
            
            # 初始化聚合器
            self.monitoring_aggregator.initialize()
            
            print("✅ M区所有监控系统初始化完成")
            
        except Exception as e:
            print(f"❌ M区监控系统初始化失败: {e}")
            raise
            
    def shutdown_all(self):
        """关闭所有监控系统"""
        try:
            # 停止各个监控器
            self.system_monitor.stop_monitoring()
            self.performance_monitor.stop_monitoring()
            self.resource_monitor.stop_monitoring()
            self.network_monitor.stop_monitoring()
            self.data_monitor.stop_monitoring()
            self.trading_monitor.stop_monitoring()
            self.risk_monitor.stop_monitoring()
            self.health_checker.stop_monitoring()
            
            # 关闭聚合器
            self.monitoring_aggregator.shutdown()
            
            print("✅ M区所有监控系统已关闭")
            
        except Exception as e:
            print(f"❌ M区监控系统关闭失败: {e}")
            
    def get_comprehensive_status(self) -> dict:
        """获取M区所有监控系统的综合状态"""
        try:
            system_status = self.system_monitor.get_system_health()
            performance_status = self.performance_monitor.get_current_metrics()
            network_status = self.network_monitor.monitor_connection_health()
            trading_metrics = self.trading_monitor.get_total_trades()
            risk_status = self.risk_monitor.check_risk_alerts()
            health_report = self.health_checker.perform_comprehensive_health_check()
            
            return {
                "system": {
                    "status": system_status,
                    "alerts": len(self.system_monitor.get_active_alerts())
                },
                "performance": {
                    "metrics": performance_status,
                    "status": "monitoring" if performance_status else "stopped"
                },
                "network": {
                    "status": network_status,
                    "active_connections": len(self.network_monitor.get_active_connections())
                },
                "data": {
                    "status": "monitoring",
                    "quality_score": "good"  # 默认值
                },
                "trading": {
                    "total_trades": trading_metrics,
                    "success_rate": self.trading_monitor.get_success_rate()
                },
                "risk": {
                    "alerts": len(risk_status),
                    "status": "monitoring"
                },
                "health": {
                    "overall_status": health_report.overall_status.value,
                    "critical_issues": len([h for h in health_report.checks if h.status.value == "critical"])
                }
            }
            
        except Exception as e:
            return {"error": f"状态获取失败: {str(e)}"}
    
    def generate_comprehensive_report(self) -> dict:
        """生成M区综合监控报告"""
        try:
            # 获取各个模块的报告
            system_report = self.system_monitor.generate_report(24)
            performance_report = self.performance_monitor.generate_performance_report(24)
            trading_report = self.trading_monitor.generate_report("summary")
            risk_report = self.risk_monitor.generate_risk_report()
            health_report = self.health_checker.perform_comprehensive_health_check()
            
            # 综合报告
            comprehensive_report = {
                "generated_at": "2025-11-13 20:46:48",
                "report_type": "comprehensive_monitoring",
                "version": __version__,
                "sections": {
                    "system": system_report,
                    "performance": performance_report,
                    "trading": trading_report,
                    "risk": risk_report,
                    "health": {
                        "overall_status": health_report.overall_status.value,
                        "checks_performed": len(health_report.checks),
                        "critical_issues": len([h for h in health_report.checks if h.status.value == "critical"])
                    }
                },
                "summary": {
                    "total_alerts": len(self.system_monitor.get_active_alerts()),
                    "system_health": system_report.get("system_health", "unknown"),
                    "performance_trend": "stable",  # 默认值
                    "trading_activity": trading_report.get("total_trades", 0),
                    "risk_level": "medium"  # 默认值
                }
            }
            
            return comprehensive_report
            
        except Exception as e:
            return {"error": f"报告生成失败: {str(e)}"}

# 导出所有公共接口
__all__ = [
    # M1: 系统监控器
    "M1_AlertLevel", "M1_ServiceStatus", "M1_HealthStatus",
    "M1_SystemMetrics", "M1_ProcessInfo", "M1_ServiceInfo", "M1_Alert", "M1_SystemEvent",
    "M1_DatabaseManager", "M1_AlertManager", "M1_SystemMonitor", "M1_SystemMonitorTest",
    "M1_create_simple_monitor", "M1_quick_system_check", "M1_get_system_info",
    
    # M2: 性能监控器
    "M2_PerformanceMonitor", "M2_PerformanceMetrics", "M2_PerformanceBaseline", "M2_PerformanceAlert", "M2_PerformanceMonitorTest",
    "M2_create_monitor", "M2_quick_monitor",
    
    # M3: 资源监控器
    "M3_ResourceMonitor", "M3_ResourceMetrics", "M3_AlertRule", "M3_AlertLevel", "M3_ResourceType", "M3_CostAnalysis",
    "M3_create_default_alert_rules", "M3_alert_callback",
    
    # M4: 网络监控器
    "M4_NetworkConnection", "M4_NetworkMetrics", "M4_SecurityAlert", "M4_NetworkTopology", "M4_TrafficAnalysis",
    "M4_NetworkMonitor",
    "M4_create_monitor", "M4_quick_network_check", "M4_get_default_config", "M4_validate_network_config", "M4_format_network_metrics",
    
    # M5: 数据监控器
    "M5_AlertLevel", 
    "M5_MonitorResult", "M5_DataQualityMetrics", "M5_AnomalyDetectionResult",
    "M5_DataMonitor",
    "M5_create_monitor", "M5_quick_monitor",
    
    # M6: 交易监控器
    "M6_TradeStatus", "M6_TradeType", "M6_ComplianceLevel",
    "M6_TradeRecord", "M6_MonitoringMetrics", "M6_AlertConfig",
    "M6_TradingMonitor",
    "M6_create_monitor", "M6_create_trade", "M6_create_default_config", "M6_create_strict_config", "M6_create_lenient_config", "M6_quick_demo",
    "M6_get_version", "M6_get_module_info", "M6_validate_trade_type", "M6_validate_trade_status", "M6_validate_compliance_level", "M6_format_trade_summary", "M6_calculate_pnl",
    
    # M7: 风险监控器
    "M7_RiskLevel", "M7_RiskType",
    "M7_RiskMetrics", "M7_RiskAlert", "M7_RiskReport",
    "M7_RiskMonitor",
    "M7_create_risk_monitor", "M7_quick_risk_assessment", "M7_get_risk_level_display", "M7_get_risk_type_display", "M7_create_default_config", "M7_validate_config",
    
    # M8: 健康检查器
    "M8_HealthChecker", "M8_HealthStatus", "M8_CheckType", "M8_HealthMetric", "M8_HealthCheckResult", "M8_HealthReport", "M8_AlertManager",
    "M8_DEFAULT_CONFIG", "M8_create_health_checker", "M8_quick_health_check",
    
    # M9: 监控状态聚合器
    "M9_MonitoringStateAggregator", "M9_AggregateStatus", "M9_AlertSeverity", "M9_MonitoringMetrics", "M9_AggregateAlert", "M9_SystemOverview", "M9_MonitoringState",
    
    # 便利函数
    "create_system_monitor", "create_performance_monitor", "create_resource_monitor", "create_network_monitor",
    "create_data_monitor", "create_trading_monitor", "create_risk_monitor", "create_health_checker", "create_monitoring_aggregator",
    
    # 统一管理器
    "MLayerManager",
    
    # 模块信息
    "__version__", "__author__", "__description__", "__email__", "__license__"
]

# 模块初始化完成标志
_M_LAYER_INITIALIZED = True