#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L区：日志层 (Logging Layer)

L区是智能量化框架的日志层，负责提供全面的日志记录、监控、审计和管理功能。

功能模块:
- L1: 系统日志记录器 (System Logger) - 系统级日志管理
- L2: 交易日志记录器 (Trading Logger) - 交易相关日志记录
- L3: 错误日志记录器 (Error Logger) - 错误追踪和分析
- L4: 性能日志记录器 (Performance Logger) - 性能监控日志
- L5: 学习日志记录器 (Learning Logger) - 机器学习日志记录
- L6: 决策日志记录器 (Decision Logger) - 决策过程日志
- L7: 调试日志记录器 (Debug Logger) - 调试和诊断日志
- L8: 审计日志记录器 (Audit Logger) - 企业级审计日志
- L9: 日志状态聚合器 (Logging State Aggregator) - 日志系统聚合

Author: MiniMax Agent
Version: 1.0.0
Date: 2025-11-13
"""

# L1: 系统日志记录器
from .L1 import (
    # 枚举类
    LogLevel as L1_LogLevel,
    LogFormat as L1_LogFormat,
    RotationStrategy as L1_RotationStrategy,
    CompressionType as L1_CompressionType,
    # 数据结构
    LogRecord as L1_LogRecord,
    SystemLoggerConfig as L1_SystemLoggerConfig,
    # 格式化器
    LogFormatter as L1_LogFormatter,
    SimpleFormatter as L1_SimpleFormatter,
    DetailedFormatter as L1_DetailedFormatter,
    JsonFormatter as L1_JsonFormatter,
    CustomFormatter as L1_CustomFormatter,
    # 过滤器
    LogFilter as L1_LogFilter,
    LevelFilter as L1_LevelFilter,
    PatternFilter as L1_PatternFilter,
    ModuleFilter as L1_ModuleFilter,
    CompositeFilter as L1_CompositeFilter,
    # 输出目标
    OutputTarget as L1_OutputTarget,
    FileTarget as L1_FileTarget,
    ConsoleTarget as L1_ConsoleTarget,
    NetworkTarget as L1_NetworkTarget,
    DatabaseTarget as L1_DatabaseTarget,
    # 管理器
    LogRotationManager as L1_LogRotationManager,
    AsyncLogProcessor as L1_AsyncLogProcessor,
    LogSearchEngine as L1_LogSearchEngine,
    # 主要类
    SystemLogger as L1_SystemLogger,
    # 工厂函数
    create_system_logger as L1_create_system_logger,
    setup_system_logging as L1_setup_system_logging
)

# L2: 交易日志记录器
from .L2 import (
    # 主要类
    TradingLogger as L2_TradingLogger,
    ExtendedTradingLogger as L2_ExtendedTradingLogger,
    AdvancedAnalytics as L2_AdvancedAnalytics,
    AlertManager as L2_AlertManager,
    DataValidator as L2_DataValidator,
    ConfigManager as L2_ConfigManager,
    HealthMonitor as L2_HealthMonitor,
    DatabaseManager as L2_DatabaseManager,
    AsyncLogProcessor as L2_AsyncLogProcessor,
    LogRotator as L2_LogRotator,
    # 数据类
    TradeEvent as L2_TradeEvent,
    PerformanceMetrics as L2_PerformanceMetrics,
    ErrorEvent as L2_ErrorEvent,
    StrategyEvent as L2_StrategyEvent,
    MarketDataEvent as L2_MarketDataEvent,
    StatisticsSummary as L2_StatisticsSummary,
    # 枚举类
    LogLevel as L2_LogLevel,
    LogType as L2_LogType,
    TradeEventType as L2_TradeEventType,
    OrderSide as L2_OrderSide,
    OrderType as L2_OrderType
)

# L3: 错误日志记录器
from .L3 import (
    ErrorLogger as L3_ErrorLogger,
    ErrorContext as L3_ErrorContext,
    ErrorStatistics as L3_ErrorStatistics,
    AlertManager as L3_AlertManager,
    RecoveryManager as L3_RecoveryManager,
    ErrorType as L3_ErrorType,
    ErrorSeverity as L3_ErrorSeverity,
    ErrorStatus as L3_ErrorStatus,
    AlertChannel as L3_AlertChannel,
    RecoveryStrategy as L3_RecoveryStrategy
)

# L4: 性能日志记录器
from .L4 import (
    # 主要类
    PerformanceLogger as L4_PerformanceLogger,
    # 监控器类
    SystemPerformanceMonitor as L4_SystemPerformanceMonitor,
    ApplicationPerformanceMonitor as L4_ApplicationPerformanceMonitor,
    DatabasePerformanceMonitor as L4_DatabasePerformanceMonitor,
    NetworkPerformanceMonitor as L4_NetworkPerformanceMonitor,
    # 管理器类
    AlertManager as L4_AlertManager,
    TrendAnalyzer as L4_TrendAnalyzer,
    AsyncLogProcessor as L4_AsyncLogProcessor,
    NotificationManager as L4_NotificationManager,
    PerformanceReportGenerator as L4_PerformanceReportGenerator,
    # 数据结构
    PerformanceData as L4_PerformanceData,
    AlertRule as L4_AlertRule,
    Alert as L4_Alert,
    TrendAnalysis as L4_TrendAnalysis,
    # 枚举
    PerformanceMetric as L4_PerformanceMetric,
    AlertLevel as L4_AlertLevel,
    LogLevel as L4_LogLevel,
    DatabaseType as L4_DatabaseType,
    NetworkProtocol as L4_NetworkProtocol
)

# L5: 学习日志记录器
from .L5 import (
    # 主要类
    LearningLogger as L5_LearningLogger,
    # 枚举类
    LogLevel as L5_LogLevel,
    ExperimentStatus as L5_ExperimentStatus,
    ModelVersionStatus as L5_ModelVersionStatus,
    DataProcessingStage as L5_DataProcessingStage,
    HyperparameterSearchStatus as L5_HyperparameterSearchStatus,
    # 数据类
    TrainingMetrics as L5_TrainingMetrics,
    EvaluationMetrics as L5_EvaluationMetrics,
    HyperparameterConfig as L5_HyperparameterConfig,
    HyperparameterTrial as L5_HyperparameterTrial,
    DataProcessingLog as L5_DataProcessingLog,
    ExperimentConfig as L5_ExperimentConfig,
    ExperimentResult as L5_ExperimentResult,
    ModelVersion as L5_ModelVersion,
    AsyncLogEntry as L5_AsyncLogEntry,
    # 异常类
    LearningLoggerError as L5_LearningLoggerError,
    DatabaseError as L5_DatabaseError,
    ExperimentError as L5_ExperimentError,
    ModelVersionError as L5_ModelVersionError,
    AsyncProcessingError as L5_AsyncProcessingError,
    # 存储后端
    LogStorageBackend as L5_LogStorageBackend,
    SQLiteStorageBackend as L5_SQLiteStorageBackend
)

# L6: 决策日志记录器
from .L6 import (
    # 主类
    DecisionLogger as L6_DecisionLogger,
    # 枚举类
    LogLevel as L6_LogLevel,
    DecisionType as L6_DecisionType,
    StrategyType as L6_StrategyType,
    RiskLevel as L6_RiskLevel,
    ExecutionStatus as L6_ExecutionStatus,
    # 数据类
    DecisionInput as L6_DecisionInput,
    DecisionLogic as L6_DecisionLogic,
    DecisionResult as L6_DecisionResult,
    StrategyStep as L6_StrategyStep,
    RiskAssessment as L6_RiskAssessment,
    MarketAnalysis as L6_MarketAnalysis,
    DecisionExplanation as L6_DecisionExplanation,
    EffectEvaluation as L6_EffectEvaluation,
    # 存储相关类
    LogStorage as L6_LogStorage,
    FileStorage as L6_FileStorage,
    MemoryStorage as L6_MemoryStorage,
    # 核心处理器
    AsyncLogProcessor as L6_AsyncLogProcessor,
    # 功能扩展类
    LogValidator as L6_LogValidator,
    LogCompressor as L6_LogCompressor,
    DecisionAnalytics as L6_DecisionAnalytics,
    LogExporter as L6_LogExporter,
    LogMonitor as L6_LogMonitor,
    LogArchiver as L6_LogArchiver,
    # 高级功能类
    DecisionPredictor as L6_DecisionPredictor,
    DecisionOptimizer as L6_DecisionOptimizer,
    DecisionSimulator as L6_DecisionSimulator,
    # 工具函数
    log_execution_time as L6_log_execution_time,
    validate_input as L6_validate_input,
    cache_result as L6_cache_result,
    get_log_type_stats as L6_get_log_type_stats,
    create_decision_summary as L6_create_decision_summary,
    calculate_risk_score as L6_calculate_risk_score,
    calculate_performance_score as L6_calculate_performance_score,
    generate_correlation_matrix as L6_generate_correlation_matrix,
    calculate_pearson_correlation as L6_calculate_pearson_correlation,
    format_file_size as L6_format_file_size,
    validate_timestamp as L6_validate_timestamp,
    sanitize_filename as L6_sanitize_filename,
    create_backup_name as L6_create_backup_name,
    # 便利函数
    create_logger as L6_create_logger,
    quick_log_decision as L6_quick_log_decision,
    quick_batch_log_decisions as L6_quick_batch_log_decisions,
    quick_log_risk as L6_quick_log_risk
)

# L7: 调试日志记录器
from .L7 import (
    # 主要类
    DebugLogger as L7_DebugLogger,
    # 异常类
    DebugLoggerError as L7_DebugLoggerError,
    MemoryLeakError as L7_MemoryLeakError,
    PerformanceThresholdError as L7_PerformanceThresholdError,
    NetworkDebugError as L7_NetworkDebugError,
    DatabaseDebugError as L7_DatabaseDebugError,
    # 常量类
    LogLevel as L7_LogLevel,
    DebugCategory as L7_DebugCategory,
    # 装饰器
    debug_logger as L7_debug_logger,
    performance_profile as L7_performance_profile,
    # 便捷函数
    get_default_logger as L7_get_default_logger,
    debug as L7_debug,
    info as L7_info,
    warning as L7_warning,
    error as L7_error,
    critical as L7_critical,
    # 基础类
    BaseLogHandler as L7_BaseLogHandler,
    # 数据结构类
    LogEntry as L7_LogEntry,
    MemorySnapshot as L7_MemorySnapshot,
    PerformanceMetrics as L7_PerformanceMetrics,
    # 工具类
    AsyncLogHandler as L7_AsyncLogHandler,
    FileLogHandler as L7_FileLogHandler,
    ConsoleLogHandler as L7_ConsoleLogHandler,
    DatabaseLogHandler as L7_DatabaseLogHandler,
    MemoryMonitor as L7_MemoryMonitor,
    PerformanceProfiler as L7_PerformanceProfiler,
    NetworkDebugger as L7_NetworkDebugger,
    DatabaseDebugger as L7_DatabaseDebugger,
    DebugToolIntegrator as L7_DebugToolIntegrator
)

# L8: 审计日志记录器
from .L8 import (
    # 枚举类
    AuditLevel as L8_AuditLevel,
    AuditCategory as L8_AuditCategory,
    UserOperation as L8_UserOperation,
    DataOperation as L8_DataOperation,
    ConfigOperation as L8_ConfigOperation,
    TransactionOperation as L8_TransactionOperation,
    SecurityEvent as L8_SecurityEvent,
    ComplianceType as L8_ComplianceType,
    # 数据结构
    AuditContext as L8_AuditContext,
    AuditRecord as L8_AuditRecord,
    AuditFilter as L8_AuditFilter,
    # 存储接口和实现
    AuditStorage as L8_AuditStorage,
    SQLiteAuditStorage as L8_SQLiteAuditStorage,
    FileAuditStorage as L8_FileAuditStorage,
    # 处理器
    AuditHandler as L8_AuditHandler,
    ConsoleAuditHandler as L8_ConsoleAuditHandler,
    FileAuditHandler as L8_FileAuditHandler,
    EmailAuditHandler as L8_EmailAuditHandler,
    WebhookAuditHandler as L8_WebhookAuditHandler,
    # 异步处理
    AsyncAuditQueue as L8_AsyncAuditQueue,
    # 主要记录器
    AuditLogger as L8_AuditLogger,
    # 分析和验证
    AuditAnalyzer as L8_AuditAnalyzer,
    AuditValidator as L8_AuditValidator,
    AuditCacheManager as L8_AuditCacheManager,
    AuditRuleEngine as L8_AuditRuleEngine,
    AuditRules as L8_AuditRules,
    AdvancedAuditAnalyzer as L8_AdvancedAuditAnalyzer,
    # 报告和显示
    AuditReportGenerator as L8_AuditReportGenerator,
    AuditDashboard as L8_AuditDashboard,
    # 压缩和归档
    AuditCompressor as L8_AuditCompressor,
    AuditArchiver as L8_AuditArchiver,
    # 集成和工具
    AuditIntegrationTool as L8_AuditIntegrationTool,
    AuditPerformanceMonitor as L8_AuditPerformanceMonitor,
    AuditConfigManager as L8_AuditConfigManager,
    AuditUtils as L8_AuditUtils,
    AuditTestTool as L8_AuditTestTool,
    # 创建函数
    create_audit_context as L8_create_audit_context,
    # 单元测试
    AuditLoggerTest as L8_AuditLoggerTest
)

# L9: 日志状态聚合器
from .L9 import (
    # 主要类
    LoggingStateAggregator as L9_LoggingStateAggregator,
    # 数据结构
    LogEntry as L9_LogEntry,
    LogStatistics as L9_LogStatistics,
    HealthCheckResult as L9_HealthCheckResult,
    Alert as L9_Alert,
    # 枚举
    LogLevel as L9_LogLevel,
    LogStatus as L9_LogStatus,
    HealthStatus as L9_HealthStatus,
    AlertLevel as L9_AlertLevel,
    AggregationType as L9_AggregationType,
    # 异常
    LoggingStateAggregatorError as L9_LoggingStateAggregatorError,
    LogProcessingError as L9_LogProcessingError,
    LogStorageError as L9_LogStorageError,
    LogTransmissionError as L9_LogTransmissionError,
    LogHealthCheckError as L9_LogHealthCheckError,
    LogAggregationError as L9_LogAggregationError,
    # 组件类
    LogStateMonitor as L9_LogStateMonitor,
    LogLifecycleManager as L9_LogLifecycleManager,
    LogStatisticsManager as L9_LogStatisticsManager,
    LogHealthChecker as L9_LogHealthChecker,
    LogAlertSystem as L9_LogAlertSystem,
    LogCoordinator as L9_LogCoordinator,
    LogStorageManager as L9_LogStorageManager,
    LogTransmissionManager as L9_LogTransmissionManager,
    check_dependencies as L9_check_dependencies
)

# 模块信息
__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__description__ = "L区：日志层 - 智能量化框架的日志记录和管理系统"
__email__ = "l@minimax.com"
__license__ = "MIT"

# L区统一接口便利函数
def create_system_logger(name: str = "L_System") -> "L1_SystemLogger":
    """创建系统日志记录器"""
    return L1_create_system_logger(name)

def create_trading_logger(config: dict = None) -> "L2_TradingLogger":
    """创建交易日志记录器"""
    return L2_TradingLogger(config)

def create_error_logger() -> "L3_ErrorLogger":
    """创建错误日志记录器"""
    return L3_ErrorLogger()

def create_performance_logger(name: str = "L_Performance") -> "L4_PerformanceLogger":
    """创建性能日志记录器"""
    return L4_PerformanceLogger(name=name)

def create_learning_logger(db_path: str = None) -> "L5_LearningLogger":
    """创建学习日志记录器"""
    return L5_LearningLogger(db_path=db_path)

def create_decision_logger(storage_type: str = "memory") -> "L6_DecisionLogger":
    """创建决策日志记录器"""
    return L6_create_logger(storage_type=storage_type)

def create_debug_logger(name: str = "L_Debug") -> "L7_DebugLogger":
    """创建调试日志记录器"""
    return L7_get_default_logger(name)

def create_audit_logger(storage_type: str = "sqlite") -> "L8_AuditLogger":
    """创建审计日志记录器"""
    if storage_type == "sqlite":
        storage = L8_SQLiteAuditStorage()
    else:
        storage = L8_FileAuditStorage()
    return L8_AuditLogger(storage=storage)

def create_logging_aggregator(config: dict = None) -> "L9_LoggingStateAggregator":
    """创建日志状态聚合器"""
    return L9_LoggingStateAggregator(config)

# L区统一管理器类
class LLayerManager:
    """
    L区统一管理器
    
    提供对所有L区日志模块的统一管理和协调功能
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.system_logger = create_system_logger()
        self.trading_logger = create_trading_logger()
        self.error_logger = create_error_logger()
        self.performance_logger = create_performance_logger()
        self.learning_logger = create_learning_logger()
        self.decision_logger = create_decision_logger()
        self.debug_logger = create_debug_logger()
        self.audit_logger = create_audit_logger()
        self.aggregator = create_logging_aggregator()
        
    def initialize_all(self):
        """初始化所有日志系统"""
        try:
            # 启动各个日志系统
            self.system_logger.start()
            self.performance_logger.start()
            self.decision_logger.start()
            self.audit_logger.start()
            
            # 初始化聚合器
            self.aggregator.initialize()
            
            print("✅ L区所有日志系统初始化完成")
            
        except Exception as e:
            print(f"❌ L区日志系统初始化失败: {e}")
            raise
            
    def shutdown_all(self):
        """关闭所有日志系统"""
        try:
            # 停止各个日志系统
            self.system_logger.stop()
            self.performance_logger.stop()
            self.decision_logger.stop()
            self.audit_logger.stop()
            
            # 关闭聚合器
            self.aggregator.shutdown()
            
            print("✅ L区所有日志系统已关闭")
            
        except Exception as e:
            print(f"❌ L区日志系统关闭失败: {e}")
            
    def get_status(self) -> dict:
        """获取L区所有日志系统状态"""
        return {
            "system_logger": "running" if hasattr(self.system_logger, 'is_running') and self.system_logger.is_running else "unknown",
            "trading_logger": "available",
            "error_logger": "available", 
            "performance_logger": "running" if hasattr(self.performance_logger, 'is_running') and self.performance_logger.is_running else "unknown",
            "learning_logger": "available",
            "decision_logger": "running" if hasattr(self.decision_logger, 'is_running') and self.decision_logger.is_running else "unknown",
            "debug_logger": "available",
            "audit_logger": "running" if hasattr(self.audit_logger, 'is_running') and self.audit_logger.is_running else "unknown",
            "aggregator": "available"
        }

# 导出所有公共接口
__all__ = [
    # L1: 系统日志记录器
    "L1_LogLevel", "L1_LogFormat", "L1_RotationStrategy", "L1_CompressionType",
    "L1_LogRecord", "L1_SystemLoggerConfig",
    "L1_LogFormatter", "L1_SimpleFormatter", "L1_DetailedFormatter", "L1_JsonFormatter", "L1_CustomFormatter",
    "L1_LogFilter", "L1_LevelFilter", "L1_PatternFilter", "L1_ModuleFilter", "L1_CompositeFilter",
    "L1_OutputTarget", "L1_FileTarget", "L1_ConsoleTarget", "L1_NetworkTarget", "L1_DatabaseTarget",
    "L1_LogRotationManager", "L1_AsyncLogProcessor", "L1_LogSearchEngine",
    "L1_SystemLogger", "L1_create_system_logger", "L1_setup_system_logging",
    
    # L2: 交易日志记录器
    "L2_TradingLogger", "L2_ExtendedTradingLogger", "L2_AdvancedAnalytics", "L2_AlertManager", "L2_DataValidator",
    "L2_ConfigManager", "L2_HealthMonitor", "L2_DatabaseManager", "L2_AsyncLogProcessor", "L2_LogRotator",
    "L2_TradeEvent", "L2_PerformanceMetrics", "L2_ErrorEvent", "L2_StrategyEvent", "L2_MarketDataEvent", "L2_StatisticsSummary",
    "L2_LogLevel", "L2_LogType", "L2_TradeEventType", "L2_OrderSide", "L2_OrderType",
    
    # L3: 错误日志记录器
    "L3_ErrorLogger", "L3_ErrorContext", "L3_ErrorStatistics", "L3_AlertManager", "L3_RecoveryManager",
    "L3_ErrorType", "L3_ErrorSeverity", "L3_ErrorStatus", "L3_AlertChannel", "L3_RecoveryStrategy",
    
    # L4: 性能日志记录器
    "L4_PerformanceLogger", "L4_SystemPerformanceMonitor", "L4_ApplicationPerformanceMonitor", 
    "L4_DatabasePerformanceMonitor", "L4_NetworkPerformanceMonitor",
    "L4_AlertManager", "L4_TrendAnalyzer", "L4_AsyncLogProcessor", "L4_NotificationManager", "L4_PerformanceReportGenerator",
    "L4_PerformanceData", "L4_AlertRule", "L4_Alert", "L4_TrendAnalysis",
    "L4_PerformanceMetric", "L4_AlertLevel", "L4_LogLevel", "L4_DatabaseType", "L4_NetworkProtocol",
    
    # L5: 学习日志记录器
    "L5_LearningLogger",
    "L5_LogLevel", "L5_ExperimentStatus", "L5_ModelVersionStatus", "L5_DataProcessingStage", "L5_HyperparameterSearchStatus",
    "L5_TrainingMetrics", "L5_EvaluationMetrics", "L5_HyperparameterConfig", "L5_HyperparameterTrial", 
    "L5_DataProcessingLog", "L5_ExperimentConfig", "L5_ExperimentResult", "L5_ModelVersion", "L5_AsyncLogEntry",
    "L5_LearningLoggerError", "L5_DatabaseError", "L5_ExperimentError", "L5_ModelVersionError", "L5_AsyncProcessingError",
    "L5_LogStorageBackend", "L5_SQLiteStorageBackend",
    
    # L6: 决策日志记录器
    "L6_DecisionLogger",
    "L6_LogLevel", "L6_DecisionType", "L6_StrategyType", "L6_RiskLevel", "L6_ExecutionStatus",
    "L6_DecisionInput", "L6_DecisionLogic", "L6_DecisionResult", "L6_StrategyStep", "L6_RiskAssessment",
    "L6_MarketAnalysis", "L6_DecisionExplanation", "L6_EffectEvaluation",
    "L6_LogStorage", "L6_FileStorage", "L6_MemoryStorage",
    "L6_AsyncLogProcessor",
    "L6_LogValidator", "L6_LogCompressor", "L6_DecisionAnalytics", "L6_LogExporter", "L6_LogMonitor", "L6_LogArchiver",
    "L6_DecisionPredictor", "L6_DecisionOptimizer", "L6_DecisionSimulator",
    "L6_log_execution_time", "L6_validate_input", "L6_cache_result", "L6_get_log_type_stats", "L6_create_decision_summary",
    "L6_calculate_risk_score", "L6_calculate_performance_score", "L6_generate_correlation_matrix", "L6_calculate_pearson_correlation",
    "L6_format_file_size", "L6_validate_timestamp", "L6_sanitize_filename", "L6_create_backup_name",
    "L6_create_logger", "L6_quick_log_decision", "L6_quick_batch_log_decisions", "L6_quick_log_risk",
    
    # L7: 调试日志记录器
    "L7_DebugLogger",
    "L7_DebugLoggerError", "L7_MemoryLeakError", "L7_PerformanceThresholdError", "L7_NetworkDebugError", "L7_DatabaseDebugError",
    "L7_LogLevel", "L7_DebugCategory",
    "L7_debug_logger", "L7_performance_profile",
    "L7_get_default_logger", "L7_debug", "L7_info", "L7_warning", "L7_error", "L7_critical",
    "L7_BaseLogHandler", "L7_LogEntry", "L7_MemorySnapshot", "L7_PerformanceMetrics",
    "L7_AsyncLogHandler", "L7_FileLogHandler", "L7_ConsoleLogHandler", "L7_DatabaseLogHandler",
    "L7_MemoryMonitor", "L7_PerformanceProfiler", "L7_NetworkDebugger", "L7_DatabaseDebugger", "L7_DebugToolIntegrator",
    
    # L8: 审计日志记录器
    "L8_AuditLevel", "L8_AuditCategory", "L8_UserOperation", "L8_DataOperation", "L8_ConfigOperation", 
    "L8_TransactionOperation", "L8_SecurityEvent", "L8_ComplianceType",
    "L8_AuditContext", "L8_AuditRecord", "L8_AuditFilter",
    "L8_AuditStorage", "L8_SQLiteAuditStorage", "L8_FileAuditStorage",
    "L8_AuditHandler", "L8_ConsoleAuditHandler", "L8_FileAuditHandler", "L8_EmailAuditHandler", "L8_WebhookAuditHandler",
    "L8_AsyncAuditQueue", "L8_AuditLogger",
    "L8_AuditAnalyzer", "L8_AuditValidator", "L8_AuditCacheManager", "L8_AuditRuleEngine", "L8_AuditRules", "L8_AdvancedAuditAnalyzer",
    "L8_AuditReportGenerator", "L8_AuditDashboard",
    "L8_AuditCompressor", "L8_AuditArchiver",
    "L8_AuditIntegrationTool", "L8_AuditPerformanceMonitor", "L8_AuditConfigManager", "L8_AuditUtils", "L8_AuditTestTool",
    "L8_create_audit_context", "L8_AuditLoggerTest",
    
    # L9: 日志状态聚合器
    "L9_LoggingStateAggregator",
    "L9_LogEntry", "L9_LogStatistics", "L9_HealthCheckResult", "L9_Alert",
    "L9_LogLevel", "L9_LogStatus", "L9_HealthStatus", "L9_AlertLevel", "L9_AggregationType",
    "L9_LoggingStateAggregatorError", "L9_LogProcessingError", "L9_LogStorageError", "L9_LogTransmissionError", 
    "L9_LogHealthCheckError", "L9_LogAggregationError",
    "L9_LogStateMonitor", "L9_LogLifecycleManager", "L9_LogStatisticsManager", "L9_LogHealthChecker", 
    "L9_LogAlertSystem", "L9_LogCoordinator", "L9_LogStorageManager", "L9_LogTransmissionManager", "L9_check_dependencies",
    
    # 便利函数
    "create_system_logger", "create_trading_logger", "create_error_logger", "create_performance_logger",
    "create_learning_logger", "create_decision_logger", "create_debug_logger", "create_audit_logger", "create_logging_aggregator",
    
    # 统一管理器
    "LLayerManager",
    
    # 模块信息
    "__version__", "__author__", "__description__", "__email__", "__license__"
]

# 模块初始化完成标志
_L_LAYER_INITIALIZED = True