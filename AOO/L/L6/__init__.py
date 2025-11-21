"""
L6决策日志记录器模块

提供全面的决策日志记录功能，包括智能决策过程、策略执行、风险评估、
市场分析、决策解释和效果评估等多个维度的日志记录。

主要类:
    DecisionLogger: 决策日志记录器主类
    LogLevel: 日志级别枚举
    DecisionType: 决策类型枚举
    StrategyType: 策略类型枚举
    RiskLevel: 风险级别枚举
    ExecutionStatus: 执行状态枚举
    DecisionInput: 决策输入数据类
    DecisionLogic: 决策逻辑数据类
    DecisionResult: 决策结果数据类
    StrategyStep: 策略执行步骤数据类
    RiskAssessment: 风险评估数据类
    MarketAnalysis: 市场分析数据类
    DecisionExplanation: 决策解释数据类
    EffectEvaluation: 决策效果评估数据类
    LogStorage: 日志存储抽象基类
    FileStorage: 文件存储实现
    MemoryStorage: 内存存储实现
    AsyncLogProcessor: 异步日志处理器
    LogValidator: 日志验证器
    LogCompressor: 日志压缩器
    DecisionAnalytics: 决策分析器
    LogExporter: 日志导出器
    LogMonitor: 日志监控器
    LogArchiver: 日志归档器
    DecisionPredictor: 决策预测器
    DecisionOptimizer: 决策优化器
    DecisionSimulator: 决策模拟器


"""

# 从DecisionLogger模块导入所有主要类和函数
from DecisionLogger import (
    # 主类
    DecisionLogger,
    
    # 枚举类
    LogLevel,
    DecisionType,
    StrategyType,
    RiskLevel,
    ExecutionStatus,
    
    # 数据类
    DecisionInput,
    DecisionLogic,
    DecisionResult,
    StrategyStep,
    RiskAssessment,
    MarketAnalysis,
    DecisionExplanation,
    EffectEvaluation,
    
    # 存储相关类
    LogStorage,
    FileStorage,
    MemoryStorage,
    
    # 核心处理器
    AsyncLogProcessor,
    
    # 功能扩展类
    LogValidator,
    LogCompressor,
    DecisionAnalytics,
    LogExporter,
    LogMonitor,
    LogArchiver,
    
    # 高级功能类
    DecisionPredictor,
    DecisionOptimizer,
    DecisionSimulator,
    
    # 工具函数
    log_execution_time,
    validate_input,
    cache_result,
    get_log_type_stats,
    create_decision_summary,
    calculate_risk_score,
    calculate_performance_score,
    generate_correlation_matrix,
    calculate_pearson_correlation,
    format_file_size,
    validate_timestamp,
    sanitize_filename,
    create_backup_name
)

# 版本信息
__version__ = "1.0.0"
__author__ = "L6系统"
__email__ = "l6@system.com"
__description__ = "L6决策日志记录器模块"

# 模块信息
__all__ = [
    # 主类
    "DecisionLogger",
    
    # 枚举类
    "LogLevel",
    "DecisionType", 
    "StrategyType",
    "RiskLevel",
    "ExecutionStatus",
    
    # 数据类
    "DecisionInput",
    "DecisionLogic",
    "DecisionResult",
    "StrategyStep",
    "RiskAssessment",
    "MarketAnalysis",
    "DecisionExplanation",
    "EffectEvaluation",
    
    # 存储相关类
    "LogStorage",
    "FileStorage",
    "MemoryStorage",
    
    # 核心处理器
    "AsyncLogProcessor",
    
    # 功能扩展类
    "LogValidator",
    "LogCompressor",
    "DecisionAnalytics",
    "LogExporter",
    "LogMonitor",
    "LogArchiver",
    
    # 高级功能类
    "DecisionPredictor",
    "DecisionOptimizer",
    "DecisionSimulator",
    
    # 工具函数
    "log_execution_time",
    "validate_input",
    "cache_result",
    "get_log_type_stats",
    "create_decision_summary",
    "calculate_risk_score",
    "calculate_performance_score",
    "generate_correlation_matrix",
    "calculate_pearson_correlation",
    "format_file_size",
    "validate_timestamp",
    "sanitize_filename",
    "create_backup_name",
    
    # 模块信息
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]

# 便利函数 - 创建默认的决策日志记录器实例
def create_logger(
    storage_type: str = "memory",
    storage_config: dict = None,
    enable_async: bool = True,
    log_level: str = "INFO",
    enable_analytics: bool = True
) -> DecisionLogger:
    """
    创建一个决策日志记录器实例
    
    Args:
        storage_type: 存储类型 ("memory", "file")
        storage_config: 存储配置
        enable_async: 是否启用异步处理
        log_level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        enable_analytics: 是否启用分析功能
        
    Returns:
        DecisionLogger: 决策日志记录器实例
    """
    level = LogLevel.INFO
    if hasattr(LogLevel, log_level):
        level = getattr(LogLevel, log_level)
    
    return DecisionLogger(
        storage_type=storage_type,
        storage_config=storage_config or {},
        enable_async=enable_async,
        log_level=level,
        enable_analytics=enable_analytics
    )


# 便利函数 - 快速记录决策日志
async def quick_log_decision(
    decision_type: str,
    outcome: str,
    success: bool,
    confidence: float = 0.0,
    value: any = 0,
    execution_time: float = 0.0,
    storage_type: str = "memory"
) -> str:
    """
    快速记录决策日志
    
    Args:
        decision_type: 决策类型 ("strategic", "tactical", "operational", "emergency", "routine")
        outcome: 决策结果描述
        success: 是否成功
        confidence: 置信度 (0.0-1.0)
        value: 决策值
        execution_time: 执行时间
        storage_type: 存储类型
        
    Returns:
        str: 日志条目ID
    """
    logger = create_logger(storage_type=storage_type)
    await logger.start()
    
    try:
        # 转换决策类型
        dt = DecisionType.OPERATIONAL
        if hasattr(DecisionType, decision_type.upper()):
            dt = getattr(DecisionType, decision_type.upper())
        
        result = DecisionResult(
            decision_type=dt,
            outcome=outcome,
            success=success,
            confidence=confidence,
            value=value,
            execution_time=execution_time
        )
        
        log_id = await logger.log_decision_result(result)
        return log_id
        
    finally:
        await logger.stop()


# 便利函数 - 批量记录决策
async def quick_batch_log_decisions(
    decisions: list,
    storage_type: str = "memory"
) -> list:
    """
    快速批量记录决策日志
    
    Args:
        decisions: 决策数据列表，每个决策包含 decision_type, outcome, success 等字段
        storage_type: 存储类型
        
    Returns:
        list: 日志条目ID列表
    """
    logger = create_logger(storage_type=storage_type)
    await logger.start()
    
    try:
        log_ids = await logger.log_batch_decisions(decisions)
        return log_ids
        
    finally:
        await logger.stop()


# 便利函数 - 快速风险评估
async def quick_log_risk(
    risk_type: str,
    level: str,
    probability: float,
    impact: float,
    description: str = "",
    mitigation: str = "",
    storage_type: str = "memory"
) -> str:
    """
    快速记录风险评估
    
    Args:
        risk_type: 风险类型
        level: 风险级别 ("low", "medium", "high", "critical")
        probability: 发生概率 (0.0-1.0)
        impact: 影响程度 (0.0-1.0)
        description: 风险描述
        mitigation: 缓解措施
        storage_type: 存储类型
        
    Returns:
        str: 日志条目ID
    """
    logger = create_logger(storage_type=storage_type)
    await logger.start()
    
    try:
        # 转换风险级别
        rl = RiskLevel.LOW
        if hasattr(RiskLevel, level.upper()):
            rl = getattr(RiskLevel, level.upper())
        
        risk = RiskAssessment(
            risk_type=risk_type,
            level=rl,
            probability=probability,
            impact=impact,
            score=probability * impact,
            description=description,
            mitigation=mitigation
        )
        
        log_id = await logger.log_risk_identification(risk, "quick_assessment")
        return log_id
        
    finally:
        await logger.stop()


# 模块初始化完成标志
_L6_MODULE_INITIALIZED = True