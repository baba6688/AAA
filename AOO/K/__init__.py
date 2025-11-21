"""
K模块 - 配置管理系统

该模块提供了一套完整的配置管理解决方案，包含多个配置处理器：
- K1: 系统配置管理
- K2: 模块配置处理
- K3: 策略配置管理
- K4: 风险配置处理
- K5: 交易配置管理
- K6: 数据配置处理
- K7: 学习配置管理
- K8: 环境配置处理
- K9: 配置状态聚合

作者：配置管理团队
版本：1.0.0
"""

# 导入所有子模块（使用try-except处理依赖问题）
try:
    from . import K1
except ImportError as e:
    print(f"Warning: Failed to import K1: {e}")

try:
    from . import K2
except ImportError as e:
    print(f"Warning: Failed to import K2: {e}")

try:
    from . import K3
except ImportError as e:
    print(f"Warning: Failed to import K3: {e}")

try:
    from . import K4
except ImportError as e:
    print(f"Warning: Failed to import K4: {e}")

try:
    from . import K5
except ImportError as e:
    print(f"Warning: Failed to import K5: {e}")

try:
    from . import K6
except ImportError as e:
    print(f"Warning: Failed to import K6: {e}")

try:
    from . import K7
except ImportError as e:
    print(f"Warning: Failed to import K7: {e}")

try:
    from . import K8
except ImportError as e:
    print(f"Warning: Failed to import K8: {e}")

try:
    from . import K9
except ImportError as e:
    print(f"Warning: Failed to import K9: {e}")

# K1 - 系统配置管理器导出
from .K1 import (
    SystemConfigurationManager,
    ExtendedSystemConfigurationManager,
    UltimateSystemConfigurationManager,
    ConfigurationValidator,
    ConfigurationCache,
    ConfigurationEncryptor,
    ConfigurationLoader,
    ConfigurationTemplate,
    ConfigurationMerger,
    ConfigurationBackupManager,
    ConfigurationHealthChecker,
    ConfigurationImporter,
    ConfigurationExporter,
    ConfigurationError,
    ConfigurationValidationError,
    ConfigurationLoadError,
    ConfigurationFormat,
    LogLevel,
    ConfigurationMetadata,
    create_config_manager,
    create_basic_config_manager,
)

# K2 - 模块配置处理器导出
from .K2 import (
    ModuleConfigurationProcessor,
    ModuleConfig,
    ConfigTemplate,
    ConfigDependency,
    ConfigValidator,
    ConfigLoader,
    ConfigHotUpdate,
    AsyncConfigProcessor,
    ConfigError,
    ConfigValidationError,
    ConfigLoadError,
    ConfigType,
    ConfigStatus,
    ConfigEvent,
)

# K3 - 策略配置管理器导出
from .K3 import (
    StrategyConfigurationManager,
    StrategyConfig,
    StrategyTemplate,
    StrategyLibrary,
    ParameterOptimizer,
    BacktestConfig,
    RiskManager,
    PerformanceEvaluator,
    VersionManager,
    ConfigError,
    ValidationError,
    StrategyType,
    OptimizationMethod,
    RiskLevel,
)

# K4 - 风险配置处理器导出
from .K4 import (
    RiskConfigurationProcessor,
    AlertSystem,
    RiskDatabase,
    RiskValidator,
    RiskType,
    RiskLevel,
    AlertSeverity,
    HedgeStrategy,
    RiskThreshold,
    RiskLimit,
    MarketRiskConfig,
    CreditRiskConfig,
    OperationalRiskConfig,
    MonitoringConfig,
    AlertConfig,
    RiskConfigurationError,
    ValidationError,
    ProcessingError,
)

# K5 - 交易配置管理器导出
from .K5 import (
    TradingConfigurationManager,
    TradingMode,
    TimeRange,
    VolumeConfig,
    FrequencyConfig,
    APIConfig,
    ExchangeConfig,
    PositionLimit,
    CapitalLimit,
    TradingLimits,
    CommissionConfig,
    SlippageConfig,
    ExecutionConfig,
    MonitoringConfig,
    ReportConfig,
    ConfigurationValidator,
    ConfigurationBackupManager,
    RiskManager,
    PerformanceAnalyzer,
)

# K6 - 数据配置处理器导出
from .K6 import (
    DataConfigurationProcessor,
    DataSourceManager,
    StorageManager,
    DataUpdateManager,
    DataQualityManager,
    SecurityManager,
    BackupManager,
    AsyncDataProcessor,
    ConfigError,
    DataSourceError,
    StorageError,
    SecurityError,
    QualityError,
    BackupError,
    AsyncProcessingError,
    DataFormat,
    CompressionType,
    StorageType,
    UpdateStrategy,
    SecurityLevel,
    LogLevel,
    DataSourceParameters,
    AuthenticationInfo,
    DataFormatConfig,
    StoragePathConfig,
    StorageStrategyConfig,
    UpdateFrequency,
    QualityValidationRule,
    EncryptionConfig,
    AccessControlConfig,
    AuditLogConfig,
    BackupConfig,
    RecoveryConfig,
    ConfigDict,
    DataSourceConfig,
    StorageConfig,
    SecurityConfig,
)

# K7 - 学习配置管理器导出
from .K7 import (
    LearningConfigurationManager,
    ConfigurationManager,
    AdvancedConfigurationManager,
    ExtendedLearningConfigurationManager,
    AsyncConfigurationProcessor,
    ConfigurationValidator,
    ConfigurationTemplate,
    TemplateManager,
    AdvancedConfigurationValidator,
    ConfigurationAnalyzer,
    ConfigurationPerformanceMonitor,
    ConfigurationOptimizer,
    MLModelConfiguration,
    TrainingConfiguration,
    ValidationConfiguration,
    EvaluationConfiguration,
    DeploymentConfiguration,
    ModelUpdateConfiguration,
    ExperimentConfiguration,
    MonitoringConfiguration,
    DataPipelineConfiguration,
    ABTestConfiguration,
    ModelRegistryConfiguration,
    FeatureStoreConfiguration,
    BaseConfiguration,
    ConfigurationError,
    ValidationError,
    DeploymentError,
    ExperimentError,
    AsyncProcessingError,
    ConfigurationStatus,
    DeploymentStatus,
    ExperimentStatus,
    ModelTypeEnum,
    create_learning_manager,
    create_extended_manager,
)

# K8 - 环境配置处理器导出（已有）

# K9 - 配置状态聚合器导出（带依赖检查）
try:
    from .K9 import (
        ConfigurationStateAggregator,
        ConfigurationMonitor,
        ConfigurationCoordinator,
        ConfigurationLifecycleManager,
        ConfigurationPerformanceTracker,
        ConfigurationHealthChecker,
        ConfigurationAlertSystem,
        ConfigurationAPI,
        ConfigurationEventHandler,
    )
except ImportError as e:
    # 如果K9导入失败，设置这些变量为None
    ConfigurationStateAggregator = None
    ConfigurationMonitor = None
    ConfigurationCoordinator = None
    ConfigurationLifecycleManager = None
    ConfigurationPerformanceTracker = None
    ConfigurationHealthChecker = None
    ConfigurationAlertSystem = None
    ConfigurationAPI = None
    ConfigurationEventHandler = None
    print(f"Warning: K9模块导入失败，部分功能可能不可用: {e}")

# 版本信息
__version__ = "1.0.0"
__author__ = "配置管理团队"

# 导出的模块和类列表
__all__ = [
    # 子模块
    "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9",
    
    # K1 - 系统配置管理器
    'SystemConfigurationManager', 'ExtendedSystemConfigurationManager', 'UltimateSystemConfigurationManager',
    'ConfigurationValidator', 'ConfigurationCache', 'ConfigurationEncryptor', 'ConfigurationLoader',
    'ConfigurationTemplate', 'ConfigurationMerger', 'ConfigurationBackupManager', 'ConfigurationHealthChecker',
    'ConfigurationImporter', 'ConfigurationExporter', 'ConfigurationError', 'ConfigurationValidationError',
    'ConfigurationLoadError', 'ConfigurationFormat', 'LogLevel', 'ConfigurationMetadata',
    'create_config_manager', 'create_basic_config_manager',
    
    # K2 - 模块配置处理器
    'ModuleConfigurationProcessor', 'ModuleConfig', 'ConfigTemplate', 'ConfigDependency',
    'ConfigValidator', 'ConfigLoader', 'ConfigHotUpdate', 'AsyncConfigProcessor',
    'ConfigError', 'ConfigValidationError', 'ConfigLoadError', 'ConfigType', 'ConfigStatus', 'ConfigEvent',
    
    # K3 - 策略配置管理器
    'StrategyConfigurationManager', 'StrategyConfig', 'StrategyTemplate', 'StrategyLibrary',
    'ParameterOptimizer', 'BacktestConfig', 'RiskManager', 'PerformanceEvaluator', 'VersionManager',
    'ConfigError', 'ValidationError', 'StrategyType', 'OptimizationMethod', 'RiskLevel',
    
    # K4 - 风险配置处理器
    'RiskConfigurationProcessor', 'AlertSystem', 'RiskDatabase', 'RiskValidator',
    'RiskType', 'RiskLevel', 'AlertSeverity', 'HedgeStrategy', 'RiskThreshold', 'RiskLimit',
    'MarketRiskConfig', 'CreditRiskConfig', 'OperationalRiskConfig', 'MonitoringConfig',
    'AlertConfig', 'RiskConfigurationError', 'ValidationError', 'ProcessingError',
    
    # K5 - 交易配置管理器
    'TradingConfigurationManager', 'TradingMode', 'TimeRange', 'VolumeConfig', 'FrequencyConfig',
    'APIConfig', 'ExchangeConfig', 'PositionLimit', 'CapitalLimit', 'TradingLimits',
    'CommissionConfig', 'SlippageConfig', 'ExecutionConfig', 'MonitoringConfig', 'ReportConfig',
    'ConfigurationValidator', 'ConfigurationBackupManager', 'RiskManager', 'PerformanceAnalyzer',
    
    # K6 - 数据配置处理器
    'DataConfigurationProcessor', 'DataSourceManager', 'StorageManager', 'DataUpdateManager',
    'DataQualityManager', 'SecurityManager', 'BackupManager', 'AsyncDataProcessor',
    'ConfigError', 'DataSourceError', 'StorageError', 'SecurityError', 'QualityError',
    'BackupError', 'AsyncProcessingError', 'DataFormat', 'CompressionType', 'StorageType',
    'UpdateStrategy', 'SecurityLevel', 'LogLevel', 'DataSourceParameters', 'AuthenticationInfo',
    'DataFormatConfig', 'StoragePathConfig', 'StorageStrategyConfig', 'UpdateFrequency',
    'QualityValidationRule', 'EncryptionConfig', 'AccessControlConfig', 'AuditLogConfig',
    'BackupConfig', 'RecoveryConfig', 'ConfigDict', 'DataSourceConfig', 'StorageConfig',
    'SecurityConfig',
    
    # K7 - 学习配置管理器
    'LearningConfigurationManager', 'ConfigurationManager', 'AdvancedConfigurationManager',
    'ExtendedLearningConfigurationManager', 'AsyncConfigurationProcessor', 'ConfigurationValidator',
    'ConfigurationTemplate', 'TemplateManager', 'AdvancedConfigurationValidator', 'ConfigurationAnalyzer',
    'ConfigurationPerformanceMonitor', 'ConfigurationOptimizer', 'MLModelConfiguration',
    'TrainingConfiguration', 'ValidationConfiguration', 'EvaluationConfiguration', 'DeploymentConfiguration',
    'ModelUpdateConfiguration', 'ExperimentConfiguration', 'MonitoringConfiguration', 'DataPipelineConfiguration',
    'ABTestConfiguration', 'ModelRegistryConfiguration', 'FeatureStoreConfiguration', 'BaseConfiguration',
    'ConfigurationError', 'ValidationError', 'DeploymentError', 'ExperimentError', 'AsyncProcessingError',
    'ConfigurationStatus', 'DeploymentStatus', 'ExperimentStatus', 'ModelTypeEnum', 'create_learning_manager',
    'create_extended_manager',
    
    # K8 - 环境配置处理器
    'EnvironmentConfigurationProcessor', 'ExtendedEnvironmentConfigurationProcessor',
    'EnvironmentType', 'EnvironmentConfig', 'ConfigurationError',
    
    # K9 - 配置状态聚合器
    'ConfigurationStateAggregator', 'ConfigurationMonitor', 'ConfigurationCoordinator',
    'ConfigurationLifecycleManager', 'ConfigurationPerformanceTracker', 'ConfigurationHealthChecker',
    'ConfigurationAlertSystem', 'ConfigurationAPI', 'ConfigurationEventHandler',
    
    "__version__", "__author__",
]