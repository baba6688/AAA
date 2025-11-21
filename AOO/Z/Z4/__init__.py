"""
Z4第三方集成接口包
提供完整的第三方服务集成功能
"""

from .ThirdPartyIntegration import (
    # 核心类
    ThirdPartyIntegration,
    IntegrationConfig,
    IntegrationMetrics,
    
    # 异常类
    ThirdPartyError,
    AuthenticationError,
    RateLimitError,
    DataTransformationError,
    
    # 组件类
    BaseDataTransformer,
    JSONDataTransformer,
    XMLDataTransformer,
    AuthenticationManager,
    APIAdapter,
    SecurityManager,
    PerformanceMonitor,
    RateLimiter,
    
    # 工具函数
    create_integration,
    get_integration
)

__version__ = "1.0.0"
__author__ = "Z4 Team"
__description__ = "Z4第三方集成接口 - 提供完整的第三方服务集成功能"

__all__ = [
    "ThirdPartyIntegration",
    "IntegrationConfig", 
    "IntegrationMetrics",
    "ThirdPartyError",
    "AuthenticationError", 
    "RateLimitError",
    "DataTransformationError",
    "BaseDataTransformer",
    "JSONDataTransformer",
    "XMLDataTransformer",
    "AuthenticationManager",
    "APIAdapter",
    "SecurityManager",
    "PerformanceMonitor", 
    "RateLimiter",
    "create_integration",
    "get_integration"
]