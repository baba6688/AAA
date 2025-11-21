"""
I1模块 - API接口管理器

该模块提供完整的API接口管理功能，包括：
- RESTful API管理
- GraphQL接口支持
- API版本管理
- 请求限流和认证
- 响应格式标准化
- 错误处理和日志记录
- 异步API调用支持
- API文档自动生成
- 接口状态监控
"""

from .APIInterfaceManager import (
    APIInterfaceManager,
    APIEndpoint,
    APIVersion,
    RateLimiter,
    APIAuth,
    ResponseFormatter,
    ErrorHandler,
    AsyncRequestHandler,
    APIDocumentationGenerator,
    APIMonitor,
    APIResponse,
    APIRequest,
    GraphQLHandler,
    APIStatus
)

__version__ = "1.0.0"
__all__ = [
    "APIInterfaceManager",
    "APIEndpoint", 
    "APIVersion",
    "RateLimiter",
    "APIAuth",
    "ResponseFormatter",
    "ErrorHandler",
    "AsyncRequestHandler",
    "APIDocumentationGenerator",
    "APIMonitor",
    "APIResponse",
    "APIRequest",
    "GraphQLHandler",
    "APIStatus"
]