"""
I1 API接口管理器

该模块实现了完整的API接口管理器，提供RESTful API管理、GraphQL支持、
API版本管理、请求限流、认证、响应标准化、错误处理、异步调用、
文档生成和状态监控等功能。

主要功能：
1. RESTful API管理（GET/POST/PUT/DELETE）
2. GraphQL接口支持
3. API版本管理
4. 请求限流和认证
5. 响应格式标准化
6. 错误处理和日志记录
7. 异步API调用支持
8. API文档自动生成
9. 接口状态监控


版本: 1.0.0
日期: 2025-11-05
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import (
    Any, Dict, List, Optional, Union, Callable, 
    Tuple, Set, AsyncGenerator, Literal
)
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import aiohttp
import aiofiles
from urllib.parse import urljoin, urlparse
import re
from contextlib import asynccontextmanager


# =============================================================================
# 基础数据结构和枚举
# =============================================================================

class HTTPMethod(Enum):
    """HTTP方法枚举"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class APIStatus(Enum):
    """API状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class AuthType(Enum):
    """认证类型枚举"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    JWT = "jwt"


class RateLimitStrategy(Enum):
    """限流策略枚举"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


# =============================================================================
# 数据模型
# =============================================================================

@dataclass
class APIEndpoint:
    """API端点数据模型"""
    path: str
    method: HTTPMethod
    version: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    responses: Dict[str, Any] = field(default_factory=dict)
    auth_required: bool = False
    rate_limit: Optional[int] = None
    timeout: int = 30
    retry_count: int = 3
    middleware: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['method'] = self.method.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class APIRequest:
    """API请求数据模型"""
    url: str
    method: HTTPMethod
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None
    json_data: Optional[Dict[str, Any]] = None
    auth: Optional[Dict[str, Any]] = None
    timeout: int = 30
    verify_ssl: bool = True
    follow_redirects: bool = True
    allow_redirects: bool = True
    max_redirects: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class APIResponse:
    """API响应数据模型"""
    status_code: int
    headers: Dict[str, str]
    content: Union[str, bytes]
    json_data: Optional[Dict[str, Any]] = None
    response_time: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class APIVersion:
    """API版本数据模型"""
    version: str
    description: str = ""
    status: APIStatus = APIStatus.ACTIVE
    endpoints: List[str] = field(default_factory=list)
    deprecated_at: Optional[datetime] = None
    sunset_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.deprecated_at:
            data['deprecated_at'] = self.deprecated_at.isoformat()
        if self.sunset_at:
            data['sunset_at'] = self.sunset_at.isoformat()
        return data


# =============================================================================
# 限流器
# =============================================================================

class RateLimiter:
    """请求限流器"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60, strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW):
        """
        初始化限流器
        
        Args:
            max_requests: 最大请求数
            time_window: 时间窗口（秒）
            strategy: 限流策略
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.strategy = strategy
        
        # 不同策略的数据存储
        self.requests = deque()
        self.tokens = max_requests
        self.last_refill = time.time()
        
        # 统计信息
        self.total_requests = 0
        self.blocked_requests = 0
        
    async def acquire(self, endpoint: str = "") -> bool:
        """
        获取请求许可
        
        Args:
            endpoint: 端点标识符
            
        Returns:
            bool: 是否获得许可
        """
        now = time.time()
        
        if self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window_acquire(now)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_acquire(now)
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket_acquire(now)
        
        return True
    
    async def _fixed_window_acquire(self, now: float) -> bool:
        """固定窗口限流策略"""
        # 清理过期请求
        cutoff = now - self.time_window
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            self.total_requests += 1
            return True
        
        self.blocked_requests += 1
        return False
    
    async def _sliding_window_acquire(self, now: float) -> bool:
        """滑动窗口限流策略"""
        # 清理过期请求
        cutoff = now - self.time_window
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            self.total_requests += 1
            return True
        
        self.blocked_requests += 1
        return False
    
    async def _token_bucket_acquire(self, now: float) -> bool:
        """令牌桶限流策略"""
        # 更新令牌
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * (self.max_requests / self.time_window)
        self.tokens = min(self.max_requests, self.tokens + tokens_to_add)
        self.last_refill = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            self.total_requests += 1
            return True
        
        self.blocked_requests += 1
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": self.blocked_requests / max(1, self.total_requests),
            "current_requests": len(self.requests),
            "available_tokens": self.tokens if self.strategy == RateLimitStrategy.TOKEN_BUCKET else None
        }


# =============================================================================
# 认证管理器
# =============================================================================

class APIAuth:
    """API认证管理器"""
    
    def __init__(self):
        """初始化认证管理器"""
        self.auth_methods: Dict[str, Dict[str, Any]] = {}
        self.active_auth: Optional[str] = None
        
    def register_auth(self, name: str, auth_type: AuthType, config: Dict[str, Any]):
        """
        注册认证方法
        
        Args:
            name: 认证方法名称
            auth_type: 认证类型
            config: 认证配置
        """
        self.auth_methods[name] = {
            "type": auth_type,
            "config": config,
            "created_at": datetime.now()
        }
    
    def set_active_auth(self, name: str):
        """
        设置活跃认证方法
        
        Args:
            name: 认证方法名称
        """
        if name in self.auth_methods:
            self.active_auth = name
        else:
            raise ValueError(f"认证方法 '{name}' 不存在")
    
    async def get_auth_headers(self, endpoint: str = "") -> Dict[str, str]:
        """
        获取认证头信息
        
        Args:
            endpoint: 端点路径
            
        Returns:
            Dict[str, str]: 认证头信息
        """
        if not self.active_auth or self.active_auth not in self.auth_methods:
            return {}
        
        auth_info = self.auth_methods[self.active_auth]
        auth_type = auth_info["type"]
        config = auth_info["config"]
        
        if auth_type == AuthType.API_KEY:
            return self._get_api_key_headers(config)
        elif auth_type == AuthType.BEARER:
            return self._get_bearer_headers(config)
        elif auth_type == AuthType.BASIC:
            return self._get_basic_headers(config)
        elif auth_type == AuthType.OAUTH2:
            return await self._get_oauth2_headers(config)
        elif auth_type == AuthType.JWT:
            return self._get_jwt_headers(config)
        
        return {}
    
    def _get_api_key_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        """API Key认证头"""
        key_name = config.get("key_name", "X-API-Key")
        key_value = config.get("key_value", "")
        return {key_name: key_value}
    
    def _get_bearer_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Bearer Token认证头"""
        token = config.get("token", "")
        return {"Authorization": f"Bearer {token}"}
    
    def _get_basic_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Basic认证头"""
        import base64
        username = config.get("username", "")
        password = config.get("password", "")
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {credentials}"}
    
    async def _get_oauth2_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        """OAuth2认证头"""
        # 简化实现，实际应该处理token刷新
        access_token = config.get("access_token", "")
        return {"Authorization": f"Bearer {access_token}"}
    
    def _get_jwt_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        """JWT认证头"""
        token = config.get("token", "")
        return {"Authorization": f"Bearer {token}"}
    
    def get_available_auths(self) -> List[str]:
        """获取可用的认证方法"""
        return list(self.auth_methods.keys())


# =============================================================================
# 响应格式化器
# =============================================================================

class ResponseFormatter:
    """响应格式化器"""
    
    def __init__(self, default_format: str = "json"):
        """
        初始化响应格式化器
        
        Args:
            default_format: 默认响应格式
        """
        self.default_format = default_format
        self.formatters = {
            "json": self._format_json,
            "xml": self._format_xml,
            "csv": self._format_csv,
            "yaml": self._format_yaml
        }
    
    def format_response(self, response: APIResponse, format_type: Optional[str] = None) -> APIResponse:
        """
        格式化响应
        
        Args:
            response: 原始响应
            format_type: 格式类型
            
        Returns:
            APIResponse: 格式化后的响应
        """
        format_type = format_type or self.default_format
        
        if format_type in self.formatters:
            return self.formatters[format_type](response)
        
        return response
    
    def _format_json(self, response: APIResponse) -> APIResponse:
        """JSON格式响应"""
        if response.json_data is None and isinstance(response.content, str):
            try:
                response.json_data = json.loads(response.content)
            except json.JSONDecodeError:
                pass
        return response
    
    def _format_xml(self, response: APIResponse) -> APIResponse:
        """XML格式响应"""
        # 简化实现，实际应该使用XML解析器
        return response
    
    def _format_csv(self, response: APIResponse) -> APIResponse:
        """CSV格式响应"""
        # 简化实现，实际应该使用CSV解析器
        return response
    
    def _format_yaml(self, response: APIResponse) -> APIResponse:
        """YAML格式响应"""
        # 简化实现，实际应该使用YAML解析器
        return response
    
    def standardize_response(self, response: APIResponse) -> Dict[str, Any]:
        """
        标准化响应格式
        
        Args:
            response: API响应
            
        Returns:
            Dict[str, Any]: 标准化响应
        """
        return {
            "success": response.success,
            "status_code": response.status_code,
            "data": response.json_data if response.json_data else response.content,
            "message": response.error_message or "Success",
            "timestamp": response.timestamp.isoformat(),
            "request_id": response.request_id,
            "response_time": response.response_time
        }


# =============================================================================
# 错误处理器
# =============================================================================

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化错误处理器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = defaultdict(int)
        self.last_errors = deque(maxlen=100)
    
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> APIResponse:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文
            
        Returns:
            APIResponse: 错误响应
        """
        context = context or {}
        
        # 记录错误
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
        
        self.last_errors.append(error_info)
        self.error_counts[type(error).__name__] += 1
        
        # 记录到日志
        self.logger.error(f"API Error: {error_info}")
        
        # 根据错误类型生成响应
        if isinstance(error, aiohttp.ClientError):
            return self._handle_client_error(error, context)
        elif isinstance(error, asyncio.TimeoutError):
            return self._handle_timeout_error(error, context)
        else:
            return self._handle_generic_error(error, context)
    
    def _handle_client_error(self, error: aiohttp.ClientError, context: Dict[str, Any]) -> APIResponse:
        """处理客户端错误"""
        status_code = 500
        message = "客户端请求错误"
        
        if isinstance(error, aiohttp.ClientResponseError):
            status_code = error.status
            message = f"HTTP {status_code}: {error.message}"
        elif isinstance(error, aiohttp.ClientConnectorError):
            status_code = 503
            message = "服务连接错误"
        
        return APIResponse(
            status_code=status_code,
            headers={},
            content=message,
            success=False,
            error_message=message
        )
    
    def _handle_timeout_error(self, error: asyncio.TimeoutError, context: Dict[str, Any]) -> APIResponse:
        """处理超时错误"""
        return APIResponse(
            status_code=408,
            headers={},
            content="请求超时",
            success=False,
            error_message="请求超时"
        )
    
    def _handle_generic_error(self, error: Exception, context: Dict[str, Any]) -> APIResponse:
        """处理通用错误"""
        return APIResponse(
            status_code=500,
            headers={},
            content="内部服务器错误",
            success=False,
            error_message=str(error)
        )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        return {
            "error_counts": dict(self.error_counts),
            "total_errors": sum(self.error_counts.values()),
            "recent_errors": list(self.last_errors)[-10:]  # 最近10个错误
        }


# =============================================================================
# 异步请求处理器
# =============================================================================

class AsyncRequestHandler:
    """异步请求处理器"""
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """
        初始化异步请求处理器
        
        Args:
            session: aiohttp会话
        """
        self.session = session
        self._default_headers = {
            "User-Agent": "APIInterfaceManager/1.0.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    @asynccontextmanager
    async def get_session(self):
        """获取HTTP会话"""
        if self.session and not self.session.closed:
            yield self.session
        else:
            async with aiohttp.ClientSession() as session:
                yield session
    
    async def execute_request(self, request: APIRequest) -> APIResponse:
        """
        执行HTTP请求
        
        Args:
            request: API请求
            
        Returns:
            APIResponse: API响应
        """
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                # 构建请求参数
                kwargs = {
                    "url": request.url,
                    "headers": {**self._default_headers, **request.headers},
                    "timeout": aiohttp.ClientTimeout(total=request.timeout),
                    "ssl": request.verify_ssl,
                    "allow_redirects": request.allow_redirects,
                    "max_redirects": request.max_redirects
                }
                
                # 添加请求数据
                if request.json_data:
                    kwargs["json"] = request.json_data
                elif request.data:
                    kwargs["data"] = request.data
                
                # 添加查询参数
                if request.params:
                    kwargs["params"] = request.params
                
                # 执行请求
                async with session.request(request.method.value, **kwargs) as response:
                    # 读取响应内容
                    content = await response.read()
                    
                    # 尝试解析JSON
                    json_data = None
                    try:
                        if content:
                            json_data = json.loads(content.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    
                    # 计算响应时间
                    response_time = time.time() - start_time
                    
                    return APIResponse(
                        status_code=response.status,
                        headers=dict(response.headers),
                        content=content.decode('utf-8', errors='ignore'),
                        json_data=json_data,
                        response_time=response_time,
                        success=200 <= response.status < 300
                    )
        
        except Exception as error:
            response_time = time.time() - start_time
            return APIResponse(
                status_code=500,
                headers={},
                content=str(error),
                response_time=response_time,
                success=False,
                error_message=str(error)
            )
    
    async def execute_batch_requests(self, requests: List[APIRequest], max_concurrent: int = 10) -> List[APIResponse]:
        """
        批量执行请求
        
        Args:
            requests: 请求列表
            max_concurrent: 最大并发数
            
        Returns:
            List[APIResponse]: 响应列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(request: APIRequest) -> APIResponse:
            async with semaphore:
                return await self.execute_request(request)
        
        tasks = [execute_with_semaphore(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stream_request(self, request: APIRequest) -> AsyncGenerator[APIResponse, None]:
        """
        流式请求处理
        
        Args:
            request: API请求
            
        Yields:
            APIResponse: 流式响应
        """
        async with self.get_session() as session:
            async with session.request(request.method.value, url=request.url, headers=request.headers) as response:
                async for chunk in response.content.iter_chunked(1024):
                    yield APIResponse(
                        status_code=response.status,
                        headers=dict(response.headers),
                        content=chunk,
                        success=200 <= response.status < 300
                    )


# =============================================================================
# GraphQL处理器
# =============================================================================

class GraphQLHandler:
    """GraphQL请求处理器"""
    
    def __init__(self, request_handler: AsyncRequestHandler):
        """
        初始化GraphQL处理器
        
        Args:
            request_handler: 异步请求处理器
        """
        self.request_handler = request_handler
    
    async def execute_query(self, endpoint: str, query: str, variables: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        执行GraphQL查询
        
        Args:
            endpoint: GraphQL端点
            query: GraphQL查询语句
            variables: 查询变量
            
        Returns:
            APIResponse: GraphQL响应
        """
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        request = APIRequest(
            url=endpoint,
            method=HTTPMethod.POST,
            json_data=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return await self.request_handler.execute_request(request)
    
    async def execute_mutation(self, endpoint: str, mutation: str, variables: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        执行GraphQL变更
        
        Args:
            endpoint: GraphQL端点
            mutation: GraphQL变更语句
            variables: 变更变量
            
        Returns:
            APIResponse: GraphQL响应
        """
        return await self.execute_query(endpoint, mutation, variables)
    
    async def introspect_schema(self, endpoint: str) -> APIResponse:
        """
        获取GraphQL模式信息
        
        Args:
            endpoint: GraphQL端点
            
        Returns:
            APIResponse: 模式信息响应
        """
        introspection_query = """
        query IntrospectionQuery {
            __schema {
                queryType { name }
                mutationType { name }
                subscriptionType { name }
                types {
                    ...FullType
                }
                directives {
                    name
                    description
                    locations
                    args {
                        ...InputValue
                    }
                }
            }
        }
        
        fragment FullType on __Type {
            kind
            name
            description
            fields(includeDeprecated: true) {
                name
                description
                args {
                    ...InputValue
                }
                type {
                    ...TypeRef
                }
                isDeprecated
                deprecationReason
            }
            inputFields {
                ...InputValue
            }
            interfaces {
                ...TypeRef
            }
            enumValues(includeDeprecated: true) {
                name
                description
                isDeprecated
                deprecationReason
            }
            possibleTypes {
                ...TypeRef
            }
        }
        
        fragment InputValue on __InputValue {
            name
            description
            type { ...TypeRef }
            defaultValue
        }
        
        fragment TypeRef on __Type {
            kind
            name
            ofType {
                kind
                name
                ofType {
                    kind
                    name
                    ofType {
                        kind
                        name
                        ofType {
                            kind
                            name
                            ofType {
                                kind
                                name
                                ofType {
                                    kind
                                    name
                                    ofType {
                                        kind
                                        name
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        
        return await self.execute_query(endpoint, introspection_query)


# =============================================================================
# API文档生成器
# =============================================================================

class APIDocumentationGenerator:
    """API文档生成器"""
    
    def __init__(self, api_manager: 'APIInterfaceManager'):
        """
        初始化文档生成器
        
        Args:
            api_manager: API管理器实例
        """
        self.api_manager = api_manager
    
    def generate_openapi_spec(self, title: str = "API Documentation", version: str = "1.0.0") -> Dict[str, Any]:
        """
        生成OpenAPI规范文档
        
        Args:
            title: API标题
            version: API版本
            
        Returns:
            Dict[str, Any]: OpenAPI规范
        """
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "version": version,
                "description": "API接口文档"
            },
            "servers": [{"url": "https://api.example.com", "description": "生产环境"}],
            "paths": {},
            "components": {
                "securitySchemes": {},
                "schemas": {}
            }
        }
        
        # 生成路径文档
        for endpoint in self.api_manager.endpoints.values():
            path_item = self._generate_path_item(endpoint)
            if endpoint.path in spec["paths"]:
                spec["paths"][endpoint.path].update(path_item)
            else:
                spec["paths"][endpoint.path] = path_item
        
        # 生成安全方案
        spec["components"]["securitySchemes"] = self._generate_security_schemes()
        
        return spec
    
    def _generate_path_item(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """生成路径项文档"""
        path_item = {}
        
        operation = {
            "summary": endpoint.description or f"{endpoint.method.value} {endpoint.path}",
            "description": endpoint.description,
            "tags": endpoint.tags,
            "parameters": self._generate_parameters(endpoint.parameters),
            "responses": self._generate_responses(endpoint.responses),
            "deprecated": endpoint.deprecated
        }
        
        # 添加认证信息
        if endpoint.auth_required:
            operation["security"] = [{"BearerAuth": []}]
        
        path_item[endpoint.method.value.lower()] = operation
        
        return path_item
    
    def _generate_parameters(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成参数文档"""
        params = []
        for name, param in parameters.items():
            params.append({
                "name": name,
                "in": param.get("in", "query"),
                "description": param.get("description", ""),
                "required": param.get("required", False),
                "schema": param.get("schema", {"type": "string"})
            })
        return params
    
    def _generate_responses(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """生成响应文档"""
        response_docs = {
            "200": {
                "description": "成功",
                "content": {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                }
            },
            "400": {
                "description": "请求错误"
            },
            "500": {
                "description": "服务器错误"
            }
        }
        
        for status_code, response in responses.items():
            response_docs[status_code] = {
                "description": response.get("description", ""),
                "content": {
                    "application/json": {
                        "schema": response.get("schema", {"type": "object"})
                    }
                }
            }
        
        return response_docs
    
    def _generate_security_schemes(self) -> Dict[str, Any]:
        """生成安全方案"""
        return {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            }
        }
    
    async def save_documentation(self, output_path: str, format: str = "json"):
        """
        保存文档到文件
        
        Args:
            output_path: 输出路径
            format: 文档格式 (json/yaml)
        """
        spec = self.generate_openapi_spec()
        
        if format.lower() == "json":
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(spec, indent=2, ensure_ascii=False))
        elif format.lower() == "yaml":
            import yaml
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(yaml.dump(spec, default_flow_style=False, allow_unicode=True))
        else:
            raise ValueError(f"不支持的文档格式: {format}")


# =============================================================================
# API监控器
# =============================================================================

class APIMonitor:
    """API状态监控器"""
    
    def __init__(self, api_manager: 'APIInterfaceManager'):
        """
        初始化API监控器
        
        Args:
            api_manager: API管理器实例
        """
        self.api_manager = api_manager
        self.metrics = defaultdict(list)
        self.alerts = []
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval: int = 60):
        """
        开始监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
    
    async def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                await self._check_all_endpoints()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"监控循环错误: {e}")
                await asyncio.sleep(interval)
    
    async def _check_all_endpoints(self):
        """检查所有端点"""
        for endpoint_id, endpoint in self.api_manager.endpoints.items():
            try:
                # 创建健康检查请求
                health_request = APIRequest(
                    url=endpoint.path,
                    method=HTTPMethod.GET,
                    timeout=endpoint.timeout
                )
                
                # 执行检查
                response = await self.api_manager.request_handler.execute_request(health_request)
                
                # 记录指标
                self._record_metrics(endpoint_id, response)
                
                # 检查告警条件
                self._check_alerts(endpoint_id, response)
                
            except Exception as e:
                logging.error(f"检查端点 {endpoint_id} 失败: {e}")
                self._record_error_metrics(endpoint_id, str(e))
    
    def _record_metrics(self, endpoint_id: str, response: APIResponse):
        """记录指标"""
        self.metrics[f"{endpoint_id}_response_time"].append({
            "timestamp": datetime.now().isoformat(),
            "value": response.response_time
        })
        
        self.metrics[f"{endpoint_id}_status_code"].append({
            "timestamp": datetime.now().isoformat(),
            "value": response.status_code
        })
        
        # 保持最近1000个数据点
        for key in self.metrics:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-1000:]
    
    def _record_error_metrics(self, endpoint_id: str, error: str):
        """记录错误指标"""
        self.metrics[f"{endpoint_id}_errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": error
        })
    
    def _check_alerts(self, endpoint_id: str, response: APIResponse):
        """检查告警条件"""
        # 检查响应时间
        if response.response_time > 5.0:  # 5秒告警
            self._create_alert(endpoint_id, "high_response_time", 
                             f"响应时间过长: {response.response_time:.2f}s")
        
        # 检查状态码
        if not response.success:
            self._create_alert(endpoint_id, "request_failed", 
                             f"请求失败: HTTP {response.status_code}")
    
    def _create_alert(self, endpoint_id: str, alert_type: str, message: str):
        """创建告警"""
        alert = {
            "id": str(uuid.uuid4()),
            "endpoint_id": endpoint_id,
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        
        # 保持最近100个告警
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logging.warning(f"API告警 [{endpoint_id}]: {message}")
    
    def get_metrics(self, endpoint_id: Optional[str] = None, metric_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取监控指标
        
        Args:
            endpoint_id: 端点ID
            metric_type: 指标类型
            
        Returns:
            Dict[str, Any]: 指标数据
        """
        if endpoint_id:
            return {k: v for k, v in self.metrics.items() if k.startswith(endpoint_id)}
        elif metric_type:
            return {k: v for k, v in self.metrics.items() if k.endswith(metric_type)}
        else:
            return dict(self.metrics)
    
    def get_alerts(self, acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        获取告警列表
        
        Args:
            acknowledged: 是否已确认
            
        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        if acknowledged is None:
            return self.alerts
        return [alert for alert in self.alerts if alert["acknowledged"] == acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        确认告警
        
        Args:
            alert_id: 告警ID
            
        Returns:
            bool: 是否成功确认
        """
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                return True
        return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        total_endpoints = len(self.api_manager.endpoints)
        recent_alerts = [a for a in self.alerts if 
                        datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(hours=1)]
        
        return {
            "total_endpoints": total_endpoints,
            "recent_alerts": len(recent_alerts),
            "monitoring_active": self.monitoring_active,
            "last_check": datetime.now().isoformat(),
            "status": "healthy" if len(recent_alerts) == 0 else "degraded"
        }


# =============================================================================
# 主要的API接口管理器类
# =============================================================================

class APIInterfaceManager:
    """
    API接口管理器
    
    提供完整的API接口管理功能，包括RESTful API管理、GraphQL支持、
    API版本管理、请求限流、认证、响应标准化、错误处理、异步调用、
    文档生成和状态监控等功能。
    """
    
    def __init__(self, base_url: str = "", default_timeout: int = 30, max_concurrent: int = 10):
        """
        初始化API接口管理器
        
        Args:
            base_url: 基础URL
            default_timeout: 默认超时时间
            max_concurrent: 最大并发数
        """
        self.base_url = base_url
        self.default_timeout = default_timeout
        self.max_concurrent = max_concurrent
        
        # 核心组件
        self.request_handler = AsyncRequestHandler()
        self.graphql_handler = GraphQLHandler(self.request_handler)
        self.rate_limiter = RateLimiter()
        self.auth_manager = APIAuth()
        self.response_formatter = ResponseFormatter()
        self.error_handler = ErrorHandler()
        self.documentation_generator = APIDocumentationGenerator(self)
        self.monitor = APIMonitor(self)
        
        # 数据存储
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.versions: Dict[str, APIVersion] = {}
        self.request_history: List[APIResponse] = []
        
        # 配置
        self.config = {
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_timeout": 60,
            "enable_monitoring": True,
            "enable_caching": False,
            "cache_ttl": 300
        }
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "start_time": datetime.now()
        }
        
        # 设置日志
        self._setup_logging()
        
        # 初始化默认版本
        self._init_default_version()
    
    def _setup_logging(self):
        """设置日志记录"""
        self.logger = logging.getLogger(f"{__name__}.APIInterfaceManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _init_default_version(self):
        """初始化默认版本"""
        default_version = APIVersion(
            version="v1",
            description="默认API版本",
            status=APIStatus.ACTIVE
        )
        self.versions["v1"] = default_version
    
    # =============================================================================
    # 端点管理
    # =============================================================================
    
    def register_endpoint(self, endpoint: APIEndpoint) -> str:
        """
        注册API端点
        
        Args:
            endpoint: API端点对象
            
        Returns:
            str: 端点ID
        """
        endpoint_id = f"{endpoint.method.value}_{endpoint.path}"
        
        # 检查版本是否存在
        if endpoint.version not in self.versions:
            self.versions[endpoint.version] = APIVersion(
                version=endpoint.version,
                status=APIStatus.ACTIVE
            )
        
        # 添加端点到版本
        self.versions[endpoint.version].endpoints.append(endpoint_id)
        
        # 注册端点
        self.endpoints[endpoint_id] = endpoint
        
        self.logger.info(f"注册端点: {endpoint_id}")
        return endpoint_id
    
    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """
        注销API端点
        
        Args:
            endpoint_id: 端点ID
            
        Returns:
            bool: 是否成功注销
        """
        if endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            
            # 从版本中移除
            if endpoint.version in self.versions:
                if endpoint_id in self.versions[endpoint.version].endpoints:
                    self.versions[endpoint.version].endpoints.remove(endpoint_id)
            
            # 注销端点
            del self.endpoints[endpoint_id]
            
            self.logger.info(f"注销端点: {endpoint_id}")
            return True
        
        return False
    
    def get_endpoint(self, endpoint_id: str) -> Optional[APIEndpoint]:
        """
        获取API端点
        
        Args:
            endpoint_id: 端点ID
            
        Returns:
            Optional[APIEndpoint]: API端点对象
        """
        return self.endpoints.get(endpoint_id)
    
    def list_endpoints(self, version: Optional[str] = None, method: Optional[HTTPMethod] = None) -> List[APIEndpoint]:
        """
        列出API端点
        
        Args:
            version: API版本
            method: HTTP方法
            
        Returns:
            List[APIEndpoint]: 端点列表
        """
        endpoints = list(self.endpoints.values())
        
        if version:
            endpoints = [ep for ep in endpoints if ep.version == version]
        
        if method:
            endpoints = [ep for ep in endpoints if ep.method == method]
        
        return endpoints
    
    # =============================================================================
    # 版本管理
    # =============================================================================
    
    def create_version(self, version: APIVersion) -> bool:
        """
        创建API版本
        
        Args:
            version: API版本对象
            
        Returns:
            bool: 是否成功创建
        """
        if version.version in self.versions:
            return False
        
        self.versions[version.version] = version
        self.logger.info(f"创建API版本: {version.version}")
        return True
    
    def deprecate_version(self, version: str, sunset_date: Optional[datetime] = None) -> bool:
        """
        废弃API版本
        
        Args:
            version: API版本
            sunset_date: 停用日期
            
        Returns:
            bool: 是否成功废弃
        """
        if version not in self.versions:
            return False
        
        version_obj = self.versions[version]
        version_obj.status = APIStatus.DEPRECATED
        version_obj.deprecated_at = datetime.now()
        version_obj.sunset_at = sunset_date
        
        self.logger.info(f"废弃API版本: {version}")
        return True
    
    def get_version_info(self, version: str) -> Optional[APIVersion]:
        """
        获取版本信息
        
        Args:
            version: API版本
            
        Returns:
            Optional[APIVersion]: 版本信息
        """
        return self.versions.get(version)
    
    def list_versions(self) -> List[APIVersion]:
        """
        列出所有版本
        
        Returns:
            List[APIVersion]: 版本列表
        """
        return list(self.versions.values())
    
    # =============================================================================
    # 认证管理
    # =============================================================================
    
    def setup_api_key_auth(self, name: str, key_name: str, key_value: str):
        """
        设置API Key认证
        
        Args:
            name: 认证方法名称
            key_name: 密钥参数名
            key_value: 密钥值
        """
        self.auth_manager.register_auth(name, AuthType.API_KEY, {
            "key_name": key_name,
            "key_value": key_value
        })
    
    def setup_bearer_auth(self, name: str, token: str):
        """
        设置Bearer Token认证
        
        Args:
            name: 认证方法名称
            token: Bearer Token
        """
        self.auth_manager.register_auth(name, AuthType.BEARER, {
            "token": token
        })
    
    def setup_basic_auth(self, name: str, username: str, password: str):
        """
        设置Basic认证
        
        Args:
            name: 认证方法名称
            username: 用户名
            password: 密码
        """
        self.auth_manager.register_auth(name, AuthType.BASIC, {
            "username": username,
            "password": password
        })
    
    def set_active_auth(self, name: str):
        """
        设置活跃认证方法
        
        Args:
            name: 认证方法名称
        """
        self.auth_manager.set_active_auth(name)
    
    # =============================================================================
    # 请求执行
    # =============================================================================
    
    async def request(self, 
                     endpoint_id: str,
                     params: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None,
                     json_data: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     timeout: Optional[int] = None,
                     retries: Optional[int] = None) -> APIResponse:
        """
        执行API请求
        
        Args:
            endpoint_id: 端点ID
            params: 查询参数
            data: 表单数据
            json_data: JSON数据
            headers: 请求头
            timeout: 超时时间
            retries: 重试次数
            
        Returns:
            APIResponse: API响应
        """
        # 获取端点信息
        endpoint = self.get_endpoint(endpoint_id)
        if not endpoint:
            raise ValueError(f"端点不存在: {endpoint_id}")
        
        # 检查版本状态
        version_info = self.get_version_info(endpoint.version)
        if version_info and version_info.status == APIStatus.DEPRECATED:
            self.logger.warning(f"使用已废弃的API版本: {endpoint.version}")
        
        # 检查限流
        if not await self.rate_limiter.acquire(endpoint_id):
            raise Exception(f"请求被限流: {endpoint_id}")
        
        # 构建请求URL
        url = urljoin(self.base_url, endpoint.path)
        
        # 获取认证头
        auth_headers = await self.auth_manager.get_auth_headers(endpoint.path)
        
        # 合并请求头
        request_headers = {**auth_headers, **(headers or {})}
        
        # 创建请求对象
        request = APIRequest(
            url=url,
            method=endpoint.method,
            headers=request_headers,
            params=params or {},
            data=data,
            json_data=json_data,
            timeout=timeout or endpoint.timeout
        )
        
        # 执行请求（带重试）
        retry_count = retries or self.config["retry_attempts"]
        last_error = None
        
        for attempt in range(retry_count + 1):
            try:
                response = await self.request_handler.execute_request(request)
                
                # 记录统计信息
                self._update_stats(response)
                
                # 格式化响应
                formatted_response = self.response_formatter.format_response(response)
                
                # 添加到历史记录
                self.request_history.append(formatted_response)
                
                # 保持最近1000个记录
                if len(self.request_history) > 1000:
                    self.request_history = self.request_history[-1000:]
                
                return formatted_response
                
            except Exception as error:
                last_error = error
                if attempt < retry_count:
                    delay = self.config["retry_delay"] * (2 ** attempt)  # 指数退避
                    await asyncio.sleep(delay)
                    continue
                break
        
        # 所有重试都失败，处理错误
        return await self.error_handler.handle_error(last_error, {
            "endpoint_id": endpoint_id,
            "attempt": retry_count + 1
        })
    
    def _update_stats(self, response: APIResponse):
        """更新统计信息"""
        self.stats["total_requests"] += 1
        
        if response.success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # 更新平均响应时间
        total_time = (self.stats["average_response_time"] * 
                     (self.stats["total_requests"] - 1) + response.response_time)
        self.stats["average_response_time"] = total_time / self.stats["total_requests"]
    
    # =============================================================================
    # 便捷方法
    # =============================================================================
    
    async def get(self, endpoint_id: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """
        执行GET请求
        
        Args:
            endpoint_id: 端点ID
            params: 查询参数
            **kwargs: 其他参数
            
        Returns:
            APIResponse: API响应
        """
        return await self.request(endpoint_id, params=params, **kwargs)
    
    async def post(self, endpoint_id: str, json_data: Optional[Dict[str, Any]] = None, 
                   data: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """
        执行POST请求
        
        Args:
            endpoint_id: 端点ID
            json_data: JSON数据
            data: 表单数据
            **kwargs: 其他参数
            
        Returns:
            APIResponse: API响应
        """
        return await self.request(endpoint_id, json_data=json_data, data=data, **kwargs)
    
    async def put(self, endpoint_id: str, json_data: Optional[Dict[str, Any]] = None,
                  data: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """
        执行PUT请求
        
        Args:
            endpoint_id: 端点ID
            json_data: JSON数据
            data: 表单数据
            **kwargs: 其他参数
            
        Returns:
            APIResponse: API响应
        """
        return await self.request(endpoint_id, json_data=json_data, data=data, **kwargs)
    
    async def delete(self, endpoint_id: str, **kwargs) -> APIResponse:
        """
        执行DELETE请求
        
        Args:
            endpoint_id: 端点ID
            **kwargs: 其他参数
            
        Returns:
            APIResponse: API响应
        """
        return await self.request(endpoint_id, **kwargs)
    
    # =============================================================================
    # GraphQL支持
    # =============================================================================
    
    async def graphql_query(self, endpoint: str, query: str, 
                           variables: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        执行GraphQL查询
        
        Args:
            endpoint: GraphQL端点
            query: GraphQL查询语句
            variables: 查询变量
            
        Returns:
            APIResponse: GraphQL响应
        """
        return await self.graphql_handler.execute_query(endpoint, query, variables)
    
    async def graphql_mutation(self, endpoint: str, mutation: str,
                              variables: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        执行GraphQL变更
        
        Args:
            endpoint: GraphQL端点
            mutation: GraphQL变更语句
            variables: 变更变量
            
        Returns:
            APIResponse: GraphQL响应
        """
        return await self.graphql_handler.execute_mutation(endpoint, mutation, variables)
    
    async def graphql_introspect(self, endpoint: str) -> APIResponse:
        """
        获取GraphQL模式信息
        
        Args:
            endpoint: GraphQL端点
            
        Returns:
            APIResponse: 模式信息响应
        """
        return await self.graphql_handler.introspect_schema(endpoint)
    
    # =============================================================================
    # 批量操作
    # =============================================================================
    
    async def batch_request(self, requests: List[Dict[str, Any]]) -> List[APIResponse]:
        """
        批量执行请求
        
        Args:
            requests: 请求配置列表
            
        Returns:
            List[APIResponse]: 响应列表
        """
        api_requests = []
        
        for req_config in requests:
            endpoint_id = req_config["endpoint_id"]
            endpoint = self.get_endpoint(endpoint_id)
            
            if not endpoint:
                # 创建错误响应
                error_response = APIResponse(
                    status_code=400,
                    headers={},
                    content=f"端点不存在: {endpoint_id}",
                    success=False,
                    error_message=f"端点不存在: {endpoint_id}"
                )
                api_requests.append(error_response)
                continue
            
            # 构建请求URL
            url = urljoin(self.base_url, endpoint.path)
            
            # 获取认证头
            auth_headers = await self.auth_manager.get_auth_headers(endpoint.path)
            
            # 创建请求对象
            request = APIRequest(
                url=url,
                method=endpoint.method,
                headers={**auth_headers, **req_config.get("headers", {})},
                params=req_config.get("params", {}),
                data=req_config.get("data"),
                json_data=req_config.get("json_data"),
                timeout=req_config.get("timeout", endpoint.timeout)
            )
            
            api_requests.append(request)
        
        # 执行批量请求
        responses = await self.request_handler.execute_batch_requests(
            api_requests, self.max_concurrent
        )
        
        # 格式化响应
        formatted_responses = []
        for response in responses:
            if isinstance(response, APIResponse):
                formatted_response = self.response_formatter.format_response(response)
                formatted_responses.append(formatted_response)
                
                # 更新统计信息
                self._update_stats(formatted_response)
            else:
                # 处理异常
                error_response = APIResponse(
                    status_code=500,
                    headers={},
                    content=str(response),
                    success=False,
                    error_message=str(response)
                )
                formatted_responses.append(error_response)
        
        return formatted_responses
    
    # =============================================================================
    # 文档生成
    # =============================================================================
    
    def generate_api_docs(self, title: str = "API Documentation", 
                         version: str = "1.0.0") -> Dict[str, Any]:
        """
        生成API文档
        
        Args:
            title: 文档标题
            version: 文档版本
            
        Returns:
            Dict[str, Any]: OpenAPI规范文档
        """
        return self.documentation_generator.generate_openapi_spec(title, version)
    
    async def save_api_docs(self, output_path: str, format: str = "json"):
        """
        保存API文档
        
        Args:
            output_path: 输出路径
            format: 文档格式 (json/yaml)
        """
        await self.documentation_generator.save_documentation(output_path, format)
    
    # =============================================================================
    # 监控和统计
    # =============================================================================
    
    async def start_monitoring(self, interval: int = 60):
        """
        开始监控
        
        Args:
            interval: 监控间隔（秒）
        """
        await self.monitor.start_monitoring(interval)
    
    async def stop_monitoring(self):
        """停止监控"""
        await self.monitor.stop_monitoring()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            **self.stats,
            "rate_limiter_stats": self.rate_limiter.get_stats(),
            "auth_methods": self.auth_manager.get_available_auths(),
            "endpoint_count": len(self.endpoints),
            "version_count": len(self.versions),
            "request_history_size": len(self.request_history)
        }
    
    def get_request_history(self, limit: int = 100) -> List[APIResponse]:
        """
        获取请求历史
        
        Args:
            limit: 返回记录数限制
            
        Returns:
            List[APIResponse]: 请求历史
        """
        return self.request_history[-limit:]
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """
        获取监控指标
        
        Returns:
            Dict[str, Any]: 监控指标
        """
        return {
            "metrics": self.monitor.get_metrics(),
            "alerts": self.monitor.get_alerts(),
            "health_status": self.monitor.get_health_status()
        }
    
    # =============================================================================
    # 配置管理
    # =============================================================================
    
    def update_config(self, config: Dict[str, Any]):
        """
        更新配置
        
        Args:
            config: 配置字典
        """
        self.config.update(config)
        self.logger.info(f"更新配置: {config}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取配置
        
        Returns:
            Dict[str, Any]: 当前配置
        """
        return self.config.copy()
    
    # =============================================================================
    # 清理和关闭
    # =============================================================================
    
    async def close(self):
        """关闭管理器"""
        # 停止监控
        await self.stop_monitoring()
        
        # 关闭HTTP会话
        if self.request_handler.session:
            await self.request_handler.session.close()
        
        self.logger.info("API接口管理器已关闭")
    
    def __enter__(self):
        """同步上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """同步上下文管理器出口"""
        asyncio.create_task(self.close())
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()


# =============================================================================
# 测试用例
# =============================================================================

async def test_api_interface_manager():
    """测试API接口管理器"""
    
    print("=== API接口管理器测试 ===\n")
    
    # 创建API管理器
    api_manager = APIInterfaceManager(
        base_url="https://jsonplaceholder.typicode.com",
        default_timeout=30
    )
    
    try:
        # 1. 测试端点注册
        print("1. 测试端点注册...")
        
        # 注册GET端点
        get_endpoint = APIEndpoint(
            path="/posts/1",
            method=HTTPMethod.GET,
            version="v1",
            description="获取单个文章",
            auth_required=False
        )
        
        endpoint_id = api_manager.register_endpoint(get_endpoint)
        print(f"   注册端点ID: {endpoint_id}")
        
        # 注册POST端点
        post_endpoint = APIEndpoint(
            path="/posts",
            method=HTTPMethod.POST,
            version="v1",
            description="创建文章",
            auth_required=False
        )
        
        post_endpoint_id = api_manager.register_endpoint(post_endpoint)
        print(f"   注册端点ID: {post_endpoint_id}")
        
        # 2. 测试认证设置
        print("\n2. 测试认证设置...")
        
        # 设置API Key认证
        api_manager.setup_api_key_auth("api_key_auth", "X-API-Key", "test-api-key")
        api_manager.set_active_auth("api_key_auth")
        print("   API Key认证设置完成")
        
        # 3. 测试GET请求
        print("\n3. 测试GET请求...")
        
        response = await api_manager.get(endpoint_id)
        print(f"   状态码: {response.status_code}")
        print(f"   响应时间: {response.response_time:.2f}s")
        print(f"   成功: {response.success}")
        
        if response.json_data:
            print(f"   数据预览: {json.dumps(response.json_data, indent=2, ensure_ascii=False)[:200]}...")
        
        # 4. 测试POST请求
        print("\n4. 测试POST请求...")
        
        post_data = {
            "title": "测试文章",
            "body": "这是一个测试文章的内容",
            "userId": 1
        }
        
        post_response = await api_manager.post(post_endpoint_id, json_data=post_data)
        print(f"   状态码: {post_response.status_code}")
        print(f"   响应时间: {post_response.response_time:.2f}s")
        print(f"   成功: {post_response.success}")
        
        if post_response.json_data:
            print(f"   创建的文章ID: {post_response.json_data.get('id')}")
        
        # 5. 测试GraphQL
        print("\n5. 测试GraphQL...")
        
        # 使用公开的GraphQL API进行测试
        graphql_query = """
        query {
            countries {
                code
                name
                continent {
                    name
                }
            }
        }
        """
        
        # 注意：这里使用了一个公开的GraphQL端点
        try:
            graphql_response = await api_manager.graphql_query(
                "https://countries.trevorblades.com/", 
                graphql_query
            )
            print(f"   GraphQL状态码: {graphql_response.status_code}")
            print(f"   GraphQL成功: {graphql_response.success}")
        except Exception as e:
            print(f"   GraphQL测试跳过: {e}")
        
        # 6. 测试批量请求
        print("\n6. 测试批量请求...")
        
        batch_requests = [
            {
                "endpoint_id": endpoint_id,
                "params": {"_limit": "5"}
            }
        ]
        
        batch_responses = await api_manager.batch_request(batch_requests)
        print(f"   批量请求数量: {len(batch_responses)}")
        print(f"   第一个响应状态码: {batch_responses[0].status_code}")
        
        # 7. 测试版本管理
        print("\n7. 测试版本管理...")
        
        # 创建新版本
        v2_version = APIVersion(
            version="v2",
            description="API第二版本",
            status=APIStatus.ACTIVE
        )
        
        api_manager.create_version(v2_version)
        print(f"   创建版本: v2")
        
        # 列出所有版本
        versions = api_manager.list_versions()
        print(f"   可用版本: {[v.version for v in versions]}")
        
        # 8. 测试文档生成
        print("\n8. 测试文档生成...")
        
        docs = api_manager.generate_api_docs("测试API文档", "1.0.0")
        print(f"   文档标题: {docs['info']['title']}")
        print(f"   文档路径数量: {len(docs['paths'])}")
        
        # 9. 测试监控
        print("\n9. 测试监控...")
        
        # 启动监控
        await api_manager.start_monitoring(interval=30)
        print("   监控已启动")
        
        # 获取统计信息
        stats = api_manager.get_stats()
        print(f"   总请求数: {stats['total_requests']}")
        print(f"   成功请求数: {stats['successful_requests']}")
        print(f"   平均响应时间: {stats['average_response_time']:.2f}s")
        
        # 获取监控指标
        metrics = api_manager.get_monitoring_metrics()
        print(f"   监控状态: {metrics['health_status']['status']}")
        
        # 10. 测试限流
        print("\n10. 测试限流...")
        
        # 设置更严格的限流
        api_manager.rate_limiter = RateLimiter(max_requests=5, time_window=10)
        
        # 发送多个请求来测试限流
        for i in range(7):
            try:
                response = await api_manager.get(endpoint_id)
                print(f"   请求 {i+1}: {'成功' if response.success else '失败'}")
            except Exception as e:
                print(f"   请求 {i+1}: 限流 - {str(e)[:50]}...")
        
        # 获取限流统计
        rate_stats = api_manager.rate_limiter.get_stats()
        print(f"   限流统计: 总请求 {rate_stats['total_requests']}, "
              f"被阻止 {rate_stats['blocked_requests']}, "
              f"阻止率 {rate_stats['block_rate']:.2%}")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        await api_manager.close()
        print("\n资源清理完成")


# =============================================================================
# 使用示例
# =============================================================================

async def example_usage():
    """使用示例"""
    
    print("=== API接口管理器使用示例 ===\n")
    
    # 创建API管理器
    async with APIInterfaceManager(base_url="https://api.github.com") as api_manager:
        
        # 1. 设置认证
        print("1. 设置GitHub API认证...")
        # 注意：实际使用时需要真实的GitHub token
        # api_manager.setup_bearer_auth("github_auth", "your-github-token")
        # api_manager.set_active_auth("github_auth")
        
        # 2. 注册端点
        print("2. 注册GitHub API端点...")
        
        # 获取用户信息端点
        user_endpoint = APIEndpoint(
            path="/user",
            method=HTTPMethod.GET,
            version="v3",
            description="获取当前用户信息",
            auth_required=True
        )
        
        user_endpoint_id = api_manager.register_endpoint(user_endpoint)
        
        # 获取仓库列表端点
        repos_endpoint = APIEndpoint(
            path="/user/repos",
            method=HTTPMethod.GET,
            version="v3",
            description="获取用户仓库列表",
            auth_required=True,
            params={
                "sort": {"in": "query", "description": "排序方式", "schema": {"type": "string"}},
                "per_page": {"in": "query", "description": "每页数量", "schema": {"type": "integer"}}
            }
        )
        
        repos_endpoint_id = api_manager.register_endpoint(repos_endpoint)
        
        # 3. 执行请求
        print("3. 执行API请求...")
        
        try:
            # 获取仓库列表
            response = await api_manager.get(repos_endpoint_id, params={
                "sort": "updated",
                "per_page": 5
            })
            
            print(f"   状态码: {response.status_code}")
            print(f"   响应时间: {response.response_time:.2f}s")
            
            if response.success and response.json_data:
                repos = response.json_data
                print(f"   仓库数量: {len(repos)}")
                if repos:
                    print(f"   最新仓库: {repos[0]['name']}")
            
        except Exception as e:
            print(f"   请求失败（预期，因为没有真实认证）: {e}")
        
        # 4. 生成文档
        print("\n4. 生成API文档...")
        
        docs = api_manager.generate_api_docs("GitHub API客户端", "1.0.0")
        print(f"   文档包含 {len(docs['paths'])} 个端点")
        
        # 5. 查看统计信息
        print("\n5. 查看统计信息...")
        
        stats = api_manager.get_stats()
        print(f"   注册端点数: {stats['endpoint_count']}")
        print(f"   API版本数: {stats['version_count']}")
        print(f"   请求历史记录: {stats['request_history_size']}")
        
        # 6. 启动监控
        print("\n6. 启动监控...")
        
        await api_manager.start_monitoring(interval=60)
        print("   监控已启动，60秒间隔")
        
        # 等待一段时间来收集监控数据
        print("   等待监控数据收集...")
        await asyncio.sleep(5)
        
        # 获取监控状态
        monitoring = api_manager.get_monitoring_metrics()
        print(f"   监控状态: {monitoring['health_status']['status']}")
        
    print("\n=== 示例完成 ===")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        """主函数"""
        print("API接口管理器演示\n")
        
        # 运行测试
        await test_api_interface_manager()
        
        print("\n" + "="*50 + "\n")
        
        # 运行示例
        await example_usage()
    
    # 运行主程序
    asyncio.run(main())