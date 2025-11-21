"""
Z4第三方集成接口模块
提供完整的第三方服务集成功能，包括API适配、数据转换、认证管理等功能
"""

import json
import time
import logging
import hashlib
import base64
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """第三方集成配置"""
    service_name: str
    base_url: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    rate_limit: int = 100  # 每分钟请求限制
    headers: Dict[str, str] = field(default_factory=dict)
    auth_type: str = "bearer"  # bearer, basic, oauth2, api_key
    data_format: str = "json"  # json, xml, form


@dataclass
class IntegrationMetrics:
    """集成性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    rate_limit_hits: int = 0


class ThirdPartyError(Exception):
    """第三方服务错误基类"""
    def __init__(self, message: str, error_code: str = None, status_code: int = None):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationError(ThirdPartyError):
    """认证错误"""
    pass


class RateLimitError(ThirdPartyError):
    """速率限制错误"""
    pass


class DataTransformationError(ThirdPartyError):
    """数据转换错误"""
    pass


class BaseDataTransformer(ABC):
    """数据转换器基类"""
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """转换数据"""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: Any) -> Any:
        """反向转换数据"""
        pass


class JSONDataTransformer(BaseDataTransformer):
    """JSON数据转换器"""
    
    def transform(self, data: Any) -> Any:
        """转换为JSON格式"""
        try:
            if isinstance(data, str):
                return json.loads(data)
            return data
        except (json.JSONDecodeError, TypeError) as e:
            raise DataTransformationError(f"JSON转换失败: {str(e)}")
    
    def inverse_transform(self, data: Any) -> str:
        """转换为JSON字符串"""
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            raise DataTransformationError(f"JSON反向转换失败: {str(e)}")


class XMLDataTransformer(BaseDataTransformer):
    """XML数据转换器"""
    
    def transform(self, data: Any) -> Any:
        """转换XML数据为字典"""
        try:
            import xml.etree.ElementTree as ET
            if isinstance(data, str):
                root = ET.fromstring(data)
                return self._element_to_dict(root)
            return data
        except Exception as e:
            raise DataTransformationError(f"XML转换失败: {str(e)}")
    
    def inverse_transform(self, data: Any) -> str:
        """转换字典为XML"""
        try:
            import xml.etree.ElementTree as ET
            if isinstance(data, dict):
                root = ET.Element("root")
                self._dict_to_element(data, root)
                return ET.tostring(root, encoding='unicode')
            return str(data)
        except Exception as e:
            raise DataTransformationError(f"XML反向转换失败: {str(e)}")
    
    def _element_to_dict(self, element):
        """将XML元素转换为字典"""
        result = {}
        if element.text:
            result['text'] = element.text
        if element.attrib:
            result['attributes'] = element.attrib
        for child in element:
            child_data = self._element_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        return result
    
    def _dict_to_element(self, data, element):
        """将字典转换为XML元素"""
        if isinstance(data, dict):
            for key, value in data.items():
                child = element
                if key != 'text' and key != 'attributes':
                    child = ET.SubElement(element, key)
                if key == 'attributes' and isinstance(value, dict):
                    element.attrib.update(value)
                elif key == 'text':
                    element.text = str(value)
                else:
                    self._dict_to_element(value, child)
        elif isinstance(data, list):
            for item in data:
                self._dict_to_element(item, element)
        else:
            element.text = str(data)


class AuthenticationManager:
    """认证管理器"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._token_cache = {}
        self._token_expires = {}
    
    def get_auth_headers(self) -> Dict[str, str]:
        """获取认证头"""
        headers = self.config.headers.copy()
        
        if self.config.auth_type == "bearer" and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        elif self.config.auth_type == "api_key":
            headers["X-API-Key"] = self.config.api_key
        elif self.config.auth_type == "basic":
            if self.config.api_key and self.config.api_secret:
                credentials = base64.b64encode(
                    f"{self.config.api_key}:{self.config.api_secret}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        return headers
    
    def refresh_token_if_needed(self, service_name: str) -> bool:
        """必要时刷新令牌"""
        if service_name in self._token_expires:
            if datetime.now() >= self._token_expires[service_name]:
                return self._refresh_token(service_name)
        return True
    
    def _refresh_token(self, service_name: str) -> bool:
        """刷新令牌"""
        # 这里可以实现具体的令牌刷新逻辑
        logger.info(f"刷新服务 {service_name} 的认证令牌")
        return True


class APIAdapter:
    """API适配器"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """设置会话"""
        # 配置重试策略
        retry_strategy = Retry(
            total=self.config.retry_count,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """发送HTTP请求"""
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # 设置默认参数
        kwargs.setdefault('timeout', self.config.timeout)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"请求频率超限: {e}")
            else:
                raise ThirdPartyError(f"HTTP错误: {e}", status_code=e.response.status_code)
        except requests.exceptions.RequestException as e:
            raise ThirdPartyError(f"请求异常: {e}")


class SecurityManager:
    """安全控制器"""
    
    @staticmethod
    def validate_input(data: Any) -> bool:
        """验证输入数据安全性"""
        if isinstance(data, str):
            # 检查潜在的恶意内容
            dangerous_patterns = [
                '<script', 'javascript:', 'vbscript:', 'onload=',
                'onerror=', 'onclick=', 'eval(', 'document.cookie'
            ]
            data_lower = data.lower()
            return not any(pattern in data_lower for pattern in dangerous_patterns)
        return True
    
    @staticmethod
    def sanitize_output(data: Any) -> Any:
        """清理输出数据"""
        if isinstance(data, str):
            # 移除或转义危险字符
            dangerous_chars = ['<', '>', '"', "'", '&']
            for char in dangerous_chars:
                data = data.replace(char, f'\\x{ord(char):02x}')
        return data
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """对敏感数据进行哈希处理"""
        return hashlib.sha256(data.encode()).hexdigest()


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, IntegrationMetrics] = {}
    
    def record_request(self, service_name: str, response_time: float, success: bool):
        """记录请求指标"""
        if service_name not in self.metrics:
            self.metrics[service_name] = IntegrationMetrics()
        
        metrics = self.metrics[service_name]
        metrics.total_requests += 1
        metrics.last_request_time = datetime.now()
        
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # 更新平均响应时间
        if metrics.total_requests == 1:
            metrics.average_response_time = response_time
        else:
            metrics.average_response_time = (
                (metrics.average_response_time * (metrics.total_requests - 1) + response_time) 
                / metrics.total_requests
            )
        
        # 更新错误率
        metrics.error_rate = metrics.failed_requests / metrics.total_requests
    
    def get_metrics(self, service_name: str) -> Optional[IntegrationMetrics]:
        """获取性能指标"""
        return self.metrics.get(service_name)
    
    def get_all_metrics(self) -> Dict[str, IntegrationMetrics]:
        """获取所有性能指标"""
        return self.metrics.copy()


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    def is_allowed(self, key: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        if key not in self.requests:
            self.requests[key] = []
        
        # 清理过期请求记录
        self.requests[key] = [
            req_time for req_time in self.requests[key] 
            if now - req_time < self.time_window
        ]
        
        # 检查是否超限
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # 记录本次请求
        self.requests[key].append(now)
        return True


class ThirdPartyIntegration:
    """第三方集成主类"""
    
    def __init__(self):
        self.configs: Dict[str, IntegrationConfig] = {}
        self.adapters: Dict[str, APIAdapter] = {}
        self.auth_managers: Dict[str, AuthenticationManager] = {}
        self.transformers: Dict[str, BaseDataTransformer] = {}
        self.security_manager = SecurityManager()
        self.performance_monitor = PerformanceMonitor()
        self.rate_limiters: Dict[str, RateLimiter] = {}
    
    def register_service(self, config: IntegrationConfig):
        """注册第三方服务"""
        self.configs[config.service_name] = config
        self.adapters[config.service_name] = APIAdapter(config)
        self.auth_managers[config.service_name] = AuthenticationManager(config)
        
        # 根据数据格式创建转换器
        if config.data_format == "json":
            self.transformers[config.service_name] = JSONDataTransformer()
        elif config.data_format == "xml":
            self.transformers[config.service_name] = XMLDataTransformer()
        
        # 创建速率限制器
        self.rate_limiters[config.service_name] = RateLimiter(config.rate_limit)
        
        logger.info(f"注册第三方服务: {config.service_name}")
    
    def call_api(self, service_name: str, method: str, endpoint: str, 
                 data: Any = None, params: Dict[str, Any] = None) -> Any:
        """调用第三方API"""
        if service_name not in self.configs:
            raise ThirdPartyError(f"未注册的服务: {service_name}")
        
        config = self.configs[service_name]
        adapter = self.adapters[service_name]
        auth_manager = self.auth_managers[service_name]
        rate_limiter = self.rate_limiters[service_name]
        
        # 检查速率限制
        if not rate_limiter.is_allowed(service_name):
            raise RateLimitError(f"服务 {service_name} 请求频率超限")
        
        # 验证输入数据
        if data and not self.security_manager.validate_input(str(data)):
            raise ThirdPartyError("输入数据包含不安全内容")
        
        start_time = time.time()
        success = False
        
        try:
            # 获取认证头
            headers = auth_manager.get_auth_headers()
            
            # 准备请求参数
            request_kwargs = {
                'headers': headers,
                'params': params or {}
            }
            
            # 处理请求数据
            if data:
                if config.data_format == "json":
                    request_kwargs['json'] = data
                elif config.data_format == "form":
                    request_kwargs['data'] = data
                else:
                    request_kwargs['data'] = str(data)
            
            # 发送请求
            response = adapter.request(method, endpoint, **request_kwargs)
            
            # 处理响应数据
            transformer = self.transformers.get(service_name)
            if transformer:
                result = transformer.transform(response.text)
            else:
                result = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            
            success = True
            return result
            
        except Exception as e:
            logger.error(f"API调用失败 [{service_name}]: {str(e)}")
            raise
        finally:
            # 记录性能指标
            response_time = time.time() - start_time
            self.performance_monitor.record_request(service_name, response_time, success)
    
    def batch_call(self, service_name: str, requests_list: List[Dict[str, Any]]) -> List[Any]:
        """批量调用API"""
        results = []
        for req in requests_list:
            try:
                result = self.call_api(
                    service_name,
                    req['method'],
                    req['endpoint'],
                    data=req.get('data'),
                    params=req.get('params')
                )
                results.append({'success': True, 'data': result})
            except Exception as e:
                results.append({'success': False, 'error': str(e)})
        return results
    
    def transform_data(self, service_name: str, data: Any, inverse: bool = False) -> Any:
        """转换数据格式"""
        transformer = self.transformers.get(service_name)
        if not transformer:
            raise DataTransformationError(f"服务 {service_name} 未配置数据转换器")
        
        try:
            if inverse:
                return transformer.inverse_transform(data)
            else:
                return transformer.transform(data)
        except Exception as e:
            raise DataTransformationError(f"数据转换失败: {str(e)}")
    
    def get_performance_metrics(self, service_name: str = None) -> Union[IntegrationMetrics, Dict[str, IntegrationMetrics]]:
        """获取性能指标"""
        if service_name:
            return self.performance_monitor.get_metrics(service_name)
        else:
            return self.performance_monitor.get_all_metrics()
    
    def health_check(self, service_name: str) -> Dict[str, Any]:
        """健康检查"""
        if service_name not in self.configs:
            return {'status': 'error', 'message': '服务未注册'}
        
        try:
            # 尝试调用一个简单的API端点进行健康检查
            # 这里可以根据具体服务调整
            result = self.call_api(service_name, 'GET', '/health')
            return {
                'status': 'healthy',
                'message': '服务正常',
                'data': result
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'服务异常: {str(e)}',
                'error': str(e)
            }
    
    def get_service_info(self, service_name: str) -> Dict[str, Any]:
        """获取服务信息"""
        if service_name not in self.configs:
            raise ThirdPartyError(f"未注册的服务: {service_name}")
        
        config = self.configs[service_name]
        metrics = self.performance_monitor.get_metrics(service_name)
        
        return {
            'service_name': config.service_name,
            'base_url': config.base_url,
            'auth_type': config.auth_type,
            'data_format': config.data_format,
            'timeout': config.timeout,
            'retry_count': config.retry_count,
            'rate_limit': config.rate_limit,
            'metrics': metrics.__dict__ if metrics else None
        }


# 全局集成实例
integration = ThirdPartyIntegration()


def create_integration(config: IntegrationConfig) -> ThirdPartyIntegration:
    """创建第三方集成实例"""
    integration.register_service(config)
    return integration


def get_integration() -> ThirdPartyIntegration:
    """获取全局集成实例"""
    return integration