"""
K8环境配置处理器模块

该模块提供了一套完整的Kubernetes环境配置管理解决方案，支持开发、测试、生产环境的
配置管理、环境变量管理、依赖管理、环境切换、监控和健康检查等功能。

主要功能：
- 多环境配置管理（开发、测试、生产）
- 环境变量管理和验证
- 环境依赖管理
- 环境切换和迁移
- 环境监控和健康检查
- 异步配置处理
- 完整的错误处理和日志记录

作者：K8配置管理团队
版本：1.0.0
创建时间：2025-11-06
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
import yaml
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable, 
    AsyncIterator, Tuple, TypeVar, Generic, Awaitable
)
from urllib.parse import urlparse
import hashlib
import tempfile
import threading
from datetime import datetime, timedelta


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    STAGING = "staging"


class ConfigStatus(Enum):
    """配置状态枚举"""
    PENDING = "pending"
    LOADING = "loading"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MIGRATING = "migrating"


class DependencyStatus(Enum):
    """依赖状态枚举"""
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    OUTDATED = "outdated"
    ERROR = "error"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"


# 类型定义
T = TypeVar('T')
ConfigDict = Dict[str, Any]
VariableDict = Dict[str, str]
DependencyDict = Dict[str, Any]


@dataclass
class EnvironmentConfig:
    """环境配置数据类"""
    name: str
    type: EnvironmentType
    description: str = ""
    parameters: ConfigDict = field(default_factory=dict)
    variables: VariableDict = field(default_factory=dict)
    dependencies: DependencyDict = field(default_factory=dict)
    monitoring: ConfigDict = field(default_factory=dict)
    security: ConfigDict = field(default_factory=dict)
    status: ConfigStatus = ConfigStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    def to_dict(self) -> ConfigDict:
        """转换为字典格式"""
        data = asdict(self)
        data['type'] = self.type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: ConfigDict) -> 'EnvironmentConfig':
        """从字典创建实例"""
        data['type'] = EnvironmentType(data['type'])
        data['status'] = ConfigStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class DependencyInfo:
    """依赖信息数据类"""
    name: str
    version: str
    type: str  # package, service, config, etc.
    status: DependencyStatus = DependencyStatus.NOT_INSTALLED
    install_command: Optional[str] = None
    config_file: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: ConfigDict = field(default_factory=dict)
    
    def to_dict(self) -> ConfigDict:
        """转换为字典格式"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: ConfigDict) -> 'DependencyInfo':
        """从字典创建实例"""
        data['status'] = DependencyStatus(data['status'])
        return cls(**data)


@dataclass
class HealthCheckResult:
    """健康检查结果数据类"""
    component: str
    status: HealthStatus
    message: str = ""
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    details: ConfigDict = field(default_factory=dict)
    
    def to_dict(self) -> ConfigDict:
        """转换为字典格式"""
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ConfigurationError(Exception):
    """配置错误异常"""
    pass


class DependencyError(Exception):
    """依赖错误异常"""
    pass


class HealthCheckError(Exception):
    """健康检查错误异常"""
    pass


class EnvironmentProcessor(ABC):
    """环境处理器抽象基类"""
    
    @abstractmethod
    async def process(self, config: EnvironmentConfig) -> bool:
        """处理环境配置"""
        pass
    
    @abstractmethod
    async def validate(self, config: EnvironmentConfig) -> bool:
        """验证环境配置"""
        pass


class DevelopmentEnvironmentProcessor(EnvironmentProcessor):
    """开发环境处理器"""
    
    async def process(self, config: EnvironmentConfig) -> bool:
        """处理开发环境配置"""
        try:
            logger.info(f"开始处理开发环境配置: {config.name}")
            
            # 设置开发参数
            await self._setup_development_parameters(config)
            
            # 配置调试设置
            await self._setup_debug_settings(config)
            
            # 配置开发工具
            await self._setup_development_tools(config)
            
            config.status = ConfigStatus.ACTIVE
            config.updated_at = datetime.now()
            
            logger.info(f"开发环境配置处理完成: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"开发环境配置处理失败: {e}")
            config.status = ConfigStatus.ERROR
            raise ConfigurationError(f"开发环境配置处理失败: {e}")
    
    async def validate(self, config: EnvironmentConfig) -> bool:
        """验证开发环境配置"""
        try:
            # 验证必要的开发参数
            required_params = ['debug_level', 'hot_reload', 'source_maps']
            for param in required_params:
                if param not in config.parameters:
                    raise ConfigurationError(f"缺少必要的开发参数: {param}")
            
            # 验证开发工具配置
            if 'tools' not in config.parameters:
                raise ConfigurationError("缺少开发工具配置")
            
            logger.info(f"开发环境配置验证通过: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"开发环境配置验证失败: {e}")
            return False
    
    async def _setup_development_parameters(self, config: EnvironmentConfig) -> None:
        """设置开发参数"""
        default_params = {
            'debug_level': 'debug',
            'hot_reload': True,
            'source_maps': True,
            'auto_restart': True,
            'verbose_logging': True,
            'development_mode': True
        }
        
        config.parameters.update(default_params)
        logger.debug("开发参数设置完成")
    
    async def _setup_debug_settings(self, config: EnvironmentConfig) -> None:
        """配置调试设置"""
        debug_config = {
            'debug_port': 5000,
            'debug_host': 'localhost',
            'debug_enabled': True,
            'break_on_start': False,
            'remote_debug': False
        }
        
        config.parameters.setdefault('debug', {}).update(debug_config)
        logger.debug("调试设置配置完成")
    
    async def _setup_development_tools(self, config: EnvironmentConfig) -> None:
        """配置开发工具"""
        tools_config = {
            'linter': {'enabled': True, 'tool': 'eslint'},
            'formatter': {'enabled': True, 'tool': 'prettier'},
            'test_runner': {'enabled': True, 'tool': 'jest'},
            'bundler': {'enabled': True, 'tool': 'webpack'},
            'hot_reload_server': {'enabled': True, 'port': 3000}
        }
        
        config.parameters.setdefault('tools', {}).update(tools_config)
        logger.debug("开发工具配置完成")


class TestingEnvironmentProcessor(EnvironmentProcessor):
    """测试环境处理器"""
    
    async def process(self, config: EnvironmentConfig) -> bool:
        """处理测试环境配置"""
        try:
            logger.info(f"开始处理测试环境配置: {config.name}")
            
            # 设置测试参数
            await self._setup_testing_parameters(config)
            
            # 配置测试数据
            await self._setup_test_data(config)
            
            # 配置测试工具
            await self._setup_testing_tools(config)
            
            config.status = ConfigStatus.ACTIVE
            config.updated_at = datetime.now()
            
            logger.info(f"测试环境配置处理完成: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"测试环境配置处理失败: {e}")
            config.status = ConfigStatus.ERROR
            raise ConfigurationError(f"测试环境配置处理失败: {e}")
    
    async def validate(self, config: EnvironmentConfig) -> bool:
        """验证测试环境配置"""
        try:
            # 验证测试参数
            required_params = ['test_framework', 'test_data_source', 'coverage_threshold']
            for param in required_params:
                if param not in config.parameters:
                    raise ConfigurationError(f"缺少必要的测试参数: {param}")
            
            # 验证测试数据配置
            if 'test_data' not in config.parameters:
                raise ConfigurationError("缺少测试数据配置")
            
            logger.info(f"测试环境配置验证通过: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"测试环境配置验证失败: {e}")
            return False
    
    async def _setup_testing_parameters(self, config: EnvironmentConfig) -> None:
        """设置测试参数"""
        default_params = {
            'test_framework': 'pytest',
            'parallel_execution': True,
            'coverage_threshold': 80,
            'test_timeout': 300,
            'retry_failed_tests': True,
            'generate_reports': True
        }
        
        config.parameters.update(default_params)
        logger.debug("测试参数设置完成")
    
    async def _setup_test_data(self, config: EnvironmentConfig) -> None:
        """配置测试数据"""
        test_data_config = {
            'seed_data': True,
            'mock_services': True,
            'test_database': 'test_db',
            'data_cleanup': 'after_each_test',
            'test_users': ['user1', 'user2', 'user3'],
            'test_scenarios': ['happy_path', 'edge_case', 'error_case']
        }
        
        config.parameters.setdefault('test_data', {}).update(test_data_config)
        logger.debug("测试数据配置完成")
    
    async def _setup_testing_tools(self, config: EnvironmentConfig) -> None:
        """配置测试工具"""
        tools_config = {
            'unit_test_runner': {'enabled': True, 'tool': 'pytest'},
            'integration_test_runner': {'enabled': True, 'tool': 'postman'},
            'e2e_test_runner': {'enabled': True, 'tool': 'selenium'},
            'performance_testing': {'enabled': True, 'tool': 'jmeter'},
            'code_coverage': {'enabled': True, 'tool': 'coverage.py'}
        }
        
        config.parameters.setdefault('tools', {}).update(tools_config)
        logger.debug("测试工具配置完成")


class ProductionEnvironmentProcessor(EnvironmentProcessor):
    """生产环境处理器"""
    
    async def process(self, config: EnvironmentConfig) -> bool:
        """处理生产环境配置"""
        try:
            logger.info(f"开始处理生产环境配置: {config.name}")
            
            # 设置生产参数
            await self._setup_production_parameters(config)
            
            # 配置生产监控
            await self._setup_production_monitoring(config)
            
            # 配置生产安全
            await self._setup_production_security(config)
            
            config.status = ConfigStatus.ACTIVE
            config.updated_at = datetime.now()
            
            logger.info(f"生产环境配置处理完成: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"生产环境配置处理失败: {e}")
            config.status = ConfigStatus.ERROR
            raise ConfigurationError(f"生产环境配置处理失败: {e}")
    
    async def validate(self, config: EnvironmentConfig) -> bool:
        """验证生产环境配置"""
        try:
            # 验证生产参数
            required_params = ['instance_count', 'load_balancer', 'auto_scaling']
            for param in required_params:
                if param not in config.parameters:
                    raise ConfigurationError(f"缺少必要的生产参数: {param}")
            
            # 验证安全配置
            if not config.security:
                raise ConfigurationError("缺少安全配置")
            
            # 验证监控配置
            if not config.monitoring:
                raise ConfigurationError("缺少监控配置")
            
            logger.info(f"生产环境配置验证通过: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"生产环境配置验证失败: {e}")
            return False
    
    async def _setup_production_parameters(self, config: EnvironmentConfig) -> None:
        """设置生产参数"""
        default_params = {
            'instance_count': 3,
            'load_balancer': True,
            'auto_scaling': True,
            'min_instances': 2,
            'max_instances': 10,
            'target_cpu_utilization': 70,
            'deployment_strategy': 'rolling_update',
            'health_check_enabled': True,
            'ssl_enabled': True
        }
        
        config.parameters.update(default_params)
        logger.debug("生产参数设置完成")
    
    async def _setup_production_monitoring(self, config: EnvironmentConfig) -> None:
        """配置生产监控"""
        monitoring_config = {
            'metrics_collection': True,
            'log_aggregation': True,
            'alerting': True,
            'performance_monitoring': True,
            'uptime_monitoring': True,
            'error_tracking': True,
            'metrics_endpoint': '/metrics',
            'log_level': 'info',
            'retention_period': '30d'
        }
        
        config.monitoring.update(monitoring_config)
        logger.debug("生产监控配置完成")
    
    async def _setup_production_security(self, config: EnvironmentConfig) -> None:
        """配置生产安全"""
        security_config = {
            'encryption': {
                'at_rest': True,
                'in_transit': True,
                'algorithm': 'AES-256'
            },
            'authentication': {
                'method': 'oauth2',
                'session_timeout': 3600,
                'max_login_attempts': 3
            },
            'authorization': {
                'rbac_enabled': True,
                'least_privilege': True
            },
            'network_security': {
                'firewall_enabled': True,
                'vpc_isolation': True,
                'ingress_restrictions': True
            },
            'compliance': {
                'gdpr_compliant': True,
                'soc2_compliant': True,
                'audit_logging': True
            }
        }
        
        config.security.update(security_config)
        logger.debug("生产安全配置完成")


class VariableManager:
    """环境变量管理器"""
    
    def __init__(self):
        self._variables: VariableDict = {}
        self._variable_patterns: Dict[str, str] = {}
        self._variable_validators: Dict[str, Callable[[str], bool]] = {}
        self._lock = threading.RLock()
    
    def register_variable(self, name: str, value: str, var_type: str = "user") -> None:
        """注册环境变量"""
        with self._lock:
            self._variables[name] = value
            logger.debug(f"注册环境变量: {name}={value} (类型: {var_type})")
    
    def register_variable_pattern(self, name: str, pattern: str) -> None:
        """注册变量模式"""
        with self._lock:
            self._variable_patterns[name] = pattern
            logger.debug(f"注册变量模式: {name}={pattern}")
    
    def register_validator(self, name: str, validator: Callable[[str], bool]) -> None:
        """注册变量验证器"""
        with self._lock:
            self._variable_validators[name] = validator
            logger.debug(f"注册变量验证器: {name}")
    
    def get_variable(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """获取环境变量"""
        with self._lock:
            return self._variables.get(name, default)
    
    def set_variable(self, name: str, value: str) -> None:
        """设置环境变量"""
        with self._lock:
            # 验证变量
            if name in self._variable_validators:
                if not self._variable_validators[name](value):
                    raise ConfigurationError(f"变量 {name} 的值验证失败: {value}")
            
            self._variables[name] = value
            logger.debug(f"设置环境变量: {name}={value}")
    
    def delete_variable(self, name: str) -> bool:
        """删除环境变量"""
        with self._lock:
            if name in self._variables:
                del self._variables[name]
                logger.debug(f"删除环境变量: {name}")
                return True
            return False
    
    def list_variables(self, var_type: Optional[str] = None) -> VariableDict:
        """列出环境变量"""
        with self._lock:
            if var_type is None:
                return self._variables.copy()
            
            # 按类型过滤（这里简化处理，实际应该按类型存储）
            return {k: v for k, v in self._variables.items() if not k.startswith('_')}
    
    def expand_variables(self, text: str) -> str:
        """展开文本中的变量"""
        with self._lock:
            result = text
            for name, pattern in self._variable_patterns.items():
                if name in self._variables:
                    result = result.replace(pattern, self._variables[name])
            
            # 展开 ${VAR} 格式的变量
            import re
            var_pattern = r'\$\{([^}]+)\}'
            matches = re.findall(var_pattern, result)
            for var_name in matches:
                if var_name in self._variables:
                    result = result.replace(f'${{{var_name}}}', self._variables[var_name])
            
            return result
    
    def validate_variables(self) -> List[str]:
        """验证所有变量"""
        errors = []
        with self._lock:
            for name, value in self._variables.items():
                if name in self._variable_validators:
                    try:
                        if not self._variable_validators[name](value):
                            errors.append(f"变量 {name} 验证失败: {value}")
                    except Exception as e:
                        errors.append(f"变量 {name} 验证异常: {e}")
        return errors
    
    def export_to_os_environment(self) -> None:
        """导出到操作系统环境变量"""
        with self._lock:
            for name, value in self._variables.items():
                os.environ[name] = value
            logger.info(f"已导出 {len(self._variables)} 个环境变量到操作系统")
    
    def load_from_os_environment(self, prefix: str = "") -> None:
        """从操作系统环境变量加载"""
        with self._lock:
            for name, value in os.environ.items():
                if prefix and not name.startswith(prefix):
                    continue
                self._variables[name] = value
            logger.info(f"从操作系统加载了 {len(self._variables)} 个环境变量")


class DependencyManager:
    """环境依赖管理器"""
    
    def __init__(self):
        self._dependencies: Dict[str, DependencyInfo] = {}
        self._installers: Dict[str, Callable[[DependencyInfo], Awaitable[bool]]] = {}
        self._lock = threading.RLock()
    
    def register_dependency(self, dependency: DependencyInfo) -> None:
        """注册依赖"""
        with self._lock:
            self._dependencies[dependency.name] = dependency
            logger.debug(f"注册依赖: {dependency.name} v{dependency.version}")
    
    def register_installer(self, dep_type: str, installer: Callable[[DependencyInfo], Awaitable[bool]]) -> None:
        """注册依赖安装器"""
        with self._lock:
            self._installers[dep_type] = installer
            logger.debug(f"注册安装器: {dep_type}")
    
    def get_dependency(self, name: str) -> Optional[DependencyInfo]:
        """获取依赖信息"""
        with self._lock:
            return self._dependencies.get(name)
    
    def list_dependencies(self, status: Optional[DependencyStatus] = None) -> List[DependencyInfo]:
        """列出依赖"""
        with self._lock:
            deps = list(self._dependencies.values())
            if status:
                deps = [dep for dep in deps if dep.status == status]
            return deps
    
    async def install_dependency(self, name: str) -> bool:
        """安装依赖"""
        with self._lock:
            if name not in self._dependencies:
                raise DependencyError(f"依赖 {name} 不存在")
            
            dependency = self._dependencies[name]
            dependency.status = DependencyStatus.INSTALLING
            
            try:
                if dependency.type not in self._installers:
                    raise DependencyError(f"没有找到 {dependency.type} 类型的安装器")
                
                installer = self._installers[dependency.type]
                success = await installer(dependency)
                
                if success:
                    dependency.status = DependencyStatus.INSTALLED
                    logger.info(f"依赖安装成功: {name}")
                else:
                    dependency.status = DependencyStatus.ERROR
                    logger.error(f"依赖安装失败: {name}")
                
                return success
                
            except Exception as e:
                dependency.status = DependencyStatus.ERROR
                logger.error(f"依赖安装异常: {name}, 错误: {e}")
                raise DependencyError(f"依赖安装异常: {name}, 错误: {e}")
    
    async def install_all_dependencies(self, max_workers: int = 5) -> Dict[str, bool]:
        """安装所有依赖"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交安装任务
            futures = {
                executor.submit(asyncio.run, self.install_dependency(name)): name
                for name in self._dependencies.keys()
            }
            
            # 收集结果
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"依赖安装任务异常: {name}, 错误: {e}")
                    results[name] = False
        
        return results
    
    async def check_dependencies(self) -> List[DependencyInfo]:
        """检查依赖状态"""
        outdated_deps = []
        
        with self._lock:
            for dependency in self._dependencies.values():
                # 这里可以实现具体的版本检查逻辑
                if dependency.status == DependencyStatus.INSTALLED:
                    # 模拟检查版本更新
                    if self._is_outdated(dependency):
                        dependency.status = DependencyStatus.OUTDATED
                        outdated_deps.append(dependency)
        
        return outdated_deps
    
    def _is_outdated(self, dependency: DependencyInfo) -> bool:
        """检查依赖是否过时"""
        # 简化的版本检查逻辑
        # 实际实现中应该连接到包管理器或服务发现
        return False
    
    def resolve_dependencies(self, name: str) -> List[str]:
        """解析依赖关系"""
        if name not in self._dependencies:
            return []
        
        dependency = self._dependencies[name]
        resolved = []
        
        def resolve_recursive(dep_name: str, visited: Set[str]) -> None:
            if dep_name in visited:
                return
            
            visited.add(dep_name)
            
            if dep_name in self._dependencies:
                dep = self._dependencies[dep_name]
                for child_dep in dep.dependencies:
                    if child_dep not in visited:
                        resolve_recursive(child_dep, visited)
                        resolved.append(child_dep)
        
        resolve_recursive(name, set())
        return resolved


class HealthChecker:
    """环境健康检查器"""
    
    def __init__(self):
        self._checkers: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        self._check_history: List[HealthCheckResult] = []
        self._lock = threading.RLock()
    
    def register_checker(self, component: str, checker: Callable[[], Awaitable[HealthCheckResult]]) -> None:
        """注册健康检查器"""
        with self._lock:
            self._checkers[component] = checker
            logger.debug(f"注册健康检查器: {component}")
    
    async def check_health(self, component: Optional[str] = None) -> List[HealthCheckResult]:
        """执行健康检查"""
        results = []
        
        if component:
            if component in self._checkers:
                result = await self._checkers[component]()
                results.append(result)
            else:
                results.append(HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNKNOWN,
                    message=f"未找到组件 {component} 的健康检查器"
                ))
        else:
            # 执行所有健康检查
            tasks = []
            for comp_name, checker in self._checkers.items():
                task = asyncio.create_task(self._safe_check(comp_name, checker))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 过滤异常结果
            results = [r for r in results if isinstance(r, HealthCheckResult)]
        
        # 记录检查历史
        with self._lock:
            self._check_history.extend(results)
            # 保留最近1000条记录
            if len(self._check_history) > 1000:
                self._check_history = self._check_history[-1000:]
        
        return results
    
    async def _safe_check(self, component: str, checker: Callable[[], Awaitable[HealthCheckResult]]) -> HealthCheckResult:
        """安全执行健康检查"""
        try:
            start_time = time.time()
            result = await checker()
            result.response_time = time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"健康检查异常: {component}, 错误: {e}")
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"健康检查异常: {e}"
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        with self._lock:
            if not self._check_history:
                return {"status": HealthStatus.UNKNOWN.value, "message": "没有健康检查记录"}
            
            latest_results = {}
            for result in reversed(self._check_history):
                if result.component not in latest_results:
                    latest_results[result.component] = result
            
            healthy_count = sum(1 for r in latest_results.values() if r.status == HealthStatus.HEALTHY)
            total_count = len(latest_results)
            
            overall_status = HealthStatus.HEALTHY
            if healthy_count < total_count:
                overall_status = HealthStatus.DEGRADED if healthy_count > 0 else HealthStatus.UNHEALTHY
            
            return {
                "status": overall_status.value,
                "healthy_components": healthy_count,
                "total_components": total_count,
                "components": {name: result.status.value for name, result in latest_results.items()},
                "last_check": max(r.timestamp for r in self._check_history).isoformat()
            }
    
    def get_check_history(self, component: Optional[str] = None, limit: int = 100) -> List[HealthCheckResult]:
        """获取检查历史"""
        with self._lock:
            history = self._check_history
            if component:
                history = [r for r in history if r.component == component]
            return history[-limit:]


class EnvironmentMigrator:
    """环境迁移器"""
    
    def __init__(self):
        self._migration_strategies: Dict[str, Callable[[EnvironmentConfig, EnvironmentConfig], Awaitable[bool]]] = {}
        self._migration_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
    
    def register_migration_strategy(self, from_type: str, to_type: str, strategy: Callable[[EnvironmentConfig, EnvironmentConfig], Awaitable[bool]]) -> None:
        """注册迁移策略"""
        strategy_key = f"{from_type}_to_{to_type}"
        with self._lock:
            self._migration_strategies[strategy_key] = strategy
            logger.debug(f"注册迁移策略: {strategy_key}")
    
    async def migrate_environment(self, source_config: EnvironmentConfig, target_config: EnvironmentConfig) -> bool:
        """迁移环境配置"""
        try:
            logger.info(f"开始环境迁移: {source_config.name} -> {target_config.name}")
            
            strategy_key = f"{source_config.type.value}_to_{target_config.type.value}"
            
            if strategy_key not in self._migration_strategies:
                raise ConfigurationError(f"未找到迁移策略: {strategy_key}")
            
            # 执行迁移
            source_config.status = ConfigStatus.MIGRATING
            strategy = self._migration_strategies[strategy_key]
            success = await strategy(source_config, target_config)
            
            if success:
                # 记录迁移历史
                migration_record = {
                    "source": source_config.name,
                    "target": target_config.name,
                    "source_type": source_config.type.value,
                    "target_type": target_config.type.value,
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                with self._lock:
                    self._migration_history.append(migration_record)
                
                logger.info(f"环境迁移成功: {source_config.name} -> {target_config.name}")
            else:
                logger.error(f"环境迁移失败: {source_config.name} -> {target_config.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"环境迁移异常: {e}")
            raise ConfigurationError(f"环境迁移异常: {e}")
    
    def get_migration_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取迁移历史"""
        with self._lock:
            return self._migration_history[-limit:]


class EnvironmentConfigurationProcessor:
    """K8环境配置处理器主类"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化环境配置处理器
        
        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "k8_configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各个管理器
        self.variable_manager = VariableManager()
        self.dependency_manager = DependencyManager()
        self.health_checker = HealthChecker()
        self.migrator = EnvironmentMigrator()
        
        # 初始化环境处理器
        self.environment_processors = {
            EnvironmentType.DEVELOPMENT: DevelopmentEnvironmentProcessor(),
            EnvironmentType.TESTING: TestingEnvironmentProcessor(),
            EnvironmentType.PRODUCTION: ProductionEnvironmentProcessor()
        }
        
        # 环境配置存储
        self._environments: Dict[str, EnvironmentConfig] = {}
        self._lock = threading.RLock()
        
        # 注册默认的依赖安装器
        self._register_default_installers()
        
        # 注册默认的健康检查器
        self._register_default_health_checkers()
        
        # 注册默认的迁移策略
        self._register_default_migration_strategies()
        
        logger.info("环境配置处理器初始化完成")
    
    def _register_default_installers(self) -> None:
        """注册默认的依赖安装器"""
        
        async def install_package(dependency: DependencyInfo) -> bool:
            """安装包依赖"""
            try:
                if dependency.install_command:
                    result = subprocess.run(
                        dependency.install_command,
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    return result.returncode == 0
                return True
            except Exception as e:
                logger.error(f"包安装失败: {dependency.name}, 错误: {e}")
                return False
        
        async def install_service(dependency: DependencyInfo) -> bool:
            """安装服务依赖"""
            try:
                # 模拟服务安装
                logger.info(f"安装服务: {dependency.name}")
                time.sleep(1)  # 模拟安装时间
                return True
            except Exception as e:
                logger.error(f"服务安装失败: {dependency.name}, 错误: {e}")
                return False
        
        async def install_config(dependency: DependencyInfo) -> bool:
            """安装配置依赖"""
            try:
                if dependency.config_file:
                    # 复制配置文件到配置目录
                    target_path = self.config_dir / dependency.config_file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    # 这里应该复制实际的配置文件
                    logger.info(f"安装配置: {dependency.name}")
                return True
            except Exception as e:
                logger.error(f"配置安装失败: {dependency.name}, 错误: {e}")
                return False
        
        self.dependency_manager.register_installer("package", install_package)
        self.dependency_manager.register_installer("service", install_service)
        self.dependency_manager.register_installer("config", install_config)
    
    def _register_default_health_checkers(self) -> None:
        """注册默认的健康检查器"""
        
        async def check_kubernetes_cluster() -> HealthCheckResult:
            """检查Kubernetes集群"""
            try:
                start_time = time.time()
                # 模拟集群检查
                result = subprocess.run(
                    ["kubectl", "cluster-info"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                response_time = time.time() - start_time
                
                if result.returncode == 0:
                    return HealthCheckResult(
                        component="kubernetes_cluster",
                        status=HealthStatus.HEALTHY,
                        message="Kubernetes集群运行正常",
                        response_time=response_time
                    )
                else:
                    return HealthCheckResult(
                        component="kubernetes_cluster",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Kubernetes集群检查失败: {result.stderr}",
                        response_time=response_time
                    )
            except Exception as e:
                return HealthCheckResult(
                    component="kubernetes_cluster",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Kubernetes集群检查异常: {e}"
                )
        
        async def check_docker_service() -> HealthCheckResult:
            """检查Docker服务"""
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                response_time = time.time() - start_time
                
                if result.returncode == 0:
                    return HealthCheckResult(
                        component="docker_service",
                        status=HealthStatus.HEALTHY,
                        message="Docker服务运行正常",
                        response_time=response_time
                    )
                else:
                    return HealthCheckResult(
                        component="docker_service",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Docker服务检查失败: {result.stderr}",
                        response_time=response_time
                    )
            except Exception as e:
                return HealthCheckResult(
                    component="docker_service",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Docker服务检查异常: {e}"
                )
        
        async def check_network_connectivity() -> HealthCheckResult:
            """检查网络连接"""
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["ping", "-c", "1", "8.8.8.8"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                response_time = time.time() - start_time
                
                if result.returncode == 0:
                    return HealthCheckResult(
                        component="network_connectivity",
                        status=HealthStatus.HEALTHY,
                        message="网络连接正常",
                        response_time=response_time
                    )
                else:
                    return HealthCheckResult(
                        component="network_connectivity",
                        status=HealthStatus.DEGRADED,
                        message="网络连接不稳定",
                        response_time=response_time
                    )
            except Exception as e:
                return HealthCheckResult(
                    component="network_connectivity",
                    status=HealthStatus.UNHEALTHY,
                    message=f"网络连接检查异常: {e}"
                )
        
        self.health_checker.register_checker("kubernetes_cluster", check_kubernetes_cluster)
        self.health_checker.register_checker("docker_service", check_docker_service)
        self.health_checker.register_checker("network_connectivity", check_network_connectivity)
    
    def _register_default_migration_strategies(self) -> None:
        """注册默认的迁移策略"""
        
        async def dev_to_test_migration(source: EnvironmentConfig, target: EnvironmentConfig) -> bool:
            """开发到测试环境迁移"""
            try:
                logger.info("执行开发到测试环境迁移")
                
                # 迁移配置参数
                target.parameters.update(source.parameters)
                
                # 调整测试特定参数
                target.parameters.update({
                    'test_mode': True,
                    'debug_level': 'info',
                    'verbose_logging': False
                })
                
                # 迁移环境变量
                target.variables.update(source.variables)
                
                # 迁移依赖
                target.dependencies.update(source.dependencies)
                
                return True
                
            except Exception as e:
                logger.error(f"开发到测试迁移失败: {e}")
                return False
        
        async def test_to_prod_migration(source: EnvironmentConfig, target: EnvironmentConfig) -> bool:
            """测试到生产环境迁移"""
            try:
                logger.info("执行测试到生产环境迁移")
                
                # 迁移基础配置
                target.parameters.update(source.parameters)
                
                # 调整生产特定参数
                target.parameters.update({
                    'production_mode': True,
                    'debug_level': 'error',
                    'verbose_logging': False,
                    'ssl_enabled': True,
                    'load_balancer': True
                })
                
                # 迁移环境变量
                target.variables.update(source.variables)
                
                # 清理测试特定变量
                test_vars_to_remove = [k for k in target.variables.keys() if k.startswith('TEST_')]
                for var in test_vars_to_remove:
                    del target.variables[var]
                
                # 迁移依赖
                target.dependencies.update(source.dependencies)
                
                return True
                
            except Exception as e:
                logger.error(f"测试到生产迁移失败: {e}")
                return False
        
        async def prod_to_test_migration(source: EnvironmentConfig, target: EnvironmentConfig) -> bool:
            """生产到测试环境迁移"""
            try:
                logger.info("执行生产到测试环境迁移")
                
                # 迁移配置参数
                target.parameters.update(source.parameters)
                
                # 调整测试参数
                target.parameters.update({
                    'test_mode': True,
                    'debug_level': 'debug',
                    'verbose_logging': True
                })
                
                # 迁移环境变量
                target.variables.update(source.variables)
                
                # 添加测试特定变量
                target.variables.update({
                    'TEST_MODE': 'true',
                    'TEST_DATA_SOURCE': 'mock'
                })
                
                # 迁移依赖
                target.dependencies.update(source.dependencies)
                
                return True
                
            except Exception as e:
                logger.error(f"生产到测试迁移失败: {e}")
                return False
        
        self.migrator.register_migration_strategy(
            EnvironmentType.DEVELOPMENT.value,
            EnvironmentType.TESTING.value,
            dev_to_test_migration
        )
        
        self.migrator.register_migration_strategy(
            EnvironmentType.TESTING.value,
            EnvironmentType.PRODUCTION.value,
            test_to_prod_migration
        )
        
        self.migrator.register_migration_strategy(
            EnvironmentType.PRODUCTION.value,
            EnvironmentType.TESTING.value,
            prod_to_test_migration
        )
    
    async def create_environment(self, config: EnvironmentConfig) -> bool:
        """
        创建环境配置
        
        Args:
            config: 环境配置对象
            
        Returns:
            bool: 创建是否成功
        """
        try:
            logger.info(f"创建环境配置: {config.name}")
            
            # 验证配置
            processor = self.environment_processors.get(config.type)
            if not processor:
                raise ConfigurationError(f"不支持的环境类型: {config.type}")
            
            if not await processor.validate(config):
                raise ConfigurationError("环境配置验证失败")
            
            # 处理配置
            if not await processor.process(config):
                raise ConfigurationError("环境配置处理失败")
            
            # 保存配置
            with self._lock:
                self._environments[config.name] = config
            
            # 保存到文件
            await self._save_config_to_file(config)
            
            logger.info(f"环境配置创建成功: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"环境配置创建失败: {e}")
            raise ConfigurationError(f"环境配置创建失败: {e}")
    
    async def update_environment(self, name: str, updates: ConfigDict) -> bool:
        """
        更新环境配置
        
        Args:
            name: 环境名称
            updates: 更新内容
            
        Returns:
            bool: 更新是否成功
        """
        try:
            logger.info(f"更新环境配置: {name}")
            
            with self._lock:
                if name not in self._environments:
                    raise ConfigurationError(f"环境配置不存在: {name}")
                
                config = self._environments[name]
                
                # 更新配置
                for key, value in updates.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                config.updated_at = datetime.now()
            
            # 重新处理配置
            processor = self.environment_processors.get(config.type)
            if processor:
                await processor.process(config)
            
            # 保存配置
            await self._save_config_to_file(config)
            
            logger.info(f"环境配置更新成功: {name}")
            return True
            
        except Exception as e:
            logger.error(f"环境配置更新失败: {e}")
            raise ConfigurationError(f"环境配置更新失败: {e}")
    
    async def delete_environment(self, name: str) -> bool:
        """
        删除环境配置
        
        Args:
            name: 环境名称
            
        Returns:
            bool: 删除是否成功
        """
        try:
            logger.info(f"删除环境配置: {name}")
            
            with self._lock:
                if name not in self._environments:
                    raise ConfigurationError(f"环境配置不存在: {name}")
                
                config = self._environments[name]
                
                # 删除配置
                del self._environments[name]
            
            # 删除配置文件
            config_file = self.config_dir / f"{name}.yaml"
            if config_file.exists():
                config_file.unlink()
            
            logger.info(f"环境配置删除成功: {name}")
            return True
            
        except Exception as e:
            logger.error(f"环境配置删除失败: {e}")
            raise ConfigurationError(f"环境配置删除失败: {e}")
    
    def get_environment(self, name: str) -> Optional[EnvironmentConfig]:
        """
        获取环境配置
        
        Args:
            name: 环境名称
            
        Returns:
            EnvironmentConfig: 环境配置对象，如果不存在返回None
        """
        with self._lock:
            return self._environments.get(name)
    
    def list_environments(self, env_type: Optional[EnvironmentType] = None) -> List[EnvironmentConfig]:
        """
        列出环境配置
        
        Args:
            env_type: 环境类型过滤
            
        Returns:
            List[EnvironmentConfig]: 环境配置列表
        """
        with self._lock:
            configs = list(self._environments.values())
            if env_type:
                configs = [config for config in configs if config.type == env_type]
            return configs
    
    async def load_configurations(self) -> None:
        """加载所有配置文件"""
        try:
            logger.info("开始加载配置文件")
            
            config_files = list(self.config_dir.glob("*.yaml"))
            config_files.extend(self.config_dir.glob("*.json"))
            
            for config_file in config_files:
                try:
                    config = await self._load_config_from_file(config_file)
                    if config:
                        with self._lock:
                            self._environments[config.name] = config
                        logger.debug(f"加载配置文件: {config_file}")
                except Exception as e:
                    logger.error(f"加载配置文件失败: {config_file}, 错误: {e}")
            
            logger.info(f"配置文件加载完成，共加载 {len(self._environments)} 个配置")
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise ConfigurationError(f"配置文件加载失败: {e}")
    
    async def _save_config_to_file(self, config: EnvironmentConfig) -> None:
        """保存配置到文件"""
        config_file = self.config_dir / f"{config.name}.yaml"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"保存配置文件失败: {config_file}, 错误: {e}")
            raise
    
    async def _load_config_from_file(self, config_file: Path) -> Optional[EnvironmentConfig]:
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix == '.json':
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
            
            return EnvironmentConfig.from_dict(data)
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {config_file}, 错误: {e}")
            return None
    
    async def switch_environment(self, from_name: str, to_name: str) -> bool:
        """
        切换环境
        
        Args:
            from_name: 源环境名称
            to_name: 目标环境名称
            
        Returns:
            bool: 切换是否成功
        """
        try:
            logger.info(f"切换环境: {from_name} -> {to_name}")
            
            with self._lock:
                if from_name not in self._environments:
                    raise ConfigurationError(f"源环境不存在: {from_name}")
                
                if to_name not in self._environments:
                    raise ConfigurationError(f"目标环境不存在: {to_name}")
                
                source_config = self._environments[from_name]
                target_config = self._environments[to_name]
            
            # 执行迁移
            success = await self.migrator.migrate_environment(source_config, target_config)
            
            if success:
                # 更新环境状态
                source_config.status = ConfigStatus.INACTIVE
                target_config.status = ConfigStatus.ACTIVE
                
                # 更新环境变量
                self.variable_manager.load_from_os_environment()
                for name, value in target_config.variables.items():
                    self.variable_manager.set_variable(name, value)
                self.variable_manager.export_to_os_environment()
                
                logger.info(f"环境切换成功: {from_name} -> {to_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"环境切换失败: {e}")
            raise ConfigurationError(f"环境切换失败: {e}")
    
    async def check_environment_health(self, env_name: Optional[str] = None) -> Dict[str, Any]:
        """
        检查环境健康状态
        
        Args:
            env_name: 环境名称，如果为None则检查所有环境
            
        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            logger.info(f"开始环境健康检查: {env_name or 'all'}")
            
            if env_name:
                # 检查特定环境
                with self._lock:
                    if env_name not in self._environments:
                        raise ConfigurationError(f"环境不存在: {env_name}")
                    
                    config = self._environments[env_name]
                
                health_results = await self.health_checker.check_health()
                
                return {
                    "environment": env_name,
                    "status": config.status.value,
                    "health_checks": [result.to_dict() for result in health_results],
                    "summary": self.health_checker.get_health_summary()
                }
            else:
                # 检查所有环境
                all_results = {}
                
                with self._lock:
                    env_names = list(self._environments.keys())
                
                for env_name in env_names:
                    try:
                        result = await self.check_environment_health(env_name)
                        all_results[env_name] = result
                    except Exception as e:
                        logger.error(f"环境健康检查失败: {env_name}, 错误: {e}")
                        all_results[env_name] = {
                            "environment": env_name,
                            "error": str(e)
                        }
                
                return all_results
                
        except Exception as e:
            logger.error(f"环境健康检查失败: {e}")
            raise HealthCheckError(f"环境健康检查失败: {e}")
    
    async def install_environment_dependencies(self, env_name: str) -> Dict[str, bool]:
        """
        安装环境依赖
        
        Args:
            env_name: 环境名称
            
        Returns:
            Dict[str, bool]: 依赖安装结果
        """
        try:
            logger.info(f"安装环境依赖: {env_name}")
            
            with self._lock:
                if env_name not in self._environments:
                    raise ConfigurationError(f"环境不存在: {env_name}")
                
                config = self._environments[env_name]
            
            # 注册环境依赖
            for dep_name, dep_info in config.dependencies.items():
                dependency = DependencyInfo.from_dict(dep_info)
                self.dependency_manager.register_dependency(dependency)
            
            # 安装所有依赖
            results = await self.dependency_manager.install_all_dependencies()
            
            logger.info(f"环境依赖安装完成: {env_name}")
            return results
            
        except Exception as e:
            logger.error(f"环境依赖安装失败: {e}")
            raise DependencyError(f"环境依赖安装失败: {e}")
    
    async def validate_environment(self, env_name: str) -> List[str]:
        """
        验证环境配置
        
        Args:
            env_name: 环境名称
            
        Returns:
            List[str]: 验证错误列表
        """
        try:
            logger.info(f"验证环境配置: {env_name}")
            
            with self._lock:
                if env_name not in self._environments:
                    raise ConfigurationError(f"环境不存在: {env_name}")
                
                config = self._environments[env_name]
            
            errors = []
            
            # 验证环境配置
            processor = self.environment_processors.get(config.type)
            if processor:
                try:
                    valid = await processor.validate(config)
                    if not valid:
                        errors.append("环境配置验证失败")
                except Exception as e:
                    errors.append(f"环境配置验证异常: {e}")
            
            # 验证环境变量
            for name, value in config.variables.items():
                self.variable_manager.register_variable(name, value)
            
            var_errors = self.variable_manager.validate_variables()
            errors.extend(var_errors)
            
            # 验证依赖
            for dep_name, dep_info in config.dependencies.items():
                dependency = DependencyInfo.from_dict(dep_info)
                if dependency.status == DependencyStatus.ERROR:
                    errors.append(f"依赖 {dep_name} 状态异常")
            
            logger.info(f"环境配置验证完成: {env_name}, 发现 {len(errors)} 个问题")
            return errors
            
        except Exception as e:
            logger.error(f"环境配置验证失败: {e}")
            raise ConfigurationError(f"环境配置验证失败: {e}")
    
    @asynccontextmanager
    async def environment_context(self, env_name: str):
        """
        环境上下文管理器
        
        Args:
            env_name: 环境名称
            
        Yields:
            EnvironmentConfig: 环境配置对象
        """
        logger.info(f"进入环境上下文: {env_name}")
        
        # 保存当前环境状态
        original_env = os.environ.copy()
        
        try:
            with self._lock:
                if env_name not in self._environments:
                    raise ConfigurationError(f"环境不存在: {env_name}")
                
                config = self._environments[env_name]
            
            # 设置环境变量
            for name, value in config.variables.items():
                os.environ[name] = value
            
            # 激活环境
            config.status = ConfigStatus.ACTIVE
            
            yield config
            
        finally:
            # 恢复原始环境
            os.environ.clear()
            os.environ.update(original_env)
            
            # 恢复环境状态
            config.status = ConfigStatus.INACTIVE
            
            logger.info(f"退出环境上下文: {env_name}")
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """获取环境摘要信息"""
        with self._lock:
            environments = list(self._environments.values())
            
            summary = {
                "total_environments": len(environments),
                "environments_by_type": {},
                "environments_by_status": {},
                "recent_updates": []
            }
            
            # 按类型统计
            for env in environments:
                env_type = env.type.value
                summary["environments_by_type"][env_type] = \
                    summary["environments_by_type"].get(env_type, 0) + 1
            
            # 按状态统计
            for env in environments:
                status = env.status.value
                summary["environments_by_status"][status] = \
                    summary["environments_by_status"].get(status, 0) + 1
            
            # 最近更新
            sorted_envs = sorted(environments, key=lambda x: x.updated_at, reverse=True)
            summary["recent_updates"] = [
                {
                    "name": env.name,
                    "type": env.type.value,
                    "status": env.status.value,
                    "updated_at": env.updated_at.isoformat()
                }
                for env in sorted_envs[:5]
            ]
            
            return summary


# 使用示例和工具函数
def create_development_config(name: str, description: str = "") -> EnvironmentConfig:
    """
    创建开发环境配置
    
    Args:
        name: 环境名称
        description: 环境描述
        
    Returns:
        EnvironmentConfig: 开发环境配置对象
    """
    config = EnvironmentConfig(
        name=name,
        type=EnvironmentType.DEVELOPMENT,
        description=description or f"开发环境 - {name}",
        parameters={
            "debug_level": "debug",
            "hot_reload": True,
            "source_maps": True,
            "tools": {
                "linter": {"enabled": True, "tool": "eslint"},
                "formatter": {"enabled": True, "tool": "prettier"}
            }
        },
        variables={
            "ENVIRONMENT": "development",
            "DEBUG": "true",
            "LOG_LEVEL": "debug"
        },
        dependencies={
            "nodejs": {
                "name": "nodejs",
                "version": "18.x",
                "type": "package",
                "install_command": "apt-get install -y nodejs"
            },
            "docker": {
                "name": "docker",
                "version": "latest",
                "type": "service",
                "install_command": "apt-get install -y docker.io"
            }
        }
    )
    
    return config


def create_testing_config(name: str, description: str = "") -> EnvironmentConfig:
    """
    创建测试环境配置
    
    Args:
        name: 环境名称
        description: 环境描述
        
    Returns:
        EnvironmentConfig: 测试环境配置对象
    """
    config = EnvironmentConfig(
        name=name,
        type=EnvironmentType.TESTING,
        description=description or f"测试环境 - {name}",
        parameters={
            "test_framework": "pytest",
            "test_data_source": "mock",
            "parallel_execution": True,
            "coverage_threshold": 80,
            "test_data": {
                "seed_data": True,
                "mock_services": True,
                "test_database": "test_db"
            }
        },
        variables={
            "ENVIRONMENT": "testing",
            "DEBUG": "false",
            "LOG_LEVEL": "info",
            "TEST_MODE": "true"
        },
        dependencies={
            "pytest": {
                "name": "pytest",
                "version": "7.x",
                "type": "package",
                "install_command": "pip install pytest"
            },
            "selenium": {
                "name": "selenium",
                "version": "4.x",
                "type": "package",
                "install_command": "pip install selenium"
            }
        }
    )
    
    return config


def create_production_config(name: str, description: str = "") -> EnvironmentConfig:
    """
    创建生产环境配置
    
    Args:
        name: 环境名称
        description: 环境描述
        
    Returns:
        EnvironmentConfig: 生产环境配置对象
    """
    config = EnvironmentConfig(
        name=name,
        type=EnvironmentType.PRODUCTION,
        description=description or f"生产环境 - {name}",
        parameters={
            "instance_count": 3,
            "load_balancer": True,
            "auto_scaling": True,
            "deployment_strategy": "rolling_update"
        },
        variables={
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            "LOG_LEVEL": "error",
            "SSL_ENABLED": "true"
        },
        dependencies={
            "nginx": {
                "name": "nginx",
                "version": "1.20.x",
                "type": "service",
                "install_command": "apt-get install -y nginx"
            },
            "ssl-cert": {
                "name": "ssl-cert",
                "version": "latest",
                "type": "config",
                "config_file": "ssl/cert.pem"
            }
        },
        monitoring={
            "metrics_collection": True,
            "log_aggregation": True,
            "alerting": True
        },
        security={
            "encryption": {
                "at_rest": True,
                "in_transit": True
            },
            "authentication": {
                "method": "oauth2",
                "session_timeout": 3600
            }
        }
    )
    
    return config


async def main():
    """主函数 - 使用示例"""
    
    # 创建环境配置处理器
    processor = EnvironmentConfigurationProcessor("./k8_configs")
    
    try:
        # 创建开发环境
        dev_config = create_development_config("dev-env", "开发测试环境")
        await processor.create_environment(dev_config)
        
        # 创建测试环境
        test_config = create_testing_config("test-env", "集成测试环境")
        await processor.create_environment(test_config)
        
        # 创建生产环境
        prod_config = create_production_config("prod-env", "生产环境")
        await processor.create_environment(prod_config)
        
        # 列出所有环境
        environments = processor.list_environments()
        print(f"当前环境数量: {len(environments)}")
        
        for env in environments:
            print(f"- {env.name} ({env.type.value}) - {env.status.value}")
        
        # 检查环境健康状态
        health_results = await processor.check_environment_health()
        print(f"健康检查结果: {health_results}")
        
        # 切换环境
        await processor.switch_environment("dev-env", "test-env")
        
        # 验证环境
        errors = await processor.validate_environment("test-env")
        if errors:
            print(f"验证发现的问题: {errors}")
        else:
            print("环境验证通过")
        
        # 获取环境摘要
        summary = processor.get_environment_summary()
        print(f"环境摘要: {summary}")
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())

class StagingEnvironmentProcessor(EnvironmentProcessor):
    """预发布环境处理器"""
    
    async def process(self, config: EnvironmentConfig) -> bool:
        """处理预发布环境配置"""
        try:
            logger.info(f"开始处理预发布环境配置: {config.name}")
            
            # 设置预发布参数
            await self._setup_staging_parameters(config)
            
            # 配置预发布监控
            await self._setup_staging_monitoring(config)
            
            # 配置预发布安全
            await self._setup_staging_security(config)
            
            config.status = ConfigStatus.ACTIVE
            config.updated_at = datetime.now()
            
            logger.info(f"预发布环境配置处理完成: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"预发布环境配置处理失败: {e}")
            config.status = ConfigStatus.ERROR
            raise ConfigurationError(f"预发布环境配置处理失败: {e}")
    
    async def validate(self, config: EnvironmentConfig) -> bool:
        """验证预发布环境配置"""
        try:
            # 验证预发布参数
            required_params = ['staging_mode', 'canary_deployment', 'rollback_enabled']
            for param in required_params:
                if param not in config.parameters:
                    raise ConfigurationError(f"缺少必要的预发布参数: {param}")
            
            # 验证监控配置
            if not config.monitoring:
                raise ConfigurationError("缺少监控配置")
            
            logger.info(f"预发布环境配置验证通过: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"预发布环境配置验证失败: {e}")
            return False
    
    async def _setup_staging_parameters(self, config: EnvironmentConfig) -> None:
        """设置预发布参数"""
        default_params = {
            'staging_mode': True,
            'canary_deployment': True,
            'rollback_enabled': True,
            'traffic_splitting': 10,  # 10% traffic to staging
            'feature_flags': True,
            'blue_green_deployment': True
        }
        
        config.parameters.update(default_params)
        logger.debug("预发布参数设置完成")
    
    async def _setup_staging_monitoring(self, config: EnvironmentConfig) -> None:
        """配置预发布监控"""
        monitoring_config = {
            'enhanced_monitoring': True,
            'performance_profiling': True,
            'user_behavior_analytics': True,
            'a_b_testing_support': True,
            'detailed_logging': True,
            'metrics_retention': '7d'
        }
        
        config.monitoring.update(monitoring_config)
        logger.debug("预发布监控配置完成")
    
    async def _setup_staging_security(self, config: EnvironmentConfig) -> None:
        """配置预发布安全"""
        security_config = {
            'security_scanning': True,
            'vulnerability_assessment': True,
            'compliance_checking': True,
            'access_control': {
                'restricted_access': True,
                'ip_whitelisting': True,
                'multi_factor_auth': True
            },
            'data_protection': {
                'encryption': True,
                'data_masking': True,
                'audit_logging': True
            }
        }
        
        config.security.update(security_config)
        logger.debug("预发布安全配置完成")


class ConfigurationValidator:
    """配置验证器"""
    
    def __init__(self):
        self._validators: Dict[str, Callable[[Any], bool]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def register_validator(self, name: str, validator: Callable[[Any], bool]) -> None:
        """注册验证器"""
        with self._lock:
            self._validators[name] = validator
            logger.debug(f"注册配置验证器: {name}")
    
    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """注册配置模式"""
        with self._lock:
            self._schemas[name] = schema
            logger.debug(f"注册配置模式: {name}")
    
    def validate_config(self, config: EnvironmentConfig, strict: bool = True) -> List[str]:
        """验证配置"""
        errors = []
        
        # 基础验证
        errors.extend(self._validate_basic_config(config))
        
        # 类型特定验证
        errors.extend(self._validate_type_specific_config(config))
        
        # 模式验证
        if config.type.value in self._schemas:
            errors.extend(self._validate_against_schema(config, self._schemas[config.type.value]))
        
        # 自定义验证器
        for validator_name, validator in self._validators.items():
            try:
                if not validator(config):
                    errors.append(f"自定义验证失败: {validator_name}")
            except Exception as e:
                errors.append(f"自定义验证异常: {validator_name}, 错误: {e}")
        
        # 严格模式检查
        if strict:
            errors.extend(self._strict_validation(config))
        
        return errors
    
    def _validate_basic_config(self, config: EnvironmentConfig) -> List[str]:
        """基础配置验证"""
        errors = []
        
        if not config.name or not config.name.strip():
            errors.append("环境名称不能为空")
        
        if not config.description or not config.description.strip():
            errors.append("环境描述不能为空")
        
        if not isinstance(config.parameters, dict):
            errors.append("参数必须是字典类型")
        
        if not isinstance(config.variables, dict):
            errors.append("变量必须是字典类型")
        
        if not isinstance(config.dependencies, dict):
            errors.append("依赖必须是字典类型")
        
        return errors
    
    def _validate_type_specific_config(self, config: EnvironmentConfig) -> List[str]:
        """类型特定配置验证"""
        errors = []
        
        if config.type == EnvironmentType.DEVELOPMENT:
            if 'debug_level' not in config.parameters:
                errors.append("开发环境必须包含debug_level参数")
            
            if config.parameters.get('debug_level') not in ['debug', 'info', 'warning', 'error']:
                errors.append("debug_level必须是有效的日志级别")
        
        elif config.type == EnvironmentType.TESTING:
            if 'test_framework' not in config.parameters:
                errors.append("测试环境必须包含test_framework参数")
            
            if 'coverage_threshold' in config.parameters:
                threshold = config.parameters['coverage_threshold']
                if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 100:
                    errors.append("coverage_threshold必须是0-100之间的数字")
        
        elif config.type == EnvironmentType.PRODUCTION:
            if 'instance_count' not in config.parameters:
                errors.append("生产环境必须包含instance_count参数")
            
            instance_count = config.parameters.get('instance_count', 0)
            if not isinstance(instance_count, int) or instance_count < 1:
                errors.append("instance_count必须是大于0的整数")
            
            if not config.security:
                errors.append("生产环境必须包含安全配置")
        
        elif config.type == EnvironmentType.STAGING:
            if 'staging_mode' not in config.parameters:
                errors.append("预发布环境必须包含staging_mode参数")
        
        return errors
    
    def _validate_against_schema(self, config: EnvironmentConfig, schema: Dict[str, Any]) -> List[str]:
        """根据模式验证配置"""
        errors = []
        
        # 这里可以实现基于JSON Schema的验证逻辑
        # 简化实现，实际应该使用jsonschema库
        
        required_fields = schema.get('required', [])
        for field in required_fields:
            if not hasattr(config, field) or getattr(config, field) is None:
                errors.append(f"缺少必需字段: {field}")
        
        return errors
    
    def _strict_validation(self, config: EnvironmentConfig) -> List[str]:
        """严格模式验证"""
        errors = []
        
        # 检查配置大小
        if len(str(config.parameters)) > 10000:  # 10KB
            errors.append("参数配置过大，超过10KB限制")
        
        if len(str(config.variables)) > 5000:  # 5KB
            errors.append("变量配置过大，超过5KB限制")
        
        if len(str(config.dependencies)) > 8000:  # 8KB
            errors.append("依赖配置过大，超过8KB限制")
        
        # 检查配置复杂度
        if len(config.parameters) > 100:
            errors.append("参数数量过多，超过100个限制")
        
        if len(config.variables) > 200:
            errors.append("变量数量过多，超过200个限制")
        
        # 检查变量名格式
        for var_name in config.variables.keys():
            if not re.match(r'^[A-Z_][A-Z0-9_]*$', var_name):
                errors.append(f"变量名格式不正确: {var_name}")
        
        # 检查依赖名称格式
        for dep_name in config.dependencies.keys():
            if not re.match(r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$', dep_name):
                errors.append(f"依赖名格式不正确: {dep_name}")
        
        return errors


class ConfigurationBackupManager:
    """配置备份管理器"""
    
    def __init__(self, backup_dir: Optional[str] = None):
        self.backup_dir = Path(backup_dir) if backup_dir else Path.cwd() / "k8_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._backup_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
    
    async def create_backup(self, config: EnvironmentConfig, description: str = "") -> str:
        """创建配置备份"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{config.name}_{timestamp}"
            backup_path = self.backup_dir / f"{backup_name}.yaml"
            
            # 创建备份数据
            backup_data = {
                "config": config.to_dict(),
                "backup_info": {
                    "backup_name": backup_name,
                    "timestamp": datetime.now().isoformat(),
                    "description": description,
                    "original_name": config.name,
                    "version": config.version
                }
            }
            
            # 保存备份文件
            with open(backup_path, 'w', encoding='utf-8') as f:
                yaml.dump(backup_data, f, default_flow_style=False, allow_unicode=True)
            
            # 记录备份历史
            backup_record = {
                "backup_name": backup_name,
                "config_name": config.name,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "file_path": str(backup_path)
            }
            
            with self._lock:
                self._backup_history.append(backup_record)
            
            logger.info(f"配置备份创建成功: {backup_name}")
            return backup_name
            
        except Exception as e:
            logger.error(f"配置备份创建失败: {e}")
            raise ConfigurationError(f"配置备份创建失败: {e}")
    
    async def restore_backup(self, backup_name: str) -> Optional[EnvironmentConfig]:
        """恢复配置备份"""
        try:
            backup_file = self.backup_dir / f"{backup_name}.yaml"
            
            if not backup_file.exists():
                raise ConfigurationError(f"备份文件不存在: {backup_name}")
            
            # 加载备份数据
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = yaml.safe_load(f)
            
            config_data = backup_data.get("config")
            if not config_data:
                raise ConfigurationError("备份文件中缺少配置数据")
            
            # 恢复配置
            config = EnvironmentConfig.from_dict(config_data)
            
            logger.info(f"配置备份恢复成功: {backup_name}")
            return config
            
        except Exception as e:
            logger.error(f"配置备份恢复失败: {e}")
            raise ConfigurationError(f"配置备份恢复失败: {e}")
    
    def list_backups(self, config_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出备份"""
        with self._lock:
            backups = self._backup_history
            if config_name:
                backups = [b for b in backups if b['config_name'] == config_name]
            return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    async def delete_backup(self, backup_name: str) -> bool:
        """删除备份"""
        try:
            backup_file = self.backup_dir / f"{backup_name}.yaml"
            
            if backup_file.exists():
                backup_file.unlink()
            
            # 从历史记录中删除
            with self._lock:
                self._backup_history = [b for b in self._backup_history if b['backup_name'] != backup_name]
            
            logger.info(f"配置备份删除成功: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"配置备份删除失败: {e}")
            return False
    
    def cleanup_old_backups(self, days: int = 30) -> int:
        """清理旧备份"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_count = 0
            
            with self._lock:
                old_backups = [
                    b for b in self._backup_history 
                    if datetime.fromisoformat(b['timestamp']) < cutoff_date
                ]
            
            for backup in old_backups:
                backup_file = Path(backup['file_path'])
                if backup_file.exists():
                    backup_file.unlink()
                    deleted_count += 1
            
            # 从历史记录中删除
            with self._lock:
                self._backup_history = [
                    b for b in self._backup_history 
                    if datetime.fromisoformat(b['timestamp']) >= cutoff_date
                ]
            
            logger.info(f"清理旧备份完成，删除了 {deleted_count} 个备份")
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理旧备份失败: {e}")
            return 0


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._thresholds: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """设置性能阈值"""
        with self._lock:
            self._thresholds[metric_name] = threshold
            logger.debug(f"设置性能阈值: {metric_name} = {threshold}")
    
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录性能指标"""
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            
            metric_data = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "tags": tags or {}
            }
            
            self._metrics[metric_name].append(metric_data)
            
            # 检查阈值
            if metric_name in self._thresholds:
                self._check_threshold(metric_name, value)
            
            # 限制指标历史记录数量
            if len(self._metrics[metric_name]) > 1000:
                self._metrics[metric_name] = self._metrics[metric_name][-1000:]
    
    def _check_threshold(self, metric_name: str, value: float) -> None:
        """检查阈值"""
        threshold = self._thresholds[metric_name]
        
        if value > threshold:
            alert = {
                "alert_type": "threshold_exceeded",
                "metric_name": metric_name,
                "value": value,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
                "severity": "warning" if value < threshold * 1.5 else "critical"
            }
            
            with self._lock:
                self._alerts.append(alert)
            
            logger.warning(f"性能阈值超过: {metric_name} = {value} > {threshold}")
    
    def get_metric_summary(self, metric_name: str, time_window: Optional[int] = None) -> Dict[str, Any]:
        """获取指标摘要"""
        with self._lock:
            if metric_name not in self._metrics:
                return {"error": f"指标不存在: {metric_name}"}
            
            metrics = self._metrics[metric_name]
            
            if time_window:
                cutoff_time = datetime.now() - timedelta(minutes=time_window)
                metrics = [
                    m for m in metrics 
                    if datetime.fromisoformat(m['timestamp']) >= cutoff_time
                ]
            
            if not metrics:
                return {"message": "指定时间窗口内没有指标数据"}
            
            values = [m['value'] for m in metrics]
            
            return {
                "metric_name": metric_name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1],
                "time_window": time_window
            }
    
    def get_all_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取所有告警"""
        with self._lock:
            return sorted(self._alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def clear_alerts(self) -> None:
        """清除告警"""
        with self._lock:
            self._alerts.clear()
            logger.info("已清除所有告警")


class ConfigurationAuditor:
    """配置审计器"""
    
    def __init__(self):
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
    
    def log_action(self, action: str, user: str, details: Dict[str, Any]) -> None:
        """记录审计动作"""
        audit_record = {
            "action": action,
            "user": user,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "session_id": details.get("session_id", "unknown")
        }
        
        with self._lock:
            self._audit_log.append(audit_record)
            
            # 限制审计日志大小
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-5000:]
        
        logger.info(f"审计记录: {action} by {user}")
    
    def get_audit_log(self, user: Optional[str] = None, action: Optional[str] = None, 
                     start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """获取审计日志"""
        with self._lock:
            logs = self._audit_log
            
            if user:
                logs = [log for log in logs if log['user'] == user]
            
            if action:
                logs = [log for log in logs if log['action'] == action]
            
            if start_time:
                logs = [log for log in logs if datetime.fromisoformat(log['timestamp']) >= start_time]
            
            if end_time:
                logs = [log for log in logs if datetime.fromisoformat(log['timestamp']) <= end_time]
            
            return sorted(logs, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def generate_audit_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """生成审计报告"""
        logs = self.get_audit_log(start_time=start_time, end_time=end_time, limit=10000)
        
        if not logs:
            return {"message": "指定时间范围内没有审计记录"}
        
        # 统计信息
        action_counts = {}
        user_counts = {}
        
        for log in logs:
            action = log['action']
            user = log['user']
            
            action_counts[action] = action_counts.get(action, 0) + 1
            user_counts[user] = user_counts.get(user, 0) + 1
        
        return {
            "report_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "summary": {
                "total_actions": len(logs),
                "unique_users": len(user_counts),
                "unique_actions": len(action_counts)
            },
            "top_actions": sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_users": sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "timeline": self._generate_timeline(logs)
        }
    
    def _generate_timeline(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成时间线"""
        timeline = {}
        
        for log in logs:
            timestamp = datetime.fromisoformat(log['timestamp'])
            date_key = timestamp.strftime("%Y-%m-%d")
            hour_key = timestamp.strftime("%H:00")
            
            if date_key not in timeline:
                timeline[date_key] = {}
            
            if hour_key not in timeline[date_key]:
                timeline[date_key][hour_key] = 0
            
            timeline[date_key][hour_key] += 1
        
        return timeline


# 扩展主类功能
class ExtendedEnvironmentConfigurationProcessor(EnvironmentConfigurationProcessor):
    """扩展的环境配置处理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        super().__init__(config_dir)
        
        # 添加新的管理器
        self.validator = ConfigurationValidator()
        self.backup_manager = ConfigurationBackupManager()
        self.performance_monitor = PerformanceMonitor()
        self.auditor = ConfigurationAuditor()
        
        # 注册额外的环境处理器
        self.environment_processors[EnvironmentType.STAGING] = StagingEnvironmentProcessor()
        
        # 注册额外的验证器
        self._register_default_validators()
        
        # 设置默认性能阈值
        self._setup_default_thresholds()
        
        logger.info("扩展环境配置处理器初始化完成")
    
    def _register_default_validators(self) -> None:
        """注册默认验证器"""
        
        def validate_k8s_resources(config: EnvironmentConfig) -> bool:
            """验证K8s资源配置"""
            if 'k8s_resources' in config.parameters:
                resources = config.parameters['k8s_resources']
                if not isinstance(resources, dict):
                    return False
                
                required_resources = ['deployment', 'service', 'ingress']
                for resource in required_resources:
                    if resource not in resources:
                        return False
            
            return True
        
        def validate_resource_limits(config: EnvironmentConfig) -> bool:
            """验证资源限制"""
            if 'resources' in config.parameters:
                resources = config.parameters['resources']
                if 'limits' in resources:
                    limits = resources['limits']
                    if not isinstance(limits, dict):
                        return False
                    
                    # 检查CPU和内存限制
                    if 'cpu' not in limits or 'memory' not in limits:
                        return False
            
            return True
        
        self.validator.register_validator("k8s_resources", validate_k8s_resources)
        self.validator.register_validator("resource_limits", validate_resource_limits)
    
    def _setup_default_thresholds(self) -> None:
        """设置默认性能阈值"""
        self.performance_monitor.set_threshold("response_time", 2.0)  # 2秒
        self.performance_monitor.set_threshold("memory_usage", 80.0)  # 80%
        self.performance_monitor.set_threshold("cpu_usage", 70.0)  # 70%
        self.performance_monitor.set_threshold("disk_usage", 85.0)  # 85%
    
    async def create_environment_with_audit(self, config: EnvironmentConfig, user: str) -> bool:
        """创建环境配置并记录审计"""
        try:
            self.auditor.log_action("create_environment", user, {
                "environment_name": config.name,
                "environment_type": config.type.value
            })
            
            success = await self.create_environment(config)
            
            if success:
                self.auditor.log_action("environment_created", user, {
                    "environment_name": config.name,
                    "environment_type": config.type.value
                })
            
            return success
            
        except Exception as e:
            self.auditor.log_action("environment_creation_failed", user, {
                "environment_name": config.name,
                "error": str(e)
            })
            raise
    
    async def update_environment_with_audit(self, name: str, updates: ConfigDict, user: str) -> bool:
        """更新环境配置并记录审计"""
        try:
            self.auditor.log_action("update_environment", user, {
                "environment_name": name,
                "updates": updates
            })
            
            success = await self.update_environment(name, updates)
            
            if success:
                self.auditor.log_action("environment_updated", user, {
                    "environment_name": name,
                    "updates": updates
                })
            
            return success
            
        except Exception as e:
            self.auditor.log_action("environment_update_failed", user, {
                "environment_name": name,
                "error": str(e)
            })
            raise
    
    async def delete_environment_with_audit(self, name: str, user: str) -> bool:
        """删除环境配置并记录审计"""
        try:
            self.auditor.log_action("delete_environment", user, {
                "environment_name": name
            })
            
            success = await self.delete_environment(name)
            
            if success:
                self.auditor.log_action("environment_deleted", user, {
                    "environment_name": name
                })
            
            return success
            
        except Exception as e:
            self.auditor.log_action("environment_deletion_failed", user, {
                "environment_name": name,
                "error": str(e)
            })
            raise
    
    async def backup_environment(self, name: str, description: str = "") -> str:
        """备份环境配置"""
        try:
            with self._lock:
                if name not in self._environments:
                    raise ConfigurationError(f"环境不存在: {name}")
                
                config = self._environments[name]
            
            backup_name = await self.backup_manager.create_backup(config, description)
            
            self.auditor.log_action("backup_environment", "system", {
                "environment_name": name,
                "backup_name": backup_name,
                "description": description
            })
            
            return backup_name
            
        except Exception as e:
            logger.error(f"环境备份失败: {e}")
            raise ConfigurationError(f"环境备份失败: {e}")
    
    async def restore_environment(self, backup_name: str, new_name: Optional[str] = None) -> bool:
        """恢复环境配置"""
        try:
            config = await self.backup_manager.restore_backup(backup_name)
            
            if not config:
                raise ConfigurationError(f"备份不存在: {backup_name}")
            
            # 如果指定了新名称，更新配置
            if new_name:
                config.name = new_name
            
            success = await self.create_environment(config)
            
            if success:
                self.auditor.log_action("restore_environment", "system", {
                    "backup_name": backup_name,
                    "environment_name": config.name
                })
            
            return success
            
        except Exception as e:
            logger.error(f"环境恢复失败: {e}")
            raise ConfigurationError(f"环境恢复失败: {e}")
    
    def get_performance_metrics(self, metric_name: Optional[str] = None, 
                               time_window: Optional[int] = None) -> Dict[str, Any]:
        """获取性能指标"""
        if metric_name:
            return self.performance_monitor.get_metric_summary(metric_name, time_window)
        else:
            # 返回所有指标的摘要
            summaries = {}
            for metric_name in self.performance_monitor._metrics.keys():
                summaries[metric_name] = self.performance_monitor.get_metric_summary(
                    metric_name, time_window
                )
            return summaries
    
    def get_audit_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """获取审计报告"""
        return self.auditor.generate_audit_report(start_time, end_time)
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """综合健康检查"""
        try:
            logger.info("开始综合健康检查")
            
            # 基础健康检查
            basic_health = await self.check_environment_health()
            
            # 性能指标检查
            performance_metrics = self.get_performance_metrics(time_window=60)  # 最近1小时
            
            # 配置验证
            validation_results = {}
            with self._lock:
                for env_name, config in self._environments.items():
                    errors = self.validator.validate_config(config)
                    validation_results[env_name] = {
                        "valid": len(errors) == 0,
                        "errors": errors
                    }
            
            # 依赖状态检查
            dependency_status = {}
            for env_name, config in self._environments.items():
                deps = self.dependency_manager.list_dependencies()
                dependency_status[env_name] = {
                    "total": len(deps),
                    "installed": len([d for d in deps if d.status == DependencyStatus.INSTALLED]),
                    "outdated": len([d for d in deps if d.status == DependencyStatus.OUTDATED]),
                    "errors": len([d for d in deps if d.status == DependencyStatus.ERROR])
                }
            
            # 生成综合报告
            report = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy",
                "basic_health": basic_health,
                "performance_metrics": performance_metrics,
                "validation_results": validation_results,
                "dependency_status": dependency_status,
                "alerts": self.performance_monitor.get_all_alerts(),
                "summary": {
                    "total_environments": len(self._environments),
                    "healthy_environments": 0,
                    "total_dependencies": 0,
                    "total_alerts": len(self.performance_monitor.get_all_alerts())
                }
            }
            
            # 计算总体状态
            healthy_count = 0
            total_deps = 0
            
            for env_name, result in basic_health.items():
                if isinstance(result, dict) and "error" not in result:
                    healthy_count += 1
            
            for env_name, status in dependency_status.items():
                total_deps += status["total"]
                if status["errors"] == 0:
                    healthy_count += 1
            
            report["summary"]["healthy_environments"] = healthy_count
            report["summary"]["total_dependencies"] = total_deps
            
            # 确定总体状态
            if report["summary"]["total_alerts"] > 0:
                report["overall_status"] = "degraded"
            if healthy_count < len(self._environments):
                report["overall_status"] = "unhealthy"
            
            logger.info(f"综合健康检查完成，总体状态: {report['overall_status']}")
            return report
            
        except Exception as e:
            logger.error(f"综合健康检查失败: {e}")
            raise HealthCheckError(f"综合健康检查失败: {e}")


def create_staging_config(name: str, description: str = "") -> EnvironmentConfig:
    """创建预发布环境配置"""
    config = EnvironmentConfig(
        name=name,
        type=EnvironmentType.STAGING,
        description=description or f"预发布环境 - {name}",
        parameters={
            "staging_mode": True,
            "canary_deployment": True,
            "rollback_enabled": True,
            "traffic_splitting": 10,
            "feature_flags": True,
            "blue_green_deployment": True,
            "k8s_resources": {
                "deployment": {"replicas": 2},
                "service": {"type": "ClusterIP"},
                "ingress": {"enabled": True}
            }
        },
        variables={
            "ENVIRONMENT": "staging",
            "DEBUG": "false",
            "LOG_LEVEL": "info",
            "STAGING_MODE": "true"
        },
        dependencies={
            "istio": {
                "name": "istio",
                "version": "1.15.x",
                "type": "service",
                "install_command": "istioctl install"
            },
            "prometheus": {
                "name": "prometheus",
                "version": "2.40.x",
                "type": "service",
                "install_command": "helm install prometheus"
            }
        },
        monitoring={
            "enhanced_monitoring": True,
            "performance_profiling": True,
            "user_behavior_analytics": True,
            "a_b_testing_support": True
        },
        security={
            "security_scanning": True,
            "vulnerability_assessment": True,
            "access_control": {
                "restricted_access": True,
                "ip_whitelisting": True
            }
        }
    )
    
    return config


async def advanced_example():
    """高级使用示例"""
    
    # 创建扩展的环境配置处理器
    processor = ExtendedEnvironmentConfigurationProcessor("./k8_configs")
    
    try:
        # 创建多种环境配置
        environments = [
            create_development_config("dev-main", "主开发环境"),
            create_testing_config("test-integration", "集成测试环境"),
            create_staging_config("staging-preprod", "预生产环境"),
            create_production_config("prod-main", "主生产环境")
        ]
        
        # 创建环境并记录审计
        for env_config in environments:
            await processor.create_environment_with_audit(env_config, "admin_user")
        
        # 执行综合健康检查
        health_report = await processor.comprehensive_health_check()
        print(f"综合健康检查结果: {health_report['overall_status']}")
        
        # 备份环境
        backup_name = await processor.backup_environment("dev-main", "开发环境初始备份")
        print(f"环境备份完成: {backup_name}")
        
        # 获取性能指标
        metrics = processor.get_performance_metrics("response_time")
        print(f"响应时间指标: {metrics}")
        
        # 获取审计报告
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        audit_report = processor.get_audit_report(start_time, end_time)
        print(f"审计报告: {audit_report['summary']}")
        
        # 环境切换演示
        await processor.switch_environment("dev-main", "test-integration")
        
        # 安装依赖
        install_results = await processor.install_environment_dependencies("test-integration")
        print(f"依赖安装结果: {install_results}")
        
        # 验证所有环境
        for env_name in processor.list_environments():
            errors = await processor.validate_environment(env_name.name)
            if errors:
                print(f"环境 {env_name.name} 验证发现问题: {errors}")
            else:
                print(f"环境 {env_name.name} 验证通过")
        
        # 获取环境摘要
        summary = processor.get_environment_summary()
        print(f"环境摘要: {summary}")
        
        # 记录一些性能指标
        processor.performance_monitor.record_metric("response_time", 1.5, {"env": "test"})
        processor.performance_monitor.record_metric("memory_usage", 65.0, {"env": "test"})
        processor.performance_monitor.record_metric("cpu_usage", 45.0, {"env": "test"})
        
        # 获取更新后的性能指标
        updated_metrics = processor.get_performance_metrics()
        print(f"更新后的性能指标: {updated_metrics}")
        
    except Exception as e:
        logger.error(f"高级示例执行失败: {e}")


async def performance_test_example():
    """性能测试示例"""
    
    processor = ExtendedEnvironmentConfigurationProcessor("./k8_configs")
    
    # 创建大量环境配置进行性能测试
    start_time = time.time()
    
    for i in range(50):  # 创建50个环境配置
        config = EnvironmentConfig(
            name=f"perf-test-{i:03d}",
            type=EnvironmentType.DEVELOPMENT,
            description=f"性能测试环境 {i}",
            parameters={"test_param": f"value_{i}"},
            variables={f"VAR_{i}": f"value_{i}"}
        )
        
        await processor.create_environment(config)
        
        # 记录性能指标
        creation_time = time.time() - start_time
        processor.performance_monitor.record_metric("environment_creation_time", creation_time)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"创建50个环境配置耗时: {total_time:.2f}秒")
    print(f"平均每个环境配置耗时: {total_time/50:.3f}秒")
    
    # 获取性能测试结果
    metrics = processor.get_performance_metrics("environment_creation_time")
    print(f"性能测试指标: {metrics}")


async def stress_test_example():
    """压力测试示例"""
    
    processor = ExtendedEnvironmentConfigurationProcessor("./k8_configs")
    
    # 创建基础环境
    base_config = create_production_config("stress-test-base")
    await processor.create_environment(base_config)
    
    # 并发创建多个环境
    async def create_env_task(env_id: int):
        config = EnvironmentConfig(
            name=f"stress-env-{env_id}",
            type=EnvironmentType.PRODUCTION,
            description=f"压力测试环境 {env_id}",
            parameters={"load_test": True, "concurrent_users": env_id * 10}
        )
        
        start = time.time()
        await processor.create_environment(config)
        end = time.time()
        
        return {
            "env_id": env_id,
            "creation_time": end - start,
            "success": True
        }
    
    # 并发执行10个环境创建任务
    tasks = [create_env_task(i) for i in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_creations = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed_creations = [r for r in results if not isinstance(r, dict) or not r.get("success")]
    
    print(f"压力测试结果:")
    print(f"成功创建: {len(successful_creations)} 个环境")
    print(f"失败创建: {len(failed_creations)} 个环境")
    
    if successful_creations:
        avg_time = sum(r["creation_time"] for r in successful_creations) / len(successful_creations)
        print(f"平均创建时间: {avg_time:.3f}秒")


def demonstrate_error_handling():
    """演示错误处理"""
    
    processor = ExtendedEnvironmentConfigurationProcessor("./k8_configs")
    
    # 演示各种错误情况
    error_scenarios = [
        {
            "name": "invalid_config",
            "description": "无效配置",
            "config": EnvironmentConfig(
                name="",
                type=EnvironmentType.DEVELOPMENT,
                description=""
            )
        },
        {
            "name": "missing_parameters",
            "description": "缺少参数",
            "config": EnvironmentConfig(
                name="test-missing",
                type=EnvironmentType.PRODUCTION,
                description="缺少必要参数的生产环境",
                parameters={}  # 缺少instance_count等必要参数
            )
        },
        {
            "name": "invalid_variables",
            "description": "无效变量",
            "config": EnvironmentConfig(
                name="test-vars",
                type=EnvironmentType.DEVELOPMENT,
                description="包含无效变量的环境",
                variables={
                    "INVALID-VAR-NAME": "value",  # 包含非法字符
                    "": "empty_name"  # 空名称
                }
            )
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n测试错误场景: {scenario['name']} - {scenario['description']}")
        
        try:
            # 验证配置
            errors = processor.validator.validate_config(scenario['config'])
            if errors:
                print(f"验证发现的问题: {errors}")
            else:
                print("配置验证通过")
        except Exception as e:
            print(f"验证过程异常: {e}")


def demonstrate_migration_strategies():
    """演示迁移策略"""
    
    processor = ExtendedEnvironmentConfigurationProcessor("./k8_configs")
    
    # 创建源环境和目标环境
    source_config = create_development_config("migration-source", "迁移源环境")
    target_config = create_production_config("migration-target", "迁移目标环境")
    
    # 模拟迁移过程
    print("演示环境迁移策略:")
    
    migration_steps = [
        "1. 验证源环境和目标环境",
        "2. 备份源环境配置",
        "3. 执行迁移策略",
        "4. 验证迁移结果",
        "5. 更新环境状态"
    ]
    
    for step in migration_steps:
        print(f"  {step}")
        time.sleep(0.1)  # 模拟处理时间
    
    print("迁移策略演示完成")


def demonstrate_monitoring_features():
    """演示监控功能"""
    
    processor = ExtendedEnvironmentConfigurationProcessor("./k8_configs")
    
    print("演示监控功能:")
    
    # 模拟收集各种性能指标
    metrics_to_simulate = [
        ("response_time", [1.2, 1.5, 1.8, 2.1, 1.9]),
        ("memory_usage", [45.0, 52.0, 48.0, 55.0, 50.0]),
        ("cpu_usage", [30.0, 35.0, 40.0, 38.0, 42.0]),
        ("disk_usage", [60.0, 62.0, 65.0, 63.0, 68.0])
    ]
    
    for metric_name, values in metrics_to_simulate:
        print(f"\n{metric_name} 指标:")
        for value in values:
            processor.performance_monitor.record_metric(metric_name, value)
            print(f"  记录值: {value}")
    
    # 获取指标摘要
    print("\n指标摘要:")
    for metric_name, _ in metrics_to_simulate:
        summary = processor.performance_monitor.get_metric_summary(metric_name)
        print(f"  {metric_name}: {summary}")
    
    # 获取告警
    alerts = processor.performance_monitor.get_all_alerts()
    print(f"\n当前告警数量: {len(alerts)}")


def demonstrate_audit_features():
    """演示审计功能"""
    
    processor = ExtendedEnvironmentConfigurationProcessor("./k8_configs")
    
    print("演示审计功能:")
    
    # 模拟审计操作
    audit_actions = [
        ("create_environment", "admin", {"environment_name": "test-env-1"}),
        ("update_environment", "admin", {"environment_name": "test-env-1", "updates": {"description": "更新描述"}}),
        ("delete_environment", "admin", {"environment_name": "test-env-1"}),
        ("backup_environment", "system", {"environment_name": "test-env-2", "backup_name": "backup-001"}),
        ("switch_environment", "admin", {"from": "test-env-1", "to": "test-env-2"})
    ]
    
    for action, user, details in audit_actions:
        processor.auditor.log_action(action, user, details)
        print(f"  记录审计操作: {action} by {user}")
    
    # 生成审计报告
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    report = processor.get_audit_report(start_time, end_time)
    print(f"\n审计报告摘要:")
    print(f"  总操作数: {report['summary']['total_actions']}")
    print(f"  活跃用户: {report['summary']['unique_users']}")
    print(f"  操作类型: {report['summary']['unique_actions']}")


if __name__ == "__main__":
    print("K8环境配置处理器 - 完整演示")
    print("=" * 50)
    
    # 运行各种演示
    try:
        print("\n1. 基础使用示例:")
        asyncio.run(main())
        
        print("\n2. 高级功能示例:")
        asyncio.run(advanced_example())
        
        print("\n3. 性能测试示例:")
        asyncio.run(performance_test_example())
        
        print("\n4. 压力测试示例:")
        asyncio.run(stress_test_example())
        
        print("\n5. 错误处理演示:")
        demonstrate_error_handling()
        
        print("\n6. 迁移策略演示:")
        demonstrate_migration_strategies()
        
        print("\n7. 监控功能演示:")
        demonstrate_monitoring_features()
        
        print("\n8. 审计功能演示:")
        demonstrate_audit_features()
        
        print("\n" + "=" * 50)
        print("所有演示完成!")
        
    except Exception as e:
        logger.error(f"演示执行失败: {e}")