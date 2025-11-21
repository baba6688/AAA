"""
Z3自定义算法插槽系统 - 主要实现

这个模块提供了完整的自定义算法插槽功能，包括算法管理、执行、验证等。
"""

import time
import threading
import json
import inspect
import functools
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Type, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import weakref
import uuid


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlgorithmStatus(Enum):
    """算法状态枚举"""
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ERROR = "error"


class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AlgorithmInfo:
    """算法信息数据类"""
    name: str
    description: str
    version: str
    author: str
    category: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    status: AlgorithmStatus = AlgorithmStatus.REGISTERED
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    usage_count: int = 0
    documentation: str = ""
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'category': self.category,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'status': self.status.value,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'usage_count': self.usage_count,
            'documentation': self.documentation,
            'examples': self.examples
        }


@dataclass
class AlgorithmResult:
    """算法结果数据类"""
    algorithm_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'algorithm_name': self.algorithm_name,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


@dataclass
class ExecutionContext:
    """执行上下文数据类"""
    algorithm_name: str
    parameters: Dict[str, Any]
    timeout: Optional[float] = None
    priority: int = 0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'algorithm_name': self.algorithm_name,
            'parameters': self.parameters,
            'timeout': self.timeout,
            'priority': self.priority,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'metadata': self.metadata
        }


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': self.suggestions
        }


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    original_performance: Dict[str, float]
    optimized_performance: Dict[str, float]
    improvements: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'original_performance': self.original_performance,
            'optimized_performance': self.optimized_performance,
            'improvements': self.improvements,
            'recommendations': self.recommendations
        }


class AlgorithmSlot:
    """算法插槽类"""
    
    def __init__(self, slot_id: str, algorithm_func: Callable, info: AlgorithmInfo):
        self.slot_id = slot_id
        self.algorithm_func = algorithm_func
        self.info = info
        self.is_occupied = True
        self.lock = threading.RLock()
        
    def execute(self, *args, **kwargs) -> AlgorithmResult:
        """执行算法"""
        start_time = time.time()
        try:
            with self.lock:
                if not self.is_occupied:
                    raise ValueError(f"Algorithm slot {self.slot_id} is not occupied")
                
                result = self.algorithm_func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                return AlgorithmResult(
                    algorithm_name=self.info.name,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    metadata={'slot_id': self.slot_id}
                )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Algorithm {self.info.name} execution failed: {str(e)}")
            return AlgorithmResult(
                algorithm_name=self.info.name,
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={'slot_id': self.slot_id}
            )
    
    def release(self):
        """释放插槽"""
        with self.lock:
            self.is_occupied = False
    
    def occupy(self, algorithm_func: Callable, info: AlgorithmInfo):
        """占用插槽"""
        with self.lock:
            self.algorithm_func = algorithm_func
            self.info = info
            self.is_occupied = True


class AlgorithmRegistry:
    """算法注册表"""
    
    def __init__(self):
        self._algorithms: Dict[str, AlgorithmInfo] = {}
        self._functions: Dict[str, Callable] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, func: Callable, info: AlgorithmInfo) -> bool:
        """注册算法"""
        with self._lock:
            if name in self._algorithms:
                logger.warning(f"Algorithm {name} already registered, updating...")
            
            self._algorithms[name] = info
            self._functions[name] = func
            logger.info(f"Algorithm {name} registered successfully")
            return True
    
    def unregister(self, name: str) -> bool:
        """注销算法"""
        with self._lock:
            if name in self._algorithms:
                del self._algorithms[name]
                del self._functions[name]
                logger.info(f"Algorithm {name} unregistered successfully")
                return True
            return False
    
    def get_algorithm(self, name: str) -> Optional[Tuple[Callable, AlgorithmInfo]]:
        """获取算法"""
        with self._lock:
            if name in self._algorithms:
                return self._functions[name], self._algorithms[name]
            return None
    
    def list_algorithms(self) -> List[AlgorithmInfo]:
        """列出所有算法"""
        with self._lock:
            return list(self._algorithms.values())
    
    def search_algorithms(self, query: str) -> List[AlgorithmInfo]:
        """搜索算法"""
        with self._lock:
            query = query.lower()
            results = []
            for info in self._algorithms.values():
                if (query in info.name.lower() or 
                    query in info.description.lower() or
                    query in info.category.lower() or
                    any(query in tag.lower() for tag in info.tags)):
                    results.append(info)
            return results


class AlgorithmExecutor:
    """算法执行器"""
    
    def __init__(self, registry: AlgorithmRegistry):
        self.registry = registry
        self.execution_history: deque = deque(maxlen=1000)
        self.active_executions: Dict[str, ExecutionContext] = {}
        self._lock = threading.Lock()
    
    def execute(self, name: str, parameters: Dict[str, Any], 
                timeout: Optional[float] = None) -> AlgorithmResult:
        """执行算法"""
        with self._lock:
            algorithm_data = self.registry.get_algorithm(name)
            if not algorithm_data:
                return AlgorithmResult(
                    algorithm_name=name,
                    success=False,
                    error=f"Algorithm {name} not found"
                )
            
            func, info = algorithm_data
            
            # 创建执行上下文
            context = ExecutionContext(
                algorithm_name=name,
                parameters=parameters,
                timeout=timeout
            )
            
            execution_id = str(uuid.uuid4())
            self.active_executions[execution_id] = context
            
            try:
                # 执行算法
                start_time = time.time()
                result = func(**parameters)
                execution_time = time.time() - start_time
                
                # 更新算法使用统计
                info.usage_count += 1
                info.last_used = time.time()
                
                algorithm_result = AlgorithmResult(
                    algorithm_name=name,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    metadata={'execution_id': execution_id}
                )
                
                # 添加到执行历史
                self.execution_history.append({
                    'execution_id': execution_id,
                    'algorithm_name': name,
                    'timestamp': time.time(),
                    'execution_time': execution_time,
                    'success': True
                })
                
                return algorithm_result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_result = AlgorithmResult(
                    algorithm_name=name,
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                    metadata={'execution_id': execution_id}
                )
                
                self.execution_history.append({
                    'execution_id': execution_id,
                    'algorithm_name': name,
                    'timestamp': time.time(),
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e)
                })
                
                return error_result
            finally:
                if execution_id in self.active_executions:
                    del self.active_executions[execution_id]
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return list(self.execution_history)[-limit:]
    
    def get_active_executions(self) -> Dict[str, ExecutionContext]:
        """获取活跃执行"""
        return self.active_executions.copy()


class AlgorithmConfig:
    """算法配置管理器"""
    
    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def set_config(self, algorithm_name: str, config: Dict[str, Any]):
        """设置配置"""
        with self._lock:
            self._configs[algorithm_name] = config.copy()
    
    def get_config(self, algorithm_name: str) -> Optional[Dict[str, Any]]:
        """获取配置"""
        with self._lock:
            return self._configs.get(algorithm_name, {}).copy()
    
    def update_config(self, algorithm_name: str, updates: Dict[str, Any]):
        """更新配置"""
        with self._lock:
            if algorithm_name not in self._configs:
                self._configs[algorithm_name] = {}
            self._configs[algorithm_name].update(updates)
    
    def delete_config(self, algorithm_name: str):
        """删除配置"""
        with self._lock:
            if algorithm_name in self._configs:
                del self._configs[algorithm_name]


class AlgorithmValidator:
    """算法验证器"""
    
    def __init__(self, registry: AlgorithmRegistry):
        self.registry = registry
    
    def validate_algorithm(self, name: str) -> ValidationResult:
        """验证算法"""
        algorithm_data = self.registry.get_algorithm(name)
        if not algorithm_data:
            return ValidationResult(
                is_valid=False,
                errors=[f"Algorithm {name} not found"]
            )
        
        func, info = algorithm_data
        errors = []
        warnings = []
        suggestions = []
        
        # 验证算法信息
        if not info.name:
            errors.append("Algorithm name is empty")
        if not info.description:
            warnings.append("Algorithm description is missing")
        if not info.version:
            warnings.append("Algorithm version is missing")
        
        # 验证函数签名
        try:
            sig = inspect.signature(func)
            if len(sig.parameters) == 0:
                warnings.append("Algorithm function has no parameters")
        except Exception as e:
            errors.append(f"Cannot inspect function signature: {str(e)}")
        
        # 验证依赖
        for dep in info.dependencies:
            if not self.registry.get_algorithm(dep):
                errors.append(f"Dependency {dep} not found")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_all(self) -> Dict[str, ValidationResult]:
        """验证所有算法"""
        results = {}
        for info in self.registry.list_algorithms():
            results[info.name] = self.validate_algorithm(info.name)
        return results


class AlgorithmStatistics:
    """算法统计器"""
    
    def __init__(self, executor: AlgorithmExecutor):
        self.executor = executor
        self._lock = threading.RLock()
    
    def get_algorithm_stats(self, algorithm_name: str) -> Dict[str, Any]:
        """获取算法统计"""
        with self._lock:
            history = [h for h in self.executor.execution_history 
                      if h['algorithm_name'] == algorithm_name]
            
            if not history:
                return {'total_executions': 0}
            
            total_executions = len(history)
            successful_executions = len([h for h in history if h['success']])
            failed_executions = total_executions - successful_executions
            
            execution_times = [h['execution_time'] for h in history if h['success']]
            
            stats = {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
                'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                'min_execution_time': min(execution_times) if execution_times else 0,
                'max_execution_time': max(execution_times) if execution_times else 0,
                'last_execution': max(h['timestamp'] for h in history)
            }
            
            return stats
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计"""
        with self._lock:
            history = list(self.executor.execution_history)
            
            if not history:
                return {'total_executions': 0}
            
            total_executions = len(history)
            successful_executions = len([h for h in history if h['success']])
            
            algorithm_usage = defaultdict(int)
            for h in history:
                algorithm_usage[h['algorithm_name']] += 1
            
            return {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': total_executions - successful_executions,
                'success_rate': successful_executions / total_executions,
                'algorithm_usage': dict(algorithm_usage),
                'most_used_algorithm': max(algorithm_usage.items(), key=lambda x: x[1])[0] if algorithm_usage else None
            }


class AlgorithmOptimizer:
    """算法优化器"""
    
    def __init__(self, statistics: AlgorithmStatistics):
        self.statistics = statistics
    
    def optimize_algorithm(self, algorithm_name: str) -> OptimizationResult:
        """优化算法"""
        stats = self.statistics.get_algorithm_stats(algorithm_name)
        
        if stats['total_executions'] == 0:
            return OptimizationResult(
                original_performance={},
                optimized_performance={},
                improvements={},
                recommendations=["No execution history available for optimization"]
            )
        
        original_performance = {
            'average_execution_time': stats['average_execution_time'],
            'success_rate': stats['success_rate']
        }
        
        # 简单的优化建议
        recommendations = []
        improvements = {}
        
        if stats['success_rate'] < 0.9:
            recommendations.append("Consider adding error handling to improve success rate")
            improvements['success_rate_improvement'] = 0.1
        
        if stats['average_execution_time'] > 1.0:
            recommendations.append("Consider optimizing algorithm performance")
            improvements['execution_time_reduction'] = 0.2
        
        optimized_performance = {
            'average_execution_time': original_performance['average_execution_time'] * 0.8,
            'success_rate': min(1.0, original_performance['success_rate'] + 0.05)
        }
        
        return OptimizationResult(
            original_performance=original_performance,
            optimized_performance=optimized_performance,
            improvements=improvements,
            recommendations=recommendations
        )
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        global_stats = self.statistics.get_global_stats()
        
        if global_stats['success_rate'] < 0.95:
            suggestions.append("Overall success rate is low, consider reviewing error handling")
        
        if global_stats['total_executions'] > 100:
            suggestions.append("High execution volume detected, consider implementing caching")
        
        return suggestions


class AlgorithmDocumentation:
    """算法文档管理器"""
    
    def __init__(self, registry: AlgorithmRegistry):
        self.registry = registry
    
    def add_documentation(self, algorithm_name: str, documentation: str, examples: List[str] = None):
        """添加文档"""
        algorithm_data = self.registry.get_algorithm(algorithm_name)
        if algorithm_data:
            _, info = algorithm_data
            info.documentation = documentation
            if examples:
                info.examples = examples
    
    def get_documentation(self, algorithm_name: str) -> Optional[str]:
        """获取文档"""
        algorithm_data = self.registry.get_algorithm(algorithm_name)
        if algorithm_data:
            _, info = algorithm_data
            return info.documentation
        return None
    
    def generate_api_docs(self) -> str:
        """生成API文档"""
        algorithms = self.registry.list_algorithms()
        
        doc_lines = ["# Z3算法插槽API文档\n"]
        
        for info in algorithms:
            doc_lines.append(f"## {info.name} (v{info.version})\n")
            doc_lines.append(f"**描述**: {info.description}\n")
            doc_lines.append(f"**作者**: {info.author}\n")
            doc_lines.append(f"**分类**: {info.category}\n")
            
            if info.parameters:
                doc_lines.append("**参数**:")
                for param_name, param_info in info.parameters.items():
                    doc_lines.append(f"- {param_name}: {param_info}")
                doc_lines.append("")
            
            if info.examples:
                doc_lines.append("**示例**:")
                for example in info.examples:
                    doc_lines.append(f"```python\n{example}\n```")
                doc_lines.append("")
            
            doc_lines.append("---\n")
        
        return "\n".join(doc_lines)


class CustomAlgorithmSlot:
    """自定义算法插槽主类"""
    
    def __init__(self):
        self.registry = AlgorithmRegistry()
        self.executor = AlgorithmExecutor(self.registry)
        self.config = AlgorithmConfig()
        self.validator = AlgorithmValidator(self.registry)
        self.statistics = AlgorithmStatistics(self.executor)
        self.optimizer = AlgorithmOptimizer(self.statistics)
        self.documentation = AlgorithmDocumentation(self.registry)
        self.slots: Dict[str, AlgorithmSlot] = {}
        self._lock = threading.RLock()
    
    def create_slot(self, slot_id: str, algorithm_func: Callable, info: AlgorithmInfo) -> bool:
        """创建算法插槽"""
        with self._lock:
            if slot_id in self.slots:
                logger.warning(f"Slot {slot_id} already exists, updating...")
            
            slot = AlgorithmSlot(slot_id, algorithm_func, info)
            self.slots[slot_id] = slot
            
            # 注册算法
            self.registry.register(info.name, algorithm_func, info)
            
            logger.info(f"Algorithm slot {slot_id} created successfully")
            return True
    
    def register_algorithm(self, name: str, func: Callable, info: AlgorithmInfo) -> bool:
        """注册算法"""
        return self.registry.register(name, func, info)
    
    def execute_algorithm(self, name: str, parameters: Dict[str, Any], 
                         timeout: Optional[float] = None) -> AlgorithmResult:
        """执行算法"""
        return self.executor.execute(name, parameters, timeout)
    
    def execute_slot(self, slot_id: str, *args, **kwargs) -> AlgorithmResult:
        """执行插槽中的算法"""
        with self._lock:
            if slot_id not in self.slots:
                return AlgorithmResult(
                    algorithm_name=slot_id,
                    success=False,
                    error=f"Slot {slot_id} not found"
                )
            
            return self.slots[slot_id].execute(*args, **kwargs)
    
    def configure_algorithm(self, algorithm_name: str, config: Dict[str, Any]):
        """配置算法"""
        self.config.set_config(algorithm_name, config)
    
    def validate_algorithm(self, algorithm_name: str) -> ValidationResult:
        """验证算法"""
        return self.validator.validate_algorithm(algorithm_name)
    
    def get_statistics(self, algorithm_name: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息"""
        if algorithm_name:
            return self.statistics.get_algorithm_stats(algorithm_name)
        else:
            return self.statistics.get_global_stats()
    
    def optimize_algorithm(self, algorithm_name: str) -> OptimizationResult:
        """优化算法"""
        return self.optimizer.optimize_algorithm(algorithm_name)
    
    def add_documentation(self, algorithm_name: str, documentation: str, examples: List[str] = None):
        """添加文档"""
        self.documentation.add_documentation(algorithm_name, documentation, examples)
    
    def generate_documentation(self) -> str:
        """生成文档"""
        return self.documentation.generate_api_docs()
    
    def list_algorithms(self) -> List[AlgorithmInfo]:
        """列出所有算法"""
        return self.registry.list_algorithms()
    
    def search_algorithms(self, query: str) -> List[AlgorithmInfo]:
        """搜索算法"""
        return self.registry.search_algorithms(query)
    
    def get_slot_info(self, slot_id: str) -> Optional[Dict[str, Any]]:
        """获取插槽信息"""
        with self._lock:
            if slot_id in self.slots:
                slot = self.slots[slot_id]
                return {
                    'slot_id': slot.slot_id,
                    'algorithm_name': slot.info.name,
                    'is_occupied': slot.is_occupied,
                    'algorithm_info': slot.info.to_dict()
                }
            return None
    
    def release_slot(self, slot_id: str) -> bool:
        """释放插槽"""
        with self._lock:
            if slot_id in self.slots:
                self.slots[slot_id].release()
                logger.info(f"Slot {slot_id} released")
                return True
            return False
    
    def occupy_slot(self, slot_id: str, algorithm_func: Callable, info: AlgorithmInfo) -> bool:
        """占用插槽"""
        with self._lock:
            if slot_id not in self.slots:
                return False
            
            self.slots[slot_id].occupy(algorithm_func, info)
            self.registry.register(info.name, algorithm_func, info)
            logger.info(f"Slot {slot_id} occupied with algorithm {info.name}")
            return True
    
    def export_config(self) -> str:
        """导出配置"""
        config_data = {
            'algorithms': [info.to_dict() for info in self.registry.list_algorithms()],
            'slots': {slot_id: self.get_slot_info(slot_id) for slot_id in self.slots},
            'statistics': self.get_statistics(),
            'timestamp': time.time()
        }
        return json.dumps(config_data, indent=2, ensure_ascii=False)
    
    def import_config(self, config_json: str) -> bool:
        """导入配置"""
        try:
            config_data = json.loads(config_json)
            
            # 导入算法信息
            for algo_data in config_data.get('algorithms', []):
                info = AlgorithmInfo(**algo_data)
                # 这里需要重新创建函数，实际应用中需要从其他地方获取
                logger.info(f"Would import algorithm {info.name}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to import config: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'total_algorithms': len(self.registry.list_algorithms()),
            'total_slots': len(self.slots),
            'occupied_slots': len([s for s in self.slots.values() if s.is_occupied]),
            'total_executions': len(self.executor.execution_history),
            'active_executions': len(self.executor.active_executions),
            'system_uptime': time.time() - getattr(self, '_start_time', time.time()),
            'timestamp': time.time()
        }


# 装饰器函数
def algorithm_slot(slot_id: str, info: AlgorithmInfo):
    """算法插槽装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 这里需要访问全局的 CustomAlgorithmSlot 实例
            # 实际使用中需要通过其他方式传递
            return func(*args, **kwargs)
        
        # 将算法信息附加到函数上
        wrapper._algorithm_info = info
        wrapper._slot_id = slot_id
        return wrapper
    return decorator


def register_algorithm(name: str, category: str = "general", version: str = "1.0.0"):
    """算法注册装饰器"""
    def decorator(func: Callable) -> Callable:
        info = AlgorithmInfo(
            name=name,
            description=func.__doc__ or f"Algorithm {name}",
            version=version,
            author="Unknown",
            category=category
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._algorithm_info = info
        return wrapper
    return decorator