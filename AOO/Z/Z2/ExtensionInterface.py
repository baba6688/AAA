"""
Z2 扩展接口系统 - 主要实现

这个模块提供了完整的扩展接口管理功能，包括：
- 接口定义和标准
- 扩展点注册和管理
- 接口适配和转换
- 接口验证和检查
- 接口文档生成
- 版本管理
- 统计分析
- 性能优化
"""

import abc
import inspect
import json
import logging
import threading
import time
import weakref
from collections import defaultdict, Counter
from datetime import datetime
from typing import (
    Any, Dict, List, Optional, Type, Callable, Union, Set,
    Generic, TypeVar, Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from enum import Enum


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


T = TypeVar('T')


class InterfaceStatus(Enum):
    """接口状态枚举"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


class CompatibilityLevel(Enum):
    """兼容性级别枚举"""
    FULL = "full"
    PARTIAL = "partial"
    INCOMPATIBLE = "incompatible"


@dataclass
class InterfaceMetadata:
    """接口元数据"""
    name: str
    version: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    status: InterfaceStatus
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    compatibility: Dict[str, CompatibilityLevel] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.value,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'compatibility': {k: v.value for k, v in self.compatibility.items()}
        }


@dataclass
class InterfaceDefinition:
    """接口定义"""
    interface: Type
    metadata: InterfaceMetadata
    methods: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.methods:
            self.methods = [name for name, method in inspect.getmembers(
                self.interface, predicate=inspect.isfunction
            )]
        if not self.properties:
            self.properties = [name for name, prop in inspect.getmembers(
                self.interface, predicate=lambda x: isinstance(x, property)
            )]
    
    def validate_interface(self) -> List[str]:
        """验证接口定义"""
        errors = []
        
        # 检查是否是抽象基类
        if not inspect.isabstract(self.interface):
            errors.append(f"接口 {self.metadata.name} 不是抽象基类")
        
        # 检查方法签名
        for method_name in self.methods:
            if not hasattr(self.interface, method_name):
                errors.append(f"方法 {method_name} 不存在于接口中")
            else:
                method = getattr(self.interface, method_name)
                if not callable(method):
                    errors.append(f"{method_name} 不是可调用方法")
        
        return errors


class ExtensionPoint(Generic[T]):
    """扩展点管理"""
    
    def __init__(self, name: str, interface: Type[T]):
        self.name = name
        self.interface = interface
        self._extensions: Dict[str, T] = {}
        self._lock = threading.RLock()
        self._observers: List[Callable] = []
    
    def register(self, extension_id: str, extension: T) -> bool:
        """注册扩展"""
        with self._lock:
            if not isinstance(extension, self.interface):
                logger.error(f"扩展 {extension_id} 不符合接口 {self.interface.__name__}")
                return False
            
            self._extensions[extension_id] = extension
            logger.info(f"扩展点 {self.name} 注册扩展: {extension_id}")
            
            # 通知观察者
            self._notify_observers('register', extension_id, extension)
            return True
    
    def unregister(self, extension_id: str) -> bool:
        """注销扩展"""
        with self._lock:
            if extension_id in self._extensions:
                extension = self._extensions.pop(extension_id)
                logger.info(f"扩展点 {self.name} 注销扩展: {extension_id}")
                
                # 通知观察者
                self._notify_observers('unregister', extension_id, extension)
                return True
            return False
    
    def get_extension(self, extension_id: str) -> Optional[T]:
        """获取扩展"""
        return self._extensions.get(extension_id)
    
    def get_all_extensions(self) -> Dict[str, T]:
        """获取所有扩展"""
        return self._extensions.copy()
    
    def add_observer(self, observer: Callable):
        """添加观察者"""
        self._observers.append(observer)
    
    def remove_observer(self, observer: Callable):
        """移除观察者"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, event: str, extension_id: str, extension: T):
        """通知观察者"""
        for observer in self._observers:
            try:
                observer(event, extension_id, extension, self)
            except Exception as e:
                logger.error(f"观察者通知失败: {e}")


class ExtensionRegistry:
    """扩展注册表"""
    
    def __init__(self):
        self._extension_points: Dict[str, ExtensionPoint] = {}
        self._lock = threading.RLock()
    
    def create_extension_point(self, name: str, interface: Type) -> ExtensionPoint:
        """创建扩展点"""
        with self._lock:
            if name in self._extension_points:
                raise ValueError(f"扩展点 {name} 已存在")
            
            extension_point = ExtensionPoint(name, interface)
            self._extension_points[name] = extension_point
            logger.info(f"创建扩展点: {name}")
            return extension_point
    
    def get_extension_point(self, name: str) -> Optional[ExtensionPoint]:
        """获取扩展点"""
        return self._extension_points.get(name)
    
    def list_extension_points(self) -> List[str]:
        """列出所有扩展点"""
        return list(self._extension_points.keys())


class InterfaceAdapter:
    """接口适配器"""
    
    def __init__(self):
        self._adapters: Dict[Type, Type] = {}
        self._lock = threading.RLock()
    
    def register_adapter(self, source_interface: Type, target_interface: Type, adapter_class: Type):
        """注册适配器"""
        with self._lock:
            self._adapters[(source_interface, target_interface)] = adapter_class
            logger.info(f"注册适配器: {source_interface.__name__} -> {target_interface.__name__}")
    
    def adapt(self, obj: Any, target_interface: Type) -> Optional[Any]:
        """适配对象到目标接口"""
        source_interface = type(obj)
        
        # 查找适配器
        adapter_class = self._adapters.get((source_interface, target_interface))
        if not adapter_class:
            # 尝试查找通用适配器
            for (src, tgt), adapter in self._adapters.items():
                if issubclass(source_interface, src) and issubclass(target_interface, tgt):
                    adapter_class = adapter
                    break
        
        if not adapter_class:
            logger.warning(f"找不到适配器: {source_interface.__name__} -> {target_interface.__name__}")
            return None
        
        try:
            adapter_instance = adapter_class()
            return adapter_instance.adapt(obj)
        except Exception as e:
            logger.error(f"适配失败: {e}")
            return None
    
    def can_adapt(self, source_interface: Type, target_interface: Type) -> bool:
        """检查是否可以适配"""
        return (source_interface, target_interface) in self._adapters


class InterfaceValidator:
    """接口验证器"""
    
    @staticmethod
    def validate_implementation(interface: Type, implementation: Any) -> List[str]:
        """验证实现是否符合接口"""
        errors = []
        
        if not hasattr(implementation, '__class__'):
            errors.append("实现不是有效的对象")
            return errors
        
        impl_class = implementation.__class__
        
        # 检查所有抽象方法是否被实现
        for name, method in inspect.getmembers(interface, predicate=inspect.isfunction):
            if getattr(interface, name, None) is not None:
                if not hasattr(impl_class, name):
                    errors.append(f"方法 {name} 未实现")
                else:
                    impl_method = getattr(impl_class, name)
                    if not callable(impl_method):
                        errors.append(f"{name} 不是可调用方法")
        
        # 检查属性
        for name, prop in inspect.getmembers(interface, predicate=lambda x: isinstance(x, property)):
            if not hasattr(impl_class, name):
                errors.append(f"属性 {name} 未实现")
        
        return errors
    
    @staticmethod
    def validate_signature_compatibility(interface_method: Callable, impl_method: Callable) -> bool:
        """验证方法签名兼容性"""
        try:
            interface_sig = inspect.signature(interface_method)
            impl_sig = inspect.signature(impl_method)
            
            # 检查参数数量
            if len(interface_sig.parameters) != len(impl_sig.parameters):
                return False
            
            # 检查参数类型注解
            for (name, interface_param), (impl_name, impl_param) in zip(
                interface_sig.parameters.items(), impl_sig.parameters.items()
            ):
                if (interface_param.annotation != inspect.Parameter.empty and
                    impl_param.annotation != inspect.Parameter.empty and
                    not issubclass(impl_param.annotation, interface_param.annotation)):
                    return False
            
            return True
        except Exception:
            return False


class InterfaceDocumenter:
    """接口文档生成器"""
    
    def __init__(self):
        self._document_templates = {
            'markdown': self._generate_markdown_doc,
            'html': self._generate_html_doc,
            'json': self._generate_json_doc
        }
    
    def generate_documentation(self, interface_def: InterfaceDefinition, 
                             format_type: str = 'markdown') -> str:
        """生成接口文档"""
        generator = self._document_templates.get(format_type)
        if not generator:
            raise ValueError(f"不支持的文档格式: {format_type}")
        
        return generator(interface_def)
    
    def _generate_markdown_doc(self, interface_def: InterfaceDefinition) -> str:
        """生成Markdown格式文档"""
        md = interface_def.metadata
        
        doc = f"""# {md.name} 接口文档

## 概述
{md.description}

## 基本信息
- **版本**: {md.version}
- **作者**: {md.author}
- **状态**: {md.status.value}
- **创建时间**: {md.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **更新时间**: {md.updated_at.strftime('%Y-%m-%d %H:%M:%S')}

## 标签
{', '.join(md.tags) if md.tags else '无'}

## 依赖
{', '.join(md.dependencies) if md.dependencies else '无'}

## 方法列表
"""
        
        for method_name in interface_def.methods:
            doc += f"- `{method_name}()`\n"
        
        doc += "\n## 属性列表\n"
        for prop_name in interface_def.properties:
            doc += f"- `{prop_name}`\n"
        
        if interface_def.events:
            doc += "\n## 事件列表\n"
            for event_name in interface_def.events:
                doc += f"- `{event_name}`\n"
        
        return doc
    
    def _generate_html_doc(self, interface_def: InterfaceDefinition) -> str:
        """生成HTML格式文档"""
        md = interface_def.metadata
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{md.name} 接口文档</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .method {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 3px solid #007acc; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{md.name} 接口文档</h1>
        <p>{md.description}</p>
    </div>
    
    <div class="section">
        <h2>基本信息</h2>
        <ul>
            <li><strong>版本:</strong> {md.version}</li>
            <li><strong>作者:</strong> {md.author}</li>
            <li><strong>状态:</strong> {md.status.value}</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>方法列表</h2>
        {''.join(f'<div class="method">{method}</div>' for method in interface_def.methods)}
    </div>
</body>
</html>
"""
        return html
    
    def _generate_json_doc(self, interface_def: InterfaceDefinition) -> str:
        """生成JSON格式文档"""
        return json.dumps({
            'interface': interface_def.metadata.to_dict(),
            'methods': interface_def.methods,
            'properties': interface_def.properties,
            'events': interface_def.events
        }, ensure_ascii=False, indent=2)


class VersionManager:
    """版本管理器"""
    
    def __init__(self):
        self._versions: Dict[str, str] = {}
        self._compatibility_matrix: Dict[str, Dict[str, CompatibilityLevel]] = {}
        self._lock = threading.RLock()
    
    def register_version(self, interface_name: str, version: str):
        """注册接口版本"""
        with self._lock:
            self._versions[interface_name] = version
            logger.info(f"注册接口版本: {interface_name} = {version}")
    
    def get_version(self, interface_name: str) -> Optional[str]:
        """获取接口版本"""
        return self._versions.get(interface_name)
    
    def set_compatibility(self, interface_name: str, target_version: str, 
                         compatibility: CompatibilityLevel):
        """设置兼容性"""
        with self._lock:
            if interface_name not in self._compatibility_matrix:
                self._compatibility_matrix[interface_name] = {}
            self._compatibility_matrix[interface_name][target_version] = compatibility
            logger.info(f"设置兼容性: {interface_name} -> {target_version} = {compatibility.value}")
    
    def check_compatibility(self, interface_name: str, version1: str, version2: str) -> CompatibilityLevel:
        """检查版本兼容性"""
        matrix = self._compatibility_matrix.get(interface_name, {})
        return matrix.get(version2, CompatibilityLevel.INCOMPATIBLE)
    
    def is_compatible(self, interface_name: str, version1: str, version2: str) -> bool:
        """检查是否兼容"""
        compatibility = self.check_compatibility(interface_name, version1, version2)
        return compatibility in [CompatibilityLevel.FULL, CompatibilityLevel.PARTIAL]


class InterfaceStats:
    """接口统计分析"""
    
    def __init__(self):
        self._call_counts: Counter = Counter()
        self._error_counts: Counter = Counter()
        self._response_times: List[float] = []
        self._lock = threading.RLock()
    
    def record_call(self, interface_name: str, method_name: str):
        """记录接口调用"""
        with self._lock:
            key = f"{interface_name}.{method_name}"
            self._call_counts[key] += 1
    
    def record_error(self, interface_name: str, method_name: str, error_type: str):
        """记录错误"""
        with self._lock:
            key = f"{interface_name}.{method_name}.{error_type}"
            self._error_counts[key] += 1
    
    def record_response_time(self, response_time: float):
        """记录响应时间"""
        with self._lock:
            self._response_times.append(response_time)
            # 保持最近1000次记录
            if len(self._response_times) > 1000:
                self._response_times.pop(0)
    
    def get_call_stats(self) -> Dict[str, int]:
        """获取调用统计"""
        return dict(self._call_counts)
    
    def get_error_stats(self) -> Dict[str, int]:
        """获取错误统计"""
        return dict(self._error_counts)
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """获取响应时间统计"""
        if not self._response_times:
            return {}
        
        times = self._response_times
        return {
            'min': min(times),
            'max': max(times),
            'avg': sum(times) / len(times),
            'count': len(times)
        }
    
    def reset_stats(self):
        """重置统计"""
        with self._lock:
            self._call_counts.clear()
            self._error_counts.clear()
            self._response_times.clear()


class InterfaceOptimizer:
    """接口性能优化器"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}
        self._optimization_rules: List[Callable] = []
        self._lock = threading.RLock()
    
    def add_optimization_rule(self, rule: Callable):
        """添加优化规则"""
        self._optimization_rules.append(rule)
    
    def cache_result(self, key: str, result: Any, ttl: float = 300):
        """缓存结果"""
        with self._lock:
            self._cache[key] = result
            self._cache_ttl[key] = time.time() + ttl
            logger.debug(f"缓存结果: {key}")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """获取缓存结果"""
        with self._lock:
            if key in self._cache:
                if time.time() < self._cache_ttl[key]:
                    logger.debug(f"缓存命中: {key}")
                    return self._cache[key]
                else:
                    # 缓存过期，清理
                    del self._cache[key]
                    del self._cache_ttl[key]
            return None
    
    def clear_cache(self):
        """清理缓存"""
        with self._lock:
            self._cache.clear()
            self._cache_ttl.clear()
    
    def optimize_call(self, func: Callable, *args, **kwargs) -> Any:
        """优化方法调用"""
        # 应用优化规则
        for rule in self._optimization_rules:
            try:
                result = rule(func, *args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"优化规则失败: {e}")
        
        # 执行原始方法
        return func(*args, **kwargs)


class ExtensionInterface:
    """扩展接口主类"""
    
    def __init__(self):
        self.registry = ExtensionRegistry()
        self.adapter = InterfaceAdapter()
        self.validator = InterfaceValidator()
        self.documenter = InterfaceDocumenter()
        self.version_manager = VersionManager()
        self.stats = InterfaceStats()
        self.optimizer = InterfaceOptimizer()
        self._interface_definitions: Dict[str, InterfaceDefinition] = {}
        self._lock = threading.RLock()
    
    def define_interface(self, name: str, interface: Type, metadata: InterfaceMetadata) -> InterfaceDefinition:
        """定义接口"""
        with self._lock:
            if name in self._interface_definitions:
                raise ValueError(f"接口 {name} 已存在")
            
            interface_def = InterfaceDefinition(interface, metadata)
            
            # 验证接口定义
            errors = interface_def.validate_interface()
            if errors:
                raise ValueError(f"接口定义验证失败: {errors}")
            
            self._interface_definitions[name] = interface_def
            
            # 注册版本
            self.version_manager.register_version(name, metadata.version)
            
            logger.info(f"定义接口: {name}")
            return interface_def
    
    def get_interface_definition(self, name: str) -> Optional[InterfaceDefinition]:
        """获取接口定义"""
        return self._interface_definitions.get(name)
    
    def list_interfaces(self) -> List[str]:
        """列出所有接口"""
        return list(self._interface_definitions.keys())
    
    def generate_documentation(self, interface_name: str, format_type: str = 'markdown') -> Optional[str]:
        """生成接口文档"""
        interface_def = self._interface_definitions.get(interface_name)
        if not interface_def:
            logger.error(f"接口 {interface_name} 不存在")
            return None
        
        return self.documenter.generate_documentation(interface_def, format_type)
    
    def export_documentation(self, interface_name: str, format_type: str = 'markdown', 
                           output_file: Optional[str] = None) -> bool:
        """导出接口文档"""
        doc = self.generate_documentation(interface_name, format_type)
        if not doc:
            return False
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(doc)
                logger.info(f"文档已导出到: {output_file}")
                return True
            except Exception as e:
                logger.error(f"导出文档失败: {e}")
                return False
        else:
            print(doc)
            return True
    
    def validate_implementation(self, interface_name: str, implementation: Any) -> List[str]:
        """验证实现"""
        interface_def = self._interface_definitions.get(interface_name)
        if not interface_def:
            return [f"接口 {interface_name} 不存在"]
        
        return self.validator.validate_implementation(interface_def.interface, implementation)
    
    def adapt_interface(self, obj: Any, target_interface: Type) -> Optional[Any]:
        """适配接口"""
        return self.adapter.adapt(obj, target_interface)
    
    def check_version_compatibility(self, interface_name: str, version1: str, version2: str) -> bool:
        """检查版本兼容性"""
        return self.version_manager.is_compatible(interface_name, version1, version2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'calls': self.stats.get_call_stats(),
            'errors': self.stats.get_error_stats(),
            'response_times': self.stats.get_response_time_stats(),
            'interfaces': len(self._interface_definitions),
            'extension_points': len(self.registry.list_extension_points())
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats.reset_stats()
    
    def optimize_performance(self):
        """优化性能"""
        # 清理过期缓存
        current_time = time.time()
        with self.optimizer._lock:
            expired_keys = [
                key for key, ttl in self.optimizer._cache_ttl.items()
                if current_time >= ttl
            ]
            for key in expired_keys:
                del self.optimizer._cache[key]
                del self.optimizer._cache_ttl[key]
        
        logger.info("性能优化完成")


# 全局扩展接口实例
extension_interface = ExtensionInterface()


# 装饰器函数
def define_extension_point(name: str, interface: Type):
    """定义扩展点装饰器"""
    def decorator(func):
        extension_point = extension_interface.registry.create_extension_point(name, interface)
        func._extension_point = extension_point
        return func
    return decorator


def implement_interface(interface_name: str):
    """实现接口装饰器"""
    def decorator(cls):
        interface_def = extension_interface.get_interface_definition(interface_name)
        if interface_def:
            errors = extension_interface.validate_implementation(interface_name, cls())
            if errors:
                logger.warning(f"接口实现验证警告: {errors}")
        return cls
    return decorator


def measure_performance(func: Callable) -> Callable:
    """性能测量装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            extension_interface.stats.record_call(func.__name__, 'call')
            return result
        except Exception as e:
            extension_interface.stats.record_error(func.__name__, 'call', type(e).__name__)
            raise
        finally:
            end_time = time.time()
            extension_interface.stats.record_response_time(end_time - start_time)
    return wrapper