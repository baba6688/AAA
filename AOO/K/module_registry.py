"""
模块注册表
负责管理所有自动发现的模块和类，提供高效的查询和管理功能
支持依赖关系跟踪、版本管理和生命周期状态监控
"""

import threading
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, OrderedDict
from threading import RLock
import hashlib
import json

class RegistrationStatus(Enum):
    """注册状态枚举"""
    PENDING = "pending"
    REGISTERED = "registered"
    ACTIVATED = "activated"
    SUSPENDED = "suspended"
    UNREGISTERED = "unregistered"
    ERROR = "error"

class DependencyType(Enum):
    """依赖类型枚举"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    PROVIDED = "provided"

@dataclass
class DependencyInfo:
    """依赖信息数据类"""
    name: str
    type: DependencyType
    version: str = "*"
    optional: bool = False
    description: str = ""
    satisfied: bool = False
    provider: str = ""

@dataclass
class ModuleVersion:
    """模块版本数据类"""
    major: int
    minor: int
    patch: int
    build: str = ""
    
    def __str__(self):
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}+{self.build}" if self.build else base
    
    def compatible_with(self, other: 'ModuleVersion') -> bool:
        """检查版本兼容性"""
        return self.major == other.major and self.minor == other.minor

@dataclass
class ModuleRegistration:
    """模块注册信息数据类"""
    # 基本信息
    id: str
    zone: str
    module_name: str
    full_path: str
    version: ModuleVersion
    
    # 类信息
    classes: List[Dict[str, Any]]
    discoverable_classes: List[Dict[str, Any]]
    
    # 状态信息
    status: RegistrationStatus
    registration_time: float
    activation_time: Optional[float]
    
    # 依赖信息
    dependencies: List[DependencyInfo]
    dependents: List[str]  # 依赖此模块的模块ID列表
    
    # 元数据
    metadata: Dict[str, Any]
    tags: Set[str]
    
    # 性能统计
    access_count: int
    last_access_time: float
    creation_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['version'] = str(self.version)
        return data

class ModuleRegistry:
    """模块注册表 - 生产环境级别实现"""
    
    
    def get_all_classes(self):
        """获取所有已注册的类"""
        return getattr(self, '_classes', {})
    
    def get_class_info(self, class_name):
        """获取指定类的信息"""
        classes = getattr(self, 'classes', {})
        return classes.get(class_name)
    
    def get_classes_by_zone(self, zone):
        """获取指定区域的所有类"""
        classes = getattr(self, 'classes', {})
        return [cls for cls in classes.values() if getattr(cls, 'zone', None) == zone]
    
    def get_discoverable_classes(self):
        """获取所有可发现类"""
        classes = getattr(self, 'classes', {})
        return [cls for cls in classes.values() if getattr(cls, 'is_discoverable', False)]

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        
        # 核心存储结构
        self._modules = OrderedDict()  # 模块ID -> ModuleRegistration
        self._classes = {}  # 类名 -> 模块ID
        self._zone_index = defaultdict(OrderedDict)  # 区域 -> 模块名称 -> 模块ID
        self._discoverable_index = defaultdict(list)  # 区域 -> 可发现类列表
        
        # 依赖关系图
        self._dependency_graph = defaultdict(set)  # 模块ID -> 依赖的模块ID集合
        self._reverse_dependency_graph = defaultdict(set)  # 模块ID -> 被依赖的模块ID集合
        
        # 缓存和索引
        self._name_index = {}  # 模块名称 -> 模块ID集合
        self._tag_index = defaultdict(set)  # 标签 -> 模块ID集合
        self._version_index = defaultdict(dict)  # 模块名称 -> 版本 -> 模块ID
        
        # 状态管理
        self._status_counters = defaultdict(int)
        self._module_states = {}  # 模块ID -> 状态历史
        
        # 线程安全
        self._lock = RLock()
        self._version_lock = RLock()
        
        # 配置
        self.auto_resolve_dependencies = True
        self.enable_version_management = True
        self.allow_multiple_versions = False
        self.default_version_constraint = ">=1.0.0"
        
        # 性能监控
        self._stats = {
            'total_registrations': 0,
            'total_unregistrations': 0,
            'total_queries': 0,
            'dependency_resolutions': 0,
            'conflict_resolutions': 0,
            'start_time': time.time()
        }
        
        # 事件回调
        self._registration_callbacks = []
        self._unregistration_callbacks = []
        self._activation_callbacks = []
        
        # 日志
        self.logger = logging.getLogger('AOO.Registry')
        
        # 初始化
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        if self.config_manager:
            try:
                registry_config = self.config_manager.get_section('registry') or {}
                self.auto_resolve_dependencies = registry_config.get('auto_resolve_dependencies', True)
                self.enable_version_management = registry_config.get('enable_version_management', True)
                self.allow_multiple_versions = registry_config.get('allow_multiple_versions', False)
                self.default_version_constraint = registry_config.get('default_version_constraint', ">=1.0.0")
            except Exception as e:
                self.logger.warning(f"加载注册表配置失败: {e}")
    
    def register_module(self, module_path: str, zone: str, module_info: Dict[str, Any]) -> str:
        """
        注册模块
        
        Args:
            module_path: 模块文件路径
            zone: 区域名称
            module_info: 模块信息
            
        Returns:
            str: 模块注册ID
        """
        with self._lock:
            try:
                # 生成模块ID
                module_id = self._generate_module_id(zone, module_info.name)
                
                # 检查是否已注册
                if module_id in self._modules:
                    self.logger.warning(f"模块已注册: {module_id}")
                    return module_id
                
                # 解析版本信息
                version = self._parse_version(getattr(module_info, 'version', 'unknown'))
                
                # 创建注册信息
                registration = ModuleRegistration(
                    id=module_id,
                    zone=zone,
                    module_name=module_info.name,
                    full_path=module_path,
                    version=version,
                    classes=getattr(module_info, 'classes', 'unknown'),
                    discoverable_classes=getattr(module_info, 'discoverable_classes', 'unknown'),
                    status=RegistrationStatus.REGISTERED,
                    registration_time=time.time(),
                    activation_time=None,
                    dependencies=[],
                    dependents=[],
                    metadata=getattr(module_info, 'metadata', 'unknown'),
                    tags=set(getattr(module_info, 'tags', 'unknown')),
                    access_count=0,
                    last_access_time=time.time(),
                    creation_time=time.time()
                )
                
                # 解析依赖关系
                registration.dependencies = self._parse_dependencies(module_info, registration)
                
                # 存储模块信息
                self._modules[module_id] = registration
                
                # 更新索引
                self._update_indexes(registration)
                
                # 解析依赖关系图
                self._update_dependency_graph(registration)
                
                # 自动解析依赖（如果启用）
                if self.auto_resolve_dependencies:
                    self._resolve_dependencies(registration)
                
                # 更新统计
                self._stats['total_registrations'] += 1
                self._status_counters[RegistrationStatus.REGISTERED] += 1
                
                # 记录状态历史
                self._module_states[module_id] = [{
                    'status': RegistrationStatus.REGISTERED,
                    'timestamp': time.time(),
                    'reason': 'initial_registration'
                }]
                
                # 触发注册回调
                self._trigger_registration_callbacks(registration)
                
                self.logger.info(f"模块注册成功: {module_id} (v{version})")
                return module_id
                
            except Exception as e:
                self.logger.error(f"模块注册失败 {zone}.{module_info.name}: {e}")
                raise
    
    def unregister_module(self, module_id: str, reason: str = "manual") -> bool:
        """
        注销模块
        
        Args:
            module_id: 模块ID
            reason: 注销原因
            
        Returns:
            bool: 是否成功注销
        """
        with self._lock:
            if module_id not in self._modules:
                self.logger.warning(f"模块未注册: {module_id}")
                return False
            
            try:
                registration = self._modules[module_id]
                
                # 检查是否有依赖此模块的其他模块
                dependents = self._reverse_dependency_graph.get(module_id, set())
                if dependents and not self._force_unregister:
                    self.logger.error(f"无法注销模块 {module_id}: 有 {len(dependents)} 个模块依赖于此模块")
                    return False
                
                # 更新依赖关系图
                for dep_id in self._dependency_graph.get(module_id, set()):
                    self._reverse_dependency_graph[dep_id].discard(module_id)
                
                # 清理索引
                self._remove_from_indexes(registration)
                
                # 清理依赖关系图
                if module_id in self._dependency_graph:
                    del self._dependency_graph[module_id]
                if module_id in self._reverse_dependency_graph:
                    del self._reverse_dependency_graph[module_id]
                
                # 更新状态
                registration.status = RegistrationStatus.UNREGISTERED
                self._update_state_history(module_id, RegistrationStatus.UNREGISTERED, reason)
                
                # 更新统计
                self._status_counters[registration.status] -= 1
                self._status_counters[RegistrationStatus.UNREGISTERED] += 1
                self._stats['total_unregistrations'] += 1
                
                # 触发注销回调
                self._trigger_unregistration_callbacks(registration, reason)
                
                # 从主存储中移除
                del self._modules[module_id]
                
                self.logger.info(f"模块注销成功: {module_id} ({reason})")
                return True
                
            except Exception as e:
                self.logger.error(f"模块注销失败 {module_id}: {e}")
                return False
    
    def activate_module(self, module_id: str) -> bool:
        """
        激活模块
        
        Args:
            module_id: 模块ID
            
        Returns:
            bool: 是否成功激活
        """
        with self._lock:
            if module_id not in self._modules:
                self.logger.error(f"模块未注册: {module_id}")
                return False
            
            registration = self._modules[module_id]
            
            # 检查依赖是否满足
            if not self._check_dependencies_satisfied(registration):
                self.logger.error(f"无法激活模块 {module_id}: 依赖不满足")
                return False
            
            try:
                # 更新状态
                old_status = registration.status
                registration.status = RegistrationStatus.ACTIVATED
                registration.activation_time = time.time()
                
                # 更新状态历史
                self._update_state_history(module_id, RegistrationStatus.ACTIVATED, "manual_activation")
                
                # 更新统计
                self._status_counters[old_status] -= 1
                self._status_counters[RegistrationStatus.ACTIVATED] += 1
                
                # 触发激活回调
                self._trigger_activation_callbacks(registration)
                
                self.logger.info(f"模块激活成功: {module_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"模块激活失败 {module_id}: {e}")
                registration.status = RegistrationStatus.ERROR
                self._update_state_history(module_id, RegistrationStatus.ERROR, f"activation_failed: {e}")
                return False
    
    def suspend_module(self, module_id: str, reason: str = "manual") -> bool:
        """
        挂起模块
        
        Args:
            module_id: 模块ID
            reason: 挂起原因
            
        Returns:
            bool: 是否成功挂起
        """
        with self._lock:
            if module_id not in self._modules:
                return False
            
            registration = self._modules[module_id]
            
            # 检查是否有激活的依赖模块
            active_dependents = [
                dep_id for dep_id in self._reverse_dependency_graph.get(module_id, set())
                if self._modules[dep_id].status == RegistrationStatus.ACTIVATED
            ]
            
            if active_dependents:
                self.logger.warning(f"挂起模块 {module_id}: 有 {len(active_dependents)} 个激活模块依赖于此模块")
            
            try:
                # 更新状态
                old_status = registration.status
                registration.status = RegistrationStatus.SUSPENDED
                
                # 更新状态历史
                self._update_state_history(module_id, RegistrationStatus.SUSPENDED, reason)
                
                # 更新统计
                self._status_counters[old_status] -= 1
                self._status_counters[RegistrationStatus.SUSPENDED] += 1
                
                self.logger.info(f"模块挂起成功: {module_id} ({reason})")
                return True
                
            except Exception as e:
                self.logger.error(f"模块挂起失败 {module_id}: {e}")
                return False
    
    def get_module(self, module_id: str) -> Optional[ModuleRegistration]:
        """获取模块信息"""
        with self._lock:
            if module_id in self._modules:
                registration = self._modules[module_id]
                registration.access_count += 1
                registration.last_access_time = time.time()
                self._stats['total_queries'] += 1
                return registration
            return None
    
    def get_class(self, class_name: str) -> Optional[Dict[str, Any]]:
        """获取类信息"""
        with self._lock:
            if class_name in self._classes:
                module_id = self._classes[class_name]
                module = self.get_module(module_id)
                if module:
                    for class_info in module.classes:
                        if class_info.name == class_name:
                            return class_info
            return None
    
    def find_modules(self, 
                    zone: str = None, 
                    status: RegistrationStatus = None,
                    tags: List[str] = None,
                    name_pattern: str = None) -> List[ModuleRegistration]:
        """
        查找模块
        
        Args:
            zone: 区域过滤
            status: 状态过滤
            tags: 标签过滤
            name_pattern: 名称模式匹配
            
        Returns:
            List[ModuleRegistration]: 匹配的模块列表
        """
        with self._lock:
            results = []
            
            for module_id, registration in self._modules.items():
                # 区域过滤
                if zone and registration.zone != zone:
                    continue
                
                # 状态过滤
                if status and registration.status != status:
                    continue
                
                # 标签过滤
                if tags and not all(tag in registration.tags for tag in tags):
                    continue
                
                # 名称模式过滤
                if name_pattern and name_pattern not in registration.module_name:
                    continue
                
                results.append(registration)
            
            self._stats['total_queries'] += 1
            return results
    
    def get_zone_modules(self, zone: str) -> Dict[str, ModuleRegistration]:
        """获取指定区域的所有模块"""
        with self._lock:
            return dict(self._zone_index.get(zone, {}))
    
    def get_discoverable_classes(self, zone: str = None) -> List[Dict[str, Any]]:
        """获取可发现类"""
        with self._lock:
            if zone:
                return self._discoverable_index.get(zone, [])
            else:
                all_classes = []
                for zone_classes in self._discoverable_index.values():
                    all_classes.extend(zone_classes)
                return all_classes
    
    def resolve_dependencies(self, module_id: str) -> List[Tuple[str, bool]]:
        """
        解析模块依赖
        
        Args:
            module_id: 模块ID
            
        Returns:
            List[Tuple[str, bool]]: 依赖解析结果列表 (依赖名称, 是否满足)
        """
        with self._lock:
            if module_id not in self._modules:
                return []
            
            registration = self._modules[module_id]
            results = []
            
            for dep in registration.dependencies:
                # 查找满足依赖的模块
                satisfied = self._find_dependency_provider(dep)
                dep.satisfied = satisfied
                dep.provider = self._find_provider_module_id(dep.name) if satisfied else ""
                results.append((dep.name, satisfied))
            
            self._stats['dependency_resolutions'] += 1
            return results
    
    def get_dependency_tree(self, module_id: str) -> Dict[str, Any]:
        """
        获取依赖树
        
        Args:
            module_id: 模块ID
            
        Returns:
            Dict[str, Any]: 依赖树结构
        """
        with self._lock:
            if module_id not in self._modules:
                return {}
            
            def build_tree(current_id: str, visited: set) -> Dict[str, Any]:
                if current_id in visited:
                    return {'module_id': current_id, 'circular': True}
                
                visited.add(current_id)
                module = self._modules[current_id]
                
                tree = {
                    'module_id': current_id,
                    'module_name': module.module_name,
                    'zone': module.zone,
                    'version': str(module.version),
                    'status': module.status.value,
                    'dependencies': []
                }
                
                for dep_id in self._dependency_graph.get(current_id, set()):
                    dep_tree = build_tree(dep_id, visited.copy())
                    tree['dependencies'].append(dep_tree)
                
                return tree
            
            return build_tree(module_id, set())
    
    def check_circular_dependencies(self) -> List[List[str]]:
        """检查循环依赖"""
        with self._lock:
            visited = set()
            recursion_stack = set()
            circular_dependencies = []
            
            def dfs(module_id: str, path: List[str]):
                if module_id in recursion_stack:
                    # 找到循环依赖
                    cycle_start = path.index(module_id)
                    circular_dependencies.append(path[cycle_start:])
                    return
                
                if module_id in visited:
                    return
                
                visited.add(module_id)
                recursion_stack.add(module_id)
                path.append(module_id)
                
                for dep_id in self._dependency_graph.get(module_id, set()):
                    dfs(dep_id, path.copy())
                
                recursion_stack.remove(module_id)
                path.pop()
            
            for module_id in self._modules:
                if module_id not in visited:
                    dfs(module_id, [])
            
            return circular_dependencies
    
    def add_registration_callback(self, callback: Callable[[ModuleRegistration], None]):
        """添加注册回调"""
        with self._lock:
            self._registration_callbacks.append(callback)
    
    def add_unregistration_callback(self, callback: Callable[[ModuleRegistration, str], None]):
        """添加注销回调"""
        with self._lock:
            self._unregistration_callbacks.append(callback)
    
    def add_activation_callback(self, callback: Callable[[ModuleRegistration], None]):
        """添加激活回调"""
        with self._lock:
            self._activation_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            uptime = time.time() - self._stats['start_time']
            
            return {
                'modules': {
                    'total': len(self._modules),
                    'by_status': {status.value: count for status, count in self._status_counters.items()},
                    'by_zone': {zone: len(modules) for zone, modules in self._zone_index.items()}
                },
                'classes': {
                    'total': len(self._classes),
                    'discoverable': sum(len(classes) for classes in self._discoverable_index.values())
                },
                'dependencies': {
                    'total_edges': sum(len(deps) for deps in self._dependency_graph.values()),
                    'circular_dependencies': len(self.check_circular_dependencies())
                },
                'performance': {
                    'uptime': uptime,
                    'queries_per_second': self._stats['total_queries'] / uptime if uptime > 0 else 0,
                    'registration_rate': self._stats['total_registrations'] / uptime if uptime > 0 else 0
                },
                'operations': self._stats.copy()
            }
    
    def clear(self):
        """清空注册表"""
        with self._lock:
            self._modules.clear()
            self._classes.clear()
            self._zone_index.clear()
            self._discoverable_index.clear()
            self._dependency_graph.clear()
            self._reverse_dependency_graph.clear()
            self._name_index.clear()
            self._tag_index.clear()
            self._version_index.clear()
            self._status_counters.clear()
            self._module_states.clear()
            
            # 重置统计
            self._stats.update({
                'total_registrations': 0,
                'total_unregistrations': 0,
                'total_queries': 0,
                'dependency_resolutions': 0,
                'conflict_resolutions': 0,
                'start_time': time.time()
            })
            
            self.logger.info("注册表已清空")
    
    # 私有方法
    def _generate_module_id(self, zone: str, module_name: str) -> str:
        """生成模块ID"""
        base_id = f"{zone}.{module_name}"
        return hashlib.md5(base_id.encode()).hexdigest()[:16]
    
    def _parse_version(self, version_str: str) -> ModuleVersion:
        """解析版本字符串"""
        try:
            # 处理版本字符串，如 "1.2.3+build123"
            version_parts = version_str.split('+')
            version_base = version_parts[0]
            build = version_parts[1] if len(version_parts) > 1 else ""
            
            # 解析主版本号
            major, minor, patch = version_base.split('.')
            
            return ModuleVersion(
                major=int(major),
                minor=int(minor),
                patch=int(patch),
                build=build
            )
        except Exception:
            # 如果解析失败，返回默认版本
            return ModuleVersion(1, 0, 0)
    
    def _parse_dependencies(self, module_info: Dict[str, Any], registration: ModuleRegistration) -> List[DependencyInfo]:
        """解析依赖关系"""
        dependencies = []
        
        # 从元数据中提取依赖信息
        metadata = getattr(module_info, 'metadata', 'unknown')
        deps_config = metadata.get('dependencies', {}) if isinstance(metadata, dict) else {}
        
        for dep_name, dep_config in deps_config.items():
            if isinstance(dep_config, str):
                # 简单格式: "dependency_name": "version_constraint"
                dep_info = DependencyInfo(
                    name=dep_name,
                    type=DependencyType.REQUIRED,
                    version=dep_config
                )
            elif isinstance(dep_config, dict):
                # 详细格式
                dep_info = DependencyInfo(
                    name=dep_name,
                    type=DependencyType(dep_config.get('type', 'required')),
                    version=dep_config.get('version', '*'),
                    optional=dep_config.get('optional', False),
                    description=dep_config.get('description', '')
                )
            else:
                continue
            
            dependencies.append(dep_info)
        
        return dependencies
    
    def _update_indexes(self, registration: ModuleRegistration):
        """更新所有索引"""
        # 类索引
        for class_info in registration.classes:
            class_name = class_info.name
            if class_name:
                self._classes[class_name] = registration.id
        
        # 区域索引
        self._zone_index[registration.zone][registration.module_name] = registration.id
        
        # 可发现类索引
        for class_info in registration.discoverable_classes:
            self._discoverable_index[registration.zone].append(class_info)
        
        # 名称索引
        if registration.module_name not in self._name_index:
            self._name_index[registration.module_name] = set()
        self._name_index[registration.module_name].add(registration.id)
        
        # 标签索引
        for tag in registration.tags:
            self._tag_index[tag].add(registration.id)
        
        # 版本索引
        if registration.module_name not in self._version_index:
            self._version_index[registration.module_name] = {}
        self._version_index[registration.module_name][str(registration.version)] = registration.id
    
    def _remove_from_indexes(self, registration: ModuleRegistration):
        """从所有索引中移除"""
        # 类索引
        for class_info in registration.classes:
            class_name = class_info.name
            if class_name in self._classes and self._classes[class_name] == registration.id:
                del self._classes[class_name]
        
        # 区域索引
        if registration.zone in self._zone_index:
            if registration.module_name in self._zone_index[registration.zone]:
                del self._zone_index[registration.zone][registration.module_name]
        
        # 可发现类索引
        if registration.zone in self._discoverable_index:
            self._discoverable_index[registration.zone] = [
                cls for cls in self._discoverable_index[registration.zone]
                if cls.get('module') != f"{registration.zone}.{registration.module_name}"
            ]
        
        # 名称索引
        if registration.module_name in self._name_index:
            self._name_index[registration.module_name].discard(registration.id)
            if not self._name_index[registration.module_name]:
                del self._name_index[registration.module_name]
        
        # 标签索引
        for tag in registration.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(registration.id)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
        
        # 版本索引
        if registration.module_name in self._version_index:
            version_str = str(registration.version)
            if version_str in self._version_index[registration.module_name]:
                del self._version_index[registration.module_name][version_str]
            if not self._version_index[registration.module_name]:
                del self._version_index[registration.module_name]
    
    def _update_dependency_graph(self, registration: ModuleRegistration):
        """更新依赖关系图"""
        module_id = registration.id
        
        for dep in registration.dependencies:
            # 查找依赖的模块
            provider_id = self._find_provider_module_id(dep.name)
            if provider_id:
                # 添加正向依赖
                self._dependency_graph[module_id].add(provider_id)
                # 添加反向依赖
                self._reverse_dependency_graph[provider_id].add(module_id)
    
    def _resolve_dependencies(self, registration: ModuleRegistration):
        """解析依赖关系"""
        for dep in registration.dependencies:
            if not dep.satisfied:
                provider_id = self._find_provider_module_id(dep.name)
                if provider_id:
                    dep.satisfied = True
                    dep.provider = provider_id
    
    def _find_provider_module_id(self, dependency_name: str) -> Optional[str]:
        """查找提供指定依赖的模块ID"""
        # 在已注册的模块中查找
        for module_id, module in self._modules.items():
            if (module.status in [RegistrationStatus.REGISTERED, RegistrationStatus.ACTIVATED] and
                module.module_name == dependency_name):
                return module_id
        
        return None
    
    def _find_dependency_provider(self, dependency: DependencyInfo) -> bool:
        """查找依赖提供者"""
        return self._find_provider_module_id(dependency.name) is not None
    
    def _check_dependencies_satisfied(self, registration: ModuleRegistration) -> bool:
        """检查依赖是否全部满足"""
        required_deps = [dep for dep in registration.dependencies 
                        if dep.type == DependencyType.REQUIRED and not dep.optional]
        
        for dep in required_deps:
            if not dep.satisfied:
                return False
        
        return True
    
    def _update_state_history(self, module_id: str, status: RegistrationStatus, reason: str):
        """更新状态历史"""
        if module_id not in self._module_states:
            self._module_states[module_id] = []
        
        self._module_states[module_id].append({
            'status': status,
            'timestamp': time.time(),
            'reason': reason
        })
    
    def _trigger_registration_callbacks(self, registration: ModuleRegistration):
        """触发注册回调"""
        for callback in self._registration_callbacks:
            try:
                callback(registration)
            except Exception as e:
                self.logger.error(f"注册回调执行失败: {e}")
    
    def _trigger_unregistration_callbacks(self, registration: ModuleRegistration, reason: str):
        """触发注销回调"""
        for callback in self._unregistration_callbacks:
            try:
                callback(registration, reason)
            except Exception as e:
                self.logger.error(f"注销回调执行失败: {e}")
    
    def _trigger_activation_callbacks(self, registration: ModuleRegistration):
        """触发激活回调"""
        for callback in self._activation_callbacks:
            try:
                callback(registration)
            except Exception as e:
                self.logger.error(f"激活回调执行失败: {e}")


# 全局注册表实例
global_registry = None

def get_global_registry(config_manager=None) -> ModuleRegistry:
    """获取全局注册表实例"""
    global global_registry
    if global_registry is None:
        global_registry = ModuleRegistry(config_manager)
    return global_registry


class RegistryBuilder:
    """注册表构建器"""
    
    def __init__(self):
        self._config = {
            'auto_resolve_dependencies': True,
            'enable_version_management': True,
            'allow_multiple_versions': False,
            'default_version_constraint': ">=1.0.0"
        }
        self._callbacks = {
            'registration': [],
            'unregistration': [],
            'activation': []
        }
    
    def set_auto_resolve_dependencies(self, enabled: bool) -> 'RegistryBuilder':
        """设置自动解析依赖"""
        self._config['auto_resolve_dependencies'] = enabled
        return self
    
    def set_version_management(self, enabled: bool, allow_multiple: bool = False) -> 'RegistryBuilder':
        """设置版本管理"""
        self._config['enable_version_management'] = enabled
        self._config['allow_multiple_versions'] = allow_multiple
        return self
    
    def add_registration_callback(self, callback: Callable) -> 'RegistryBuilder':
        """添加注册回调"""
        self._callbacks['registration'].append(callback)
        return self
    
    def add_unregistration_callback(self, callback: Callable) -> 'RegistryBuilder':
        """添加注销回调"""
        self._callbacks['unregistration'].append(callback)
        return self
    
    def add_activation_callback(self, callback: Callable) -> 'RegistryBuilder':
        """添加激活回调"""
        self._callbacks['activation'].append(callback)
        return self
    
    def build(self, config_manager=None) -> ModuleRegistry:
        """构建注册表实例"""
        registry = ModuleRegistry(config_manager)
        
        # 应用配置
        for key, value in self._config.items():
            if hasattr(registry, key):
                setattr(registry, key, value)
        
        # 注册回调
        for callback in self._callbacks['registration']:
            registry.add_registration_callback(callback)
        
        for callback in self._callbacks['unregistration']:
            registry.add_unregistration_callback(callback)
        
        for callback in self._callbacks['activation']:
            registry.add_activation_callback(callback)
        
        return registry