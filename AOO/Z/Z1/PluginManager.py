"""
Z1插件管理器 - 主要实现
提供完整的插件加载、管理、执行、配置等功能
"""

import os
import sys
import json
import importlib
import importlib.util
import inspect
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from datetime import datetime
import traceback


class PluginStatus(Enum):
    """插件状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"


class SecurityLevel(Enum):
    """安全级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PluginInfo:
    """插件信息数据类"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    entry_point: str
    security_level: SecurityLevel
    file_path: str
    checksum: str
    created_time: str
    last_modified: str
    usage_count: int = 0
    status: PluginStatus = PluginStatus.UNLOADED
    config: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PluginExecutionResult:
    """插件执行结果数据类"""
    plugin_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = ""


class PluginSecurityValidator:
    """插件安全验证器"""
    
    def __init__(self):
        self.forbidden_imports = {
            'os.system', 'subprocess', 'eval', 'compile',
            'open', 'file', 'input', 'raw_input', '__import__'
        }
        self.allowed_modules = {
            'json', 'math', 'datetime', 'time', 'random', 'hashlib',
            'typing', 'dataclasses', 'enum', 'pathlib', 're'
        }
    
    def validate_plugin(self, plugin_path: str) -> Tuple[bool, List[str]]:
        """
        验证插件安全性
        
        Args:
            plugin_path: 插件文件路径
            
        Returns:
            Tuple[bool, List[str]]: (是否安全, 警告列表)
        """
        warnings = []
        
        try:
            # 检查文件是否存在
            if not os.path.exists(plugin_path):
                warnings.append(f"插件文件不存在: {plugin_path}")
                return False, warnings
            
            # 检查文件扩展名
            if not plugin_path.endswith('.py'):
                warnings.append(f"不支持的插件文件格式: {plugin_path}")
                return False, warnings
            
            # 读取并分析代码
            with open(plugin_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 检查危险函数调用
            for forbidden in self.forbidden_imports:
                if forbidden in code:
                    warnings.append(f"检测到危险函数调用: {forbidden}")
            
            # 检查文件大小
            file_size = os.path.getsize(plugin_path)
            if file_size > 1024 * 1024:  # 1MB
                warnings.append(f"插件文件过大: {file_size} bytes")
            
            # 检查代码行数
            lines = code.split('\n')
            if len(lines) > 10000:
                warnings.append(f"插件代码行数过多: {len(lines)} 行")
            
            # 检查是否有__init__.py中的危险操作
            # 更精确的模式匹配，避免误报
            import re
            
            # 检查危险的import语句（但排除注释）
            lines = code.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    continue
                
                # 检查危险的import（但允许注释中的说明）
                if ('import os' in line and 'os.path' not in line and 'os.system' not in line) or \
                   ('import sys' in line and 'sys.modules' not in line) or \
                   'import subprocess' in line:
                    warnings.append(f"检测到潜在危险的import: {line}")
            
            # 检查危险的函数调用（但排除注释和字符串）
            # 使用正则表达式更精确地匹配，排除安全的调用
            dangerous_calls = [
                r'\beval\s*\(',
                r'\bexec\s*\([^)]*\)',  # 匹配exec()但不包括exec_module
                r'\bcompile\s*\(',
                r'\bos\.system\s*\(',
                r'\bsubprocess\.',
                r'\b__import__\s*\('
            ]
            
            for pattern in dangerous_calls:
                matches = re.finditer(pattern, code)
                for match in matches:
                    start_pos = match.start()
                    # 检查前面的内容，排除安全的调用
                    before_code = code[max(0, start_pos-30):start_pos]
                    
                    # 排除安全的调用
                    safe_patterns = [
                        'importlib.exec_module',
                        'type('  # 排除type()调用
                    ]
                    
                    is_safe = any(safe_pattern in before_code for safe_pattern in safe_patterns)
                    
                    if not is_safe:
                        warnings.append(f"检测到潜在危险函数调用: {match.group()}")
                        break  # 每个pattern只警告一次
            
            return len(warnings) == 0, warnings
            
        except Exception as e:
            warnings.append(f"安全验证失败: {str(e)}")
            return False, warnings
    
    def calculate_checksum(self, plugin_path: str) -> str:
        """计算文件校验和"""
        try:
            with open(plugin_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""


class PluginDependencyResolver:
    """插件依赖解析器"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, List[str]] = {}
    
    def add_plugin(self, plugin_info: PluginInfo):
        """添加插件到依赖图"""
        self.dependency_graph[plugin_info.name] = plugin_info.dependencies.copy()
    
    def resolve_dependencies(self, plugin_name: str) -> Tuple[bool, List[str]]:
        """
        解析插件依赖
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Tuple[bool, List[str]]: (是否成功, 依赖列表)
        """
        if plugin_name not in self.dependency_graph:
            return True, []
        
        visited = set()
        recursion_stack = set()
        dependencies = []
        
        def dfs(name: str) -> bool:
            if name in recursion_stack:
                return False  # 循环依赖
            if name in visited:
                return True
            
            visited.add(name)
            recursion_stack.add(name)
            
            if name in self.dependency_graph:
                for dep in self.dependency_graph[name]:
                    if not dfs(dep):
                        return False
                    if dep not in dependencies:
                        dependencies.append(dep)
            
            recursion_stack.remove(name)
            return True
        
        if not dfs(plugin_name):
            return False, []
        
        return True, dependencies


class PluginStatistics:
    """插件统计器"""
    
    def __init__(self):
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def record_execution(self, plugin_name: str, execution_time: float, success: bool):
        """记录插件执行统计"""
        with self.lock:
            if plugin_name not in self.stats:
                self.stats[plugin_name] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'total_execution_time': 0.0,
                    'average_execution_time': 0.0,
                    'last_execution': None,
                    'first_execution': None
                }
            
            stats = self.stats[plugin_name]
            stats['total_executions'] += 1
            stats['total_execution_time'] += execution_time
            
            if success:
                stats['successful_executions'] += 1
            else:
                stats['failed_executions'] += 1
            
            stats['average_execution_time'] = (
                stats['total_execution_time'] / stats['total_executions']
            )
            
            now = datetime.now().isoformat()
            stats['last_execution'] = now
            if stats['first_execution'] is None:
                stats['first_execution'] = now
    
    def get_statistics(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """获取插件统计信息"""
        with self.lock:
            return self.stats.get(plugin_name, {}).copy()
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有插件统计信息"""
        with self.lock:
            return {k: v.copy() for k, v in self.stats.items()}


class PluginManager:
    """Z1插件管理器主类"""
    
    def __init__(self, plugin_dir: str = "plugins", config_file: str = "plugin_config.json"):
        """
        初始化插件管理器
        
        Args:
            plugin_dir: 插件目录路径
            config_file: 配置文件路径
        """
        self.plugin_dir = Path(plugin_dir)
        self.config_file = Path(config_file)
        
        # 创建目录
        self.plugin_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.security_validator = PluginSecurityValidator()
        self.dependency_resolver = PluginDependencyResolver()
        self.statistics = PluginStatistics()
        
        # 插件存储
        self.plugins: Dict[str, PluginInfo] = {}
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_instances: Dict[str, Any] = {}
        
        # 配置
        self.config = self._load_config()
        
        # 日志
        self.logger = self._setup_logging()
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 加载已安装的插件
        self._discover_plugins()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('Z1PluginManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "auto_load": True,
            "security_level": "medium",
            "max_execution_time": 30.0,
            "enable_statistics": True,
            "enable_dependency_check": True,
            "allowed_imports": [],
            "forbidden_imports": [],
            "plugin_timeout": 60.0
        }
        
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并配置
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                # 创建默认配置文件
                self._save_config(default_config)
                return default_config
        except Exception as e:
            self.logger.warning(f"加载配置文件失败，使用默认配置: {e}")
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
    
    def _discover_plugins(self):
        """发现插件"""
        try:
            for plugin_file in self.plugin_dir.glob("*.py"):
                if plugin_file.name.startswith('_'):
                    continue
                
                try:
                    plugin_info = self._analyze_plugin(plugin_file)
                    if plugin_info:
                        self.plugins[plugin_info.name] = plugin_info
                        self.dependency_resolver.add_plugin(plugin_info)
                        self.logger.info(f"发现插件: {plugin_info.name}")
                except Exception as e:
                    self.logger.error(f"分析插件失败 {plugin_file}: {e}")
        except Exception as e:
            self.logger.error(f"发现插件失败: {e}")
    
    def _analyze_plugin(self, plugin_file: Path) -> Optional[PluginInfo]:
        """分析插件文件"""
        try:
            # 安全验证
            is_safe, warnings = self.security_validator.validate_plugin(str(plugin_file))
            if not is_safe:
                self.logger.warning(f"插件安全验证失败: {plugin_file}, 警告: {warnings}")
                return None
            
            # 计算校验和
            checksum = self.security_validator.calculate_checksum(str(plugin_file))
            
            # 尝试导入插件模块
            spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # 临时添加到sys.modules
            module_name = f"z1_plugins.{plugin_file.stem}"
            sys.modules[module_name] = module
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                self.logger.error(f"导入插件模块失败: {e}")
                return None
            
            # 提取插件信息
            plugin_info = self._extract_plugin_info(module, str(plugin_file), checksum)
            
            return plugin_info
            
        except Exception as e:
            self.logger.error(f"分析插件失败 {plugin_file}: {e}")
            return None
    
    def _extract_plugin_info(self, module: Any, file_path: str, checksum: str) -> Optional[PluginInfo]:
        """从模块中提取插件信息"""
        try:
            # 查找插件信息
            plugin_info = getattr(module, 'PLUGIN_INFO', None)
            
            if plugin_info is None:
                # 尝试从docstring中提取
                docstring = inspect.getdoc(module)
                if docstring and 'PLUGIN_INFO' in docstring:
                    self.logger.warning("在docstring中找到PLUGIN_INFO，建议将其定义为模块级变量")
                    return None
            
            if plugin_info is None:
                return None
            
            # 验证插件信息格式
            required_fields = ['name', 'version', 'description']
            for field in required_fields:
                if not hasattr(plugin_info, field):
                    self.logger.error(f"插件信息缺少必需字段: {field}")
                    return None
            
            # 获取依赖
            dependencies = getattr(plugin_info, 'dependencies', [])
            
            # 获取安全级别
            security_level_str = getattr(plugin_info, 'security_level', 'medium')
            try:
                security_level = SecurityLevel(security_level_str)
            except ValueError:
                security_level = SecurityLevel.MEDIUM
            
            # 获取入口点
            entry_point = getattr(plugin_info, 'entry_point', 'main')
            
            # 获取作者
            author = getattr(plugin_info, 'author', 'Unknown')
            
            # 获取配置
            config = getattr(plugin_info, 'config', {})
            
            # 获取元数据
            metadata = getattr(plugin_info, 'metadata', {})
            
            # 文件时间
            stat = os.stat(file_path)
            created_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
            last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            return PluginInfo(
                name=plugin_info.name,
                version=plugin_info.version,
                description=plugin_info.description,
                author=author,
                dependencies=dependencies,
                entry_point=entry_point,
                security_level=security_level,
                file_path=file_path,
                checksum=checksum,
                created_time=created_time,
                last_modified=last_modified,
                config=config,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"提取插件信息失败: {e}")
            return None
    
    def load_plugin(self, plugin_name: str, force_reload: bool = False) -> bool:
        """
        加载插件
        
        Args:
            plugin_name: 插件名称
            force_reload: 是否强制重新加载
            
        Returns:
            bool: 是否成功
        """
        with self.lock:
            try:
                if plugin_name not in self.plugins:
                    self.logger.error(f"插件不存在: {plugin_name}")
                    return False
                
                plugin_info = self.plugins[plugin_name]
                
                # 检查是否已加载
                if plugin_name in self.loaded_plugins and not force_reload:
                    self.logger.info(f"插件已加载: {plugin_name}")
                    return True
                
                # 更新状态
                plugin_info.status = PluginStatus.LOADING
                
                # 检查依赖
                if self.config.get('enable_dependency_check', True):
                    deps_success, dependencies = self.dependency_resolver.resolve_dependencies(plugin_name)
                    if not deps_success:
                        plugin_info.status = PluginStatus.ERROR
                        self.logger.error(f"依赖解析失败: {plugin_name}")
                        return False
                    
                    # 加载依赖
                    for dep in dependencies:
                        if dep not in self.loaded_plugins:
                            if not self.load_plugin(dep):
                                plugin_info.status = PluginStatus.ERROR
                                self.logger.error(f"依赖插件加载失败: {dep}")
                                return False
                
                # 导入插件模块
                module = self._import_plugin_module(plugin_info)
                if module is None:
                    plugin_info.status = PluginStatus.ERROR
                    return False
                
                # 验证插件接口
                if not self._validate_plugin_interface(module, plugin_info):
                    plugin_info.status = PluginStatus.ERROR
                    return False
                
                # 创建插件实例
                instance = self._create_plugin_instance(module, plugin_info)
                if instance is None:
                    plugin_info.status = PluginStatus.ERROR
                    return False
                
                # 保存插件
                self.loaded_plugins[plugin_name] = module
                self.plugin_instances[plugin_name] = instance
                
                # 更新状态
                plugin_info.status = PluginStatus.LOADED
                
                self.logger.info(f"插件加载成功: {plugin_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"加载插件失败 {plugin_name}: {e}")
                if plugin_name in self.plugins:
                    self.plugins[plugin_name].status = PluginStatus.ERROR
                return False
    
    def _import_plugin_module(self, plugin_info: PluginInfo) -> Optional[Any]:
        """导入插件模块"""
        try:
            spec = importlib.util.spec_from_file_location(
                f"z1_plugins.{plugin_info.name}", 
                plugin_info.file_path
            )
            
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # 检查校验和
            current_checksum = self.security_validator.calculate_checksum(plugin_info.file_path)
            if current_checksum != plugin_info.checksum:
                self.logger.warning(f"插件文件校验和不匹配: {plugin_info.name}")
            
            # 执行模块
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            self.logger.error(f"导入插件模块失败: {e}")
            return None
    
    def _validate_plugin_interface(self, module: Any, plugin_info: PluginInfo) -> bool:
        """验证插件接口"""
        try:
            # 检查入口函数
            if not hasattr(module, plugin_info.entry_point):
                self.logger.error(f"插件缺少入口函数: {plugin_info.entry_point}")
                return False
            
            # 检查函数签名
            func = getattr(module, plugin_info.entry_point)
            if not callable(func):
                self.logger.error(f"入口点不是可调用对象: {plugin_info.entry_point}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证插件接口失败: {e}")
            return False
    
    def _create_plugin_instance(self, module: Any, plugin_info: PluginInfo) -> Optional[Any]:
        """创建插件实例"""
        try:
            # 如果有初始化函数，调用它
            if hasattr(module, 'initialize'):
                init_func = getattr(module, 'initialize')
                if callable(init_func):
                    instance = init_func(plugin_info.config)
                    return instance
            
            # 否则返回模块本身
            return module
            
        except Exception as e:
            self.logger.error(f"创建插件实例失败: {e}")
            return None
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> PluginExecutionResult:
        """
        执行插件
        
        Args:
            plugin_name: 插件名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            PluginExecutionResult: 执行结果
        """
        start_time = time.time()
        
        try:
            # 检查插件是否加载
            if plugin_name not in self.loaded_plugins:
                if not self.load_plugin(plugin_name):
                    return PluginExecutionResult(
                        plugin_name=plugin_name,
                        success=False,
                        error="插件加载失败",
                        execution_time=time.time() - start_time,
                        timestamp=datetime.now().isoformat()
                    )
            
            plugin_info = self.plugins[plugin_name]
            
            # 检查插件状态
            if plugin_info.status not in [PluginStatus.LOADED, PluginStatus.ACTIVE]:
                return PluginExecutionResult(
                    plugin_name=plugin_name,
                    success=False,
                    error=f"插件状态不正确: {plugin_info.status}",
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
            
            # 获取插件实例
            if plugin_name not in self.plugin_instances:
                return PluginExecutionResult(
                    plugin_name=plugin_name,
                    success=False,
                    error="插件实例不存在",
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
            
            instance = self.plugin_instances[plugin_name]
            module = self.loaded_plugins[plugin_name]
            
            # 执行插件
            func = getattr(module, plugin_info.entry_point)
            
            # 设置执行超时
            execution_timeout = self.config.get('plugin_timeout', 60.0)
            
            # 在线程中执行以支持超时
            result_container = []
            exception_container = []
            
            def target():
                try:
                    result = func(instance, *args, **kwargs)
                    result_container.append(result)
                except Exception as e:
                    exception_container.append(e)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=execution_timeout)
            
            if thread.is_alive():
                # 超时
                return PluginExecutionResult(
                    plugin_name=plugin_name,
                    success=False,
                    error=f"插件执行超时 ({execution_timeout}s)",
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
            
            # 检查异常
            if exception_container:
                exception = exception_container[0]
                return PluginExecutionResult(
                    plugin_name=plugin_name,
                    success=False,
                    error=str(exception),
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
            
            # 成功执行
            result = result_container[0] if result_container else None
            execution_time = time.time() - start_time
            
            # 更新统计
            if self.config.get('enable_statistics', True):
                self.statistics.record_execution(plugin_name, execution_time, True)
            
            # 更新插件信息
            plugin_info.usage_count += 1
            plugin_info.status = PluginStatus.ACTIVE
            
            return PluginExecutionResult(
                plugin_name=plugin_name,
                success=True,
                result=result,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # 更新统计
            if self.config.get('enable_statistics', True):
                self.statistics.record_execution(plugin_name, execution_time, False)
            
            # 更新插件状态
            if plugin_name in self.plugins:
                self.plugins[plugin_name].status = PluginStatus.ERROR
            
            return PluginExecutionResult(
                plugin_name=plugin_name,
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        卸载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 是否成功
        """
        with self.lock:
            try:
                if plugin_name not in self.plugins:
                    return False
                
                # 检查是否有其他插件依赖此插件
                dependent_plugins = []
                for name, info in self.plugins.items():
                    if plugin_name in info.dependencies and name in self.loaded_plugins:
                        dependent_plugins.append(name)
                
                if dependent_plugins:
                    self.logger.warning(f"插件 {plugin_name} 被其他插件依赖: {dependent_plugins}")
                    # 可以选择强制卸载或拒绝卸载
                    # 这里选择拒绝卸载
                    return False
                
                # 调用插件清理函数
                if plugin_name in self.plugin_instances:
                    instance = self.plugin_instances[plugin_name]
                    if hasattr(instance, 'cleanup'):
                        try:
                            instance.cleanup()
                        except Exception as e:
                            self.logger.warning(f"插件清理失败 {plugin_name}: {e}")
                
                # 移除插件
                if plugin_name in self.loaded_plugins:
                    del self.loaded_plugins[plugin_name]
                
                if plugin_name in self.plugin_instances:
                    del self.plugin_instances[plugin_name]
                
                # 更新状态
                self.plugins[plugin_name].status = PluginStatus.UNLOADED
                
                self.logger.info(f"插件卸载成功: {plugin_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"卸载插件失败 {plugin_name}: {e}")
                return False
    
    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[PluginInfo]:
        """
        列出插件
        
        Args:
            status_filter: 状态过滤条件
            
        Returns:
            List[PluginInfo]: 插件信息列表
        """
        plugins = list(self.plugins.values())
        
        if status_filter:
            plugins = [p for p in plugins if p.status == status_filter]
        
        return sorted(plugins, key=lambda x: x.name)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """获取插件信息"""
        return self.plugins.get(plugin_name)
    
    def update_plugin(self, plugin_name: str, new_file_path: str) -> bool:
        """
        更新插件
        
        Args:
            plugin_name: 插件名称
            new_file_path: 新插件文件路径
            
        Returns:
            bool: 是否成功
        """
        with self.lock:
            try:
                if plugin_name not in self.plugins:
                    return False
                
                plugin_info = self.plugins[plugin_name]
                
                # 检查插件是否正在使用
                if plugin_name in self.loaded_plugins:
                    self.logger.warning(f"插件正在使用中，尝试卸载: {plugin_name}")
                    if not self.unload_plugin(plugin_name):
                        self.logger.error(f"无法卸载正在使用的插件: {plugin_name}")
                        return False
                
                # 备份原文件
                backup_path = f"{plugin_info.file_path}.backup"
                try:
                    import shutil
                    shutil.copy2(plugin_info.file_path, backup_path)
                except Exception as e:
                    self.logger.warning(f"备份原插件文件失败: {e}")
                
                # 复制新文件
                try:
                    import shutil
                    shutil.copy2(new_file_path, plugin_info.file_path)
                except Exception as e:
                    self.logger.error(f"复制新插件文件失败: {e}")
                    return False
                
                # 重新分析插件
                plugin_info.status = PluginStatus.UPDATING
                
                new_plugin_info = self._analyze_plugin(Path(plugin_info.file_path))
                if new_plugin_info is None:
                    # 恢复备份
                    try:
                        shutil.copy2(backup_path, plugin_info.file_path)
                    except Exception:
                        pass
                    plugin_info.status = PluginStatus.ERROR
                    return False
                
                # 更新插件信息
                self.plugins[plugin_name] = new_plugin_info
                self.dependency_resolver.add_plugin(new_plugin_info)
                
                # 重新加载插件
                success = self.load_plugin(plugin_name, force_reload=True)
                
                if success:
                    self.logger.info(f"插件更新成功: {plugin_name}")
                    # 删除备份文件
                    try:
                        os.remove(backup_path)
                    except Exception:
                        pass
                else:
                    # 恢复备份
                    try:
                        shutil.copy2(backup_path, plugin_info.file_path)
                        self.load_plugin(plugin_name, force_reload=True)
                    except Exception:
                        pass
                
                return success
                
            except Exception as e:
                self.logger.error(f"更新插件失败 {plugin_name}: {e}")
                return False
    
    def get_plugin_statistics(self, plugin_name: str = None) -> Dict[str, Any]:
        """
        获取插件统计信息
        
        Args:
            plugin_name: 插件名称，如果为None则获取所有插件统计
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if plugin_name:
            return self.statistics.get_statistics(plugin_name)
        else:
            return self.statistics.get_all_statistics()
    
    def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """
        配置插件
        
        Args:
            plugin_name: 插件名称
            config: 配置参数
            
        Returns:
            bool: 是否成功
        """
        try:
            if plugin_name not in self.plugins:
                return False
            
            # 合并配置
            plugin_info = self.plugins[plugin_name]
            plugin_info.config.update(config)
            
            # 如果插件已加载，更新实例配置
            if plugin_name in self.plugin_instances:
                instance = self.plugin_instances[plugin_name]
                if hasattr(instance, 'update_config'):
                    try:
                        instance.update_config(config)
                    except Exception as e:
                        self.logger.warning(f"更新插件配置失败 {plugin_name}: {e}")
            
            self.logger.info(f"插件配置更新成功: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"配置插件失败 {plugin_name}: {e}")
            return False
    
    def validate_all_plugins(self) -> Dict[str, List[str]]:
        """验证所有插件的安全性"""
        validation_results = {}
        
        for plugin_name, plugin_info in self.plugins.items():
            is_safe, warnings = self.security_validator.validate_plugin(plugin_info.file_path)
            validation_results[plugin_name] = warnings
        
        return validation_results
    
    def save_plugin_registry(self, file_path: str = None):
        """保存插件注册表"""
        if file_path is None:
            file_path = "plugin_registry.json"
        
        try:
            registry = {
                'plugins': {name: asdict(info) for name, info in self.plugins.items()},
                'statistics': self.statistics.get_all_statistics(),
                'config': self.config,
                'saved_time': datetime.now().isoformat()
            }
            
            # 转换枚举值为字符串
            for plugin_data in registry['plugins'].values():
                plugin_data['status'] = plugin_data['status'].value
                plugin_data['security_level'] = plugin_data['security_level'].value
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"插件注册表已保存: {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存插件注册表失败: {e}")
    
    def load_plugin_registry(self, file_path: str = None):
        """加载插件注册表"""
        if file_path is None:
            file_path = "plugin_registry.json"
        
        try:
            if not os.path.exists(file_path):
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            # 恢复插件信息
            for name, plugin_data in registry['plugins'].items():
                try:
                    plugin_data['status'] = PluginStatus(plugin_data['status'])
                    plugin_data['security_level'] = SecurityLevel(plugin_data['security_level'])
                    plugin_info = PluginInfo(**plugin_data)
                    self.plugins[name] = plugin_info
                except Exception as e:
                    self.logger.warning(f"恢复插件信息失败 {name}: {e}")
            
            self.logger.info(f"插件注册表已加载: {file_path}")
            
        except Exception as e:
            self.logger.error(f"加载插件注册表失败: {e}")
    
    def get_plugin_dependencies(self, plugin_name: str) -> List[str]:
        """获取插件依赖列表"""
        if plugin_name not in self.plugins:
            return []
        
        plugin_info = self.plugins[plugin_name]
        return plugin_info.dependencies.copy()
    
    def get_dependent_plugins(self, plugin_name: str) -> List[str]:
        """获取依赖指定插件的其他插件"""
        dependents = []
        for name, info in self.plugins.items():
            if plugin_name in info.dependencies:
                dependents.append(name)
        return dependents
    
    def shutdown(self):
        """关闭插件管理器"""
        self.logger.info("正在关闭插件管理器...")
        
        # 卸载所有插件
        loaded_plugins = list(self.loaded_plugins.keys())
        for plugin_name in loaded_plugins:
            self.unload_plugin(plugin_name)
        
        # 保存注册表
        self.save_plugin_registry()
        
        self.logger.info("插件管理器已关闭")


# 示例插件模板
PLUGIN_TEMPLATE = '''"""
{name} 插件
{description}

作者: {author}
版本: {version}
"""

# 插件信息
PLUGIN_INFO = type('PluginInfo', (), {{
    'name': '{name}',
    'version': '{version}',
    'description': '{description}',
    'author': '{author}',
    'dependencies': {dependencies},
    'entry_point': '{entry_point}',
    'security_level': '{security_level}',
    'config': {config},
    'metadata': {metadata}
}})()

def {entry_point}(plugin_instance, *args, **kwargs):
    """
    插件主函数
    
    Args:
        plugin_instance: 插件实例
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        Any: 执行结果
    """
    # 在这里实现插件的主要功能
    result = {{
        'message': 'Hello from {name} plugin!',
        'args': args,
        'kwargs': kwargs,
        'config': plugin_instance.config if hasattr(plugin_instance, 'config') else {{}}
    }}
    return result

def initialize(config=None):
    """
    插件初始化函数
    
    Args:
        config: 配置参数
        
    Returns:
        Any: 插件实例
    """
    class {class_name}:
        def __init__(self, config=None):
            self.config = config or {{}}
        
        def cleanup(self):
            """清理资源"""
            pass
        
        def update_config(self, new_config):
            """更新配置"""
            self.config.update(new_config)
    
    return {class_name}(config)

# 可选：添加插件清理函数
def cleanup(plugin_instance):
    """清理插件资源"""
    if hasattr(plugin_instance, 'cleanup'):
        plugin_instance.cleanup()
'''

def create_plugin_template(name: str, version: str = "1.0.0", 
                          description: str = "", author: str = "Unknown",
                          dependencies: List[str] = None, 
                          entry_point: str = "main",
                          security_level: str = "medium",
                          config: Dict[str, Any] = None,
                          metadata: Dict[str, Any] = None,
                          output_path: str = None) -> str:
    """
    创建插件模板
    
    Args:
        name: 插件名称
        version: 版本号
        description: 描述
        author: 作者
        dependencies: 依赖列表
        entry_point: 入口点
        security_level: 安全级别
        config: 默认配置
        metadata: 元数据
        output_path: 输出路径
        
    Returns:
        str: 生成的插件文件路径
    """
    if dependencies is None:
        dependencies = []
    if config is None:
        config = {}
    if metadata is None:
        metadata = {}
    
    if output_path is None:
        output_path = f"{name}.py"
    
    class_name = f"{name.replace('-', '_').title()}Plugin"
    
    plugin_code = PLUGIN_TEMPLATE.format(
        name=name,
        version=version,
        description=description or f"{name} 插件",
        author=author,
        dependencies=dependencies,
        entry_point=entry_point,
        security_level=security_level,
        config=json.dumps(config, ensure_ascii=False),
        metadata=json.dumps(metadata, ensure_ascii=False),
        class_name=class_name
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(plugin_code)
    
    return output_path


if __name__ == "__main__":
    # 示例用法
    print("Z1插件管理器示例")
    
    # 创建插件管理器
    manager = PluginManager()
    
    # 列出插件
    plugins = manager.list_plugins()
    print(f"发现 {len(plugins)} 个插件:")
    for plugin in plugins:
        print(f"  - {plugin.name} v{plugin.version} ({plugin.status.value})")
    
    # 如果有插件，尝试执行第一个
    if plugins:
        plugin_name = plugins[0].name
        print(f"\\n尝试执行插件: {plugin_name}")
        result = manager.execute_plugin(plugin_name, "test", key="value")
        print(f"执行结果: {result.success}")
        if result.success:
            print(f"返回结果: {result.result}")
        else:
            print(f"错误: {result.error}")