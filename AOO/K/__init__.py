"""
AOO自动发现工厂系统 - K区主入口
集成所有核心组件，提供统一的工厂系统接口
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import asdict 
import time 
# 导入K区各个模块
from .base_module import (
    BaseModule, DiscoverableModule, FactoryModule, ServiceModule, TradingModule,
    ModuleState, ServiceType, discoverable, trading_module
)
from .module_scanner import ModuleScanner, ScannerBuilder
from .module_registry import ModuleRegistry, RegistryBuilder, get_global_registry
from .dependency_resolver import DependencyResolver, ResolverBuilder
from .auto_wiring_factory import AutoWiringFactory, FactoryBuilder, InstanceScope, InstanceStatus
from .lifecycle_manager import LifecycleManager, LifecycleBuilder, LifecycleState, HealthStatus
from .config_manager import ConfigManager, get_global_config_manager

class AOOFactorySystem:
    """
    AOO工厂系统
    集成所有核心组件，提供完整的自动发现和依赖注入功能
    """
    
    def __init__(self, aoo_root: str, config_path: str = None, environment: str = None):
        self.aoo_root = Path(aoo_root)
        self.config_path = Path(config_path) if config_path else self.aoo_root / 'config' / 'trading_config.json'
        self.environment = environment or 'development'
        
        # 核心组件
        self.config_manager: Optional[ConfigManager] = None
        self.module_scanner: Optional[ModuleScanner] = None
        self.module_registry: Optional[ModuleRegistry] = None
        self.dependency_resolver: Optional[DependencyResolver] = None
        self.auto_wiring_factory: Optional[AutoWiringFactory] = None
        self.lifecycle_manager: Optional[LifecycleManager] = None
        
        # 状态
        self._initialized = False
        self._started = False
        
        # 日志
        self.logger = logging.getLogger('AOO.FactorySystem')
        
        # 初始化日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def initialize(self) -> bool:
        """
        初始化工厂系统
        
        Returns:
            bool: 初始化是否成功
        """
        if self._initialized:
            self.logger.warning("工厂系统已经初始化")
            return True
        
        try:
            self.logger.info("🏭 初始化AOO自动发现系统工厂...")
            
            # 1. 初始化配置管理器
            from .config_manager import ConfigManager
            self.config_manager = ConfigManager(
                config_path=str(self.config_path),
                environment=self.environment
            )
            
            # 2. 初始化模块扫描器
            self.logger.info("🔧 初始化模块扫描器...")
            self.module_scanner = ScannerBuilder()\
                .build(str(self.aoo_root), self.config_manager)
            
            # 3. 初始化模块注册表
            self.logger.info("🔧 初始化模块注册表...")
            self.module_registry = RegistryBuilder()\
                .build(self.config_manager)
            
            # 4. 初始化依赖解析器
            self.logger.info("🔧 初始化依赖解析器...")
            self.dependency_resolver = ResolverBuilder()\
                .build(self.module_registry, self.config_manager)
            
            # 5. 初始化自动装配工厂
            self.logger.info("🔧 初始化自动装配工厂...")
            self.auto_wiring_factory = FactoryBuilder()\
                .build(self.module_registry, self.dependency_resolver, self.config_manager)
            
            # 6. 初始化生命周期管理器
            self.logger.info("🔧 初始化生命周期管理器...")
            self.lifecycle_manager = LifecycleBuilder()\
                .build(self.module_registry, self.auto_wiring_factory, self.config_manager)
            
            # 7. 执行模块扫描和注册
            self.logger.info("🔍 开始深度扫描AOO框架...")
            scan_results = self.module_scanner.deep_scan()
            
            # 8. 注册发现的模块
            self.logger.info("📦 注册发现的模块...")
            for zone, modules in scan_results.items():
                for module_info in modules:
                    try:
                        self.module_registry.register_module(
                            module_info.file_path,
                            zone,
                            {
                                'module_name': module_info.name,
                                'classes': [asdict(cls) for cls in module_info.classes],
                                'discoverable_classes': [
                                    asdict(cls) for cls in module_info.classes 
                                    if cls.is_discoverable
                                ],
                                'analysis_success': module_info.analysis_success,
                                'error': module_info.error,
                                'metadata': {
                                    'file_size': module_info.file_size,
                                    'analysis_time': module_info.analysis_time
                                }
                            }
                        )
                        
                        # 注册到生命周期管理器
                        for class_info in module_info.classes:
                            if class_info.is_discoverable:
                                module_id = f"{zone}.{module_info.name}.{class_info.name}"
                                self.lifecycle_manager.register_module(
                                    module_id,
                                    class_info.name,
                                    zone,
                                    []  # 依赖会在后续解析
                                )
                    
                    except Exception as e:
                        self.logger.error(f"模块注册失败 {zone}.{module_info.name}: {e}")
            
            self._initialized = True
            self.logger.info("✅ AOO工厂系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ AOO工厂系统初始化失败: {e}")
            return False
    
    def start(self) -> bool:
        """
        启动工厂系统
        
        Returns:
            bool: 启动是否成功
        """
        if not self._initialized:
            self.logger.error("工厂系统未初始化")
            return False
        
        if self._started:
            self.logger.warning("工厂系统已经启动")
            return True
        
        try:
            self.logger.info("🚀 启动AOO工厂系统...")
            
            # 1. 初始化所有模块
            self.logger.info("🔧 初始化所有模块...")
            if not self.lifecycle_manager.initialize_all_modules():
                self.logger.error("模块初始化失败")
                return False
            
            # 2. 启动所有模块
            self.logger.info("🔧 启动所有模块...")
            if not self.lifecycle_manager.start_all_modules():
                self.logger.error("模块启动失败")
                return False
            
            self._started = True
            self.logger.info("✅ AOO工厂系统启动完成")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ AOO工厂系统启动失败: {e}")
            return False
    
    def shutdown(self):
        """关闭工厂系统"""
        self.logger.info("🛑 关闭AOO工厂系统...")
        
        if self.lifecycle_manager:
            self.lifecycle_manager.shutdown()
        
        if self.auto_wiring_factory:
            self.auto_wiring_factory.shutdown()
        
        self._started = False
        self._initialized = False
        self.logger.info("✅ AOO工厂系统关闭完成")
    
    def get_instance(self, class_name: str, **kwargs) -> Any:
        """
        获取类的实例
        
        Args:
            class_name: 类名
            **kwargs: 创建参数
            
        Returns:
            Any: 类的实例
        """
        if not self._initialized:
            raise RuntimeError("工厂系统未初始化")
        
        return self.auto_wiring_factory.create_instance(class_name, **kwargs)
    
    def get_singleton(self, class_name: str) -> Any:
        """
        获取单例实例
        
        Args:
            class_name: 类名
            
        Returns:
            Any: 单例实例
        """
        if not self._initialized:
            raise RuntimeError("工厂系统未初始化")
        
        return self.auto_wiring_factory.get_singleton(class_name)
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        获取系统健康报告
        
        Returns:
            Dict[str, Any]: 健康报告
        """
        if not self._initialized:
            return {
                'system': 'not_initialized',
                'overall_status': 'unknown',
                'timestamp': time.time()
            }
        
        health_report = self.lifecycle_manager.get_health_report()
        return asdict(health_report)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self._initialized:
            return {'system': 'not_initialized'}
        
        stats = {
            'system': {
                'initialized': self._initialized,
                'started': self._started,
                'environment': self.environment
            },
            'components': {
                'config_manager': self.config_manager.get_statistics() if self.config_manager else None,
                'module_scanner': self.module_scanner.get_statistics() if self.module_scanner else None,
                'module_registry': self.module_registry.get_statistics() if self.module_registry else None,
                'dependency_resolver': self.dependency_resolver.get_statistics() if self.dependency_resolver else None,
                'auto_wiring_factory': self.auto_wiring_factory.get_statistics() if self.auto_wiring_factory else None,
                'lifecycle_manager': self.lifecycle_manager.get_statistics() if self.lifecycle_manager else None
            }
        }
        
        return stats
    
    @property
    def is_initialized(self) -> bool:
        """获取初始化状态"""
        return self._initialized
    
    @property
    def is_started(self) -> bool:
        """获取启动状态"""
        return self._started


# 全局工厂系统实例
global_factory_system = None

def get_global_factory_system(aoo_root: str = None, config_path: str = None, environment: str = None) -> AOOFactorySystem:
    """获取全局工厂系统实例"""
    global global_factory_system
    if global_factory_system is None:
        if aoo_root is None:
            # 尝试自动检测AOO根目录
            aoo_root = Path(__file__).parent.parent
        global_factory_system = AOOFactorySystem(aoo_root, config_path, environment)
    return global_factory_system

def initialize_global_factory(aoo_root: str = None, config_path: str = None, environment: str = None) -> bool:
    """初始化全局工厂系统"""
    factory = get_global_factory_system(aoo_root, config_path, environment)
    return factory.initialize()

def start_global_factory() -> bool:
    """启动全局工厂系统"""
    factory = get_global_factory_system()
    return factory.start()

def shutdown_global_factory():
    """关闭全局工厂系统"""
    factory = get_global_factory_system()
    factory.shutdown()

def get_instance(class_name: str, **kwargs) -> Any:
    """获取类的实例（快捷方式）"""
    factory = get_global_factory_system()
    return factory.get_instance(class_name, **kwargs)

def get_singleton(class_name: str) -> Any:
    """获取单例实例（快捷方式）"""
    factory = get_global_factory_system()
    return factory.get_singleton(class_name)

# 导出主要类和函数
__all__ = [
    # 基础模块
    'BaseModule', 'DiscoverableModule', 'FactoryModule', 'ServiceModule', 'TradingModule',
    'ModuleState', 'ServiceType', 'discoverable', 'trading_module',
    
    # 模块扫描
    'ModuleScanner', 'ScannerBuilder',
    
    # 模块注册
    'ModuleRegistry', 'RegistryBuilder', 'get_global_registry',
    
    # 依赖解析
    'DependencyResolver', 'ResolverBuilder',
    
    # 自动装配
    'AutoWiringFactory', 'FactoryBuilder', 'InstanceScope', 'InstanceStatus',
    
    # 生命周期管理
    'LifecycleManager', 'LifecycleBuilder', 'LifecycleState', 'HealthStatus',
    
    # 配置管理
    'ConfigManager', 'ConfigBuilder', 'get_global_config_manager',
    
    # 工厂系统
    'AOOFactorySystem', 'get_global_factory_system',
    
    # 快捷函数
    'initialize_global_factory', 'start_global_factory', 'shutdown_global_factory',
    'get_instance', 'get_singleton'
]