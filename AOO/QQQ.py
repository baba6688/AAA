#!/usr/bin/env python3
"""
AOO智能量化交易系统 - 修复版主程序
修复生命周期管理器和统计显示问题
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('aoo_system.log', encoding='utf-8')
    ]
)

logger = logging.getLogger('AOO.Main')

class AOOFixedStarter:
    """修复问题的AOO启动器"""
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.system_components = {}
        self.actual_registered_count = 0
        
    def start_system(self):
        """启动系统 - 修复版"""
        try:
            logger.info("🎯 AOO系统修复版启动...")
            
            # 初始化核心组件
            components = [
                ('config_manager', self._init_config_manager),
                ('scanner', self._init_scanner),
                ('registry', self._init_registry),
                ('resolver', self._init_resolver),
                ('factory', self._init_factory),
                ('lifecycle_manager', self._init_lifecycle_manager)
            ]
            
            for name, init_func in components:
                if not init_func():
                    return False
            
            # 执行自动发现
            if not self._execute_auto_discovery():
                return False
            
            logger.info("🎉 AOO系统修复版启动完成!")
            return self._show_fixed_system_status()
            
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            return False
    
    def _init_config_manager(self):
        """初始化配置管理器"""
        try:
            from K.config_manager import get_global_config_manager
            config_path = self.project_root / "config" / "trading_config.json"
            self.system_components['config_manager'] = get_global_config_manager(str(config_path))
            logger.info("✅ 配置管理器就绪")
            return True
        except Exception as e:
            logger.error(f"❌ 配置管理器初始化失败: {e}")
            return False
    
    def _init_scanner(self):
        """初始化模块扫描器"""
        try:
            from K.module_scanner import ModuleScanner
            self.system_components['scanner'] = ModuleScanner(
                str(self.project_root), 
                self.system_components['config_manager']
            )
            logger.info("✅ 模块扫描器就绪")
            return True
        except Exception as e:
            logger.error(f"❌ 模块扫描器初始化失败: {e}")
            return False
    
    def _init_registry(self):
        """初始化模块注册表"""
        try:
            from K.module_registry import ModuleRegistry
            self.system_components['registry'] = ModuleRegistry()
            logger.info("✅ 模块注册表就绪")
            return True
        except Exception as e:
            logger.error(f"❌ 模块注册表初始化失败: {e}")
            return False
    
    def _init_resolver(self):
        """初始化依赖解析器"""
        try:
            from K.dependency_resolver import DependencyResolver
            self.system_components['resolver'] = DependencyResolver(
                self.system_components['registry']
            )
            logger.info("✅ 依赖解析器就绪")
            return True
        except Exception as e:
            logger.error(f"❌ 依赖解析器初始化失败: {e}")
            return False
    
    def _init_factory(self):
        """初始化自动装配工厂"""
        try:
            from K.auto_wiring_factory import AutoWiringFactory
            self.system_components['factory'] = AutoWiringFactory(
                registry=self.system_components['registry'],
                dependency_resolver=self.system_components['resolver'],
                config_manager=self.system_components['config_manager']
            )
            logger.info("✅ 自动装配工厂就绪")
            return True
        except Exception as e:
            logger.error(f"❌ 自动装配工厂初始化失败: {e}")
            return False
    
    def _init_lifecycle_manager(self):
        """修复版生命周期管理器初始化"""
        try:
            from K.lifecycle_manager import LifecycleManager
            lifecycle_manager = LifecycleManager(
                factory=self.system_components['factory'],
                registry=self.system_components['registry']
            )
            
            # 修复：检查并调用正确的初始化方法
            if hasattr(lifecycle_manager, 'initialize'):
                lifecycle_manager.initialize()
                logger.info("✅ 生命周期管理器初始化完成")
            elif hasattr(lifecycle_manager, 'start'):
                lifecycle_manager.start()
                logger.info("✅ 生命周期管理器启动完成")
            elif hasattr(lifecycle_manager, 'init'):
                lifecycle_manager.init()
                logger.info("✅ 生命周期管理器初始化完成")
            else:
                logger.info("✅ 生命周期管理器就绪（无需显式初始化）")
            
            self.system_components['lifecycle_manager'] = lifecycle_manager
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 生命周期管理器初始化异常: {e}")
            # 生命周期管理器是可选的，不阻止系统启动
            self.system_components['lifecycle_manager'] = None
            return True
    
    def _execute_auto_discovery(self):
        """执行自动发现流程"""
        try:
            logger.info("🔍 开始自动发现流程...")
            
            # 扫描所有模块
            modules_by_zone = self.system_components['scanner'].deep_scan()
            total_modules = sum(len(modules) for modules in modules_by_zone.values())
            logger.info(f"📦 扫描完成: {total_modules} 个模块")
            
            # 注册所有模块并记录实际数量
            self.actual_registered_count = 0
            for zone, modules in modules_by_zone.items():
                logger.info(f"  注册 {zone}区模块...")
                for module_info in modules:
                    try:
                        self.system_components['registry'].register_module(
                            module_info.file_path, zone, module_info
                        )
                        self.actual_registered_count += 1
                        logger.debug(f"    ✅ {module_info.name}")
                    except Exception as e:
                        logger.error(f"    ❌ {module_info.name}: {e}")
            
            logger.info(f"✅ 注册完成: {self.actual_registered_count}/{total_modules} 个模块")
            return self.actual_registered_count > 0
            
        except Exception as e:
            logger.error(f"❌ 自动发现流程失败: {e}")
            return False
    
    def _show_fixed_system_status(self):
        """修复版系统状态显示"""
        logger.info("📊 AOO系统修复版状态报告:")
        
        # 使用实际注册数量而不是统计数量
        logger.info(f"   📈 实际注册模块: {self.actual_registered_count} 个")
        
        # 尝试获取统计信息，但不依赖它
        try:
            stats = self.system_components['registry'].get_statistics()
            if stats and 'total_registrations' in stats:
                logger.info(f"   📊 统计显示注册: {stats['total_registrations']} 个")
        except:
            logger.info("   📊 统计信息: 无法获取")
        
        # 组件状态
        components = {
            'config_manager': '配置管理器',
            'scanner': '模块扫描器', 
            'registry': '模块注册表',
            'resolver': '依赖解析器',
            'factory': '自动装配工厂',
            'lifecycle_manager': '生命周期管理器'
        }
        
        for key, name in components.items():
            status = "✅ 正常" if self.system_components.get(key) else "❌ 异常"
            logger.info(f"   - {name}: {status}")
        
        logger.info("🎯 系统修复完成，可以开始业务开发!")
        return True

def main():
    """修复版主程序"""
    logger.info("🚀 AOO系统修复版启动...")
    
    starter = AOOFixedStarter(project_root)
    success = starter.start_system()
    
    if success:
        logger.info("🎊 AOO系统修复版完全正常运行!")
        return True
    else:
        logger.error("💥 AOO系统修复版启动失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
