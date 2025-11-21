"""
Z1插件管理器包

这是一个功能完整的Python插件管理系统，提供插件的加载、管理、执行、配置等功能。

主要功能:
- 插件动态加载和初始化
- 插件生命周期管理
- 插件安全验证
- 插件依赖解析
- 插件配置管理
- 插件使用统计
- 插件版本更新
- 插件注册表管理

使用示例:
    from Z1 import PluginManager, create_plugin_template
    
    # 创建插件管理器
    manager = PluginManager(plugin_dir="plugins")
    
    # 加载插件
    manager.load_plugin("my_plugin")
    
    # 执行插件
    result = manager.execute_plugin("my_plugin", arg1="value1")
    
    # 创建插件模板
    template_path = create_plugin_template(
        name="my_plugin",
        description="My custom plugin"
    )
"""

# 导入主要类
from .PluginManager import (
    # 主要类
    PluginManager,
    
    # 数据类
    PluginInfo,
    PluginExecutionResult,
    
    # 枚举类
    PluginStatus,
    SecurityLevel,
    
    # 组件类
    PluginSecurityValidator,
    PluginDependencyResolver,
    PluginStatistics,
    
    # 工具函数
    create_plugin_template,
)

# 包版本
__version__ = "1.0.0"
__author__ = "Z1 Plugin Manager Team"

# 导出的公共API
__all__ = [
    # 主要类
    "PluginManager",
    
    # 数据类
    "PluginInfo",
    "PluginExecutionResult",
    
    # 枚举
    "PluginStatus",
    "SecurityLevel",
    
    # 组件
    "PluginSecurityValidator",
    "PluginDependencyResolver", 
    "PluginStatistics",
    
    # 工具函数
    "create_plugin_template",
]

# 便捷函数
def quick_start(plugin_dir="plugins", auto_load=True):
    """
    快速启动插件管理器
    
    Args:
        plugin_dir: 插件目录
        auto_load: 是否自动加载插件
        
    Returns:
        PluginManager: 配置好的插件管理器实例
    """
    manager = PluginManager(plugin_dir=plugin_dir)
    
    if auto_load:
        # 自动加载所有插件
        plugins = manager.list_plugins()
        for plugin in plugins:
            try:
                manager.load_plugin(plugin.name)
                print(f"✓ 已加载插件: {plugin.name}")
            except Exception as e:
                print(f"✗ 加载插件失败 {plugin.name}: {e}")
    
    return manager


def create_simple_plugin(name, description="", output_dir="plugins"):
    """
    创建简单插件的便捷函数
    
    Args:
        name: 插件名称
        description: 插件描述
        output_dir: 输出目录
        
    Returns:
        str: 创建的插件文件路径
    """
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成插件文件路径
    plugin_path = os.path.join(output_dir, f"{name}.py")
    
    # 创建插件
    return create_plugin_template(
        name=name,
        description=description or f"{name} 插件",
        output_path=plugin_path
    )


# 插件开发指南
PLUGIN_DEVELOPMENT_GUIDE = """
=== Z1插件开发指南 ===

1. 创建插件:
   from Z1 import create_plugin_template
   
   # 创建插件模板
   create_plugin_template(
       name="my_plugin",
       description="我的自定义插件",
       author="开发者姓名"
   )

2. 插件结构:
   - PLUGIN_INFO: 插件信息配置
   - main(): 插件主函数
   - initialize(): 插件初始化（可选）
   - cleanup(): 插件清理（可选）

3. 插件信息字段:
   - name: 插件名称（唯一）
   - version: 版本号
   - description: 描述
   - author: 作者
   - dependencies: 依赖列表
   - entry_point: 入口函数名
   - security_level: 安全级别
   - config: 默认配置
   - metadata: 元数据

4. 安全级别:
   - low: 低安全级别
   - medium: 中等安全级别（默认）
   - high: 高安全级别
   - critical: 关键安全级别

5. 插件执行:
   result = manager.execute_plugin("plugin_name", arg1="value1", arg2="value2")

6. 插件配置:
   manager.configure_plugin("plugin_name", {"param": "value"})

7. 插件管理:
   # 列出插件
   plugins = manager.list_plugins()
   
   # 加载插件
   manager.load_plugin("plugin_name")
   
   # 卸载插件
   manager.unload_plugin("plugin_name")
   
   # 获取插件信息
   info = manager.get_plugin_info("plugin_name")
   
   # 获取统计信息
   stats = manager.get_plugin_statistics("plugin_name")
"""

# 版本信息
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable",
    "build_date": "2025-11-06"
}


def get_version():
    """获取版本字符串"""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"


def get_version_info():
    """获取详细版本信息"""
    return VERSION_INFO.copy()


# 初始化日志配置
import logging

# 配置默认日志级别
logging.getLogger('Z1PluginManager').setLevel(logging.INFO)

# 添加包级别的文档
__doc__ = """
Z1插件管理器 - 完整的Python插件管理系统

主要特性:
- 🔌 动态插件加载和初始化
- 🛡️ 插件安全验证和检查
- 📊 插件使用统计和分析
- 🔗 插件依赖关系管理
- ⚙️ 插件配置管理
- 🔄 插件版本更新
- 📝 详细的日志记录
- 🧪 完整的测试覆盖

快速开始:
    from Z1 import PluginManager, create_plugin_template
    
    # 创建插件
    create_plugin_template("my_plugin", "我的插件")
    
    # 初始化管理器
    manager = PluginManager()
    
    # 加载并执行插件
    manager.load_plugin("my_plugin")
    result = manager.execute_plugin("my_plugin")

详细文档请参考 Z1插件管理器使用指南.md
"""