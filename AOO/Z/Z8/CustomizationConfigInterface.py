"""
Z8定制化配置接口主要实现

提供完整的系统定制化配置解决方案，包括配置定制、参数调整、
界面定制、主题管理、布局定制、功能定制、行为定制和配置导出等功能。
"""

import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import copy

# 可选导入yaml
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False


class ConfigType(Enum):
    """配置类型枚举"""
    SYSTEM = "system"
    USER = "user"
    THEME = "theme"
    LAYOUT = "layout"
    FEATURE = "feature"
    BEHAVIOR = "behavior"


class ThemeType(Enum):
    """主题类型枚举"""
    LIGHT = "light"
    DARK = "dark"
    CUSTOM = "custom"


class LayoutType(Enum):
    """布局类型枚举"""
    GRID = "grid"
    FLEX = "flex"
    ABSOLUTE = "absolute"


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.configs = {}
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
    
    def load_config(self, config_name: str, config_type: ConfigType) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = os.path.join(self.config_dir, f"{config_name}_{config_type.value}.json")
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                self.configs[config_name] = json.load(f)
                return self.configs[config_name]
        return {}
    
    def save_config(self, config_name: str, config_data: Dict[str, Any], config_type: ConfigType):
        """保存配置文件"""
        config_file = os.path.join(self.config_dir, f"{config_name}_{config_type.value}.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        self.configs[config_name] = config_data
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """获取配置"""
        return self.configs.get(config_name, {})
    
    def list_configs(self) -> List[str]:
        """列出所有配置"""
        return list(self.configs.keys())


class ParameterAdjuster:
    """参数调整器"""
    
    def __init__(self):
        self.parameters = {}
        self.parameter_ranges = {}
        self.parameter_defaults = {}
    
    def set_parameter(self, param_name: str, value: Any, param_type: str = "general"):
        """设置参数值"""
        if param_name in self.parameter_ranges:
            min_val, max_val = self.parameter_ranges[param_name]
            if not (min_val <= value <= max_val):
                raise ValueError(f"参数 {param_name} 的值 {value} 超出允许范围 [{min_val}, {max_val}]")
        
        if param_name not in self.parameters:
            self.parameters[param_name] = {}
        
        self.parameters[param_name][param_type] = value
        return True
    
    def get_parameter(self, param_name: str, param_type: str = "general") -> Any:
        """获取参数值"""
        return self.parameters.get(param_name, {}).get(param_type, 
                self.parameter_defaults.get(param_name))
    
    def set_parameter_range(self, param_name: str, min_val: Any, max_val: Any):
        """设置参数范围"""
        self.parameter_ranges[param_name] = (min_val, max_val)
    
    def set_parameter_default(self, param_name: str, default_value: Any):
        """设置参数默认值"""
        self.parameter_defaults[param_name] = default_value
    
    def reset_parameter(self, param_name: str, param_type: str = "general"):
        """重置参数到默认值"""
        if param_name in self.parameter_defaults:
            self.set_parameter(param_name, self.parameter_defaults[param_name], param_type)
    
    def list_parameters(self) -> List[str]:
        """列出所有参数"""
        return list(self.parameters.keys())


class InterfaceCustomizer:
    """界面定制器"""
    
    def __init__(self):
        self.interface_configs = {}
        self.component_styles = {}
        self.ui_themes = {}
    
    def customize_component(self, component_name: str, style_config: Dict[str, Any]):
        """定制组件样式"""
        self.component_styles[component_name] = style_config
        return True
    
    def get_component_style(self, component_name: str) -> Dict[str, Any]:
        """获取组件样式"""
        return self.component_styles.get(component_name, {})
    
    def set_interface_config(self, interface_name: str, config: Dict[str, Any]):
        """设置界面配置"""
        self.interface_configs[interface_name] = config
    
    def get_interface_config(self, interface_name: str) -> Dict[str, Any]:
        """获取界面配置"""
        return self.interface_configs.get(interface_name, {})
    
    def apply_ui_theme(self, theme_name: str, theme_data: Dict[str, Any]):
        """应用UI主题"""
        self.ui_themes[theme_name] = theme_data
    
    def get_ui_theme(self, theme_name: str) -> Dict[str, Any]:
        """获取UI主题"""
        return self.ui_themes.get(theme_name, {})


class ThemeManager:
    """主题管理器"""
    
    def __init__(self):
        self.themes = {}
        self.current_theme = None
        self.theme_assets = {}
    
    def create_theme(self, theme_name: str, theme_config: Dict[str, Any]):
        """创建主题"""
        theme_config['created_at'] = datetime.now().isoformat()
        theme_config['theme_type'] = theme_config.get('theme_type', ThemeType.CUSTOM.value)
        self.themes[theme_name] = theme_config
    
    def apply_theme(self, theme_name: str) -> bool:
        """应用主题"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            return True
        return False
    
    def get_current_theme(self) -> Optional[Dict[str, Any]]:
        """获取当前主题"""
        if self.current_theme:
            return self.themes.get(self.current_theme)
        return None
    
    def list_themes(self) -> List[str]:
        """列出所有主题"""
        return list(self.themes.keys())
    
    def delete_theme(self, theme_name: str) -> bool:
        """删除主题"""
        if theme_name in self.themes:
            del self.themes[theme_name]
            if self.current_theme == theme_name:
                self.current_theme = None
            return True
        return False
    
    def export_theme(self, theme_name: str, file_path: str) -> bool:
        """导出主题"""
        if theme_name in self.themes:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.themes[theme_name], f, ensure_ascii=False, indent=2)
            return True
        return False
    
    def import_theme(self, file_path: str) -> Optional[str]:
        """导入主题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                theme_data = json.load(f)
            theme_name = theme_data.get('name', f"imported_theme_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.create_theme(theme_name, theme_data)
            return theme_name
        except Exception:
            return None


class LayoutCustomizer:
    """布局定制器"""
    
    def __init__(self):
        self.layouts = {}
        self.layout_templates = {}
    
    def create_layout(self, layout_name: str, layout_config: Dict[str, Any]):
        """创建布局"""
        layout_config['created_at'] = datetime.now().isoformat()
        layout_config['layout_type'] = layout_config.get('layout_type', LayoutType.GRID.value)
        self.layouts[layout_name] = layout_config
    
    def apply_layout(self, layout_name: str) -> bool:
        """应用布局"""
        if layout_name in self.layouts:
            return True
        return False
    
    def get_layout(self, layout_name: str) -> Dict[str, Any]:
        """获取布局配置"""
        return self.layouts.get(layout_name, {})
    
    def list_layouts(self) -> List[str]:
        """列出所有布局"""
        return list(self.layouts.keys())
    
    def create_layout_template(self, template_name: str, template_config: Dict[str, Any]):
        """创建布局模板"""
        self.layout_templates[template_name] = template_config
    
    def apply_layout_template(self, template_name: str, layout_name: str) -> bool:
        """应用布局模板"""
        if template_name in self.layout_templates and layout_name in self.layouts:
            template = self.layout_templates[template_name]
            self.layouts[layout_name].update(template)
            return True
        return False


class FeatureCustomizer:
    """功能定制器"""
    
    def __init__(self):
        self.feature_configs = {}
        self.enabled_features = set()
        self.feature_dependencies = {}
    
    def enable_feature(self, feature_name: str) -> bool:
        """启用功能"""
        if self._check_dependencies(feature_name):
            self.enabled_features.add(feature_name)
            return True
        return False
    
    def disable_feature(self, feature_name: str) -> bool:
        """禁用功能"""
        if feature_name in self.enabled_features:
            # 检查是否有其他功能依赖此功能
            dependents = self._get_dependents(feature_name)
            if dependents:
                return False  # 不能禁用被依赖的功能
            self.enabled_features.remove(feature_name)
            return True
        return False
    
    def set_feature_config(self, feature_name: str, config: Dict[str, Any]):
        """设置功能配置"""
        self.feature_configs[feature_name] = config
    
    def get_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """获取功能配置"""
        return self.feature_configs.get(feature_name, {})
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """检查功能是否启用"""
        return feature_name in self.enabled_features
    
    def list_enabled_features(self) -> List[str]:
        """列出启用的功能"""
        return list(self.enabled_features)
    
    def set_feature_dependency(self, feature_name: str, dependencies: List[str]):
        """设置功能依赖"""
        self.feature_dependencies[feature_name] = dependencies
    
    def _check_dependencies(self, feature_name: str) -> bool:
        """检查依赖是否满足"""
        if feature_name not in self.feature_dependencies:
            return True
        dependencies = self.feature_dependencies[feature_name]
        return all(dep in self.enabled_features for dep in dependencies)
    
    def _get_dependents(self, feature_name: str) -> List[str]:
        """获取依赖指定功能的功能列表"""
        dependents = []
        for feat, deps in self.feature_dependencies.items():
            if feature_name in deps and feat in self.enabled_features:
                dependents.append(feat)
        return dependents


class BehaviorCustomizer:
    """行为定制器"""
    
    def __init__(self):
        self.behavior_configs = {}
        self.interaction_rules = {}
        self.response_templates = {}
    
    def set_behavior_config(self, behavior_name: str, config: Dict[str, Any]):
        """设置行为配置"""
        self.behavior_configs[behavior_name] = config
    
    def get_behavior_config(self, behavior_name: str) -> Dict[str, Any]:
        """获取行为配置"""
        return self.behavior_configs.get(behavior_name, {})
    
    def set_interaction_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """设置交互规则"""
        self.interaction_rules[rule_name] = rule_config
    
    def get_interaction_rule(self, rule_name: str) -> Dict[str, Any]:
        """获取交互规则"""
        return self.interaction_rules.get(rule_name, {})
    
    def set_response_template(self, template_name: str, template: str):
        """设置响应模板"""
        self.response_templates[template_name] = template
    
    def get_response_template(self, template_name: str) -> str:
        """获取响应模板"""
        return self.response_templates.get(template_name, "")
    
    def customize_system_behavior(self, behavior_type: str, custom_config: Dict[str, Any]):
        """定制系统行为"""
        if behavior_type not in self.behavior_configs:
            self.behavior_configs[behavior_type] = {}
        self.behavior_configs[behavior_type].update(custom_config)


class ConfigExporter:
    """配置导出器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def export_config(self, config_name: str, export_path: str, format: str = "json") -> bool:
        """导出配置"""
        config_data = self.config_manager.get_config(config_name)
        if not config_data:
            return False
        
        try:
            if format.lower() == "json":
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
            elif format.lower() == "yaml":
                if not YAML_AVAILABLE:
                    raise ImportError("YAML support not available. Install PyYAML to use YAML format.")
                with open(export_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)
            return True
        except Exception:
            return False
    
    def import_config(self, import_path: str, config_name: str, format: str = "json") -> bool:
        """导入配置"""
        try:
            if format.lower() == "json":
                with open(import_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            elif format.lower() == "yaml":
                if not YAML_AVAILABLE:
                    raise ImportError("YAML support not available. Install PyYAML to use YAML format.")
                with open(import_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            else:
                return False
            
            # 保存到配置管理器
            config_type = ConfigType.USER  # 默认为用户配置
            self.config_manager.save_config(config_name, config_data, config_type)
            return True
        except Exception:
            return False
    
    def export_all_configs(self, export_dir: str, format: str = "json") -> List[str]:
        """导出所有配置"""
        exported_configs = []
        for config_name in self.config_manager.list_configs():
            export_path = os.path.join(export_dir, f"{config_name}.{format}")
            if self.export_config(config_name, export_path, format):
                exported_configs.append(config_name)
        return exported_configs


class CustomizationConfigInterface:
    """Z8定制化配置接口主类"""
    
    def __init__(self, config_dir: str = "configs"):
        """初始化定制化配置接口"""
        self.config_manager = ConfigManager(config_dir)
        self.parameter_adjuster = ParameterAdjuster()
        self.interface_customizer = InterfaceCustomizer()
        self.theme_manager = ThemeManager()
        self.layout_customizer = LayoutCustomizer()
        self.feature_customizer = FeatureCustomizer()
        self.behavior_customizer = BehaviorCustomizer()
        self.config_exporter = ConfigExporter(self.config_manager)
        
        # 初始化默认配置
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """初始化默认配置"""
        # 默认主题
        default_theme = {
            "name": "默认主题",
            "theme_type": ThemeType.LIGHT.value,
            "colors": {
                "primary": "#007bff",
                "secondary": "#6c757d",
                "success": "#28a745",
                "danger": "#dc3545",
                "warning": "#ffc107",
                "info": "#17a2b8"
            },
            "fonts": {
                "primary": "Arial, sans-serif",
                "monospace": "Courier New, monospace"
            },
            "spacing": {
                "small": "8px",
                "medium": "16px",
                "large": "24px"
            }
        }
        self.theme_manager.create_theme("default", default_theme)
        self.theme_manager.apply_theme("default")
        
        # 默认布局
        default_layout = {
            "name": "默认布局",
            "layout_type": LayoutType.GRID.value,
            "structure": {
                "header": {"height": "60px"},
                "sidebar": {"width": "250px"},
                "main": {"flex": 1},
                "footer": {"height": "40px"}
            }
        }
        self.layout_customizer.create_layout("default", default_layout)
        
        # 默认功能配置
        default_features = ["basic_ui", "navigation", "search"]
        for feature in default_features:
            self.feature_customizer.enable_feature(feature)
    
    # 配置定制方法
    def customize_system_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """定制系统配置"""
        try:
            self.config_manager.save_config(config_name, config_data, ConfigType.SYSTEM)
            return True
        except Exception:
            return False
    
    def get_system_config(self, config_name: str) -> Dict[str, Any]:
        """获取系统配置"""
        return self.config_manager.get_config(config_name)
    
    # 参数调整方法
    def adjust_parameter(self, param_name: str, value: Any, param_type: str = "general") -> bool:
        """调整系统参数"""
        try:
            return self.parameter_adjuster.set_parameter(param_name, value, param_type)
        except Exception:
            return False
    
    def get_parameter(self, param_name: str, param_type: str = "general") -> Any:
        """获取参数值"""
        return self.parameter_adjuster.get_parameter(param_name, param_type)
    
    # 界面定制方法
    def customize_interface(self, interface_name: str, config: Dict[str, Any]) -> bool:
        """定制用户界面"""
        try:
            self.interface_customizer.set_interface_config(interface_name, config)
            return True
        except Exception:
            return False
    
    def get_interface_config(self, interface_name: str) -> Dict[str, Any]:
        """获取界面配置"""
        return self.interface_customizer.get_interface_config(interface_name)
    
    # 主题管理方法
    def create_custom_theme(self, theme_name: str, theme_config: Dict[str, Any]) -> bool:
        """创建自定义主题"""
        try:
            self.theme_manager.create_theme(theme_name, theme_config)
            return True
        except Exception:
            return False
    
    def apply_theme(self, theme_name: str) -> bool:
        """应用主题"""
        return self.theme_manager.apply_theme(theme_name)
    
    def get_current_theme(self) -> Optional[Dict[str, Any]]:
        """获取当前主题"""
        return self.theme_manager.get_current_theme()
    
    # 布局定制方法
    def create_custom_layout(self, layout_name: str, layout_config: Dict[str, Any]) -> bool:
        """创建自定义布局"""
        try:
            self.layout_customizer.create_layout(layout_name, layout_config)
            return True
        except Exception:
            return False
    
    def apply_layout(self, layout_name: str) -> bool:
        """应用布局"""
        return self.layout_customizer.apply_layout(layout_name)
    
    # 功能定制方法
    def enable_feature(self, feature_name: str) -> bool:
        """启用功能模块"""
        return self.feature_customizer.enable_feature(feature_name)
    
    def disable_feature(self, feature_name: str) -> bool:
        """禁用功能模块"""
        return self.feature_customizer.disable_feature(feature_name)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """检查功能是否启用"""
        return self.feature_customizer.is_feature_enabled(feature_name)
    
    # 行为定制方法
    def customize_behavior(self, behavior_type: str, config: Dict[str, Any]) -> bool:
        """定制系统行为"""
        try:
            self.behavior_customizer.customize_system_behavior(behavior_type, config)
            return True
        except Exception:
            return False
    
    # 配置导出导入方法
    def export_config(self, config_name: str, file_path: str, format: str = "json") -> bool:
        """导出配置"""
        return self.config_exporter.export_config(config_name, file_path, format)
    
    def import_config(self, file_path: str, config_name: str, format: str = "json") -> bool:
        """导入配置"""
        return self.config_exporter.import_config(file_path, config_name, format)
    
    def export_all_configs(self, export_dir: str, format: str = "json") -> List[str]:
        """导出所有配置"""
        return self.config_exporter.export_all_configs(export_dir, format)
    
    # 状态查询方法
    def get_customization_status(self) -> Dict[str, Any]:
        """获取定制化状态"""
        return {
            "current_theme": self.theme_manager.current_theme,
            "available_themes": self.theme_manager.list_themes(),
            "available_layouts": self.layout_customizer.list_layouts(),
            "enabled_features": self.feature_customizer.list_enabled_features(),
            "configured_interfaces": list(self.interface_customizer.interface_configs.keys()),
            "system_configs": self.config_manager.list_configs()
        }
    
    def reset_to_defaults(self) -> bool:
        """重置为默认配置"""
        try:
            # 重新初始化
            self.__init__(self.config_manager.config_dir)
            return True
        except Exception:
            return False
    
    def backup_configuration(self, backup_path: str) -> bool:
        """备份配置"""
        try:
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "themes": self.theme_manager.themes,
                "layouts": self.layout_customizer.layouts,
                "features": {
                    "enabled": list(self.feature_customizer.enabled_features),
                    "configs": self.feature_customizer.feature_configs,
                    "dependencies": self.feature_customizer.feature_dependencies
                },
                "interfaces": self.interface_customizer.interface_configs,
                "behaviors": self.behavior_customizer.behavior_configs,
                "parameters": self.parameter_adjuster.parameters
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False
    
    def restore_configuration(self, backup_path: str) -> bool:
        """恢复配置"""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # 恢复主题
            for theme_name, theme_data in backup_data.get("themes", {}).items():
                self.theme_manager.create_theme(theme_name, theme_data)
            
            # 恢复布局
            for layout_name, layout_data in backup_data.get("layouts", {}).items():
                self.layout_customizer.create_layout(layout_name, layout_data)
            
            # 恢复功能
            features_data = backup_data.get("features", {})
            self.feature_customizer.enabled_features = set(features_data.get("enabled", []))
            self.feature_customizer.feature_configs = features_data.get("configs", {})
            self.feature_customizer.feature_dependencies = features_data.get("dependencies", {})
            
            # 恢复界面配置
            self.interface_customizer.interface_configs = backup_data.get("interfaces", {})
            
            # 恢复行为配置
            self.behavior_customizer.behavior_configs = backup_data.get("behaviors", {})
            
            # 恢复参数
            self.parameter_adjuster.parameters = backup_data.get("parameters", {})
            
            return True
        except Exception:
            return False