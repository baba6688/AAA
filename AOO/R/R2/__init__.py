#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2配置备份器 - 包初始化文件

提供完整的配置备份、恢复、版本管理功能，包括：
- 配置文件的备份和恢复
- 配置版本管理
- 配置验证和完整性检查
- 敏感数据保护
- 配置迁移和转换
- 多格式支持（JSON、YAML、TOML）
- 配置差异比较
- 回滚功能

作者: R2配置备份器团队
版本: 1.0.0
"""

# 核心类
from .ConfigBackup import (
    # 主要类
    ConfigBackup,
    
    # 备份管理器
    BackupManager,
    
    # 验证器
    ConfigValidator,
    
    # 敏感数据保护
    SensitiveDataProtector,
    
    # 数据结构
    ConfigType,
    BackupStatus,
    ConfigMetadata,
    BackupRecord,
    
    # 便利函数
    create_config_backup,
    restore_config,
    validate_config,
    protect_sensitive_data,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "R2配置备份器团队"
__email__ = "support@r2config.com"
__description__ = "R2配置备份器 - 完整的配置备份、恢复和版本管理解决方案"

# 包级别的默认配置
DEFAULT_CONFIG = {
    'backup_dir': './config_backups',
    'backup_format': 'yaml',
    'compression': 'gzip',
    'encryption': False,
    'retention_days': 90,
    'validation': {
        'strict_mode': False,
        'schema_validation': True,
        'type_checking': True
    },
    'protection': {
        'mask_sensitive': True,
        'sensitive_keywords': ['password', 'secret', 'key', 'token'],
        'protection_level': 'medium'
    },
    'migration': {
        'auto_migrate': True,
        'backup_before_migration': True,
        'validation_after_migration': True
    }
}

# 支持的配置格式
SUPPORTED_FORMATS = [
    'json',       # JSON格式
    'yaml',       # YAML格式
    'toml',       # TOML格式
    'ini',        # INI格式
    'properties', # Java Properties格式
    'xml',        # XML格式
    'env',        # 环境变量格式
]

# 配置类型
CONFIG_TYPES = [
    'application',    # 应用配置
    'database',       # 数据库配置
    'network',        # 网络配置
    'security',       # 安全配置
    'logging',        # 日志配置
    'cache',          # 缓存配置
    'api',            # API配置
    'custom',         # 自定义配置
]

# 验证级别
VALIDATION_LEVELS = [
    'none',      # 无验证
    'basic',     # 基础验证
    'strict',    # 严格验证
    'complete',  # 完整验证
]

# 保护级别
PROTECTION_LEVELS = [
    'low',       # 低级保护
    'medium',    # 中级保护
    'high',      # 高级保护
    'maximum',   # 最大保护
]

# 备份类型
BACKUP_TYPES = [
    'full',         # 完整备份
    'incremental',  # 增量备份
    'differential', # 差异备份
]

# 迁移状态
MIGRATION_STATUS = [
    'pending',      # 待执行
    'running',      # 执行中
    'success',      # 成功
    'failed',       # 失败
    'rollback',     # 回滚中
]

# 公开的API函数
__all__ = [
    # 核心类
    'ConfigBackup',
    
    # 管理器
    'BackupManager',
    'ConfigValidator',
    'SensitiveDataProtector',
    
    # 数据结构
    'ConfigType',
    'BackupStatus',
    'ConfigMetadata',
    'BackupRecord',
    
    # 便利函数
    'create_config_backup',
    'restore_config',
    'validate_config',
    'protect_sensitive_data',
    
    # 常量
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'DEFAULT_CONFIG',
    'SUPPORTED_FORMATS',
    'CONFIG_TYPES',
    'VALIDATION_LEVELS',
    'PROTECTION_LEVELS',
    'BACKUP_TYPES',
    'MIGRATION_STATUS',
]

# 快速入门指南
def quick_start():
    """
    快速入门指南
    
    返回一个包含基本使用示例的字符串。
    """
    return """
    R2配置备份器快速入门
    ====================
    
    1. 基本配置备份:
       ```python
       from R2 import create_config_backup
       
       result = create_config_backup(
           config_path="/path/to/config.yaml",
           backup_path="/path/to/backup",
           backup_id="my_config"
       )
       ```
    
    2. 配置恢复:
       ```python
       from R2 import restore_config
       
       result = restore_config(
           backup_id="my_config",
           restore_path="/path/to/restore"
       )
       ```
    
    3. 配置验证:
       ```python
       from R2 import validate_config
       
       is_valid = validate_config(
           config_path="/path/to/config.yaml",
           validation_level="strict"
       )
       ```
    
    4. 敏感数据保护:
       ```python
       from R2 import protect_sensitive_data
       
       protected_config = protect_sensitive_data(
           config_path="/path/to/config.yaml",
           protection_level="high"
       )
       ```
    
    5. 格式转换:
       ```python
       from R2 import convert_config_format
       
       convert_config_format(
           source_path="/path/to/config.yaml",
           target_path="/path/to/config.json",
           target_format="json"
       )
       ```
    
    更多信息请查看文档或运行测试文件。
    """

# 临时格式转换器类
class FormatConverter:
    """临时的格式转换器类"""
    def convert(self, source_path, target_path, target_format, **kwargs):
        """
        转换配置格式的临时实现
        """
        import os
        import json
        
        # 简单的格式转换实现
        if target_format.lower() == 'json':
            try:
                # 尝试读取源文件
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 简单的格式转换（这里仅作示例）
                if target_format == 'json':
                    result = {'content': content, 'format': target_format}
                    with open(target_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    with open(target_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                return {'success': True, 'message': f'转换成功: {source_path} -> {target_path}'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        else:
            # 对于其他格式的简单实现
            try:
                import shutil
                shutil.copy2(source_path, target_path)
                return {'success': True, 'message': f'文件复制成功: {source_path} -> {target_path}'}
            except Exception as e:
                return {'success': False, 'error': str(e)}

# 便利函数实现
def create_config_backup(config_path, backup_path, backup_id=None, **kwargs):
    """
    创建配置备份的便利函数
    """
    backup_manager = BackupManager()
    return backup_manager.create_backup(config_path, backup_path, backup_id, **kwargs)

def restore_config(backup_id, restore_path, **kwargs):
    """
    恢复配置的便利函数
    """
    backup_manager = BackupManager()
    return backup_manager.restore_backup(backup_id, restore_path, **kwargs)

def validate_config(config_path, validation_level='basic', **kwargs):
    """
    验证配置的便利函数
    """
    validator = ConfigValidator()
    return validator.validate(config_path, validation_level, **kwargs)

def protect_sensitive_data(config_path, protection_level='medium', **kwargs):
    """
    保护敏感数据的便利函数
    """
    protector = SensitiveDataProtector()
    return protector.protect(config_path, protection_level, **kwargs)

def convert_config_format(source_path, target_path, target_format, **kwargs):
    """
    转换配置格式的便利函数
    """
    converter = FormatConverter()
    return converter.convert(source_path, target_path, target_format, **kwargs)

# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    print(f"R2配置备份器版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")

if __name__ == "__main__":
    check_version()
    print(quick_start())