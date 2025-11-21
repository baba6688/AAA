#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y3对象存储管理器

提供完整的云对象存储服务管理功能，包括对象上传、下载、管理、
权限控制、版本管理、统计分析和备份恢复等功能。

作者: Y3开发团队
版本: 1.0.0
日期: 2025-11-06

主要功能：
- 对象存储：支持多种云存储服务（AWS S3、阿里云OSS、腾讯云COS等）
- 对象上传：支持文件上传、元数据设置、标签管理
- 对象下载：支持对象下载、断点续传
- 对象管理：支持对象生命周期管理、列表查询
- 对象权限：支持权限设置和访问控制
- 对象版本：支持版本管理和版本控制
- 对象统计：支持使用统计和性能监控
- 对象备份：支持对象备份和恢复

使用示例：
    from Y3 import create_local_storage_manager
    
    # 创建本地存储管理器
    manager = create_local_storage_manager('/path/to/storage', 'my_bucket')
    
    # 上传文件
    metadata = manager.upload_object('local_file.txt', 'remote_file.txt')
    
    # 下载文件
    manager.download_object('remote_file.txt', 'downloaded_file.txt')
    
    # 获取对象信息
    metadata = manager.get_object_metadata('remote_file.txt')
    
    # 列出对象
    objects = manager.list_objects()
    
    # 删除对象
    manager.delete_object('remote_file.txt')
"""

from .ObjectStorageManager import (
    ObjectStorageManager,
    StorageConfig,
    StorageProvider,
    ObjectStatus,
    Permission,
    ObjectMetadata,
    create_local_storage_manager,
    create_aws_s3_manager,
    create_aliyun_oss_manager
)

__version__ = "1.0.0"
__author__ = "Y3开发团队"
__email__ = "y3-team@example.com"
__description__ = "Y3对象存储管理器 - 完整的云对象存储解决方案"

# 导出的公共API
__all__ = [
    "ObjectStorageManager",
    "StorageConfig", 
    "StorageProvider",
    "ObjectStatus",
    "Permission",
    "ObjectMetadata",
    "create_local_storage_manager",
    "create_aws_s3_manager", 
    "create_aliyun_oss_manager",
    "__version__",
    "__author__",
    "__description__"
]

# 版本信息
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version():
    """获取版本信息"""
    return __version__

def get_version_info():
    """获取详细版本信息"""
    return VERSION_INFO.copy()

# 默认配置
DEFAULT_CONFIG = {
    "timeout": 300,
    "retry_count": 3,
    "chunk_size": 8192,
    "max_keys": 1000
}

def get_default_config():
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()

# 支持的存储提供商
SUPPORTED_PROVIDERS = {
    "aws_s3": "Amazon S3",
    "aliyun_oss": "阿里云OSS", 
    "tencent_cos": "腾讯云COS",
    "qiniu": "七牛云",
    "local": "本地存储"
}

def get_supported_providers():
    """获取支持的存储提供商列表"""
    return SUPPORTED_PROVIDERS.copy()

# 功能特性
FEATURES = {
    "upload": "对象上传",
    "download": "对象下载", 
    "delete": "对象删除",
    "list": "对象列表",
    "metadata": "元数据管理",
    "permissions": "权限控制",
    "versioning": "版本管理",
    "backup": "备份恢复",
    "statistics": "统计分析",
    "health_check": "健康检查"
}

def get_features():
    """获取功能特性列表"""
    return FEATURES.copy()

# 快速入门指南
QUICK_START = """
快速入门指南：

1. 安装依赖：
   pip install boto3 oss2

2. 创建存储管理器：
   from Y3 import create_local_storage_manager
   manager = create_local_storage_manager('/path/to/storage', 'my_bucket')

3. 上传文件：
   metadata = manager.upload_object('local_file.txt', 'remote_file.txt')

4. 下载文件：
   manager.download_object('remote_file.txt', 'downloaded_file.txt')

5. 查看文档：
   详细使用说明请参考 Y3对象存储管理器使用指南.md
"""

def print_quick_start():
    """打印快速入门指南"""
    print(QUICK_START)

# 模块初始化日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"Y3对象存储管理器 v{__version__} 初始化完成")
logger.info(f"支持的存储提供商: {', '.join(SUPPORTED_PROVIDERS.values())}")
logger.info(f"主要功能: {', '.join(FEATURES.values())}")