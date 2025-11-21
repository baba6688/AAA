#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y1文件存储管理器包
一个功能完整的文件存储管理系统
"""

from .FileStorageManager import (
    FileStorageManager,
    FileInfo,
    StorageStats,
    FileType,
    PermissionLevel
)

__version__ = "1.0.0"
__author__ = "Y1开发团队"
__description__ = "Y1文件存储管理器 - 功能完整的文件存储管理系统"

__all__ = [
    "FileStorageManager",
    "FileInfo", 
    "StorageStats",
    "FileType",
    "PermissionLevel"
]

# 包级别的便捷函数
def create_storage_manager(storage_root="storage"):
    """
    创建一个新的文件存储管理器实例
    
    Args:
        storage_root (str): 存储根目录路径
        
    Returns:
        FileStorageManager: 文件存储管理器实例
    """
    return FileStorageManager(storage_root)


def get_file_type_from_extension(extension):
    """
    根据文件扩展名获取文件类型
    
    Args:
        extension (str): 文件扩展名（包含点号）
        
    Returns:
        FileType: 文件类型
    """
    from .FileStorageManager import FileStorageManager
    manager = FileStorageManager()
    return manager._get_file_type(extension)


def format_file_size(size_bytes):
    """
    格式化文件大小显示
    
    Args:
        size_bytes (int): 文件大小（字节）
        
    Returns:
        str: 格式化后的文件大小字符串
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def is_valid_file_path(file_path):
    """
    检查文件路径是否有效
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        bool: 路径是否有效
    """
    try:
        from pathlib import Path
        path = Path(file_path)
        # 检查路径是否包含无效字符
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        return not any(char in str(path) for char in invalid_chars)
    except Exception:
        return False


# 版本信息
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}


def get_version():
    """
    获取版本信息
    
    Returns:
        str: 版本字符串
    """
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"


def get_version_info():
    """
    获取详细版本信息
    
    Returns:
        dict: 版本信息字典
    """
    return VERSION_INFO.copy()


# 默认配置
DEFAULT_CONFIG = {
    "storage_root": "storage",
    "backup_enabled": True,
    "max_backup_count": 10,
    "temp_file_max_age_hours": 24,
    "metadata_backup_enabled": True,
    "auto_cleanup_enabled": True,
    "default_permission": "private",
    "supported_image_types": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"],
    "supported_video_types": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"],
    "supported_audio_types": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma"],
    "supported_document_types": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"],
    "supported_archive_types": [".zip", ".rar", ".7z", ".tar", ".gz"],
    "supported_code_types": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".php"]
}


def get_default_config():
    """
    获取默认配置
    
    Returns:
        dict: 默认配置字典
    """
    return DEFAULT_CONFIG.copy()


def validate_config(config):
    """
    验证配置字典
    
    Args:
        config (dict): 配置字典
        
    Returns:
        tuple: (是否有效, 错误信息)
    """
    required_keys = ["storage_root"]
    
    for key in required_keys:
        if key not in config:
            return False, f"缺少必需的配置项: {key}"
    
    # 验证存储根目录
    if not isinstance(config["storage_root"], str):
        return False, "storage_root必须是字符串"
    
    # 验证布尔值配置
    bool_keys = ["backup_enabled", "metadata_backup_enabled", "auto_cleanup_enabled"]
    for key in bool_keys:
        if key in config and not isinstance(config[key], bool):
            return False, f"{key}必须是布尔值"
    
    # 验证整数配置
    int_keys = ["max_backup_count", "temp_file_max_age_hours"]
    for key in int_keys:
        if key in config and not isinstance(config[key], int):
            return False, f"{key}必须是整数"
    
    return True, "配置有效"


# 工具函数
def calculate_directory_size(directory_path):
    """
    计算目录大小
    
    Args:
        directory_path (str): 目录路径
        
    Returns:
        int: 目录大小（字节）
    """
    try:
        from pathlib import Path
        total_size = 0
        directory = Path(directory_path)
        
        if not directory.exists():
            return 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    except Exception:
        return 0


def find_duplicate_files(directory_path, by_content=True):
    """
    查找重复文件
    
    Args:
        directory_path (str): 目录路径
        by_content (bool): 是否按内容比较（否则按大小）
        
    Returns:
        dict: 重复文件组字典
    """
    try:
        from pathlib import Path
        from collections import defaultdict
        import hashlib
        
        directory = Path(directory_path)
        if not directory.exists():
            return {}
        
        file_groups = defaultdict(list)
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                if by_content:
                    # 按内容比较
                    file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
                else:
                    # 按大小比较
                    file_hash = str(file_path.stat().st_size)
                
                file_groups[file_hash].append(str(file_path))
        
        # 返回只有多个文件的组
        return {k: v for k, v in file_groups.items() if len(v) > 1}
    
    except Exception:
        return {}


def clean_empty_directories(directory_path):
    """
    清理空目录
    
    Args:
        directory_path (str): 目录路径
        
    Returns:
        int: 清理的目录数量
    """
    try:
        from pathlib import Path
        
        directory = Path(directory_path)
        if not directory.exists():
            return 0
        
        cleaned_count = 0
        
        # 从最深层开始清理
        for subdir in sorted(directory.rglob('*'), key=lambda x: len(str(x)), reverse=True):
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()
                cleaned_count += 1
        
        return cleaned_count
    
    except Exception:
        return 0


# 异常类
class FileStorageError(Exception):
    """文件存储基础异常"""
    pass


class FileNotFoundError(FileStorageError):
    """文件未找到异常"""
    pass


class PermissionError(FileStorageError):
    """权限错误异常"""
    pass


class StorageFullError(FileStorageError):
    """存储空间不足异常"""
    pass


class InvalidFileError(FileStorageError):
    """无效文件异常"""
    pass


# 导出异常类
__all__.extend([
    "FileStorageError",
    "FileNotFoundError", 
    "PermissionError",
    "StorageFullError",
    "InvalidFileError",
    "create_storage_manager",
    "get_file_type_from_extension",
    "format_file_size",
    "is_valid_file_path",
    "get_version",
    "get_version_info",
    "get_default_config",
    "validate_config",
    "calculate_directory_size",
    "find_duplicate_files",
    "clean_empty_directories"
])