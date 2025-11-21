#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R6版本控制器 - 包初始化文件

提供完整的版本控制功能，包括：
- 版本管理和跟踪
- 分支管理和合并
- 用户权限管理
- 版本比较和差异分析
- 版本回滚和恢复
- 版本标签和里程碑
- 合并冲突解决
- 版本发布管理

作者: R6版本控制器团队
版本: 1.0.0
"""

# 核心类
from .VersionController import (
    # 主要类
    VersionController,
    
    # 版本管理
    Version,
    
    # 分支管理
    Branch,
    
    # 用户管理
    User,
    PermissionLevel,
    
    # 便利函数
    create_version,
    merge_branches,
    compare_versions,
    rollback_version,
    create_branch,
    create_tag,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "R6版本控制器团队"
__email__ = "support@r6version.com"
__description__ = "R6版本控制器 - 完整的版本控制系统"

# 包级别的默认配置
DEFAULT_CONFIG = {
    'repository_path': './repository',
    'backup_path': './version_backups',
    'default_branch': 'main',
    'max_versions': 1000,
    'merge': {
        'auto_merge': False,
        'conflict_resolution': 'manual',
        'merge_validation': True,
        'squash_commits': False
    },
    'permissions': {
        'default_permission': 'READ',
        'admin_users': [],
        'restricted_paths': []
    },
    'versioning': {
        'auto_increment': True,
        'semantic_versioning': True,
        'include_timestamp': True
    },
    'release': {
        'auto_tag': False,
        'generate_release_notes': True,
        'notify_on_release': False
    }
}

# 版本类型
VERSION_TYPES = [
    'MAJOR',         # 主版本
    'MINOR',         # 次版本
    'PATCH',         # 补丁版本
    'PRE_RELEASE',   # 预发布版本
    'BUILD',         # 构建版本
]

# 分支类型
BRANCH_TYPES = [
    'FEATURE',       # 功能分支
    'HOTFIX',        # 紧急修复分支
    'RELEASE',       # 发布分支
    'DEVELOPMENT',   # 开发分支
    'MAIN',          # 主分支
]

# 权限级别
PERMISSION_LEVELS = [
    'READ',          # 读取权限
    'WRITE',         # 写入权限
    'DELETE',        # 删除权限
    'ADMIN',         # 管理员权限
    'OWNER',         # 拥有者权限
]

# 合并策略
MERGE_STRATEGIES = [
    'FAST_FORWARD',  # 快进合并
    'THREE_WAY',     # 三方合并
    'SQUASH',        # 压缩合并
    'REBASE',        # 变基合并
    'CHERRY_PICK',   # 樱桃挑选合并
]

# 冲突解决策略
CONFLICT_RESOLUTION = [
    'MANUAL',        # 手动解决
    'OURS',          # 保留我们的版本
    'THEIRS',        # 保留他们的版本
    'AUTO',          # 自动解决
]

# 版本状态
VERSION_STATUS = [
    'DRAFT',         # 草稿
    'ACTIVE',        # 活跃
    'RELEASED',      # 已发布
    'DEPRECATED',    # 已废弃
    'ARCHIVED',      # 已归档
]

# 用户角色
USER_ROLES = [
    'VIEWER',        # 查看者
    'CONTRIBUTOR',   # 贡献者
    'MAINTAINER',    # 维护者
    'ADMIN',         # 管理员
    'OWNER',         # 拥有者
]

# 公开的API函数
__all__ = [
    # 核心类
    'VersionController',
    
    # 版本管理
    'Version',
    
    # 分支管理
    'Branch',
    
    # 用户管理
    'User',
    'PermissionLevel',
    
    # 便利函数
    'create_version',
    'merge_branches',
    'compare_versions',
    'rollback_version',
    'create_branch',
    'create_tag',
    
    # 常量
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'DEFAULT_CONFIG',
    'VERSION_TYPES',
    'BRANCH_TYPES',
    'PERMISSION_LEVELS',
    'MERGE_STRATEGIES',
    'CONFLICT_RESOLUTION',
    'VERSION_STATUS',
    'USER_ROLES',
]

# 快速入门指南
def quick_start():
    """
    快速入门指南
    
    返回一个包含基本使用示例的字符串。
    """
    return """
    R6版本控制器快速入门
    ====================
    
    1. 创建版本:
       ```python
       from R6 import create_version
       
       version = create_version(
           version_type="MINOR",
           description="Add new feature",
           author="developer@example.com"
       )
       ```
    
    2. 创建分支:
       ```python
       from R6 import create_branch
       
       branch = create_branch(
           branch_name="feature/user-management",
           parent_branch="main",
           branch_type="FEATURE"
       )
       ```
    
    3. 合并分支:
       ```python
       from R6 import merge_branches
       
       result = merge_branches(
           source_branch="feature/user-management",
           target_branch="main",
           merge_strategy="THREE_WAY"
       )
       ```
    
    4. 比较版本:
       ```python
       from R6 import compare_versions
       
       diff = compare_versions(
           version_a="v1.0.0",
           version_b="v1.1.0"
       )
       ```
    
    5. 版本回滚:
       ```python
       from R6 import rollback_version
       
       result = rollback_version(
           target_version="v1.0.0",
           reason="Critical bug found"
       )
       ```
    
    6. 创建标签:
       ```python
       from R6 import create_tag
       
       tag = create_tag(
           tag_name="v1.1.0",
           version="v1.1.0",
           message="Release version 1.1.0"
       )
       ```
    
    更多信息请查看文档或运行测试文件。
    """

# 便利函数实现
def create_version(version_type, description, author=None, **kwargs):
    """
    创建版本的便利函数
    """
    version_controller = VersionController()
    return version_controller.create_version(version_type, description, author, **kwargs)

def merge_branches(source_branch, target_branch, **kwargs):
    """
    合并分支的便利函数
    """
    version_controller = VersionController()
    return version_controller.merge(source_branch, target_branch, **kwargs)

def compare_versions(version_a, version_b, **kwargs):
    """
    比较版本的便利函数
    """
    version_controller = VersionController()
    return version_controller.compare(version_a, version_b, **kwargs)

def rollback_version(target_version, reason=None, **kwargs):
    """
    版本回滚的便利函数
    """
    version_controller = VersionController()
    return version_controller.rollback(target_version, reason, **kwargs)

def create_branch(branch_name, parent_branch=None, **kwargs):
    """
    创建分支的便利函数
    """
    version_controller = VersionController()
    return version_controller.create_branch(branch_name, parent_branch, **kwargs)

def create_tag(tag_name, version, message=None, **kwargs):
    """
    创建标签的便利函数
    """
    version_controller = VersionController()
    return version_controller.create_tag(tag_name, version, message, **kwargs)

# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    print(f"R6版本控制器版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")

if __name__ == "__main__":
    check_version()
    print(quick_start())