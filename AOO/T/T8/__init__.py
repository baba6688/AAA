#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T区T8子模块 - 数据版本控制器

一个功能完整的数据版本管理系统，支持数据版本的全生命周期管理，
包括版本追踪、历史记录、差异分析、回滚恢复、分支合并、标签发布、
依赖管理、访问控制和统计分析等功能。

版本信息:
    版本: 1.0.0
    创建时间: 2025-11-05
    作者: T8开发团队
    许可证: MIT

快速开始:
    >>> from T.T8 import DataVersionController, VersionStatus, AccessLevel
    >>> 
    >>> # 创建控制器实例
    >>> controller = DataVersionController("my_data_versions")
    >>> 
    >>> # 授予用户权限
    >>> controller.grant_permission("user1", "data", "*", AccessLevel.READ_WRITE, "admin")
    >>> 
    >>> # 创建数据版本
    >>> version = controller.create_version(
    ...     data_id="my_data",
    ...     data={"name": "测试数据", "value": 100},
    ...     user="user1",
    ...     description="初始版本"
    ... )
    >>> 
    >>> print(f"版本ID: {version.version_id}")

主要特性:
    ✓ 版本管理 - 创建、更新、标记数据版本
    ✓ 历史记录 - 追踪所有变更历史
    ✓ 差异分析 - 比较版本间的差异
    ✓ 回滚恢复 - 支持版本回滚和恢复
    ✓ 分支管理 - 支持版本分支和合并
    ✓ 标签发布 - 版本标签和发布管理
    ✓ 依赖管理 - 版本间依赖关系管理
    ✓ 访问控制 - 细粒度权限控制
    ✓ 统计分析 - 生成版本控制报告

更多信息请访问项目文档。
"""

from .DataVersionController import (
    # 枚举类
    VersionStatus,
    AccessLevel,
    ChangeType,
    
    # 数据类
    DataVersion,
    ChangeRecord,
    VersionBranch,
    AccessPermission,
    
    # 主控制器类
    DataVersionController,
    
    # 工具函数
    test_data_version_controller
)

# ==================== 版本信息 ====================
__version__ = "1.0.0"
__author__ = "T8开发团队"
__email__ = "t8-team@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 T8开发团队"
__description__ = "一个功能完整的数据版本管理系统"

# ==================== 默认配置 ====================

# 默认存储配置
DEFAULT_STORAGE_CONFIG = {
    "storage_path": "data_version_storage",
    "auto_backup": False,
    "backup_interval": 3600,  # 秒
    "max_versions_per_data": 1000,
    "cleanup_expired_versions": True,
    "enable_compression": False,
    "cache_size": 1000
}

# 默认权限配置
DEFAULT_PERMISSIONS = {
    "admin_user": {
        "data": AccessLevel.OWNER,
        "version": AccessLevel.ADMIN,
        "branch": AccessLevel.ADMIN,
        "permission": AccessLevel.ADMIN
    },
    "standard_user": {
        "data": AccessLevel.READ_WRITE,
        "version": AccessLevel.READ_ONLY,
        "branch": AccessLevel.READ_ONLY
    },
    "guest_user": {
        "data": AccessLevel.READ_ONLY,
        "version": AccessLevel.READ_ONLY,
        "branch": AccessLevel.READ_ONLY
    }
}

# 默认版本状态配置
DEFAULT_VERSION_STATUS = {
    "auto_promote": {
        VersionStatus.DRAFT: VersionStatus.ACTIVE,
        VersionStatus.ACTIVE: VersionStatus.STABLE
    },
    "retention_days": {
        VersionStatus.DRAFT: 7,
        VersionStatus.ACTIVE: 30,
        VersionStatus.STABLE: -1,  # 永久保留
        VersionStatus.DEPRECATED: 90,
        VersionStatus.ARCHIVED: -1
    }
}

# ==================== 常量定义 ====================

# 资源类型常量
RESOURCE_TYPES = {
    "DATA": "data",
    "VERSION": "version", 
    "BRANCH": "branch",
    "TAG": "tag",
    "PERMISSION": "permission"
}

# 变更类型映射
CHANGE_TYPE_LABELS = {
    ChangeType.CREATE: "创建",
    ChangeType.UPDATE: "更新", 
    ChangeType.DELETE: "删除",
    ChangeType.MERGE: "合并",
    ChangeType.REVERT: "回滚"
}

# 状态标签映射
STATUS_LABELS = {
    VersionStatus.DRAFT: "草稿",
    VersionStatus.ACTIVE: "活跃",
    VersionStatus.STABLE: "稳定",
    VersionStatus.DEPRECATED: "废弃",
    VersionStatus.ARCHIVED: "归档"
}

# 访问级别标签映射
ACCESS_LEVEL_LABELS = {
    AccessLevel.READ_ONLY: "只读",
    AccessLevel.READ_WRITE: "读写",
    AccessLevel.ADMIN: "管理员",
    AccessLevel.OWNER: "所有者"
}

# ==================== 便利函数 ====================

def create_controller(storage_path: str = None, **kwargs) -> DataVersionController:
    """
    创建数据版本控制器实例的便利函数
    
    Args:
        storage_path: 存储路径，如果为None则使用默认路径
        **kwargs: 其他传递给DataVersionController的参数
        
    Returns:
        DataVersionController: 控制器实例
    """
    if storage_path is None:
        storage_path = DEFAULT_STORAGE_CONFIG["storage_path"]
    
    return DataVersionController(storage_path=storage_path, **kwargs)


def quick_setup(storage_path: str = None, admin_user: str = "admin") -> DataVersionController:
    """
    快速设置数据版本控制器的便利函数
    
    自动创建控制器实例并授予管理员权限
    
    Args:
        storage_path: 存储路径
        admin_user: 管理员用户名
        
    Returns:
        DataVersionController: 配置好的控制器实例
    """
    controller = create_controller(storage_path)
    
    # 授予管理员完整权限
    for resource_type, level in DEFAULT_PERMISSIONS["admin_user"].items():
        controller.grant_permission(admin_user, resource_type, "*", level, "system")
    
    return controller


def create_sample_data(controller: DataVersionController, user: str = "admin") -> dict:
    """
    创建示例数据版本的便利函数
    
    Args:
        controller: 数据版本控制器实例
        user: 创建用户
        
    Returns:
        dict: 创建的版本ID字典
    """
    # 确保用户有权限
    controller.grant_permission(user, "data", "*", AccessLevel.READ_WRITE, "system")
    
    # 创建示例数据
    sample_data = [
        {
            "data_id": "user_profile_001",
            "data": {
                "name": "张三",
                "email": "zhangsan@example.com", 
                "role": "developer",
                "skills": ["Python", "JavaScript", "Docker"]
            },
            "description": "用户档案初始版本",
            "metadata": {"category": "user", "type": "profile"}
        },
        {
            "data_id": "config_app",
            "data": {
                "app_name": "MyApp",
                "version": "1.0.0",
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "name": "myapp_db"
                },
                "features": ["auth", "api", "admin"]
            },
            "description": "应用程序配置文件",
            "metadata": {"category": "config", "type": "application"}
        },
        {
            "data_id": "product_catalog",
            "data": [
                {"id": 1, "name": "智能手机", "price": 2999, "stock": 100},
                {"id": 2, "name": "笔记本电脑", "price": 5999, "stock": 50},
                {"id": 3, "name": "无线耳机", "price": 299, "stock": 200}
            ],
            "description": "产品目录数据",
            "metadata": {"category": "business", "type": "catalog"}
        }
    ]
    
    version_ids = {}
    
    for item in sample_data:
        version = controller.create_version(
            data_id=item["data_id"],
            data=item["data"],
            user=user,
            description=item["description"],
            metadata=item["metadata"]
        )
        version_ids[item["data_id"]] = version.version_id
        
        # 添加标签
        if item["data_id"] == "user_profile_001":
            controller.add_version_tag(version.version_id, "initial", user)
        elif item["data_id"] == "config_app":
            controller.add_version_tag(version.version_id, "v1.0.0", user)
        elif item["data_id"] == "product_catalog":
            controller.add_version_tag(version.version_id, "baseline", user)
    
    return version_ids


def export_statistics(controller: DataVersionController, format_type: str = "dict") -> any:
    """
    导出统计信息的便利函数
    
    Args:
        controller: 数据版本控制器实例
        format_type: 导出格式 ("dict", "json", "markdown")
        
    Returns:
        any: 格式化的统计数据
    """
    stats = controller.get_statistics()
    
    if format_type == "dict":
        return stats
    elif format_type == "json":
        import json
        return json.dumps(stats, indent=2, ensure_ascii=False)
    elif format_type == "markdown":
        md_report = f"""
# 数据版本控制统计报告

## 总体概览
- 总版本数: {stats['total_versions']}
- 总分支数: {stats['total_branches']}  
- 总用户数: {stats['total_users']}
- 总数据大小: {stats['total_size_mb']} MB

## 版本状态分布
"""
        for status, count in stats['status_distribution'].items():
            status_label = STATUS_LABELS.get(VersionStatus(status), status)
            md_report += f"- {status_label}: {count}\n"
        
        md_report += f"""
## 变更活动
- 过去30天变更次数: {stats['recent_activities_30d']}

## 变更类型分布
"""
        for change_type, count in stats['change_type_distribution'].items():
            change_label = CHANGE_TYPE_LABELS.get(ChangeType(change_type), change_type)
            md_report += f"- {change_label}: {count}\n"
        
        md_report += f"\n*报告生成时间: {stats['generated_at']}*\n"
        return md_report
    
    return stats


def cleanup_old_versions(controller: DataVersionController, 
                        days_threshold: int = 30,
                        dry_run: bool = True) -> dict:
    """
    清理旧版本的便利函数
    
    Args:
        controller: 数据版本控制器实例
        days_threshold: 超过多少天的版本将被清理
        dry_run: 是否只预览不实际执行
        
    Returns:
        dict: 清理结果摘要
    """
    from datetime import datetime, timedelta
    import sqlite3
    from pathlib import Path
    
    cutoff_date = datetime.now() - timedelta(days=days_threshold)
    
    with sqlite3.connect(controller.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT version_id, data_id, created_at, status FROM versions WHERE created_at < ?",
            (cutoff_date.isoformat(),)
        )
        old_versions = cursor.fetchall()
    
    cleanup_summary = {
        "cutoff_date": cutoff_date.isoformat(),
        "versions_found": len(old_versions),
        "total_size_bytes": 0,
        "by_status": {},
        "dry_run": dry_run
    }
    
    for version_id, data_id, created_at, status in old_versions:
        data_file = Path(controller.storage_path) / f"{version_id}.dat"
        if data_file.exists():
            cleanup_summary["total_size_bytes"] += data_file.stat().st_size
        
        if status not in cleanup_summary["by_status"]:
            cleanup_summary["by_status"][status] = 0
        cleanup_summary["by_status"][status] += 1
        
        if not dry_run:
            # 执行清理操作
            try:
                # 删除数据文件
                if data_file.exists():
                    data_file.unlink()
                
                # 从数据库删除记录
                with sqlite3.connect(controller.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM versions WHERE version_id = ?", (version_id,))
                    cursor.execute("DELETE FROM change_records WHERE version_id = ?", (version_id,))
                    conn.commit()
                
                # 从缓存中移除
                if version_id in controller._version_cache:
                    del controller._version_cache[version_id]
                    
            except Exception as e:
                cleanup_summary.setdefault("errors", []).append(f"删除版本 {version_id} 时出错: {e}")
    
    return cleanup_summary


def generate_migration_script(from_version: str, to_version: str,
                            controller: DataVersionController) -> str:
    """
    生成版本迁移脚本的便利函数
    
    Args:
        from_version: 源版本ID
        to_version: 目标版本ID  
        controller: 数据版本控制器实例
        
    Returns:
        str: 迁移脚本
    """
    from datetime import datetime
    import json
    
    version_from = controller.get_version(from_version)
    version_to = controller.get_version(to_version)
    
    if not version_from or not version_to:
        raise ValueError("源版本或目标版本不存在")
    
    comparison = controller.compare_versions(from_version, to_version)
    
    script = f"""# 数据迁移脚本
# 从版本 {from_version} 迁移到版本 {to_version}

import json

# 迁移配置
MIGRATION_CONFIG = {{
    "from_version": "{from_version}",
    "to_version": "{to_version}",
    "description": "{version_to.description}",
    "migration_date": "{datetime.now().isoformat()}"
}}

# 版本信息
FROM_VERSION = {json.dumps(version_from.to_dict(), indent=2, ensure_ascii=False)}
TO_VERSION = {json.dumps(version_to.to_dict(), indent=2, ensure_ascii=False)}

# 变更详情
CHANGES = {json.dumps(comparison["differences"], indent=2, ensure_ascii=False)}

def migrate_data(source_data):
    \"\"\"数据迁移函数\"\"\"
    # 在这里实现具体的数据迁移逻辑
    # 例如：
    # if "field_renamed" in CHANGES.get("metadata", {{}}):
    #     source_data["new_field"] = source_data.pop("old_field")
    
    return source_data

if __name__ == "__main__":
    # 迁移示例
    source_data = {{}}  # 从源版本获取数据
    migrated_data = migrate_data(source_data)
    print("数据迁移完成")
"""
    
    return script


# ==================== 异常类定义 ====================

class DataVersionError(Exception):
    """数据版本控制基础异常类"""
    pass


class PermissionError(DataVersionError):
    """权限相关异常"""
    pass


class VersionNotFoundError(DataVersionError):
    """版本不存在异常"""
    pass


class BranchNotFoundError(DataVersionError):
    """分支不存在异常"""
    pass


class DependencyCycleError(DataVersionError):
    """依赖循环异常"""
    pass


# ==================== 快速入门指南 ====================

QUICK_START_GUIDE = """
# 数据版本控制器快速入门指南

## 安装和导入

```python
from T.T8 import (
    DataVersionController, VersionStatus, AccessLevel,
    create_controller, quick_setup, create_sample_data
)
```

## 基本使用

### 1. 创建控制器
```python
# 方法1: 直接创建
controller = DataVersionController("my_versions")

# 方法2: 使用便利函数
controller = create_controller("my_versions")

# 方法3: 快速设置（自动配置管理员权限）
controller = quick_setup("my_versions", "admin")
```

### 2. 配置权限
```python
# 授予用户完整权限
controller.grant_permission("user1", "data", "*", AccessLevel.READ_WRITE, "admin")
controller.grant_permission("user1", "version", "*", AccessLevel.ADMIN, "admin")

# 授予只读权限
controller.grant_permission("guest", "data", "*", AccessLevel.READ_ONLY, "admin")
```

### 3. 创建和管理版本
```python
# 创建第一个版本
data1 = {"name": "产品A", "price": 299, "stock": 100}
version1 = controller.create_version(
    data_id="product_a",
    data=data1,
    user="admin",
    description="产品初始版本"
)

# 创建更新版本
data2 = {"name": "产品A", "price": 279, "stock": 120, "discount": "限时优惠"}
version2 = controller.create_version(
    data_id="product_a", 
    data=data2,
    user="admin",
    description="价格调整和库存更新"
)

# 添加标签
controller.add_version_tag(version1.version_id, "v1.0", "admin")
controller.add_version_tag(version2.version_id, "v1.1", "admin")
```

### 4. 版本比较和分析
```python
# 比较版本差异
comparison = controller.compare_versions(version1.version_id, version2.version_id)
print(f"哈希值不同: {comparison['differences']['hash']}")

# 获取变更历史
history = controller.get_change_history(data_id="product_a")
for record in history:
    print(f"{record.timestamp}: {record.description}")
```

### 5. 分支管理
```python
# 创建新分支
branch = controller.create_branch(
    branch_name="feature_discount",
    from_version=version1.version_id,
    user="admin",
    description="折扣功能分支"
)

# 合并分支
merge_version = controller.merge_branches(
    source_branch="feature_discount",
    target_branch="main", 
    user="admin",
    description="合并折扣功能"
)
```

### 6. 标签和发布
```python
# 发布版本
controller.release_version(
    version_id=version2.version_id,
    release_name="v1.1.0",
    user="admin", 
    description="产品A价格调整版本"
)

# 获取统计信息
stats = controller.get_statistics()
print(f"总版本数: {stats['total_versions']}")
print(f"总大小: {stats['total_size_mb']} MB")
```

### 7. 便利函数使用
```python
# 创建示例数据
version_ids = create_sample_data(controller, "admin")

# 导出统计信息
markdown_report = export_statistics(controller, "markdown")
print(markdown_report)

# 清理旧版本
cleanup_result = cleanup_old_versions(controller, days_threshold=30, dry_run=True)
print(f"发现 {cleanup_result['versions_found']} 个旧版本")
```

## 最佳实践

1. **权限管理**: 使用最小权限原则，仅授予必要的访问权限
2. **版本标签**: 为重要版本添加有意义的标签
3. **分支策略**: 为不同功能创建独立分支
4. **定期清理**: 定期清理旧版本和草稿
5. **备份重要数据**: 对重要版本进行额外备份
6. **监控使用**: 定期查看统计报告了解使用情况

## 常见问题

### Q: 如何恢复误删的版本？
A: 使用 `recover_deleted_version()` 方法恢复标记为删除的版本。

### Q: 如何处理大型数据集？
A: 考虑启用数据压缩和分批处理。

### Q: 如何实现自动版本管理？
A: 可以结合定时任务和状态转换规则实现。

## 更多帮助

- 查看完整文档
- 运行测试用例: `test_data_version_controller()`
- 生成示例报告: `controller.generate_report()`
"""

# ==================== 导出所有公共接口 ====================

__all__ = [
    # 主要类
    "DataVersionController",
    "VersionStatus", 
    "AccessLevel",
    "ChangeType",
    "DataVersion",
    "ChangeRecord", 
    "VersionBranch",
    "AccessPermission",
    
    # 便利函数
    "create_controller",
    "quick_setup",
    "create_sample_data", 
    "export_statistics",
    "cleanup_old_versions",
    "generate_migration_script",
    
    # 配置常量
    "DEFAULT_STORAGE_CONFIG",
    "DEFAULT_PERMISSIONS",
    "DEFAULT_VERSION_STATUS", 
    "RESOURCE_TYPES",
    "CHANGE_TYPE_LABELS",
    "STATUS_LABELS",
    "ACCESS_LEVEL_LABELS",
    
    # 异常类
    "DataVersionError",
    "PermissionError", 
    "VersionNotFoundError",
    "BranchNotFoundError",
    "DependencyCycleError",
    
    # 文档
    "QUICK_START_GUIDE",
    
    # 测试
    "test_data_version_controller"
]

# ==================== 模块初始化 ====================

def _init_module():
    """模块初始化函数"""
    # 设置日志级别
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 输出欢迎信息
    print(f"T8数据版本控制器 v{__version__} 已加载")
    print("快速开始: from T.T8 import quick_setup")

# 模块加载时自动初始化
_init_module()