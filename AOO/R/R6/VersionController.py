#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R6版本控制器
实现完整的版本控制功能，包括版本管理、分支管理、合并操作等
"""

import os
import json
import hashlib
import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import shutil


class PermissionLevel(Enum):
    """权限级别枚举"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class Version:
    """版本数据类"""
    version_id: str
    message: str
    timestamp: str
    author: str
    files: Dict[str, str]  # 文件路径 -> 文件内容哈希
    parent_version: Optional[str] = None
    branch: str = "main"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class Branch:
    """分支数据类"""
    name: str
    head_version: Optional[str]
    created_at: str
    created_by: str
    description: str = ""


@dataclass
class User:
    """用户数据类"""
    username: str
    permission_level: PermissionLevel
    created_at: str


class VersionController:
    """R6版本控制器主类"""
    
    def __init__(self, repository_path: str):
        """
        初始化版本控制器
        
        Args:
            repository_path: 仓库路径
        """
        self.repository_path = repository_path
        self.metadata_path = os.path.join(repository_path, ".r6_metadata")
        self.versions_path = os.path.join(self.metadata_path, "versions")
        self.branches_path = os.path.join(self.metadata_path, "branches")
        self.users_path = os.path.join(self.metadata_path, "users")
        self.config_path = os.path.join(self.metadata_path, "config.json")
        
        # 创建必要的目录
        self._ensure_directories()
        
        # 加载配置
        self.config = self._load_config()
        self.current_user = None
        
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.metadata_path,
            self.versions_path,
            self.branches_path,
            self.users_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 创建默认配置
            default_config = {
                "repository_name": "R6 Repository",
                "created_at": datetime.datetime.now().isoformat(),
                "current_branch": "main",
                "default_permission": PermissionLevel.WRITE.value
            }
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config: Dict):
        """保存配置文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件内容的哈希值"""
        if not os.path.exists(file_path):
            return ""
        
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _save_version_metadata(self, version: Version):
        """保存版本元数据"""
        version_file = os.path.join(self.versions_path, f"{version.version_id}.json")
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(version), f, indent=2, ensure_ascii=False)
    
    def _load_version_metadata(self, version_id: str) -> Optional[Version]:
        """加载版本元数据"""
        version_file = os.path.join(self.versions_path, f"{version_id}.json")
        if os.path.exists(version_file):
            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Version(**data)
        return None
    
    def _save_branch_metadata(self, branch: Branch):
        """保存分支元数据"""
        # 将分支名称中的特殊字符替换为安全字符
        safe_branch_name = branch.name.replace('/', '_').replace('\\', '_')
        branch_file = os.path.join(self.branches_path, f"{safe_branch_name}.json")
        with open(branch_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(branch), f, indent=2, ensure_ascii=False)
    
    def _load_branch_metadata(self, branch_name: str) -> Optional[Branch]:
        """加载分支元数据"""
        # 将分支名称中的特殊字符替换为安全字符
        safe_branch_name = branch_name.replace('/', '_').replace('\\', '_')
        branch_file = os.path.join(self.branches_path, f"{safe_branch_name}.json")
        if os.path.exists(branch_file):
            with open(branch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Branch(**data)
        return None
    
    def authenticate_user(self, username: str, permission_level: PermissionLevel = PermissionLevel.WRITE) -> bool:
        """
        用户认证
        
        Args:
            username: 用户名
            permission_level: 权限级别
            
        Returns:
            bool: 认证是否成功
        """
        user = User(
            username=username,
            permission_level=permission_level,
            created_at=datetime.datetime.now().isoformat()
        )
        
        user_file = os.path.join(self.users_path, f"{username}.json")
        user_dict = asdict(user)
        user_dict['permission_level'] = permission_level.value  # 转换为字符串
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(user_dict, f, indent=2, ensure_ascii=False)
        
        self.current_user = user
        return True
    
    def check_permission(self, required_level: PermissionLevel) -> bool:
        """
        检查权限
        
        Args:
            required_level: 所需权限级别
            
        Returns:
            bool: 是否有权限
        """
        if not self.current_user:
            return False
        
        level_order = [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.ADMIN]
        user_level_index = level_order.index(self.current_user.permission_level)
        required_level_index = level_order.index(required_level)
        
        return user_level_index >= required_level_index
    
    def create_version(self, message: str, files: List[str] = None) -> str:
        """
        创建新版本
        
        Args:
            message: 版本描述信息
            files: 要包含的文件列表
            
        Returns:
            str: 版本ID
        """
        if not self.check_permission(PermissionLevel.WRITE):
            raise PermissionError("没有创建版本的权限")
        
        if files is None:
            # 获取所有跟踪的文件
            files = self._get_tracked_files()
        
        # 计算文件哈希
        file_hashes = {}
        for file_path in files:
            full_path = os.path.join(self.repository_path, file_path)
            if os.path.exists(full_path):
                file_hashes[file_path] = self._get_file_hash(full_path)
        
        # 生成版本ID
        timestamp = datetime.datetime.now()
        version_id = hashlib.md5(f"{timestamp.isoformat()}{message}".encode()).hexdigest()[:12]
        
        # 获取当前分支
        current_branch = self.config["current_branch"]
        
        # 获取父版本
        current_branch_obj = self._load_branch_metadata(current_branch)
        parent_version = current_branch_obj.head_version if current_branch_obj else None
        
        # 创建版本
        version = Version(
            version_id=version_id,
            message=message,
            timestamp=timestamp.isoformat(),
            author=self.current_user.username if self.current_user else "unknown",
            files=file_hashes,
            parent_version=parent_version,
            branch=current_branch
        )
        
        # 保存版本
        self._save_version_metadata(version)
        
        # 更新分支头指针
        if current_branch_obj:
            current_branch_obj.head_version = version_id
            self._save_branch_metadata(current_branch_obj)
        else:
            # 创建主分支
            main_branch = Branch(
                name="main",
                head_version=version_id,
                created_at=timestamp.isoformat(),
                created_by=self.current_user.username if self.current_user else "unknown"
            )
            self._save_branch_metadata(main_branch)
        
        return version_id
    
    def _get_tracked_files(self) -> List[str]:
        """获取所有跟踪的文件"""
        tracked_files = []
        for root, dirs, files in os.walk(self.repository_path):
            # 跳过元数据目录
            dirs[:] = [d for d in dirs if d != ".r6_metadata"]
            
            for file in files:
                if not file.startswith('.'):
                    file_path = os.path.relpath(os.path.join(root, file), self.repository_path)
                    tracked_files.append(file_path)
        return tracked_files
    
    def get_version_history(self, branch: str = None, limit: int = None) -> List[Version]:
        """
        获取版本历史
        
        Args:
            branch: 分支名称，为None时获取所有分支
            limit: 限制返回数量
            
        Returns:
            List[Version]: 版本列表
        """
        if not self.check_permission(PermissionLevel.READ):
            raise PermissionError("没有查看版本历史的权限")
        
        versions = []
        
        # 加载所有版本
        for filename in os.listdir(self.versions_path):
            if filename.endswith('.json'):
                version = self._load_version_metadata(filename[:-5])
                if version:
                    if branch is None or version.branch == branch:
                        versions.append(version)
        
        # 按时间戳排序
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """
        比较两个版本
        
        Args:
            version1_id: 第一个版本ID
            version2_id: 第二个版本ID
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        if not self.check_permission(PermissionLevel.READ):
            raise PermissionError("没有版本比较的权限")
        
        version1 = self._load_version_metadata(version1_id)
        version2 = self._load_version_metadata(version2_id)
        
        if not version1 or not version2:
            raise ValueError("版本不存在")
        
        # 分析差异
        files1 = set(version1.files.keys())
        files2 = set(version2.files.keys())
        
        added_files = files2 - files1
        removed_files = files1 - files2
        modified_files = files1 & files2
        
        # 检查修改的文件
        changed_files = []
        for file_path in modified_files:
            if version1.files[file_path] != version2.files[file_path]:
                changed_files.append(file_path)
        
        return {
            "version1": {
                "id": version1_id,
                "message": version1.message,
                "timestamp": version1.timestamp,
                "author": version1.author
            },
            "version2": {
                "id": version2_id,
                "message": version2.message,
                "timestamp": version2.timestamp,
                "author": version2.author
            },
            "added_files": list(added_files),
            "removed_files": list(removed_files),
            "modified_files": changed_files,
            "summary": {
                "files_added": len(added_files),
                "files_removed": len(removed_files),
                "files_modified": len(changed_files)
            }
        }
    
    def create_branch(self, branch_name: str, from_version: str = None, description: str = "") -> bool:
        """
        创建新分支
        
        Args:
            branch_name: 分支名称
            from_version: 基于的版本ID，为None时基于当前分支头
            description: 分支描述
            
        Returns:
            bool: 是否创建成功
        """
        if not self.check_permission(PermissionLevel.WRITE):
            raise PermissionError("没有创建分支的权限")
        
        # 检查分支是否已存在
        if self._load_branch_metadata(branch_name):
            raise ValueError(f"分支 {branch_name} 已存在")
        
        # 获取起始版本
        if from_version:
            head_version = from_version
        else:
            current_branch = self.config["current_branch"]
            current_branch_obj = self._load_branch_metadata(current_branch)
            head_version = current_branch_obj.head_version if current_branch_obj else None
        
        # 创建分支
        branch = Branch(
            name=branch_name,
            head_version=head_version,
            created_at=datetime.datetime.now().isoformat(),
            created_by=self.current_user.username if self.current_user else "unknown",
            description=description
        )
        
        self._save_branch_metadata(branch)
        return True
    
    def switch_branch(self, branch_name: str) -> bool:
        """
        切换分支
        
        Args:
            branch_name: 要切换到的分支名称
            
        Returns:
            bool: 是否切换成功
        """
        if not self.check_permission(PermissionLevel.WRITE):
            raise PermissionError("没有切换分支的权限")
        
        branch = self._load_branch_metadata(branch_name)
        if not branch:
            raise ValueError(f"分支 {branch_name} 不存在")
        
        self.config["current_branch"] = branch_name
        self._save_config(self.config)
        return True
    
    def delete_branch(self, branch_name: str) -> bool:
        """
        删除分支
        
        Args:
            branch_name: 要删除的分支名称
            
        Returns:
            bool: 是否删除成功
        """
        if not self.check_permission(PermissionLevel.ADMIN):
            raise PermissionError("没有删除分支的管理员权限")
        
        if branch_name == "main":
            raise ValueError("不能删除主分支")
        
        # 将分支名称中的特殊字符替换为安全字符
        safe_branch_name = branch_name.replace('/', '_').replace('\\', '_')
        branch_file = os.path.join(self.branches_path, f"{safe_branch_name}.json")
        if os.path.exists(branch_file):
            os.remove(branch_file)
            return True
        return False
    
    def merge_branch(self, source_branch: str, target_branch: str = None, 
                    resolve_conflicts: bool = True) -> Tuple[bool, List[str]]:
        """
        合并分支
        
        Args:
            source_branch: 源分支名称
            target_branch: 目标分支名称，为None时使用当前分支
            resolve_conflicts: 是否自动解决冲突
            
        Returns:
            Tuple[bool, List[str]]: (是否成功, 冲突列表)
        """
        if not self.check_permission(PermissionLevel.WRITE):
            raise PermissionError("没有合并分支的权限")
        
        if target_branch is None:
            target_branch = self.config["current_branch"]
        
        source = self._load_branch_metadata(source_branch)
        target = self._load_branch_metadata(target_branch)
        
        if not source or not target:
            raise ValueError("分支不存在")
        
        # 获取两个分支的版本历史
        source_versions = self.get_version_history(source_branch)
        target_versions = self.get_version_history(target_branch)
        
        # 找到共同祖先
        common_ancestor = None
        for source_ver in source_versions:
            for target_ver in target_versions:
                if source_ver.parent_version == target_ver.parent_version:
                    common_ancestor = source_ver.parent_version
                    break
            if common_ancestor:
                break
        
        conflicts = []
        
        # 分析冲突
        if source.head_version and target.head_version:
            source_files = self._load_version_metadata(source.head_version).files
            target_files = self._load_version_metadata(target.head_version).files
            
            for file_path in source_files:
                if file_path in target_files:
                    if source_files[file_path] != target_files[file_path]:
                        conflicts.append(file_path)
        
        if conflicts and not resolve_conflicts:
            return False, conflicts
        
        # 创建合并版本
        merge_message = f"Merge branch '{source_branch}' into '{target_branch}'"
        if conflicts:
            merge_message += f" (with {len(conflicts)} conflicts resolved)"
        
        # 模拟合并（实际实现中需要更复杂的逻辑）
        merged_version_id = self.create_version(merge_message)
        
        return True, conflicts
    
    def add_tag(self, version_id: str, tag_name: str) -> bool:
        """
        添加标签
        
        Args:
            version_id: 版本ID
            tag_name: 标签名称
            
        Returns:
            bool: 是否添加成功
        """
        if not self.check_permission(PermissionLevel.WRITE):
            raise PermissionError("没有添加标签的权限")
        
        version = self._load_version_metadata(version_id)
        if not version:
            raise ValueError("版本不存在")
        
        if tag_name not in version.tags:
            version.tags.append(tag_name)
            self._save_version_metadata(version)
        
        return True
    
    def get_tags(self, version_id: str = None) -> List[str]:
        """
        获取标签列表
        
        Args:
            version_id: 版本ID，为None时获取所有标签
            
        Returns:
            List[str]: 标签列表
        """
        if not self.check_permission(PermissionLevel.READ):
            raise PermissionError("没有查看标签的权限")
        
        if version_id:
            version = self._load_version_metadata(version_id)
            return version.tags if version else []
        else:
            all_tags = set()
            for filename in os.listdir(self.versions_path):
                if filename.endswith('.json'):
                    version = self._load_version_metadata(filename[:-5])
                    if version:
                        all_tags.update(version.tags)
            return list(all_tags)
    
    def rollback_version(self, version_id: str) -> bool:
        """
        回滚到指定版本
        
        Args:
            version_id: 目标版本ID
            
        Returns:
            bool: 是否回滚成功
        """
        if not self.check_permission(PermissionLevel.ADMIN):
            raise PermissionError("没有版本回滚的管理员权限")
        
        version = self._load_version_metadata(version_id)
        if not version:
            raise ValueError("版本不存在")
        
        # 更新当前分支的头指针
        current_branch = self.config["current_branch"]
        branch = self._load_branch_metadata(current_branch)
        if branch:
            branch.head_version = version_id
            self._save_branch_metadata(branch)
        
        return True
    
    def get_repository_info(self) -> Dict[str, Any]:
        """
        获取仓库信息
        
        Returns:
            Dict[str, Any]: 仓库信息
        """
        if not self.check_permission(PermissionLevel.READ):
            raise PermissionError("没有查看仓库信息的权限")
        
        branches = []
        for filename in os.listdir(self.branches_path):
            if filename.endswith('.json'):
                branch = self._load_branch_metadata(filename[:-5])
                if branch:
                    branches.append({
                        "name": branch.name,
                        "head_version": branch.head_version,
                        "created_at": branch.created_at,
                        "created_by": branch.created_by,
                        "description": branch.description
                    })
        
        versions = self.get_version_history()
        
        return {
            "repository_name": self.config["repository_name"],
            "created_at": self.config["created_at"],
            "current_branch": self.config["current_branch"],
            "total_versions": len(versions),
            "total_branches": len(branches),
            "branches": branches,
            "recent_versions": [
                {
                    "id": v.version_id,
                    "message": v.message,
                    "timestamp": v.timestamp,
                    "author": v.author,
                    "branch": v.branch,
                    "tags": v.tags
                }
                for v in versions[:10]  # 最近10个版本
            ]
        }
    
    def export_version(self, version_id: str, export_path: str) -> bool:
        """
        导出指定版本
        
        Args:
            version_id: 版本ID
            export_path: 导出路径
            
        Returns:
            bool: 是否导出成功
        """
        if not self.check_permission(PermissionLevel.READ):
            raise PermissionError("没有导出版本的权限")
        
        version = self._load_version_metadata(version_id)
        if not version:
            raise ValueError("版本不存在")
        
        # 创建导出目录
        os.makedirs(export_path, exist_ok=True)
        
        # 导出版本信息
        version_info = {
            "version_id": version.version_id,
            "message": version.message,
            "timestamp": version.timestamp,
            "author": version.author,
            "branch": version.branch,
            "tags": version.tags,
            "files": version.files
        }
        
        with open(os.path.join(export_path, "version_info.json"), 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
        
        return True