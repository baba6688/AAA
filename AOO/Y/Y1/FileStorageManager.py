#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y1文件存储管理器
一个功能完整的文件存储管理系统，支持文件上传、下载、管理、存储等功能
"""

import os
import shutil
import hashlib
import json
import time
import glob
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FileType(Enum):
    """文件类型枚举"""
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    CODE = "code"
    OTHER = "other"


class PermissionLevel(Enum):
    """权限级别枚举"""
    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"


@dataclass
class FileInfo:
    """文件信息数据类"""
    name: str
    path: str
    size: int
    file_type: FileType
    extension: str
    created_time: str
    modified_time: str
    access_time: str
    checksum: str
    permission: PermissionLevel
    owner: str
    description: str = ""
    tags: List[str] = None
    backup_count: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class StorageStats:
    """存储统计信息"""
    total_files: int
    total_size: int
    file_types: Dict[str, int]
    storage_usage: Dict[str, int]
    recent_files: List[str]
    largest_files: List[Tuple[str, int]]
    oldest_files: List[Tuple[str, str]]


class FileStorageManager:
    """文件存储管理器主类"""
    
    def __init__(self, storage_root: str = "storage"):
        """
        初始化文件存储管理器
        
        Args:
            storage_root: 存储根目录路径
        """
        self.storage_root = Path(storage_root)
        self.metadata_file = self.storage_root / "metadata.json"
        self.backup_dir = self.storage_root / "backups"
        self.temp_dir = self.storage_root / "temp"
        
        # 确保目录存在
        self.storage_root.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # 加载元数据
        self.metadata = self._load_metadata()
        
        logger.info(f"文件存储管理器已初始化，存储根目录: {self.storage_root}")
    
    def _load_metadata(self) -> Dict:
        """加载元数据文件"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # 转换字符串回枚举类型
                for path, info in metadata.items():
                    if 'file_type' in info and isinstance(info['file_type'], str):
                        info['file_type'] = FileType(info['file_type'])
                    if 'permission' in info and isinstance(info['permission'], str):
                        info['permission'] = PermissionLevel(info['permission'])
                
                return metadata
            except Exception as e:
                logger.error(f"加载元数据失败: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """保存元数据文件"""
        try:
            # 转换枚举类型为字符串
            serializable_metadata = {}
            for path, info in self.metadata.items():
                serializable_info = info.copy()
                if 'file_type' in serializable_info and hasattr(serializable_info['file_type'], 'value'):
                    serializable_info['file_type'] = serializable_info['file_type'].value
                if 'permission' in serializable_info and hasattr(serializable_info['permission'], 'value'):
                    serializable_info['permission'] = serializable_info['permission'].value
                serializable_metadata[path] = serializable_info
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def _calculate_checksum(self, file_path: Union[str, Path]) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算校验和失败: {e}")
            return ""
    
    def _get_file_type(self, file_path: Union[str, Path]) -> FileType:
        """根据文件扩展名判断文件类型"""
        extension = Path(file_path).suffix.lower()
        
        # 图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'}
        if extension in image_extensions:
            return FileType.IMAGE
        
        # 视频文件
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
        if extension in video_extensions:
            return FileType.VIDEO
        
        # 音频文件
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'}
        if extension in audio_extensions:
            return FileType.AUDIO
        
        # 文档文件
        doc_extensions = {'.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'}
        if extension in doc_extensions:
            return FileType.DOCUMENT
        
        # 压缩文件
        archive_extensions = {'.zip', '.rar', '.7z', '.tar', '.gz'}
        if extension in archive_extensions:
            return FileType.ARCHIVE
        
        # 代码文件
        code_extensions = {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php'}
        if extension in code_extensions:
            return FileType.CODE
        
        return FileType.OTHER
    
    def _get_file_info(self, file_path: Union[str, Path]) -> Optional[FileInfo]:
        """获取文件信息"""
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return None
            
            stat = path_obj.stat()
            file_type = self._get_file_type(path_obj)
            checksum = self._calculate_checksum(path_obj)
            
            return FileInfo(
                name=path_obj.name,
                path=str(path_obj),
                size=stat.st_size,
                file_type=file_type,
                extension=path_obj.suffix.lower(),
                created_time=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                access_time=datetime.fromtimestamp(stat.st_atime).isoformat(),
                checksum=checksum,
                permission=PermissionLevel.PRIVATE,
                owner="system"
            )
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return None
    
    def upload_file(self, source_path: Union[str, Path], 
                   target_path: Optional[Union[str, Path]] = None,
                   overwrite: bool = False,
                   create_backup: bool = True) -> bool:
        """
        上传文件
        
        Args:
            source_path: 源文件路径
            target_path: 目标路径（可选，默认使用源文件名）
            overwrite: 是否覆盖已存在的文件
            create_backup: 是否创建备份
            
        Returns:
            bool: 上传是否成功
        """
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                logger.error(f"源文件不存在: {source_path}")
                return False
            
            if target_path is None:
                target_path = self.storage_root / source_path.name
            else:
                target_path = Path(target_path)
            
            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查文件是否已存在
            if target_path.exists() and not overwrite:
                logger.error(f"文件已存在: {target_path}")
                return False
            
            # 创建备份
            if target_path.exists() and create_backup:
                self._create_backup(target_path)
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            
            # 更新元数据
            file_info = self._get_file_info(target_path)
            if file_info:
                self.metadata[str(target_path)] = asdict(file_info)
                self._save_metadata()
            
            logger.info(f"文件上传成功: {source_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            return False
    
    def download_file(self, source_path: Union[str, Path],
                     target_path: Optional[Union[str, Path]] = None) -> bool:
        """
        下载文件
        
        Args:
            source_path: 源文件路径
            target_path: 目标路径（可选）
            
        Returns:
            bool: 下载是否成功
        """
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                logger.error(f"源文件不存在: {source_path}")
                return False
            
            if target_path is None:
                target_path = Path.cwd() / source_path.name
            else:
                target_path = Path(target_path)
            
            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            
            # 更新访问时间
            Path(source_path).touch()
            
            logger.info(f"文件下载成功: {source_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件下载失败: {e}")
            return False
    
    def delete_file(self, file_path: Union[str, Path], 
                   create_backup: bool = True) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径
            create_backup: 是否创建备份
            
        Returns:
            bool: 删除是否成功
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return False
            
            # 创建备份
            if create_backup:
                self._create_backup(file_path)
            
            # 从元数据中删除
            if str(file_path) in self.metadata:
                del self.metadata[str(file_path)]
                self._save_metadata()
            
            # 删除文件
            file_path.unlink()
            
            logger.info(f"文件删除成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    def move_file(self, source_path: Union[str, Path],
                 target_path: Union[str, Path]) -> bool:
        """
        移动文件
        
        Args:
            source_path: 源文件路径
            target_path: 目标文件路径
            
        Returns:
            bool: 移动是否成功
        """
        try:
            source_path = Path(source_path)
            target_path = Path(target_path)
            
            if not source_path.exists():
                logger.error(f"源文件不存在: {source_path}")
                return False
            
            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 移动文件
            shutil.move(str(source_path), str(target_path))
            
            # 更新元数据
            if str(source_path) in self.metadata:
                file_info_dict = self.metadata[str(source_path)]
                del self.metadata[str(source_path)]
                
                # 更新文件信息
                file_info_dict['path'] = str(target_path)
                file_info_dict['name'] = target_path.name
                self.metadata[str(target_path)] = file_info_dict
                self._save_metadata()
            
            logger.info(f"文件移动成功: {source_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件移动失败: {e}")
            return False
    
    def rename_file(self, file_path: Union[str, Path], new_name: str) -> bool:
        """
        重命名文件
        
        Args:
            file_path: 文件路径
            new_name: 新文件名
            
        Returns:
            bool: 重命名是否成功
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return False
            
            new_path = file_path.parent / new_name
            return self.move_file(file_path, new_path)
            
        except Exception as e:
            logger.error(f"文件重命名失败: {e}")
            return False
    
    def create_file(self, file_path: Union[str, Path], content: str = "") -> bool:
        """
        创建文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
            
        Returns:
            bool: 创建是否成功
        """
        try:
            file_path = Path(file_path)
            
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 更新元数据
            file_info = self._get_file_info(file_path)
            if file_info:
                file_dict = asdict(file_info)
                # 确保枚举类型正确
                file_dict['file_type'] = file_info.file_type
                file_dict['permission'] = file_info.permission
                self.metadata[str(file_path)] = file_dict
                self._save_metadata()
            
            logger.info(f"文件创建成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件创建失败: {e}")
            return False
    
    def search_files(self, pattern: str = "", 
                    file_type: Optional[FileType] = None,
                    date_from: Optional[datetime] = None,
                    date_to: Optional[datetime] = None,
                    size_min: Optional[int] = None,
                    size_max: Optional[int] = None,
                    tags: Optional[List[str]] = None) -> List[FileInfo]:
        """
        搜索文件
        
        Args:
            pattern: 文件名模式（支持通配符）
            file_type: 文件类型过滤
            date_from: 创建时间起始
            date_to: 创建时间结束
            size_min: 最小文件大小（字节）
            size_max: 最大文件大小（字节）
            tags: 标签过滤
            
        Returns:
            List[FileInfo]: 匹配的文件信息列表
        """
        try:
            results = []
            
            for file_path_str, file_info_dict in self.metadata.items():
                file_path = Path(file_path_str)
                
                # 模式匹配
                if pattern and not file_path.match(pattern):
                    continue
                
                # 文件类型过滤
                if file_type and FileType(file_info_dict['file_type']) != file_type:
                    continue
                
                # 时间过滤
                if date_from or date_to:
                    created_time = datetime.fromisoformat(file_info_dict['created_time'])
                    if date_from and created_time < date_from:
                        continue
                    if date_to and created_time > date_to:
                        continue
                
                # 大小过滤
                if size_min and file_info_dict['size'] < size_min:
                    continue
                if size_max and file_info_dict['size'] > size_max:
                    continue
                
                # 标签过滤
                if tags:
                    file_tags = file_info_dict.get('tags', [])
                    if not any(tag in file_tags for tag in tags):
                        continue
                
                # 转换为FileInfo对象
                file_info = FileInfo(**file_info_dict)
                results.append(file_info)
            
            return results
            
        except Exception as e:
            logger.error(f"文件搜索失败: {e}")
            return []
    
    def get_storage_stats(self) -> StorageStats:
        """
        获取存储统计信息
        
        Returns:
            StorageStats: 存储统计信息
        """
        try:
            total_files = len(self.metadata)
            total_size = 0
            file_types = {}
            storage_usage = {}
            recent_files = []
            largest_files = []
            oldest_files = []
            
            file_items = []
            
            for file_path_str, file_info_dict in self.metadata.items():
                file_info = FileInfo(**file_info_dict)
                file_items.append((file_path_str, file_info))
                
                total_size += file_info.size
                
                # 统计文件类型
                file_type = file_info.file_type.value
                file_types[file_type] = file_types.get(file_type, 0) + 1
                
                # 存储使用情况
                parent_dir = Path(file_path_str).parent.name
                storage_usage[parent_dir] = storage_usage.get(parent_dir, 0) + file_info.size
            
            # 最近文件（按修改时间排序）
            file_items.sort(key=lambda x: x[1].modified_time, reverse=True)
            recent_files = [item[0] for item in file_items[:10]]
            
            # 最大文件
            file_items.sort(key=lambda x: x[1].size, reverse=True)
            largest_files = [(item[0], item[1].size) for item in file_items[:10]]
            
            # 最旧文件
            file_items.sort(key=lambda x: x[1].created_time)
            oldest_files = [(item[0], item[1].created_time) for item in file_items[:10]]
            
            return StorageStats(
                total_files=total_files,
                total_size=total_size,
                file_types=file_types,
                storage_usage=storage_usage,
                recent_files=recent_files,
                largest_files=largest_files,
                oldest_files=oldest_files
            )
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return StorageStats(0, 0, {}, {}, [], [], [])
    
    def create_backup(self, file_path: Union[str, Path], 
                     backup_name: Optional[str] = None) -> bool:
        """
        创建文件备份
        
        Args:
            file_path: 要备份的文件路径
            backup_name: 备份文件名（可选）
            
        Returns:
            bool: 备份是否成功
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return False
            
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            backup_path = self.backup_dir / backup_name
            
            # 复制文件到备份目录
            shutil.copy2(file_path, backup_path)
            
            # 更新元数据中的备份计数
            if str(file_path) in self.metadata:
                self.metadata[str(file_path)]['backup_count'] += 1
                self._save_metadata()
            
            logger.info(f"备份创建成功: {file_path} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return False
    
    def restore_backup(self, backup_path: Union[str, Path],
                      target_path: Optional[Union[str, Path]] = None) -> bool:
        """
        恢复文件备份
        
        Args:
            backup_path: 备份文件路径
            target_path: 目标路径（可选）
            
        Returns:
            bool: 恢复是否成功
        """
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"备份文件不存在: {backup_path}")
                return False
            
            if target_path is None:
                # 从备份文件名推断原始文件名
                target_path = self.storage_root / backup_path.name
            else:
                target_path = Path(target_path)
            
            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制备份文件
            shutil.copy2(backup_path, target_path)
            
            # 更新元数据
            file_info = self._get_file_info(target_path)
            if file_info:
                file_dict = asdict(file_info)
                # 确保枚举类型正确
                file_dict['file_type'] = file_info.file_type
                file_dict['permission'] = file_info.permission
                self.metadata[str(target_path)] = file_dict
                self._save_metadata()
            
            logger.info(f"备份恢复成功: {backup_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"恢复备份失败: {e}")
            return False
    
    def list_backups(self, file_path: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        列出备份文件
        
        Args:
            file_path: 文件路径（可选，列出特定文件的备份）
            
        Returns:
            List[Path]: 备份文件路径列表
        """
        try:
            if file_path is None:
                # 列出所有备份
                return list(self.backup_dir.glob("*"))
            else:
                # 列出特定文件的备份
                file_path = Path(file_path)
                pattern = f"{file_path.stem}_*{file_path.suffix}"
                return list(self.backup_dir.glob(pattern))
                
        except Exception as e:
            logger.error(f"列出备份失败: {e}")
            return []
    
    def _create_backup(self, file_path: Union[str, Path]) -> bool:
        """内部方法：创建文件备份"""
        return self.create_backup(file_path)
    
    def set_file_permission(self, file_path: Union[str, Path],
                           permission: PermissionLevel) -> bool:
        """
        设置文件权限
        
        Args:
            file_path: 文件路径
            permission: 权限级别
            
        Returns:
            bool: 设置是否成功
        """
        try:
            file_path = Path(file_path)
            if str(file_path) not in self.metadata:
                logger.error(f"文件不在存储管理器中: {file_path}")
                return False
            
            # 更新元数据
            self.metadata[str(file_path)]['permission'] = permission.value
            self._save_metadata()
            
            # 设置文件系统权限
            if permission == PermissionLevel.PUBLIC:
                os.chmod(file_path, 0o644)
            elif permission == PermissionLevel.PROTECTED:
                os.chmod(file_path, 0o664)
            else:  # PRIVATE
                os.chmod(file_path, 0o600)
            
            logger.info(f"文件权限设置成功: {file_path} -> {permission.value}")
            return True
            
        except Exception as e:
            logger.error(f"设置文件权限失败: {e}")
            return False
    
    def get_file_permission(self, file_path: Union[str, Path]) -> Optional[PermissionLevel]:
        """
        获取文件权限
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[PermissionLevel]: 权限级别
        """
        try:
            file_path = Path(file_path)
            if str(file_path) not in self.metadata:
                return None
            
            permission_str = self.metadata[str(file_path)].get('permission', 'private')
            return PermissionLevel(permission_str)
            
        except Exception as e:
            logger.error(f"获取文件权限失败: {e}")
            return None
    
    def add_file_tags(self, file_path: Union[str, Path], tags: List[str]) -> bool:
        """
        添加文件标签
        
        Args:
            file_path: 文件路径
            tags: 标签列表
            
        Returns:
            bool: 添加是否成功
        """
        try:
            file_path = Path(file_path)
            if str(file_path) not in self.metadata:
                logger.error(f"文件不在存储管理器中: {file_path}")
                return False
            
            # 更新标签
            current_tags = self.metadata[str(file_path)].get('tags', [])
            for tag in tags:
                if tag not in current_tags:
                    current_tags.append(tag)
            
            self.metadata[str(file_path)]['tags'] = current_tags
            self._save_metadata()
            
            logger.info(f"文件标签添加成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"添加文件标签失败: {e}")
            return False
    
    def remove_file_tags(self, file_path: Union[str, Path], tags: List[str]) -> bool:
        """
        移除文件标签
        
        Args:
            file_path: 文件路径
            tags: 标签列表
            
        Returns:
            bool: 移除是否成功
        """
        try:
            file_path = Path(file_path)
            if str(file_path) not in self.metadata:
                logger.error(f"文件不在存储管理器中: {file_path}")
                return False
            
            # 移除标签
            current_tags = self.metadata[str(file_path)].get('tags', [])
            for tag in tags:
                if tag in current_tags:
                    current_tags.remove(tag)
            
            self.metadata[str(file_path)]['tags'] = current_tags
            self._save_metadata()
            
            logger.info(f"文件标签移除成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"移除文件标签失败: {e}")
            return False
    
    def verify_file_integrity(self, file_path: Union[str, Path]) -> bool:
        """
        验证文件完整性
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 验证是否通过
        """
        try:
            file_path = Path(file_path)
            if str(file_path) not in self.metadata:
                logger.error(f"文件不在存储管理器中: {file_path}")
                return False
            
            # 计算当前校验和
            current_checksum = self._calculate_checksum(file_path)
            stored_checksum = self.metadata[str(file_path)].get('checksum', '')
            
            if current_checksum == stored_checksum:
                logger.info(f"文件完整性验证通过: {file_path}")
                return True
            else:
                logger.warning(f"文件完整性验证失败: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"验证文件完整性失败: {e}")
            return False
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        清理临时文件
        
        Args:
            max_age_hours: 最大保留时间（小时）
            
        Returns:
            int: 清理的文件数量
        """
        try:
            cleaned_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.is_file():
                    file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        temp_file.unlink()
                        cleaned_count += 1
            
            logger.info(f"清理临时文件完成，清理数量: {cleaned_count}")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")
            return 0
    
    def export_metadata(self, export_path: Union[str, Path]) -> bool:
        """
        导出元数据
        
        Args:
            export_path: 导出路径
            
        Returns:
            bool: 导出是否成功
        """
        try:
            export_path = Path(export_path)
            
            # 确保目录存在
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换枚举类型为字符串
            serializable_metadata = {}
            for path, info in self.metadata.items():
                serializable_info = info.copy()
                if 'file_type' in serializable_info and hasattr(serializable_info['file_type'], 'value'):
                    serializable_info['file_type'] = serializable_info['file_type'].value
                if 'permission' in serializable_info and hasattr(serializable_info['permission'], 'value'):
                    serializable_info['permission'] = serializable_info['permission'].value
                serializable_metadata[path] = serializable_info
            
            # 导出元数据
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"元数据导出成功: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出元数据失败: {e}")
            return False
    
    def import_metadata(self, import_path: Union[str, Path]) -> bool:
        """
        导入元数据
        
        Args:
            import_path: 导入路径
            
        Returns:
            bool: 导入是否成功
        """
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                logger.error(f"导入文件不存在: {import_path}")
                return False
            
            # 读取元数据
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_metadata = json.load(f)
            
            # 合并元数据
            self.metadata.update(imported_metadata)
            self._save_metadata()
            
            logger.info(f"元数据导入成功: {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"导入元数据失败: {e}")
            return False
    
    def list_files(self, directory: Optional[Union[str, Path]] = None,
                  recursive: bool = False) -> List[FileInfo]:
        """
        列出文件
        
        Args:
            directory: 目录路径（可选）
            recursive: 是否递归列出子目录
            
        Returns:
            List[FileInfo]: 文件信息列表
        """
        try:
            if directory is None:
                directory = self.storage_root
            else:
                directory = Path(directory)
            
            results = []
            
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_info = self._get_file_info(file_path)
                    if file_info:
                        results.append(file_info)
            
            return results
            
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    def get_file_info(self, file_path: Union[str, Path]) -> Optional[FileInfo]:
        """
        获取文件详细信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[FileInfo]: 文件信息
        """
        file_path = Path(file_path)
        
        # 首先从元数据中查找
        if str(file_path) in self.metadata:
            file_info_dict = self.metadata[str(file_path)]
            try:
                # 转换为FileInfo对象
                file_info = FileInfo(**file_info_dict)
                return file_info
            except Exception as e:
                logger.error(f"从元数据创建FileInfo失败: {e}")
        
        # 如果元数据中没有，则重新获取
        return self._get_file_info(file_path)
    
    def compress_directory(self, source_dir: Union[str, Path],
                          archive_path: Union[str, Path],
                          format: str = "zip") -> bool:
        """
        压缩目录
        
        Args:
            source_dir: 源目录
            archive_path: 压缩文件路径
            format: 压缩格式（zip, tar, gz）
            
        Returns:
            bool: 压缩是否成功
        """
        try:
            source_dir = Path(source_dir)
            archive_path = Path(archive_path)
            
            if not source_dir.exists() or not source_dir.is_dir():
                logger.error(f"源目录不存在: {source_dir}")
                return False
            
            # 确保目标目录存在
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "zip":
                shutil.make_archive(str(archive_path.with_suffix('')), 'zip', source_dir)
            elif format.lower() == "tar":
                shutil.make_archive(str(archive_path.with_suffix('')), 'tar', source_dir)
            elif format.lower() == "gz":
                shutil.make_archive(str(archive_path.with_suffix('')), 'gztar', source_dir)
            else:
                logger.error(f"不支持的压缩格式: {format}")
                return False
            
            logger.info(f"目录压缩成功: {source_dir} -> {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"压缩目录失败: {e}")
            return False
    
    def extract_archive(self, archive_path: Union[str, Path],
                       extract_dir: Union[str, Path]) -> bool:
        """
        解压缩文件
        
        Args:
            archive_path: 压缩文件路径
            extract_dir: 提取目录
            
        Returns:
            bool: 解压缩是否成功
        """
        try:
            archive_path = Path(archive_path)
            extract_dir = Path(extract_dir)
            
            if not archive_path.exists():
                logger.error(f"压缩文件不存在: {archive_path}")
                return False
            
            # 确保提取目录存在
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            # 解压缩
            shutil.unpack_archive(str(archive_path), str(extract_dir))
            
            logger.info(f"解压缩成功: {archive_path} -> {extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"解压缩失败: {e}")
            return False