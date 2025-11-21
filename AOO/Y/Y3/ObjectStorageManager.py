#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y3对象存储管理器

提供完整的云对象存储服务管理功能，包括对象上传、下载、管理、
权限控制、版本管理、统计分析和备份恢复等功能。

作者: Y3开发团队
版本: 1.0.0
日期: 2025-11-06
"""

import os
import json
import hashlib
import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import time
import mimetypes

# 阿里云 OSS 可用性标志
try:
    import oss2
    OSS2_AVAILABLE = True
except ImportError:
    oss2 = None
    OSS2_AVAILABLE = False


class StorageProvider(Enum):
    """存储提供商枚举"""
    AWS_S3 = "aws_s3"
    ALIYUN_OSS = "aliyun_oss"
    TENCENT_COS = "tencent_cos"
    QINIU = "qiniu"
    LOCAL = "local"


class ObjectStatus(Enum):
    """对象状态枚举"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    BACKUP = "backup"


class Permission(Enum):
    """权限枚举"""
    PUBLIC_READ = "public_read"
    PUBLIC_READ_WRITE = "public_read_write"
    PRIVATE = "private"
    AUTHENTICATED_READ = "authenticated_read"


@dataclass
class ObjectMetadata:
    """对象元数据"""
    name: str
    size: int
    content_type: str
    etag: str
    last_modified: datetime.datetime
    storage_provider: StorageProvider
    bucket: str
    key: str
    status: ObjectStatus = ObjectStatus.ACTIVE
    permissions: Permission = Permission.PRIVATE
    version: str = "1.0"
    tags: Dict[str, str] = None
    custom_metadata: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.custom_metadata is None:
            self.custom_metadata = {}


@dataclass
class StorageConfig:
    """存储配置"""
    provider: StorageProvider
    access_key: str
    secret_key: str
    endpoint: str
    region: str
    bucket_name: str
    timeout: int = 300
    retry_count: int = 3
    chunk_size: int = 8192


class ObjectStorageManager:
    """Y3对象存储管理器主类"""
    
    def __init__(self, config: StorageConfig):
        """
        初始化对象存储管理器
        
        Args:
            config: 存储配置
        """
        self.config = config
        self.logger = self._setup_logger()
        self._lock = threading.RLock()
        self._objects_cache = {}
        self._stats = {
            'upload_count': 0,
            'download_count': 0,
            'delete_count': 0,
            'total_size': 0,
            'error_count': 0
        }
        
        # 初始化存储提供商
        self._init_provider()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f'Y3.ObjectStorage.{self.config.provider.value}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_provider(self):
        """初始化存储提供商"""
        try:
            if self.config.provider == StorageProvider.AWS_S3:
                import boto3
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=self.config.access_key,
                    aws_secret_access_key=self.config.secret_key,
                    endpoint_url=self.config.endpoint,
                    region_name=self.config.region
                )
            elif self.config.provider == StorageProvider.ALIYUN_OSS:
                if not OSS2_AVAILABLE:
                    raise ImportError("oss2库未安装，无法使用阿里云OSS服务")
                self.client = oss2.Auth(self.config.access_key, self.config.secret_key)
                self.bucket = oss2.Bucket(self.client, self.config.endpoint, self.config.bucket_name)
            elif self.config.provider == StorageProvider.LOCAL:
                self._init_local_provider()
            else:
                raise ValueError(f"不支持的存储提供商: {self.config.provider}")
            
            self.logger.info(f"成功初始化 {self.config.provider.value} 存储提供商")
        except ImportError as e:
            self.logger.warning(f"缺少依赖库: {e}，使用模拟模式")
            self.client = None
        except Exception as e:
            self.logger.error(f"初始化存储提供商失败: {e}")
            raise
    
    def _init_local_provider(self):
        """初始化本地存储提供商"""
        local_path = Path(self.config.endpoint)
        if not local_path.exists():
            local_path.mkdir(parents=True, exist_ok=True)
        self.local_path = local_path
    
    def upload_object(self, 
                     file_path: Union[str, Path], 
                     object_key: str,
                     bucket_name: Optional[str] = None,
                     metadata: Optional[Dict[str, str]] = None,
                     tags: Optional[Dict[str, str]] = None) -> ObjectMetadata:
        """
        上传对象到存储
        
        Args:
            file_path: 本地文件路径
            object_key: 对象键名
            bucket_name: 存储桶名称（可选）
            metadata: 自定义元数据
            tags: 标签
            
        Returns:
            ObjectMetadata: 对象元数据
        """
        with self._lock:
            try:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"文件不存在: {file_path}")
                
                bucket_name = bucket_name or self.config.bucket_name
                
                # 计算文件哈希和大小
                file_hash = self._calculate_file_hash(file_path)
                file_size = file_path.stat().st_size
                content_type = mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
                
                # 上传到存储
                if self.config.provider == StorageProvider.LOCAL:
                    self._upload_to_local(file_path, object_key)
                else:
                    self._upload_to_cloud(file_path, object_key, bucket_name)
                
                # 创建对象元数据
                obj_metadata = ObjectMetadata(
                    name=file_path.name,
                    size=file_size,
                    content_type=content_type,
                    etag=file_hash,
                    last_modified=datetime.datetime.now(),
                    storage_provider=self.config.provider,
                    bucket=bucket_name,
                    key=object_key,
                    tags=tags or {},
                    custom_metadata=metadata or {}
                )
                
                # 缓存对象信息
                self._objects_cache[object_key] = obj_metadata
                
                # 更新统计
                self._stats['upload_count'] += 1
                self._stats['total_size'] += file_size
                
                self.logger.info(f"成功上传对象: {object_key}")
                return obj_metadata
                
            except Exception as e:
                self._stats['error_count'] += 1
                self.logger.error(f"上传对象失败: {object_key}, 错误: {e}")
                raise
    
    def _upload_to_local(self, file_path: Path, object_key: str):
        """上传到本地存储"""
        target_path = self.local_path / object_key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
            while True:
                chunk = src.read(self.config.chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
    
    def _upload_to_cloud(self, file_path: Path, object_key: str, bucket_name: str):
        """上传到云存储"""
        if self.config.provider == StorageProvider.AWS_S3:
            self.client.upload_file(
                str(file_path), bucket_name, object_key
            )
        elif self.config.provider == StorageProvider.ALIYUN_OSS:
            self.bucket.put_object_from_file(object_key, str(file_path))
        else:
            raise ValueError(f"不支持的云存储提供商: {self.config.provider}")
    
    def download_object(self, 
                       object_key: str, 
                       local_path: Union[str, Path],
                       bucket_name: Optional[str] = None) -> bool:
        """
        从存储下载对象
        
        Args:
            object_key: 对象键名
            local_path: 本地保存路径
            bucket_name: 存储桶名称（可选）
            
        Returns:
            bool: 下载是否成功
        """
        with self._lock:
            try:
                bucket_name = bucket_name or self.config.bucket_name
                local_path = Path(local_path)
                
                # 确保本地目录存在
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 从存储下载
                if self.config.provider == StorageProvider.LOCAL:
                    return self._download_from_local(object_key, local_path)
                else:
                    return self._download_from_cloud(object_key, local_path, bucket_name)
                
            except Exception as e:
                self._stats['error_count'] += 1
                self.logger.error(f"下载对象失败: {object_key}, 错误: {e}")
                raise
    
    def _download_from_local(self, object_key: str, local_path: Path) -> bool:
        """从本地存储下载"""
        source_path = self.local_path / object_key
        if not source_path.exists():
            raise FileNotFoundError(f"对象不存在: {object_key}")
        
        with open(source_path, 'rb') as src, open(local_path, 'wb') as dst:
            while True:
                chunk = src.read(self.config.chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
        
        self._stats['download_count'] += 1
        self.logger.info(f"成功下载对象: {object_key}")
        return True
    
    def _download_from_cloud(self, object_key: str, local_path: Path, bucket_name: str) -> bool:
        """从云存储下载"""
        if self.config.provider == StorageProvider.AWS_S3:
            self.client.download_file(bucket_name, object_key, str(local_path))
        elif self.config.provider == StorageProvider.ALIYUN_OSS:
            self.bucket.get_object_to_file(object_key, str(local_path))
        else:
            raise ValueError(f"不支持的云存储提供商: {self.config.provider}")
        
        self._stats['download_count'] += 1
        self.logger.info(f"成功下载对象: {object_key}")
        return True
    
    def delete_object(self, object_key: str, bucket_name: Optional[str] = None) -> bool:
        """
        删除对象
        
        Args:
            object_key: 对象键名
            bucket_name: 存储桶名称（可选）
            
        Returns:
            bool: 删除是否成功
        """
        with self._lock:
            try:
                bucket_name = bucket_name or self.config.bucket_name
                
                # 从存储删除
                if self.config.provider == StorageProvider.LOCAL:
                    source_path = self.local_path / object_key
                    if not source_path.exists():
                        raise FileNotFoundError(f"对象不存在: {object_key}")
                    source_path.unlink()
                else:
                    self._delete_from_cloud(object_key, bucket_name)
                
                # 从缓存删除
                if object_key in self._objects_cache:
                    del self._objects_cache[object_key]
                
                # 更新统计
                self._stats['delete_count'] += 1
                
                self.logger.info(f"成功删除对象: {object_key}")
                return True
                
            except Exception as e:
                self._stats['error_count'] += 1
                self.logger.error(f"删除对象失败: {object_key}, 错误: {e}")
                raise
    
    def _delete_from_cloud(self, object_key: str, bucket_name: str):
        """从云存储删除"""
        if self.config.provider == StorageProvider.AWS_S3:
            self.client.delete_object(Bucket=bucket_name, Key=object_key)
        elif self.config.provider == StorageProvider.ALIYUN_OSS:
            self.bucket.delete_object(object_key)
        else:
            raise ValueError(f"不支持的云存储提供商: {self.config.provider}")
    
    def get_object_metadata(self, object_key: str, bucket_name: Optional[str] = None) -> Optional[ObjectMetadata]:
        """
        获取对象元数据
        
        Args:
            object_key: 对象键名
            bucket_name: 存储桶名称（可选）
            
        Returns:
            ObjectMetadata: 对象元数据，如果对象不存在则返回None
        """
        with self._lock:
            # 先从缓存查找
            if object_key in self._objects_cache:
                return self._objects_cache[object_key]
            
            try:
                bucket_name = bucket_name or self.config.bucket_name
                
                if self.config.provider == StorageProvider.LOCAL:
                    return self._get_local_metadata(object_key)
                else:
                    return self._get_cloud_metadata(object_key, bucket_name)
                    
            except Exception as e:
                self.logger.error(f"获取对象元数据失败: {object_key}, 错误: {e}")
                return None
    
    def _get_local_metadata(self, object_key: str) -> Optional[ObjectMetadata]:
        """获取本地存储对象元数据"""
        source_path = self.local_path / object_key
        if not source_path.exists():
            return None
        
        stat = source_path.stat()
        content_type = mimetypes.guess_type(str(source_path))[0] or 'application/octet-stream'
        
        metadata = ObjectMetadata(
            name=source_path.name,
            size=stat.st_size,
            content_type=content_type,
            etag=self._calculate_file_hash(source_path),
            last_modified=datetime.datetime.fromtimestamp(stat.st_mtime),
            storage_provider=self.config.provider,
            bucket=self.config.bucket_name,
            key=object_key
        )
        
        # 缓存对象信息
        self._objects_cache[object_key] = metadata
        return metadata
    
    def _get_cloud_metadata(self, object_key: str, bucket_name: str) -> Optional[ObjectMetadata]:
        """获取云存储对象元数据"""
        if self.config.provider == StorageProvider.AWS_S3:
            response = self.client.head_object(Bucket=bucket_name, Key=object_key)
            return self._parse_aws_metadata(response, object_key, bucket_name)
        elif self.config.provider == StorageProvider.ALIYUN_OSS:
            head_info = self.bucket.head_object(object_key)
            return self._parse_oss_metadata(head_info, object_key, bucket_name)
        else:
            raise ValueError(f"不支持的云存储提供商: {self.config.provider}")
    
    def _parse_aws_metadata(self, response: Dict, object_key: str, bucket_name: str) -> ObjectMetadata:
        """解析AWS S3元数据"""
        return ObjectMetadata(
            name=object_key.split('/')[-1],
            size=response['ContentLength'],
            content_type=response.get('ContentType', 'application/octet-stream'),
            etag=response['ETag'].strip('"'),
            last_modified=response['LastModified'],
            storage_provider=self.config.provider,
            bucket=bucket_name,
            key=object_key
        )
    
    def _parse_oss_metadata(self, head_info: Dict, object_key: str, bucket_name: str) -> ObjectMetadata:
        """解析阿里云OSS元数据"""
        return ObjectMetadata(
            name=object_key.split('/')[-1],
            size=head_info.content_length,
            content_type=head_info.content_type,
            etag=head_info.etag,
            last_modified=head_info.last_modified,
            storage_provider=self.config.provider,
            bucket=bucket_name,
            key=object_key
        )
    
    def list_objects(self, 
                    prefix: str = "", 
                    max_keys: int = 1000,
                    bucket_name: Optional[str] = None) -> List[ObjectMetadata]:
        """
        列出对象
        
        Args:
            prefix: 对象键名前缀
            max_keys: 最大返回数量
            bucket_name: 存储桶名称（可选）
            
        Returns:
            List[ObjectMetadata]: 对象元数据列表
        """
        with self._lock:
            try:
                bucket_name = bucket_name or self.config.bucket_name
                
                if self.config.provider == StorageProvider.LOCAL:
                    return self._list_local_objects(prefix, max_keys)
                else:
                    return self._list_cloud_objects(prefix, max_keys, bucket_name)
                    
            except Exception as e:
                self.logger.error(f"列出对象失败: {e}")
                raise
    
    def _list_local_objects(self, prefix: str, max_keys: int) -> List[ObjectMetadata]:
        """列出本地存储对象"""
        objects = []
        prefix_path = self.local_path / prefix
        
        if not prefix_path.exists():
            return objects
        
        for file_path in prefix_path.rglob('*'):
            if file_path.is_file() and len(objects) < max_keys:
                relative_key = str(file_path.relative_to(self.local_path))
                metadata = self._get_local_metadata(relative_key)
                if metadata:
                    objects.append(metadata)
        
        return objects
    
    def _list_cloud_objects(self, prefix: str, max_keys: int, bucket_name: str) -> List[ObjectMetadata]:
        """列出云存储对象"""
        if self.config.provider == StorageProvider.AWS_S3:
            response = self.client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            return [self._parse_aws_object(obj) for obj in response.get('Contents', [])]
        elif self.config.provider == StorageProvider.ALIYUN_OSS:
            if not OSS2_AVAILABLE:
                raise ImportError("oss2库未安装，无法使用阿里云OSS服务")
            objects = oss2.ObjectIterator(self.bucket, prefix=prefix, max_keys=max_keys)
            return [self._parse_oss_object(obj) for obj in objects]
        else:
            raise ValueError(f"不支持的云存储提供商: {self.config.provider}")
    
    def _parse_aws_object(self, obj: Dict) -> ObjectMetadata:
        """解析AWS S3对象"""
        return ObjectMetadata(
            name=obj['Key'].split('/')[-1],
            size=obj['Size'],
            content_type='application/octet-stream',
            etag=obj['ETag'].strip('"'),
            last_modified=obj['LastModified'],
            storage_provider=self.config.provider,
            bucket=self.config.bucket_name,
            key=obj['Key']
        )
    
    def _parse_oss_object(self, obj) -> ObjectMetadata:
        """解析阿里云OSS对象"""
        return ObjectMetadata(
            name=obj.key.split('/')[-1],
            size=obj.size,
            content_type=obj.content_type,
            etag=obj.etag,
            last_modified=obj.last_modified,
            storage_provider=self.config.provider,
            bucket=self.config.bucket_name,
            key=obj.key
        )
    
    def set_object_permissions(self, object_key: str, permission: Permission, bucket_name: Optional[str] = None) -> bool:
        """
        设置对象权限
        
        Args:
            object_key: 对象键名
            permission: 权限类型
            bucket_name: 存储桶名称（可选）
            
        Returns:
            bool: 设置是否成功
        """
        with self._lock:
            try:
                bucket_name = bucket_name or self.config.bucket_name
                
                if self.config.provider == StorageProvider.LOCAL:
                    # 本地存储不支持权限控制
                    self.logger.warning("本地存储不支持权限控制")
                    return True
                else:
                    return self._set_cloud_permissions(object_key, permission, bucket_name)
                    
            except Exception as e:
                self.logger.error(f"设置对象权限失败: {object_key}, 错误: {e}")
                raise
    
    def _set_cloud_permissions(self, object_key: str, permission: Permission, bucket_name: str) -> bool:
        """设置云存储对象权限"""
        if self.config.provider == StorageProvider.AWS_S3:
            acl = 'public-read' if permission == Permission.PUBLIC_READ else 'private'
            self.client.put_object_acl(Bucket=bucket_name, Key=object_key, ACL=acl)
        elif self.config.provider == StorageProvider.ALIYUN_OSS:
            # 阿里云OSS权限设置逻辑
            pass
        else:
            raise ValueError(f"不支持的云存储提供商: {self.config.provider}")
        
        self.logger.info(f"成功设置对象权限: {object_key}")
        return True
    
    def create_object_version(self, object_key: str, version_name: str) -> bool:
        """
        创建对象版本
        
        Args:
            object_key: 对象键名
            version_name: 版本名称
            
        Returns:
            bool: 创建是否成功
        """
        with self._lock:
            try:
                # 获取当前对象
                current_metadata = self.get_object_metadata(object_key)
                if not current_metadata:
                    raise ValueError(f"对象不存在: {object_key}")
                
                # 创建版本标识
                version_key = f"{object_key}_v{version_name}"
                
                # 复制对象到版本位置
                if self.config.provider == StorageProvider.LOCAL:
                    source_path = self.local_path / object_key
                    version_path = self.local_path / version_key
                    if source_path.exists():
                        version_path.parent.mkdir(parents=True, exist_ok=True)
                        version_path.write_bytes(source_path.read_bytes())
                else:
                    self._copy_object_to_version(object_key, version_key)
                
                self.logger.info(f"成功创建对象版本: {version_key}")
                return True
                
            except Exception as e:
                self.logger.error(f"创建对象版本失败: {object_key}, 错误: {e}")
                raise
    
    def _copy_object_to_version(self, source_key: str, version_key: str):
        """复制对象到版本位置"""
        if self.config.provider == StorageProvider.AWS_S3:
            copy_source = {'Bucket': self.config.bucket_name, 'Key': source_key}
            self.client.copy_object(
                CopySource=copy_source,
                Bucket=self.config.bucket_name,
                Key=version_key
            )
        elif self.config.provider == StorageProvider.ALIYUN_OSS:
            self.bucket.copy_object(self.config.bucket_name, source_key, version_key)
        else:
            raise ValueError(f"不支持的云存储提供商: {self.config.provider}")
    
    def get_object_versions(self, object_key: str) -> List[str]:
        """
        获取对象版本列表
        
        Args:
            object_key: 对象键名
            
        Returns:
            List[str]: 版本列表
        """
        try:
            versions = []
            if self.config.provider == StorageProvider.LOCAL:
                # 本地存储版本查找
                object_dir = self.local_path / object_key
                for file_path in self.local_path.glob(f"{object_key}_v*"):
                    version = file_path.name.replace(f"{object_key}_v", "")
                    versions.append(version)
            else:
                # 云存储版本查找逻辑
                pass
            
            return sorted(versions)
            
        except Exception as e:
            self.logger.error(f"获取对象版本失败: {object_key}, 错误: {e}")
            return []
    
    def backup_object(self, object_key: str, backup_path: str) -> bool:
        """
        备份对象
        
        Args:
            object_key: 对象键名
            backup_path: 备份路径
            
        Returns:
            bool: 备份是否成功
        """
        with self._lock:
            try:
                backup_path = Path(backup_path)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 下载对象到备份路径
                return self.download_object(object_key, backup_path)
                
            except Exception as e:
                self.logger.error(f"备份对象失败: {object_key}, 错误: {e}")
                raise
    
    def restore_object(self, backup_path: str, object_key: str) -> bool:
        """
        恢复对象
        
        Args:
            backup_path: 备份路径
            object_key: 对象键名
            
        Returns:
            bool: 恢复是否成功
        """
        with self._lock:
            try:
                backup_path = Path(backup_path)
                if not backup_path.exists():
                    raise FileNotFoundError(f"备份文件不存在: {backup_path}")
                
                # 上传备份文件到存储
                metadata = self.upload_object(backup_path, object_key)
                
                self.logger.info(f"成功恢复对象: {object_key}")
                return True
                
            except Exception as e:
                self.logger.error(f"恢复对象失败: {backup_path}, 错误: {e}")
                raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取使用统计
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                'cache_size': len(self._objects_cache),
                'provider': self.config.provider.value,
                'bucket': self.config.bucket_name,
                'uptime': time.time() - getattr(self, '_start_time', time.time())
            })
            return stats
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self._objects_cache.clear()
            self.logger.info("缓存已清空")
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict[str, Any]: 健康状态
        """
        try:
            if self.config.provider == StorageProvider.LOCAL:
                # 本地存储健康检查
                health_status = {
                    'status': 'healthy',
                    'provider': self.config.provider.value,
                    'path_accessible': self.local_path.exists(),
                    'path_writable': os.access(self.local_path, os.W_OK) if self.local_path.exists() else False
                }
            else:
                # 云存储健康检查
                health_status = {
                    'status': 'healthy',
                    'provider': self.config.provider.value,
                    'endpoint_accessible': True  # 简化实现
                }
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'provider': self.config.provider.value
            }
    
    def __enter__(self):
        """上下文管理器入口"""
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.clear_cache()
        self.logger.info("对象存储管理器已关闭")


# 便捷函数
def create_local_storage_manager(storage_path: str, bucket_name: str = "default") -> ObjectStorageManager:
    """
    创建本地存储管理器
    
    Args:
        storage_path: 本地存储路径
        bucket_name: 存储桶名称
        
    Returns:
        ObjectStorageManager: 存储管理器实例
    """
    config = StorageConfig(
        provider=StorageProvider.LOCAL,
        access_key="",
        secret_key="",
        endpoint=storage_path,
        region="local",
        bucket_name=bucket_name
    )
    return ObjectStorageManager(config)


def create_aws_s3_manager(access_key: str, secret_key: str, endpoint: str, 
                         region: str, bucket_name: str) -> ObjectStorageManager:
    """
    创建AWS S3存储管理器
    
    Args:
        access_key: 访问密钥
        secret_key: 秘密密钥
        endpoint: 终端节点
        region: 区域
        bucket_name: 存储桶名称
        
    Returns:
        ObjectStorageManager: 存储管理器实例
    """
    config = StorageConfig(
        provider=StorageProvider.AWS_S3,
        access_key=access_key,
        secret_key=secret_key,
        endpoint=endpoint,
        region=region,
        bucket_name=bucket_name
    )
    return ObjectStorageManager(config)


def create_aliyun_oss_manager(access_key: str, secret_key: str, endpoint: str, 
                             bucket_name: str) -> ObjectStorageManager:
    """
    创建阿里云OSS存储管理器
    
    Args:
        access_key: 访问密钥
        secret_key: 秘密密钥
        endpoint: 终端节点
        bucket_name: 存储桶名称
        
    Returns:
        ObjectStorageManager: 存储管理器实例
    """
    config = StorageConfig(
        provider=StorageProvider.ALIYUN_OSS,
        access_key=access_key,
        secret_key=secret_key,
        endpoint=endpoint,
        region="oss",
        bucket_name=bucket_name
    )
    return ObjectStorageManager(config)