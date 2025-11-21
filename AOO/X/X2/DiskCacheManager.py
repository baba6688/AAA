#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X2磁盘缓存管理器

提供高效的磁盘缓存管理功能，包括文件存储、缓存优化、空间管理等功能。
"""

import os
import json
import time
import gzip
import shutil
import hashlib
import threading
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import OrderedDict
import fcntl
import stat


@dataclass
class CacheConfig:
    """缓存配置类"""
    max_cache_size: int = 1024 * 1024 * 100  # 100MB
    max_file_size: int = 1024 * 1024 * 10    # 10MB
    cleanup_interval: int = 3600             # 1小时
    compression_enabled: bool = True
    backup_enabled: bool = True
    index_file: str = "cache_index.json"
    backup_dir: str = "backup"
    temp_dir: str = "temp"
    gc_threshold: float = 0.8                # 垃圾回收阈值


@dataclass
class CacheEntry:
    """缓存条目类"""
    key: str
    file_path: str
    size: int
    created_time: float
    access_time: float
    compressed: bool
    checksum: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class CacheStats:
    """缓存统计类"""
    total_files: int = 0
    total_size: int = 0
    compressed_files: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_cleanup: float = 0
    last_backup: float = 0
    gc_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheStats':
        """从字典创建实例"""
        return cls(**data)


class DiskCacheManager:
    """磁盘缓存管理器"""
    
    def __init__(self, cache_dir: str, config: Optional[CacheConfig] = None):
        """
        初始化磁盘缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            config: 缓存配置
        """
        self.cache_dir = Path(cache_dir)
        self.config = config or CacheConfig()
        self.lock = threading.RLock()
        
        # 创建必要的目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / self.config.backup_dir).mkdir(exist_ok=True)
        (self.cache_dir / self.config.temp_dir).mkdir(exist_ok=True)
        
        # 初始化组件
        self._init_logging()
        self._init_index()
        self._init_stats()
        
        # 启动后台任务
        self._start_background_tasks()
        
        self.logger.info(f"磁盘缓存管理器初始化完成: {self.cache_dir}")
    
    def _init_logging(self):
        """初始化日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DiskCacheManager')
    
    def _init_index(self):
        """初始化缓存索引"""
        index_file = self.cache_dir / self.config.index_file
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.index = OrderedDict()
                    for key, entry_data in data.items():
                        self.index[key] = CacheEntry.from_dict(entry_data)
            except Exception as e:
                self.logger.error(f"加载缓存索引失败: {e}")
                self.index = OrderedDict()
        else:
            self.index = OrderedDict()
    
    def _save_index(self):
        """保存缓存索引"""
        index_file = self.cache_dir / self.config.index_file
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                data = {key: entry.to_dict() for key, entry in self.index.items()}
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存缓存索引失败: {e}")
    
    def _init_stats(self):
        """初始化统计信息"""
        stats_file = self.cache_dir / "cache_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.stats = CacheStats.from_dict(data)
            except Exception as e:
                self.logger.error(f"加载统计信息失败: {e}")
                self.stats = CacheStats()
        else:
            self.stats = CacheStats()
    
    def _save_stats(self):
        """保存统计信息"""
        stats_file = self.cache_dir / "cache_stats.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存统计信息失败: {e}")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        def cleanup_task():
            while True:
                time.sleep(self.config.cleanup_interval)
                try:
                    self.cleanup()
                except Exception as e:
                    self.logger.error(f"后台清理任务失败: {e}")
        
        def backup_task():
            while True:
                time.sleep(self.config.cleanup_interval * 6)  # 6小时备份一次
                try:
                    if self.config.backup_enabled:
                        self.backup()
                except Exception as e:
                    self.logger.error(f"后台备份任务失败: {e}")
        
        # 启动后台线程
        threading.Thread(target=cleanup_task, daemon=True).start()
        threading.Thread(target=backup_task, daemon=True).start()
    
    def _get_file_path(self, key: str) -> Path:
        """获取缓存文件的路径"""
        # 使用哈希值创建子目录
        hash_obj = hashlib.md5(key.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        subdir = hash_hex[:2] + hash_hex[2:4]
        return self.cache_dir / subdir / f"{key}.cache"
    
    def _get_checksum(self, data: bytes) -> str:
        """计算数据校验和"""
        return hashlib.md5(data).hexdigest()
    
    def _compress_data(self, data: bytes) -> bytes:
        """压缩数据"""
        if not self.config.compression_enabled:
            return data
        return gzip.compress(data)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """解压数据"""
        try:
            return gzip.decompress(data)
        except:
            return data
    
    def _acquire_lock(self, file_path: Path) -> Optional[Any]:
        """获取文件锁"""
        try:
            f = open(file_path, 'w')
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except:
            return None
    
    def _release_lock(self, lock_file):
        """释放文件锁"""
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except:
                pass
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒）
            
        Returns:
            bool: 是否设置成功
        """
        with self.lock:
            try:
                # 序列化数据
                data = pickle.dumps(value)
                
                # 检查文件大小
                if len(data) > self.config.max_file_size:
                    self.logger.warning(f"缓存项过大: {key}")
                    return False
                
                # 压缩数据
                if self.config.compression_enabled:
                    data = self._compress_data(data)
                    compressed = True
                else:
                    compressed = False
                
                # 计算校验和
                checksum = self._get_checksum(data)
                
                # 获取文件路径
                file_path = self._get_file_path(key)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 写入文件
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # 设置文件权限
                os.chmod(file_path, 0o644)
                
                # 更新索引
                now = time.time()
                entry = CacheEntry(
                    key=key,
                    file_path=str(file_path),
                    size=len(data),
                    created_time=now,
                    access_time=now,
                    compressed=compressed,
                    checksum=checksum
                )
                
                # 如果存在TTL，设置过期时间
                if ttl:
                    entry.expires_time = now + ttl
                
                # 移除旧条目（如果存在）
                if key in self.index:
                    old_entry = self.index[key]
                    old_path = Path(old_entry.file_path)
                    if old_path.exists():
                        old_path.unlink()
                
                self.index[key] = entry
                
                # 保存索引
                self._save_index()
                
                self.logger.debug(f"缓存项已设置: {key}")
                return True
                
            except Exception as e:
                self.logger.error(f"设置缓存项失败: {key}, {e}")
                return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值或默认值
        """
        with self.lock:
            try:
                # 检查索引中是否存在
                if key not in self.index:
                    self.stats.cache_misses += 1
                    self._save_stats()
                    return default
                
                entry = self.index[key]
                
                # 检查是否过期
                if hasattr(entry, 'expires_time') and time.time() > entry.expires_time:
                    self._remove_entry(key)
                    self.stats.cache_misses += 1
                    self._save_stats()
                    return default
                
                # 检查文件是否存在
                file_path = Path(entry.file_path)
                if not file_path.exists():
                    self._remove_entry(key)
                    self.stats.cache_misses += 1
                    self._save_stats()
                    return default
                
                # 读取文件
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # 验证校验和
                current_checksum = self._get_checksum(data)
                if current_checksum != entry.checksum:
                    self.logger.warning(f"缓存项校验和验证失败: {key}")
                    self._remove_entry(key)
                    self.stats.cache_misses += 1
                    self._save_stats()
                    return default
                
                # 解压数据
                if entry.compressed:
                    data = self._decompress_data(data)
                
                # 反序列化数据
                value = pickle.loads(data)
                
                # 更新访问时间
                entry.access_time = time.time()
                self._save_index()
                
                # 更新统计
                self.stats.cache_hits += 1
                self._save_stats()
                
                self.logger.debug(f"缓存项已获取: {key}")
                return value
                
            except Exception as e:
                self.logger.error(f"获取缓存项失败: {key}, {e}")
                self.stats.cache_misses += 1
                self._save_stats()
                return default
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        with self.lock:
            try:
                if key in self.index:
                    self._remove_entry(key)
                    self.logger.debug(f"缓存项已删除: {key}")
                    return True
                return False
            except Exception as e:
                self.logger.error(f"删除缓存项失败: {key}, {e}")
                return False
    
    def _remove_entry(self, key: str):
        """移除缓存条目"""
        if key in self.index:
            entry = self.index[key]
            file_path = Path(entry.file_path)
            if file_path.exists():
                file_path.unlink()
            del self.index[key]
            self._save_index()
    
    def exists(self, key: str) -> bool:
        """
        检查缓存项是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在
        """
        return key in self.index
    
    def keys(self) -> List[str]:
        """
        获取所有缓存键
        
        Returns:
            List[str]: 缓存键列表
        """
        return list(self.index.keys())
    
    def clear(self) -> bool:
        """
        清空所有缓存
        
        Returns:
            bool: 是否清空成功
        """
        with self.lock:
            try:
                # 删除所有缓存文件
                for entry in self.index.values():
                    file_path = Path(entry.file_path)
                    if file_path.exists():
                        file_path.unlink()
                
                # 清空索引
                self.index.clear()
                self._save_index()
                
                self.logger.info("所有缓存已清空")
                return True
            except Exception as e:
                self.logger.error(f"清空缓存失败: {e}")
                return False
    
    def cleanup(self) -> bool:
        """
        清理过期和无效的缓存项
        
        Returns:
            bool: 是否清理成功
        """
        with self.lock:
            try:
                current_time = time.time()
                removed_count = 0
                
                # 清理过期项
                keys_to_remove = []
                for key, entry in self.index.items():
                    # 检查过期时间
                    if hasattr(entry, 'expires_time') and current_time > entry.expires_time:
                        keys_to_remove.append(key)
                        continue
                    
                    # 检查文件是否存在
                    file_path = Path(entry.file_path)
                    if not file_path.exists():
                        keys_to_remove.append(key)
                
                # 移除过期项
                for key in keys_to_remove:
                    self._remove_entry(key)
                    removed_count += 1
                
                # 检查缓存大小，如果超过限制则清理最久未使用的项
                total_size = sum(entry.size for entry in self.index.values())
                max_size = self.config.max_cache_size
                
                if total_size > max_size:
                    # 按访问时间排序，清理最久未使用的项
                    sorted_items = sorted(
                        self.index.items(),
                        key=lambda x: x[1].access_time
                    )
                    
                    for key, entry in sorted_items:
                        if total_size <= max_size * self.config.gc_threshold:
                            break
                        
                        self._remove_entry(key)
                        total_size -= entry.size
                        removed_count += 1
                
                # 更新统计
                self.stats.last_cleanup = current_time
                self._save_stats()
                
                self.logger.info(f"清理完成，移除 {removed_count} 个缓存项")
                return True
                
            except Exception as e:
                self.logger.error(f"清理缓存失败: {e}")
                return False
    
    def backup(self) -> bool:
        """
        备份缓存数据
        
        Returns:
            bool: 是否备份成功
        """
        with self.lock:
            try:
                if not self.config.backup_enabled:
                    return True
                
                backup_dir = self.cache_dir / self.config.backup_dir
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"backup_{timestamp}"
                
                # 创建备份目录
                backup_path.mkdir(exist_ok=True)
                
                # 复制缓存文件
                for entry in self.index.values():
                    src_path = Path(entry.file_path)
                    if src_path.exists():
                        dst_path = backup_path / src_path.name
                        shutil.copy2(src_path, dst_path)
                
                # 备份索引文件
                index_src = self.cache_dir / self.config.index_file
                if index_src.exists():
                    shutil.copy2(index_src, backup_path / self.config.index_file)
                
                # 备份统计文件
                stats_src = self.cache_dir / "cache_stats.json"
                if stats_src.exists():
                    shutil.copy2(stats_src, backup_path / "cache_stats.json")
                
                # 清理旧备份（保留最近5个）
                self._cleanup_old_backups()
                
                # 更新统计
                self.stats.last_backup = time.time()
                self._save_stats()
                
                self.logger.info(f"备份完成: {backup_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"备份缓存失败: {e}")
                return False
    
    def _cleanup_old_backups(self):
        """清理旧备份"""
        backup_dir = self.cache_dir / self.config.backup_dir
        if not backup_dir.exists():
            return
        
        # 获取所有备份目录
        backup_dirs = [d for d in backup_dir.iterdir() if d.is_dir() and d.name.startswith("backup_")]
        backup_dirs.sort(key=lambda x: x.name, reverse=True)
        
        # 删除超过5个的旧备份
        for backup_dir in backup_dirs[5:]:
            shutil.rmtree(backup_dir)
    
    def restore(self, backup_name: str) -> bool:
        """
        恢复缓存数据
        
        Args:
            backup_name: 备份名称
            
        Returns:
            bool: 是否恢复成功
        """
        with self.lock:
            try:
                backup_dir = self.cache_dir / self.config.backup_dir / backup_name
                if not backup_dir.exists():
                    self.logger.error(f"备份不存在: {backup_name}")
                    return False
                
                # 清空当前缓存
                self.clear()
                
                # 恢复缓存文件
                for file_path in backup_dir.glob("*.cache"):
                    shutil.copy2(file_path, self.cache_dir / file_path.name)
                
                # 恢复索引文件
                index_backup = backup_dir / self.config.index_file
                if index_backup.exists():
                    shutil.copy2(index_backup, self.cache_dir / self.config.index_file)
                
                # 恢复统计文件
                stats_backup = backup_dir / "cache_stats.json"
                if stats_backup.exists():
                    shutil.copy2(stats_backup, self.cache_dir / "cache_stats.json")
                
                # 重新加载索引和统计
                self._init_index()
                self._init_stats()
                
                self.logger.info(f"恢复完成: {backup_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"恢复缓存失败: {backup_name}, {e}")
                return False
    
    def get_stats(self) -> CacheStats:
        """
        获取缓存统计信息
        
        Returns:
            CacheStats: 统计信息
        """
        with self.lock:
            # 更新实时统计
            self.stats.total_files = len(self.index)
            self.stats.total_size = sum(entry.size for entry in self.index.values())
            self.stats.compressed_files = sum(1 for entry in self.index.values() if entry.compressed)
            
            self._save_stats()
            return self.stats
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存详细信息
        
        Returns:
            Dict[str, Any]: 缓存信息
        """
        stats = self.get_stats()
        
        # 计算磁盘使用情况
        disk_usage = shutil.disk_usage(self.cache_dir)
        
        return {
            "缓存目录": str(self.cache_dir),
            "总文件数": stats.total_files,
            "总大小": f"{stats.total_size / 1024 / 1024:.2f} MB",
            "压缩文件数": stats.compressed_files,
            "缓存命中率": f"{stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1) * 100:.2f}%",
            "磁盘使用": f"{(disk_usage.used / disk_usage.total * 100):.2f}%",
            "剩余空间": f"{disk_usage.free / 1024 / 1024 / 1024:.2f} GB",
            "最后清理": datetime.fromtimestamp(stats.last_cleanup).strftime("%Y-%m-%d %H:%M:%S") if stats.last_cleanup else "从未",
            "最后备份": datetime.fromtimestamp(stats.last_backup).strftime("%Y-%m-%d %H:%M:%S") if stats.last_backup else "从未",
            "垃圾回收次数": stats.gc_count
        }
    
    def optimize(self) -> bool:
        """
        优化缓存性能
        
        Returns:
            bool: 是否优化成功
        """
        with self.lock:
            try:
                # 重新组织文件结构
                self._reorganize_files()
                
                # 重建索引
                self._rebuild_index()
                
                # 清理碎片文件
                self._cleanup_fragments()
                
                self.stats.gc_count += 1
                self._save_stats()
                
                self.logger.info("缓存优化完成")
                return True
                
            except Exception as e:
                self.logger.error(f"缓存优化失败: {e}")
                return False
    
    def _reorganize_files(self):
        """重新组织文件结构"""
        for key in list(self.index.keys()):
            entry = self.index[key]
            old_path = Path(entry.file_path)
            new_path = self._get_file_path(key)
            
            if old_path != new_path and old_path.exists():
                # 创建新目录
                new_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 移动文件
                shutil.move(str(old_path), str(new_path))
                
                # 更新索引
                entry.file_path = str(new_path)
    
    def _rebuild_index(self):
        """重建索引"""
        new_index = OrderedDict()
        
        for file_path in self.cache_dir.rglob("*.cache"):
            try:
                # 读取文件头部获取信息
                with open(file_path, 'rb') as f:
                    # 这里可以读取文件头部的元数据
                    # 简化处理，使用文件属性
                    stat_info = file_path.stat()
                    
                    key = file_path.stem
                    size = stat_info.st_size
                    created_time = stat_info.st_ctime
                    access_time = stat_info.st_atime
                    
                    # 计算校验和
                    with open(file_path, 'rb') as f2:
                        data = f2.read()
                        checksum = self._get_checksum(data)
                    
                    entry = CacheEntry(
                        key=key,
                        file_path=str(file_path),
                        size=size,
                        created_time=created_time,
                        access_time=access_time,
                        compressed=False,  # 需要根据文件内容判断
                        checksum=checksum
                    )
                    
                    new_index[key] = entry
                    
            except Exception as e:
                self.logger.warning(f"重建索引时跳过文件: {file_path}, {e}")
        
        self.index = new_index
        self._save_index()
    
    def _cleanup_fragments(self):
        """清理碎片文件"""
        for file_path in self.cache_dir.rglob("*.tmp"):
            try:
                file_path.unlink()
            except Exception as e:
                self.logger.warning(f"删除临时文件失败: {file_path}, {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self._save_index()
        self._save_stats()
    
    def __len__(self) -> int:
        """返回缓存项数量"""
        return len(self.index)
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        return self.exists(key)