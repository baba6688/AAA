#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R8归档管理器
实现数据归档、压缩、检索等功能的完整归档管理系统
"""

import os
import json
import hashlib
import shutil
import sqlite3
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import gzip
import bz2
import lzma
import zlib
import pickle
import tempfile
import fcntl
from contextlib import contextmanager


class CompressionType(Enum):
    """压缩算法类型"""
    NONE = "none"
    GZIP = "gzip"
    BZ2 = "bz2"
    LZMA = "lzma"
    ZLIB = "zlib"


class ArchiveStatus(Enum):
    """归档状态"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    CORRUPTED = "corrupted"


@dataclass
class ArchiveEntry:
    """归档条目数据结构"""
    id: str
    original_path: str
    archive_path: str
    compression_type: CompressionType
    size: int
    compressed_size: int
    checksum: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    status: ArchiveStatus = ArchiveStatus.ARCHIVED
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ArchiveStrategy(Enum):
    """归档策略"""
    TIME_BASED = "time_based"  # 基于时间
    SIZE_BASED = "size_based"  # 基于大小
    ACCESS_BASED = "access_based"  # 基于访问
    CUSTOM = "custom"  # 自定义


@dataclass
class ArchiveRule:
    """归档规则"""
    name: str
    strategy: ArchiveStrategy
    conditions: Dict[str, Any]
    compression_type: CompressionType
    retention_days: int
    enabled: bool = True


class ArchiveManager:
    """R8归档管理器主类"""
    
    def __init__(self, archive_root: str, config_file: str = None):
        """
        初始化归档管理器
        
        Args:
            archive_root: 归档根目录
            config_file: 配置文件路径
        """
        self.archive_root = Path(archive_root)
        self.config_file = config_file or str(self.archive_root / "config.json")
        self.index_db = str(self.archive_root / "archive_index.db")
        self.temp_dir = self.archive_root / "temp"
        self.logs_dir = self.archive_root / "logs"
        
        # 创建必要目录
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # 初始化日志
        self._setup_logging()
        
        # 初始化数据库
        self._init_database()
        
        # 加载配置和规则
        self.config = self._load_config()
        self.archive_rules = self._load_archive_rules()
        
        # 线程锁
        self._lock = threading.RLock()
        
        self.logger.info("归档管理器初始化完成")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.logs_dir / f"archive_manager_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """初始化SQLite数据库"""
        with sqlite3.connect(self.index_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archives (
                    id TEXT PRIMARY KEY,
                    original_path TEXT NOT NULL,
                    archive_path TEXT NOT NULL,
                    compression_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    compressed_size INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    UNIQUE(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archive_rules (
                    name TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    compression_type TEXT NOT NULL,
                    retention_days INTEGER NOT NULL,
                    enabled BOOLEAN NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    archive_id TEXT NOT NULL,
                    access_time TEXT NOT NULL,
                    access_type TEXT NOT NULL,
                    FOREIGN KEY (archive_id) REFERENCES archives (id)
                )
            """)
            
            conn.commit()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {e}")
        
        # 默认配置
        return {
            "max_archive_size": 1024 * 1024 * 1024,  # 1GB
            "compression_level": 6,
            "auto_cleanup": True,
            "cleanup_interval": 24 * 3600,  # 24小时
            "verify_on_archive": True,
            "thread_pool_size": 4
        }
    
    def _save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
    
    def _load_archive_rules(self) -> List[ArchiveRule]:
        """加载归档规则"""
        rules = []
        try:
            with sqlite3.connect(self.index_db) as conn:
                cursor = conn.execute("SELECT * FROM archive_rules WHERE enabled = 1")
                for row in cursor.fetchall():
                    rule = ArchiveRule(
                        name=row[0],
                        strategy=ArchiveStrategy(row[1]),
                        conditions=json.loads(row[2]),
                        compression_type=CompressionType(row[3]),
                        retention_days=row[4],
                        enabled=bool(row[5])
                    )
                    rules.append(rule)
        except Exception as e:
            self.logger.error(f"加载归档规则失败: {e}")
        
        # 如果没有规则，创建默认规则
        if not rules:
            default_rule = ArchiveRule(
                name="默认规则",
                strategy=ArchiveStrategy.TIME_BASED,
                conditions={"days": 30},
                compression_type=CompressionType.GZIP,
                retention_days=365
            )
            # 直接保存到数据库，不调用add_archive_rule避免循环引用
            try:
                with sqlite3.connect(self.index_db) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO archive_rules 
                        (name, strategy, conditions, compression_type, retention_days, enabled)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        default_rule.name,
                        default_rule.strategy.value,
                        json.dumps(default_rule.conditions),
                        default_rule.compression_type.value,
                        default_rule.retention_days,
                        default_rule.enabled
                    ))
                    conn.commit()
                rules.append(default_rule)
            except Exception as e:
                self.logger.error(f"创建默认规则失败: {e}")
        
        return rules
    
    def _calculate_checksum(self, data: bytes) -> str:
        """计算数据校验和"""
        return hashlib.sha256(data).hexdigest()
    
    def _compress_data(self, data: bytes, compression_type: CompressionType) -> Tuple[bytes, int]:
        """压缩数据"""
        if compression_type == CompressionType.NONE:
            return data, len(data)
        elif compression_type == CompressionType.GZIP:
            compressed = gzip.compress(data, compresslevel=self.config["compression_level"])
        elif compression_type == CompressionType.BZ2:
            compressed = bz2.compress(data, compresslevel=self.config["compression_level"])
        elif compression_type == CompressionType.LZMA:
            compressed = lzma.compress(data, preset=self.config["compression_level"])
        elif compression_type == CompressionType.ZLIB:
            compressed = zlib.compress(data, level=self.config["compression_level"])
        else:
            raise ValueError(f"不支持的压缩类型: {compression_type}")
        
        return compressed, len(compressed)
    
    def _decompress_data(self, data: bytes, compression_type: CompressionType) -> bytes:
        """解压缩数据"""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression_type == CompressionType.BZ2:
            return bz2.decompress(data)
        elif compression_type == CompressionType.LZMA:
            return lzma.decompress(data)
        elif compression_type == CompressionType.ZLIB:
            return zlib.decompress(data)
        else:
            raise ValueError(f"不支持的压缩类型: {compression_type}")
    
    @contextmanager
    def _file_lock(self, file_path: Path):
        """文件锁上下文管理器"""
        lock_file = file_path.with_suffix(file_path.suffix + '.lock')
        with open(lock_file, 'w') as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
            except IOError:
                raise RuntimeError(f"无法获取文件锁: {file_path}")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                if lock_file.exists():
                    lock_file.unlink()
    
    def add_archive_rule(self, rule: ArchiveRule) -> bool:
        """添加归档规则"""
        try:
            with sqlite3.connect(self.index_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO archive_rules 
                    (name, strategy, conditions, compression_type, retention_days, enabled)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    rule.name,
                    rule.strategy.value,
                    json.dumps(rule.conditions),
                    rule.compression_type.value,
                    rule.retention_days,
                    rule.enabled
                ))
                conn.commit()
            
            self.archive_rules.append(rule)
            self.logger.info(f"归档规则添加成功: {rule.name}")
            return True
        except Exception as e:
            self.logger.error(f"添加归档规则失败: {e}")
            return False
    
    def remove_archive_rule(self, rule_name: str) -> bool:
        """删除归档规则"""
        try:
            with sqlite3.connect(self.index_db) as conn:
                conn.execute("DELETE FROM archive_rules WHERE name = ?", (rule_name,))
                conn.commit()
            
            # 从内存中移除
            self.archive_rules = [r for r in self.archive_rules if r.name != rule_name]
            self.logger.info(f"归档规则删除成功: {rule_name}")
            return True
        except Exception as e:
            self.logger.error(f"删除归档规则失败: {e}")
            return False
    
    def archive_file(self, file_path: str, rule_name: str = None, custom_compression: CompressionType = None) -> Optional[str]:
        """归档文件"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"文件不存在: {file_path}")
            return None
        
        with self._lock:
            try:
                # 选择归档规则
                if rule_name:
                    rule = next((r for r in self.archive_rules if r.name == rule_name), None)
                    if not rule:
                        self.logger.error(f"归档规则不存在: {rule_name}")
                        return None
                else:
                    rule = self._select_archive_rule(file_path)
                
                # 读取文件数据
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # 计算校验和
                checksum = self._calculate_checksum(data)
                
                # 压缩数据
                compression_type = custom_compression or rule.compression_type
                compressed_data, compressed_size = self._compress_data(data, compression_type)
                
                # 生成归档ID和路径
                archive_id = hashlib.md5(f"{file_path}_{time.time()}".encode()).hexdigest()
                archive_subdir = datetime.now().strftime("%Y/%m/%d")
                archive_dir = self.archive_root / "archives" / archive_subdir
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_file_path = archive_dir / f"{archive_id}.archive"
                
                # 保存压缩数据
                with open(archive_file_path, 'wb') as f:
                    f.write(compressed_data)
                
                # 创建归档条目
                expires_at = datetime.now() + timedelta(days=rule.retention_days)
                entry = ArchiveEntry(
                    id=archive_id,
                    original_path=str(file_path),
                    archive_path=str(archive_file_path),
                    compression_type=compression_type,
                    size=len(data),
                    compressed_size=compressed_size,
                    checksum=checksum,
                    created_at=datetime.now(),
                    expires_at=expires_at,
                    metadata={"rule_name": rule.name, "original_name": file_path.name}
                )
                
                # 保存到数据库
                self._save_archive_entry(entry)
                
                self.logger.info(f"文件归档成功: {file_path} -> {archive_id}")
                return archive_id
                
            except Exception as e:
                self.logger.error(f"文件归档失败: {e}")
                return None
    
    def _select_archive_rule(self, file_path: Path) -> ArchiveRule:
        """选择合适的归档规则"""
        for rule in self.archive_rules:
            if self._check_rule_conditions(file_path, rule):
                return rule
        
        # 如果没有匹配的规则，使用默认规则
        return self.archive_rules[0]
    
    def _check_rule_conditions(self, file_path: Path, rule: ArchiveRule) -> bool:
        """检查文件是否满足归档规则条件"""
        if rule.strategy == ArchiveStrategy.TIME_BASED:
            # 基于文件修改时间
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            days_diff = (datetime.now() - file_mtime).days
            return days_diff >= rule.conditions.get("days", 30)
        
        elif rule.strategy == ArchiveStrategy.SIZE_BASED:
            # 基于文件大小
            file_size = file_path.stat().st_size
            return file_size >= rule.conditions.get("size", 1024 * 1024)  # 默认1MB
        
        elif rule.strategy == ArchiveStrategy.ACCESS_BASED:
            # 基于文件访问时间
            file_atime = datetime.fromtimestamp(file_path.stat().st_atime)
            days_diff = (datetime.now() - file_atime).days
            return days_diff >= rule.conditions.get("days", 90)
        
        return False
    
    def _save_archive_entry(self, entry: ArchiveEntry):
        """保存归档条目到数据库"""
        with sqlite3.connect(self.index_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO archives 
                (id, original_path, archive_path, compression_type, size, compressed_size, 
                 checksum, created_at, expires_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.original_path,
                entry.archive_path,
                entry.compression_type.value,
                entry.size,
                entry.compressed_size,
                entry.checksum,
                entry.created_at.isoformat(),
                entry.expires_at.isoformat() if entry.expires_at else None,
                entry.status.value,
                json.dumps(entry.metadata)
            ))
            conn.commit()
    
    def retrieve_archive(self, archive_id: str, output_path: str = None) -> bool:
        """检索归档文件"""
        with self._lock:
            try:
                # 从数据库获取归档信息
                entry = self._get_archive_entry(archive_id)
                if not entry:
                    self.logger.error(f"归档条目不存在: {archive_id}")
                    return False
                
                # 读取压缩数据
                with open(entry.archive_path, 'rb') as f:
                    compressed_data = f.read()
                
                # 解压缩数据
                decompressed_data = self._decompress_data(compressed_data, entry.compression_type)
                
                # 验证校验和
                if self._calculate_checksum(decompressed_data) != entry.checksum:
                    self.logger.error(f"归档文件校验失败: {archive_id}")
                    return False
                
                # 保存到输出路径
                if output_path:
                    output_path = Path(output_path)
                else:
                    output_path = Path(entry.original_path)
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(decompressed_data)
                
                # 记录访问日志
                self._log_access(archive_id, "retrieve")
                
                self.logger.info(f"归档文件检索成功: {archive_id} -> {output_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"归档文件检索失败: {e}")
                return False
    
    def _get_archive_entry(self, archive_id: str) -> Optional[ArchiveEntry]:
        """从数据库获取归档条目"""
        with sqlite3.connect(self.index_db) as conn:
            cursor = conn.execute("SELECT * FROM archives WHERE id = ?", (archive_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return ArchiveEntry(
                id=row[0],
                original_path=row[1],
                archive_path=row[2],
                compression_type=CompressionType(row[3]),
                size=row[4],
                compressed_size=row[5],
                checksum=row[6],
                created_at=datetime.fromisoformat(row[7]),
                expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
                status=ArchiveStatus(row[9]),
                metadata=json.loads(row[10]) if row[10] else {}
            )
    
    def _log_access(self, archive_id: str, access_type: str):
        """记录访问日志"""
        with sqlite3.connect(self.index_db) as conn:
            conn.execute("""
                INSERT INTO access_log (archive_id, access_time, access_type)
                VALUES (?, ?, ?)
            """, (archive_id, datetime.now().isoformat(), access_type))
            conn.commit()
    
    def verify_archive(self, archive_id: str) -> bool:
        """验证归档文件完整性"""
        try:
            entry = self._get_archive_entry(archive_id)
            if not entry:
                return False
            
            # 检查文件是否存在
            if not os.path.exists(entry.archive_path):
                self.logger.error(f"归档文件不存在: {entry.archive_path}")
                return False
            
            # 读取并验证校验和
            with open(entry.archive_path, 'rb') as f:
                compressed_data = f.read()
            
            decompressed_data = self._decompress_data(compressed_data, entry.compression_type)
            current_checksum = self._calculate_checksum(decompressed_data)
            
            if current_checksum != entry.checksum:
                self.logger.error(f"归档文件校验失败: {archive_id}")
                return False
            
            self.logger.info(f"归档文件验证成功: {archive_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"归档文件验证失败: {e}")
            return False
    
    def list_archives(self, status: ArchiveStatus = None, limit: int = 100) -> List[Dict[str, Any]]:
        """列出归档文件"""
        try:
            with sqlite3.connect(self.index_db) as conn:
                if status:
                    cursor = conn.execute(
                        "SELECT * FROM archives WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                        (status.value, limit)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM archives ORDER BY created_at DESC LIMIT ?",
                        (limit,)
                    )
                
                archives = []
                for row in cursor.fetchall():
                    archive = {
                        "id": row[0],
                        "original_path": row[1],
                        "archive_path": row[2],
                        "compression_type": row[3],
                        "size": row[4],
                        "compressed_size": row[5],
                        "compression_ratio": f"{(1 - row[5]/row[4])*100:.1f}%",
                        "checksum": row[6][:16] + "...",  # 只显示前16位
                        "created_at": row[7],
                        "expires_at": row[8],
                        "status": row[9],
                        "metadata": json.loads(row[10]) if row[10] else {}
                    }
                    archives.append(archive)
                
                return archives
                
        except Exception as e:
            self.logger.error(f"列出归档文件失败: {e}")
            return []
    
    def delete_archive(self, archive_id: str) -> bool:
        """删除归档文件"""
        with self._lock:
            try:
                entry = self._get_archive_entry(archive_id)
                if not entry:
                    return False
                
                # 删除物理文件
                if os.path.exists(entry.archive_path):
                    os.remove(entry.archive_path)
                
                # 从数据库删除
                with sqlite3.connect(self.index_db) as conn:
                    conn.execute("DELETE FROM archives WHERE id = ?", (archive_id,))
                    conn.execute("DELETE FROM access_log WHERE archive_id = ?", (archive_id,))
                    conn.commit()
                
                self.logger.info(f"归档文件删除成功: {archive_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"删除归档文件失败: {e}")
                return False
    
    def cleanup_expired_archives(self) -> int:
        """清理过期的归档文件"""
        cleaned_count = 0
        try:
            with sqlite3.connect(self.index_db) as conn:
                cursor = conn.execute("""
                    SELECT id FROM archives 
                    WHERE expires_at IS NOT NULL 
                    AND expires_at < ? 
                    AND status != 'expired'
                """, (datetime.now().isoformat(),))
                
                expired_ids = [row[0] for row in cursor.fetchall()]
                
                for archive_id in expired_ids:
                    if self.delete_archive(archive_id):
                        cleaned_count += 1
                
                self.logger.info(f"清理过期归档文件完成，清理数量: {cleaned_count}")
                return cleaned_count
                
        except Exception as e:
            self.logger.error(f"清理过期归档文件失败: {e}")
            return cleaned_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            with sqlite3.connect(self.index_db) as conn:
                # 总统计
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_count,
                        SUM(size) as total_size,
                        SUM(compressed_size) as total_compressed_size,
                        AVG(compressed_size * 1.0 / size) as avg_compression_ratio
                    FROM archives
                """)
                stats = cursor.fetchone()
                
                # 按压缩类型统计
                cursor = conn.execute("""
                    SELECT compression_type, COUNT(*), SUM(size), SUM(compressed_size)
                    FROM archives
                    GROUP BY compression_type
                """)
                compression_stats = {}
                for row in cursor.fetchall():
                    compression_stats[row[0]] = {
                        "count": row[1],
                        "original_size": row[2],
                        "compressed_size": row[3],
                        "compression_ratio": f"{(1 - row[3]/row[2])*100:.1f}%" if row[2] > 0 else "0%"
                    }
                
                # 状态统计
                cursor = conn.execute("""
                    SELECT status, COUNT(*)
                    FROM archives
                    GROUP BY status
                """)
                status_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_archives": stats[0] or 0,
                    "total_size": stats[1] or 0,
                    "total_compressed_size": stats[2] or 0,
                    "overall_compression_ratio": f"{(1 - (stats[2] or 0)/(stats[1] or 1))*100:.1f}%",
                    "average_compression_ratio": f"{(1 - (stats[3] or 0))*100:.1f}%" if stats[3] else "0%",
                    "compression_by_type": compression_stats,
                    "status_distribution": status_stats,
                    "storage_saved": (stats[1] or 0) - (stats[2] or 0)
                }
                
        except Exception as e:
            self.logger.error(f"获取存储统计失败: {e}")
            return {}
    
    def optimize_storage(self) -> Dict[str, Any]:
        """存储优化"""
        results = {
            "recompressed_count": 0,
            "deleted_corrupted": 0,
            "freed_space": 0
        }
        
        try:
            archives = self.list_archives()
            
            for archive in archives:
                archive_id = archive["id"]
                
                # 验证完整性
                if not self.verify_archive(archive_id):
                    self.logger.warning(f"发现损坏的归档文件，删除: {archive_id}")
                    if self.delete_archive(archive_id):
                        results["deleted_corrupted"] += 1
                    continue
                
                # 检查是否可以重新压缩以节省空间
                entry = self._get_archive_entry(archive_id)
                if entry and entry.compression_type != CompressionType.LZMA:
                    # 尝试使用LZMA重新压缩
                    try:
                        with open(entry.archive_path, 'rb') as f:
                            compressed_data = f.read()
                        
                        decompressed_data = self._decompress_data(compressed_data, entry.compression_type)
                        new_compressed_data, new_compressed_size = self._compress_data(
                            decompressed_data, CompressionType.LZMA
                        )
                        
                        # 如果新压缩版本更小，则替换
                        if new_compressed_size < entry.compressed_size:
                            backup_path = entry.archive_path + ".backup"
                            shutil.move(entry.archive_path, backup_path)
                            
                            with open(entry.archive_path, 'wb') as f:
                                f.write(new_compressed_data)
                            
                            # 更新数据库
                            entry.compression_type = CompressionType.LZMA
                            entry.compressed_size = new_compressed_size
                            self._save_archive_entry(entry)
                            
                            # 删除备份文件
                            os.remove(backup_path)
                            
                            results["recompressed_count"] += 1
                            results["freed_space"] += entry.compressed_size - new_compressed_size
                            
                            self.logger.info(f"重新压缩归档文件: {archive_id}")
                    
                    except Exception as e:
                        self.logger.error(f"重新压缩失败: {archive_id}, {e}")
                        # 恢复备份
                        if os.path.exists(backup_path):
                            shutil.move(backup_path, entry.archive_path)
            
            self.logger.info(f"存储优化完成: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"存储优化失败: {e}")
            return results
    
    def export_archive_metadata(self, output_file: str) -> bool:
        """导出归档元数据"""
        try:
            metadata = {
                "export_time": datetime.now().isoformat(),
                "archive_manager_version": "1.0.0",
                "archives": self.list_archives(limit=10000),  # 导出所有
                "rules": [asdict(rule) for rule in self.archive_rules],
                "stats": self.get_storage_stats()
            }
            
            # 转换枚举值为字符串
            for archive in metadata["archives"]:
                archive["compression_type"] = archive["compression_type"].value if hasattr(archive["compression_type"], 'value') else str(archive["compression_type"])
                archive["status"] = archive["status"].value if hasattr(archive["status"], 'value') else str(archive["status"])
            
            for rule in metadata["rules"]:
                rule["strategy"] = rule["strategy"].value if hasattr(rule["strategy"], 'value') else str(rule["strategy"])
                rule["compression_type"] = rule["compression_type"].value if hasattr(rule["compression_type"], 'value') else str(rule["compression_type"])
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"归档元数据导出成功: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出归档元数据失败: {e}")
            return False
    
    def import_archive_metadata(self, input_file: str) -> bool:
        """导入归档元数据"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 导入规则
            for rule_data in metadata.get("rules", []):
                rule = ArchiveRule(
                    name=rule_data["name"],
                    strategy=ArchiveStrategy(rule_data["strategy"]),
                    conditions=rule_data["conditions"],
                    compression_type=CompressionType(rule_data["compression_type"]),
                    retention_days=rule_data["retention_days"],
                    enabled=rule_data["enabled"]
                )
                self.add_archive_rule(rule)
            
            self.logger.info(f"归档元数据导入成功: {input_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"导入归档元数据失败: {e}")
            return False
    
    def run_auto_archive(self) -> Dict[str, Any]:
        """运行自动归档"""
        results = {
            "archived_files": 0,
            "total_size": 0,
            "errors": []
        }
        
        try:
            # 遍历所有规则
            for rule in self.archive_rules:
                if not rule.enabled:
                    continue
                
                # 根据规则策略查找需要归档的文件
                files_to_archive = self._find_files_for_archive(rule)
                
                for file_path in files_to_archive:
                    try:
                        archive_id = self.archive_file(str(file_path), rule.name)
                        if archive_id:
                            results["archived_files"] += 1
                            results["total_size"] += file_path.stat().st_size
                    except Exception as e:
                        error_msg = f"归档文件失败 {file_path}: {e}"
                        results["errors"].append(error_msg)
                        self.logger.error(error_msg)
            
            # 清理过期文件
            cleaned = self.cleanup_expired_archives()
            results["cleaned_expired"] = cleaned
            
            self.logger.info(f"自动归档完成: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"自动归档失败: {e}")
            results["errors"].append(str(e))
            return results
    
    def _find_files_for_archive(self, rule: ArchiveRule) -> List[Path]:
        """根据规则查找需要归档的文件"""
        files = []
        
        if rule.strategy == ArchiveStrategy.TIME_BASED:
            days = rule.conditions.get("days", 30)
            cutoff_time = time.time() - (days * 24 * 3600)
            
            # 搜索归档根目录外的文件
            for root, dirs, filenames in os.walk("/"):
                if self.archive_root in Path(root).parents:
                    continue
                
                for filename in filenames:
                    file_path = Path(root) / filename
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            files.append(file_path)
                    except (OSError, FileNotFoundError):
                        continue
        
        return files[:1000]  # 限制处理文件数量
    
    def close(self):
        """关闭归档管理器"""
        self.logger.info("归档管理器关闭")
        # 这里可以添加清理资源的代码


# 便捷函数
def create_archive_manager(archive_root: str, config_file: str = None) -> ArchiveManager:
    """创建归档管理器实例"""
    return ArchiveManager(archive_root, config_file)


if __name__ == "__main__":
    # 示例用法
    archive_manager = create_archive_manager("/tmp/r8_archive")
    
    # 添加自定义规则
    rule = ArchiveRule(
        name="大文件规则",
        strategy=ArchiveStrategy.SIZE_BASED,
        conditions={"size": 10 * 1024 * 1024},  # 10MB
        compression_type=CompressionType.LZMA,
        retention_days=730  # 2年
    )
    archive_manager.add_archive_rule(rule)
    
    print("R8归档管理器初始化完成")