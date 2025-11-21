#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R4日志备份器 - 主要实现文件

功能包括：
- 日志备份（系统日志、应用日志、交易日志）
- 日志归档（按时间、类型归档日志）
- 日志清理（自动清理过期日志）
- 日志压缩（日志文件压缩存储）
- 日志索引（快速日志检索）
- 日志分析（日志数据分析和报告）
- 日志恢复（日志数据恢复）
- 日志监控（日志备份状态监控）
"""

import os
import sys
import json
import time
import shutil
import gzip
import sqlite3
import threading
import logging
import hashlib
import zipfile
import tarfile
import glob
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile


class LogType(Enum):
    """日志类型枚举"""
    SYSTEM = "system"
    APPLICATION = "application"
    TRANSACTION = "transaction"
    ERROR = "error"
    ACCESS = "access"


class BackupStatus(Enum):
    """备份状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    RUNNING = "running"
    PENDING = "pending"


@dataclass
class LogEntry:
    """日志条目数据结构"""
    timestamp: datetime
    level: str
    message: str
    source: str
    log_type: LogType
    metadata: Dict[str, Any] = None


@dataclass
class BackupRecord:
    """备份记录数据结构"""
    id: str
    timestamp: datetime
    log_type: LogType
    source_path: str
    backup_path: str
    file_count: int
    file_size: int
    compressed_size: int
    status: BackupStatus
    checksum: str
    error_message: str = None


class LogBackup:
    """R4日志备份器主类"""
    
    def __init__(self, config_file: str = "log_backup_config.json"):
        """
        初始化日志备份器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.db_path = os.path.join(self.config['backup_dir'], 'log_backup.db')
        self.index_db_path = os.path.join(self.config['backup_dir'], 'log_index.db')
        self.logger = self._setup_logging()
        self._init_database()
        self._init_directories()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "backup_dir": "./backups",
            "log_sources": {
                "system": ["/var/log/syslog", "/var/log/messages"],
                "application": ["./logs/app.log"],
                "transaction": ["./logs/transaction.log"],
                "error": ["./logs/error.log"],
                "access": ["./logs/access.log"]
            },
            "retention_days": 30,
            "compression": True,
            "max_backup_size": 1073741824,  # 1GB
            "backup_interval": 3600,  # 1小时
            "auto_cleanup": True,
            "index_enabled": True,
            "monitoring": {
                "enabled": True,
                "alert_email": "",
                "webhook_url": ""
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并默认配置
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"加载配置文件失败，使用默认配置: {e}")
                
        # 创建默认配置文件
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"创建配置文件失败: {e}")
            
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志记录"""
        logger = logging.getLogger('LogBackup')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_dir = os.path.join(self.config['backup_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'log_backup.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_database(self):
        """初始化数据库"""
        os.makedirs(self.config['backup_dir'], exist_ok=True)
        
        # 初始化备份记录数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backup_records (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                log_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                backup_path TEXT NOT NULL,
                file_count INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                compressed_size INTEGER,
                status TEXT NOT NULL,
                checksum TEXT,
                error_message TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON backup_records(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_log_type ON backup_records(log_type)
        ''')
        
        conn.commit()
        conn.close()
        
        # 初始化日志索引数据库
        if self.config['index_enabled']:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS log_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backup_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT NOT NULL,
                    log_type TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (backup_id) REFERENCES backup_records(id)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_index_timestamp ON log_index(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_index_level ON log_index(level)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_index_log_type ON log_index(log_type)
            ''')
            
            conn.commit()
            conn.close()
    
    def _init_directories(self):
        """初始化目录结构"""
        backup_dir = self.config['backup_dir']
        
        # 创建主要目录
        directories = [
            backup_dir,
            os.path.join(backup_dir, 'system'),
            os.path.join(backup_dir, 'application'),
            os.path.join(backup_dir, 'transaction'),
            os.path.join(backup_dir, 'error'),
            os.path.join(backup_dir, 'access'),
            os.path.join(backup_dir, 'compressed'),
            os.path.join(backup_dir, 'archive'),
            os.path.join(backup_dir, 'restores')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def backup_logs(self, log_type: LogType = None) -> List[BackupRecord]:
        """
        备份日志文件
        
        Args:
            log_type: 要备份的日志类型，None表示备份所有类型
            
        Returns:
            备份记录列表
        """
        self.logger.info(f"开始备份日志，类型: {log_type.value if log_type else 'all'}")
        
        backup_records = []
        
        try:
            log_types_to_backup = [log_type] if log_type else list(LogType)
            
            for lt in log_types_to_backup:
                if lt.value not in self.config['log_sources']:
                    continue
                    
                source_paths = self.config['log_sources'][lt.value]
                backup_path = self._create_backup_path(lt)
                
                record = self._backup_log_type(lt, source_paths, backup_path)
                if record:
                    backup_records.append(record)
                    self._save_backup_record(record)
                    
                    if self.config['index_enabled']:
                        self._index_backup_logs(record)
                        
        except Exception as e:
            self.logger.error(f"备份日志时发生错误: {e}")
            
        self.logger.info(f"完成备份，共 {len(backup_records)} 个记录")
        return backup_records
    
    def _backup_log_type(self, log_type: LogType, source_paths: List[str], backup_path: str) -> Optional[BackupRecord]:
        """备份特定类型的日志"""
        backup_id = self._generate_backup_id()
        timestamp = datetime.now()
        
        try:
            # 创建备份目录
            os.makedirs(backup_path, exist_ok=True)
            
            file_count = 0
            total_size = 0
            compressed_size = 0
            
            for source_path in source_paths:
                if os.path.exists(source_path):
                    if os.path.isfile(source_path):
                        # 备份单个文件
                        dest_path = os.path.join(backup_path, os.path.basename(source_path))
                        shutil.copy2(source_path, dest_path)
                        
                        file_count += 1
                        file_size = os.path.getsize(dest_path)
                        total_size += file_size
                        
                    elif os.path.isdir(source_path):
                        # 备份目录
                        dest_path = os.path.join(backup_path, os.path.basename(source_path))
                        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                        
                        # 统计文件和大小
                        for root, dirs, files in os.walk(dest_path):
                            file_count += len(files)
                            for file in files:
                                file_path = os.path.join(root, file)
                                total_size += os.path.getsize(file_path)
            
            # 压缩备份（如果启用）
            if self.config['compression'] and file_count > 0:
                compressed_path = self._compress_backup(backup_path, log_type, timestamp)
                compressed_size = os.path.getsize(compressed_path)
                
                # 删除未压缩的备份
                shutil.rmtree(backup_path)
            
            # 计算校验和
            checksum = self._calculate_checksum(backup_path if not self.config['compression'] else compressed_path)
            
            # 创建备份记录
            record = BackupRecord(
                id=backup_id,
                timestamp=timestamp,
                log_type=log_type,
                source_path=str(source_paths),
                backup_path=backup_path if not self.config['compression'] else compressed_path,
                file_count=file_count,
                file_size=total_size,
                compressed_size=compressed_size,
                status=BackupStatus.SUCCESS,
                checksum=checksum
            )
            
            self.logger.info(f"成功备份 {log_type.value} 日志: {file_count} 个文件，{total_size} 字节")
            return record
            
        except Exception as e:
            error_msg = f"备份 {log_type.value} 日志失败: {e}"
            self.logger.error(error_msg)
            
            # 创建失败记录
            record = BackupRecord(
                id=backup_id,
                timestamp=timestamp,
                log_type=log_type,
                source_path=str(source_paths),
                backup_path=backup_path,
                file_count=0,
                file_size=0,
                compressed_size=0,
                status=BackupStatus.FAILED,
                checksum="",
                error_message=str(e)
            )
            
            return record
    
    def _create_backup_path(self, log_type: LogType) -> str:
        """创建备份路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config['backup_dir']
        return os.path.join(backup_dir, log_type.value, f"{log_type.value}_{timestamp}")
    
    def _compress_backup(self, backup_path: str, log_type: LogType, timestamp: datetime) -> str:
        """压缩备份文件"""
        compressed_dir = os.path.join(self.config['backup_dir'], 'compressed')
        os.makedirs(compressed_dir, exist_ok=True)
        
        compressed_filename = f"{log_type.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}.tar.gz"
        compressed_path = os.path.join(compressed_dir, compressed_filename)
        
        with tarfile.open(compressed_path, "w:gz") as tar:
            tar.add(backup_path, arcname=os.path.basename(backup_path))
        
        return compressed_path
    
    def _generate_backup_id(self) -> str:
        """生成备份ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"backup_{timestamp}_{random_str}"
    
    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _save_backup_record(self, record: BackupRecord):
        """保存备份记录到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backup_records 
            (id, timestamp, log_type, source_path, backup_path, file_count, 
             file_size, compressed_size, status, checksum, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.id,
            record.timestamp.isoformat(),
            record.log_type.value,
            record.source_path,
            record.backup_path,
            record.file_count,
            record.file_size,
            record.compressed_size,
            record.status.value,
            record.checksum,
            record.error_message
        ))
        
        conn.commit()
        conn.close()
    
    def _index_backup_logs(self, record: BackupRecord):
        """索引备份的日志"""
        if not self.config['index_enabled']:
            return
            
        try:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            # 解析日志文件并索引
            if os.path.isfile(record.backup_path):
                # 处理压缩文件
                if record.backup_path.endswith('.tar.gz'):
                    with tarfile.open(record.backup_path, "r:gz") as tar:
                        for member in tar.getmembers():
                            if member.isfile() and member.name.endswith('.log'):
                                content = tar.extractfile(member).read().decode('utf-8', errors='ignore')
                                self._index_log_content(cursor, record, content, member.name)
                else:
                    # 处理普通文件
                    with open(record.backup_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        self._index_log_content(cursor, record, content, os.path.basename(record.backup_path))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"索引备份日志失败: {e}")
    
    def _index_log_content(self, cursor, record: BackupRecord, content: str, filename: str):
        """索引日志内容"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 简单的日志解析（可以扩展为更复杂的解析器）
            timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', line)
            level_match = re.search(r'\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b', line, re.IGNORECASE)
            
            timestamp = timestamp_match.group() if timestamp_match else ""
            level = level_match.group().upper() if level_match else "INFO"
            
            cursor.execute('''
                INSERT INTO log_index 
                (backup_id, timestamp, level, message, source, log_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.id,
                timestamp,
                level,
                line[:500],  # 限制消息长度
                filename,
                record.log_type.value,
                json.dumps({"source_file": filename})
            ))
    
    def cleanup_old_backups(self, days: int = None) -> int:
        """
        清理过期的备份
        
        Args:
            days: 保留天数，None使用配置中的值
            
        Returns:
            清理的备份数量
        """
        retention_days = days or self.config['retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        self.logger.info(f"开始清理 {retention_days} 天前的备份")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查询过期的备份
        cursor.execute('''
            SELECT id, backup_path FROM backup_records 
            WHERE timestamp < ? AND status = ?
        ''', (cutoff_date.isoformat(), BackupStatus.SUCCESS.value))
        
        old_backups = cursor.fetchall()
        cleaned_count = 0
        
        for backup_id, backup_path in old_backups:
            try:
                # 删除备份文件
                if os.path.exists(backup_path):
                    if os.path.isfile(backup_path):
                        os.remove(backup_path)
                    else:
                        shutil.rmtree(backup_path)
                
                # 从数据库中删除记录
                cursor.execute('DELETE FROM backup_records WHERE id = ?', (backup_id,))
                
                # 删除相关的索引记录
                if self.config['index_enabled']:
                    cursor.execute('DELETE FROM log_index WHERE backup_id = ?', (backup_id,))
                
                cleaned_count += 1
                self.logger.info(f"已清理过期备份: {backup_id}")
                
            except Exception as e:
                self.logger.error(f"清理备份 {backup_id} 失败: {e}")
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"清理完成，共清理 {cleaned_count} 个备份")
        return cleaned_count
    
    def search_logs(self, query: str, log_type: LogType = None, 
                   start_time: datetime = None, end_time: datetime = None,
                   level: str = None) -> List[Dict[str, Any]]:
        """
        搜索日志
        
        Args:
            query: 搜索关键词
            log_type: 日志类型过滤
            start_time: 开始时间
            end_time: 结束时间
            level: 日志级别过滤
            
        Returns:
            搜索结果列表
        """
        if not self.config['index_enabled']:
            self.logger.warning("日志索引未启用，无法搜索")
            return []
        
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        
        # 构建查询
        sql = '''
            SELECT timestamp, level, message, source, log_type, metadata
            FROM log_index
            WHERE message LIKE ?
        '''
        params = [f"%{query}%"]
        
        if log_type:
            sql += ' AND log_type = ?'
            params.append(log_type.value)
        
        if start_time:
            sql += ' AND timestamp >= ?'
            params.append(start_time.isoformat())
        
        if end_time:
            sql += ' AND timestamp <= ?'
            params.append(end_time.isoformat())
        
        if level:
            sql += ' AND level = ?'
            params.append(level.upper())
        
        sql += ' ORDER BY timestamp DESC LIMIT 1000'
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        # 格式化结果
        search_results = []
        for row in results:
            search_results.append({
                'timestamp': row[0],
                'level': row[1],
                'message': row[2],
                'source': row[3],
                'log_type': row[4],
                'metadata': json.loads(row[5]) if row[5] else {}
            })
        
        return search_results
    
    def analyze_logs(self, start_time: datetime = None, end_time: datetime = None) -> Dict[str, Any]:
        """
        分析日志数据
        
        Args:
            start_time: 分析开始时间
            end_time: 分析结束时间
            
        Returns:
            分析结果
        """
        if not self.config['index_enabled']:
            self.logger.warning("日志索引未启用，无法分析")
            return {}
        
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        
        # 设置时间范围
        time_condition = ""
        params = []
        
        if start_time:
            time_condition += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            time_condition += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        # 统计日志级别分布
        cursor.execute(f'''
            SELECT level, COUNT(*) as count
            FROM log_index
            WHERE 1=1 {time_condition}
            GROUP BY level
        ''', params)
        level_stats = dict(cursor.fetchall())
        
        # 统计日志类型分布
        cursor.execute(f'''
            SELECT log_type, COUNT(*) as count
            FROM log_index
            WHERE 1=1 {time_condition}
            GROUP BY log_type
        ''', params)
        type_stats = dict(cursor.fetchall())
        
        # 统计每日日志数量
        cursor.execute(f'''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM log_index
            WHERE 1=1 {time_condition}
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''', params)
        daily_stats = dict(cursor.fetchall())
        
        # 查找错误模式
        cursor.execute(f'''
            SELECT message, COUNT(*) as count
            FROM log_index
            WHERE level = 'ERROR' {time_condition}
            GROUP BY message
            ORDER BY count DESC
            LIMIT 10
        ''', params)
        error_patterns = cursor.fetchall()
        
        conn.close()
        
        analysis_result = {
            'time_range': {
                'start': start_time.isoformat() if start_time else None,
                'end': end_time.isoformat() if end_time else None
            },
            'level_distribution': level_stats,
            'type_distribution': type_stats,
            'daily_counts': daily_stats,
            'top_error_patterns': [{'pattern': pattern, 'count': count} for pattern, count in error_patterns],
            'total_logs': sum(level_stats.values()) if level_stats else 0
        }
        
        return analysis_result
    
    def restore_logs(self, backup_id: str, restore_path: str) -> bool:
        """
        恢复日志备份
        
        Args:
            backup_id: 备份ID
            restore_path: 恢复路径
            
        Returns:
            恢复是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查询备份信息
        cursor.execute('''
            SELECT backup_path, log_type FROM backup_records 
            WHERE id = ? AND status = ?
        ''', (backup_id, BackupStatus.SUCCESS.value))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            self.logger.error(f"备份记录不存在: {backup_id}")
            return False
        
        backup_path, log_type = result
        
        try:
            # 创建恢复目录
            os.makedirs(restore_path, exist_ok=True)
            
            # 解压或复制备份文件
            if backup_path.endswith('.tar.gz'):
                with tarfile.open(backup_path, "r:gz") as tar:
                    tar.extractall(restore_path)
            else:
                shutil.copytree(backup_path, os.path.join(restore_path, log_type))
            
            self.logger.info(f"成功恢复备份 {backup_id} 到 {restore_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复备份 {backup_id} 失败: {e}")
            return False
    
    def get_backup_status(self) -> Dict[str, Any]:
        """
        获取备份状态
        
        Returns:
            备份状态信息
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 统计总体情况
        cursor.execute('SELECT COUNT(*) FROM backup_records')
        total_backups = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM backup_records WHERE status = ?', (BackupStatus.SUCCESS.value,))
        successful_backups = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM backup_records WHERE status = ?', (BackupStatus.FAILED.value,))
        failed_backups = cursor.fetchone()[0]
        
        # 最近备份
        cursor.execute('''
            SELECT * FROM backup_records 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''')
        recent_backups = cursor.fetchall()
        
        # 磁盘使用情况
        backup_dir = self.config['backup_dir']
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(backup_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except OSError:
                    pass
        
        conn.close()
        
        status_info = {
            'total_backups': total_backups,
            'successful_backups': successful_backups,
            'failed_backups': failed_backups,
            'success_rate': (successful_backups / total_backups * 100) if total_backups > 0 else 0,
            'disk_usage': {
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'file_count': file_count
            },
            'recent_backups': [
                {
                    'id': row[0],
                    'timestamp': row[1],
                    'log_type': row[2],
                    'status': row[8],
                    'file_count': row[5]
                } for row in recent_backups
            ]
        }
        
        return status_info
    
    def start_monitoring(self):
        """启动监控"""
        if not self.config['monitoring']['enabled']:
            return
        
        def monitor_task():
            while True:
                try:
                    status = self.get_backup_status()
                    
                    # 检查失败率
                    if status['success_rate'] < 90 and status['total_backups'] > 10:
                        self._send_alert(f"备份成功率过低: {status['success_rate']:.1f}%")
                    
                    # 检查磁盘使用
                    if status['disk_usage']['total_size_mb'] > 10240:  # 10GB
                        self._send_alert(f"备份目录磁盘使用过高: {status['disk_usage']['total_size_mb']:.1f}MB")
                    
                    time.sleep(300)  # 5分钟检查一次
                    
                except Exception as e:
                    self.logger.error(f"监控任务异常: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_task, daemon=True)
        monitor_thread.start()
        self.logger.info("日志备份监控已启动")
    
    def _send_alert(self, message: str):
        """发送告警"""
        self.logger.warning(f"告警: {message}")
        
        # 这里可以添加邮件、webhook等告警方式
        if self.config['monitoring']['alert_email']:
            # 发送邮件告警
            pass
        
        if self.config['monitoring']['webhook_url']:
            # 发送webhook告警
            pass
    
    def schedule_backup(self, interval: int = None):
        """
        定时备份
        
        Args:
            interval: 备份间隔（秒），None使用配置中的值
        """
        backup_interval = interval or self.config['backup_interval']
        
        def backup_task():
            while True:
                try:
                    self.logger.info("开始定时备份")
                    self.backup_logs()
                    time.sleep(backup_interval)
                except Exception as e:
                    self.logger.error(f"定时备份异常: {e}")
                    time.sleep(60)
        
        backup_thread = threading.Thread(target=backup_task, daemon=True)
        backup_thread.start()
        self.logger.info(f"定时备份已启动，间隔: {backup_interval}秒")
    
    def export_backup_report(self, output_file: str):
        """
        导出备份报告
        
        Args:
            output_file: 输出文件路径
        """
        status = self.get_backup_status()
        analysis = self.analyze_logs()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'backup_status': status,
            'log_analysis': analysis
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"备份报告已导出到: {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='R4日志备份器')
    parser.add_argument('command', choices=['backup', 'cleanup', 'search', 'analyze', 'restore', 'status', 'report'], 
                       help='要执行的命令')
    parser.add_argument('--log-type', type=str, choices=[lt.value for lt in LogType], 
                       help='日志类型')
    parser.add_argument('--query', type=str, help='搜索关键词')
    parser.add_argument('--days', type=int, help='保留天数')
    parser.add_argument('--backup-id', type=str, help='备份ID')
    parser.add_argument('--restore-path', type=str, help='恢复路径')
    parser.add_argument('--output', type=str, help='输出文件')
    
    args = parser.parse_args()
    
    backup = LogBackup()
    
    if args.command == 'backup':
        log_type = LogType(args.log_type) if args.log_type else None
        records = backup.backup_logs(log_type)
        print(f"备份完成，共 {len(records)} 个记录")
        
    elif args.command == 'cleanup':
        count = backup.cleanup_old_backups(args.days)
        print(f"清理完成，共清理 {count} 个备份")
        
    elif args.command == 'search':
        if not args.query:
            print("请提供搜索关键词")
            return
        results = backup.search_logs(args.query)
        print(f"找到 {len(results)} 条匹配结果")
        for result in results[:10]:  # 显示前10条
            print(f"[{result['timestamp']}] {result['level']}: {result['message'][:100]}...")
            
    elif args.command == 'analyze':
        analysis = backup.analyze_logs()
        print("日志分析结果:")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        
    elif args.command == 'restore':
        if not args.backup_id or not args.restore_path:
            print("请提供备份ID和恢复路径")
            return
        success = backup.restore_logs(args.backup_id, args.restore_path)
        print(f"恢复{'成功' if success else '失败'}")
        
    elif args.command == 'status':
        status = backup.get_backup_status()
        print("备份状态:")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
    elif args.command == 'report':
        if not args.output:
            args.output = f"backup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup.export_backup_report(args.output)
        print(f"报告已导出到: {args.output}")


if __name__ == "__main__":
    main()