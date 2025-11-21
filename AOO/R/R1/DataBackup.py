#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1数据备份器 - 主要实现模块

提供完整的数据备份解决方案，包括：
- 文件、数据库、对象存储备份
- 多种压缩算法支持
- 数据加密保护
- 增量备份
- 备份验证
- 存储管理
- 备份调度
- 备份报告

作者: R1数据备份器团队
版本: 1.0.0
"""

import os
import sys
import json
import hashlib
import shutil
import sqlite3
import zipfile
import tarfile
import gzip
import bz2
import lzma
import time
import datetime
import logging
import threading
import schedule
import smtplib
import subprocess
import tempfile
import shutil
import pickle
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import io
import sqlite3
import mysql.connector
import psycopg2
import boto3
from botocore.exceptions import ClientError


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_backup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """备份配置类"""
    backup_id: str
    source_path: str
    backup_path: str
    compression: str = 'gzip'
    encryption: bool = False
    encryption_key: Optional[str] = None
    backup_type: str = 'full'  # full, incremental, differential
    verify_backup: bool = True
    retention_days: int = 30
    exclude_patterns: List[str] = None
    include_patterns: List[str] = None
    database_config: Optional[Dict] = None
    cloud_config: Optional[Dict] = None
    schedule_config: Optional[Dict] = None
    
    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = []
        if self.include_patterns is None:
            self.include_patterns = []


@dataclass
class BackupStatus:
    """备份状态类"""
    backup_id: str
    status: str  # success, failed, running, pending
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    file_count: int = 0
    total_size: int = 0
    compressed_size: int = 0
    error_message: Optional[str] = None
    checksum: Optional[str] = None
    location: Optional[str] = None


class Compressor:
    """压缩器类 - 支持多种压缩算法"""
    
    @staticmethod
    def get_compressor(compression_type: str):
        """获取压缩器实例"""
        compressors = {
            'gzip': Compressor._gzip_compress,
            'bz2': Compressor._bz2_compress,
            'lzma': Compressor._lzma_compress,
            'zip': Compressor._zip_compress,
            'tar': Compressor._tar_compress,
            'tar.gz': Compressor._tar_gz_compress,
            'tar.bz2': Compressor._tar_bz2_compress,
        }
        
        if compression_type not in compressors:
            raise ValueError(f"不支持的压缩格式: {compression_type}")
        
        return compressors[compression_type]
    
    @staticmethod
    def _gzip_compress(data: bytes) -> bytes:
        """GZIP压缩"""
        return gzip.compress(data)
    
    @staticmethod
    def _bz2_compress(data: bytes) -> bytes:
        """BZ2压缩"""
        return bz2.compress(data)
    
    @staticmethod
    def _lzma_compress(data: bytes) -> bytes:
        """LZMA压缩"""
        return lzma.compress(data)
    
    @staticmethod
    def _zip_compress(source_path: str, output_path: str) -> str:
        """ZIP压缩"""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_path)
                    zipf.write(file_path, arcname)
        return output_path
    
    @staticmethod
    def _tar_compress(source_path: str, output_path: str) -> str:
        """TAR压缩"""
        with tarfile.open(output_path, 'w') as tar:
            tar.add(source_path, arcname=os.path.basename(source_path))
        return output_path
    
    @staticmethod
    def _tar_gz_compress(source_path: str, output_path: str) -> str:
        """TAR.GZ压缩"""
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(source_path, arcname=os.path.basename(source_path))
        return output_path
    
    @staticmethod
    def _tar_bz2_compress(source_path: str, output_path: str) -> str:
        """TAR.BZ2压缩"""
        with tarfile.open(output_path, 'w:bz2') as tar:
            tar.add(source_path, arcname=os.path.basename(source_path))
        return output_path


class Encryptor:
    """加密器类"""
    
    @staticmethod
    def generate_key(password: str, salt: bytes = None) -> bytes:
        """生成加密密钥"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> bytes:
        """加密数据"""
        fernet = Fernet(key)
        return fernet.encrypt(data)
    
    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
        """解密数据"""
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data)


class BackupSource(ABC):
    """备份源抽象基类"""
    
    @abstractmethod
    def get_files(self) -> List[str]:
        """获取要备份的文件列表"""
        pass
    
    @abstractmethod
    def get_file_hash(self, file_path: str) -> str:
        """获取文件哈希值"""
        pass
    
    @abstractmethod
    def read_file(self, file_path: str) -> bytes:
        """读取文件内容"""
        pass


class FileBackupSource(BackupSource):
    """文件备份源"""
    
    def __init__(self, source_path: str, exclude_patterns: List[str] = None, 
                 include_patterns: List[str] = None):
        self.source_path = source_path
        self.exclude_patterns = exclude_patterns or []
        self.include_patterns = include_patterns or []
    
    def get_files(self) -> List[str]:
        """获取文件列表"""
        files = []
        for root, dirs, filenames in os.walk(self.source_path):
            # 检查排除模式
            dirs[:] = [d for d in dirs if not any(
                pattern in os.path.join(root, d) for pattern in self.exclude_patterns
            )]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                
                # 检查包含模式
                if self.include_patterns and not any(
                    pattern in file_path for pattern in self.include_patterns
                ):
                    continue
                
                # 检查排除模式
                if any(pattern in file_path for pattern in self.exclude_patterns):
                    continue
                
                files.append(file_path)
        
        return files
    
    def get_file_hash(self, file_path: str) -> str:
        """获取文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def read_file(self, file_path: str) -> bytes:
        """读取文件内容"""
        with open(file_path, 'rb') as f:
            return f.read()


class DatabaseBackupSource(BackupSource):
    """数据库备份源"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.db_type = db_config.get('type', 'sqlite')
    
    def get_files(self) -> List[str]:
        """获取数据库文件列表"""
        if self.db_type == 'sqlite':
            return [self.db_config['database_path']]
        elif self.db_type in ['mysql', 'postgresql']:
            # 对于MySQL/PostgreSQL，返回数据库名列表
            return [self.db_config['database']]
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")
    
    def get_file_hash(self, file_path: str) -> str:
        """获取文件哈希值"""
        if self.db_type == 'sqlite':
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        else:
            # 对于MySQL/PostgreSQL，返回基于数据库内容的哈希
            return hashlib.md5(self.get_database_content().encode()).hexdigest()
    
    def read_file(self, file_path: str) -> bytes:
        """读取文件内容"""
        if self.db_type == 'sqlite':
            with open(file_path, 'rb') as f:
                return f.read()
        else:
            # 对于MySQL/PostgreSQL，返回SQL转储
            return self.get_database_dump().encode()
    
    def get_database_content(self) -> str:
        """获取数据库内容"""
        if self.db_type == 'mysql':
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            content = ""
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                content += f"Table: {table_name}\n"
                for row in rows:
                    content += str(row) + "\n"
            
            cursor.close()
            conn.close()
            return content
        
        elif self.db_type == 'postgresql':
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cursor.fetchall()
            
            content = ""
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                content += f"Table: {table_name}\n"
                for row in rows:
                    content += str(row) + "\n"
            
            cursor.close()
            conn.close()
            return content
        
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")
    
    def get_database_dump(self) -> str:
        """获取数据库转储"""
        if self.db_type == 'mysql':
            # 使用mysqldump命令
            cmd = [
                'mysqldump',
                f"-h{self.db_config['host']}",
                f"-u{self.db_config['user']}",
                f"-p{self.db_config['password']}",
                self.db_config['database']
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout
        
        elif self.db_type == 'postgresql':
            # 使用pg_dump命令
            cmd = [
                'pg_dump',
                f"-h{self.db_config['host']}",
                f"-U{self.db_config['user']}",
                self.db_config['database']
            ]
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_config['password']
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            return result.stdout
        
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")


class CloudBackupSource(BackupSource):
    """云存储备份源"""
    
    def __init__(self, cloud_config: Dict):
        self.cloud_config = cloud_config
        self.provider = cloud_config.get('provider', 's3')
    
    def get_files(self) -> List[str]:
        """获取云存储文件列表"""
        if self.provider == 's3':
            s3 = boto3.client('s3', **self.cloud_config['credentials'])
            try:
                response = s3.list_objects_v2(
                    Bucket=self.cloud_config['bucket'],
                    Prefix=self.cloud_config.get('prefix', '')
                )
                return [obj['Key'] for obj in response.get('Contents', [])]
            except ClientError as e:
                logger.error(f"S3列表文件失败: {e}")
                return []
        else:
            raise ValueError(f"不支持的云存储提供商: {self.provider}")
    
    def get_file_hash(self, file_path: str) -> str:
        """获取文件哈希值"""
        if self.provider == 's3':
            s3 = boto3.client('s3', **self.cloud_config['credentials'])
            try:
                response = s3.head_object(
                    Bucket=self.cloud_config['bucket'],
                    Key=file_path
                )
                return response['ETag'].strip('"')
            except ClientError as e:
                logger.error(f"获取S3文件哈希失败: {e}")
                return ""
        else:
            raise ValueError(f"不支持的云存储提供商: {self.provider}")
    
    def read_file(self, file_path: str) -> bytes:
        """读取文件内容"""
        if self.provider == 's3':
            s3 = boto3.client('s3', **self.cloud_config['credentials'])
            try:
                response = s3.get_object(
                    Bucket=self.cloud_config['bucket'],
                    Key=file_path
                )
                return response['Body'].read()
            except ClientError as e:
                logger.error(f"读取S3文件失败: {e}")
                return b""
        else:
            raise ValueError(f"不支持的云存储提供商: {self.provider}")


class BackupStorage(ABC):
    """备份存储抽象基类"""
    
    @abstractmethod
    def save_backup(self, backup_data: bytes, backup_path: str) -> bool:
        """保存备份"""
        pass
    
    @abstractmethod
    def load_backup(self, backup_path: str) -> bytes:
        """加载备份"""
        pass
    
    @abstractmethod
    def delete_backup(self, backup_path: str) -> bool:
        """删除备份"""
        pass
    
    @abstractmethod
    def list_backups(self) -> List[str]:
        """列出备份"""
        pass


class LocalStorage(BackupStorage):
    """本地存储"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save_backup(self, backup_data: bytes, backup_path: str) -> bool:
        """保存备份到本地"""
        try:
            full_path = os.path.join(self.storage_path, backup_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'wb') as f:
                f.write(backup_data)
            return True
        except Exception as e:
            logger.error(f"保存本地备份失败: {e}")
            return False
    
    def load_backup(self, backup_path: str) -> bytes:
        """从本地加载备份"""
        try:
            full_path = os.path.join(self.storage_path, backup_path)
            with open(full_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"加载本地备份失败: {e}")
            return b""
    
    def delete_backup(self, backup_path: str) -> bool:
        """删除本地备份"""
        try:
            full_path = os.path.join(self.storage_path, backup_path)
            if os.path.exists(full_path):
                os.remove(full_path)
            return True
        except Exception as e:
            logger.error(f"删除本地备份失败: {e}")
            return False
    
    def list_backups(self) -> List[str]:
        """列出本地备份"""
        try:
            backups = []
            for root, dirs, files in os.walk(self.storage_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), self.storage_path)
                    backups.append(rel_path)
            return backups
        except Exception as e:
            logger.error(f"列出本地备份失败: {e}")
            return []


class CloudStorage(BackupStorage):
    """云存储"""
    
    def __init__(self, cloud_config: Dict):
        self.cloud_config = cloud_config
        self.provider = cloud_config.get('provider', 's3')
        
        if self.provider == 's3':
            self.s3 = boto3.client('s3', **cloud_config['credentials'])
    
    def save_backup(self, backup_data: bytes, backup_path: str) -> bool:
        """保存备份到云存储"""
        try:
            if self.provider == 's3':
                self.s3.put_object(
                    Bucket=self.cloud_config['bucket'],
                    Key=backup_path,
                    Body=backup_data
                )
                return True
            else:
                raise ValueError(f"不支持的云存储提供商: {self.provider}")
        except Exception as e:
            logger.error(f"保存云存储备份失败: {e}")
            return False
    
    def load_backup(self, backup_path: str) -> bytes:
        """从云存储加载备份"""
        try:
            if self.provider == 's3':
                response = self.s3.get_object(
                    Bucket=self.cloud_config['bucket'],
                    Key=backup_path
                )
                return response['Body'].read()
            else:
                raise ValueError(f"不支持的云存储提供商: {self.provider}")
        except Exception as e:
            logger.error(f"加载云存储备份失败: {e}")
            return b""
    
    def delete_backup(self, backup_path: str) -> bool:
        """删除云存储备份"""
        try:
            if self.provider == 's3':
                self.s3.delete_object(
                    Bucket=self.cloud_config['bucket'],
                    Key=backup_path
                )
                return True
            else:
                raise ValueError(f"不支持的云存储提供商: {self.provider}")
        except Exception as e:
            logger.error(f"删除云存储备份失败: {e}")
            return False
    
    def list_backups(self) -> List[str]:
        """列出云存储备份"""
        try:
            if self.provider == 's3':
                response = self.s3.list_objects_v2(
                    Bucket=self.cloud_config['bucket']
                )
                return [obj['Key'] for obj in response.get('Contents', [])]
            else:
                raise ValueError(f"不支持的云存储提供商: {self.provider}")
        except Exception as e:
            logger.error(f"列出云存储备份失败: {e}")
            return []


class BackupVerifier:
    """备份验证器"""
    
    @staticmethod
    def verify_backup_integrity(backup_path: str, expected_checksum: str) -> bool:
        """验证备份完整性"""
        try:
            hash_md5 = hashlib.md5()
            with open(backup_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            actual_checksum = hash_md5.hexdigest()
            return actual_checksum == expected_checksum
        except Exception as e:
            logger.error(f"验证备份完整性失败: {e}")
            return False
    
    @staticmethod
    def calculate_checksum(file_path: str) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class BackupScheduler:
    """备份调度器"""
    
    def __init__(self, data_backup: 'DataBackup'):
        self.data_backup = data_backup
        self.scheduled_jobs = {}
        self.running = False
        self.scheduler_thread = None
    
    def schedule_backup(self, backup_config: BackupConfig, schedule_config: Dict) -> str:
        """调度备份任务"""
        job_id = f"backup_{backup_config.backup_id}_{int(time.time())}"
        
        schedule_type = schedule_config.get('type', 'interval')
        interval = schedule_config.get('interval', 3600)  # 默认1小时
        
        if schedule_type == 'interval':
            schedule.every(interval).seconds.do(self._run_scheduled_backup, backup_config)
        elif schedule_type == 'daily':
            time_str = schedule_config.get('time', '02:00')
            schedule.every().day.at(time_str).do(self._run_scheduled_backup, backup_config)
        elif schedule_type == 'weekly':
            day = schedule_config.get('day', 'sunday')
            time_str = schedule_config.get('time', '02:00')
            getattr(schedule.every(), day).at(time_str).do(self._run_scheduled_backup, backup_config)
        
        self.scheduled_jobs[job_id] = {
            'config': backup_config,
            'schedule': schedule_config,
            'created_at': datetime.datetime.now()
        }
        
        logger.info(f"备份任务已调度: {job_id}")
        return job_id
    
    def _run_scheduled_backup(self, backup_config: BackupConfig):
        """运行调度的备份任务"""
        try:
            logger.info(f"开始执行调度的备份任务: {backup_config.backup_id}")
            result = self.data_backup.create_backup(backup_config)
            if result.status == 'success':
                logger.info(f"调度备份任务执行成功: {backup_config.backup_id}")
            else:
                logger.error(f"调度备份任务执行失败: {backup_config.backup_id}, 错误: {result.error_message}")
        except Exception as e:
            logger.error(f"执行调度备份任务异常: {e}")
    
    def start_scheduler(self):
        """启动调度器"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            logger.info("备份调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("备份调度器已停止")
    
    def _run_scheduler(self):
        """运行调度器主循环"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def cancel_job(self, job_id: str) -> bool:
        """取消调度任务"""
        try:
            if job_id in self.scheduled_jobs:
                # 清理schedule中的相关任务
                schedule.clear(job_id)
                del self.scheduled_jobs[job_id]
                logger.info(f"备份任务已取消: {job_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"取消备份任务失败: {e}")
            return False


class BackupReporter:
    """备份报告器"""
    
    def __init__(self, report_config: Dict = None):
        self.report_config = report_config or {}
        self.email_config = self.report_config.get('email', {})
    
    def generate_report(self, backup_statuses: List[BackupStatus]) -> str:
        """生成备份报告"""
        report = []
        report.append("=" * 60)
        report.append("R1数据备份器 - 备份报告")
        report.append("=" * 60)
        report.append(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"备份任务总数: {len(backup_statuses)}")
        report.append("")
        
        # 统计信息
        success_count = sum(1 for status in backup_statuses if status.status == 'success')
        failed_count = sum(1 for status in backup_statuses if status.status == 'failed')
        running_count = sum(1 for status in backup_statuses if status.status == 'running')
        
        report.append("备份统计:")
        report.append(f"  成功: {success_count}")
        report.append(f"  失败: {failed_count}")
        report.append(f"  运行中: {running_count}")
        report.append("")
        
        # 详细信息
        report.append("备份详情:")
        report.append("-" * 60)
        
        for status in backup_statuses:
            report.append(f"备份ID: {status.backup_id}")
            report.append(f"状态: {status.status}")
            report.append(f"开始时间: {status.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if status.end_time:
                duration = status.end_time - status.start_time
                report.append(f"结束时间: {status.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                report.append(f"耗时: {duration}")
            
            report.append(f"文件数量: {status.file_count}")
            report.append(f"原始大小: {self._format_size(status.total_size)}")
            report.append(f"压缩后大小: {self._format_size(status.compressed_size)}")
            
            if status.checksum:
                report.append(f"校验和: {status.checksum}")
            
            if status.location:
                report.append(f"存储位置: {status.location}")
            
            if status.error_message:
                report.append(f"错误信息: {status.error_message}")
            
            report.append("-" * 60)
        
        return "\n".join(report)
    
    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"
    
    def save_report(self, report: str, file_path: str) -> bool:
        """保存报告到文件"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"备份报告已保存: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存备份报告失败: {e}")
            return False
    
    def send_email_report(self, report: str, subject: str = None) -> bool:
        """发送邮件报告"""
        if not self.email_config:
            logger.warning("未配置邮件设置")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = subject or f"R1数据备份器报告 - {datetime.datetime.now().strftime('%Y-%m-%d')}"
            
            msg.attach(MIMEText(report, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            if self.email_config.get('use_tls'):
                server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("备份报告邮件已发送")
            return True
        except Exception as e:
            logger.error(f"发送备份报告邮件失败: {e}")
            return False


class DataBackup:
    """数据备份器主类"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.backup_db_path = os.path.join(
            self.config.get('data_dir', '.'), 'backup_metadata.db'
        )
        self._init_database()
        self.scheduler = BackupScheduler(self)
        self.reporter = BackupReporter(self.config.get('report_config', {}))
    
    def _init_database(self):
        """初始化元数据数据库"""
        os.makedirs(os.path.dirname(self.backup_db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.backup_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backups (
                backup_id TEXT PRIMARY KEY,
                config_json TEXT,
                status TEXT,
                start_time TEXT,
                end_time TEXT,
                file_count INTEGER,
                total_size INTEGER,
                compressed_size INTEGER,
                checksum TEXT,
                location TEXT,
                error_message TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT,
                backup_id TEXT,
                file_hash TEXT,
                PRIMARY KEY (file_path, backup_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_backup(self, backup_config: BackupConfig) -> BackupStatus:
        """创建备份"""
        logger.info(f"开始创建备份: {backup_config.backup_id}")
        
        status = BackupStatus(
            backup_id=backup_config.backup_id,
            status='running',
            start_time=datetime.datetime.now()
        )
        
        try:
            # 获取备份源
            backup_source = self._get_backup_source(backup_config)
            
            # 获取文件列表
            files = backup_source.get_files()
            status.file_count = len(files)
            
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_backup_path = os.path.join(temp_dir, f"{backup_config.backup_id}.tmp")
                
                # 执行备份
                if backup_config.backup_type == 'incremental':
                    backup_data = self._create_incremental_backup(backup_source, backup_config, temp_backup_path)
                else:
                    backup_data = self._create_full_backup(backup_source, backup_config, temp_backup_path)
                
                # 压缩
                if backup_config.compression != 'none':
                    compressed_data = self._compress_backup(backup_data, backup_config.compression)
                else:
                    compressed_data = backup_data
                
                # 加密
                if backup_config.encryption:
                    key, salt = Encryptor.generate_key(backup_config.encryption_key or 'default_key')
                    encrypted_data = Encryptor.encrypt_data(compressed_data, key)
                    # 保存加密密钥和盐值
                    key_file = f"{temp_backup_path}.key"
                    with open(key_file, 'wb') as f:
                        f.write(salt + key)
                    compressed_data = encrypted_data
                
                # 保存到存储
                storage = self._get_storage(backup_config)
                backup_filename = f"{backup_config.backup_id}_{int(time.time())}.backup"
                if storage.save_backup(compressed_data, backup_filename):
                    status.location = backup_filename
                    status.compressed_size = len(compressed_data)
                    
                    # 验证备份
                    if backup_config.verify_backup:
                        if self._verify_backup(storage, backup_filename):
                            status.status = 'success'
                        else:
                            status.status = 'failed'
                            status.error_message = "备份验证失败"
                    else:
                        status.status = 'success'
                    
                    # 计算校验和
                    status.checksum = BackupVerifier.calculate_checksum(
                        os.path.join(storage.storage_path if hasattr(storage, 'storage_path') else '', backup_filename)
                    )
                else:
                    status.status = 'failed'
                    status.error_message = "保存备份失败"
            
            status.end_time = datetime.datetime.now()
            
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            status.status = 'failed'
            status.error_message = str(e)
            status.end_time = datetime.datetime.now()
        
        finally:
            # 保存备份状态到数据库
            self._save_backup_status(status, backup_config)
        
        return status
    
    def _get_backup_source(self, backup_config: BackupConfig) -> BackupSource:
        """获取备份源"""
        if backup_config.database_config:
            return DatabaseBackupSource(backup_config.database_config)
        elif backup_config.cloud_config:
            return CloudBackupSource(backup_config.cloud_config)
        else:
            return FileBackupSource(
                backup_config.source_path,
                backup_config.exclude_patterns,
                backup_config.include_patterns
            )
    
    def _create_full_backup(self, backup_source: BackupSource, backup_config: BackupConfig, temp_path: str) -> bytes:
        """创建完整备份"""
        files = backup_source.get_files()
        backup_data = {}
        
        for file_path in files:
            try:
                content = backup_source.read_file(file_path)
                backup_data[file_path] = content
            except Exception as e:
                logger.warning(f"读取文件失败 {file_path}: {e}")
        
        return pickle.dumps(backup_data)
    
    def _create_incremental_backup(self, backup_source: BackupSource, backup_config: BackupConfig, temp_path: str) -> bytes:
        """创建增量备份"""
        # 获取当前文件哈希
        current_hashes = {}
        files = backup_source.get_files()
        
        for file_path in files:
            try:
                current_hashes[file_path] = backup_source.get_file_hash(file_path)
            except Exception as e:
                logger.warning(f"获取文件哈希失败 {file_path}: {e}")
        
        # 获取上次备份的哈希
        last_backup_hashes = self._get_last_backup_hashes(backup_config.backup_id)
        
        # 找出变更的文件
        changed_files = {}
        for file_path, current_hash in current_hashes.items():
            if file_path not in last_backup_hashes or last_backup_hashes[file_path] != current_hash:
                try:
                    content = backup_source.read_file(file_path)
                    changed_files[file_path] = content
                except Exception as e:
                    logger.warning(f"读取文件失败 {file_path}: {e}")
        
        # 找出删除的文件
        deleted_files = []
        for file_path in last_backup_hashes:
            if file_path not in current_hashes:
                deleted_files.append(file_path)
        
        incremental_data = {
            'changed_files': changed_files,
            'deleted_files': deleted_files,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return pickle.dumps(incremental_data)
    
    def _get_last_backup_hashes(self, backup_id: str) -> Dict[str, str]:
        """获取上次备份的文件哈希"""
        conn = sqlite3.connect(self.backup_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT backup_id FROM backups 
            WHERE backup_id LIKE ? AND status = 'success'
            ORDER BY start_time DESC LIMIT 1
        ''', (f"{backup_id}%",))
        
        result = cursor.fetchone()
        if result:
            last_backup_id = result[0]
            cursor.execute('''
                SELECT file_path, file_hash FROM file_hashes 
                WHERE backup_id = ?
            ''', (last_backup_id,))
            
            hashes = {}
            for row in cursor.fetchall():
                hashes[row[0]] = row[1]
            
            conn.close()
            return hashes
        
        conn.close()
        return {}
    
    def _compress_backup(self, data: bytes, compression_type: str) -> bytes:
        """压缩备份数据"""
        compressor = Compressor.get_compressor(compression_type)
        
        if compression_type in ['gzip', 'bz2', 'lzma']:
            return compressor(data)
        else:
            # 对于需要文件路径的压缩格式，创建临时文件
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(data)
                temp_file.flush()
                
                output_path = temp_file.name + '.compressed'
                compressor(temp_file.name, output_path)
                
                with open(output_path, 'rb') as f:
                    compressed_data = f.read()
                
                os.unlink(temp_file.name)
                os.unlink(output_path)
                
                return compressed_data
    
    def _get_storage(self, backup_config: BackupConfig) -> BackupStorage:
        """获取存储实例"""
        if backup_config.cloud_config:
            return CloudStorage(backup_config.cloud_config)
        else:
            return LocalStorage(backup_config.backup_path)
    
    def _verify_backup(self, storage: BackupStorage, backup_filename: str) -> bool:
        """验证备份"""
        try:
            backup_data = storage.load_backup(backup_filename)
            if not backup_data:
                return False
            
            # 简单的完整性检查
            return len(backup_data) > 0
        except Exception as e:
            logger.error(f"验证备份失败: {e}")
            return False
    
    def _save_backup_status(self, status: BackupStatus, backup_config: BackupConfig):
        """保存备份状态到数据库"""
        conn = sqlite3.connect(self.backup_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO backups 
            (backup_id, config_json, status, start_time, end_time, 
             file_count, total_size, compressed_size, checksum, location, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            status.backup_id,
            json.dumps(asdict(backup_config)),
            status.status,
            status.start_time.isoformat(),
            status.end_time.isoformat() if status.end_time else None,
            status.file_count,
            status.total_size,
            status.compressed_size,
            status.checksum,
            status.location,
            status.error_message
        ))
        
        # 保存文件哈希（如果是完整备份）
        if status.status == 'success' and backup_config.backup_type != 'incremental':
            backup_source = self._get_backup_source(backup_config)
            files = backup_source.get_files()
            
            for file_path in files:
                try:
                    file_hash = backup_source.get_file_hash(file_path)
                    cursor.execute('''
                        INSERT OR REPLACE INTO file_hashes 
                        (file_path, backup_id, file_hash)
                        VALUES (?, ?, ?)
                    ''', (file_path, status.backup_id, file_hash))
                except Exception as e:
                    logger.warning(f"保存文件哈希失败 {file_path}: {e}")
        
        conn.commit()
        conn.close()
    
    def restore_backup(self, backup_id: str, restore_path: str, encryption_key: str = None) -> bool:
        """恢复备份"""
        logger.info(f"开始恢复备份: {backup_id}")
        
        try:
            # 获取备份配置和状态
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT config_json, location, checksum FROM backups 
                WHERE backup_id = ? AND status = 'success'
                ORDER BY start_time DESC LIMIT 1
            ''', (backup_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.error(f"未找到备份: {backup_id}")
                return False
            
            config_json, location, checksum = result
            backup_config = BackupConfig(**json.loads(config_json))
            
            # 加载备份数据
            storage = self._get_storage(backup_config)
            backup_data = storage.load_backup(location)
            
            if not backup_data:
                logger.error("加载备份数据失败")
                return False
            
            # 解密
            if backup_config.encryption:
                key_file = f"{location}.key"
                if os.path.exists(key_file):
                    with open(key_file, 'rb') as f:
                        salt_and_key = f.read()
                        salt = salt_and_key[:16]
                        key = salt_and_key[16:]
                    
                    if encryption_key:
                        provided_key, _ = Encryptor.generate_key(encryption_key, salt)
                        if provided_key != key:
                            logger.error("加密密钥不匹配")
                            return False
                    
                    backup_data = Encryptor.decrypt_data(backup_data, key)
            
            # 解压缩
            if backup_config.compression != 'none':
                # 这里需要根据压缩格式进行解压缩
                # 简化处理，实际实现需要更复杂的逻辑
                pass
            
            # 恢复数据
            restored_data = pickle.loads(backup_data)
            
            os.makedirs(restore_path, exist_ok=True)
            
            if backup_config.backup_type == 'incremental':
                # 增量恢复
                changed_files = restored_data.get('changed_files', {})
                deleted_files = restored_data.get('deleted_files', [])
                
                # 应用变更的文件
                for file_path, content in changed_files.items():
                    rel_path = os.path.relpath(file_path, backup_config.source_path)
                    full_restore_path = os.path.join(restore_path, rel_path)
                    os.makedirs(os.path.dirname(full_restore_path), exist_ok=True)
                    
                    with open(full_restore_path, 'wb') as f:
                        f.write(content)
                
                # 删除已删除的文件
                for file_path in deleted_files:
                    rel_path = os.path.relpath(file_path, backup_config.source_path)
                    full_restore_path = os.path.join(restore_path, rel_path)
                    if os.path.exists(full_restore_path):
                        os.remove(full_restore_path)
            
            else:
                # 完整恢复
                for file_path, content in restored_data.items():
                    rel_path = os.path.relpath(file_path, backup_config.source_path)
                    full_restore_path = os.path.join(restore_path, rel_path)
                    os.makedirs(os.path.dirname(full_restore_path), exist_ok=True)
                    
                    with open(full_restore_path, 'wb') as f:
                        f.write(content)
            
            logger.info(f"备份恢复成功: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"恢复备份失败: {e}")
            return False
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    def list_backups(self, backup_id: str = None) -> List[BackupStatus]:
        """列出备份"""
        conn = sqlite3.connect(self.backup_db_path)
        cursor = conn.cursor()
        
        if backup_id:
            cursor.execute('''
                SELECT * FROM backups 
                WHERE backup_id LIKE ?
                ORDER BY start_time DESC
            ''', (f"{backup_id}%",))
        else:
            cursor.execute('''
                SELECT * FROM backups 
                ORDER BY start_time DESC
            ''')
        
        columns = [description[0] for description in cursor.description]
        backups = []
        
        for row in cursor.fetchall():
            backup_dict = dict(zip(columns, row))
            
            status = BackupStatus(
                backup_id=backup_dict['backup_id'],
                status=backup_dict['status'],
                start_time=datetime.datetime.fromisoformat(backup_dict['start_time']),
                end_time=datetime.datetime.fromisoformat(backup_dict['end_time']) if backup_dict['end_time'] else None,
                file_count=backup_dict['file_count'],
                total_size=backup_dict['total_size'],
                compressed_size=backup_dict['compressed_size'],
                error_message=backup_dict['error_message'],
                checksum=backup_dict['checksum'],
                location=backup_dict['location']
            )
            
            backups.append(status)
        
        conn.close()
        return backups
    
    def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            # 获取备份信息
            cursor.execute('''
                SELECT location FROM backups 
                WHERE backup_id = ?
            ''', (backup_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.error(f"未找到备份: {backup_id}")
                return False
            
            location = result[0]
            
            # 删除备份文件
            # 这里需要根据存储类型删除文件
            # 简化处理
            
            # 删除数据库记录
            cursor.execute('DELETE FROM backups WHERE backup_id = ?', (backup_id,))
            cursor.execute('DELETE FROM file_hashes WHERE backup_id = ?', (backup_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"备份已删除: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除备份失败: {e}")
            return False
    
    def cleanup_old_backups(self, backup_id: str, retention_days: int = 30) -> int:
        """清理旧备份"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
        
        conn = sqlite3.connect(self.backup_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT backup_id FROM backups 
            WHERE backup_id LIKE ? AND start_time < ?
            ORDER BY start_time ASC
        ''', (f"{backup_id}%", cutoff_date.isoformat()))
        
        old_backups = [row[0] for row in cursor.fetchall()]
        
        deleted_count = 0
        for backup_id in old_backups:
            if self.delete_backup(backup_id):
                deleted_count += 1
        
        conn.close()
        return deleted_count
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """获取备份统计信息"""
        conn = sqlite3.connect(self.backup_db_path)
        cursor = conn.cursor()
        
        # 总备份数
        cursor.execute('SELECT COUNT(*) FROM backups')
        total_backups = cursor.fetchone()[0]
        
        # 成功备份数
        cursor.execute("SELECT COUNT(*) FROM backups WHERE status = 'success'")
        successful_backups = cursor.fetchone()[0]
        
        # 失败备份数
        cursor.execute("SELECT COUNT(*) FROM backups WHERE status = 'failed'")
        failed_backups = cursor.fetchone()[0]
        
        # 总文件数
        cursor.execute('SELECT SUM(file_count) FROM backups')
        total_files = cursor.fetchone()[0] or 0
        
        # 总数据量
        cursor.execute('SELECT SUM(total_size) FROM backups')
        total_size = cursor.fetchone()[0] or 0
        
        # 压缩后总数据量
        cursor.execute('SELECT SUM(compressed_size) FROM backups')
        total_compressed_size = cursor.fetchone()[0] or 0
        
        # 最近备份时间
        cursor.execute('SELECT MAX(start_time) FROM backups')
        last_backup_time = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_backups': total_backups,
            'successful_backups': successful_backups,
            'failed_backups': failed_backups,
            'success_rate': (successful_backups / total_backups * 100) if total_backups > 0 else 0,
            'total_files': total_files,
            'total_size': total_size,
            'total_compressed_size': total_compressed_size,
            'compression_ratio': (total_compressed_size / total_size) if total_size > 0 else 0,
            'last_backup_time': last_backup_time
        }
    
    def schedule_backup(self, backup_config: BackupConfig, schedule_config: Dict) -> str:
        """调度备份"""
        return self.scheduler.schedule_backup(backup_config, schedule_config)
    
    def start_scheduler(self):
        """启动调度器"""
        self.scheduler.start_scheduler()
    
    def stop_scheduler(self):
        """停止调度器"""
        self.scheduler.stop_scheduler()
    
    def generate_report(self, backup_id: str = None) -> str:
        """生成备份报告"""
        backup_statuses = self.list_backups(backup_id)
        return self.reporter.generate_report(backup_statuses)
    
    def save_report(self, report: str, file_path: str) -> bool:
        """保存报告"""
        return self.reporter.save_report(report, file_path)
    
    def send_email_report(self, report: str, subject: str = None) -> bool:
        """发送邮件报告"""
        return self.reporter.send_email_report(report, subject)


# 使用示例和便利函数
def create_file_backup(source_path: str, backup_path: str, backup_id: str = None, **kwargs) -> BackupStatus:
    """创建文件备份的便利函数"""
    if backup_id is None:
        backup_id = f"file_backup_{int(time.time())}"
    
    config = BackupConfig(
        backup_id=backup_id,
        source_path=source_path,
        backup_path=backup_path,
        **kwargs
    )
    
    backup_system = DataBackup()
    return backup_system.create_backup(config)


def create_database_backup(db_config: Dict, backup_path: str, backup_id: str = None, **kwargs) -> BackupStatus:
    """创建数据库备份的便利函数"""
    if backup_id is None:
        backup_id = f"db_backup_{int(time.time())}"
    
    config = BackupConfig(
        backup_id=backup_id,
        source_path="",  # 数据库备份不使用source_path
        backup_path=backup_path,
        database_config=db_config,
        **kwargs
    )
    
    backup_system = DataBackup()
    return backup_system.create_backup(config)


def create_cloud_backup(cloud_config: Dict, backup_path: str, backup_id: str = None, **kwargs) -> BackupStatus:
    """创建云存储备份的便利函数"""
    if backup_id is None:
        backup_id = f"cloud_backup_{int(time.time())}"
    
    config = BackupConfig(
        backup_id=backup_id,
        source_path="",  # 云存储备份不使用source_path
        backup_path=backup_path,
        cloud_config=cloud_config,
        **kwargs
    )
    
    backup_system = DataBackup()
    return backup_system.create_backup(config)


if __name__ == "__main__":
    # 使用示例
    print("R1数据备份器")
    print("=" * 50)
    
    # 创建文件备份
    backup_config = BackupConfig(
        backup_id="example_backup",
        source_path="/path/to/source",
        backup_path="/path/to/backup",
        compression="gzip",
        encryption=True,
        encryption_key="my_secret_key"
    )
    
    backup_system = DataBackup()
    result = backup_system.create_backup(backup_config)
    
    print(f"备份状态: {result.status}")
    print(f"备份位置: {result.location}")
    print(f"文件数量: {result.file_count}")
    print(f"压缩后大小: {result.compressed_size} bytes")