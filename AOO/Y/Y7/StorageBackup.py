"""
Y7存储备份器主要实现
提供完整的数据备份、恢复、管理功能
"""

import os
import json
import shutil
import hashlib
import datetime
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging


class BackupType(Enum):
    """备份类型枚举"""
    FULL = "full"           # 全量备份
    INCREMENTAL = "incremental"  # 增量备份
    DIFFERENTIAL = "differential"  # 差异备份


class BackupStatus(Enum):
    """备份状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackupPriority(Enum):
    """备份优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BackupConfig:
    """备份配置类"""
    backup_dir: str = "./backups"
    max_backups: int = 10
    compression: bool = True
    encryption: bool = False
    verify_backup: bool = True
    parallel_jobs: int = 4
    schedule_enabled: bool = False
    schedule_interval: str = "daily"  # daily, weekly, monthly
    schedule_time: str = "02:00"
    exclude_patterns: List[str] = None
    include_patterns: List[str] = None
    
    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = ['*.tmp', '*.log', '__pycache__', '.git']
        if self.include_patterns is None:
            self.include_patterns = ['*']


@dataclass
class BackupTask:
    """备份任务类"""
    task_id: str
    source_path: str
    backup_path: str
    backup_type: BackupType
    priority: BackupPriority = BackupPriority.NORMAL
    status: BackupStatus = BackupStatus.PENDING
    created_time: datetime.datetime = None
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    file_count: int = 0
    total_size: int = 0
    processed_size: int = 0
    error_message: str = ""
    checksum: str = ""
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.datetime.now()


@dataclass
class BackupStatistics:
    """备份统计类"""
    total_backups: int = 0
    successful_backups: int = 0
    failed_backups: int = 0
    total_size: int = 0
    total_files: int = 0
    avg_backup_time: float = 0.0
    last_backup_time: Optional[datetime.datetime] = None
    storage_used: int = 0
    compression_ratio: float = 0.0


class BackupValidator:
    """备份验证类"""
    
    @staticmethod
    def validate_backup_integrity(backup_path: str, expected_checksum: str = None) -> Tuple[bool, str]:
        """验证备份完整性"""
        try:
            if not os.path.exists(backup_path):
                return False, "备份文件不存在"
            
            # 计算实际校验和
            actual_checksum = BackupValidator._calculate_checksum(backup_path)
            
            if expected_checksum and actual_checksum != expected_checksum:
                return False, f"校验和不匹配: 期望 {expected_checksum}, 实际 {actual_checksum}"
            
            return True, "备份完整性验证通过"
            
        except Exception as e:
            return False, f"验证失败: {str(e)}"
    
    @staticmethod
    def _calculate_checksum(file_path: str) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""


class BackupReport:
    """备份报告类"""
    
    def __init__(self):
        self.reports: List[Dict] = []
    
    def add_report(self, task: BackupTask, success: bool, message: str):
        """添加备份报告"""
        report = {
            'task_id': task.task_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'source_path': task.source_path,
            'backup_path': task.backup_path,
            'backup_type': task.backup_type.value,
            'status': task.status.value,
            'success': success,
            'message': message,
            'file_count': task.file_count,
            'total_size': task.total_size,
            'duration': (task.end_time - task.start_time).total_seconds() if task.start_time and task.end_time else 0
        }
        self.reports.append(report)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """生成汇总报告"""
        if not self.reports:
            return {}
        
        total_reports = len(self.reports)
        successful_reports = sum(1 for r in self.reports if r['success'])
        failed_reports = total_reports - successful_reports
        
        total_size = sum(r['total_size'] for r in self.reports)
        total_files = sum(r['file_count'] for r in self.reports)
        total_duration = sum(r['duration'] for r in self.reports)
        
        return {
            'total_backups': total_reports,
            'successful_backups': successful_reports,
            'failed_backups': failed_reports,
            'success_rate': successful_reports / total_reports * 100 if total_reports > 0 else 0,
            'total_size': total_size,
            'total_files': total_files,
            'avg_duration': total_duration / total_reports if total_reports > 0 else 0,
            'latest_backup': max(self.reports, key=lambda x: x['timestamp'])['timestamp']
        }
    
    def export_report(self, file_path: str):
        """导出报告到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': self.generate_summary_report(),
                'details': self.reports
            }, f, ensure_ascii=False, indent=2)


class BackupRecovery:
    """备份恢复类"""
    
    def __init__(self, backup_dir: str):
        self.backup_dir = backup_dir
    
    def list_available_backups(self, source_path: str = None) -> List[Dict]:
        """列出可用的备份"""
        backups = []
        backup_root = Path(self.backup_dir)
        
        for backup_dir in backup_root.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / "backup_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        if source_path is None or metadata.get('source_path') == source_path:
                            backups.append({
                                'backup_id': backup_dir.name,
                                'path': str(backup_dir),
                                'created_time': metadata.get('created_time'),
                                'backup_type': metadata.get('backup_type'),
                                'size': metadata.get('total_size', 0),
                                'file_count': metadata.get('file_count', 0)
                            })
                    except Exception:
                        continue
        
        return sorted(backups, key=lambda x: x['created_time'], reverse=True)
    
    def restore_backup(self, backup_id: str, target_path: str, 
                      verify_integrity: bool = True) -> Tuple[bool, str]:
        """恢复备份"""
        try:
            backup_path = Path(self.backup_dir) / backup_id
            if not backup_path.exists():
                return False, f"备份 {backup_id} 不存在"
            
            # 验证备份完整性
            if verify_integrity:
                metadata_file = backup_path / "backup_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    expected_checksum = metadata.get('checksum')
                    
                    valid, message = BackupValidator.validate_backup_integrity(
                        str(backup_path), expected_checksum
                    )
                    if not valid:
                        return False, f"备份完整性验证失败: {message}"
            
            # 创建目标目录
            os.makedirs(target_path, exist_ok=True)
            
            # 恢复文件
            if (backup_path / "backup_data").exists():
                source_data = backup_path / "backup_data"
                shutil.copytree(source_data, target_path, dirs_exist_ok=True)
            else:
                # 如果没有备份数据目录，直接复制整个备份目录内容
                for item in backup_path.iterdir():
                    if item.name != "backup_metadata.json":
                        if item.is_file():
                            shutil.copy2(item, target_path)
                        elif item.is_dir():
                            shutil.copytree(item, target_path / item.name, dirs_exist_ok=True)
            
            return True, f"备份 {backup_id} 恢复成功"
            
        except Exception as e:
            return False, f"恢复失败: {str(e)}"


class BackupManager:
    """备份管理器类"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.tasks: Dict[str, BackupTask] = {}
        self.running_tasks: Dict[str, threading.Thread] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.parallel_jobs)
        self.scheduler_running = False
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('BackupManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_backup_task(self, source_path: str, backup_type: BackupType = BackupType.FULL,
                          priority: BackupPriority = BackupPriority.NORMAL) -> str:
        """创建备份任务"""
        task_id = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = os.path.join(self.config.backup_dir, task_id)
        task = BackupTask(
            task_id=task_id,
            source_path=source_path,
            backup_path=backup_dir,
            backup_type=backup_type,
            priority=priority
        )
        
        self.tasks[task_id] = task
        self.logger.info(f"创建备份任务: {task_id}")
        return task_id
    
    def start_backup_task(self, task_id: str) -> bool:
        """启动备份任务"""
        if task_id not in self.tasks:
            self.logger.error(f"任务 {task_id} 不存在")
            return False
        
        task = self.tasks[task_id]
        if task.status == BackupStatus.RUNNING:
            self.logger.warning(f"任务 {task_id} 已在运行中")
            return False
        
        # 启动备份线程
        thread = threading.Thread(target=self._execute_backup, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        self.running_tasks[task_id] = thread
        task.status = BackupStatus.RUNNING
        task.start_time = datetime.datetime.now()
        
        self.logger.info(f"启动备份任务: {task_id}")
        return True
    
    def _execute_backup(self, task_id: str):
        """执行备份任务"""
        task = self.tasks[task_id]
        try:
            self.logger.info(f"开始执行备份任务: {task_id}")
            
            # 创建备份目录
            os.makedirs(task.backup_path, exist_ok=True)
            
            # 执行备份
            if task.backup_type == BackupType.FULL:
                success = self._execute_full_backup(task)
            elif task.backup_type == BackupType.INCREMENTAL:
                success = self._execute_incremental_backup(task)
            else:  # DIFFERENTIAL
                success = self._execute_differential_backup(task)
            
            if success:
                task.status = BackupStatus.COMPLETED
                task.end_time = datetime.datetime.now()
                self.logger.info(f"备份任务 {task_id} 完成")
            else:
                task.status = BackupStatus.FAILED
                task.end_time = datetime.datetime.now()
                self.logger.error(f"备份任务 {task_id} 失败")
            
        except Exception as e:
            task.status = BackupStatus.FAILED
            task.end_time = datetime.datetime.now()
            task.error_message = str(e)
            self.logger.error(f"备份任务 {task_id} 异常: {str(e)}")
        finally:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def _execute_full_backup(self, task: BackupTask) -> bool:
        """执行全量备份"""
        try:
            source_path = Path(task.source_path)
            backup_data_path = Path(task.backup_path) / "backup_data"
            backup_data_path.mkdir(exist_ok=True)
            
            # 复制文件
            if source_path.is_file():
                return self._copy_file(source_path, backup_data_path / source_path.name, task)
            elif source_path.is_dir():
                return self._copy_directory(source_path, backup_data_path, task)
            else:
                task.error_message = "源路径不存在"
                return False
                
        except Exception as e:
            task.error_message = str(e)
            return False
    
    def _execute_incremental_backup(self, task: BackupTask) -> bool:
        """执行增量备份"""
        # 简化实现：复制所有文件并在文件名中添加时间戳
        try:
            source_path = Path(task.source_path)
            backup_data_path = Path(task.backup_path) / "backup_data"
            backup_data_path.mkdir(exist_ok=True)
            
            if source_path.is_file():
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
                return self._copy_file(source_path, backup_data_path / backup_filename, task)
            elif source_path.is_dir():
                return self._copy_directory(source_path, backup_data_path, task)
            else:
                task.error_message = "源路径不存在"
                return False
                
        except Exception as e:
            task.error_message = str(e)
            return False
    
    def _execute_differential_backup(self, task: BackupTask) -> bool:
        """执行差异备份"""
        # 简化实现：与增量备份相同
        return self._execute_incremental_backup(task)
    
    def _copy_file(self, source: Path, target: Path, task: BackupTask) -> bool:
        """复制文件"""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            
            task.file_count += 1
            task.total_size += source.stat().st_size
            task.processed_size = task.total_size
            
            return True
        except Exception as e:
            task.error_message = f"文件复制失败: {str(e)}"
            return False
    
    def _copy_directory(self, source: Path, target: Path, task: BackupTask) -> bool:
        """复制目录"""
        try:
            target.mkdir(parents=True, exist_ok=True)
            
            for item in source.rglob('*'):
                if self._should_skip_file(item):
                    continue
                
                relative_path = item.relative_to(source)
                target_path = target / relative_path
                
                if item.is_file():
                    if not self._copy_file(item, target_path, task):
                        return False
                elif item.is_dir():
                    target_path.mkdir(exist_ok=True)
            
            return True
        except Exception as e:
            task.error_message = f"目录复制失败: {str(e)}"
            return False
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        # 检查排除模式
        for pattern in self.config.exclude_patterns:
            if file_path.match(pattern):
                return True
        
        # 检查包含模式
        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if file_path.match(pattern):
                    return False
            return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[BackupTask]:
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def list_tasks(self, status: BackupStatus = None) -> List[BackupTask]:
        """列出任务"""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_time, reverse=True)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status in [BackupStatus.COMPLETED, BackupStatus.FAILED]:
            return False
        
        task.status = BackupStatus.CANCELLED
        if task_id in self.running_tasks:
            # 注意：这里只是标记任务为取消状态，实际停止需要更复杂的实现
            del self.running_tasks[task_id]
        
        self.logger.info(f"取消备份任务: {task_id}")
        return True
    
    def start_scheduler(self):
        """启动调度器"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        self.logger.info("启动备份调度器")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.scheduler_running = False
        self.logger.info("停止备份调度器")
    
    def _run_scheduler(self):
        """运行调度器"""
        # 简化实现：定期检查并执行调度的备份
        while self.scheduler_running:
            try:
                # 这里可以实现更复杂的调度逻辑
                # 当前简化为定期执行（实际使用中需要更精确的调度）
                current_time = datetime.datetime.now()
                if hasattr(self, '_last_schedule_check'):
                    time_diff = (current_time - self._last_schedule_check).total_seconds()
                    if time_diff >= 60:  # 每分钟检查一次
                        self._check_scheduled_backups()
                        self._last_schedule_check = current_time
                else:
                    self._last_schedule_check = current_time
                
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                self.logger.error(f"调度器错误: {str(e)}")
                time.sleep(60)
    
    def _check_scheduled_backups(self):
        """检查并执行调度的备份"""
        # 简化实现：这里可以根据配置的调度时间执行备份
        # 实际应用中需要更复杂的调度逻辑
        pass
    
    def schedule_backup(self, source_path: str, backup_type: BackupType = BackupType.FULL,
                       interval: str = None, time_str: str = None):
        """调度备份"""
        interval = interval or self.config.schedule_interval
        time_str = time_str or self.config.schedule_time
        
        # 简化实现：记录调度信息但不实际执行复杂调度
        # 实际应用中需要更复杂的调度逻辑
        self.logger.info(f"调度备份: {source_path}, 类型: {backup_type.value}, 间隔: {interval}, 时间: {time_str}")
        
        # 这里可以添加更复杂的调度逻辑
        # 例如：保存调度配置到文件，定期读取并执行
    
    def _scheduled_backup(self, source_path: str, backup_type: BackupType):
        """执行调度的备份"""
        task_id = self.create_backup_task(source_path, backup_type)
        self.start_backup_task(task_id)
        self.logger.info(f"执行调度备份: {task_id}")


class StorageBackup:
    """主要的存储备份器类"""
    
    def __init__(self, config: BackupConfig = None):
        self.config = config or BackupConfig()
        self.backup_manager = BackupManager(self.config)
        self.backup_recovery = BackupRecovery(self.config.backup_dir)
        self.backup_statistics = BackupStatistics()
        self.backup_report = BackupReport()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('StorageBackup')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def backup(self, source_path: str, backup_type: BackupType = BackupType.FULL,
               priority: BackupPriority = BackupPriority.NORMAL) -> str:
        """执行备份"""
        # 创建备份目录
        os.makedirs(self.config.backup_dir, exist_ok=True)
        
        # 创建并启动备份任务
        task_id = self.backup_manager.create_backup_task(source_path, backup_type, priority)
        self.backup_manager.start_backup_task(task_id)
        
        self.logger.info(f"启动备份: {source_path} -> {task_id}")
        return task_id
    
    def restore(self, backup_id: str, target_path: str, verify_integrity: bool = True) -> Tuple[bool, str]:
        """恢复备份"""
        success, message = self.backup_recovery.restore_backup(
            backup_id, target_path, verify_integrity
        )
        
        self.logger.info(f"恢复备份: {backup_id} -> {target_path}, 结果: {message}")
        return success, message
    
    def list_backups(self, source_path: str = None) -> List[Dict]:
        """列出可用备份"""
        return self.backup_recovery.list_available_backups(source_path)
    
    def verify_backup(self, backup_id: str) -> Tuple[bool, str]:
        """验证备份"""
        backup_path = os.path.join(self.config.backup_dir, backup_id)
        metadata_file = os.path.join(backup_path, "backup_metadata.json")
        
        expected_checksum = None
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                expected_checksum = metadata.get('checksum')
            except Exception:
                pass
        
        return BackupValidator.validate_backup_integrity(backup_path, expected_checksum)
    
    def get_statistics(self) -> BackupStatistics:
        """获取备份统计信息"""
        # 简化实现：基于任务记录计算统计信息
        tasks = self.backup_manager.list_tasks()
        
        completed_tasks = [t for t in tasks if t.status == BackupStatus.COMPLETED]
        failed_tasks = [t for t in tasks if t.status == BackupStatus.FAILED]
        
        self.backup_statistics.total_backups = len(tasks)
        self.backup_statistics.successful_backups = len(completed_tasks)
        self.backup_statistics.failed_backups = len(failed_tasks)
        self.backup_statistics.total_files = sum(t.file_count for t in completed_tasks)
        self.backup_statistics.total_size = sum(t.total_size for t in completed_tasks)
        
        if completed_tasks:
            durations = [(t.end_time - t.start_time).total_seconds() 
                        for t in completed_tasks if t.start_time and t.end_time]
            if durations:
                self.backup_statistics.avg_backup_time = sum(durations) / len(durations)
            
            self.backup_statistics.last_backup_time = max(
                t.end_time for t in completed_tasks if t.end_time
            )
        
        return self.backup_statistics
    
    def generate_report(self, file_path: str = None) -> str:
        """生成备份报告"""
        if file_path is None:
            file_path = os.path.join(
                self.config.backup_dir, 
                f"backup_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        self.backup_report.export_report(file_path)
        self.logger.info(f"备份报告已生成: {file_path}")
        return file_path
    
    def start_scheduled_backups(self):
        """启动定时备份"""
        self.backup_manager.start_scheduler()
    
    def stop_scheduled_backups(self):
        """停止定时备份"""
        self.backup_manager.stop_scheduler()
    
    def schedule_backup(self, source_path: str, backup_type: BackupType = BackupType.FULL,
                       interval: str = None, time_str: str = None):
        """调度备份"""
        self.backup_manager.schedule_backup(source_path, backup_type, interval, time_str)
    
    def get_task_status(self, task_id: str) -> Optional[BackupTask]:
        """获取任务状态"""
        return self.backup_manager.get_task_status(task_id)
    
    def list_tasks(self, status: BackupStatus = None) -> List[BackupTask]:
        """列出任务"""
        return self.backup_manager.list_tasks(status)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self.backup_manager.cancel_task(task_id)