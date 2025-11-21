"""
S8维护服务模块

提供完整的系统维护、数据清理、性能优化等功能
"""

import os
import sys
import json
import time
import shutil
import logging
import sqlite3
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """任务类型枚举"""
    SYSTEM_CLEANUP = "system_cleanup"
    DATA_CLEANUP = "data_cleanup"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BACKUP_MAINTENANCE = "backup_maintenance"
    UPDATE_MANAGEMENT = "update_management"
    LOG_CLEANUP = "log_cleanup"
    DATABASE_OPTIMIZATION = "database_optimization"


@dataclass
class MaintenanceTask:
    """维护任务数据类"""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[str] = None
    error_message: Optional[str] = None
    progress: int = 0
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()


@dataclass
class MaintenancePlan:
    """维护计划数据类"""
    plan_id: str
    name: str
    description: str
    tasks: List[MaintenanceTask]
    schedule: str  # cron表达式或simple格式
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


class MaintenanceService:
    """S8维护服务主类"""
    
    def __init__(self, config_path: str = "maintenance_config.json"):
        """
        初始化维护服务
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.tasks_db_path = os.path.join(os.path.dirname(__file__), "maintenance_tasks.db")
        self.running_tasks: Dict[str, MaintenanceTask] = {}
        self.task_threads: Dict[str, threading.Thread] = {}
        
        # 设置日志
        self._setup_logging()
        
        # 初始化数据库
        self._init_database()
        
        # 加载维护计划
        self.maintenance_plans = self._load_maintenance_plans()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "log_level": "INFO",
            "log_file": "maintenance.log",
            "max_log_size": 10485760,  # 10MB
            "backup_dir": "./backups",
            "temp_dir": "./temp",
            "log_dir": "./logs",
            "db_optimization": {
                "vacuum_interval": 86400,  # 24小时
                "analyze_interval": 3600   # 1小时
            },
            "cleanup": {
                "temp_file_age": 86400,    # 24小时
                "log_file_age": 604800,    # 7天
                "backup_retention": 30     # 30天
            },
            "performance": {
                "memory_threshold": 80,    # 80%
                "disk_threshold": 85,      # 85%
                "cpu_threshold": 90        # 90%
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并默认配置
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        log_file = self.config.get("log_file", "maintenance.log")
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """初始化任务数据库"""
        conn = sqlite3.connect(self.tasks_db_path)
        cursor = conn.cursor()
        
        # 创建任务表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                created_time TEXT,
                start_time TEXT,
                end_time TEXT,
                result TEXT,
                error_message TEXT,
                progress INTEGER DEFAULT 0
            )
        ''')
        
        # 创建维护计划表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_plans (
                plan_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tasks_config TEXT,
                schedule TEXT,
                enabled BOOLEAN DEFAULT 1,
                last_run TEXT,
                next_run TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_maintenance_plans(self) -> List[MaintenancePlan]:
        """加载维护计划"""
        plans = []
        conn = sqlite3.connect(self.tasks_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM maintenance_plans WHERE enabled = 1")
        rows = cursor.fetchall()
        
        for row in rows:
            plan_id, name, description, tasks_config, schedule, enabled, last_run, next_run = row
            
            # 解析任务配置
            tasks = []
            if tasks_config:
                try:
                    tasks_data = json.loads(tasks_config)
                    for task_data in tasks_data:
                        task = MaintenanceTask(
                            task_id=task_data['task_id'],
                            task_type=TaskType(task_data['task_type']),
                            name=task_data['name'],
                            description=task_data['description']
                        )
                        tasks.append(task)
                except Exception as e:
                    self.logger.error(f"解析任务配置失败: {e}")
            
            plan = MaintenancePlan(
                plan_id=plan_id,
                name=name,
                description=description,
                tasks=tasks,
                schedule=schedule,
                enabled=bool(enabled)
            )
            
            if last_run:
                plan.last_run = datetime.fromisoformat(last_run)
            if next_run:
                plan.next_run = datetime.fromisoformat(next_run)
            
            plans.append(plan)
        
        conn.close()
        return plans
    
    def create_maintenance_task(self, task_type: TaskType, name: str, description: str) -> str:
        """
        创建维护任务
        
        Args:
            task_type: 任务类型
            name: 任务名称
            description: 任务描述
            
        Returns:
            任务ID
        """
        task_id = f"{task_type.value}_{int(time.time() * 1000000)}"
        task = MaintenanceTask(
            task_id=task_id,
            task_type=task_type,
            name=name,
            description=description
        )
        
        # 保存到数据库
        self._save_task_to_db(task)
        
        self.logger.info(f"创建维护任务: {name} ({task_id})")
        return task_id
    
    def _save_task_to_db(self, task: MaintenanceTask):
        """保存任务到数据库"""
        conn = sqlite3.connect(self.tasks_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO maintenance_tasks 
            (task_id, task_type, name, description, status, created_time, 
             start_time, end_time, result, error_message, progress)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id,
            task.task_type.value,
            task.name,
            task.description,
            task.status.value,
            task.created_time.isoformat() if task.created_time else None,
            task.start_time.isoformat() if task.start_time else None,
            task.end_time.isoformat() if task.end_time else None,
            task.result,
            task.error_message,
            task.progress
        ))
        
        conn.commit()
        conn.close()
    
    def execute_task(self, task_id: str) -> bool:
        """
        执行维护任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            执行是否成功
        """
        if task_id in self.running_tasks:
            self.logger.warning(f"任务 {task_id} 已在运行中")
            return False
        
        # 获取任务信息
        task = self._get_task_from_db(task_id)
        if not task:
            self.logger.error(f"任务 {task_id} 不存在")
            return False
        
        # 更新任务状态
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        self._save_task_to_db(task)
        
        self.running_tasks[task_id] = task
        
        # 创建执行线程
        thread = threading.Thread(target=self._execute_task_worker, args=(task,))
        thread.daemon = True
        thread.start()
        
        self.task_threads[task_id] = thread
        
        self.logger.info(f"开始执行任务: {task.name} ({task_id})")
        return True
    
    def _execute_task_worker(self, task: MaintenanceTask):
        """任务执行工作线程"""
        try:
            self.logger.info(f"执行任务: {task.name}")
            
            # 根据任务类型执行相应操作
            if task.task_type == TaskType.SYSTEM_CLEANUP:
                result = self._execute_system_cleanup(task)
            elif task.task_type == TaskType.DATA_CLEANUP:
                result = self._execute_data_cleanup(task)
            elif task.task_type == TaskType.PERFORMANCE_OPTIMIZATION:
                result = self._execute_performance_optimization(task)
            elif task.task_type == TaskType.BACKUP_MAINTENANCE:
                result = self._execute_backup_maintenance(task)
            elif task.task_type == TaskType.UPDATE_MANAGEMENT:
                result = self._execute_update_management(task)
            elif task.task_type == TaskType.LOG_CLEANUP:
                result = self._execute_log_cleanup(task)
            elif task.task_type == TaskType.DATABASE_OPTIMIZATION:
                result = self._execute_database_optimization(task)
            else:
                result = f"未知的任务类型: {task.task_type}"
            
            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            task.result = result
            task.progress = 100
            
        except Exception as e:
            # 更新任务状态为失败
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            task.error_message = str(e)
            self.logger.error(f"任务执行失败: {task.name} - {e}")
        
        finally:
            # 保存任务状态
            self._save_task_to_db(task)
            
            # 从运行任务中移除
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            if task.task_id in self.task_threads:
                del self.task_threads[task.task_id]
    
    def _execute_system_cleanup(self, task: MaintenanceTask) -> str:
        """执行系统清理"""
        cleaned_files = 0
        cleaned_size = 0
        
        # 清理临时文件
        temp_dir = self.config.get("temp_dir", "./temp")
        if os.path.exists(temp_dir):
            temp_age = self.config.get("cleanup", {}).get("temp_file_age", 86400)
            cutoff_time = time.time() - temp_age
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if os.path.getmtime(file_path) < cutoff_time:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_files += 1
                            cleaned_size += file_size
                    except Exception as e:
                        self.logger.warning(f"删除临时文件失败 {file_path}: {e}")
        
        # 清理系统缓存
        try:
            # 清理Python缓存
            for root, dirs, files in os.walk("."):
                for dir_name in dirs:
                    if dir_name == "__pycache__":
                        cache_path = os.path.join(root, dir_name)
                        shutil.rmtree(cache_path)
                        cleaned_files += 1
        except Exception as e:
            self.logger.warning(f"清理系统缓存失败: {e}")
        
        result = f"系统清理完成: 清理了 {cleaned_files} 个文件，释放空间 {cleaned_size / 1024 / 1024:.2f} MB"
        self.logger.info(result)
        return result
    
    def _execute_data_cleanup(self, task: MaintenanceTask) -> str:
        """执行数据清理"""
        cleaned_records = 0
        
        # 清理过期日志
        log_dir = self.config.get("log_dir", "./logs")
        if os.path.exists(log_dir):
            log_age = self.config.get("cleanup", {}).get("log_file_age", 604800)
            cutoff_time = time.time() - log_age
            
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith('.log'):
                        file_path = os.path.join(root, file)
                        try:
                            if os.path.getmtime(file_path) < cutoff_time:
                                os.remove(file_path)
                                cleaned_records += 1
                        except Exception as e:
                            self.logger.warning(f"删除过期日志失败 {file_path}: {e}")
        
        # 清理数据库临时数据
        try:
            conn = sqlite3.connect(self.tasks_db_path)
            cursor = conn.cursor()
            
            # 删除已完成的旧任务记录
            cutoff_date = datetime.now() - timedelta(days=30)
            cursor.execute('''
                DELETE FROM maintenance_tasks 
                WHERE status = ? AND created_time < ?
            ''', (TaskStatus.COMPLETED.value, cutoff_date.isoformat()))
            
            deleted_count = cursor.rowcount
            cleaned_records += deleted_count
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"清理数据库数据失败: {e}")
        
        result = f"数据清理完成: 清理了 {cleaned_records} 条记录"
        self.logger.info(result)
        return result
    
    def _execute_performance_optimization(self, task: MaintenanceTask) -> str:
        """执行性能优化"""
        optimizations = []
        
        # 内存使用优化
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > self.config.get("performance", {}).get("memory_threshold", 80):
                # 强制垃圾回收
                import gc
                gc.collect()
                optimizations.append("执行内存垃圾回收")
        except ImportError:
            optimizations.append("psutil未安装，跳过内存监控")
        except Exception as e:
            self.logger.warning(f"内存优化失败: {e}")
        
        # 磁盘空间优化
        try:
            disk_usage = shutil.disk_usage(".")
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            if usage_percent > self.config.get("performance", {}).get("disk_threshold", 85):
                optimizations.append(f"磁盘使用率过高: {usage_percent:.1f}%")
        except Exception as e:
            self.logger.warning(f"磁盘检查失败: {e}")
        
        # 数据库优化
        try:
            conn = sqlite3.connect(self.tasks_db_path)
            cursor = conn.cursor()
            
            # 执行VACUUM
            cursor.execute("VACUUM")
            optimizations.append("执行数据库VACUUM优化")
            
            # 执行ANALYZE
            cursor.execute("ANALYZE")
            optimizations.append("执行数据库ANALYZE优化")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"数据库优化失败: {e}")
        
        result = f"性能优化完成: {', '.join(optimizations)}"
        self.logger.info(result)
        return result
    
    def _execute_backup_maintenance(self, task: MaintenanceTask) -> str:
        """执行备份维护"""
        backup_dir = self.config.get("backup_dir", "./backups")
        backup_count = 0
        verified_count = 0
        
        # 确保备份目录存在
        os.makedirs(backup_dir, exist_ok=True)
        
        # 创建备份
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"maintenance_backup_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # 备份数据库
            if os.path.exists(self.tasks_db_path):
                shutil.copy2(self.tasks_db_path, f"{backup_path}.db")
                backup_count += 1
            
            # 备份配置文件
            if os.path.exists(self.config_path):
                shutil.copy2(self.config_path, f"{backup_path}.json")
                backup_count += 1
            
            # 验证备份
            if os.path.exists(f"{backup_path}.db"):
                conn = sqlite3.connect(f"{backup_path}.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM maintenance_tasks")
                record_count = cursor.fetchone()[0]
                conn.close()
                verified_count += 1
            
        except Exception as e:
            self.logger.error(f"备份维护失败: {e}")
            return f"备份维护失败: {e}"
        
        # 清理过期备份
        try:
            backup_retention = self.config.get("cleanup", {}).get("backup_retention", 30)
            cutoff_time = time.time() - (backup_retention * 86400)
            
            for file in os.listdir(backup_dir):
                if file.startswith("maintenance_backup_"):
                    file_path = os.path.join(backup_dir, file)
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
        except Exception as e:
            self.logger.warning(f"清理过期备份失败: {e}")
        
        result = f"备份维护完成: 创建 {backup_count} 个备份，验证 {verified_count} 个备份"
        self.logger.info(result)
        return result
    
    def _execute_update_management(self, task: MaintenanceTask) -> str:
        """执行更新管理"""
        updates = []
        
        try:
            # 检查Python包更新
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--outdated"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                outdated_packages = result.stdout.strip().split('\n')[2:]  # 跳过标题行
                if outdated_packages and outdated_packages[0]:
                    updates.append(f"发现 {len([p for p in outdated_packages if p])} 个过期包")
                else:
                    updates.append("所有包都是最新的")
            else:
                updates.append("检查包更新失败")
                
        except subprocess.TimeoutExpired:
            updates.append("检查包更新超时")
        except Exception as e:
            updates.append(f"检查包更新异常: {e}")
        
        try:
            # 检查系统更新（仅Linux）
            if sys.platform.startswith('linux'):
                result = subprocess.run(["apt", "list", "--upgradable"], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    upgradable = [line for line in result.stdout.split('\n') if 'upgradable' in line]
                    if upgradable:
                        updates.append(f"发现 {len(upgradable)} 个系统更新")
                    else:
                        updates.append("系统已是最新版本")
            else:
                updates.append("非Linux系统，跳过系统更新检查")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            updates.append("系统更新检查不可用")
        except Exception as e:
            updates.append(f"系统更新检查异常: {e}")
        
        result = f"更新管理完成: {', '.join(updates)}"
        self.logger.info(result)
        return result
    
    def _execute_log_cleanup(self, task: MaintenanceTask) -> str:
        """执行日志清理"""
        cleaned_logs = 0
        
        log_dir = self.config.get("log_dir", "./logs")
        log_age = self.config.get("cleanup", {}).get("log_file_age", 604800)
        cutoff_time = time.time() - log_age
        
        if os.path.exists(log_dir):
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith('.log'):
                        file_path = os.path.join(root, file)
                        try:
                            if os.path.getmtime(file_path) < cutoff_time:
                                os.remove(file_path)
                                cleaned_logs += 1
                        except Exception as e:
                            self.logger.warning(f"删除日志文件失败 {file_path}: {e}")
        
        result = f"日志清理完成: 删除了 {cleaned_logs} 个过期日志文件"
        self.logger.info(result)
        return result
    
    def _execute_database_optimization(self, task: MaintenanceTask) -> str:
        """执行数据库优化"""
        optimizations = []
        
        try:
            conn = sqlite3.connect(self.tasks_db_path)
            cursor = conn.cursor()
            
            # 获取数据库信息
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            
            db_size = page_count * page_size
            optimizations.append(f"数据库大小: {db_size / 1024 / 1024:.2f} MB")
            
            # 执行VACUUM
            cursor.execute("VACUUM")
            optimizations.append("执行VACUUM优化")
            
            # 执行ANALYZE
            cursor.execute("ANALYZE")
            optimizations.append("执行ANALYZE优化")
            
            # 重建索引
            cursor.execute("REINDEX")
            optimizations.append("重建数据库索引")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"数据库优化失败: {e}")
            return f"数据库优化失败: {e}"
        
        result = f"数据库优化完成: {', '.join(optimizations)}"
        self.logger.info(result)
        return result
    
    def _get_task_from_db(self, task_id: str) -> Optional[MaintenanceTask]:
        """从数据库获取任务"""
        conn = sqlite3.connect(self.tasks_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM maintenance_tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            task_id, task_type, name, description, status, created_time, start_time, end_time, result, error_message, progress = row
            
            task = MaintenanceTask(
                task_id=task_id,
                task_type=TaskType(task_type),
                name=name,
                description=description,
                status=TaskStatus(status),
                result=result,
                error_message=error_message,
                progress=progress
            )
            
            if created_time:
                task.created_time = datetime.fromisoformat(created_time)
            if start_time:
                task.start_time = datetime.fromisoformat(start_time)
            if end_time:
                task.end_time = datetime.fromisoformat(end_time)
            
            return task
        
        return None
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self._get_task_from_db(task_id)
        if task:
            return asdict(task)
        return None
    
    def list_tasks(self, status: Optional[TaskStatus] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """列出任务"""
        conn = sqlite3.connect(self.tasks_db_path)
        cursor = conn.cursor()
        
        if status:
            cursor.execute('''
                SELECT * FROM maintenance_tasks 
                WHERE status = ? 
                ORDER BY created_time DESC 
                LIMIT ?
            ''', (status.value, limit))
        else:
            cursor.execute('''
                SELECT * FROM maintenance_tasks 
                ORDER BY created_time DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in rows:
            task_id, task_type, name, description, status, created_time, start_time, end_time, result, error_message, progress = row
            
            task_dict = {
                'task_id': task_id,
                'task_type': task_type,
                'name': name,
                'description': description,
                'status': status,
                'created_time': created_time,
                'start_time': start_time,
                'end_time': end_time,
                'result': result,
                'error_message': error_message,
                'progress': progress
            }
            tasks.append(task_dict)
        
        return tasks
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.running_tasks:
            # 标记任务为取消状态
            task = self.running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.end_time = datetime.now()
            self._save_task_to_db(task)
            
            # 从运行任务中移除
            del self.running_tasks[task_id]
            
            if task_id in self.task_threads:
                del self.task_threads[task_id]
            
            self.logger.info(f"任务已取消: {task_id}")
            return True
        
        return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "running_tasks": len(self.running_tasks),
            "total_tasks": len(self.list_tasks()),
            "completed_tasks": len(self.list_tasks(TaskStatus.COMPLETED)),
            "failed_tasks": len(self.list_tasks(TaskStatus.FAILED)),
            "maintenance_plans": len(self.maintenance_plans),
            "service_uptime": time.time() - getattr(self, '_start_time', time.time())
        }
    
    def create_maintenance_plan(self, name: str, description: str, 
                              task_configs: List[Dict[str, str]], schedule: str) -> str:
        """
        创建维护计划
        
        Args:
            name: 计划名称
            description: 计划描述
            task_configs: 任务配置列表
            schedule: 执行计划（cron表达式）
            
        Returns:
            计划ID
        """
        plan_id = f"plan_{int(time.time())}"
        
        # 创建任务对象
        tasks = []
        for config in task_configs:
            task_type = TaskType(config['task_type'])
            task = MaintenanceTask(
                task_id=f"{plan_id}_{task_type.value}",
                task_type=task_type,
                name=config['name'],
                description=config['description']
            )
            tasks.append(task)
        
        plan = MaintenancePlan(
            plan_id=plan_id,
            name=name,
            description=description,
            tasks=tasks,
            schedule=schedule
        )
        
        # 保存到数据库
        self._save_plan_to_db(plan)
        
        # 添加到内存中的计划列表
        self.maintenance_plans.append(plan)
        
        self.logger.info(f"创建维护计划: {name} ({plan_id})")
        return plan_id
    
    def _save_plan_to_db(self, plan: MaintenancePlan):
        """保存计划到数据库"""
        conn = sqlite3.connect(self.tasks_db_path)
        cursor = conn.cursor()
        
        # 序列化任务配置
        tasks_config = []
        for task in plan.tasks:
            task_config = {
                'task_id': task.task_id,
                'task_type': task.task_type.value,
                'name': task.name,
                'description': task.description
            }
            tasks_config.append(task_config)
        
        cursor.execute('''
            INSERT OR REPLACE INTO maintenance_plans 
            (plan_id, name, description, tasks_config, schedule, enabled, last_run, next_run)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            plan.plan_id,
            plan.name,
            plan.description,
            json.dumps(tasks_config),
            plan.schedule,
            plan.enabled,
            plan.last_run.isoformat() if plan.last_run else None,
            plan.next_run.isoformat() if plan.next_run else None
        ))
        
        conn.commit()
        conn.close()
    
    def execute_maintenance_plan(self, plan_id: str) -> List[str]:
        """
        执行维护计划
        
        Args:
            plan_id: 计划ID
            
        Returns:
            执行的任务ID列表
        """
        plan = None
        for p in self.maintenance_plans:
            if p.plan_id == plan_id:
                plan = p
                break
        
        if not plan:
            self.logger.error(f"维护计划不存在: {plan_id}")
            return []
        
        executed_task_ids = []
        
        for task in plan.tasks:
            task_id = self.create_maintenance_task(task.task_type, task.name, task.description)
            if self.execute_task(task_id):
                executed_task_ids.append(task_id)
        
        # 更新计划执行时间
        plan.last_run = datetime.now()
        self._save_plan_to_db(plan)
        
        self.logger.info(f"执行维护计划: {plan.name}, 执行了 {len(executed_task_ids)} 个任务")
        return executed_task_ids
    
    def generate_maintenance_report(self, days: int = 7) -> Dict[str, Any]:
        """生成维护报告"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 统计任务数据
        all_tasks = self.list_tasks(limit=1000)
        period_tasks = [
            task for task in all_tasks 
            if task['created_time'] and 
            datetime.fromisoformat(task['created_time']) >= start_date
        ]
        
        # 按状态统计
        status_counts = {}
        for task in period_tasks:
            status = task['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # 按类型统计
        type_counts = {}
        for task in period_tasks:
            task_type = task['task_type']
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        # 计算平均执行时间
        completed_tasks = [task for task in period_tasks if task['status'] == TaskStatus.COMPLETED.value]
        avg_execution_time = 0
        if completed_tasks:
            total_time = 0
            for task in completed_tasks:
                if task['start_time'] and task['end_time']:
                    start = datetime.fromisoformat(task['start_time'])
                    end = datetime.fromisoformat(task['end_time'])
                    total_time += (end - start).total_seconds()
            avg_execution_time = total_time / len(completed_tasks) if completed_tasks else 0
        
        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "task_statistics": {
                "total_tasks": len(period_tasks),
                "status_distribution": status_counts,
                "type_distribution": type_counts,
                "average_execution_time_seconds": avg_execution_time
            },
            "service_status": self.get_service_status(),
            "generated_at": datetime.now().isoformat()
        }
        
        return report


# 示例使用
if __name__ == "__main__":
    # 创建维护服务实例
    service = MaintenanceService()
    
    # 创建系统清理任务
    task_id = service.create_maintenance_task(
        TaskType.SYSTEM_CLEANUP,
        "系统清理任务",
        "清理临时文件和系统缓存"
    )
    
    # 执行任务
    service.execute_task(task_id)
    
    # 等待任务完成
    time.sleep(2)
    
    # 获取任务状态
    status = service.get_task_status(task_id)
    print(f"任务状态: {status}")
    
    # 生成维护报告
    report = service.generate_maintenance_report()
    print(f"维护报告: {json.dumps(report, indent=2, ensure_ascii=False)}")