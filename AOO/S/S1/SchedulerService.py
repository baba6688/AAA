"""
S1调度服务 - 主要调度服务实现
提供任务调度、定时任务、任务管理、并发控制等功能
"""

import asyncio
import logging
import threading
import time
import uuid
import json
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from croniter import croniter
import queue
import traceback


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"        # 等待执行
    RUNNING = "running"        # 正在执行
    COMPLETED = "completed"    # 执行完成
    FAILED = "failed"          # 执行失败
    CANCELLED = "cancelled"    # 已取消
    RETRYING = "retrying"      # 重试中


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str = None
    start_time: datetime = None
    end_time: datetime = None
    execution_time: float = 0.0
    retry_count: int = 0
    
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self.execution_time,
            'retry_count': self.retry_count
        }


@dataclass
class ScheduledTask:
    """调度任务定义"""
    task_id: str
    name: str
    func: Callable
    args: tuple = None
    kwargs: dict = None
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = None
    dependencies: List[str] = None
    cron_expression: str = None
    interval: float = None
    max_runs: int = None
    enabled: bool = True
    created_time: datetime = None
    last_run_time: datetime = None
    run_count: int = 0
    
    def __post_init__(self):
        if self.args is None:
            self.args = ()
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.created_time is None:
            self.created_time = datetime.now()


class TaskLogger:
    """任务日志记录器"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.logger = logging.getLogger(f"scheduler.task")
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_task_start(self, task_id: str, task_name: str):
        self.logger.info(f"任务开始 - ID: {task_id}, 名称: {task_name}")
    
    def log_task_complete(self, task_id: str, task_name: str, execution_time: float, result=None):
        self.logger.info(f"任务完成 - ID: {task_id}, 名称: {task_name}, 耗时: {execution_time:.2f}s")
        if result:
            self.logger.debug(f"任务结果 - ID: {task_id}, 结果: {result}")
    
    def log_task_error(self, task_id: str, task_name: str, error: str):
        self.logger.error(f"任务失败 - ID: {task_id}, 名称: {task_name}, 错误: {error}")
    
    def log_retry(self, task_id: str, retry_count: int, delay: float):
        self.logger.warning(f"任务重试 - ID: {task_id}, 重试次数: {retry_count}, 延迟: {delay}s")


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alert_callbacks: List[Callable] = []
        self.alert_history: List[Dict] = []
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def send_alert(self, alert_type: str, message: str, task_id: str = None, severity: str = "WARNING"):
        """发送告警"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'task_id': task_id,
            'severity': severity
        }
        self.alert_history.append(alert)
        
        # 调用所有告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")


class TaskQueue:
    """任务队列管理器"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.pending_queue = queue.PriorityQueue()
        self.task_locks = weakref.WeakValueDictionary()
    
    def submit_task(self, task: ScheduledTask, priority_offset: int = 0):
        """提交任务到队列"""
        priority = task.priority.value + priority_offset
        self.pending_queue.put((priority, time.time(), task))
    
    def get_next_task(self) -> Optional[ScheduledTask]:
        """获取下一个待执行任务"""
        try:
            if not self.pending_queue.empty():
                priority, timestamp, task = self.pending_queue.get_nowait()
                return task
        except queue.Empty:
            pass
        return None
    
    def is_task_running(self, task_id: str) -> bool:
        """检查任务是否正在运行"""
        return task_id in self.running_tasks
    
    def mark_task_running(self, task_id: str, asyncio_task: asyncio.Task):
        """标记任务为运行状态"""
        self.running_tasks[task_id] = asyncio_task
    
    def mark_task_completed(self, task_id: str):
        """标记任务完成"""
        self.running_tasks.pop(task_id, None)
    
    def get_running_count(self) -> int:
        """获取当前运行任务数量"""
        return len(self.running_tasks)
    
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)


class SchedulerService:
    """S1调度服务主类"""
    
    def __init__(self, max_workers: int = 10, log_file: str = None):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.task_queue = TaskQueue(max_workers)
        self.task_logger = TaskLogger(log_file)
        self.alert_manager = AlertManager()
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # 性能统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'peak_concurrent_tasks': 0
        }
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self.alert_manager.add_alert_callback(callback)
    
    def create_task(self, 
                   name: str,
                   func: Callable,
                   args: tuple = None,
                   kwargs: dict = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   max_retries: int = 3,
                   retry_delay: float = 1.0,
                   timeout: float = None,
                   dependencies: List[str] = None,
                   cron_expression: str = None,
                   interval: float = None,
                   max_runs: int = None,
                   enabled: bool = True) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        
        with self._lock:
            task = ScheduledTask(
                task_id=task_id,
                name=name,
                func=func,
                args=args or (),
                kwargs=kwargs or {},
                priority=priority,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                dependencies=dependencies or [],
                cron_expression=cron_expression,
                interval=interval,
                max_runs=max_runs,
                enabled=enabled
            )
            
            self.tasks[task_id] = task
            self.task_results[task_id] = TaskResult(task_id=task_id, status=TaskStatus.PENDING)
            
            logger.info(f"任务创建成功 - ID: {task_id}, 名称: {name}")
            return task_id
    
    def update_task(self, task_id: str, **kwargs) -> bool:
        """更新任务配置"""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # 更新允许的字段
            allowed_fields = ['name', 'priority', 'max_retries', 'retry_delay', 
                            'timeout', 'dependencies', 'cron_expression', 
                            'interval', 'max_runs', 'enabled']
            
            for field, value in kwargs.items():
                if field in allowed_fields and hasattr(task, field):
                    setattr(task, field, value)
            
            logger.info(f"任务更新成功 - ID: {task_id}")
            return True
    
    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            # 如果任务正在运行，先取消
            if self.task_queue.is_task_running(task_id):
                self.cancel_task(task_id)
            
            del self.tasks[task_id]
            del self.task_results[task_id]
            
            logger.info(f"任务删除成功 - ID: {task_id}")
            return True
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    def list_tasks(self, status: TaskStatus = None) -> List[ScheduledTask]:
        """列出任务"""
        tasks = list(self.tasks.values())
        if status:
            tasks = [task for task in tasks 
                    if self.task_results[task.task_id].status == status]
        return tasks
    
    def execute_task(self, task_id: str) -> TaskResult:
        """立即执行任务"""
        if task_id not in self.tasks:
            raise ValueError(f"任务不存在: {task_id}")
        
        task = self.tasks[task_id]
        
        # 检查依赖
        if not self._check_dependencies(task):
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error="依赖任务未完成"
            )
            self.task_results[task_id] = result
            return result
        
        # 更新任务状态
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            start_time=datetime.now()
        )
        self.task_results[task_id] = result
        
        # 记录任务开始
        self.task_logger.log_task_start(task_id, task.name)
        
        try:
            # 执行任务
            if asyncio.iscoroutinefunction(task.func):
                # 异步函数
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                task_result = loop.run_until_complete(
                    asyncio.wait_for(task.func(*task.args, **task.kwargs), timeout=task.timeout)
                )
                loop.close()
            else:
                # 同步函数
                if task.timeout:
                    # 使用线程池执行带超时的同步任务
                    future = self.task_queue.executor.submit(task.func, *task.args, **task.kwargs)
                    task_result = future.result(timeout=task.timeout)
                else:
                    task_result = task.func(*task.args, **task.kwargs)
            
            # 更新结果
            result.status = TaskStatus.COMPLETED
            result.result = task_result
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()
            
            # 更新任务统计
            task.last_run_time = result.end_time
            task.run_count += 1
            
            # 记录任务完成
            self.task_logger.log_task_complete(task_id, task.name, result.execution_time, task_result)
            
        except Exception as e:
            # 任务执行失败
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()
            
            # 记录任务错误
            self.task_logger.log_task_error(task_id, task.name, str(e))
            
            # 检查是否需要重试
            if result.retry_count < task.max_retries:
                result.status = TaskStatus.RETRYING
                self._schedule_retry(task)
            else:
                # 发送告警
                self.alert_manager.send_alert(
                    "TASK_FAILED",
                    f"任务执行失败，已达到最大重试次数: {task.name}",
                    task_id,
                    "ERROR"
                )
        
        # 更新统计信息
        self._update_stats(result)
        
        return result
    
    def _schedule_retry(self, task: ScheduledTask):
        """安排任务重试"""
        result = self.task_results[task.task_id]
        result.retry_count += 1
        
        delay = task.retry_delay * (2 ** (result.retry_count - 1))  # 指数退避
        
        self.task_logger.log_retry(task.task_id, result.retry_count, delay)
        
        # 延迟执行重试
        def retry_task():
            time.sleep(delay)
            if task.enabled:
                self.execute_task(task.task_id)
        
        retry_thread = threading.Thread(target=retry_task)
        retry_thread.daemon = True
        retry_thread.start()
    
    def _check_dependencies(self, task: ScheduledTask) -> bool:
        """检查任务依赖"""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            
            dep_result = self.task_results.get(dep_id)
            if not dep_result or dep_result.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _update_stats(self, result: TaskResult):
        """更新性能统计"""
        self.stats['total_tasks'] += 1
        
        if result.status == TaskStatus.COMPLETED:
            self.stats['completed_tasks'] += 1
        elif result.status == TaskStatus.FAILED:
            self.stats['failed_tasks'] += 1
        
        if result.execution_time > 0:
            self.stats['total_execution_time'] += result.execution_time
            self.stats['average_execution_time'] = (
                self.stats['total_execution_time'] / self.stats['completed_tasks']
                if self.stats['completed_tasks'] > 0 else 0
            )
        
        # 更新峰值并发数
        current_concurrent = self.task_queue.get_running_count()
        if current_concurrent > self.stats['peak_concurrent_tasks']:
            self.stats['peak_concurrent_tasks'] = current_concurrent
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self.tasks:
            return False
        
        if self.task_queue.is_task_running(task_id):
            # 标记为已取消
            result = self.task_results[task_id]
            result.status = TaskStatus.CANCELLED
            result.end_time = datetime.now()
            
            logger.info(f"任务已取消 - ID: {task_id}")
            return True
        
        return False
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务执行结果"""
        return self.task_results.get(task_id)
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        return self.stats.copy()
    
    async def start_scheduler(self):
        """启动调度器"""
        if self.running:
            logger.warning("调度器已在运行中")
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("调度器已启动")
    
    async def stop_scheduler(self):
        """停止调度器"""
        if not self.running:
            return
        
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("调度器已停止")
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                await self._schedule_tasks()
                await asyncio.sleep(1)  # 每秒检查一次
            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                await asyncio.sleep(5)  # 错误后等待5秒
    
    async def _schedule_tasks(self):
        """调度任务执行"""
        current_time = datetime.now()
        
        for task in list(self.tasks.values()):
            if not task.enabled:
                continue
            
            # 检查是否达到最大执行次数
            if task.max_runs and task.run_count >= task.max_runs:
                task.enabled = False
                continue
            
            # 检查依赖
            if not self._check_dependencies(task):
                continue
            
            should_run = False
            
            # Cron表达式调度
            if task.cron_expression:
                if task.last_run_time is None:
                    should_run = True
                else:
                    cron = croniter(task.cron_expression, task.last_run_time)
                    next_run = cron.get_next(datetime)
                    if next_run <= current_time:
                        should_run = True
            
            # 间隔调度
            elif task.interval:
                if task.last_run_time is None:
                    should_run = True
                else:
                    next_run = task.last_run_time + timedelta(seconds=task.interval)
                    if next_run <= current_time:
                        should_run = True
            
            # 如果需要执行且当前并发数未达到限制
            if should_run and self.task_queue.get_running_count() < self.task_queue.max_workers:
                # 异步执行任务
                asyncio.create_task(self._execute_task_async(task))
    
    async def _execute_task_async(self, task: ScheduledTask):
        """异步执行任务"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.execute_task, task.task_id
            )
            
            # 发送告警（如果需要）
            if result.status == TaskStatus.FAILED:
                self.alert_manager.send_alert(
                    "TASK_FAILED",
                    f"任务执行失败: {task.name}",
                    task.task_id,
                    "WARNING"
                )
            elif result.status == TaskStatus.COMPLETED:
                self.alert_manager.send_alert(
                    "TASK_COMPLETED",
                    f"任务执行完成: {task.name}",
                    task.task_id,
                    "INFO"
                )
                
        except Exception as e:
            logger.error(f"异步任务执行错误: {e}")
            self.alert_manager.send_alert(
                "TASK_EXECUTION_ERROR",
                f"任务执行异常: {task.name}, 错误: {str(e)}",
                task.task_id,
                "ERROR"
            )
    
    def export_tasks(self, file_path: str):
        """导出任务配置到文件"""
        tasks_data = []
        for task in self.tasks.values():
            task_dict = asdict(task)
            # 转换枚举值为字符串
            task_dict['priority'] = task.priority.value
            # 转换datetime为字符串
            if task.created_time:
                task_dict['created_time'] = task.created_time.isoformat()
            if task.last_run_time:
                task_dict['last_run_time'] = task.last_run_time.isoformat()
            tasks_data.append(task_dict)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"任务配置已导出到: {file_path}")
    
    def import_tasks(self, file_path: str):
        """从文件导入任务配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            tasks_data = json.load(f)
        
        for task_data in tasks_data:
            # 转换字符串为枚举
            if 'priority' in task_data:
                task_data['priority'] = TaskPriority(task_data['priority'])
            
            # 转换字符串为datetime
            if task_data.get('created_time'):
                task_data['created_time'] = datetime.fromisoformat(task_data['created_time'])
            if task_data.get('last_run_time'):
                task_data['last_run_time'] = datetime.fromisoformat(task_data['last_run_time'])
            
            # 移除task_id（导入时重新生成）
            task_data.pop('task_id', None)
            
            # 重新创建任务
            self.create_task(**task_data)
        
        logger.info(f"已从文件导入任务配置: {file_path}")
    
    def cleanup(self):
        """清理资源"""
        # 停止调度器
        if self.running:
            asyncio.create_task(self.stop_scheduler())
        
        # 关闭任务队列
        self.task_queue.shutdown()
        
        logger.info("调度服务资源已清理")


# 便捷函数
def create_simple_task(name: str, func: Callable, interval: float = None, 
                      cron: str = None) -> ScheduledTask:
    """创建简单任务的便捷函数"""
    return ScheduledTask(
        task_id=str(uuid.uuid4()),
        name=name,
        func=func,
        interval=interval,
        cron_expression=cron
    )


def demo_task(name: str, duration: float = 1.0):
    """演示任务函数"""
    print(f"执行任务: {name}, 预计耗时: {duration}秒")
    time.sleep(duration)
    return f"任务 {name} 执行完成"


if __name__ == "__main__":
    # 演示用法
    async def main():
        # 创建调度服务
        scheduler = SchedulerService(max_workers=5)
        
        # 添加告警回调
        def alert_callback(alert):
            print(f"告警: {alert['type']} - {alert['message']}")
        
        scheduler.add_alert_callback(alert_callback)
        
        # 创建任务
        task_id1 = scheduler.create_task(
            name="演示任务1",
            func=demo_task,
            args=("任务1", 2.0),
            interval=10,  # 每10秒执行一次
            priority=TaskPriority.HIGH
        )
        
        task_id2 = scheduler.create_task(
            name="演示任务2", 
            func=demo_task,
            args=("任务2", 1.0),
            cron_expression="*/30 * * * * *",  # 每30秒执行一次
            priority=TaskPriority.NORMAL
        )
        
        # 启动调度器
        await scheduler.start_scheduler()
        
        # 等待一段时间
        await asyncio.sleep(30)
        
        # 停止调度器
        await scheduler.stop_scheduler()
        
        # 显示统计信息
        stats = scheduler.get_performance_stats()
        print(f"性能统计: {stats}")
        
        # 清理资源
        scheduler.cleanup()
    
    # 运行演示
    asyncio.run(main())