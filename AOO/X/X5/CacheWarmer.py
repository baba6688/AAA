"""
X5缓存预热器核心实现

提供完整的缓存预热解决方案，包括数据预热、策略管理、
监控统计等功能模块。

作者：X5开发团队
版本：1.0.0
日期：2025-11-06
"""

import asyncio
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import hashlib
import pickle
import os


class PreheatingStatus(Enum):
    """预热状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class PreheatingTask:
    """预热任务数据类"""
    task_id: str
    name: str
    data_source: str
    cache_keys: List[str]
    priority: int = 1
    status: PreheatingStatus = PreheatingStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class PreheatingConfig:
    """预热配置管理类"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = {
            "max_concurrent_tasks": 5,
            "preheating_timeout": 300,
            "retry_attempts": 3,
            "retry_delay": 5,
            "batch_size": 100,
            "monitor_interval": 1,
            "statistics_retention_days": 30,
            "enable_monitoring": True,
            "enable_statistics": True,
            "enable_optimization": True,
            "cache_ttl": 3600,
            "preheating_strategies": {
                "priority_based": True,
                "access_frequency": True,
                "data_size_optimization": True,
                "dependency_aware": True
            },
            "monitoring": {
                "real_time_monitoring": True,
                "alert_thresholds": {
                    "memory_usage": 80,
                    "cpu_usage": 70,
                    "preheating_time": 600
                }
            },
            "statistics": {
                "collect_detailed_metrics": True,
                "export_formats": ["json", "csv"],
                "aggregation_intervals": ["hourly", "daily", "weekly"]
            }
        }
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def load_from_file(self, config_file: str) -> None:
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self._merge_config(self.config, file_config)
        except Exception as e:
            logging.warning(f"加载配置文件失败: {e}")
    
    def save_to_file(self, config_file: str) -> None:
        """保存配置到文件"""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
    
    def _merge_config(self, base: Dict, update: Dict) -> None:
        """合并配置"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value


class PreheatingStrategy:
    """预热策略管理类"""
    
    def __init__(self, config: PreheatingConfig):
        self.config = config
        self.strategies = {
            "priority_based": self._priority_based_strategy,
            "access_frequency": self._access_frequency_strategy,
            "data_size_optimization": self._data_size_optimization_strategy,
            "dependency_aware": self._dependency_aware_strategy
        }
    
    def select_strategy(self, tasks: List[PreheatingTask]) -> str:
        """选择最优预热策略"""
        enabled_strategies = [
            name for name, enabled in self.config.get("preheating_strategies", {}).items()
            if enabled
        ]
        
        if not enabled_strategies:
            return "priority_based"
        
        # 根据任务特征选择策略
        if len(tasks) > 100:
            return "data_size_optimization"
        elif any(task.priority > 5 for task in tasks):
            return "priority_based"
        elif self._has_access_frequency_data():
            return "access_frequency"
        else:
            return enabled_strategies[0]
    
    def sort_tasks(self, tasks: List[PreheatingTask], strategy: str) -> List[PreheatingTask]:
        """根据策略排序任务"""
        if strategy in self.strategies:
            return self.strategies[strategy](tasks)
        return tasks
    
    def _priority_based_strategy(self, tasks: List[PreheatingTask]) -> List[PreheatingTask]:
        """基于优先级的策略"""
        return sorted(tasks, key=lambda t: (-t.priority, t.created_at))
    
    def _access_frequency_strategy(self, tasks: List[PreheatingTask]) -> List[PreheatingTask]:
        """基于访问频率的策略"""
        # 模拟访问频率数据
        frequency_data = self._get_access_frequency_data()
        return sorted(tasks, key=lambda t: -frequency_data.get(t.cache_keys[0], 0))
    
    def _data_size_optimization_strategy(self, tasks: List[PreheatingTask]) -> List[PreheatingTask]:
        """基于数据大小的优化策略"""
        return sorted(tasks, key=lambda t: len(str(t.metadata.get('data', ''))))
    
    def _dependency_aware_strategy(self, tasks: List[PreheatingTask]) -> List[PreheatingTask]:
        """基于依赖关系的策略"""
        # 简单的依赖解析
        dependency_graph = self._build_dependency_graph(tasks)
        return self._topological_sort(tasks, dependency_graph)
    
    def _has_access_frequency_data(self) -> bool:
        """检查是否有访问频率数据"""
        return True  # 简化实现
    
    def _get_access_frequency_data(self) -> Dict[str, int]:
        """获取访问频率数据"""
        return {}  # 简化实现
    
    def _build_dependency_graph(self, tasks: List[PreheatingTask]) -> Dict[str, List[str]]:
        """构建依赖图"""
        graph = defaultdict(list)
        for task in tasks:
            dependencies = task.metadata.get('dependencies', [])
            for dep in dependencies:
                graph[dep].append(task.task_id)
        return graph
    
    def _topological_sort(self, tasks: List[PreheatingTask], graph: Dict[str, List[str]]) -> List[PreheatingTask]:
        """拓扑排序"""
        task_map = {t.task_id: t for t in tasks}
        in_degree = defaultdict(int)
        
        # 计算入度
        for task_id in task_map:
            in_degree[task_id] = 0
        
        for dependencies in graph.values():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # 拓扑排序
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(task_map[current])
            
            for dependent in graph.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return result


class DataPreloader:
    """数据预加载器"""
    
    def __init__(self, config: PreheatingConfig):
        self.config = config
        self.cache_backend = {}  # 模拟缓存后端
    
    async def preload_data(self, task: PreheatingTask) -> bool:
        """预加载数据"""
        try:
            logging.info(f"开始预加载任务: {task.name}")
            
            # 模拟数据加载
            batch_size = self.config.get("batch_size", 100)
            total_items = len(task.cache_keys)
            loaded_items = 0
            
            for i in range(0, total_items, batch_size):
                batch = task.cache_keys[i:i + batch_size]
                await self._load_batch(task, batch)
                loaded_items += len(batch)
                
                # 更新进度
                task.progress = (loaded_items / total_items) * 100
                
                # 模拟处理时间
                await asyncio.sleep(0.1)
            
            task.status = PreheatingStatus.COMPLETED
            task.completed_at = datetime.now()
            logging.info(f"任务预加载完成: {task.name}")
            return True
            
        except Exception as e:
            task.status = PreheatingStatus.FAILED
            task.error_message = str(e)
            logging.error(f"任务预加载失败: {task.name}, 错误: {e}")
            return False
    
    async def _load_batch(self, task: PreheatingTask, batch: List[str]) -> None:
        """加载一批数据"""
        # 模拟从数据源加载数据
        for key in batch:
            # 模拟数据生成
            data = {
                "key": key,
                "value": f"preloaded_data_{key}",
                "timestamp": datetime.now().isoformat(),
                "task_id": task.task_id
            }
            
            # 存储到缓存
            cache_key = f"cache:{key}"
            self.cache_backend[cache_key] = data
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        cache_key = f"cache:{key}"
        return self.cache_backend.get(cache_key)


class PreheatingManager:
    """预热管理类"""
    
    def __init__(self, config: PreheatingConfig):
        self.config = config
        self.tasks: Dict[str, PreheatingTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.preloader = DataPreloader(config)
        self.strategy = PreheatingStrategy(config)
        self.lock = threading.Lock()
    
    async def create_task(self, name: str, data_source: str, cache_keys: List[str], 
                         priority: int = 1, metadata: Dict[str, Any] = None) -> str:
        """创建预热任务"""
        task_id = self._generate_task_id()
        task = PreheatingTask(
            task_id=task_id,
            name=name,
            data_source=data_source,
            cache_keys=cache_keys,
            priority=priority,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.tasks[task_id] = task
        
        logging.info(f"创建预热任务: {name} (ID: {task_id})")
        return task_id
    
    async def start_task(self, task_id: str) -> bool:
        """启动预热任务"""
        with self.lock:
            if task_id not in self.tasks:
                logging.error(f"任务不存在: {task_id}")
                return False
            
            task = self.tasks[task_id]
            if task.status != PreheatingStatus.PENDING:
                logging.warning(f"任务状态不正确: {task_id}, 状态: {task.status}")
                return False
        
        try:
            task.status = PreheatingStatus.RUNNING
            task.started_at = datetime.now()
            
            # 创建异步任务
            async_task = asyncio.create_task(self._execute_task(task))
            self.running_tasks[task_id] = async_task
            
            logging.info(f"启动预热任务: {task.name}")
            return True
            
        except Exception as e:
            task.status = PreheatingStatus.FAILED
            task.error_message = str(e)
            logging.error(f"启动任务失败: {task_id}, 错误: {e}")
            return False
    
    async def stop_task(self, task_id: str) -> bool:
        """停止预热任务"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
        
        try:
            if task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
                del self.running_tasks[task_id]
            
            task.status = PreheatingStatus.CANCELLED
            logging.info(f"停止预热任务: {task.name}")
            return True
            
        except Exception as e:
            logging.error(f"停止任务失败: {task_id}, 错误: {e}")
            return False
    
    async def _execute_task(self, task: PreheatingTask) -> None:
        """执行预热任务"""
        try:
            await self.preloader.preload_data(task)
        except asyncio.CancelledError:
            task.status = PreheatingStatus.CANCELLED
            logging.info(f"任务被取消: {task.name}")
        except Exception as e:
            task.status = PreheatingStatus.FAILED
            task.error_message = str(e)
            logging.error(f"任务执行失败: {task.name}, 错误: {e}")
        finally:
            with self.lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
    
    def get_task_status(self, task_id: str) -> Optional[PreheatingTask]:
        """获取任务状态"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def list_tasks(self, status: Optional[PreheatingStatus] = None) -> List[PreheatingTask]:
        """列出任务"""
        with self.lock:
            tasks = list(self.tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            return tasks
    
    def _generate_task_id(self) -> str:
        """生成任务ID"""
        timestamp = str(int(time.time() * 1000))
        random_str = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"task_{timestamp}_{random_str}"


class PreheatingMonitor:
    """预热监控类"""
    
    def __init__(self, manager: PreheatingManager, config: PreheatingConfig):
        self.manager = manager
        self.config = config
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.metrics: Dict[str, Any] = {}
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable) -> None:
        """添加监控回调"""
        self.callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logging.info("开始预热监控")
    
    async def stop_monitoring(self) -> None:
        """停止监控"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logging.info("停止预热监控")
    
    async def _monitor_loop(self) -> None:
        """监控循环"""
        interval = self.config.get("monitor_interval", 1)
        
        while self.monitoring:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await self._notify_callbacks()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"监控循环错误: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_metrics(self) -> None:
        """收集指标"""
        tasks = self.manager.list_tasks()
        
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "running_tasks": len([t for t in tasks if t.status == PreheatingStatus.RUNNING]),
            "completed_tasks": len([t for t in tasks if t.status == PreheatingStatus.COMPLETED]),
            "failed_tasks": len([t for t in tasks if t.status == PreheatingStatus.FAILED]),
            "average_progress": sum(t.progress for t in tasks) / len(tasks) if tasks else 0,
            "total_cache_keys": sum(len(t.cache_keys) for t in tasks),
            "processed_keys": sum(int(t.progress / 100 * len(t.cache_keys)) for t in tasks)
        }
    
    async def _check_alerts(self) -> None:
        """检查告警"""
        thresholds = self.config.get("monitoring.alert_thresholds", {})
        
        # 检查预热时间告警
        max_preheating_time = thresholds.get("preheating_time", 600)
        running_tasks = [t for t in self.manager.list_tasks() if t.status == PreheatingStatus.RUNNING]
        
        for task in running_tasks:
            if task.started_at:
                elapsed = (datetime.now() - task.started_at).total_seconds()
                if elapsed > max_preheating_time:
                    logging.warning(f"预热任务超时: {task.name}, 已运行 {elapsed:.0f} 秒")
    
    async def _notify_callbacks(self) -> None:
        """通知回调"""
        for callback in self.callbacks:
            try:
                await callback(self.metrics)
            except Exception as e:
                logging.error(f"监控回调错误: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        return self.metrics.copy()


class PreheatingStatistics:
    """预热统计类"""
    
    def __init__(self, config: PreheatingConfig):
        self.config = config
        self.statistics_data: Dict[str, Any] = defaultdict(list)
        self.performance_metrics: Dict[str, Any] = {}
    
    def record_task_start(self, task: PreheatingTask) -> None:
        """记录任务开始"""
        self.statistics_data["task_starts"].append({
            "task_id": task.task_id,
            "name": task.name,
            "start_time": task.started_at.isoformat() if task.started_at else None,
            "priority": task.priority,
            "cache_keys_count": len(task.cache_keys)
        })
    
    def record_task_completion(self, task: PreheatingTask) -> None:
        """记录任务完成"""
        if task.started_at and task.completed_at:
            duration = (task.completed_at - task.started_at).total_seconds()
            
            self.statistics_data["task_completions"].append({
                "task_id": task.task_id,
                "name": task.name,
                "duration": duration,
                "status": task.status.value,
                "cache_keys_count": len(task.cache_keys),
                "keys_per_second": len(task.cache_keys) / duration if duration > 0 else 0
            })
            
            # 更新性能指标
            self._update_performance_metrics(task, duration)
    
    def _update_performance_metrics(self, task: PreheatingTask, duration: float) -> None:
        """更新性能指标"""
        keys_count = len(task.cache_keys)
        throughput = keys_count / duration if duration > 0 else 0
        
        if "average_throughput" not in self.performance_metrics:
            self.performance_metrics["average_throughput"] = []
        
        self.performance_metrics["average_throughput"].append(throughput)
        
        # 保持最近100次记录
        if len(self.performance_metrics["average_throughput"]) > 100:
            self.performance_metrics["average_throughput"] = self.performance_metrics["average_throughput"][-100:]
    
    def get_statistics(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """获取统计信息"""
        if time_range:
            cutoff_time = datetime.now() - time_range
            filtered_data = {}
            
            for key, values in self.statistics_data.items():
                if key in ["task_starts", "task_completions"]:
                    filtered_values = [
                        v for v in values 
                        if datetime.fromisoformat(v.get("start_time", v.get("completion_time", ""))) > cutoff_time
                    ]
                    filtered_data[key] = filtered_values
                else:
                    filtered_data[key] = values
            
            return dict(filtered_data)
        
        return dict(self.statistics_data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        throughput_data = self.performance_metrics.get("average_throughput", [])
        
        if not throughput_data:
            return {"message": "暂无性能数据"}
        
        return {
            "average_throughput": sum(throughput_data) / len(throughput_data),
            "max_throughput": max(throughput_data),
            "min_throughput": min(throughput_data),
            "total_tasks_completed": len(self.statistics_data["task_completions"]),
            "total_cache_keys_processed": sum(
                comp.get("cache_keys_count", 0) 
                for comp in self.statistics_data["task_completions"]
            )
        }
    
    def export_statistics(self, format_type: str = "json") -> str:
        """导出统计信息"""
        data = {
            "statistics": self.get_statistics(),
            "performance": self.get_performance_summary(),
            "export_time": datetime.now().isoformat()
        }
        
        if format_type == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif format_type == "csv":
            return self._export_to_csv(data)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")
    
    def _export_to_csv(self, data: Dict[str, Any]) -> str:
        """导出为CSV格式"""
        csv_lines = ["类别,指标,值"]
        
        # 添加性能摘要
        performance = data.get("performance", {})
        for key, value in performance.items():
            csv_lines.append(f"性能,{key},{value}")
        
        # 添加任务统计
        stats = data.get("statistics", {})
        for category, values in stats.items():
            csv_lines.append(f"统计,{category},{len(values)}")
        
        return "\n".join(csv_lines)


class PreheatingOptimizer:
    """预热优化类"""
    
    def __init__(self, config: PreheatingConfig, statistics: PreheatingStatistics):
        self.config = config
        self.statistics = statistics
        self.optimization_history: List[Dict[str, Any]] = []
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析性能并提供优化建议"""
        performance = self.statistics.get_performance_summary()
        
        recommendations = []
        
        # 分析吞吐量
        avg_throughput = performance.get("average_throughput", 0)
        if avg_throughput < 100:  # 低于100 keys/秒
            recommendations.append({
                "type": "performance",
                "issue": "吞吐量较低",
                "suggestion": "考虑增加并发任务数或优化数据加载逻辑",
                "priority": "high"
            })
        
        # 分析任务完成率
        total_tasks = performance.get("total_tasks_completed", 0)
        if total_tasks > 0:
            # 这里可以分析失败率等指标
            pass
        
        return {
            "performance_metrics": performance,
            "recommendations": recommendations,
            "optimization_score": self._calculate_optimization_score(performance)
        }
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """优化配置参数"""
        current_config = self.config.config.copy()
        optimizations = []
        
        # 基于性能数据优化并发数
        performance = self.statistics.get_performance_summary()
        avg_throughput = performance.get("average_throughput", 0)
        
        if avg_throughput > 0:
            current_concurrent = self.config.get("max_concurrent_tasks", 5)
            
            # 如果吞吐量较高，可以适当增加并发数
            if avg_throughput > 500:
                new_concurrent = min(current_concurrent + 1, 10)
                optimizations.append({
                    "parameter": "max_concurrent_tasks",
                    "current_value": current_concurrent,
                    "suggested_value": new_concurrent,
                    "reason": "高吞吐量支持增加并发"
                })
            elif avg_throughput < 100:
                new_concurrent = max(current_concurrent - 1, 2)
                optimizations.append({
                    "parameter": "max_concurrent_tasks", 
                    "current_value": current_concurrent,
                    "suggested_value": new_concurrent,
                    "reason": "低吞吐量建议减少并发"
                })
        
        return {
            "current_config": current_config,
            "optimizations": optimizations,
            "estimated_improvement": self._estimate_improvement(optimizations)
        }
    
    def _calculate_optimization_score(self, performance: Dict[str, Any]) -> float:
        """计算优化分数"""
        score = 0.0
        
        # 吞吐量分数 (0-40分)
        throughput = performance.get("average_throughput", 0)
        if throughput > 500:
            score += 40
        elif throughput > 200:
            score += 30
        elif throughput > 100:
            score += 20
        else:
            score += 10
        
        # 任务完成率分数 (0-30分)
        total_tasks = performance.get("total_tasks_completed", 0)
        if total_tasks > 50:
            score += 30
        elif total_tasks > 20:
            score += 20
        elif total_tasks > 10:
            score += 10
        
        # 处理键数量分数 (0-30分)
        total_keys = performance.get("total_cache_keys_processed", 0)
        if total_keys > 10000:
            score += 30
        elif total_keys > 5000:
            score += 20
        elif total_keys > 1000:
            score += 10
        
        return min(score, 100)
    
    def _estimate_improvement(self, optimizations: List[Dict[str, Any]]) -> str:
        """估算改进效果"""
        if not optimizations:
            return "无需优化"
        
        improvement_types = [opt["type"] for opt in optimizations if "type" in opt]
        
        if "concurrency" in improvement_types:
            return "预计可提升20-30%的处理速度"
        elif "batch_size" in improvement_types:
            return "预计可提升10-15%的内存效率"
        else:
            return "预计可提升5-10%的整体性能"


class PreheatingReporter:
    """预热报告类"""
    
    def __init__(self, manager: PreheatingManager, monitor: PreheatingMonitor, 
                 statistics: PreheatingStatistics, optimizer: PreheatingOptimizer):
        self.manager = manager
        self.monitor = monitor
        self.statistics = statistics
        self.optimizer = optimizer
    
    def generate_report(self, report_type: str = "summary") -> str:
        """生成报告"""
        if report_type == "summary":
            return self._generate_summary_report()
        elif report_type == "detailed":
            return self._generate_detailed_report()
        elif report_type == "performance":
            return self._generate_performance_report()
        else:
            raise ValueError(f"不支持的报告类型: {report_type}")
    
    def _generate_summary_report(self) -> str:
        """生成摘要报告"""
        metrics = self.monitor.get_metrics()
        performance = self.statistics.get_performance_summary()
        
        report = f"""
# X5缓存预热器 - 摘要报告

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 任务概览
- 总任务数: {metrics.get('total_tasks', 0)}
- 运行中任务: {metrics.get('running_tasks', 0)}
- 已完成任务: {metrics.get('completed_tasks', 0)}
- 失败任务: {metrics.get('failed_tasks', 0)}

## 性能指标
- 平均吞吐量: {performance.get('average_throughput', 0):.2f} keys/秒
- 最大吞吐量: {performance.get('max_throughput', 0):.2f} keys/秒
- 处理键总数: {performance.get('total_cache_keys_processed', 0)}

## 当前状态
- 平均进度: {metrics.get('average_progress', 0):.1f}%
- 总缓存键数: {metrics.get('total_cache_keys', 0)}
- 已处理键数: {metrics.get('processed_keys', 0)}
        """
        
        return report.strip()
    
    def _generate_detailed_report(self) -> str:
        """生成详细报告"""
        summary = self._generate_summary_report()
        tasks = self.manager.list_tasks()
        statistics = self.statistics.get_statistics()
        
        task_details = []
        for task in tasks:
            task_details.append(f"""
### 任务: {task.name}
- 任务ID: {task.task_id}
- 状态: {task.status.value}
- 优先级: {task.priority}
- 缓存键数量: {len(task.cache_keys)}
- 进度: {task.progress:.1f}%
- 创建时间: {task.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- 开始时间: {task.started_at.strftime('%Y-%m-%d %H:%M:%S') if task.started_at else '未开始'}
- 完成时间: {task.completed_at.strftime('%Y-%m-%d %H:%M:%S') if task.completed_at else '未完成'}
            """.strip())
        
        detailed_report = f"""
{summary}

## 任务详情
{''.join(task_details)}

## 统计详情
- 任务启动记录: {len(statistics.get('task_starts', []))} 条
- 任务完成记录: {len(statistics.get('task_completions', []))} 条
        """
        
        return detailed_report.strip()
    
    def _generate_performance_report(self) -> str:
        """生成性能报告"""
        performance = self.statistics.get_performance_summary()
        optimization_analysis = self.optimizer.analyze_performance()
        config_optimization = self.optimizer.optimize_configuration()
        
        recommendations = []
        for rec in optimization_analysis.get("recommendations", []):
            recommendations.append(f"- **{rec['type']}**: {rec['suggestion']} (优先级: {rec['priority']})")
        
        optimizations = []
        for opt in config_optimization.get("optimizations", []):
            optimizations.append(f"- **{opt['parameter']}**: {opt['current_value']} → {opt['suggested_value']} ({opt['reason']})")
        
        performance_report = f"""
# X5缓存预热器 - 性能报告

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 性能指标
- 平均吞吐量: {performance.get('average_throughput', 0):.2f} keys/秒
- 最大吞吐量: {performance.get('max_throughput', 0):.2f} keys/秒
- 最小吞吐量: {performance.get('min_throughput', 0):.2f} keys/秒
- 总完成任务: {performance.get('total_tasks_completed', 0)}
- 总处理键数: {performance.get('total_cache_keys_processed', 0)}

## 优化分析
- 优化分数: {optimization_analysis.get('optimization_score', 0):.1f}/100

### 优化建议
{chr(10).join(recommendations) if recommendations else '暂无优化建议'}

### 配置优化
{chr(10).join(optimizations) if optimizations else '无需配置优化'}

## 预期改进
{config_optimization.get('estimated_improvement', '暂无改进预期')}
        """
        
        return performance_report.strip()
    
    def export_report(self, filename: str, report_type: str = "summary") -> bool:
        """导出报告到文件"""
        try:
            report_content = self.generate_report(report_type)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logging.info(f"报告已导出到: {filename}")
            return True
            
        except Exception as e:
            logging.error(f"导出报告失败: {e}")
            return False


class CacheWarmer:
    """X5缓存预热器主类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化缓存预热器"""
        self.config = PreheatingConfig(config_file)
        self.manager = PreheatingManager(self.config)
        self.monitor = PreheatingMonitor(self.manager, self.config)
        self.statistics = PreheatingStatistics(self.config)
        self.optimizer = PreheatingOptimizer(self.config, self.statistics)
        self.reporter = PreheatingReporter(self.manager, self.monitor, self.statistics, self.optimizer)
        
        # 设置监控回调
        self.monitor.add_callback(self._on_metrics_update)
        
        logging.info("X5缓存预热器初始化完成")
    
    async def start(self) -> None:
        """启动缓存预热器"""
        if self.config.get("enable_monitoring", True):
            await self.monitor.start_monitoring()
        logging.info("缓存预热器已启动")
    
    async def stop(self) -> None:
        """停止缓存预热器"""
        await self.monitor.stop_monitoring()
        logging.info("缓存预热器已停止")
    
    async def create_preheating_task(self, name: str, data_source: str, 
                                   cache_keys: List[str], priority: int = 1,
                                   metadata: Dict[str, Any] = None) -> str:
        """创建预热任务"""
        return await self.manager.create_task(name, data_source, cache_keys, priority, metadata)
    
    async def start_preheating(self, task_id: str) -> bool:
        """开始预热"""
        return await self.manager.start_task(task_id)
    
    async def stop_preheating(self, task_id: str) -> bool:
        """停止预热"""
        return await self.manager.stop_task(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[PreheatingTask]:
        """获取任务状态"""
        return self.manager.get_task_status(task_id)
    
    def list_tasks(self, status: Optional[PreheatingStatus] = None) -> List[PreheatingTask]:
        """列出任务"""
        return self.manager.list_tasks(status)
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return self.monitor.get_metrics()
    
    def get_statistics(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """获取统计信息"""
        return self.statistics.get_statistics(time_range)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return self.statistics.get_performance_summary()
    
    def analyze_optimization(self) -> Dict[str, Any]:
        """分析优化建议"""
        return self.optimizer.analyze_performance()
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """优化配置"""
        return self.optimizer.optimize_configuration()
    
    def generate_report(self, report_type: str = "summary") -> str:
        """生成报告"""
        return self.reporter.generate_report(report_type)
    
    def export_report(self, filename: str, report_type: str = "summary") -> bool:
        """导出报告"""
        return self.reporter.export_report(filename, report_type)
    
    async def _on_metrics_update(self, metrics: Dict[str, Any]) -> None:
        """指标更新回调"""
        # 记录任务统计
        tasks = self.manager.list_tasks()
        for task in tasks:
            if task.status == PreheatingStatus.RUNNING and task.started_at:
                self.statistics.record_task_start(task)
            elif task.status == PreheatingStatus.COMPLETED and task.completed_at:
                self.statistics.record_task_completion(task)
    
    async def preload_system_data(self, data_config: Dict[str, Any]) -> List[str]:
        """预加载系统数据"""
        task_ids = []
        
        for category, config in data_config.items():
            cache_keys = config.get("cache_keys", [])
            if cache_keys:
                task_id = await self.create_preheating_task(
                    name=f"系统数据预加载-{category}",
                    data_source=config.get("data_source", "system"),
                    cache_keys=cache_keys,
                    priority=config.get("priority", 1),
                    metadata={"category": category, "type": "system_data"}
                )
                task_ids.append(task_id)
                
                # 自动启动任务
                await self.start_preheating(task_id)
        
        return task_ids


# 便捷函数
async def create_cache_warmer(config_file: Optional[str] = None) -> CacheWarmer:
    """创建缓存预热器实例"""
    warmer = CacheWarmer(config_file)
    await warmer.start()
    return warmer


def quick_preload(cache_keys: List[str], data_source: str = "quick") -> str:
    """快速预加载接口"""
    # 这里简化实现，实际应用中可能需要更复杂的逻辑
    return f"quick_task_{hashlib.md5(str(cache_keys).encode()).hexdigest()[:8]}"


# 导出主要类和函数
__all__ = [
    "CacheWarmer",
    "PreheatingStrategy", 
    "PreheatingManager",
    "PreheatingMonitor",
    "PreheatingStatistics",
    "PreheatingOptimizer",
    "PreheatingConfig",
    "PreheatingReporter",
    "DataPreloader",
    "PreheatingTask",
    "PreheatingStatus",
    "create_cache_warmer",
    "quick_preload"
]