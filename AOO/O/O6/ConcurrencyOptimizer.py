#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O6并发优化器模块

该模块提供了全面的并发编程优化解决方案，包括：
- 线程池优化（动态调整、任务调度、负载均衡）
- 锁机制优化（无锁数据结构、读写锁、乐观锁）
- 协程和异步优化（协程池、事件循环优化）
- 进程间通信优化（管道、队列、共享内存）
- 并发控制优化（信号量、屏障、条件变量）
- 死锁检测和预防
- 异步并发优化处理

版本: 1.0.0
作者: O6 Team
创建时间: 2025-11-06
"""

import asyncio
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import time
import logging
import weakref
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set, Awaitable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
import warnings
import inspect
import os
import sys
from functools import wraps, lru_cache
from threading import RLock, Condition, Event, Semaphore, Barrier, Lock
import ctypes
import traceback
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """优化级别枚举"""
    CONSERVATIVE = auto()  # 保守优化
    BALANCED = auto()      # 平衡优化
    AGGRESSIVE = auto()    # 激进优化
    ADAPTIVE = auto()      # 自适应优化


class LockType(Enum):
    """锁类型枚举"""
    MUTEX = auto()         # 互斥锁
    READ_WRITE = auto()    # 读写锁
    OPTIMISTIC = auto()    # 乐观锁
    SPIN_LOCK = auto()     # 自旋锁


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """任务数据类"""
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    task_id: Optional[str] = None
    
    def __lt__(self, other):
        """优先级比较"""
        return self.priority.value > other.priority.value


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    thread_count: int = 0
    task_queue_size: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0


class OptimizedLock:
    """优化的锁基类"""
    
    def __init__(self, lock_type: LockType):
        self.lock_type = lock_type
        self._lock = None
        self._readers = 0
        self._writers = 0
        self._reader_waiting = 0
        self._writer_waiting = 0
        self._condition = Condition()
        
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取锁"""
        raise NotImplementedError
        
    def release(self):
        """释放锁"""
        raise NotImplementedError
        
    def __enter__(self):
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class MutexLock(OptimizedLock):
    """互斥锁实现"""
    
    def __init__(self):
        super().__init__(LockType.MUTEX)
        self._lock = Lock()
        
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取互斥锁"""
        try:
            if timeout is None:
                self._lock.acquire()
                return True
            else:
                return self._lock.acquire(blocking=blocking, timeout=timeout)
        except Exception as e:
            logger.error(f"获取互斥锁失败: {e}")
            return False
            
    def release(self):
        """释放互斥锁"""
        try:
            self._lock.release()
        except Exception as e:
            logger.error(f"释放互斥锁失败: {e}")


class ReadWriteLock(OptimizedLock):
    """读写锁实现"""
    
    def __init__(self):
        super().__init__(LockType.READ_WRITE)
        self._readers = 0
        self._writers = 0
        self._reader_waiting = 0
        self._writer_waiting = 0
        self._condition = Condition()
        
    def acquire_read(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取读锁"""
        start_time = time.time()
        with self._condition:
            while self._writers > 0:
                if not blocking:
                    return False
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        return False
                    remaining = timeout - elapsed
                    if not self._condition.wait(remaining):
                        return False
                else:
                    self._condition.wait()
            
            self._readers += 1
            return True
            
    def acquire_write(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取写锁"""
        start_time = time.time()
        with self._condition:
            while self._writers > 0 or self._readers > 0:
                if not blocking:
                    return False
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        return False
                    remaining = timeout - elapsed
                    if not self._condition.wait(remaining):
                        return False
                else:
                    self._condition.wait()
            
            self._writers += 1
            return True
            
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取锁（默认写锁）"""
        return self.acquire_write(blocking, timeout)
        
    def release(self):
        """释放锁"""
        with self._condition:
            if self._writers > 0:
                self._writers -= 1
            elif self._readers > 0:
                self._readers -= 1
            
            if self._writers == 0 and self._readers == 0:
                self._condition.notify_all()


class OptimisticLock(OptimizedLock):
    """乐观锁实现"""
    
    def __init__(self):
        super().__init__(LockType.OPTIMISTIC)
        self._version = 0
        self._lock = RLock()
        
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取乐观锁（版本检查）"""
        start_time = time.time()
        while True:
            with self._lock:
                current_version = self._version
                # 这里应该检查业务逻辑条件
                # 如果条件满足，增加版本号并返回True
                if self._check_condition():
                    self._version += 1
                    return True
                    
            if not blocking:
                return False
                
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                    
            time.sleep(0.001)  # 短暂等待
            
    def _check_condition(self) -> bool:
        """检查乐观锁条件"""
        # 这里应该实现具体的业务逻辑检查
        return True
        
    def release(self):
        """释放乐观锁"""
        # 乐观锁通常不需要显式释放
        pass


class LockFactory:
    """锁工厂类"""
    
    _lock_pools: Dict[LockType, weakref.WeakSet] = {
        LockType.MUTEX: weakref.WeakSet(),
        LockType.READ_WRITE: weakref.WeakSet(),
        LockType.OPTIMISTIC: weakref.WeakSet(),
        LockType.SPIN_LOCK: weakref.WeakSet()
    }
    
    @classmethod
    def create_lock(cls, lock_type: LockType) -> OptimizedLock:
        """创建锁实例"""
        # 尝试从池中获取现有锁
        for lock in cls._lock_pools[lock_type]:
            if isinstance(lock, MutexLock) and lock_type == LockType.MUTEX:
                return lock
            elif isinstance(lock, ReadWriteLock) and lock_type == LockType.READ_WRITE:
                return lock
            elif isinstance(lock, OptimisticLock) and lock_type == LockType.OPTIMISTIC:
                return lock
        
        # 创建新锁
        if lock_type == LockType.MUTEX:
            lock = MutexLock()
        elif lock_type == LockType.READ_WRITE:
            lock = ReadWriteLock()
        elif lock_type == LockType.OPTIMISTIC:
            lock = OptimisticLock()
        else:
            raise ValueError(f"不支持的锁类型: {lock_type}")
            
        cls._lock_pools[lock_type].add(lock)
        return lock


class LockOptimizer:
    """锁机制优化器"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self._lock_stats = defaultdict(int)
        self._lock_factory = LockFactory()
        self._deadlock_detector = DeadlockDetector()
        
    def create_optimized_lock(self, lock_type: LockType) -> OptimizedLock:
        """创建优化的锁"""
        lock = self._lock_factory.create_lock(lock_type)
        self._lock_stats[lock_type] += 1
        logger.debug(f"创建{lock_type}锁，当前统计: {self._lock_stats}")
        return lock
        
    def optimize_data_structure(self, data_structure: Any) -> Any:
        """优化数据结构"""
        if hasattr(data_structure, '__getitem__') and hasattr(data_structure, '__setitem__'):
            # 为列表、字典等添加锁保护
            return self._add_lock_to_collection(data_structure)
        return data_structure
        
    def _add_lock_to_collection(self, collection: Any) -> Any:
        """为集合添加锁保护"""
        if isinstance(collection, list):
            return LockedList(collection)
        elif isinstance(collection, dict):
            return LockedDict(collection)
        elif isinstance(collection, set):
            return LockedSet(collection)
        return collection
        
    def get_lock_statistics(self) -> Dict[str, int]:
        """获取锁统计信息"""
        return dict(self._lock_stats)


class LockedList:
    """带锁保护的列表"""
    
    def __init__(self, initial_list: Optional[List] = None):
        self._list = initial_list or []
        self._lock = Lock()
        
    def append(self, item):
        with self._lock:
            return self._list.append(item)
            
    def extend(self, items):
        with self._lock:
            return self._list.extend(items)
            
    def __getitem__(self, index):
        with self._lock:
            return self._list[index]
            
    def __setitem__(self, index, value):
        with self._lock:
            self._list[index] = value
            
    def __len__(self):
        with self._lock:
            return len(self._list)
            
    def __iter__(self):
        with self._lock:
            return iter(self._list.copy())


class LockedDict:
    """带锁保护的字典"""
    
    def __init__(self, initial_dict: Optional[Dict] = None):
        self._dict = initial_dict or {}
        self._lock = Lock()
        
    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]
            
    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value
            
    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]
            
    def __contains__(self, key):
        with self._lock:
            return key in self._dict
            
    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)
            
    def keys(self):
        with self._lock:
            return self._dict.keys()
            
    def values(self):
        with self._lock:
            return self._dict.values()
            
    def items(self):
        with self._lock:
            return self._dict.items()


class LockedSet:
    """带锁保护的集合"""
    
    def __init__(self, initial_set: Optional[Set] = None):
        self._set = initial_set or set()
        self._lock = Lock()
        
    def add(self, item):
        with self._lock:
            return self._set.add(item)
            
    def remove(self, item):
        with self._lock:
            return self._set.remove(item)
            
    def discard(self, item):
        with self._lock:
            return self._set.discard(item)
            
    def __contains__(self, item):
        with self._lock:
            return item in self._set
            
    def __len__(self):
        with self._lock:
            return len(self._set)
            
    def __iter__(self):
        with self._lock:
            return iter(self._set.copy())


class DeadlockDetector:
    """死锁检测器"""
    
    def __init__(self):
        self._wait_for_graph = defaultdict(set)
        self._lock_owners = {}
        self._detection_interval = 1.0
        self._is_running = False
        self._detection_thread = None
        
    def start_detection(self):
        """启动死锁检测"""
        if not self._is_running:
            self._is_running = True
            self._detection_thread = threading.Thread(target=self._detection_loop)
            self._detection_thread.daemon = True
            self._detection_thread.start()
            logger.info("死锁检测器已启动")
            
    def stop_detection(self):
        """停止死锁检测"""
        self._is_running = False
        if self._detection_thread:
            self._detection_thread.join()
        logger.info("死锁检测器已停止")
        
    def register_lock_acquisition(self, thread_id: int, lock_id: str):
        """注册锁获取"""
        self._lock_owners[lock_id] = thread_id
        
    def register_lock_wait(self, thread_id: int, waiting_for_lock: str):
        """注册锁等待"""
        # 查找该线程拥有的锁
        owned_locks = [lock_id for lock_id, owner in self._lock_owners.items() 
                      if owner == thread_id]
        
        for owned_lock in owned_locks:
            self._wait_for_graph[owned_lock].add(waiting_for_lock)
            
    def _detection_loop(self):
        """死锁检测循环"""
        while self._is_running:
            try:
                self._check_deadlock()
                time.sleep(self._detection_interval)
            except Exception as e:
                logger.error(f"死锁检测错误: {e}")
                
    def _check_deadlock(self):
        """检查死锁"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            """检查是否有环（死锁）"""
            if node in rec_stack:
                return True
            if node in visited:
                return False
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self._wait_for_graph.get(node, set()):
                if has_cycle(neighbor):
                    return True
                    
            rec_stack.remove(node)
            return False
        
        for node in self._wait_for_graph:
            if node not in visited:
                if has_cycle(node):
                    self._handle_deadlock()
                    break
                    
    def _handle_deadlock(self):
        """处理死锁"""
        logger.warning("检测到死锁！尝试解决...")
        # 这里可以实现死锁解决策略
        # 例如：强制终止某些线程、回滚操作等


class OptimizedThreadPool:
    """优化的线程池"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 min_workers: int = 2,
                 queue_size: int = 1000,
                 keepalive_time: float = 60.0,
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        
        self.max_workers = max_workers or min(mp.cpu_count() * 4, 32)
        self.min_workers = min_workers
        self.queue_size = queue_size
        self.keepalive_time = keepalive_time
        self.optimization_level = optimization_level
        
        # 线程池管理
        self._workers = []
        self._work_queue = queue.PriorityQueue(maxsize=queue_size)
        self._shutdown = False
        self._lock = Lock()
        self._condition = Condition(self._lock)
        
        # 性能统计
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._total_execution_time = 0.0
        self._task_timings = deque(maxlen=1000)
        
        # 动态调整
        self._last_adjustment = time.time()
        self._adjustment_interval = 10.0
        
        # 启动工作线程
        self._start_workers(min_workers)
        
    def _start_workers(self, count: int):
        """启动工作线程"""
        for _ in range(count):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
            
    def _worker_loop(self):
        """工作线程主循环"""
        while not self._shutdown:
            try:
                # 从队列获取任务
                task = self._work_queue.get(timeout=1.0)
                if task is None:
                    break
                    
                # 执行任务
                start_time = time.time()
                try:
                    result = task.func(*task.args, **task.kwargs)
                    if task.callback:
                        task.callback(result)
                    self._completed_tasks += 1
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
                    if task.error_callback:
                        task.error_callback(e)
                    self._failed_tasks += 1
                finally:
                    execution_time = time.time() - start_time
                    self._total_execution_time += execution_time
                    self._task_timings.append(execution_time)
                    self._work_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"工作线程错误: {e}")
                
    def submit_task(self, 
                   func: Callable,
                   *args,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: Optional[float] = None,
                   callback: Optional[Callable] = None,
                   error_callback: Optional[Callable] = None,
                   **kwargs) -> Future:
        """提交任务"""
        if self._shutdown:
            raise RuntimeError("线程池已关闭")
            
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            callback=callback,
            error_callback=error_callback
        )
        
        future = Future()
        
        def task_wrapper():
            try:
                if timeout:
                    result = self._execute_with_timeout(task)
                else:
                    result = task.func(*task.args, **task.kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
                
        wrapped_task = Task(
            func=task_wrapper,
            priority=priority
        )
        
        try:
            self._work_queue.put(wrapped_task, block=False)
            self._maybe_adjust_pool_size()
            return future
        except queue.Full:
            raise RuntimeError("任务队列已满")
            
    def _execute_with_timeout(self, task: Task) -> Any:
        """带超时执行任务"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = task.func(*task.args, **task.kwargs)
            except Exception as e:
                exception[0] = e
                
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=task.timeout)
        
        if thread.is_alive():
            # 超时终止线程
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident), 
                ctypes.py_object(KeyboardInterrupt)
            )
            raise TimeoutError(f"任务执行超时 ({task.timeout}秒)")
            
        if exception[0]:
            raise exception[0]
            
        return result[0]
        
    def _maybe_adjust_pool_size(self):
        """动态调整线程池大小"""
        current_time = time.time()
        if current_time - self._last_adjustment < self._adjustment_interval:
            return
            
        self._last_adjustment = current_time
        
        # 根据队列大小和系统负载调整
        queue_size = self._work_queue.qsize()
        cpu_count = mp.cpu_count()
        
        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            # 激进优化：更多线程
            target_workers = min(
                max(self.min_workers, queue_size // 10 + 1),
                self.max_workers
            )
        elif self.optimization_level == OptimizationLevel.CONSERVATIVE:
            # 保守优化：较少线程
            target_workers = min(
                max(self.min_workers, cpu_count),
                self.max_workers
            )
        else:  # BALANCED or ADAPTIVE
            # 平衡优化
            target_workers = min(
                max(self.min_workers, cpu_count * 2, queue_size // 20 + 2),
                self.max_workers
            )
            
        current_workers = len([w for w in self._workers if w.is_alive()])
        
        if target_workers > current_workers:
            # 增加线程
            self._start_workers(target_workers - current_workers)
            logger.info(f"线程池扩展到 {target_workers} 个工作线程")
        elif target_workers < current_workers and queue_size < current_workers:
            # 减少线程
            self._reduce_workers(current_workers - target_workers)
            
    def _reduce_workers(self, count: int):
        """减少工作线程"""
        for _ in range(count):
            try:
                self._work_queue.put(None, block=False)
            except queue.Full:
                break
                
    def get_statistics(self) -> PerformanceMetrics:
        """获取性能统计"""
        with self._lock:
            avg_execution_time = (
                self._total_execution_time / max(self._completed_tasks, 1)
            )
            throughput = (
                self._completed_tasks / 
                max(time.time() - getattr(self, '_start_time', time.time()), 1)
            )
            
            return PerformanceMetrics(
                thread_count=len([w for w in self._workers if w.is_alive()]),
                task_queue_size=self._work_queue.qsize(),
                completed_tasks=self._completed_tasks,
                failed_tasks=self._failed_tasks,
                avg_execution_time=avg_execution_time,
                throughput=throughput
            )
            
    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        self._shutdown = True
        
        if wait:
            # 等待所有任务完成
            self._work_queue.join()
            
        # 停止工作线程
        for _ in self._workers:
            self._work_queue.put(None)
            
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=1.0)


class AsyncOptimizer:
    """异步优化器"""
    
    def __init__(self, 
                 max_concurrent: int = 1000,
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.max_concurrent = max_concurrent
        self.optimization_level = optimization_level
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks = set()
        self._completed_tasks = 0
        self._failed_tasks = 0
        
    async def execute_async(self, 
                           coro: Awaitable,
                           timeout: Optional[float] = None,
                           retry_count: int = 0,
                           retry_delay: float = 1.0) -> Any:
        """执行异步任务"""
        task = asyncio.create_task(coro)
        self._active_tasks.add(task)
        
        try:
            if timeout:
                result = await asyncio.wait_for(task, timeout=timeout)
            else:
                result = await task
                
            self._completed_tasks += 1
            return result
            
        except Exception as e:
            self._failed_tasks += 1
            
            if retry_count > 0:
                logger.warning(f"任务失败，{retry_delay}秒后重试: {e}")
                await asyncio.sleep(retry_delay)
                return await self.execute_async(
                    coro, timeout, retry_count - 1, retry_delay * 2
                )
            else:
                raise
                
        finally:
            self._active_tasks.discard(task)
            
    async def batch_execute(self, 
                           coros: List[Awaitable],
                           batch_size: Optional[int] = None,
                           return_exceptions: bool = True) -> List[Any]:
        """批量执行异步任务"""
        if batch_size is None:
            batch_size = self.max_concurrent
            
        results = []
        for i in range(0, len(coros), batch_size):
            batch = coros[i:i + batch_size]
            
            # 使用信号量限制并发数
            async with self._semaphore:
                batch_tasks = [
                    self.execute_async(coro) for coro in batch
                ]
                batch_results = await asyncio.gather(
                    *batch_tasks, 
                    return_exceptions=return_exceptions
                )
                results.extend(batch_results)
                
        return results
        
    @asynccontextmanager
    async def async_lock(self, lock: asyncio.Lock):
        """异步锁上下文管理器"""
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()
            
    def create_async_pool(self, size: Optional[int] = None) -> 'AsyncTaskPool':
        """创建异步任务池"""
        return AsyncTaskPool(size or self.max_concurrent)


class AsyncTaskPool:
    """异步任务池"""
    
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self._semaphore = asyncio.Semaphore(pool_size)
        self._tasks = set()
        
    async def submit(self, coro: Awaitable) -> asyncio.Task:
        """提交任务"""
        async def wrapped_coro():
            async with self._semaphore:
                return await coro
                
        task = asyncio.create_task(wrapped_coro())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
        
    async def shutdown(self):
        """关闭任务池"""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            'active_tasks': len(self._tasks),
            'pool_size': self.pool_size
        }


class IPCOptimizer:
    """进程间通信优化器"""
    
    def __init__(self):
        self._shared_memories = {}
        self._pipes = {}
        self._queues = {}
        
    def create_shared_memory(self, 
                           name: str, 
                           size: int,
                           create: bool = True) -> shared_memory.SharedMemory:
        """创建共享内存"""
        try:
            if create:
                shm = shared_memory.SharedMemory(create=True, name=name, size=size)
            else:
                shm = shared_memory.SharedMemory(name=name)
                
            self._shared_memories[name] = shm
            logger.info(f"创建共享内存: {name}, 大小: {size}")
            return shm
            
        except FileExistsError:
            # 共享内存已存在，尝试连接
            shm = shared_memory.SharedMemory(name=name)
            self._shared_memories[name] = shm
            return shm
            
    def create_optimized_queue(self, 
                             name: str,
                             maxsize: int = 0) -> 'OptimizedQueue':
        """创建优化的队列"""
        optimized_queue = OptimizedQueue(maxsize)
        self._queues[name] = optimized_queue
        return optimized_queue
        
    def create_pipe(self, name: str) -> Tuple[Any, Any]:
        """创建管道"""
        import multiprocessing as mp
        read_end, write_end = mp.Pipe(duplex=True)
        self._pipes[name] = (read_end, write_end)
        return read_end, write_end
        
    def cleanup(self):
        """清理资源"""
        # 清理共享内存
        for name, shm in self._shared_memories.items():
            try:
                shm.close()
                shm.unlink()
            except:
                pass
                
        # 清理其他资源
        self._shared_memories.clear()
        self._pipes.clear()
        self._queues.clear()


class OptimizedQueue:
    """优化的队列"""
    
    def __init__(self, maxsize: int = 0):
        self._queue = queue.Queue(maxsize)
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._maxsize = maxsize
        
    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        """放入项目"""
        with self._condition:
            while self._queue.full():
                if not block:
                    raise queue.Full
                if timeout is not None:
                    if not self._condition.wait(timeout):
                        raise queue.Full
                else:
                    self._condition.wait()
                    
            self._queue.put(item)
            self._condition.notify()
            
    def get(self, block: bool = True, timeout: Optional[float] = None):
        """获取项目"""
        with self._condition:
            while self._queue.empty():
                if not block:
                    raise queue.Empty
                if timeout is not None:
                    if not self._condition.wait(timeout):
                        raise queue.Empty
                else:
                    self._condition.wait()
                    
            item = self._queue.get()
            self._condition.notify()
            return item
            
    def qsize(self) -> int:
        """队列大小"""
        return self._queue.qsize()
        
    def empty(self) -> bool:
        """队列是否为空"""
        return self._queue.empty()
        
    def full(self) -> bool:
        """队列是否已满"""
        return self._queue.full()


class ConcurrencyControl:
    """并发控制器"""
    
    def __init__(self):
        self._semaphores = {}
        self._barriers = {}
        self._conditions = {}
        
    def create_semaphore(self, 
                        name: str, 
                        value: int = 1,
                        maxvalue: Optional[int] = None) -> Semaphore:
        """创建信号量"""
        semaphore = Semaphore(value, maxvalue)
        self._semaphores[name] = semaphore
        return semaphore
        
    def create_barrier(self, 
                      name: str, 
                      parties: int,
                      action: Optional[Callable] = None) -> Barrier:
        """创建屏障"""
        barrier = Barrier(parties, action)
        self._barriers[name] = barrier
        return barrier
        
    def create_condition(self, name: str) -> Condition:
        """创建条件变量"""
        condition = Condition()
        self._conditions[name] = condition
        return condition
        
    def acquire_semaphore(self, name: str, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取信号量"""
        semaphore = self._semaphores.get(name)
        if not semaphore:
            raise ValueError(f"信号量 {name} 不存在")
            
        try:
            if timeout:
                return semaphore.acquire(blocking, timeout)
            else:
                semaphore.acquire(blocking)
                return True
        except:
            return False
            
    def release_semaphore(self, name: str):
        """释放信号量"""
        semaphore = self._semaphores.get(name)
        if not semaphore:
            raise ValueError(f"信号量 {name} 不存在")
        semaphore.release()
        
    def wait_barrier(self, name: str, timeout: Optional[float] = None) -> bool:
        """等待屏障"""
        barrier = self._barriers.get(name)
        if not barrier:
            raise ValueError(f"屏障 {name} 不存在")
            
        try:
            if timeout:
                return barrier.wait(timeout)
            else:
                barrier.wait()
                return True
        except:
            return False


class DeadlockHandler:
    """死锁处理器"""
    
    def __init__(self):
        self._resource_graph = defaultdict(set)
        self._waiting_graph = defaultdict(set)
        self._lock_owners = {}
        self._prevention_strategies = [
            self._resource_ordering_prevention,
            self._timeout_prevention,
            self._bankers_algorithm
        ]
        
    def register_resource_acquisition(self, 
                                    thread_id: int, 
                                    resource_id: str):
        """注册资源获取"""
        self._lock_owners[resource_id] = thread_id
        
    def register_resource_wait(self, 
                             thread_id: int, 
                             waiting_for: str):
        """注册资源等待"""
        # 查找线程当前拥有的资源
        owned_resources = [
            resource for resource, owner in self._lock_owners.items() 
            if owner == thread_id
        ]
        
        for resource in owned_resources:
            self._waiting_graph[resource].add(waiting_for)
            
    def detect_deadlock(self) -> List[List[str]]:
        """检测死锁"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # 找到环
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._waiting_graph.get(node, set()):
                dfs(neighbor, path.copy())
                
            rec_stack.remove(node)
            
        for node in self._resource_graph:
            if node not in visited:
                dfs(node, [])
                
        return cycles
        
    def prevent_deadlock(self, 
                        thread_id: int, 
                        requested_resources: List[str]) -> bool:
        """预防死锁"""
        for strategy in self._prevention_strategies:
            if strategy(thread_id, requested_resources):
                return True
        return False
        
    def _resource_ordering_prevention(self, 
                                    thread_id: int, 
                                    resources: List[str]) -> bool:
        """资源排序预防"""
        # 检查资源是否按照预定顺序获取
        # 这里实现具体的排序逻辑
        return True
        
    def _timeout_prevention(self, 
                          thread_id: int, 
                          resources: List[str]) -> bool:
        """超时预防"""
        # 设置获取资源的超时时间
        # 如果超时，释放已持有的资源
        return True
        
    def _bankers_algorithm(self, 
                         thread_id: int, 
                         resources: List[str]) -> bool:
        """银行家算法"""
        # 实现银行家算法进行死锁预防
        return True
        
    def recover_from_deadlock(self, deadlocks: List[List[str]]):
        """从死锁中恢复"""
        for cycle in deadlocks:
            logger.warning(f"检测到死锁环: {cycle}")
            # 选择一个线程进行终止
            victim_thread = self._select_victim(cycle)
            self._terminate_thread(victim_thread)
            
    def _select_victim(self, cycle: List[str]) -> int:
        """选择受害者线程"""
        # 简单的选择策略：选择优先级最低的线程
        # 这里可以实现更复杂的策略
        return int(cycle[0]) if cycle else 0
        
    def _terminate_thread(self, thread_id: int):
        """终止线程"""
        # 尝试终止指定的线程
        # 注意：这需要特殊的权限和实现
        logger.warning(f"尝试终止线程 {thread_id}")
        # 这里实现具体的线程终止逻辑


class AsyncConcurrencyOptimizer:
    """异步并发优化器"""
    
    def __init__(self, 
                 max_concurrent_tasks: int = 1000,
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.optimization_level = optimization_level
        self._task_pool = AsyncTaskPool(max_concurrent_tasks)
        self._async_locks = {}
        self._async_semaphores = {}
        self._performance_monitor = PerformanceMonitor()
        
    async def optimize_async_execution(self, 
                                     coro: Awaitable,
                                     timeout: Optional[float] = None,
                                     priority: TaskPriority = TaskPriority.NORMAL) -> Any:
        """优化异步执行"""
        # 根据优先级调整执行策略
        if priority == TaskPriority.CRITICAL:
            # 关键任务：立即执行
            return await self._task_pool.submit(coro)
        elif priority == TaskPriority.HIGH:
            # 高优先级：使用更短的超时
            effective_timeout = timeout * 0.8 if timeout else None
            return await self._task_pool.submit(coro)
        else:
            # 普通任务：标准执行
            return await self._task_pool.submit(coro)
            
    async def create_async_lock(self, name: str) -> asyncio.Lock:
        """创建异步锁"""
        if name not in self._async_locks:
            self._async_locks[name] = asyncio.Lock()
        return self._async_locks[name]
        
    async def create_async_semaphore(self, 
                                   name: str, 
                                   value: int = 1) -> asyncio.Semaphore:
        """创建异步信号量"""
        if name not in self._async_semaphores:
            self._async_semaphores[name] = asyncio.Semaphore(value)
        return self._async_semaphores[name]
        
    async def batch_optimize_async(self, 
                                 coros: List[Awaitable],
                                 batch_size: Optional[int] = None) -> List[Any]:
        """批量优化异步任务"""
        if batch_size is None:
            batch_size = min(self.max_concurrent_tasks, len(coros))
            
        results = []
        for i in range(0, len(coros), batch_size):
            batch = coros[i:i + batch_size]
            
            # 并发执行批次
            batch_tasks = [
                self.optimize_async_execution(coro) 
                for coro in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            
        return results
        
    async def monitor_performance(self):
        """监控性能"""
        while True:
            stats = self._task_pool.get_stats()
            logger.info(f"异步任务池统计: {stats}")
            await asyncio.sleep(5.0)
            
    async def shutdown(self):
        """关闭优化器"""
        await self._task_pool.shutdown()


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._start_time = time.time()
        
    def record_metric(self, name: str, value: float):
        """记录指标"""
        self._metrics[name].append({
            'value': value,
            'timestamp': time.time()
        })
        
    def get_average(self, name: str) -> float:
        """获取平均值"""
        values = [m['value'] for m in self._metrics[name]]
        return sum(values) / len(values) if values else 0.0
        
    def get_percentile(self, name: str, percentile: float) -> float:
        """获取百分位数"""
        values = sorted([m['value'] for m in self._metrics[name]])
        if not values:
            return 0.0
            
        index = int(len(values) * percentile)
        return values[min(index, len(values) - 1)]
        
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        for name, metrics in self._metrics.items():
            values = [m['value'] for m in metrics]
            if values:
                summary[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'latest': values[-1]
                }
        return summary


class ConcurrencyOptimizer:
    """O6并发优化器主类"""
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
                 enable_deadlock_detection: bool = True,
                 enable_performance_monitoring: bool = True):
        """
        初始化O6并发优化器
        
        Args:
            optimization_level: 优化级别
            enable_deadlock_detection: 是否启用死锁检测
            enable_performance_monitoring: 是否启用性能监控
        """
        self.optimization_level = optimization_level
        self.enable_deadlock_detection = enable_deadlock_detection
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # 初始化各个优化器
        self._lock_optimizer = LockOptimizer(optimization_level)
        self._async_optimizer = AsyncOptimizer(optimization_level=optimization_level)
        self._ipc_optimizer = IPCOptimizer()
        self._concurrency_control = ConcurrencyControl()
        self._deadlock_handler = DeadlockHandler()
        self._async_concurrency_optimizer = AsyncConcurrencyOptimizer(
            optimization_level=optimization_level
        )
        
        # 线程池
        self._thread_pool = None
        self._process_pool = None
        
        # 性能监控
        if enable_performance_monitoring:
            self._performance_monitor = PerformanceMonitor()
        else:
            self._performance_monitor = None
            
        # 启动死锁检测
        if enable_deadlock_detection:
            self._deadlock_handler.start_detection()
            
        logger.info("O6并发优化器初始化完成")
        
    def create_optimized_thread_pool(self,
                                   max_workers: Optional[int] = None,
                                   min_workers: int = 2,
                                   queue_size: int = 1000) -> OptimizedThreadPool:
        """创建优化的线程池"""
        if self._thread_pool:
            self._thread_pool.shutdown()
            
        self._thread_pool = OptimizedThreadPool(
            max_workers=max_workers,
            min_workers=min_workers,
            queue_size=queue_size,
            optimization_level=self.optimization_level
        )
        
        logger.info(f"创建优化线程池: max_workers={max_workers}, min_workers={min_workers}")
        return self._thread_pool
        
    def create_optimized_process_pool(self, max_workers: Optional[int] = None):
        """创建优化的进程池"""
        if self._process_pool:
            self._process_pool.shutdown()
            
        max_workers = max_workers or min(mp.cpu_count(), 8)
        self._process_pool = ProcessPoolExecutor(max_workers=max_workers)
        
        logger.info(f"创建优化进程池: max_workers={max_workers}")
        return self._process_pool
        
    def create_optimized_lock(self, lock_type: LockType) -> OptimizedLock:
        """创建优化的锁"""
        return self._lock_optimizer.create_optimized_lock(lock_type)
        
    def create_async_optimizer(self, max_concurrent: int = 1000) -> AsyncOptimizer:
        """创建异步优化器"""
        return AsyncOptimizer(max_concurrent, self.optimization_level)
        
    def create_ipc_optimizer(self) -> IPCOptimizer:
        """创建IPC优化器"""
        return self._ipc_optimizer
        
    def create_concurrency_controller(self) -> ConcurrencyControl:
        """创建并发控制器"""
        return self._concurrency_control
        
    async def execute_async_task(self,
                                coro: Awaitable,
                                timeout: Optional[float] = None,
                                priority: TaskPriority = TaskPriority.NORMAL) -> Any:
        """执行异步任务"""
        return await self._async_concurrency_optimizer.optimize_async_execution(
            coro, timeout, priority
        )
        
    async def batch_execute_async_tasks(self,
                                      coros: List[Awaitable],
                                      batch_size: Optional[int] = None) -> List[Any]:
        """批量执行异步任务"""
        return await self._async_concurrency_optimizer.batch_optimize_async(
            coros, batch_size
        )
        
    def execute_with_thread_pool(self,
                               func: Callable,
                               *args,
                               priority: TaskPriority = TaskPriority.NORMAL,
                               timeout: Optional[float] = None,
                               **kwargs) -> Future:
        """使用线程池执行任务"""
        if not self._thread_pool:
            self.create_optimized_thread_pool()
            
        return self._thread_pool.submit_task(
            func, *args, priority=priority, timeout=timeout, **kwargs
        )
        
    def execute_with_process_pool(self,
                                func: Callable,
                                *args,
                                **kwargs) -> Future:
        """使用进程池执行任务"""
        if not self._process_pool:
            self.create_optimized_process_pool()
            
        return self._process_pool.submit(func, *args, **kwargs)
        
    def optimize_data_structure(self, data_structure: Any) -> Any:
        """优化数据结构"""
        return self._lock_optimizer.optimize_data_structure(data_structure)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {}
        
        if self._thread_pool:
            metrics['thread_pool'] = self._thread_pool.get_statistics()
            
        if self._process_pool:
            metrics['process_pool'] = 'active'
            
        if self._performance_monitor:
            metrics['performance'] = self._performance_monitor.get_summary()
            
        metrics['optimization_level'] = self.optimization_level.value
        metrics['deadlock_detection_enabled'] = self.enable_deadlock_detection
        metrics['performance_monitoring_enabled'] = self.enable_performance_monitoring
        
        return metrics
        
    def detect_deadlocks(self) -> List[List[str]]:
        """检测死锁"""
        return self._deadlock_handler.detect_deadlock()
        
    def prevent_deadlock(self, 
                       thread_id: int, 
                       resources: List[str]) -> bool:
        """预防死锁"""
        return self._deadlock_handler.prevent_deadlock(thread_id, resources)
        
    def get_lock_statistics(self) -> Dict[str, int]:
        """获取锁统计信息"""
        return self._lock_optimizer.get_lock_statistics()
        
    def cleanup(self):
        """清理资源"""
        logger.info("开始清理O6并发优化器资源...")
        
        # 关闭线程池
        if self._thread_pool:
            self._thread_pool.shutdown()
            
        # 关闭进程池
        if self._process_pool:
            self._process_pool.shutdown()
            
        # 清理IPC资源
        self._ipc_optimizer.cleanup()
        
        # 停止死锁检测
        if self.enable_deadlock_detection:
            self._deadlock_handler.stop_detection()
            
        logger.info("O6并发优化器资源清理完成")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# 装饰器工具

def optimized_synchronized(lock_type: LockType = LockType.MUTEX):
    """同步装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, '_optimization_lock', None)
            if lock is None:
                lock = LockFactory.create_lock(lock_type)
                setattr(self, '_optimization_lock', lock)
                
            with lock:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


def async_optimized(max_concurrent: int = 1000):
    """异步优化装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 这里可以实现异步优化逻辑
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def thread_pool_optimized(max_workers: Optional[int] = None):
    """线程池优化装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 这里可以实现线程池优化逻辑
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 使用示例

def example_usage():
    """使用示例"""
    
    # 1. 基本使用
    with ConcurrencyOptimizer(OptimizationLevel.BALANCED) as optimizer:
        
        # 创建优化的线程池
        thread_pool = optimizer.create_optimized_thread_pool(
            max_workers=8,
            min_workers=2
        )
        
        # 提交任务
        future = optimizer.execute_with_thread_pool(
            lambda: time.sleep(1) or "任务完成",
            priority=TaskPriority.HIGH
        )
        
        result = future.result(timeout=5.0)
        print(f"任务结果: {result}")
        
        # 2. 锁优化
        lock = optimizer.create_optimized_lock(LockType.READ_WRITE)
        
        with lock.acquire_write():
            # 执行需要写锁的操作
            pass
            
        # 3. 异步优化
        async def async_task():
            await asyncio.sleep(1)
            return "异步任务完成"
            
        async def main():
            result = await optimizer.execute_async_task(async_task())
            print(f"异步结果: {result}")
            
        asyncio.run(main())
        
        # 4. 批量异步任务
        async def batch_main():
            tasks = [async_task() for _ in range(10)]
            results = await optimizer.batch_execute_async_tasks(tasks)
            print(f"批量结果: {results}")
            
        asyncio.run(batch_main())
        
        # 5. 性能监控
        metrics = optimizer.get_performance_metrics()
        print(f"性能指标: {metrics}")
        
        # 6. 死锁检测
        deadlocks = optimizer.detect_deadlocks()
        if deadlocks:
            print(f"检测到死锁: {deadlocks}")


# 高级功能模块

class AdaptiveLoadBalancer:
    """自适应负载均衡器"""
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self._worker_loads = defaultdict(float)
        self._task_history = deque(maxlen=1000)
        self._lock = Lock()
        
    def select_worker(self, task: Task) -> int:
        """选择最优工作线程"""
        with self._lock:
            if not self._worker_loads:
                return 0  # 默认返回第一个工作线程
                
            # 根据任务类型和历史负载选择
            if task.priority == TaskPriority.CRITICAL:
                # 关键任务：选择负载最低的
                return min(self._worker_loads, key=self._worker_loads.get)
            else:
                # 普通任务：轮询选择
                current_worker = len(self._task_history) % len(self._worker_loads)
                workers = list(self._worker_loads.keys())
                return workers[current_worker] if workers else 0
                
    def update_worker_load(self, worker_id: int, load: float):
        """更新工作线程负载"""
        with self._lock:
            self._worker_loads[worker_id] = load
            
    def record_task_completion(self, task: Task, execution_time: float):
        """记录任务完成"""
        with self._lock:
            self._task_history.append({
                'task': task,
                'execution_time': execution_time,
                'timestamp': time.time()
            })


class MemoryOptimizedPool:
    """内存优化池"""
    
    def __init__(self, 
                 object_type: type,
                 initial_size: int = 10,
                 max_size: int = 100,
                 cleanup_interval: float = 60.0):
        self.object_type = object_type
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self._pool = []
        self._in_use = set()
        self._lock = Lock()
        self._last_cleanup = time.time()
        
        # 预填充池
        for _ in range(initial_size):
            self._pool.append(self._create_object())
            
    def _create_object(self) -> Any:
        """创建新对象"""
        try:
            return self.object_type()
        except Exception as e:
            logger.error(f"创建对象失败: {e}")
            return None
            
    def acquire(self) -> Any:
        """获取对象"""
        with self._lock:
            now = time.time()
            
            # 定期清理
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup()
                self._last_cleanup = now
                
            # 从池中获取
            if self._pool:
                obj = self._pool.pop()
                if obj is not None:
                    self._in_use.add(id(obj))
                    return obj
                    
            # 池为空，创建新对象
            obj = self._create_object()
            if obj is not None:
                self._in_use.add(id(obj))
                return obj
                
            raise RuntimeError("无法创建新对象")
            
    def release(self, obj: Any):
        """释放对象"""
        if obj is None:
            return
            
        obj_id = id(obj)
        with self._lock:
            if obj_id in self._in_use:
                self._in_use.remove(obj_id)
                
                # 如果池未满，回收对象
                if len(self._pool) < self.max_size:
                    self._reset_object(obj)
                    self._pool.append(obj)
                # 否则丢弃对象，让垃圾回收处理
                
    def _reset_object(self, obj: Any):
        """重置对象状态"""
        if hasattr(obj, 'reset'):
            try:
                obj.reset()
            except Exception as e:
                logger.warning(f"重置对象失败: {e}")
                
    def _cleanup(self):
        """清理过期对象"""
        # 清理池中的过期对象
        current_time = time.time()
        self._pool = [obj for obj in self._pool if obj is not None]
        
    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息"""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'in_use': len(self._in_use),
                'max_size': self.max_size,
                'utilization': len(self._in_use) / max(len(self._pool) + len(self._in_use), 1)
            }


class AdvancedScheduler:
    """高级调度器"""
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self._scheduled_tasks = []
        self._running_tasks = set()
        self._completed_tasks = []
        self._failed_tasks = []
        self._lock = Lock()
        self._condition = Condition(self._lock)
        
    def schedule_task(self, 
                     func: Callable,
                     delay: float,
                     *args,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     **kwargs) -> str:
        """调度任务"""
        task_id = f"task_{int(time.time() * 1000000)}"
        
        task = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'scheduled_time': time.time() + delay,
            'created_time': time.time()
        }
        
        with self._lock:
            self._scheduled_tasks.append(task)
            # 按优先级和调度时间排序
            self._scheduled_tasks.sort(
                key=lambda t: (t['scheduled_time'], -t['priority'].value)
            )
            
        logger.info(f"任务 {task_id} 已调度，{delay}秒后执行")
        return task_id
        
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            for i, task in enumerate(self._scheduled_tasks):
                if task['id'] == task_id:
                    del self._scheduled_tasks[i]
                    logger.info(f"任务 {task_id} 已取消")
                    return True
            return False
        
    def get_next_task(self) -> Optional[Dict]:
        """获取下一个要执行的任务"""
        with self._lock:
            now = time.time()
            
            # 查找已到时间的任务
            for i, task in enumerate(self._scheduled_tasks):
                if task['scheduled_time'] <= now:
                    del self._scheduled_tasks[i]
                    self._running_tasks.add(task['id'])
                    return task
                    
            return None
        
    def complete_task(self, task_id: str, result: Any = None, error: Exception = None):
        """完成任务"""
        with self._lock:
            if task_id in self._running_tasks:
                self._running_tasks.remove(task_id)
                
                task_info = {
                    'id': task_id,
                    'result': result,
                    'error': error,
                    'completed_time': time.time()
                }
                
                if error:
                    self._failed_tasks.append(task_info)
                else:
                    self._completed_tasks.append(task_info)
                    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        with self._lock:
            return {
                'scheduled_tasks': len(self._scheduled_tasks),
                'running_tasks': len(self._running_tasks),
                'completed_tasks': len(self._completed_tasks),
                'failed_tasks': len(self._failed_tasks),
                'next_scheduled_time': (
                    self._scheduled_tasks[0]['scheduled_time'] 
                    if self._scheduled_tasks else None
                )
            }


class IntelligentCache:
    """智能缓存"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl: float = 300.0,
                 cleanup_interval: float = 60.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval
        self._cache = {}
        self._access_times = {}
        self._lock = Lock()
        self._last_cleanup = time.time()
        
    def get(self, key: Any) -> Tuple[bool, Any]:
        """获取缓存值"""
        with self._lock:
            now = time.time()
            
            # 检查是否过期
            if key in self._cache:
                value, timestamp = self._cache[key]
                if now - timestamp <= self.ttl:
                    self._access_times[key] = now
                    return True, value
                else:
                    # 过期，删除
                    del self._cache[key]
                    if key in self._access_times:
                        del self._access_times[key]
                        
            return False, None
            
    def set(self, key: Any, value: Any):
        """设置缓存值"""
        with self._lock:
            now = time.time()
            
            # 定期清理
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup()
                self._last_cleanup = now
                
            # 如果缓存已满，删除最少使用的项
            if len(self._cache) >= self.max_size:
                self._evict_lru()
                
            self._cache[key] = (value, now)
            self._access_times[key] = now
            
    def _evict_lru(self):
        """淘汰最少最近使用的项"""
        if not self._access_times:
            return
            
        # 找到最少访问的键
        lru_key = min(self._access_times, key=self._access_times.get)
        
        # 删除
        del self._cache[lru_key]
        del self._access_times[lru_key]
        
        logger.debug(f"淘汰LRU缓存项: {lru_key}")
        
    def _cleanup(self):
        """清理过期项"""
        now = time.time()
        expired_keys = []
        
        for key, (value, timestamp) in self._cache.items():
            if now - timestamp > self.ttl:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
                
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
            
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            now = time.time()
            valid_items = sum(
                1 for _, (_, timestamp) in self._cache.items()
                if now - timestamp <= self.ttl
            )
            
            return {
                'total_items': len(self._cache),
                'valid_items': valid_items,
                'expired_items': len(self._cache) - valid_items,
                'max_size': self.max_size,
                'utilization': len(self._cache) / self.max_size
            }


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self._is_monitoring = False
        self._monitor_thread = None
        self._metrics_history = deque(maxlen=1000)
        self._alerts = []
        
    def start_monitoring(self):
        """开始监控"""
        if not self._is_monitoring:
            self._is_monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("资源监控已启动")
            
    def stop_monitoring(self):
        """停止监控"""
        self._is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("资源监控已停止")
        
    def _monitoring_loop(self):
        """监控循环"""
        while self._is_monitoring:
            try:
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)
                self._check_alerts(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                
    def _collect_metrics(self) -> Dict[str, float]:
        """收集指标"""
        import psutil
        
        try:
            process = psutil.Process()
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'threads': process.num_threads(),
                'file_descriptors': process.num_fds() if hasattr(process, 'num_fds') else 0,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"收集指标失败: {e}")
            return {'timestamp': time.time()}
            
    def _check_alerts(self, metrics: Dict[str, float]):
        """检查告警"""
        # CPU使用率告警
        if metrics.get('cpu_percent', 0) > 90:
            self._add_alert('HIGH_CPU', f"CPU使用率过高: {metrics['cpu_percent']:.1f}%")
            
        # 内存使用率告警
        if metrics.get('memory_percent', 0) > 90:
            self._add_alert('HIGH_MEMORY', f"内存使用率过高: {metrics['memory_percent']:.1f}%")
            
        # 线程数告警
        if metrics.get('threads', 0) > 100:
            self._add_alert('HIGH_THREADS', f"线程数过多: {metrics['threads']}")
            
    def _add_alert(self, alert_type: str, message: str):
        """添加告警"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': time.time()
        }
        self._alerts.append(alert)
        logger.warning(f"资源告警: {message}")
        
    def get_current_metrics(self) -> Dict[str, float]:
        """获取当前指标"""
        if self._metrics_history:
            return dict(self._metrics_history[-1])
        return {}
        
    def get_metrics_history(self, count: int = 100) -> List[Dict[str, float]]:
        """获取指标历史"""
        return list(self._metrics_history)[-count:]
        
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近告警"""
        return self._alerts[-count:]
        
    def clear_alerts(self):
        """清空告警"""
        self._alerts.clear()


class AdvancedProfiler:
    """高级性能分析器"""
    
    def __init__(self):
        self._profiles = defaultdict(list)
        self._active_profiles = {}
        self._lock = Lock()
        
    def start_profile(self, name: str):
        """开始性能分析"""
        profile_id = f"{name}_{int(time.time() * 1000000)}"
        
        frame = inspect.currentframe()
        stack_info = []
        
        # 收集调用栈信息
        for i in range(min(10, len(inspect.stack()))):
            try:
                frame_info = inspect.stack()[i]
                stack_info.append({
                    'filename': frame_info.filename,
                    'lineno': frame_info.lineno,
                    'function': frame_info.function
                })
            except:
                break
                
        profile_data = {
            'id': profile_id,
            'name': name,
            'start_time': time.time(),
            'stack_info': stack_info,
            'thread_id': threading.get_ident(),
            'memory_start': self._get_memory_usage()
        }
        
        with self._lock:
            self._active_profiles[profile_id] = profile_data
            
        logger.debug(f"开始性能分析: {name} ({profile_id})")
        return profile_id
        
    def end_profile(self, profile_id: str) -> Dict[str, Any]:
        """结束性能分析"""
        with self._lock:
            if profile_id not in self._active_profiles:
                logger.warning(f"性能分析ID不存在: {profile_id}")
                return {}
                
            profile_data = self._active_profiles.pop(profile_id)
            
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        profile_result = {
            'id': profile_id,
            'name': profile_data['name'],
            'duration': end_time - profile_data['start_time'],
            'memory_delta': end_memory - profile_data['memory_start'],
            'thread_id': profile_data['thread_id'],
            'stack_info': profile_data['stack_info']
        }
        
        with self._lock:
            self._profiles[profile_data['name']].append(profile_result)
            
        logger.debug(f"结束性能分析: {profile_data['name']} ({profile_id})")
        return profile_result
        
    def _get_memory_usage(self) -> float:
        """获取内存使用量"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
            
    def get_profile_stats(self, name: str) -> Dict[str, Any]:
        """获取性能分析统计"""
        with self._lock:
            profiles = self._profiles.get(name, [])
            
            if not profiles:
                return {}
                
            durations = [p['duration'] for p in profiles]
            memory_deltas = [p['memory_delta'] for p in profiles]
            
            return {
                'count': len(profiles),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
                'latest': profiles[-1] if profiles else None
            }
            
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有性能分析统计"""
        stats = {}
        with self._lock:
            for name in self._profiles:
                stats[name] = self.get_profile_stats(name)
        return stats


# 装饰器增强

def profile_performance(name: str = None):
    """性能分析装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profile_name = name or f"{func.__module__}.{func.__name__}"
            profiler = AdvancedProfiler()
            
            profile_id = profiler.start_profile(profile_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_profile(profile_id)
                
        return wrapper
    return decorator


def cached_optimization(cache_ttl: float = 300.0, max_size: int = 1000):
    """缓存优化装饰器"""
    cache = IntelligentCache(max_size=max_size, ttl=cache_ttl)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            hit, result = cache.get(cache_key)
            if hit:
                logger.debug(f"缓存命中: {cache_key}")
                return result
                
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            cache.set(cache_key, result)
            logger.debug(f"缓存存储: {cache_key}")
            
            return result
        return wrapper
    return decorator


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, exponential_base: float = 2.0):
    """重试装饰器（指数退避）"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = base_delay * (exponential_base ** attempt)
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, "
                            f"{delay}秒后重试"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"函数 {func.__name__} 在 {max_retries + 1} 次尝试后失败")
                        
            raise last_exception
        return wrapper
    return decorator


# 扩展的并发优化器类

class ExtendedConcurrencyOptimizer(ConcurrencyOptimizer):
    """扩展的并发优化器"""
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
                 enable_deadlock_detection: bool = True,
                 enable_performance_monitoring: bool = True,
                 enable_resource_monitoring: bool = True):
        """初始化扩展并发优化器"""
        super().__init__(
            optimization_level,
            enable_deadlock_detection,
            enable_performance_monitoring
        )
        
        # 扩展功能
        self._load_balancer = AdaptiveLoadBalancer(optimization_level)
        self._memory_pools = {}
        self._scheduler = AdvancedScheduler(optimization_level)
        self._cache = IntelligentCache()
        self._profiler = AdvancedProfiler()
        
        # 资源监控
        if enable_resource_monitoring:
            self._resource_monitor = ResourceMonitor()
            self._resource_monitor.start_monitoring()
        else:
            self._resource_monitor = None
            
        logger.info("扩展O6并发优化器初始化完成")
        
    def create_memory_pool(self, 
                          object_type: type,
                          initial_size: int = 10,
                          max_size: int = 100) -> MemoryOptimizedPool:
        """创建内存优化池"""
        pool = MemoryOptimizedPool(object_type, initial_size, max_size)
        self._memory_pools[object_type.__name__] = pool
        return pool
        
    def get_load_balancer(self) -> AdaptiveLoadBalancer:
        """获取负载均衡器"""
        return self._load_balancer
        
    def get_scheduler(self) -> AdvancedScheduler:
        """获取调度器"""
        return self._scheduler
        
    def get_cache(self) -> IntelligentCache:
        """获取缓存"""
        return self._cache
        
    def get_profiler(self) -> AdvancedProfiler:
        """获取性能分析器"""
        return self._profiler
        
    def schedule_delayed_task(self,
                            func: Callable,
                            delay: float,
                            *args,
                            priority: TaskPriority = TaskPriority.NORMAL,
                            **kwargs) -> str:
        """调度延迟任务"""
        return self._scheduler.schedule_task(func, delay, *args, priority=priority, **kwargs)
        
    def get_resource_metrics(self) -> Dict[str, Any]:
        """获取资源指标"""
        metrics = self.get_performance_metrics()
        
        if self._resource_monitor:
            metrics['resource_monitor'] = {
                'current_metrics': self._resource_monitor.get_current_metrics(),
                'recent_alerts': self._resource_monitor.get_recent_alerts()
            }
            
        # 添加内存池统计
        pool_stats = {}
        for name, pool in self._memory_pools.items():
            pool_stats[name] = pool.get_stats()
        metrics['memory_pools'] = pool_stats
        
        # 添加调度器状态
        metrics['scheduler'] = self._scheduler.get_status()
        
        # 添加缓存统计
        metrics['cache'] = self._cache.get_stats()
        
        # 添加性能分析统计
        metrics['profiler'] = self._profiler.get_all_stats()
        
        return metrics
        
    def cleanup(self):
        """清理资源"""
        logger.info("开始清理扩展O6并发优化器资源...")
        
        # 调用父类清理
        super().cleanup()
        
        # 清理内存池
        for pool in self._memory_pools.values():
            # 内存池没有明确的清理方法
            pass
        self._memory_pools.clear()
        
        # 停止资源监控
        if self._resource_monitor:
            self._resource_monitor.stop_monitoring()
            
        logger.info("扩展O6并发优化器资源清理完成")


# 完整的使用示例

def comprehensive_example():
    """综合使用示例"""
    print("=== O6并发优化器综合示例 ===\n")
    
    # 1. 创建扩展优化器
    with ExtendedConcurrencyOptimizer(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_resource_monitoring=True
    ) as optimizer:
        
        print("1. 创建各种优化组件")
        
        # 创建线程池
        thread_pool = optimizer.create_optimized_thread_pool(
            max_workers=8,
            min_workers=2,
            queue_size=500
        )
        
        # 创建内存池
        list_pool = optimizer.create_memory_pool(list, initial_size=5, max_size=20)
        dict_pool = optimizer.create_memory_pool(dict, initial_size=5, max_size=20)
        
        print(f"   - 线程池创建完成: max_workers=8")
        print(f"   - 内存池创建完成: list_pool, dict_pool")
        
        # 2. 测试锁优化
        print("\n2. 测试锁优化")
        
        read_write_lock = optimizer.create_optimized_lock(LockType.READ_WRITE)
        mutex_lock = optimizer.create_optimized_lock(LockType.MUTEX)
        
        # 读锁测试
        with read_write_lock.acquire_read():
            print("   - 获取读锁成功")
            time.sleep(0.1)
            
        # 写锁测试
        with read_write_lock.acquire_write():
            print("   - 获取写锁成功")
            time.sleep(0.1)
            
        # 3. 测试异步优化
        print("\n3. 测试异步优化")
        
        async def async_calculation(n):
            """异步计算任务"""
            await asyncio.sleep(0.1)
            return sum(i * i for i in range(n))
            
        async def test_async():
            # 单个异步任务
            result1 = await optimizer.execute_async_task(
                async_calculation(1000),
                priority=TaskPriority.HIGH
            )
            print(f"   - 单个异步任务结果: {result1}")
            
            # 批量异步任务
            tasks = [async_calculation(500) for _ in range(5)]
            results = await optimizer.batch_execute_async_tasks(tasks)
            print(f"   - 批量异步任务结果: {len(results)} 个任务完成")
            
        asyncio.run(test_async())
        
        # 4. 测试线程池任务
        print("\n4. 测试线程池任务")
        
        def cpu_intensive_task(n):
            """CPU密集型任务"""
            result = 0
            for i in range(n):
                result += i ** 0.5
            return result
            
        # 提交多个任务
        futures = []
        for i in range(10):
            future = optimizer.execute_with_thread_pool(
                cpu_intensive_task,
                5000,
                priority=TaskPriority.NORMAL
            )
            futures.append(future)
            
        # 等待结果
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=5.0)
                results.append(result)
                print(f"   - 任务 {i+1} 完成: {result:.2f}")
            except Exception as e:
                print(f"   - 任务 {i+1} 失败: {e}")
                
        # 5. 测试调度器
        print("\n5. 测试任务调度")
        
        def scheduled_task(name):
            print(f"   - 调度任务 {name} 执行")
            return f"任务 {name} 完成"
            
        # 调度延迟任务
        task_id1 = optimizer.schedule_delayed_task(scheduled_task, 1.0, "A")
        task_id2 = optimizer.schedule_delayed_task(scheduled_task, 2.0, "B")
        
        print(f"   - 任务已调度: {task_id1}, {task_id2}")
        
        # 等待调度任务执行
        time.sleep(3)
        
        # 6. 测试缓存
        print("\n6. 测试智能缓存")
        
        @cached_optimization(cache_ttl=10.0, max_size=100)
        def expensive_function(x, y):
            """昂贵的函数"""
            time.sleep(0.1)  # 模拟计算
            return x * x + y * y
            
        # 第一次调用
        start_time = time.time()
        result1 = expensive_function(10, 20)
        time1 = time.time() - start_time
        
        # 第二次调用（应该命中缓存）
        start_time = time.time()
        result2 = expensive_function(10, 20)
        time2 = time.time() - start_time
        
        print(f"   - 第一次调用: {result1}, 耗时: {time1:.3f}秒")
        print(f"   - 第二次调用: {result2}, 耗时: {time2:.3f}秒")
        print(f"   - 缓存加速比: {time1/time2:.1f}x")
        
        # 7. 测试性能分析
        print("\n7. 测试性能分析")
        
        @profile_performance("test_function")
        def test_function():
            time.sleep(0.1)
            return "分析完成"
            
        result = test_function()
        print(f"   - {result}")
        
        stats = optimizer.get_profiler().get_profile_stats("test_function")
        if stats:
            print(f"   - 分析统计: 平均耗时 {stats['avg_duration']:.3f}秒")
        
        # 8. 获取综合指标
        print("\n8. 综合性能指标")
        
        metrics = optimizer.get_resource_metrics()
        
        # 线程池指标
        if 'thread_pool' in metrics:
            thread_stats = metrics['thread_pool']
            print(f"   - 线程池: {thread_stats.completed_tasks} 个任务完成")
            print(f"   - 平均执行时间: {thread_stats.avg_execution_time:.3f}秒")
            
        # 资源监控指标
        if 'resource_monitor' in metrics:
            resource_stats = metrics['resource_monitor']['current_metrics']
            print(f"   - CPU使用率: {resource_stats.get('cpu_percent', 0):.1f}%")
            print(f"   - 内存使用: {resource_stats.get('memory_mb', 0):.1f}MB")
            
        # 缓存指标
        if 'cache' in metrics:
            cache_stats = metrics['cache']
            print(f"   - 缓存: {cache_stats['valid_items']}/{cache_stats['total_items']} 项有效")
            
        # 内存池指标
        if 'memory_pools' in metrics:
            for pool_name, pool_stats in metrics['memory_pools'].items():
                print(f"   - {pool_name}池: 利用率 {pool_stats['utilization']:.1%}")
                
        print("\n=== 示例完成 ===")


def stress_test():
    """压力测试"""
    print("\n=== O6并发优化器压力测试 ===")
    
    with ExtendedConcurrencyOptimizer(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_resource_monitoring=True
    ) as optimizer:
        
        # 创建大量工作线程
        thread_pool = optimizer.create_optimized_thread_pool(
            max_workers=32,
            min_workers=4,
            queue_size=2000
        )
        
        print(f"创建线程池: max_workers=32, queue_size=2000")
        
        # 定义测试任务
        def stress_task(task_id):
            """压力测试任务"""
            # 模拟CPU密集型工作
            result = 0
            for i in range(1000):
                result += i ** 0.5
                
            # 模拟I/O等待
            time.sleep(0.01)
            
            return {
                'task_id': task_id,
                'result': result,
                'thread_id': threading.get_ident()
            }
            
        # 提交大量任务
        num_tasks = 500
        print(f"提交 {num_tasks} 个任务...")
        
        start_time = time.time()
        futures = []
        
        for i in range(num_tasks):
            future = optimizer.execute_with_thread_pool(
                stress_task,
                i,
                priority=TaskPriority.NORMAL,
                timeout=30.0
            )
            futures.append(future)
            
        print("所有任务已提交，等待完成...")
        
        # 收集结果
        completed = 0
        failed = 0
        results = []
        
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=60.0)
                results.append(result)
                completed += 1
                
                if (i + 1) % 100 == 0:
                    print(f"已完成 {i + 1}/{num_tasks} 个任务")
                    
            except Exception as e:
                failed += 1
                print(f"任务 {i} 失败: {e}")
                
        end_time = time.time()
        total_time = end_time - start_time
        
        # 性能统计
        print(f"\n=== 压力测试结果 ===")
        print(f"总任务数: {num_tasks}")
        print(f"成功完成: {completed}")
        print(f"失败任务: {failed}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均每个任务: {total_time/num_tasks:.3f}秒")
        print(f"任务吞吐量: {completed/total_time:.1f} 任务/秒")
        
        # 获取详细统计
        metrics = optimizer.get_resource_metrics()
        
        if 'thread_pool' in metrics:
            thread_stats = metrics['thread_pool']
            print(f"\n线程池统计:")
            print(f"  - 最大线程数: {thread_stats.thread_count}")
            print(f"  - 队列大小: {thread_stats.task_queue_size}")
            print(f"  - 平均执行时间: {thread_stats.avg_execution_time:.3f}秒")
            print(f"  - 吞吐量: {thread_stats.throughput:.1f} 任务/秒")
            
        if 'resource_monitor' in metrics:
            resource_stats = metrics['resource_monitor']['current_metrics']
            print(f"\n资源使用:")
            print(f"  - CPU使用率: {resource_stats.get('cpu_percent', 0):.1f}%")
            print(f"  - 内存使用: {resource_stats.get('memory_mb', 0):.1f}MB")
            print(f"  - 线程数: {resource_stats.get('threads', 0)}")
            
        # 死锁检测
        deadlocks = optimizer.detect_deadlocks()
        if deadlocks:
            print(f"\n⚠️  检测到 {len(deadlocks)} 个死锁!")
        else:
            print(f"\n✅ 无死锁检测到")
            
        print("\n=== 压力测试完成 ===")


def benchmark_comparison():
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")
    
    def baseline_task(n):
        """基准任务"""
        result = 0
        for i in range(n):
            result += i ** 0.5
        return result
        
    def optimized_task(optimizer, n):
        """优化任务"""
        future = optimizer.execute_with_thread_pool(baseline_task, n)
        return future.result(timeout=10.0)
        
    # 测试不同优化级别
    optimization_levels = [
        OptimizationLevel.CONSERVATIVE,
        OptimizationLevel.BALANCED,
        OptimizationLevel.AGGRESSIVE
    ]
    
    num_tasks = 50
    task_size = 5000
    
    for level in optimization_levels:
        print(f"\n测试优化级别: {level.name}")
        
        with ExtendedConcurrencyOptimizer(optimization_level=level) as optimizer:
            thread_pool = optimizer.create_optimized_thread_pool(
                max_workers=8,
                min_workers=2
            )
            
            start_time = time.time()
            
            # 执行任务
            futures = []
            for i in range(num_tasks):
                future = optimizer.execute_with_thread_pool(
                    baseline_task, task_size
                )
                futures.append(future)
                
            # 等待完成
            for future in futures:
                future.result(timeout=10.0)
                
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"  - 任务数: {num_tasks}")
            print(f"  - 总耗时: {total_time:.2f}秒")
            print(f"  - 平均任务时间: {total_time/num_tasks:.3f}秒")
            print(f"  - 吞吐量: {num_tasks/total_time:.1f} 任务/秒")
            
            # 获取统计
            metrics = optimizer.get_resource_metrics()
            if 'thread_pool' in metrics:
                thread_stats = metrics['thread_pool']
                print(f"  - 线程池效率: {thread_stats.throughput:.1f} 任务/秒")
                
    print("\n=== 对比测试完成 ===")


if __name__ == "__main__":
    # 运行所有示例和测试
    try:
        # 基本使用示例
        example_usage()
        
        # 综合示例
        comprehensive_example()
        
        # 压力测试
        stress_test()
        
        # 性能对比
        benchmark_comparison()
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n所有测试完成")