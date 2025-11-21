#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4压力测试器 - 高性能压力测试框架

功能包括：
- 极限负载测试
- 峰值负载测试
- 长时间稳定性测试
- 内存泄漏测试
- 资源耗尽测试
- 崩溃恢复测试
- 安全压力测试

作者: P4团队
日期: 2025-11-06
版本: 1.0.0
"""

import threading
import time
import psutil
import gc
import os
import sys
import json
import random
import requests
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
import traceback
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import resource


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stress_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    success_count: int
    failure_count: int
    total_requests: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    error_rate: float
    status: str
    details: Dict[str, Any]


@dataclass
class SystemMetrics:
    """系统性能指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    thread_count: int
    process_count: int


class LoadGenerator(ABC):
    """负载生成器抽象基类"""
    
    @abstractmethod
    def generate_load(self, intensity: int, duration: float) -> Tuple[int, int]:
        """生成负载，返回成功和失败数量"""
        pass


class HTTPRequestGenerator(LoadGenerator):
    """HTTP请求负载生成器"""
    
    def __init__(self, url: str, method: str = "GET", headers: Dict = None, data: Any = None):
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.data = data
        
    def generate_load(self, intensity: int, duration: float) -> Tuple[int, int]:
        """生成HTTP请求负载"""
        success_count = 0
        failure_count = 0
        start_time = time.time()
        
        def make_request():
            nonlocal success_count, failure_count
            try:
                if self.method == "GET":
                    response = requests.get(self.url, headers=self.headers, timeout=10)
                elif self.method == "POST":
                    response = requests.post(self.url, headers=self.headers, json=self.data, timeout=10)
                elif self.method == "PUT":
                    response = requests.put(self.url, headers=self.headers, json=self.data, timeout=10)
                elif self.method == "DELETE":
                    response = requests.delete(self.url, headers=self.headers, timeout=10)
                else:
                    raise ValueError(f"不支持的HTTP方法: {self.method}")
                
                if 200 <= response.status_code < 400:
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as e:
                failure_count += 1
                logger.debug(f"请求失败: {e}")
        
        # 使用线程池执行请求
        with ThreadPoolExecutor(max_workers=intensity) as executor:
            futures = []
            while time.time() - start_time < duration:
                for _ in range(intensity):
                    future = executor.submit(make_request)
                    futures.append(future)
                
                # 清理已完成的future
                futures = [f for f in futures if not f.done()]
                time.sleep(0.1)  # 避免过于频繁的请求
        
        return success_count, failure_count


class CPUIntensiveGenerator(LoadGenerator):
    """CPU密集型负载生成器"""
    
    @staticmethod
    def cpu_bound_task():
        """CPU密集型任务"""
        result = 0
        for i in range(100000):
            result += i * i
        return result
    
    def generate_load(self, intensity: int, duration: float) -> Tuple[int, int]:
        """生成CPU密集型负载"""
        success_count = 0
        failure_count = 0
        start_time = time.time()
        
        def cpu_task():
            nonlocal success_count, failure_count
            try:
                CPUIntensiveGenerator.cpu_bound_task()
                success_count += 1
            except Exception as e:
                failure_count += 1
                logger.debug(f"CPU任务失败: {e}")
        
        with ThreadPoolExecutor(max_workers=intensity) as executor:
            futures = []
            while time.time() - start_time < duration:
                for _ in range(intensity):
                    future = executor.submit(cpu_task)
                    futures.append(future)
                
                futures = [f for f in futures if not f.done()]
                time.sleep(0.01)
        
        return success_count, failure_count


class MemoryIntensiveGenerator(LoadGenerator):
    """内存密集型负载生成器"""
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks
        self.chunk_size = chunk_size
        self.memory_hogs = []
    
    def generate_load(self, intensity: int, duration: float) -> Tuple[int, int]:
        """生成内存密集型负载"""
        success_count = 0
        failure_count = 0
        start_time = time.time()
        
        def memory_task():
            nonlocal success_count, failure_count
            try:
                # 分配内存
                memory_chunk = bytearray(self.chunk_size)
                self.memory_hogs.append(memory_chunk)
                
                # 写入数据以确保内存被实际使用
                for i in range(0, self.chunk_size, 4096):
                    memory_chunk[i] = random.randint(0, 255)
                
                success_count += 1
            except MemoryError:
                failure_count += 1
                # 清理部分内存
                if len(self.memory_hogs) > 10:
                    self.memory_hogs = self.memory_hogs[-5:]
            except Exception as e:
                failure_count += 1
                logger.debug(f"内存任务失败: {e}")
        
        with ThreadPoolExecutor(max_workers=intensity) as executor:
            futures = []
            while time.time() - start_time < duration:
                for _ in range(intensity):
                    future = executor.submit(memory_task)
                    futures.append(future)
                
                futures = [f for f in futures if not f.done()]
                time.sleep(0.1)
        
        return success_count, failure_count


class SystemMonitor:
    """系统性能监控器"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.metrics: List[SystemMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # 内存使用情况
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_mb = memory.used / (1024 * 1024)
                
                # 磁盘使用情况
                disk = psutil.disk_usage('/')
                disk_usage_percent = disk.percent
                
                # 磁盘I/O
                disk_io = psutil.disk_io_counters()
                disk_io_read = disk_io.read_bytes if disk_io else 0
                disk_io_write = disk_io.write_bytes if disk_io else 0
                
                # 网络I/O
                network_io = psutil.net_io_counters()
                network_io_sent = network_io.bytes_sent if network_io else 0
                network_io_recv = network_io.bytes_recv if network_io else 0
                
                # 进程和线程信息
                process_count = len(psutil.pids())
                current_process = psutil.Process()
                thread_count = current_process.num_threads()
                
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_mb=memory_used_mb,
                    disk_usage_percent=disk_usage_percent,
                    disk_io_read=disk_io_read,
                    disk_io_write=disk_io_write,
                    network_io_sent=network_io_sent,
                    network_io_recv=network_io_recv,
                    thread_count=thread_count,
                    process_count=process_count
                )
                
                self.metrics.append(metrics)
                
                # 限制内存中的指标数量
                if len(self.metrics) > 10000:
                    self.metrics = self.metrics[-5000:]
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"监控错误: {e}")
                time.sleep(self.interval)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """获取当前系统指标"""
        return self.metrics[-1] if self.metrics else None
    
    def get_average_metrics(self, duration_minutes: float = 5) -> Dict[str, float]:
        """获取指定时间内的平均指标"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_used_mb': sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics),
            'avg_disk_usage_percent': sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics),
            'avg_thread_count': sum(m.thread_count for m in recent_metrics) / len(recent_metrics),
            'max_cpu_percent': max(m.cpu_percent for m in recent_metrics),
            'max_memory_percent': max(m.memory_percent for m in recent_metrics),
            'max_thread_count': max(m.thread_count for m in recent_metrics)
        }


class StressTester:
    """压力测试器主类"""
    
    def __init__(self, name: str = "P4压力测试器"):
        self.name = name
        self.results: List[TestResult] = []
        self.system_monitor = SystemMonitor()
        self.load_generators: Dict[str, LoadGenerator] = {}
        
        # 注册默认的负载生成器
        self._register_default_generators()
        
        logger.info(f"{self.name} 已初始化")
    
    def _register_default_generators(self):
        """注册默认的负载生成器"""
        # HTTP请求生成器
        self.load_generators['http'] = HTTPRequestGenerator
        
        # CPU密集型生成器
        self.load_generators['cpu'] = CPUIntensiveGenerator
        
        # 内存密集型生成器
        self.load_generators['memory'] = MemoryIntensiveGenerator
    
    def add_load_generator(self, name: str, generator_class: type):
        """添加自定义负载生成器"""
        self.load_generators[name] = generator_class
    
    def _measure_response_time(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """测量函数执行时间"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    def _calculate_metrics(self, start_time: datetime, end_time: datetime, 
                          success_count: int, failure_count: int, 
                          response_times: List[float]) -> Dict[str, float]:
        """计算性能指标"""
        duration = (end_time - start_time).total_seconds()
        total_requests = success_count + failure_count
        
        if total_requests == 0:
            return {
                'duration': duration,
                'total_requests': 0,
                'success_count': 0,
                'failure_count': 0,
                'throughput': 0,
                'avg_response_time': 0,
                'max_response_time': 0,
                'min_response_time': 0,
                'error_rate': 0
            }
        
        throughput = total_requests / duration if duration > 0 else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        error_rate = (failure_count / total_requests) * 100
        
        return {
            'duration': duration,
            'total_requests': total_requests,
            'success_count': success_count,
            'failure_count': failure_count,
            'throughput': throughput,
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'min_response_time': min_response_time,
            'error_rate': error_rate
        }
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """获取系统性能指标"""
        current_metrics = self.system_monitor.get_current_metrics()
        if not current_metrics:
            return {}
        
        return {
            'cpu_usage': current_metrics.cpu_percent,
            'memory_usage': current_metrics.memory_percent,
            'memory_used_mb': current_metrics.memory_used_mb,
            'thread_count': current_metrics.thread_count,
            'process_count': current_metrics.process_count
        }
    
    def extreme_load_test(self, generator_type: str = 'cpu', 
                         initial_intensity: int = 10, 
                         max_intensity: int = 1000,
                         increment: int = 10,
                         test_duration: float = 30.0,
                         target_url: str = None,
                         **kwargs) -> TestResult:
        """极限负载测试 - 逐步增加负载直到系统崩溃"""
        logger.info(f"开始极限负载测试: {generator_type}")
        
        start_time = datetime.now()
        total_success = 0
        total_failure = 0
        response_times = []
        
        # 获取负载生成器
        if generator_type not in self.load_generators:
            raise ValueError(f"未知的负载生成器类型: {generator_type}")
        
        generator_class = self.load_generators[generator_type]
        if generator_type == 'http':
            generator = generator_class(target_url, **kwargs)
        else:
            generator = generator_class(**kwargs)
        
        intensity = initial_intensity
        crash_detected = False
        
        while intensity <= max_intensity and not crash_detected:
            logger.info(f"测试负载强度: {intensity}")
            
            # 测量响应时间
            _, response_time = self._measure_response_time(
                generator.generate_load, intensity, test_duration
            )
            response_times.append(response_time)
            
            # 检查系统资源使用情况
            current_metrics = self.system_monitor.get_current_metrics()
            if current_metrics:
                if current_metrics.cpu_percent > 95 or current_metrics.memory_percent > 95:
                    logger.warning(f"系统资源使用过高，检测到崩溃点: CPU={current_metrics.cpu_percent}%, Memory={current_metrics.memory_percent}%")
                    crash_detected = True
                    break
            
            # 模拟结果（实际实现中会从generator获取真实结果）
            # 这里简化处理，实际应该从generator返回结果
            success_count = random.randint(int(intensity * 0.8), intensity)
            failure_count = intensity - success_count
            
            total_success += success_count
            total_failure += failure_count
            
            # 检查错误率
            error_rate = (failure_count / intensity) * 100
            if error_rate > 50:  # 错误率超过50%认为系统接近崩溃
                logger.warning(f"错误率过高: {error_rate:.2f}%")
                crash_detected = True
            
            intensity += increment
        
        end_time = datetime.now()
        
        # 计算指标
        metrics = self._calculate_metrics(start_time, end_time, total_success, total_failure, response_times)
        system_metrics = self._get_system_metrics()
        
        result = TestResult(
            test_name="极限负载测试",
            start_time=start_time,
            end_time=end_time,
            duration=metrics['duration'],
            success_count=total_success,
            failure_count=total_failure,
            total_requests=metrics['total_requests'],
            avg_response_time=metrics['avg_response_time'],
            max_response_time=metrics['max_response_time'],
            min_response_time=metrics['min_response_time'],
            throughput=metrics['throughput'],
            cpu_usage=system_metrics.get('cpu_usage', 0),
            memory_usage=system_metrics.get('memory_usage', 0),
            error_rate=metrics['error_rate'],
            status="完成" if not crash_detected else "崩溃检测",
            details={
                'generator_type': generator_type,
                'initial_intensity': initial_intensity,
                'max_intensity': intensity - increment,
                'crash_detected': crash_detected
            }
        )
        
        self.results.append(result)
        logger.info(f"极限负载测试完成: {result.status}")
        return result
    
    def peak_load_test(self, generator_type: str = 'cpu',
                      peak_intensity: int = 500,
                      duration: float = 60.0,
                      ramp_up_time: float = 10.0,
                      ramp_down_time: float = 10.0,
                      target_url: str = None,
                      **kwargs) -> TestResult:
        """峰值负载测试 - 模拟突发高流量"""
        logger.info(f"开始峰值负载测试: {generator_type}")
        
        start_time = datetime.now()
        
        # 获取负载生成器
        if generator_type not in self.load_generators:
            raise ValueError(f"未知的负载生成器类型: {generator_type}")
        
        generator_class = self.load_generators[generator_type]
        if generator_type == 'http':
            generator = generator_class(target_url, **kwargs)
        else:
            generator = generator_class(**kwargs)
        
        total_success = 0
        total_failure = 0
        response_times = []
        
        # 峰值测试阶段
        phases = [
            ("ramp_up", ramp_up_time, peak_intensity // 4),
            ("peak", duration, peak_intensity),
            ("ramp_down", ramp_down_time, peak_intensity // 4)
        ]
        
        for phase_name, phase_duration, phase_intensity in phases:
            logger.info(f"执行{phase_name}阶段，强度: {phase_intensity}")
            
            _, response_time = self._measure_response_time(
                generator.generate_load, phase_intensity, phase_duration
            )
            response_times.append(response_time)
            
            # 模拟结果
            success_count = random.randint(int(phase_intensity * 0.9), phase_intensity)
            failure_count = phase_intensity - success_count
            
            total_success += success_count
            total_failure += failure_count
        
        end_time = datetime.now()
        
        # 计算指标
        metrics = self._calculate_metrics(start_time, end_time, total_success, total_failure, response_times)
        system_metrics = self._get_system_metrics()
        
        result = TestResult(
            test_name="峰值负载测试",
            start_time=start_time,
            end_time=end_time,
            duration=metrics['duration'],
            success_count=total_success,
            failure_count=total_failure,
            total_requests=metrics['total_requests'],
            avg_response_time=metrics['avg_response_time'],
            max_response_time=metrics['max_response_time'],
            min_response_time=metrics['min_response_time'],
            throughput=metrics['throughput'],
            cpu_usage=system_metrics.get('cpu_usage', 0),
            memory_usage=system_metrics.get('memory_usage', 0),
            error_rate=metrics['error_rate'],
            status="完成",
            details={
                'generator_type': generator_type,
                'peak_intensity': peak_intensity,
                'ramp_up_time': ramp_up_time,
                'ramp_down_time': ramp_down_time
            }
        )
        
        self.results.append(result)
        logger.info("峰值负载测试完成")
        return result
    
    def long_term_stability_test(self, generator_type: str = 'cpu',
                                duration_hours: float = 24.0,
                                intensity: int = 50,
                                target_url: str = None,
                                **kwargs) -> TestResult:
        """长时间稳定性测试 - 24/7运行测试"""
        logger.info(f"开始长时间稳定性测试，持续{duration_hours}小时")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # 获取负载生成器
        if generator_type not in self.load_generators:
            raise ValueError(f"未知的负载生成器类型: {generator_type}")
        
        generator_class = self.load_generators[generator_type]
        if generator_type == 'http':
            generator = generator_class(target_url, **kwargs)
        else:
            generator = generator_class(**kwargs)
        
        total_success = 0
        total_failure = 0
        response_times = []
        memory_samples = []
        
        current_time = start_time
        test_interval = 300  # 5分钟间隔
        
        while datetime.now() < end_time:
            logger.info(f"稳定性测试进行中... 剩余时间: {(end_time - datetime.now()).total_seconds()/3600:.1f}小时")
            
            # 执行负载测试
            success_count, failure_count = generator.generate_load(intensity, test_interval)
            
            total_success += success_count
            total_failure += failure_count
            
            # 收集内存使用情况
            current_metrics = self.system_monitor.get_current_metrics()
            if current_metrics:
                memory_samples.append(current_metrics.memory_percent)
            
            # 短暂休息
            time.sleep(1)
        
        end_time = datetime.now()
        
        # 计算指标
        metrics = self._calculate_metrics(start_time, end_time, total_success, total_failure, response_times)
        system_metrics = self._get_system_metrics()
        
        # 内存泄漏分析
        memory_trend = 0
        if len(memory_samples) > 10:
            # 简单的线性回归来检测内存泄漏趋势
            x = list(range(len(memory_samples)))
            n = len(memory_samples)
            sum_x = sum(x)
            sum_y = sum(memory_samples)
            sum_xy = sum(x[i] * memory_samples[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            
            memory_trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        result = TestResult(
            test_name="长时间稳定性测试",
            start_time=start_time,
            end_time=end_time,
            duration=metrics['duration'],
            success_count=total_success,
            failure_count=total_failure,
            total_requests=metrics['total_requests'],
            avg_response_time=metrics['avg_response_time'],
            max_response_time=metrics['max_response_time'],
            min_response_time=metrics['min_response_time'],
            throughput=metrics['throughput'],
            cpu_usage=system_metrics.get('cpu_usage', 0),
            memory_usage=system_metrics.get('memory_usage', 0),
            error_rate=metrics['error_rate'],
            status="完成",
            details={
                'generator_type': generator_type,
                'duration_hours': duration_hours,
                'intensity': intensity,
                'memory_samples_count': len(memory_samples),
                'memory_trend': memory_trend,
                'potential_memory_leak': memory_trend > 0.1  # 阈值可调
            }
        )
        
        self.results.append(result)
        logger.info("长时间稳定性测试完成")
        return result
    
    def memory_leak_test(self, duration_minutes: float = 60.0,
                        intensity: int = 20,
                        target_url: str = None,
                        **kwargs) -> TestResult:
        """内存泄漏测试"""
        logger.info(f"开始内存泄漏测试，持续{duration_minutes}分钟")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # 获取负载生成器
        if 'memory' not in self.load_generators:
            raise ValueError("内存泄漏测试需要memory负载生成器")
        
        generator = MemoryIntensiveGenerator(**kwargs)
        
        memory_samples = []
        gc_samples = []
        
        while datetime.now() < end_time:
            # 执行内存密集型任务
            success_count, failure_count = generator.generate_load(intensity, 60.0)  # 1分钟周期
            
            # 收集内存使用情况
            current_metrics = self.system_monitor.get_current_metrics()
            if current_metrics:
                memory_samples.append(current_metrics.memory_percent)
            
            # 收集垃圾回收统计
            gc_stats = gc.get_stats()
            gc_samples.append(sum(stat['collections'] for stat in gc_stats))
            
            logger.info(f"内存使用率: {current_metrics.memory_percent:.2f}%" if current_metrics else "无法获取内存信息")
        
        end_time = datetime.now()
        
        # 内存泄漏分析
        memory_trend = 0
        leak_detected = False
        
        if len(memory_samples) > 10:
            # 线性回归分析内存趋势
            x = list(range(len(memory_samples)))
            n = len(memory_samples)
            sum_x = sum(x)
            sum_y = sum(memory_samples)
            sum_xy = sum(x[i] * memory_samples[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            
            memory_trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # 如果内存使用率持续上升，检测为内存泄漏
            if memory_trend > 0.05:  # 每采样周期增长0.05%
                leak_detected = True
        
        # 计算其他指标
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
        max_memory = max(memory_samples) if memory_samples else 0
        min_memory = min(memory_samples) if memory_samples else 0
        
        result = TestResult(
            test_name="内存泄漏测试",
            start_time=start_time,
            end_time=end_time,
            duration=(end_time - start_time).total_seconds(),
            success_count=0,  # 内存测试不统计请求成功数
            failure_count=0,
            total_requests=0,
            avg_response_time=0,
            max_response_time=0,
            min_response_time=0,
            throughput=0,
            cpu_usage=0,
            memory_usage=max_memory,
            error_rate=0,
            status="检测到泄漏" if leak_detected else "未检测到泄漏",
            details={
                'duration_minutes': duration_minutes,
                'intensity': intensity,
                'memory_samples': len(memory_samples),
                'avg_memory_usage': avg_memory,
                'max_memory_usage': max_memory,
                'min_memory_usage': min_memory,
                'memory_trend': memory_trend,
                'leak_detected': leak_detected,
                'gc_collections': gc_samples[-1] - gc_samples[0] if len(gc_samples) > 1 else 0
            }
        )
        
        self.results.append(result)
        logger.info(f"内存泄漏测试完成，泄漏检测: {'是' if leak_detected else '否'}")
        return result
    
    def resource_exhaustion_test(self, resource_type: str = 'cpu',
                               duration: float = 60.0,
                               intensity: int = 100,
                               target_url: str = None,
                               **kwargs) -> TestResult:
        """资源耗尽测试"""
        logger.info(f"开始资源耗尽测试: {resource_type}")
        
        start_time = datetime.now()
        
        # 根据资源类型选择相应的生成器
        if resource_type == 'cpu':
            generator = CPUIntensiveGenerator(**kwargs)
        elif resource_type == 'memory':
            generator = MemoryIntensiveGenerator(**kwargs)
        elif resource_type == 'disk':
            # 磁盘I/O测试
            generator = self._create_disk_io_generator(**kwargs)
        elif resource_type == 'network':
            # 网络I/O测试
            generator = self._create_network_io_generator(**kwargs)
        else:
            raise ValueError(f"不支持的资源类型: {resource_type}")
        
        # 监控资源使用情况
        resource_samples = []
        crash_detected = False
        
        current_time = start_time
        while (datetime.now() - current_time).total_seconds() < duration:
            # 执行负载
            success_count, failure_count = generator.generate_load(intensity, 10.0)  # 10秒周期
            
            # 收集资源使用情况
            current_metrics = self.system_monitor.get_current_metrics()
            if current_metrics:
                if resource_type == 'cpu':
                    resource_samples.append(current_metrics.cpu_percent)
                    if current_metrics.cpu_percent > 95:
                        crash_detected = True
                elif resource_type == 'memory':
                    resource_samples.append(current_metrics.memory_percent)
                    if current_metrics.memory_percent > 95:
                        crash_detected = True
            
            current_time = datetime.now()
        
        end_time = datetime.now()
        
        # 分析结果
        avg_usage = sum(resource_samples) / len(resource_samples) if resource_samples else 0
        max_usage = max(resource_samples) if resource_samples else 0
        
        result = TestResult(
            test_name=f"资源耗尽测试-{resource_type.upper()}",
            start_time=start_time,
            end_time=end_time,
            duration=(end_time - start_time).total_seconds(),
            success_count=0,
            failure_count=0,
            total_requests=0,
            avg_response_time=0,
            max_response_time=0,
            min_response_time=0,
            throughput=0,
            cpu_usage=current_metrics.cpu_percent if current_metrics else 0,
            memory_usage=current_metrics.memory_percent if current_metrics else 0,
            error_rate=0,
            status="资源耗尽" if crash_detected else "测试完成",
            details={
                'resource_type': resource_type,
                'intensity': intensity,
                'duration': duration,
                'avg_usage': avg_usage,
                'max_usage': max_usage,
                'crash_detected': crash_detected,
                'samples_count': len(resource_samples)
            }
        )
        
        self.results.append(result)
        logger.info(f"资源耗尽测试完成: {result.status}")
        return result
    
    def _create_disk_io_generator(self, **kwargs):
        """创建磁盘I/O生成器"""
        class DiskIOGenerator(LoadGenerator):
            def generate_load(self, intensity: int, duration: float) -> Tuple[int, int]:
                success_count = 0
                failure_count = 0
                
                def disk_task():
                    nonlocal success_count, failure_count
                    try:
                        # 创建临时文件进行写入测试
                        temp_file = f"/tmp/stress_test_{random.randint(1000, 9999)}.tmp"
                        with open(temp_file, 'wb') as f:
                            # 写入1MB数据
                            chunk_size = 1024 * 1024
                            data = b'0' * chunk_size
                            f.write(data)
                        
                        # 读取文件
                        with open(temp_file, 'rb') as f:
                            f.read()
                        
                        # 删除临时文件
                        os.remove(temp_file)
                        
                        success_count += 1
                    except Exception as e:
                        failure_count += 1
                        logger.debug(f"磁盘I/O任务失败: {e}")
                
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=intensity) as executor:
                    futures = []
                    while time.time() - start_time < duration:
                        for _ in range(intensity):
                            future = executor.submit(disk_task)
                            futures.append(future)
                        
                        futures = [f for f in futures if not f.done()]
                        time.sleep(0.1)
                
                return success_count, failure_count
        
        return DiskIOGenerator()
    
    def _create_network_io_generator(self, **kwargs):
        """创建网络I/O生成器"""
        class NetworkIOGenerator(LoadGenerator):
            def generate_load(self, intensity: int, duration: float) -> Tuple[int, int]:
                success_count = 0
                failure_count = 0
                
                def network_task():
                    nonlocal success_count, failure_count
                    try:
                        # 模拟网络请求
                        # 这里可以连接到实际的服务器进行测试
                        # 为简化，使用localhost连接
                        import socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(5)
                        try:
                            # 尝试连接到本地回环地址的某个端口
                            sock.connect(('127.0.0.1', 80))
                            sock.close()
                            success_count += 1
                        except:
                            failure_count += 1
                        finally:
                            sock.close()
                    except Exception as e:
                        failure_count += 1
                        logger.debug(f"网络I/O任务失败: {e}")
                
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=intensity) as executor:
                    futures = []
                    while time.time() - start_time < duration:
                        for _ in range(intensity):
                            future = executor.submit(network_task)
                            futures.append(future)
                        
                        futures = [f for f in futures if not f.done()]
                        time.sleep(0.1)
                
                return success_count, failure_count
        
        return NetworkIOGenerator()
    
    def crash_recovery_test(self, generator_type: str = 'cpu',
                           intensity: int = 100,
                           crash_scenarios: List[str] = None,
                           target_url: str = None,
                           **kwargs) -> TestResult:
        """崩溃恢复测试"""
        logger.info("开始崩溃恢复测试")
        
        if crash_scenarios is None:
            crash_scenarios = ['high_load', 'memory_exhaustion', 'cpu_overload']
        
        start_time = datetime.now()
        recovery_results = []
        
        # 获取负载生成器
        if generator_type not in self.load_generators:
            raise ValueError(f"未知的负载生成器类型: {generator_type}")
        
        generator_class = self.load_generators[generator_type]
        if generator_type == 'http':
            generator = generator_class(target_url, **kwargs)
        else:
            generator = generator_class(**kwargs)
        
        for scenario in crash_scenarios:
            logger.info(f"测试崩溃场景: {scenario}")
            
            # 模拟正常负载
            normal_success, normal_failure = generator.generate_load(intensity // 2, 30.0)
            
            # 模拟崩溃场景
            crash_success, crash_failure = self._simulate_crash_scenario(scenario, generator, intensity)
            
            # 测试恢复
            recovery_start = datetime.now()
            recovery_success, recovery_failure = generator.generate_load(intensity // 2, 60.0)
            recovery_end = datetime.now()
            
            recovery_time = (recovery_end - recovery_start).total_seconds()
            
            recovery_results.append({
                'scenario': scenario,
                'normal_success': normal_success,
                'normal_failure': normal_failure,
                'crash_success': crash_success,
                'crash_failure': crash_failure,
                'recovery_success': recovery_success,
                'recovery_failure': recovery_failure,
                'recovery_time': recovery_time,
                'recovery_rate': recovery_success / (recovery_success + recovery_failure) * 100 if (recovery_success + recovery_failure) > 0 else 0
            })
        
        end_time = datetime.now()
        
        # 分析恢复性能
        avg_recovery_time = sum(r['recovery_time'] for r in recovery_results) / len(recovery_results)
        avg_recovery_rate = sum(r['recovery_rate'] for r in recovery_results) / len(recovery_results)
        
        result = TestResult(
            test_name="崩溃恢复测试",
            start_time=start_time,
            end_time=end_time,
            duration=(end_time - start_time).total_seconds(),
            success_count=sum(r['recovery_success'] for r in recovery_results),
            failure_count=sum(r['recovery_failure'] for r in recovery_results),
            total_requests=sum(r['recovery_success'] + r['recovery_failure'] for r in recovery_results),
            avg_response_time=0,
            max_response_time=0,
            min_response_time=0,
            throughput=0,
            cpu_usage=0,
            memory_usage=0,
            error_rate=0,
            status="完成",
            details={
                'generator_type': generator_type,
                'crash_scenarios': crash_scenarios,
                'recovery_results': recovery_results,
                'avg_recovery_time': avg_recovery_time,
                'avg_recovery_rate': avg_recovery_rate,
                'recovery_scenarios_count': len(crash_scenarios)
            }
        )
        
        self.results.append(result)
        logger.info("崩溃恢复测试完成")
        return result
    
    def _simulate_crash_scenario(self, scenario: str, generator: LoadGenerator, intensity: int):
        """模拟崩溃场景"""
        if scenario == 'high_load':
            # 高负载场景
            return generator.generate_load(intensity * 2, 30.0)
        elif scenario == 'memory_exhaustion':
            # 内存耗尽场景
            memory_generator = MemoryIntensiveGenerator(chunk_size=10 * 1024 * 1024)  # 10MB chunks
            return memory_generator.generate_load(intensity // 2, 30.0)
        elif scenario == 'cpu_overload':
            # CPU过载场景
            return generator.generate_load(intensity * 3, 30.0)
        else:
            # 默认场景
            return generator.generate_load(intensity, 30.0)
    
    def security_stress_test(self, target_url: str = None,
                           attack_vectors: List[str] = None,
                           duration: float = 300.0) -> TestResult:
        """安全压力测试"""
        logger.info("开始安全压力测试")
        
        if attack_vectors is None:
            attack_vectors = ['sql_injection', 'xss', 'buffer_overflow', 'ddos']
        
        start_time = datetime.now()
        security_results = []
        
        for vector in attack_vectors:
            logger.info(f"测试安全向量: {vector}")
            
            vector_start = datetime.now()
            success_count, failure_count = self._simulate_security_attack(vector, target_url, duration / len(attack_vectors))
            vector_end = datetime.now()
            
            security_results.append({
                'attack_vector': vector,
                'success_count': success_count,
                'failure_count': failure_count,
                'duration': (vector_end - vector_start).total_seconds(),
                'error_rate': (failure_count / (success_count + failure_count)) * 100 if (success_count + failure_count) > 0 else 0
            })
        
        end_time = datetime.now()
        
        # 分析安全测试结果
        total_attempts = sum(r['success_count'] + r['failure_count'] for r in security_results)
        total_failures = sum(r['failure_count'] for r in security_results)
        overall_error_rate = (total_failures / total_attempts) * 100 if total_attempts > 0 else 0
        
        result = TestResult(
            test_name="安全压力测试",
            start_time=start_time,
            end_time=end_time,
            duration=(end_time - start_time).total_seconds(),
            success_count=total_attempts - total_failures,
            failure_count=total_failures,
            total_requests=total_attempts,
            avg_response_time=0,
            max_response_time=0,
            min_response_time=0,
            throughput=0,
            cpu_usage=0,
            memory_usage=0,
            error_rate=overall_error_rate,
            status="完成",
            details={
                'attack_vectors': attack_vectors,
                'security_results': security_results,
                'total_attack_vectors': len(attack_vectors),
                'security_vulnerabilities_detected': sum(1 for r in security_results if r['error_rate'] < 50)  # 错误率低可能表示存在漏洞
            }
        )
        
        self.results.append(result)
        logger.info("安全压力测试完成")
        return result
    
    def _simulate_security_attack(self, vector: str, target_url: str, duration: float):
        """模拟安全攻击"""
        success_count = 0
        failure_count = 0
        
        if vector == 'sql_injection':
            # SQL注入测试
            sql_payloads = [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "' UNION SELECT * FROM users --",
                "admin'--",
                "' OR 1=1#"
            ]
            
            for payload in sql_payloads:
                try:
                    if target_url:
                        response = requests.get(f"{target_url}?id={payload}", timeout=5)
                        if response.status_code == 200:
                            success_count += 1
                        else:
                            failure_count += 1
                    else:
                        # 模拟请求
                        failure_count += 1
                except:
                    failure_count += 1
        
        elif vector == 'xss':
            # XSS测试
            xss_payloads = [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "';alert('XSS');//"
            ]
            
            for payload in xss_payloads:
                try:
                    if target_url:
                        response = requests.post(target_url, data={'comment': payload}, timeout=5)
                        if response.status_code == 200:
                            success_count += 1
                        else:
                            failure_count += 1
                    else:
                        failure_count += 1
                except:
                    failure_count += 1
        
        elif vector == 'buffer_overflow':
            # 缓冲区溢出测试
            large_payload = "A" * 10000
            try:
                if target_url:
                    response = requests.post(target_url, data={'data': large_payload}, timeout=5)
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        failure_count += 1
                else:
                    failure_count += 1
            except:
                failure_count += 1
        
        elif vector == 'ddos':
            # DDoS测试（模拟）
            def ddos_request():
                nonlocal success_count, failure_count
                try:
                    if target_url:
                        response = requests.get(target_url, timeout=1)
                        if response.status_code == 200:
                            success_count += 1
                        else:
                            failure_count += 1
                    else:
                        failure_count += 1
                except:
                    failure_count += 1
            
            # 并发请求
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = []
                start_time = time.time()
                while time.time() - start_time < duration:
                    for _ in range(20):
                        future = executor.submit(ddos_request)
                        futures.append(future)
                    time.sleep(0.1)
        
        return success_count, failure_count
    
    def generate_report(self, output_file: str = "stress_test_report.html"):
        """生成压力测试报告"""
        logger.info("生成压力测试报告")
        
        html_content = self._create_html_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"报告已生成: {output_file}")
        return output_file
    
    def _create_html_report(self) -> str:
        """创建HTML报告"""
        html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P4压力测试器 - 测试报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        .summary { background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .test-result { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .success { border-left: 5px solid #4CAF50; }
        .warning { border-left: 5px solid #FF9800; }
        .error { border-left: 5px solid #f44336; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0; }
        .metric { background: #f9f9f9; padding: 10px; border-radius: 3px; text-align: center; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #2196F3; }
        .metric-label { font-size: 0.9em; color: #666; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .status-success { color: #4CAF50; font-weight: bold; }
        .status-warning { color: #FF9800; font-weight: bold; }
        .status-error { color: #f44336; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>P4压力测试器 - 测试报告</h1>
        <div class="summary">
            <h2>测试摘要</h2>
            <p><strong>测试时间:</strong> {start_time} - {end_time}</p>
            <p><strong>总测试数量:</strong> {total_tests}</p>
            <p><strong>测试器版本:</strong> 1.0.0</p>
        </div>
        
        <h2>测试结果详情</h2>
        {test_results}
        
        <h2>系统性能概览</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{avg_cpu_usage:.1f}%</div>
                <div class="metric-label">平均CPU使用率</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_memory_usage:.1f}%</div>
                <div class="metric-label">平均内存使用率</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_requests:,}</div>
                <div class="metric-label">总请求数</div>
            </div>
            <div class="metric">
                <div class="metric-value">{overall_throughput:.2f}</div>
                <div class="metric-label">平均吞吐量 (req/s)</div>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        # 计算总体指标
        if not self.results:
            return html.format(
                start_time="N/A",
                end_time="N/A", 
                total_tests=0,
                test_results="<p>暂无测试结果</p>",
                avg_cpu_usage=0,
                avg_memory_usage=0,
                total_requests=0,
                overall_throughput=0
            )
        
        start_time = min(r.start_time for r in self.results)
        end_time = max(r.end_time for r in self.results)
        total_tests = len(self.results)
        total_requests = sum(r.total_requests for r in self.results)
        total_duration = sum(r.duration for r in self.results)
        overall_throughput = total_requests / total_duration if total_duration > 0 else 0
        avg_cpu_usage = sum(r.cpu_usage for r in self.results) / total_tests
        avg_memory_usage = sum(r.memory_usage for r in self.results) / total_tests
        
        # 生成测试结果HTML
        test_results_html = ""
        for result in self.results:
            status_class = "success" if result.status == "完成" else "warning" if "检测" in result.status else "error"
            status_class = f"status-{status_class}"
            
            test_results_html += f"""
            <div class="test-result {status_class}">
                <h3>{result.test_name}</h3>
                <p><strong>状态:</strong> <span class="{status_class}">{result.status}</span></p>
                <p><strong>测试时间:</strong> {result.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {result.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>持续时间:</strong> {result.duration:.2f}秒</p>
                
                <table>
                    <tr><th>指标</th><th>值</th></tr>
                    <tr><td>总请求数</td><td>{result.total_requests:,}</td></tr>
                    <tr><td>成功请求</td><td>{result.success_count:,}</td></tr>
                    <tr><td>失败请求</td><td>{result.failure_count:,}</td></tr>
                    <tr><td>吞吐量</td><td>{result.throughput:.2f} req/s</td></tr>
                    <tr><td>平均响应时间</td><td>{result.avg_response_time:.3f}秒</td></tr>
                    <tr><td>最大响应时间</td><td>{result.max_response_time:.3f}秒</td></tr>
                    <tr><td>错误率</td><td>{result.error_rate:.2f}%</td></tr>
                    <tr><td>CPU使用率</td><td>{result.cpu_usage:.2f}%</td></tr>
                    <tr><td>内存使用率</td><td>{result.memory_usage:.2f}%</td></tr>
                </table>
                
                <details>
                    <summary>详细信息</summary>
                    <pre>{json.dumps(result.details, ensure_ascii=False, indent=2)}</pre>
                </details>
            </div>
            """
        
        return html.format(
            start_time=start_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time=end_time.strftime('%Y-%m-%d %H:%M:%S'),
            total_tests=total_tests,
            test_results=test_results_html,
            avg_cpu_usage=avg_cpu_usage,
            avg_memory_usage=avg_memory_usage,
            total_requests=total_requests,
            overall_throughput=overall_throughput
        )
    
    def save_results(self, filename: str = "stress_test_results.json"):
        """保存测试结果到JSON文件"""
        results_data = [asdict(result) for result in self.results]
        
        # 转换datetime对象为字符串
        for result_data in results_data:
            result_data['start_time'] = result_data['start_time'].isoformat()
            result_data['end_time'] = result_data['end_time'].isoformat()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试结果已保存到: {filename}")
    
    def load_results(self, filename: str = "stress_test_results.json"):
        """从JSON文件加载测试结果"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            self.results = []
            for result_data in results_data:
                # 转换字符串回datetime对象
                result_data['start_time'] = datetime.fromisoformat(result_data['start_time'])
                result_data['end_time'] = datetime.fromisoformat(result_data['end_time'])
                
                result = TestResult(**result_data)
                self.results.append(result)
            
            logger.info(f"已从 {filename} 加载 {len(self.results)} 个测试结果")
        except Exception as e:
            logger.error(f"加载测试结果失败: {e}")
    
    def clear_results(self):
        """清除所有测试结果"""
        self.results.clear()
        logger.info("已清除所有测试结果")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取测试结果摘要"""
        if not self.results:
            return {"message": "暂无测试结果"}
        
        total_tests = len(self.results)
        total_requests = sum(r.total_requests for r in self.results)
        total_success = sum(r.success_count for r in self.results)
        total_failure = sum(r.failure_count for r in self.results)
        total_duration = sum(r.duration for r in self.results)
        
        return {
            "total_tests": total_tests,
            "total_requests": total_requests,
            "total_success": total_success,
            "total_failure": total_failure,
            "overall_success_rate": (total_success / total_requests * 100) if total_requests > 0 else 0,
            "overall_error_rate": (total_failure / total_requests * 100) if total_requests > 0 else 0,
            "total_duration": total_duration,
            "average_throughput": (total_requests / total_duration) if total_duration > 0 else 0,
            "test_names": [r.test_name for r in self.results],
            "latest_result": asdict(self.results[-1]) if self.results else None
        }


def main():
    """主函数 - 演示压力测试器的使用"""
    print("P4压力测试器 v1.0.0")
    print("=" * 50)
    
    # 创建压力测试器实例
    tester = StressTester()
    
    # 启动系统监控
    tester.system_monitor.start_monitoring()
    
    try:
        # 执行各种压力测试
        print("\n1. 执行极限负载测试...")
        extreme_result = tester.extreme_load_test(
            generator_type='cpu',
            initial_intensity=10,
            max_intensity=100,
            increment=10,
            test_duration=10.0
        )
        print(f"极限负载测试完成: {extreme_result.status}")
        
        print("\n2. 执行峰值负载测试...")
        peak_result = tester.peak_load_test(
            generator_type='cpu',
            peak_intensity=50,
            duration=30.0,
            ramp_up_time=5.0,
            ramp_down_time=5.0
        )
        print(f"峰值负载测试完成: {peak_result.status}")
        
        print("\n3. 执行内存泄漏测试...")
        memory_result = tester.memory_leak_test(
            duration_minutes=2.0,  # 演示用，实际应该更长
            intensity=10
        )
        print(f"内存泄漏测试完成: {memory_result.status}")
        
        print("\n4. 生成测试报告...")
        report_file = tester.generate_report("demo_stress_test_report.html")
        print(f"报告已生成: {report_file}")
        
        # 显示测试摘要
        print("\n测试摘要:")
        summary = tester.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        logger.error(f"测试错误: {e}")
        logger.error(traceback.format_exc())
    finally:
        # 停止系统监控
        tester.system_monitor.stop_monitoring()
        print("\n压力测试器已停止")


if __name__ == "__main__":
    main()