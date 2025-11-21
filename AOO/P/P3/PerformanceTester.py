"""
P3性能测试器主要实现
提供全面的性能测试功能，包括负载测试、压力测试、基准测试等
"""

import time
import json
import threading
import statistics
import psutil
import requests
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import sqlite3
import warnings

# 抑制matplotlib警告
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_times: List[float]
    throughput_history: List[float]
    cpu_usage_history: List[float]
    memory_usage_history: List[float]
    error_counts: List[int]
    timestamp_history: List[datetime]


class PerformanceTester:
    """性能测试器主类"""
    
    def __init__(self, base_url: str = "http://localhost:8000", db_path: str = "performance_data.db"):
        """
        初始化性能测试器
        
        Args:
            base_url: 被测试的API基础URL
            db_path: 数据库文件路径
        """
        self.base_url = base_url.rstrip('/')
        self.db_path = db_path
        self.logger = self._setup_logger()
        self._init_database()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('PerformanceTester')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _init_database(self):
        """初始化SQLite数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration REAL NOT NULL,
                    total_requests INTEGER NOT NULL,
                    successful_requests INTEGER NOT NULL,
                    failed_requests INTEGER NOT NULL,
                    avg_response_time REAL NOT NULL,
                    min_response_time REAL NOT NULL,
                    max_response_time REAL NOT NULL,
                    p95_response_time REAL NOT NULL,
                    p99_response_time REAL NOT NULL,
                    throughput REAL NOT NULL,
                    error_rate REAL NOT NULL,
                    cpu_usage REAL NOT NULL,
                    memory_usage REAL NOT NULL,
                    disk_usage REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS baseline_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    baseline_response_time REAL NOT NULL,
                    baseline_throughput REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def _make_request(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        发送HTTP请求并测量响应时间
        
        Args:
            endpoint: API端点
            method: HTTP方法
            data: 请求数据
            headers: 请求头
            
        Returns:
            Tuple[bool, float]: (是否成功, 响应时间)
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, headers=headers, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                response = requests.request(method, url, json=data, headers=headers, timeout=30)
            
            response_time = time.time() - start_time
            
            # 检查响应状态码
            success = 200 <= response.status_code < 400
            
            return success, response_time
            
        except requests.exceptions.Timeout:
            return False, 30.0
        except requests.exceptions.RequestException:
            return False, 0.0
        except Exception:
            return False, 0.0
    
    def _monitor_resources(self, duration: float, interval: float = 1.0) -> Dict[str, List[float]]:
        """
        监控系统资源使用情况
        
        Args:
            duration: 监控持续时间（秒）
            interval: 监控间隔（秒）
            
        Returns:
            Dict[str, List[float]]: 资源使用历史数据
        """
        cpu_history = []
        memory_history = []
        disk_history = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_history.append(cpu_percent)
                
                # 内存使用率
                memory = psutil.virtual_memory()
                memory_history.append(memory.percent)
                
                # 磁盘使用率
                disk = psutil.disk_usage('/')
                disk_history.append(disk.percent)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.warning(f"资源监控出错: {e}")
                break
        
        return {
            'cpu': cpu_history,
            'memory': memory_history,
            'disk': disk_history
        }
    
    def response_time_test(self, endpoint: str, num_requests: int = 100, 
                          concurrent_users: int = 1) -> TestResult:
        """
        响应时间测试
        
        Args:
            endpoint: API端点
            num_requests: 请求总数
            concurrent_users: 并发用户数
            
        Returns:
            TestResult: 测试结果
        """
        self.logger.info(f"开始响应时间测试: {endpoint}, 请求数: {num_requests}, 并发: {concurrent_users}")
        
        start_time = datetime.now()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # 启动资源监控
        resource_monitor = threading.Thread(
            target=self._monitor_resources, 
            args=(num_requests * 2, 0.5)
        )
        resource_monitor.start()
        
        if concurrent_users == 1:
            # 单线程测试
            for _ in range(num_requests):
                success, response_time = self._make_request(endpoint)
                response_times.append(response_time)
                
                if success:
                    successful_requests += 1
                else:
                    failed_requests += 1
                    
                time.sleep(0.1)  # 避免过快请求
        else:
            # 并发测试
            def make_requests(batch_size):
                batch_times = []
                batch_success = 0
                batch_failed = 0
                
                for _ in range(batch_size):
                    success, response_time = self._make_request(endpoint)
                    batch_times.append(response_time)
                    
                    if success:
                        batch_success += 1
                    else:
                        batch_failed += 1
                
                return batch_times, batch_success, batch_failed
            
            requests_per_user = num_requests // concurrent_users
            remaining_requests = num_requests % concurrent_users
            
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = []
                
                # 为每个并发用户分配任务
                for i in range(concurrent_users):
                    batch_size = requests_per_user + (1 if i < remaining_requests else 0)
                    future = executor.submit(make_requests, batch_size)
                    futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    batch_times, batch_success, batch_failed = future.result()
                    response_times.extend(batch_times)
                    successful_requests += batch_success
                    failed_requests += batch_failed
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 等待资源监控完成
        resource_monitor.join()
        
        # 计算统计指标
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            if len(response_times) >= 95:
                p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            else:
                p95_response_time = max_response_time
                
            if len(response_times) >= 99:
                p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
            else:
                p99_response_time = max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        throughput = successful_requests / duration if duration > 0 else 0
        error_rate = failed_requests / num_requests * 100 if num_requests > 0 else 0
        
        # 获取资源使用情况（简化版）
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        result = TestResult(
            test_name=f"响应时间测试_{endpoint}",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            total_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput=throughput,
            error_rate=error_rate,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage
        )
        
        self._save_test_result(result)
        self.logger.info(f"响应时间测试完成: 平均响应时间 {avg_response_time:.3f}s, 吞吐量 {throughput:.2f} req/s")
        
        return result
    
    def throughput_test(self, endpoint: str, duration: int = 60, 
                       concurrent_users: int = 10) -> TestResult:
        """
        吞吐量测试
        
        Args:
            endpoint: API端点
            duration: 测试持续时间（秒）
            concurrent_users: 并发用户数
            
        Returns:
            TestResult: 测试结果
        """
        self.logger.info(f"开始吞吐量测试: {endpoint}, 持续时间: {duration}s, 并发: {concurrent_users}")
        
        start_time = datetime.now()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # 启动资源监控
        resource_monitor = threading.Thread(
            target=self._monitor_resources, 
            args=(duration, 1.0)
        )
        resource_monitor.start()
        
        def continuous_requests():
            nonlocal successful_requests, failed_requests
            end_time = time.time() + duration
            
            while time.time() < end_time:
                success, response_time = self._make_request(endpoint)
                response_times.append(response_time)
                
                if success:
                    successful_requests += 1
                else:
                    failed_requests += 1
                
                time.sleep(0.01)  # 短暂延迟
        
        # 启动并发用户
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(continuous_requests) for _ in range(concurrent_users)]
            
            # 等待所有任务完成
            for future in as_completed(futures):
                future.result()
        
        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()
        
        # 等待资源监控完成
        resource_monitor.join()
        
        # 计算统计指标
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            if len(response_times) >= 95:
                p95_response_time = statistics.quantiles(response_times, n=20)[18]
            else:
                p95_response_time = max_response_time
                
            if len(response_times) >= 99:
                p99_response_time = statistics.quantiles(response_times, n=100)[98]
            else:
                p99_response_time = max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        total_requests = successful_requests + failed_requests
        throughput = successful_requests / actual_duration if actual_duration > 0 else 0
        error_rate = failed_requests / total_requests * 100 if total_requests > 0 else 0
        
        # 获取资源使用情况
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        result = TestResult(
            test_name=f"吞吐量测试_{endpoint}",
            start_time=start_time,
            end_time=end_time,
            duration=actual_duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput=throughput,
            error_rate=error_rate,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage
        )
        
        self._save_test_result(result)
        self.logger.info(f"吞吐量测试完成: 总请求数 {total_requests}, 吞吐量 {throughput:.2f} req/s")
        
        return result
    
    def stress_test(self, endpoint: str, max_concurrent_users: int = 100, 
                   step_size: int = 10, duration_per_step: int = 30) -> List[TestResult]:
        """
        压力测试
        
        Args:
            endpoint: API端点
            max_concurrent_users: 最大并发用户数
            step_size: 每次增加的并发用户数
            duration_per_step: 每个并发级别的测试持续时间
            
        Returns:
            List[TestResult]: 所有测试级别的结果
        """
        self.logger.info(f"开始压力测试: {endpoint}, 最大并发: {max_concurrent_users}")
        
        results = []
        
        for concurrent_users in range(step_size, max_concurrent_users + 1, step_size):
            self.logger.info(f"测试并发用户数: {concurrent_users}")
            
            result = self.throughput_test(
                endpoint=endpoint,
                duration=duration_per_step,
                concurrent_users=concurrent_users
            )
            
            results.append(result)
            
            # 如果错误率过高，停止测试
            if result.error_rate > 50:
                self.logger.warning(f"错误率过高 ({result.error_rate:.1f}%)，停止压力测试")
                break
        
        return results
    
    def benchmark_test(self, endpoints: List[str], baseline_data: Optional[Dict] = None) -> Dict[str, TestResult]:
        """
        基准测试
        
        Args:
            endpoints: 要测试的端点列表
            baseline_data: 基准数据字典 {endpoint: {response_time: float, throughput: float}}
            
        Returns:
            Dict[str, TestResult]: 各端点的测试结果
        """
        self.logger.info(f"开始基准测试: {len(endpoints)} 个端点")
        
        results = {}
        
        for endpoint in endpoints:
            self.logger.info(f"测试端点: {endpoint}")
            
            result = self.response_time_test(endpoint, num_requests=50, concurrent_users=5)
            results[endpoint] = result
            
            # 与基准数据对比
            if baseline_data and endpoint in baseline_data:
                baseline = baseline_data[endpoint]
                
                response_time_diff = (result.avg_response_time - baseline['response_time']) / baseline['response_time'] * 100
                throughput_diff = (result.throughput - baseline['throughput']) / baseline['throughput'] * 100
                
                self.logger.info(
                    f"基准对比 - {endpoint}: "
                    f"响应时间变化 {response_time_diff:+.1f}%, "
                    f"吞吐量变化 {throughput_diff:+.1f}%"
                )
        
        return results
    
    def resource_usage_test(self, endpoint: str, duration: int = 300) -> TestResult:
        """
        资源使用测试
        
        Args:
            endpoint: API端点
            duration: 测试持续时间（秒）
            
        Returns:
            TestResult: 测试结果
        """
        self.logger.info(f"开始资源使用测试: {endpoint}, 持续时间: {duration}s")
        
        start_time = datetime.now()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # 启动详细资源监控
        resource_data = {'cpu': [], 'memory': [], 'disk': []}
        
        def resource_monitor():
            start = time.time()
            while time.time() - start < duration:
                try:
                    resource_data['cpu'].append(psutil.cpu_percent())
                    resource_data['memory'].append(psutil.virtual_memory().percent)
                    resource_data['disk'].append(psutil.disk_usage('/').percent)
                    time.sleep(5)  # 每5秒监控一次
                except Exception as e:
                    self.logger.warning(f"资源监控出错: {e}")
                    break
        
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.start()
        
        def continuous_load():
            nonlocal successful_requests, failed_requests
            end_time = time.time() + duration
            
            while time.time() < end_time:
                success, response_time = self._make_request(endpoint)
                response_times.append(response_time)
                
                if success:
                    successful_requests += 1
                else:
                    failed_requests += 1
                
                time.sleep(0.1)  # 控制请求频率
        
        # 启动多个线程产生持续负载
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(continuous_load) for _ in range(10)]
            
            for future in as_completed(futures):
                future.result()
        
        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()
        
        # 等待监控线程完成
        monitor_thread.join()
        
        # 计算统计指标
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            if len(response_times) >= 95:
                p95_response_time = statistics.quantiles(response_times, n=20)[18]
            else:
                p95_response_time = max_response_time
                
            if len(response_times) >= 99:
                p99_response_time = statistics.quantiles(response_times, n=100)[98]
            else:
                p99_response_time = max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        total_requests = successful_requests + failed_requests
        throughput = successful_requests / actual_duration if actual_duration > 0 else 0
        error_rate = failed_requests / total_requests * 100 if total_requests > 0 else 0
        
        # 计算平均资源使用率
        avg_cpu = statistics.mean(resource_data['cpu']) if resource_data['cpu'] else 0
        avg_memory = statistics.mean(resource_data['memory']) if resource_data['memory'] else 0
        avg_disk = statistics.mean(resource_data['disk']) if resource_data['disk'] else 0
        
        result = TestResult(
            test_name=f"资源使用测试_{endpoint}",
            start_time=start_time,
            end_time=end_time,
            duration=actual_duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput=throughput,
            error_rate=error_rate,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            disk_usage=avg_disk
        )
        
        self._save_test_result(result)
        self.logger.info(f"资源使用测试完成: CPU {avg_cpu:.1f}%, 内存 {avg_memory:.1f}%")
        
        return result
    
    def performance_regression_test(self, endpoint: str, days_back: int = 7) -> Dict[str, Any]:
        """
        性能回归检测
        
        Args:
            endpoint: API端点
            days_back: 回溯天数
            
        Returns:
            Dict[str, Any]: 回归检测结果
        """
        self.logger.info(f"开始性能回归检测: {endpoint}, 回溯 {days_back} 天")
        
        # 获取历史数据
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM test_results 
                WHERE test_name LIKE ? AND created_at >= ?
                ORDER BY created_at DESC
            ''', (f'%{endpoint}%', datetime.now() - timedelta(days=days_back)))
            
            historical_results = [dict(row) for row in cursor.fetchall()]
        
        if len(historical_results) < 2:
            return {
                'status': 'insufficient_data',
                'message': f'历史数据不足，需要至少2条记录，当前有{len(historical_results)}条'
            }
        
        # 计算历史平均值
        avg_response_times = [r['avg_response_time'] for r in historical_results]
        avg_throughputs = [r['throughput'] for r in historical_results]
        
        historical_avg_response = statistics.mean(avg_response_times)
        historical_avg_throughput = statistics.mean(avg_throughputs)
        
        # 运行当前测试
        current_result = self.response_time_test(endpoint, num_requests=30, concurrent_users=3)
        
        # 计算性能变化
        response_time_change = (current_result.avg_response_time - historical_avg_response) / historical_avg_response * 100
        throughput_change = (current_result.throughput - historical_avg_throughput) / historical_avg_throughput * 100
        
        # 判断是否存在性能回归
        regression_detected = False
        severity = 'none'
        
        if response_time_change > 20:  # 响应时间增加超过20%
            regression_detected = True
            severity = 'high' if response_time_change > 50 else 'medium'
        elif response_time_change > 10:  # 响应时间增加超过10%
            regression_detected = True
            severity = 'low'
        
        if throughput_change < -20:  # 吞吐量下降超过20%
            regression_detected = True
            if severity == 'none':
                severity = 'high' if throughput_change < -50 else 'medium'
        
        result = {
            'status': 'completed',
            'endpoint': endpoint,
            'current_result': current_result.to_dict(),
            'historical_avg_response_time': historical_avg_response,
            'historical_avg_throughput': historical_avg_throughput,
            'response_time_change_percent': response_time_change,
            'throughput_change_percent': throughput_change,
            'regression_detected': regression_detected,
            'severity': severity,
            'historical_records_count': len(historical_results)
        }
        
        self.logger.info(f"性能回归检测完成: {endpoint}, 回归状态: {'检测到' if regression_detected else '未检测到'}")
        
        return result
    
    def identify_bottlenecks(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """
        识别性能瓶颈
        
        Args:
            test_results: 测试结果列表
            
        Returns:
            Dict[str, Any]: 瓶颈分析结果
        """
        self.logger.info("开始性能瓶颈识别")
        
        if not test_results:
            return {'status': 'error', 'message': '没有测试结果数据'}
        
        bottlenecks = []
        recommendations = []
        
        # 分析响应时间
        avg_response_times = [r.avg_response_time for r in test_results]
        max_response_times = [r.max_response_time for r in test_results]
        
        if avg_response_times:
            overall_avg_response = statistics.mean(avg_response_times)
            overall_max_response = max(max_response_times)
            
            if overall_avg_response > 2.0:  # 平均响应时间超过2秒
                bottlenecks.append({
                    'type': 'high_response_time',
                    'description': f'平均响应时间过高: {overall_avg_response:.2f}s',
                    'severity': 'high' if overall_avg_response > 5.0 else 'medium',
                    'value': overall_avg_response
                })
                
                recommendations.append('优化数据库查询或增加缓存机制')
                recommendations.append('检查网络延迟和服务器性能')
        
        # 分析错误率
        error_rates = [r.error_rate for r in test_results]
        if error_rates:
            avg_error_rate = statistics.mean(error_rates)
            
            if avg_error_rate > 5.0:  # 平均错误率超过5%
                bottlenecks.append({
                    'type': 'high_error_rate',
                    'description': f'错误率过高: {avg_error_rate:.1f}%',
                    'severity': 'high' if avg_error_rate > 20.0 else 'medium',
                    'value': avg_error_rate
                })
                
                recommendations.append('检查API稳定性和错误处理')
                recommendations.append('验证输入参数和业务逻辑')
        
        # 分析资源使用
        cpu_usages = [r.cpu_usage for r in test_results]
        memory_usages = [r.memory_usage for r in test_results]
        
        if cpu_usages:
            avg_cpu = statistics.mean(cpu_usages)
            if avg_cpu > 80.0:  # CPU使用率超过80%
                bottlenecks.append({
                    'type': 'high_cpu_usage',
                    'description': f'CPU使用率过高: {avg_cpu:.1f}%',
                    'severity': 'high' if avg_cpu > 95.0 else 'medium',
                    'value': avg_cpu
                })
                
                recommendations.append('优化算法复杂度或增加服务器资源')
                recommendations.append('考虑负载均衡和水平扩展')
        
        if memory_usages:
            avg_memory = statistics.mean(memory_usages)
            if avg_memory > 80.0:  # 内存使用率超过80%
                bottlenecks.append({
                    'type': 'high_memory_usage',
                    'description': f'内存使用率过高: {avg_memory:.1f}%',
                    'severity': 'high' if avg_memory > 95.0 else 'medium',
                    'value': avg_memory
                })
                
                recommendations.append('检查内存泄漏和优化内存使用')
                recommendations.append('增加内存资源或优化数据结构')
        
        # 分析吞吐量
        throughputs = [r.throughput for r in test_results]
        if throughputs:
            avg_throughput = statistics.mean(throughputs)
            if avg_throughput < 10.0:  # 吞吐量过低
                bottlenecks.append({
                    'type': 'low_throughput',
                    'description': f'系统吞吐量较低: {avg_throughput:.2f} req/s',
                    'severity': 'medium',
                    'value': avg_throughput
                })
                
                recommendations.append('优化并发处理能力')
                recommendations.append('考虑数据库连接池优化')
        
        # 生成总体建议
        if not bottlenecks:
            recommendations.append('系统性能良好，继续保持当前配置')
        else:
            recommendations.append('建议优先解决高严重性瓶颈问题')
            recommendations.append('定期进行性能监控和测试')
        
        result = {
            'status': 'completed',
            'total_tests': len(test_results),
            'bottlenecks_identified': len(bottlenecks),
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'analysis_summary': {
                'avg_response_time': statistics.mean(avg_response_times) if avg_response_times else 0,
                'avg_error_rate': statistics.mean(error_rates) if error_rates else 0,
                'avg_cpu_usage': statistics.mean(cpu_usages) if cpu_usages else 0,
                'avg_memory_usage': statistics.mean(memory_usages) if memory_usages else 0,
                'avg_throughput': statistics.mean(throughputs) if throughputs else 0
            }
        }
        
        self.logger.info(f"性能瓶颈识别完成: 发现 {len(bottlenecks)} 个瓶颈")
        
        return result
    
    def generate_performance_report(self, test_results: List[TestResult], 
                                   output_path: str = "performance_report.html") -> str:
        """
        生成性能报告
        
        Args:
            test_results: 测试结果列表
            output_path: 输出文件路径
            
        Returns:
            str: 报告文件路径
        """
        self.logger.info("生成性能报告")
        
        if not test_results:
            return "没有测试结果数据"
        
        # 创建图表目录
        charts_dir = Path("charts")
        charts_dir.mkdir(exist_ok=True)
        
        # 生成各种图表
        self._generate_response_time_chart(test_results, charts_dir)
        self._generate_throughput_chart(test_results, charts_dir)
        self._generate_resource_usage_chart(test_results, charts_dir)
        self._generate_error_rate_chart(test_results, charts_dir)
        
        # 生成HTML报告
        html_content = self._create_html_report(test_results, charts_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"性能报告已生成: {output_path}")
        
        return output_path
    
    def _generate_response_time_chart(self, test_results: List[TestResult], charts_dir: Path):
        """生成响应时间图表"""
        plt.figure(figsize=(12, 6))
        
        test_names = [r.test_name for r in test_results]
        avg_times = [r.avg_response_time for r in test_results]
        p95_times = [r.p95_response_time for r in test_results]
        p99_times = [r.p99_response_time for r in test_results]
        
        x = range(len(test_names))
        
        plt.subplot(1, 2, 1)
        plt.bar(x, avg_times, alpha=0.7, label='平均响应时间')
        plt.bar(x, p95_times, alpha=0.7, label='95%响应时间')
        plt.bar(x, p99_times, alpha=0.7, label='99%响应时间')
        plt.xlabel('测试')
        plt.ylabel('响应时间 (秒)')
        plt.title('响应时间对比')
        plt.xticks(x, [name.split('_')[-1] for name in test_names], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(x, avg_times, marker='o', label='平均响应时间')
        plt.plot(x, p95_times, marker='s', label='95%响应时间')
        plt.plot(x, p99_times, marker='^', label='99%响应时间')
        plt.xlabel('测试序号')
        plt.ylabel('响应时间 (秒)')
        plt.title('响应时间趋势')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(charts_dir / "response_time_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_throughput_chart(self, test_results: List[TestResult], charts_dir: Path):
        """生成吞吐量图表"""
        plt.figure(figsize=(10, 6))
        
        test_names = [r.test_name for r in test_results]
        throughputs = [r.throughput for r in test_results]
        
        x = range(len(test_names))
        plt.bar(x, throughputs, alpha=0.7, color='green')
        plt.xlabel('测试')
        plt.ylabel('吞吐量 (请求/秒)')
        plt.title('系统吞吐量对比')
        plt.xticks(x, [name.split('_')[-1] for name in test_names], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(throughputs):
            plt.text(i, v + max(throughputs) * 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(charts_dir / "throughput_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_resource_usage_chart(self, test_results: List[TestResult], charts_dir: Path):
        """生成资源使用图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        test_names = [r.test_name for r in test_results]
        cpu_usages = [r.cpu_usage for r in test_results]
        memory_usages = [r.memory_usage for r in test_results]
        
        x = range(len(test_names))
        
        # CPU使用率
        bars1 = ax1.bar(x, cpu_usages, alpha=0.7, color='red', label='CPU使用率')
        ax1.set_xlabel('测试')
        ax1.set_ylabel('CPU使用率 (%)')
        ax1.set_title('CPU使用率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name.split('_')[-1] for name in test_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, cpu_usages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 内存使用率
        bars2 = ax2.bar(x, memory_usages, alpha=0.7, color='blue', label='内存使用率')
        ax2.set_xlabel('测试')
        ax2.set_ylabel('内存使用率 (%)')
        ax2.set_title('内存使用率对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels([name.split('_')[-1] for name in test_names], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars2, memory_usages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(charts_dir / "resource_usage_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_error_rate_chart(self, test_results: List[TestResult], charts_dir: Path):
        """生成错误率图表"""
        plt.figure(figsize=(10, 6))
        
        test_names = [r.test_name for r in test_results]
        error_rates = [r.error_rate for r in test_results]
        
        x = range(len(test_names))
        colors = ['red' if rate > 10 else 'orange' if rate > 5 else 'green' for rate in error_rates]
        
        bars = plt.bar(x, error_rates, alpha=0.7, color=colors)
        plt.xlabel('测试')
        plt.ylabel('错误率 (%)')
        plt.title('系统错误率对比')
        plt.xticks(x, [name.split('_')[-1] for name in test_names], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, error_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(error_rates) * 0.01,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(charts_dir / "error_rate_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_html_report(self, test_results: List[TestResult], charts_dir: Path) -> str:
        """创建HTML报告"""
        
        # 计算总体统计
        total_tests = len(test_results)
        avg_response_time = statistics.mean([r.avg_response_time for r in test_results])
        avg_throughput = statistics.mean([r.throughput for r in test_results])
        avg_error_rate = statistics.mean([r.error_rate for r in test_results])
        avg_cpu = statistics.mean([r.cpu_usage for r in test_results])
        avg_memory = statistics.mean([r.memory_usage for r in test_results])
        
        # 生成测试结果表格
        table_rows = ""
        for i, result in enumerate(test_results, 1):
            table_rows += f"""
            <tr>
                <td>{i}</td>
                <td>{result.test_name}</td>
                <td>{result.avg_response_time:.3f}s</td>
                <td>{result.throughput:.2f} req/s</td>
                <td>{result.error_rate:.1f}%</td>
                <td>{result.cpu_usage:.1f}%</td>
                <td>{result.memory_usage:.1f}%</td>
                <td>{result.total_requests}</td>
            </tr>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>P3性能测试报告</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    border-bottom: 3px solid #007acc;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #007acc;
                    border-left: 4px solid #007acc;
                    padding-left: 15px;
                    margin-top: 30px;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .summary-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 1.2em;
                }}
                .summary-card .value {{
                    font-size: 2em;
                    font-weight: bold;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #007acc;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .chart {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .chart img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .timestamp {{
                    text-align: center;
                    color: #666;
                    margin-top: 30px;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>P3性能测试报告</h1>
                
                <h2>测试概览</h2>
                <div class="summary">
                    <div class="summary-card">
                        <h3>总测试数</h3>
                        <div class="value">{total_tests}</div>
                    </div>
                    <div class="summary-card">
                        <h3>平均响应时间</h3>
                        <div class="value">{avg_response_time:.3f}s</div>
                    </div>
                    <div class="summary-card">
                        <h3>平均吞吐量</h3>
                        <div class="value">{avg_throughput:.2f}</div>
                    </div>
                    <div class="summary-card">
                        <h3>平均错误率</h3>
                        <div class="value">{avg_error_rate:.1f}%</div>
                    </div>
                    <div class="summary-card">
                        <h3>平均CPU使用率</h3>
                        <div class="value">{avg_cpu:.1f}%</div>
                    </div>
                    <div class="summary-card">
                        <h3>平均内存使用率</h3>
                        <div class="value">{avg_memory:.1f}%</div>
                    </div>
                </div>
                
                <h2>详细测试结果</h2>
                <table>
                    <thead>
                        <tr>
                            <th>序号</th>
                            <th>测试名称</th>
                            <th>平均响应时间</th>
                            <th>吞吐量</th>
                            <th>错误率</th>
                            <th>CPU使用率</th>
                            <th>内存使用率</th>
                            <th>总请求数</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                
                <h2>性能图表</h2>
                <div class="chart">
                    <h3>响应时间分析</h3>
                    <img src="charts/response_time_chart.png" alt="响应时间图表">
                </div>
                
                <div class="chart">
                    <h3>吞吐量分析</h3>
                    <img src="charts/throughput_chart.png" alt="吞吐量图表">
                </div>
                
                <div class="chart">
                    <h3>资源使用分析</h3>
                    <img src="charts/resource_usage_chart.png" alt="资源使用图表">
                </div>
                
                <div class="chart">
                    <h3>错误率分析</h3>
                    <img src="charts/error_rate_chart.png" alt="错误率图表">
                </div>
                
                <div class="timestamp">
                    报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _save_test_result(self, result: TestResult):
        """保存测试结果到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO test_results (
                        test_name, start_time, end_time, duration, total_requests,
                        successful_requests, failed_requests, avg_response_time,
                        min_response_time, max_response_time, p95_response_time,
                        p99_response_time, throughput, error_rate, cpu_usage,
                        memory_usage, disk_usage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.test_name,
                    result.start_time.isoformat(),
                    result.end_time.isoformat(),
                    result.duration,
                    result.total_requests,
                    result.successful_requests,
                    result.failed_requests,
                    result.avg_response_time,
                    result.min_response_time,
                    result.max_response_time,
                    result.p95_response_time,
                    result.p99_response_time,
                    result.throughput,
                    result.error_rate,
                    result.cpu_usage,
                    result.memory_usage,
                    result.disk_usage
                ))
        except Exception as e:
            self.logger.warning(f"保存测试结果失败: {e}")
    
    def get_historical_data(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        获取历史测试数据
        
        Args:
            days_back: 回溯天数
            
        Returns:
            List[Dict[str, Any]]: 历史测试结果
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM test_results 
                WHERE created_at >= ?
                ORDER BY created_at DESC
            ''', (datetime.now() - timedelta(days=days_back),))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def export_results_to_json(self, test_results: List[TestResult], 
                              output_path: str = "test_results.json"):
        """
        导出测试结果到JSON文件
        
        Args:
            test_results: 测试结果列表
            output_path: 输出文件路径
        """
        results_data = {
            'export_time': datetime.now().isoformat(),
            'total_tests': len(test_results),
            'results': [result.to_dict() for result in test_results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"测试结果已导出到: {output_path}")


# 使用示例和工具函数
def create_sample_tester(base_url: str = "http://localhost:8000") -> PerformanceTester:
    """创建示例性能测试器"""
    return PerformanceTester(base_url=base_url)


def run_comprehensive_test(tester: PerformanceTester, endpoints: List[str]) -> Dict[str, Any]:
    """
    运行综合性能测试
    
    Args:
        tester: 性能测试器实例
        endpoints: 要测试的端点列表
        
    Returns:
        Dict[str, Any]: 综合测试结果
    """
    results = {}
    
    # 响应时间测试
    print("执行响应时间测试...")
    response_results = []
    for endpoint in endpoints:
        result = tester.response_time_test(endpoint, num_requests=50, concurrent_users=5)
        response_results.append(result)
    results['response_time_test'] = response_results
    
    # 吞吐量测试
    print("执行吞吐量测试...")
    throughput_results = []
    for endpoint in endpoints:
        result = tester.throughput_test(endpoint, duration=30, concurrent_users=10)
        throughput_results.append(result)
    results['throughput_test'] = throughput_results
    
    # 资源使用测试
    print("执行资源使用测试...")
    resource_results = []
    for endpoint in endpoints:
        result = tester.resource_usage_test(endpoint, duration=60)
        resource_results.append(result)
    results['resource_usage_test'] = resource_results
    
    # 性能瓶颈识别
    print("识别性能瓶颈...")
    all_results = response_results + throughput_results + resource_results
    bottleneck_analysis = tester.identify_bottlenecks(all_results)
    results['bottleneck_analysis'] = bottleneck_analysis
    
    # 生成报告
    print("生成性能报告...")
    report_path = tester.generate_performance_report(all_results, "comprehensive_performance_report.html")
    results['report_path'] = report_path
    
    return results


if __name__ == "__main__":
    # 示例用法
    tester = PerformanceTester("http://localhost:8000")
    
    # 测试端点列表
    endpoints = ["api/health", "api/users", "api/data"]
    
    # 运行综合测试
    comprehensive_results = run_comprehensive_test(tester, endpoints)
    
    print("综合性能测试完成!")
    print(f"报告路径: {comprehensive_results['report_path']}")