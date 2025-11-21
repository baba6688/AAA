"""
P9测试状态聚合器主要实现
=====================

提供完整的测试状态聚合、分析、监控和报告功能。
"""

import json
import time
import logging
import threading
import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import sqlite3
import os


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    WARNING = "warning"


class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


@dataclass
class TestResult:
    """测试结果数据结构"""
    test_id: str
    test_name: str
    module_name: str
    status: TestStatus
    duration: float
    timestamp: float
    message: str = ""
    error_details: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModuleStatus:
    """模块状态数据结构"""
    module_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    warning_tests: int
    success_rate: float
    last_update: float
    average_duration: float
    status: TestStatus

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        return data


@dataclass
class Alert:
    """预警信息数据结构"""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: float
    module_name: str = ""
    test_id: str = ""
    resolved: bool = False


class StatusCollector:
    """状态收集器"""
    
    def __init__(self):
        self._test_results: Dict[str, TestResult] = {}
        self._module_results: Dict[str, List[TestResult]] = defaultdict(list)
        self._listeners: List[Callable] = []
    
    def collect_test_result(self, result: TestResult) -> None:
        """收集测试结果"""
        self._test_results[result.test_id] = result
        self._module_results[result.module_name].append(result)
        
        # 通知监听器
        for listener in self._listeners:
            try:
                listener(result)
            except Exception as e:
                logging.error(f"监听器执行失败: {e}")
    
    def add_listener(self, listener: Callable) -> None:
        """添加状态变化监听器"""
        self._listeners.append(listener)
    
    def get_test_result(self, test_id: str) -> Optional[TestResult]:
        """获取测试结果"""
        return self._test_results.get(test_id)
    
    def get_module_results(self, module_name: str) -> List[TestResult]:
        """获取模块的所有测试结果"""
        return self._module_results.get(module_name, [])
    
    def get_all_results(self) -> Dict[str, TestResult]:
        """获取所有测试结果"""
        return self._test_results.copy()


class DataAggregator:
    """数据聚合器"""
    
    @staticmethod
    def aggregate_module_status(results: List[TestResult]) -> ModuleStatus:
        """聚合模块状态"""
        if not results:
            return ModuleStatus(
                module_name="",
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=0,
                warning_tests=0,
                success_rate=0.0,
                last_update=time.time(),
                average_duration=0.0,
                status=TestStatus.PENDING
            )
        
        module_name = results[0].module_name
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped_tests = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
        warning_tests = sum(1 for r in results if r.status == TestStatus.WARNING)
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0.0
        average_duration = sum(r.duration for r in results) / total_tests if total_tests > 0 else 0.0
        
        # 确定模块状态
        if error_tests > 0:
            status = TestStatus.ERROR
        elif failed_tests > 0:
            status = TestStatus.FAILED
        elif warning_tests > 0:
            status = TestStatus.WARNING
        elif passed_tests == total_tests:
            status = TestStatus.PASSED
        else:
            status = TestStatus.RUNNING
        
        return ModuleStatus(
            module_name=module_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            warning_tests=warning_tests,
            success_rate=success_rate,
            last_update=time.time(),
            average_duration=average_duration,
            status=status
        )
    
    @staticmethod
    def aggregate_all_modules(module_statuses: Dict[str, ModuleStatus]) -> Dict[str, Any]:
        """聚合所有模块状态"""
        if not module_statuses:
            return {
                "total_modules": 0,
                "total_tests": 0,
                "overall_success_rate": 0.0,
                "average_duration": 0.0,
                "status": TestStatus.PENDING.value
            }
        
        total_modules = len(module_statuses)
        total_tests = sum(ms.total_tests for ms in module_statuses.values())
        total_passed = sum(ms.passed_tests for ms in module_statuses.values())
        total_duration = sum(ms.average_duration * ms.total_tests for ms in module_statuses.values())
        
        overall_success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0.0
        average_duration = total_duration / total_tests if total_tests > 0 else 0.0
        
        # 确定整体状态
        all_passed = all(ms.status == TestStatus.PASSED for ms in module_statuses.values())
        any_failed = any(ms.status in [TestStatus.FAILED, TestStatus.ERROR] for ms in module_statuses.values())
        any_running = any(ms.status == TestStatus.RUNNING for ms in module_statuses.values())
        
        if any_failed:
            overall_status = TestStatus.FAILED.value
        elif any_running:
            overall_status = TestStatus.RUNNING.value
        elif all_passed:
            overall_status = TestStatus.PASSED.value
        else:
            overall_status = TestStatus.PENDING.value
        
        return {
            "total_modules": total_modules,
            "total_tests": total_tests,
            "overall_success_rate": overall_success_rate,
            "average_duration": average_duration,
            "status": overall_status,
            "module_statuses": {name: ms.to_dict() for name, ms in module_statuses.items()}
        }


class StatusAnalyzer:
    """状态分析器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._history: deque = deque(maxlen=history_size)
        self._trends: Dict[str, List[float]] = defaultdict(list)
    
    def analyze_trend(self, module_name: str, success_rates: List[float]) -> Dict[str, Any]:
        """分析趋势"""
        if len(success_rates) < 2:
            return {"trend": "insufficient_data", "change": 0.0}
        
        # 计算趋势
        recent_avg = sum(success_rates[-5:]) / min(5, len(success_rates))
        previous_avg = sum(success_rates[:-5]) / max(1, len(success_rates) - 5) if len(success_rates) > 5 else success_rates[0]
        
        change = recent_avg - previous_avg
        
        if change > 5:
            trend = "improving"
        elif change < -5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change": change,
            "recent_avg": recent_avg,
            "previous_avg": previous_avg
        }
    
    def detect_anomalies(self, module_statuses: Dict[str, ModuleStatus]) -> List[Alert]:
        """检测异常"""
        alerts = []
        
        for name, status in module_statuses.items():
            # 检测成功率异常
            if status.success_rate < 50.0 and status.total_tests > 0:
                alerts.append(Alert(
                    alert_id=f"low_success_rate_{name}_{int(time.time())}",
                    level=AlertLevel.WARNING,
                    message=f"模块 {name} 成功率较低: {status.success_rate:.1f}%",
                    timestamp=time.time(),
                    module_name=name
                ))
            
            # 检测执行时间异常
            if status.average_duration > 300.0:  # 5分钟
                alerts.append(Alert(
                    alert_id=f"slow_execution_{name}_{int(time.time())}",
                    level=AlertLevel.WARNING,
                    message=f"模块 {name} 执行时间较长: {status.average_duration:.1f}秒",
                    timestamp=time.time(),
                    module_name=name
                ))
            
            # 检测错误率
            if status.error_tests > 0:
                alerts.append(Alert(
                    alert_id=f"errors_{name}_{int(time.time())}",
                    level=AlertLevel.CRITICAL,
                    message=f"模块 {name} 存在错误: {status.error_tests} 个错误",
                    timestamp=time.time(),
                    module_name=name
                ))
        
        return alerts
    
    def add_to_history(self, data: Dict[str, Any]) -> None:
        """添加到历史记录"""
        self._history.append({
            "timestamp": time.time(),
            "data": data
        })
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return list(self._history)[-limit:]


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_summary_report(self, aggregated_data: Dict[str, Any]) -> str:
        """生成摘要报告"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_report_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 序列化 aggregated_data 以避免 TestStatus 枚举序列化问题
        # aggregated_data 中的 module_statuses 已经是字典格式，无需再次序列化
        serialized_details = dict(aggregated_data)
        
        report = {
            "report_type": "summary",
            "generated_at": time.time(),
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": {
                "total_modules": aggregated_data.get("total_modules", 0),
                "total_tests": aggregated_data.get("total_tests", 0),
                "overall_success_rate": aggregated_data.get("overall_success_rate", 0.0),
                "average_duration": aggregated_data.get("average_duration", 0.0),
                "overall_status": aggregated_data.get("status", "unknown")
            },
            "details": serialized_details
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_detailed_report(self, module_statuses: Dict[str, ModuleStatus], 
                                test_results: Dict[str, TestResult]) -> str:
        """生成详细报告"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_report_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        report = {
            "report_type": "detailed",
            "generated_at": time.time(),
            "timestamp": datetime.datetime.now().isoformat(),
            "modules": {name: ms.to_dict() for name, ms in module_statuses.items()},
            "test_results": {tid: self._serialize_test_result(result) for tid, result in test_results.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_html_dashboard(self, aggregated_data: Dict[str, Any], 
                               module_statuses: Dict[str, ModuleStatus]) -> str:
        """生成HTML仪表板"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        html_content = self._generate_dashboard_html(aggregated_data, module_statuses)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def _serialize_test_result(self, result: TestResult) -> Dict[str, Any]:
        """序列化测试结果"""
        data = asdict(result)
        data['status'] = result.status.value
        return data

    def _generate_dashboard_html(self, aggregated_data: Dict[str, Any], 
                               module_statuses: Dict[str, ModuleStatus]) -> str:
        """生成仪表板HTML内容"""
        summary = aggregated_data.get("summary", {})
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P9测试状态仪表板</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: white; padding: 20px; border-radius: 10px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        .summary-card h3 {{ margin: 0 0 10px 0; color: #333; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .modules {{ background: white; border-radius: 10px; padding: 20px; 
                   box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .module-item {{ border-bottom: 1px solid #eee; padding: 15px 0; }}
        .module-item:last-child {{ border-bottom: none; }}
        .status-badge {{ padding: 5px 10px; border-radius: 15px; color: white; 
                        font-size: 0.8em; margin-left: 10px; }}
        .status-passed {{ background-color: #28a745; }}
        .status-failed {{ background-color: #dc3545; }}
        .status-running {{ background-color: #17a2b8; }}
        .status-pending {{ background-color: #6c757d; }}
        .progress-bar {{ background-color: #e9ecef; border-radius: 10px; 
                        height: 20px; margin: 10px 0; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); 
                         transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>P9测试状态仪表板</h1>
            <p>生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>总模块数</h3>
                <div class="value">{summary.get('total_modules', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>总测试数</h3>
                <div class="value">{summary.get('total_tests', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>成功率</h3>
                <div class="value">{summary.get('overall_success_rate', 0):.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>平均耗时</h3>
                <div class="value">{summary.get('average_duration', 0):.1f}s</div>
            </div>
        </div>
        
        <div class="modules">
            <h2>模块状态详情</h2>
"""
        
        for name, status in module_statuses.items():
            status_class = f"status-{status.status.value}"
            html += f"""
            <div class="module-item">
                <h3>{name} 
                    <span class="status-badge {status_class}">{status.status.value.upper()}</span>
                </h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {status.success_rate}%"></div>
                </div>
                <p>成功率: {status.success_rate:.1f}% | 
                   通过: {status.passed_tests} | 
                   失败: {status.failed_tests} | 
                   跳过: {status.skipped_tests} | 
                   错误: {status.error_tests}</p>
                <p>平均耗时: {status.average_duration:.2f}s | 
                   最后更新: {datetime.datetime.fromtimestamp(status.last_update).strftime("%H:%M:%S")}</p>
            </div>
"""
        
        html += """
        </div>
    </div>
</body>
</html>
"""
        return html


class StatusMonitor:
    """状态监控器"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self._callbacks: List[Callable] = []
    
    def start_monitoring(self, callback: Callable) -> None:
        """开始监控"""
        if self.is_monitoring:
            return
        
        self._callbacks.append(callback)
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                for callback in self._callbacks:
                    callback()
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"监控循环错误: {e}")


class AlertManager:
    """预警管理器"""
    
    def __init__(self, storage_path: str = "alerts.db"):
        self.storage_path = storage_path
        self._alerts: Dict[str, Alert] = {}
        self._listeners: List[Callable] = []
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化数据库"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp REAL NOT NULL,
                module_name TEXT,
                test_id TEXT,
                resolved INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_alert(self, alert: Alert) -> None:
        """添加预警"""
        self._alerts[alert.alert_id] = alert
        
        # 保存到数据库
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO alerts 
            (alert_id, level, message, timestamp, module_name, test_id, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (alert.alert_id, alert.level.value, alert.message, alert.timestamp,
              alert.module_name, alert.test_id, 1 if alert.resolved else 0))
        conn.commit()
        conn.close()
        
        # 通知监听器
        for listener in self._listeners:
            try:
                listener(alert)
            except Exception as e:
                logging.error(f"预警监听器执行失败: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决预警"""
        if alert_id in self._alerts:
            self._alerts[alert_id].resolved = True
            
            # 更新数据库
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE alerts SET resolved = 1 WHERE alert_id = ?', (alert_id,))
            conn.commit()
            conn.close()
            
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        return [alert for alert in self._alerts.values() if not alert.resolved]
    
    def get_all_alerts(self) -> List[Alert]:
        """获取所有预警"""
        return list(self._alerts.values())
    
    def add_listener(self, listener: Callable) -> None:
        """添加预警监听器"""
        self._listeners.append(listener)


class HistoryManager:
    """历史记录管理器"""
    
    def __init__(self, storage_path: str = "test_history.db"):
        self.storage_path = storage_path
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化数据库"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                module_name TEXT NOT NULL,
                status TEXT NOT NULL,
                duration REAL NOT NULL,
                timestamp REAL NOT NULL,
                message TEXT,
                error_details TEXT,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_test_result(self, result: TestResult) -> TestResult:
        """保存测试结果"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO test_history 
            (test_id, test_name, module_name, status, duration, timestamp, message, error_details, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (result.test_id, result.test_name, result.module_name, result.status.value,
              result.duration, result.timestamp, result.message, result.error_details,
              json.dumps(result.metadata)))
        conn.commit()
        conn.close()
        return result
    
    def get_test_history(self, module_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取测试历史"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        if module_name:
            cursor.execute('''
                SELECT * FROM test_history 
                WHERE module_name = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (module_name, limit))
        else:
            cursor.execute('''
                SELECT * FROM test_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        results = []
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            if result['metadata']:
                result['metadata'] = json.loads(result['metadata'])
            results.append(result)
        
        conn.close()
        return results


class TestStatusAggregator:
    """测试状态聚合器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化各个组件
        self.collector = StatusCollector()
        self.aggregator = DataAggregator()
        self.analyzer = StatusAnalyzer()
        self.report_generator = ReportGenerator()
        self.monitor = StatusMonitor()
        self.alert_manager = AlertManager()
        self.history_manager = HistoryManager()
        
        # 设置监听器
        self.collector.add_listener(self._on_test_result_collected)
        self.alert_manager.add_listener(self._on_alert_generated)
        
        # 当前状态
        self._module_statuses: Dict[str, ModuleStatus] = {}
        self._is_running = False
    
    def start(self) -> None:
        """启动聚合器"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # 启动监控
        self.monitor.start_monitoring(self._periodic_check)
        
        logging.info("测试状态聚合器已启动")
    
    def stop(self) -> None:
        """停止聚合器"""
        if not self._is_running:
            return
        
        self._is_running = False
        self.monitor.stop_monitoring()
        
        logging.info("测试状态聚合器已停止")
    
    def submit_test_result(self, test_id: str, test_name: str, module_name: str,
                          status: TestStatus, duration: float, message: str = "",
                          error_details: str = "", metadata: Dict[str, Any] = None) -> None:
        """提交测试结果"""
        result = TestResult(
            test_id=test_id,
            test_name=test_name,
            module_name=module_name,
            status=status,
            duration=duration,
            timestamp=time.time(),
            message=message,
            error_details=error_details,
            metadata=metadata or {}
        )
        
        # 先收集测试结果
        self.collector.collect_test_result(result)
        
        # 然后保存到历史记录
        try:
            self.history_manager.save_test_result(result)
        except Exception as e:
            logging.error(f"保存测试结果到历史记录失败: {e}")
        
        self._update_module_status(module_name)
    
    def _update_module_status(self, module_name: str) -> None:
        """更新模块状态"""
        module_results = self.collector.get_module_results(module_name)
        self._module_statuses[module_name] = self.aggregator.aggregate_module_status(module_results)
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        aggregated_data = self.aggregator.aggregate_all_modules(self._module_statuses)
        return {
            "current_status": aggregated_data,
            "module_statuses": {name: ms.to_dict() for name, ms in self._module_statuses.items()},
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "is_running": self._is_running
        }
    
    def generate_reports(self) -> Dict[str, str]:
        """生成报告"""
        aggregated_data = self.aggregator.aggregate_all_modules(self._module_statuses)
        test_results = self.collector.get_all_results()
        
        reports = {}
        reports["summary"] = self.report_generator.generate_summary_report(aggregated_data)
        reports["detailed"] = self.report_generator.generate_detailed_report(
            self._module_statuses, test_results)
        reports["dashboard"] = self.report_generator.generate_html_dashboard(
            aggregated_data, self._module_statuses)
        
        return reports
    
    def get_trend_analysis(self, module_name: str = None) -> Dict[str, Any]:
        """获取趋势分析"""
        if module_name:
            history = self.history_manager.get_test_history(module_name, limit=50)
            success_rates = []
            # 这里应该根据实际历史数据计算成功率趋势
            # 简化实现
            return {"module": module_name, "trend": "stable", "data": success_rates}
        else:
            # 全局趋势分析
            return {"overall_trend": "stable", "modules": list(self._module_statuses.keys())}
    
    def _on_test_result_collected(self, result: TestResult) -> None:
        """测试结果收集回调"""
        # 检查是否需要生成预警
        if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
            alert = Alert(
                alert_id=f"test_failure_{result.test_id}_{int(time.time())}",
                level=AlertLevel.ERROR if result.status == TestStatus.ERROR else AlertLevel.WARNING,
                message=f"测试失败: {result.test_name} in {result.module_name}",
                timestamp=time.time(),
                module_name=result.module_name,
                test_id=result.test_id
            )
            self.alert_manager.add_alert(alert)
    
    def _on_alert_generated(self, alert: Alert) -> None:
        """预警生成回调"""
        logging.warning(f"预警 [{alert.level.value.upper()}]: {alert.message}")
    
    def _periodic_check(self) -> None:
        """定期检查"""
        # 检测异常
        anomalies = self.analyzer.detect_anomalies(self._module_statuses)
        for alert in anomalies:
            self.alert_manager.add_alert(alert)
        
        # 更新历史记录
        current_data = self.get_current_status()
        self.analyzer.add_to_history(current_data)
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


# 使用示例和工具函数
def create_sample_data(aggregator: TestStatusAggregator, num_modules: int = 3, 
                      tests_per_module: int = 10) -> None:
    """创建示例数据用于测试"""
    import random
    
    statuses = [TestStatus.PASSED, TestStatus.FAILED, TestStatus.SKIPPED, TestStatus.ERROR]
    
    for module_idx in range(num_modules):
        module_name = f"module_{module_idx + 1}"
        
        for test_idx in range(tests_per_module):
            test_id = f"{module_name}_test_{test_idx + 1}"
            test_name = f"测试用例 {test_idx + 1}"
            status = random.choice(statuses)
            duration = random.uniform(0.1, 5.0)
            message = "测试执行完成" if status == TestStatus.PASSED else "测试失败"
            
            aggregator.submit_test_result(
                test_id=test_id,
                test_name=test_name,
                module_name=module_name,
                status=status,
                duration=duration,
                message=message
            )


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)
    
    with TestStatusAggregator() as aggregator:
        # 创建示例数据
        create_sample_data(aggregator, num_modules=3, tests_per_module=15)
        
        # 获取当前状态
        status = aggregator.get_current_status()
        print("当前状态:", json.dumps(status, indent=2, ensure_ascii=False))
        
        # 生成报告
        reports = aggregator.generate_reports()
        print("生成的报告:", reports)
        
        # 等待一段时间观察监控
        time.sleep(2)
        
        # 获取活跃预警
        alerts = aggregator.alert_manager.get_active_alerts()
        print("活跃预警:", len(alerts))
        
        for alert in alerts:
            print(f"- [{alert.level.value.upper()}] {alert.message}")