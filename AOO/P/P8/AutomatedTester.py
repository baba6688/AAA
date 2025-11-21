"""
P8自动化测试器 - 主要实现模块

提供完整的自动化测试解决方案，包括测试计划管理、用例库、执行调度、
环境管理、数据管理、结果分析和报告生成等功能。
"""

import json
import time
import uuid
import logging
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import os
import yaml


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TestPriority(Enum):
    """测试优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TestCase:
    """测试用例数据结构"""
    id: str
    name: str
    description: str
    test_function: Callable
    priority: TestPriority
    tags: List[str]
    environment_requirements: List[str]
    data_requirements: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5分钟超时
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['test_function'] = self.test_function.__name__ if self.test_function else None
        return data


@dataclass
class TestResult:
    """测试结果数据结构"""
    test_case_id: str
    test_case_name: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    error_message: Optional[str]
    logs: List[str]
    environment: str
    retry_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['status'] = self.status.value
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


@dataclass
class TestPlan:
    """测试计划数据结构"""
    id: str
    name: str
    description: str
    test_case_ids: List[str]
    environment: str
    schedule: Optional[str]  # cron表达式
    enabled: bool = True
    created_time: datetime = None
    last_run_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()


class TestEnvironment:
    """测试环境管理"""
    
    def __init__(self, config_file: str = "test_environments.yaml"):
        self.config_file = config_file
        self.environments = self._load_environments()
    
    def _load_environments(self) -> Dict[str, Dict[str, Any]]:
        """加载环境配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logging.warning(f"加载环境配置失败: {e}")
        
        # 默认环境配置
        return {
            "dev": {
                "base_url": "http://localhost:3000",
                "database": "test_dev",
                "timeout": 30,
                "variables": {"env": "development"}
            },
            "staging": {
                "base_url": "https://staging.example.com",
                "database": "test_staging",
                "timeout": 60,
                "variables": {"env": "staging"}
            },
            "prod": {
                "base_url": "https://prod.example.com",
                "database": "test_prod",
                "timeout": 120,
                "variables": {"env": "production"}
            }
        }
    
    def get_environment_config(self, name: str) -> Dict[str, Any]:
        """获取环境配置"""
        return self.environments.get(name, {})
    
    def add_environment(self, name: str, config: Dict[str, Any]):
        """添加环境配置"""
        self.environments[name] = config
        self._save_environments()
    
    def _save_environments(self):
        """保存环境配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.environments, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logging.error(f"保存环境配置失败: {e}")


class TestDataManager:
    """测试数据管理"""
    
    def __init__(self, data_dir: str = "test_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.data_cache = {}
    
    def generate_test_data(self, test_case_id: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试数据"""
        if test_case_id in self.data_cache:
            return self.data_cache[test_case_id]
        
        # 简单的数据生成逻辑
        generated_data = {}
        for key, spec in requirements.items():
            if isinstance(spec, dict):
                if spec.get('type') == 'random_string':
                    generated_data[key] = self._generate_random_string(spec.get('length', 10))
                elif spec.get('type') == 'random_number':
                    generated_data[key] = self._generate_random_number(
                        spec.get('min', 1), spec.get('max', 1000)
                    )
                elif spec.get('type') == 'random_email':
                    generated_data[key] = self._generate_random_email()
                elif spec.get('type') == 'timestamp':
                    generated_data[key] = datetime.now().isoformat()
                else:
                    generated_data[key] = spec.get('default', None)
            else:
                generated_data[key] = spec
        
        self.data_cache[test_case_id] = generated_data
        return generated_data
    
    def _generate_random_string(self, length: int) -> str:
        """生成随机字符串"""
        import random
        import string
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def _generate_random_number(self, min_val: int, max_val: int) -> int:
        """生成随机数字"""
        import random
        return random.randint(min_val, max_val)
    
    def _generate_random_email(self) -> str:
        """生成随机邮箱"""
        import random
        import string
        domain = ['example.com', 'test.com', 'demo.com']
        username = ''.join(random.choices(string.ascii_lowercase, k=8))
        return f"{username}@{random.choice(domain)}"
    
    def load_test_data(self, file_path: str) -> Dict[str, Any]:
        """从文件加载测试数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"加载测试数据文件失败 {file_path}: {e}")
        return {}
    
    def save_test_data(self, data: Dict[str, Any], file_path: str):
        """保存测试数据到文件"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    json.dump(data, f, ensure_ascii=False, indent=2)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logging.error(f"保存测试数据文件失败 {file_path}: {e}")


class TestReporter:
    """测试报告生成器"""
    
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, test_results: List[TestResult], test_plan: TestPlan) -> str:
        """生成测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"test_report_{timestamp}.html")
        
        # 统计测试结果
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in test_results if r.status == TestStatus.FAILED])
        skipped_tests = len([r for r in test_results if r.status == TestStatus.SKIPPED])
        
        # 计算执行时间
        total_duration = sum(r.duration or 0 for r in test_results)
        
        # 生成HTML报告
        html_content = self._generate_html_report(
            test_results, test_plan, total_tests, passed_tests, failed_tests, 
            skipped_tests, total_duration
        )
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logging.info(f"测试报告已生成: {report_file}")
            return report_file
        except Exception as e:
            logging.error(f"生成测试报告失败: {e}")
            return ""
    
    def _generate_html_report(self, test_results: List[TestResult], test_plan: TestPlan,
                            total_tests: int, passed_tests: int, failed_tests: int,
                            skipped_tests: int, total_duration: float) -> str:
        """生成HTML报告内容"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P8自动化测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 10px; border-radius: 5px; }}
        .stat.passed {{ background-color: #d4edda; color: #155724; }}
        .stat.failed {{ background-color: #f8d7da; color: #721c24; }}
        .stat.skipped {{ background-color: #fff3cd; color: #856404; }}
        .test-details {{ margin-top: 20px; }}
        .test-case {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .test-case.passed {{ border-left: 5px solid #28a745; }}
        .test-case.failed {{ border-left: 5px solid #dc3545; }}
        .test-case.skipped {{ border-left: 5px solid #ffc107; }}
        .logs {{ background-color: #f8f9fa; padding: 10px; margin-top: 10px; border-radius: 3px; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>P8自动化测试报告</h1>
        <p><strong>测试计划:</strong> {test_plan.name}</p>
        <p><strong>执行时间:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>测试环境:</strong> {test_plan.environment}</p>
    </div>
    
    <div class="summary">
        <div class="stat passed">
            <h3>{passed_tests}</h3>
            <p>通过</p>
        </div>
        <div class="stat failed">
            <h3>{failed_tests}</h3>
            <p>失败</p>
        </div>
        <div class="stat skipped">
            <h3>{skipped_tests}</h3>
            <p>跳过</p>
        </div>
        <div class="stat">
            <h3>{total_duration:.2f}s</h3>
            <p>总耗时</p>
        </div>
    </div>
    
    <div class="test-details">
        <h2>测试详情</h2>
"""
        
        for result in test_results:
            status_class = result.status.value
            html += f"""
        <div class="test-case {status_class}">
            <h3>{result.test_case_name}</h3>
            <p><strong>状态:</strong> {result.status.value.upper()}</p>
            <p><strong>耗时:</strong> {result.duration or 0:.2f}s</p>
            <p><strong>环境:</strong> {result.environment}</p>
            <p><strong>重试次数:</strong> {result.retry_count}</p>
            {f'<p><strong>错误信息:</strong> {result.error_message}</p>' if result.error_message else ''}
            {f'<div class="logs"><strong>日志:</strong><br>{"<br>".join(result.logs)}</div>' if result.logs else ''}
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        return html


class AutomatedTester:
    """P8自动化测试器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.test_cases: Dict[str, TestCase] = {}
        self.test_plans: Dict[str, TestPlan] = {}
        self.test_results: List[TestResult] = []
        
        # 初始化各个组件
        self.environment_manager = TestEnvironment()
        self.data_manager = TestDataManager()
        self.reporter = TestReporter()
        
        # 调度器
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = self.config.get('log_level', 'INFO')
        log_file = self.config.get('log_file', 'automated_tester.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def register_test_case(self, test_case: TestCase):
        """注册测试用例"""
        self.test_cases[test_case.id] = test_case
        logging.info(f"注册测试用例: {test_case.name} ({test_case.id})")
    
    def create_test_plan(self, name: str, description: str, test_case_ids: List[str],
                        environment: str, schedule: str = None) -> str:
        """创建测试计划"""
        plan_id = str(uuid.uuid4())
        test_plan = TestPlan(
            id=plan_id,
            name=name,
            description=description,
            test_case_ids=test_case_ids,
            environment=environment,
            schedule=schedule
        )
        self.test_plans[plan_id] = test_plan
        logging.info(f"创建测试计划: {name} ({plan_id})")
        return plan_id
    
    def execute_test_case(self, test_case: TestCase, environment: str) -> TestResult:
        """执行单个测试用例"""
        result = TestResult(
            test_case_id=test_case.id,
            test_case_name=test_case.name,
            status=TestStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            error_message=None,
            logs=[],
            environment=environment,
            retry_count=0,
            metadata={}
        )
        
        # 检查测试用例是否启用
        if not test_case.enabled:
            result.status = TestStatus.SKIPPED
            result.logs.append("测试用例已禁用")
            return result
        
        # 检查环境要求
        env_config = self.environment_manager.get_environment_config(environment)
        if not env_config:
            result.status = TestStatus.FAILED
            result.error_message = f"环境配置不存在: {environment}"
            return result
        
        # 生成测试数据
        test_data = self.data_manager.generate_test_data(
            test_case.id, test_case.data_requirements
        )
        
        # 执行测试
        try:
            result.status = TestStatus.RUNNING
            result.logs.append(f"开始执行测试用例: {test_case.name}")
            
            # 执行测试函数
            start_time = time.time()
            test_function_result = test_case.test_function(
                test_data, env_config, result.logs
            )
            end_time = time.time()
            
            result.duration = end_time - start_time
            
            if test_function_result:
                result.status = TestStatus.PASSED
                result.logs.append("测试用例执行成功")
            else:
                result.status = TestStatus.FAILED
                result.error_message = "测试用例返回失败结果"
                result.logs.append("测试用例返回失败结果")
                
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            result.logs.append(f"测试用例执行异常: {e}")
            logging.error(f"测试用例执行失败 {test_case.name}: {e}")
        
        result.end_time = datetime.now()
        return result
    
    def execute_test_plan(self, plan_id: str, force_run: bool = False) -> List[TestResult]:
        """执行测试计划"""
        if plan_id not in self.test_plans:
            raise ValueError(f"测试计划不存在: {plan_id}")
        
        test_plan = self.test_plans[plan_id]
        
        # 检查调度设置
        if test_plan.schedule and not force_run:
            # 这里可以添加调度检查逻辑
            pass
        
        logging.info(f"开始执行测试计划: {test_plan.name}")
        test_plan.last_run_time = datetime.now()
        
        results = []
        
        # 按优先级排序测试用例
        sorted_test_cases = sorted(
            [self.test_cases[tid] for tid in test_plan.test_case_ids if tid in self.test_cases],
            key=lambda x: x.priority.value,
            reverse=True
        )
        
        for test_case in sorted_test_cases:
            result = self.execute_test_case(test_case, test_plan.environment)
            results.append(result)
            self.test_results.append(result)
            
            # 失败重试机制 - 持续重试直到成功或达到最大重试次数
            while (result.status == TestStatus.FAILED and 
                   test_case.retry_count < test_case.max_retries):
                result = self._retry_test_case(test_case, test_plan.environment, result)
                results.append(result)
                self.test_results.append(result)
        
        # 生成测试报告
        report_file = self.reporter.generate_report(results, test_plan)
        logging.info(f"测试计划执行完成，报告生成: {report_file}")
        
        return results
    
    def _retry_test_case(self, test_case: TestCase, environment: str, previous_result: TestResult) -> TestResult:
        """重试失败的测试用例"""
        test_case.retry_count += 1
        logging.info(f"重试测试用例: {test_case.name} (第{test_case.retry_count}次)")
        
        result = self.execute_test_case(test_case, environment)
        result.retry_count = test_case.retry_count
        result.logs.insert(0, f"重试执行 (第{test_case.retry_count}次)")
        
        return result
    
    def schedule_test_plan(self, plan_id: str, cron_expression: str):
        """调度测试计划"""
        if plan_id not in self.test_plans:
            raise ValueError(f"测试计划不存在: {plan_id}")
        
        test_plan = self.test_plans[plan_id]
        test_plan.schedule = cron_expression
        
        # 使用schedule库设置定时任务
        try:
            schedule.every().cron(cron_expression).do(self.execute_test_plan, plan_id, True)
            logging.info(f"已设置测试计划定时执行: {test_plan.name} ({cron_expression})")
        except Exception as e:
            logging.error(f"设置定时任务失败: {e}")
            raise
    
    def start_scheduler(self):
        """启动调度器"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logging.info("测试调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logging.info("测试调度器已停止")
    
    def _run_scheduler(self):
        """运行调度器"""
        while self.scheduler_running:
            schedule.run_pending()
            time.sleep(1)
    
    def get_test_results(self, plan_id: str = None) -> List[TestResult]:
        """获取测试结果"""
        if plan_id:
            return [r for r in self.test_results if r.test_case_id in 
                   self.test_plans[plan_id].test_case_ids]
        return self.test_results
    
    def export_results(self, file_path: str, format: str = 'json'):
        """导出测试结果"""
        results_data = [result.to_dict() for result in self.test_results]
        
        try:
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'yaml':
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(results_data, f, default_flow_style=False, allow_unicode=True)
            
            logging.info(f"测试结果已导出到: {file_path}")
        except Exception as e:
            logging.error(f"导出测试结果失败: {e}")
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果"""
        if not self.test_results:
            return {"error": "没有测试结果可分析"}
        
        total = len(self.test_results)
        passed = len([r for r in self.test_results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.test_results if r.status == TestStatus.FAILED])
        skipped = len([r for r in self.test_results if r.status == TestStatus.SKIPPED])
        
        # 计算成功率
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        # 分析失败原因
        failure_analysis = {}
        for result in self.test_results:
            if result.status == TestStatus.FAILED:
                error_key = result.error_message or "未知错误"
                failure_analysis[error_key] = failure_analysis.get(error_key, 0) + 1
        
        # 环境分析
        environment_analysis = {}
        for result in self.test_results:
            env = result.environment
            if env not in environment_analysis:
                environment_analysis[env] = {"total": 0, "passed": 0, "failed": 0}
            
            environment_analysis[env]["total"] += 1
            if result.status == TestStatus.PASSED:
                environment_analysis[env]["passed"] += 1
            elif result.status == TestStatus.FAILED:
                environment_analysis[env]["failed"] += 1
        
        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "skipped_tests": skipped,
            "success_rate": success_rate,
            "failure_analysis": failure_analysis,
            "environment_analysis": environment_analysis,
            "analysis_time": datetime.now().isoformat()
        }


# 装饰器函数，用于简化测试用例注册
def test_case(id: str, name: str, description: str = "", priority: TestPriority = TestPriority.MEDIUM,
              tags: List[str] = None, environment_requirements: List[str] = None,
              data_requirements: Dict[str, Any] = None, max_retries: int = 3, timeout: int = 300):
    """测试用例装饰器"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper.__test_case_metadata__ = {
            'id': id,
            'name': name,
            'description': description,
            'priority': priority,
            'tags': tags or [],
            'environment_requirements': environment_requirements or [],
            'data_requirements': data_requirements or {},
            'max_retries': max_retries,
            'timeout': timeout,
            'test_function': func
        }
        return wrapper
    return decorator


def create_test_case_from_decorator(decorated_func) -> TestCase:
    """从装饰器函数创建测试用例"""
    metadata = decorated_func.__test_case_metadata__
    return TestCase(
        id=metadata['id'],
        name=metadata['name'],
        description=metadata['description'],
        test_function=metadata['test_function'],
        priority=metadata['priority'],
        tags=metadata['tags'],
        environment_requirements=metadata['environment_requirements'],
        data_requirements=metadata['data_requirements'],
        max_retries=metadata['max_retries'],
        timeout=metadata['timeout']
    )