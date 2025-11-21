"""
P1单元测试器 - 完整的单元测试框架
提供测试用例管理、断言方法、测试执行、结果分析等功能
"""

import unittest
import time
import traceback
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import inspect


class TestResult(Enum):
    """测试结果枚举"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestCase:
    """测试用例类"""
    name: str
    description: str = ""
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    timeout: float = 30.0
    tags: List[str] = field(default_factory=list)
    _test_func: Optional[Callable] = None
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("测试用例名称不能为空")
    
    def set_test_function(self, func: Callable):
        """设置测试函数"""
        self._test_func = func
    
    def get_test_function(self) -> Optional[Callable]:
        """获取测试函数"""
        return self._test_func


@dataclass
class TestResultData:
    """测试结果数据类"""
    test_case: TestCase
    result: TestResult
    start_time: float
    end_time: float
    execution_time: float
    message: str = ""
    error_traceback: str = ""
    assertions_count: int = 0
    
    @property
    def is_passed(self) -> bool:
        return self.result == TestResult.PASSED
    
    @property
    def is_failed(self) -> bool:
        return self.result in [TestResult.FAILED, TestResult.ERROR]


class AssertionLibrary:
    """断言方法库"""
    
    @staticmethod
    def assert_equal(actual: Any, expected: Any, message: str = "") -> bool:
        """断言两个值相等"""
        if actual != expected:
            raise AssertionError(
                f"{message}期望值: {expected}, 实际值: {actual}"
            )
        return True
    
    @staticmethod
    def assert_not_equal(actual: Any, expected: Any, message: str = "") -> bool:
        """断言两个值不相等"""
        if actual == expected:
            raise AssertionError(
                f"{message}值应该不相等: {actual}"
            )
        return True
    
    @staticmethod
    def assert_true(condition: Any, message: str = "") -> bool:
        """断言条件为真"""
        if not bool(condition):
            raise AssertionError(f"{message}条件应为真: {condition}")
        return True
    
    @staticmethod
    def assert_false(condition: Any, message: str = "") -> bool:
        """断言条件为假"""
        if bool(condition):
            raise AssertionError(f"{message}条件应为假: {condition}")
        return True
    
    @staticmethod
    def assert_is_instance(obj: Any, cls: type, message: str = "") -> bool:
        """断言对象是指定类型的实例"""
        if not isinstance(obj, cls):
            raise AssertionError(
                f"{message}对象 {obj} 应该是 {cls} 的实例，实际类型: {type(obj)}"
            )
        return True
    
    @staticmethod
    def assert_is_none(obj: Any, message: str = "") -> bool:
        """断言对象为None"""
        if obj is not None:
            raise AssertionError(f"{message}对象应该为None: {obj}")
        return True
    
    @staticmethod
    def assert_is_not_none(obj: Any, message: str = "") -> bool:
        """断言对象不为None"""
        if obj is None:
            raise AssertionError(f"{message}对象不应该为None")
        return True
    
    @staticmethod
    def assert_in(item: Any, container: Any, message: str = "") -> bool:
        """断言项目包含在容器中"""
        if item not in container:
            raise AssertionError(
                f"{message}项目 {item} 应该包含在 {container} 中"
            )
        return True
    
    @staticmethod
    def assert_not_in(item: Any, container: Any, message: str = "") -> bool:
        """断言项目不包含在容器中"""
        if item in container:
            raise AssertionError(
                f"{message}项目 {item} 不应该包含在 {container} 中"
            )
        return True
    
    @staticmethod
    def assert_raises(exception_type: type, func: Callable, *args, **kwargs) -> bool:
        """断言函数抛出指定异常"""
        try:
            func(*args, **kwargs)
            raise AssertionError(
                f"函数 {func.__name__} 应该抛出 {exception_type.__name__} 异常"
            )
        except exception_type:
            return True
        except Exception as e:
            raise AssertionError(
                f"函数 {func.__name__} 抛出了错误的异常类型: "
                f"期望 {exception_type.__name__}, 实际 {type(e).__name__}"
            )
    
    @staticmethod
    def assert_greater(a: Any, b: Any, message: str = "") -> bool:
        """断言a > b"""
        if not (a > b):
            raise AssertionError(f"{message}期望 {a} > {b}")
        return True
    
    @staticmethod
    def assert_less(a: Any, b: Any, message: str = "") -> bool:
        """断言a < b"""
        if not (a < b):
            raise AssertionError(f"{message}期望 {a} < {b}")
        return True
    
    @staticmethod
    def assert_greater_equal(a: Any, b: Any, message: str = "") -> bool:
        """断言a >= b"""
        if not (a >= b):
            raise AssertionError(f"{message}期望 {a} >= {b}")
        return True
    
    @staticmethod
    def assert_less_equal(a: Any, b: Any, message: str = "") -> bool:
        """断言a <= b"""
        if not (a <= b):
            raise AssertionError(f"{message}期望 {a} <= {b}")
        return True


class MockObject:
    """Mock对象类，用于模拟依赖对象"""
    
    def __init__(self, name: str = "Mock"):
        self.name = name
        self._methods: Dict[str, Callable] = {}
        self._attributes: Dict[str, Any] = {}
        self._call_history: List[Dict[str, Any]] = []
    
    def __getattr__(self, name: str):
        if name.startswith('_') or name in ['name']:
            return super().__getattr__(name)
        
        # 检查是否是已设置的属性
        if name in self._attributes:
            return self._attributes[name]
        
        if name not in self._methods:
            # 创建一个新的mock方法
            def mock_method(*args, **kwargs):
                # 记录调用历史
                self._call_history.append({
                    'method': name,
                    'args': args,
                    'kwargs': kwargs,
                    'timestamp': time.time()
                })
                return self
            
            self._methods[name] = mock_method
        
        return self._methods[name]
    
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_') or name in ['name']:
            super().__setattr__(name, value)
        else:
            self._attributes[name] = value
    
    def __repr__(self):
        return f"<Mock {self.name}>"
    
    def reset(self):
        """重置mock对象"""
        self._methods.clear()
        self._attributes.clear()
        self._call_history.clear()
    
    def assert_called_with(self, method_name: str, *args, **kwargs):
        """断言方法被调用时使用了指定的参数"""
        if not self._call_history:
            raise AssertionError(f"方法 {method_name} 未被调用")
        
        # 查找最后一次调用
        for call in reversed(self._call_history):
            if call['method'] == method_name:
                if call['args'] == args and call['kwargs'] == kwargs:
                    return True
        
        raise AssertionError(
            f"方法 {method_name} 未使用指定参数调用。\n"
            f"期望: args={args}, kwargs={kwargs}\n"
            f"实际调用历史: {self._call_history}"
        )
    
    def get_call_count(self, method_name: str) -> int:
        """获取方法被调用的次数"""
        return sum(1 for call in self._call_history if call['method'] == method_name)
    
    def get_call_history(self, method_name: str = None) -> List[Dict[str, Any]]:
        """获取调用历史"""
        if method_name:
            return [call for call in self._call_history if call['method'] == method_name]
        return self._call_history.copy()


class TestSuite:
    """测试套件类，用于组织测试用例"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.test_cases: List[TestCase] = []
        self.sub_suites: List['TestSuite'] = []
    
    def add_test_case(self, test_case: TestCase):
        """添加测试用例"""
        self.test_cases.append(test_case)
    
    def add_test_suite(self, test_suite: 'TestSuite'):
        """添加子测试套件"""
        self.sub_suites.append(test_suite)
    
    def remove_test_case(self, test_case: TestCase):
        """移除测试用例"""
        if test_case in self.test_cases:
            self.test_cases.remove(test_case)
    
    def get_all_test_cases(self) -> List[TestCase]:
        """获取所有测试用例（包括子套件中的）"""
        cases = self.test_cases.copy()
        for suite in self.sub_suites:
            cases.extend(suite.get_all_test_cases())
        return cases
    
    def filter_by_tags(self, tags: List[str]) -> List[TestCase]:
        """根据标签过滤测试用例"""
        return [case for case in self.get_all_test_cases() 
                if any(tag in case.tags for tag in tags)]


class TestRunner:
    """测试执行器"""
    
    def __init__(self):
        self.assertions = AssertionLibrary()
        self.results: List[TestResultData] = []
        self.current_test_case: Optional[TestCase] = None
    
    def run_single_test(self, test_case: TestCase) -> TestResultData:
        """执行单个测试用例"""
        self.current_test_case = test_case
        start_time = time.time()
        
        # 检查测试函数是否存在
        test_func = test_case.get_test_function()
        if not test_func:
            return TestResultData(
                test_case=test_case,
                result=TestResult.ERROR,
                start_time=start_time,
                end_time=time.time(),
                execution_time=time.time() - start_time,
                message=f"测试用例 {test_case.name} 没有设置测试函数"
            )
        
        try:
            # 执行setup
            if test_case.setup_func:
                test_case.setup_func()
            
            # 执行测试函数
            test_func()
            
            # 执行teardown
            if test_case.teardown_func:
                test_case.teardown_func()
            
            end_time = time.time()
            return TestResultData(
                test_case=test_case,
                result=TestResult.PASSED,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                message="测试通过"
            )
            
        except AssertionError as e:
            end_time = time.time()
            return TestResultData(
                test_case=test_case,
                result=TestResult.FAILED,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                message=str(e),
                error_traceback=traceback.format_exc()
            )
        except Exception as e:
            end_time = time.time()
            return TestResultData(
                test_case=test_case,
                result=TestResult.ERROR,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                message=f"测试执行错误: {str(e)}",
                error_traceback=traceback.format_exc()
            )
    
    def run_test_suite(self, test_suite: TestSuite, 
                      filter_tags: List[str] = None) -> List[TestResultData]:
        """执行测试套件"""
        results = []
        
        # 获取要执行的测试用例
        if filter_tags:
            test_cases = test_suite.filter_by_tags(filter_tags)
        else:
            test_cases = test_suite.get_all_test_cases()
        
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def run_multiple_tests(self, test_cases: List[TestCase]) -> List[TestResultData]:
        """批量执行测试用例"""
        results = []
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def run_tests_with_timeout(self, test_cases: List[TestCase], 
                             timeout: float = 30.0) -> List[TestResultData]:
        """带超时的批量测试执行"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"测试执行超时 ({timeout}秒)")
        
        # 设置信号处理器（仅在Unix系统上有效）
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
        
        try:
            return self.run_multiple_tests(test_cases)
        finally:
            # 恢复原始信号处理器
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


class TestReporter:
    """测试报告生成器"""
    
    @staticmethod
    def generate_text_report(results: List[TestResultData]) -> str:
        """生成文本格式的测试报告"""
        if not results:
            return "没有执行任何测试"
        
        # 统计信息
        total = len(results)
        passed = sum(1 for r in results if r.is_passed)
        failed = sum(1 for r in results if r.result == TestResult.FAILED)
        errors = sum(1 for r in results if r.result == TestResult.ERROR)
        skipped = sum(1 for r in results if r.result == TestResult.SKIPPED)
        
        total_time = sum(r.execution_time for r in results)
        
        report = []
        report.append("=" * 60)
        report.append("P1单元测试器 - 测试报告")
        report.append("=" * 60)
        report.append(f"总测试数: {total}")
        report.append(f"通过: {passed}")
        report.append(f"失败: {failed}")
        report.append(f"错误: {errors}")
        report.append(f"跳过: {skipped}")
        report.append(f"通过率: {passed/total*100:.1f}%")
        report.append(f"总执行时间: {total_time:.3f}秒")
        report.append("-" * 60)
        
        # 详细结果
        for result in results:
            status_symbol = "✓" if result.is_passed else "✗"
            report.append(f"{status_symbol} {result.test_case.name}")
            report.append(f"  执行时间: {result.execution_time:.3f}秒")
            report.append(f"  结果: {result.result.value}")
            if result.message:
                report.append(f"  消息: {result.message}")
            if result.error_traceback:
                report.append(f"  错误追踪:")
                for line in result.error_traceback.split('\n')[:5]:  # 只显示前5行
                    if line.strip():
                        report.append(f"    {line}")
            report.append("")
        
        return '\n'.join(report)
    
    @staticmethod
    def generate_json_report(results: List[TestResultData]) -> str:
        """生成JSON格式的测试报告"""
        report_data = {
            'summary': {
                'total': len(results),
                'passed': sum(1 for r in results if r.is_passed),
                'failed': sum(1 for r in results if r.result == TestResult.FAILED),
                'errors': sum(1 for r in results if r.result == TestResult.ERROR),
                'skipped': sum(1 for r in results if r.result == TestResult.SKIPPED),
                'total_time': sum(r.execution_time for r in results)
            },
            'details': []
        }
        
        for result in results:
            detail = {
                'test_name': result.test_case.name,
                'result': result.result.value,
                'execution_time': result.execution_time,
                'message': result.message,
                'error_traceback': result.error_traceback,
                'tags': result.test_case.tags
            }
            report_data['details'].append(detail)
        
        return json.dumps(report_data, ensure_ascii=False, indent=2)
    
    @staticmethod
    def save_report_to_file(results: List[TestResultData], 
                          filename: str, format: str = 'text'):
        """保存报告到文件"""
        if format.lower() == 'json':
            content = TestReporter.generate_json_report(results)
        else:
            content = TestReporter.generate_text_report(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    @staticmethod
    def print_summary(results: List[TestResultData]):
        """打印测试摘要"""
        if not results:
            print("没有执行任何测试")
            return
        
        total = len(results)
        passed = sum(1 for r in results if r.is_passed)
        failed = sum(1 for r in results if r.result == TestResult.FAILED)
        errors = sum(1 for r in results if r.result == TestResult.ERROR)
        
        print(f"\n测试摘要:")
        print(f"总测试数: {total}")
        print(f"通过: {passed} ({passed/total*100:.1f}%)")
        print(f"失败: {failed}")
        print(f"错误: {errors}")
        
        if failed > 0 or errors > 0:
            print("\n失败的测试:")
            for result in results:
                if not result.is_passed:
                    print(f"  - {result.test_case.name}: {result.message}")


class TestAnalyzer:
    """测试结果分析器"""
    
    @staticmethod
    def calculate_pass_rate(results: List[TestResultData]) -> float:
        """计算通过率"""
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.is_passed)
        return passed / len(results) * 100
    
    @staticmethod
    def get_failure_analysis(results: List[TestResultData]) -> Dict[str, Any]:
        """分析失败原因"""
        failures = [r for r in results if not r.is_passed]
        
        analysis = {
            "total_failures": len(failures),
            "failure_types": {},
            "common_errors": {},
            "slow_tests": []
        }
        
        if not failures:
            analysis["message"] = "所有测试都通过了"
            return analysis
        
        for result in failures:
            # 统计失败类型
            failure_type = result.result.value
            analysis["failure_types"][failure_type] = \
                analysis["failure_types"].get(failure_type, 0) + 1
            
            # 统计常见错误
            if result.message:
                error_key = result.message.split(':')[0] if ':' in result.message else result.message
                analysis["common_errors"][error_key] = \
                    analysis["common_errors"].get(error_key, 0) + 1
            
            # 找出慢测试
            if result.execution_time > 5.0:  # 超过5秒的测试
                analysis["slow_tests"].append({
                    "name": result.test_case.name,
                    "time": result.execution_time
                })
        
        return analysis
    
    @staticmethod
    def get_performance_metrics(results: List[TestResultData]) -> Dict[str, float]:
        """获取性能指标"""
        if not results:
            return {}
        
        execution_times = [r.execution_time for r in results]
        
        return {
            "total_time": sum(execution_times),
            "average_time": sum(execution_times) / len(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "median_time": sorted(execution_times)[len(execution_times)//2]
        }


class UnitTester:
    """P1单元测试器主类"""
    
    def __init__(self):
        self.test_cases: Dict[str, TestCase] = {}
        self.test_suites: Dict[str, TestSuite] = {}
        self.runner = TestRunner()
        self.reporter = TestReporter()
        self.analyzer = TestAnalyzer()
        self.assertions = AssertionLibrary()
    
    def create_test_case(self, name: str, test_func: Callable = None, 
                        description: str = "", setup_func: Callable = None,
                        teardown_func: Callable = None, timeout: float = 30.0,
                        tags: List[str] = None) -> TestCase:
        """创建测试用例"""
        if name in self.test_cases:
            raise ValueError(f"测试用例 {name} 已存在")
        
        test_case = TestCase(
            name=name,
            description=description,
            setup_func=setup_func,
            teardown_func=teardown_func,
            timeout=timeout,
            tags=tags or []
        )
        
        if test_func:
            test_case.set_test_function(test_func)
        
        self.test_cases[name] = test_case
        return test_case
    
    def remove_test_case(self, name: str):
        """删除测试用例"""
        if name in self.test_cases:
            del self.test_cases[name]
        else:
            raise ValueError(f"测试用例 {name} 不存在")
    
    def create_test_suite(self, name: str, description: str = "") -> TestSuite:
        """创建测试套件"""
        if name in self.test_suites:
            raise ValueError(f"测试套件 {name} 已存在")
        
        test_suite = TestSuite(name, description)
        self.test_suites[name] = test_suite
        return test_suite
    
    def add_test_to_suite(self, suite_name: str, test_case: TestCase):
        """将测试用例添加到测试套件"""
        if suite_name not in self.test_suites:
            raise ValueError(f"测试套件 {suite_name} 不存在")
        
        self.test_suites[suite_name].add_test_case(test_case)
    
    def run_test(self, test_name: str) -> TestResultData:
        """运行单个测试"""
        if test_name not in self.test_cases:
            raise ValueError(f"测试用例 {test_name} 不存在")
        
        return self.runner.run_single_test(self.test_cases[test_name])
    
    def run_tests(self, test_names: List[str] = None, 
                  suite_name: str = None,
                  filter_tags: List[str] = None) -> List[TestResultData]:
        """运行测试"""
        if suite_name:
            if suite_name not in self.test_suites:
                raise ValueError(f"测试套件 {suite_name} 不存在")
            return self.runner.run_test_suite(
                self.test_suites[suite_name], filter_tags
            )
        elif test_names:
            test_cases = [self.test_cases[name] for name in test_names 
                         if name in self.test_cases]
            return self.runner.run_multiple_tests(test_cases)
        else:
            # 运行所有测试
            return self.runner.run_multiple_tests(list(self.test_cases.values()))
    
    def generate_report(self, results: List[TestResultData] = None, 
                       format: str = 'text', filename: str = None) -> str:
        """生成测试报告"""
        if results is None:
            results = self.runner.results
        
        if filename:
            self.reporter.save_report_to_file(results, filename, format)
            return f"报告已保存到 {filename}"
        
        if format.lower() == 'json':
            return self.reporter.generate_json_report(results)
        else:
            return self.reporter.generate_text_report(results)
    
    def analyze_results(self, results: List[TestResultData] = None) -> Dict[str, Any]:
        """分析测试结果"""
        if results is None:
            results = self.runner.results
        
        return {
            "pass_rate": self.analyzer.calculate_pass_rate(results),
            "failure_analysis": self.analyzer.get_failure_analysis(results),
            "performance_metrics": self.analyzer.get_performance_metrics(results)
        }
    
    def create_mock(self, name: str = "Mock") -> MockObject:
        """创建Mock对象"""
        return MockObject(name)
    
    def print_summary(self, results: List[TestResultData] = None):
        """打印测试摘要"""
        if results is None:
            results = self.runner.results
        
        self.reporter.print_summary(results)
    
    def get_all_test_cases(self) -> List[TestCase]:
        """获取所有测试用例"""
        return list(self.test_cases.values())
    
    def get_test_case(self, name: str) -> Optional[TestCase]:
        """获取指定测试用例"""
        return self.test_cases.get(name)
    
    def list_test_cases(self) -> List[str]:
        """列出所有测试用例名称"""
        return list(self.test_cases.keys())
    
    def list_test_suites(self) -> List[str]:
        """列出所有测试套件名称"""
        return list(self.test_suites.keys())


# 便利函数
def create_unit_tester() -> UnitTester:
    """创建单元测试器实例"""
    return UnitTester()


def assert_equal(actual, expected, message=""):
    """便利函数：断言相等"""
    return AssertionLibrary.assert_equal(actual, expected, message)


def assert_true(condition, message=""):
    """便利函数：断言为真"""
    return AssertionLibrary.assert_true(condition, message)


def assert_false(condition, message=""):
    """便利函数：断言为假"""
    return AssertionLibrary.assert_false(condition, message)


def assert_raises(exception_type, func, *args, **kwargs):
    """便利函数：断言抛出异常"""
    return AssertionLibrary.assert_raises(exception_type, func, *args, **kwargs)


if __name__ == "__main__":
    # 简单的使用示例
    tester = create_unit_tester()
    
    # 创建测试函数
    def test_addition():
        assert_equal(2 + 2, 4, "加法测试")
    
    def test_division():
        assert_equal(10 / 2, 5, "除法测试")
        assert_raises(ZeroDivisionError, lambda: 1 / 0)
    
    # 创建测试用例
    tester.create_test_case("test_addition", test_addition, "测试加法功能")
    tester.create_test_case("test_division", test_division, "测试除法功能")
    
    # 运行测试
    results = tester.run_tests()
    
    # 生成报告
    report = tester.generate_report(format='text')
    print(report)
    
    # 分析结果
    analysis = tester.analyze_results(results)
    print(f"\n通过率: {analysis['pass_rate']:.1f}%")