"""
P1单元测试器包
提供完整的单元测试框架功能
"""

# 核心类
from .UnitTester import (
    # 主要类
    UnitTester,
    TestCase,
    TestSuite,
    TestRunner,
    TestReporter,
    TestAnalyzer,
    
    # 断言相关
    AssertionLibrary,
    MockObject,
    
    # 数据类
    TestResult,
    TestResultData,
    
    # 便利函数
    create_unit_tester,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises
)

# 包版本
__version__ = "1.0.0"
__author__ = "P1单元测试器开发团队"

# 导出的公共API
__all__ = [
    # 主要类
    "UnitTester",
    "TestCase", 
    "TestSuite",
    "TestRunner",
    "TestReporter",
    "TestAnalyzer",
    
    # 断言和Mock
    "AssertionLibrary",
    "MockObject",
    
    # 数据类型
    "TestResult",
    "TestResultData",
    
    # 便利函数
    "create_unit_tester",
    "assert_equal",
    "assert_true", 
    "assert_false",
    "assert_raises"
]

# 使用示例
"""
基本使用示例：

from P1 import UnitTester, assert_equal, assert_true

# 创建测试器实例
tester = UnitTester()

# 创建测试函数
def test_addition():
    assert_equal(2 + 2, 4, "加法测试")

# 创建测试用例
tester.create_test_case("test_addition", test_addition, "测试加法功能")

# 运行测试
results = tester.run_tests()

# 生成报告
report = tester.generate_report(format='text')
print(report)

# 分析结果
analysis = tester.analyze_results(results)
print(f"通过率: {analysis['pass_rate']:.1f}%")
"""

# 快速开始指南
QUICK_START = """
P1单元测试器快速开始：

1. 创建测试器实例：
   tester = UnitTester()

2. 编写测试函数并使用断言：
   def test_example():
       assert_equal(1 + 1, 2)
       assert_true(5 > 3)

3. 创建测试用例：
   tester.create_test_case("test_example", test_example, "示例测试")

4. 运行测试：
   results = tester.run_tests()

5. 查看报告：
   print(tester.generate_report())
"""

print("P1单元测试器已加载")
print("使用 help(P1) 查看更多信息")
print("使用 P1.QUICK_START 查看快速开始指南")