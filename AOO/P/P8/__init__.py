"""
P8自动化测试器包

一个功能完整的自动化测试解决方案，支持测试计划管理、用例库、执行调度、
环境管理、数据管理、结果分析和报告生成等功能。

主要组件:
- AutomatedTester: 自动化测试器主类
- TestCase: 测试用例数据结构
- TestPlan: 测试计划数据结构
- TestResult: 测试结果数据结构
- TestEnvironment: 测试环境管理
- TestDataManager: 测试数据管理
- TestReporter: 测试报告生成
- test_case: 测试用例装饰器

使用示例:
    from AutomatedTester import AutomatedTester, test_case, TestPriority
    
    # 创建测试器
    tester = AutomatedTester()
    
    # 使用装饰器创建测试用例
    @test_case(
        id="test_001",
        name="示例测试",
        priority=TestPriority.HIGH
    )
    def my_test(test_data, env_config, logs):
        logs.append("执行测试")
        return True
    
    # 注册并执行测试
    # ... (详见使用指南)
"""

from .AutomatedTester import (
    AutomatedTester,
    TestCase,
    TestPlan,
    TestResult,
    TestStatus,
    TestPriority,
    TestEnvironment,
    TestDataManager,
    TestReporter,
    test_case,
    create_test_case_from_decorator
)

__version__ = "1.0.0"
__author__ = "P8 Team"
__email__ = "p8-team@example.com"
__description__ = "P8自动化测试器 - 功能完整的自动化测试解决方案"

# 包级别的公共接口
__all__ = [
    'AutomatedTester',
    'TestCase', 
    'TestPlan',
    'TestResult',
    'TestStatus',
    'TestPriority',
    'TestEnvironment',
    'TestDataManager', 
    'TestReporter',
    'test_case',
    'create_test_case_from_decorator'
]

# 包信息
PACKAGE_INFO = {
    'name': 'P8自动化测试器',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'email': __email__,
    'components': [
        'AutomatedTester - 主测试器类',
        'TestCase - 测试用例数据结构',
        'TestPlan - 测试计划数据结构', 
        'TestResult - 测试结果数据结构',
        'TestEnvironment - 测试环境管理',
        'TestDataManager - 测试数据管理',
        'TestReporter - 测试报告生成',
        'test_case - 测试用例装饰器'
    ],
    'features': [
        '测试计划管理',
        '测试用例库',
        '测试执行调度',
        '测试环境管理', 
        '测试数据管理',
        '结果收集和分析',
        '测试报告生成',
        '失败重试机制'
    ],
    'requirements': [
        'Python 3.7+',
        'pyyaml',
        'schedule'
    ]
}

def get_package_info():
    """获取包信息"""
    return PACKAGE_INFO

def print_package_info():
    """打印包信息"""
    print(f"=== {PACKAGE_INFO['name']} ===")
    print(f"版本: {PACKAGE_INFO['version']}")
    print(f"描述: {PACKAGE_INFO['description']}")
    print(f"作者: {PACKAGE_INFO['author']}")
    print(f"邮箱: {PACKAGE_INFO['email']}")
    print()
    print("主要组件:")
    for component in PACKAGE_INFO['components']:
        print(f"  - {component}")
    print()
    print("功能特性:")
    for feature in PACKAGE_INFO['features']:
        print(f"  - {feature}")
    print()
    print("系统要求:")
    for req in PACKAGE_INFO['requirements']:
        print(f"  - {req}")

# 包初始化时的欢迎信息
try:
    print_package_info()
except Exception:
    # 静默处理，避免在某些环境下出现错误
    pass