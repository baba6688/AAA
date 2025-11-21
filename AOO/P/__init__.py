"""
P区 - 企业级测试系统
==================

P区是一个完整的测试框架，提供从单元测试到企业级系统测试的全方位解决方案。

功能模块:
- P1: 单元测试器 (UnitTester) - 完整的单元测试框架
- P2: 集成测试器 (IntegrationTester) - 系统集成测试
- P3: 性能测试器 (PerformanceTester) - 性能测试和基准测试
- P4: 压力测试器 (StressTester) - 压力测试和极限负载测试
- P5: 回测引擎 (BacktestEngine) - 交易策略回测
- P6: 模拟交易器 (SimulatedTrader) - 模拟交易执行
- P7: A/B测试器 (ABTester) - A/B测试实验
- P8: 自动化测试器 (AutomatedTester) - 自动化测试管理
- P9: 测试状态聚合器 (TestStatusAggregator) - 测试状态聚合分析

主要特性:
- 全面的测试覆盖：从单元测试到系统集成测试
- 丰富的断言库：支持多种数据类型和比较操作
- 模拟对象支持：Mock对象、Stub对象等
- 并行测试执行：支持多线程/多进程测试执行
- 详细报告生成：文本、HTML、JSON多种格式
- 性能监控：实时性能指标收集和分析
- 数据持久化：SQLite数据库存储测试结果
- 配置化测试：灵活的测试配置管理

版本: 1.0.0
作者: P区开发团队
许可证: MIT
"""

# ========== P1 - 单元测试器导出 ==========
from P.P1.UnitTester import (
    # 核心类和枚举
    TestResult,
    TestCase,
    TestResultData,
    TestSuite,
    TestRunner,
    TestReporter,
    TestAnalyzer,
    AssertionLibrary,
    MockObject,
    UnitTester,
    
    # 便利函数
    create_unit_tester,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises
)

# ========== P2 - 集成测试器导出 ==========
from P.P2.IntegrationTester import (
    # 核心类和枚举
    TestResult as IntegrationTestResult,
    Environment,
    TestCase as IntegrationTestCase,
    TestExecutionResult,
    IntegrationTestConfig,
    BaseTest,
    APIIntegrationTest,
    DatabaseIntegrationTest,
    MessageQueueIntegrationTest,
    IntegrationTester,
    
    # 便利函数
    create_integration_tester
)

# ========== P3 - 性能测试器导出 ==========
from P.P3.PerformanceTester import (
    # 核心类和枚举
    TestResult as PerformanceTestResult,
    PerformanceMetrics,
    PerformanceTester,
    
    # 便利函数
    create_performance_tester
)

# ========== P4 - 压力测试器导出 ==========
from P.P4.StressTester import (
    # 核心类和枚举
    TestResult as StressTestResult,
    SystemMetrics,
    LoadGenerator,
    StressTester,
    StressTestManager,
    
    # 便利函数
    create_stress_tester
)

# ========== P5 - 回测引擎导出 ==========
from P.P5.BacktestEngine import (
    # 核心类和枚举
    TradeRecord,
    PerformanceAnalyzer,
    BacktestEngine,
    SimpleStrategy,
    
    # 便利函数
    create_backtest_engine
)

# ========== P6 - 模拟交易器导出 ==========
from P.P6.SimulatedTrader import (
    # 核心类和枚举
    OrderType,
    OrderSide,
    OrderStatus,
    SignalType,
    Order,
    Trade,
    Position,
    Portfolio,
    Signal,
    SimulatedTrader,
    MarketSimulator,
    
    # 便利函数
    create_simulated_trader
)

# ========== P7 - A/B测试器导出 ==========
from P.P7.ABTester import (
    # 核心类和枚举
    ExperimentConfig,
    UserData,
    ExperimentResult,
    ABTestManager,
    
    # 便利函数
    create_ab_tester
)

# ========== P8 - 自动化测试器导出 ==========
from P.P8.AutomatedTester import (
    # 核心类和枚举
    TestStatus as AutomatedTestStatus,
    TestPriority,
    TestCase as AutomatedTestCase,
    TestResult as AutomatedTestResult,
    TestPlan,
    TestEnvironment,
    TestScheduler,
    AutomatedTester,
    
    # 便利函数
    create_automated_tester
)

# ========== P9 - 测试状态聚合器导出 ==========
from P.P9.TestStatusAggregator import (
    # 核心类和枚举
    TestStatus as AggregatedTestStatus,
    AlertLevel,
    TestResult as AggregatedTestResult,
    ModuleStatus,
    Alert,
    SystemHealth,
    TestStatusAggregator,
    
    # 便利函数
    create_status_aggregator
)

# ========== 主包导出列表 ==========
__all__ = [
    # P1 - 单元测试器
    'TestResult', 'TestCase', 'TestResultData', 'TestSuite', 'TestRunner',
    'TestReporter', 'TestAnalyzer', 'AssertionLibrary', 'MockObject', 
    'UnitTester', 'create_unit_tester', 'assert_equal', 'assert_true', 
    'assert_false', 'assert_raises',
    
    # P2 - 集成测试器
    'IntegrationTestResult', 'Environment', 'IntegrationTestCase', 
    'TestExecutionResult', 'IntegrationTestConfig', 'BaseTest', 
    'APIIntegrationTest', 'DatabaseIntegrationTest', 'MessageQueueIntegrationTest',
    'IntegrationTester', 'create_integration_tester',
    
    # P3 - 性能测试器
    'PerformanceTestResult', 'PerformanceMetrics', 'PerformanceTester',
    'create_performance_tester',
    
    # P4 - 压力测试器
    'StressTestResult', 'SystemMetrics', 'LoadGenerator', 'StressTester',
    'StressTestManager', 'create_stress_tester',
    
    # P5 - 回测引擎
    'TradeRecord', 'PerformanceAnalyzer', 'BacktestEngine', 'SimpleStrategy',
    'create_backtest_engine',
    
    # P6 - 模拟交易器
    'OrderType', 'OrderSide', 'OrderStatus', 'SignalType', 'Order', 'Trade',
    'Position', 'Portfolio', 'Signal', 'SimulatedTrader', 'MarketSimulator',
    'create_simulated_trader',
    
    # P7 - A/B测试器
    'ExperimentConfig', 'UserData', 'ExperimentResult', 'ABTestManager',
    'create_ab_tester',
    
    # P8 - 自动化测试器
    'AutomatedTestStatus', 'TestPriority', 'AutomatedTestCase',
    'AutomatedTestResult', 'TestPlan', 'TestEnvironment', 'TestScheduler',
    'AutomatedTester', 'create_automated_tester',
    
    # P9 - 测试状态聚合器
    'AggregatedTestStatus', 'AlertLevel', 'AggregatedTestResult', 'ModuleStatus',
    'Alert', 'SystemHealth', 'TestStatusAggregator', 'create_status_aggregator'
]

# ========== 包信息 ==========
__version__ = "1.0.0"
__author__ = "P区开发团队"
__license__ = "MIT"
__email__ = "p-team@example.com"
__url__ = "https://github.com/company/p-testing-system"

# ========== 工厂函数 ==========
def create_testing_component(component_type: str, **kwargs):
    """
    创建测试组件的便利工厂函数
    
    Args:
        component_type: 组件类型 ('unit', 'integration', 'performance', 'stress', 'backtest', 'trader', 'abtest', 'automated', 'aggregator')
        **kwargs: 组件特定参数
    
    Returns:
        对应的测试组件实例
    
    Examples:
        # 创建单元测试器
        unit_tester = create_testing_component('unit')
        
        # 创建集成测试器
        integration_tester = create_testing_component('integration', config=IntegrationTestConfig(...))
        
        # 创建性能测试器
        performance_tester = create_testing_component('performance', base_url='http://api.example.com')
    """
    component_map = {
        'unit': create_unit_tester,
        'integration': create_integration_tester,
        'performance': create_performance_tester,
        'stress': create_stress_tester,
        'backtest': create_backtest_engine,
        'trader': create_simulated_trader,
        'abtest': create_ab_tester,
        'automated': create_automated_tester,
        'aggregator': create_status_aggregator
    }
    
    if component_type not in component_map:
        raise ValueError(
            f"不支持的组件类型: {component_type}。支持的类型: {list(component_map.keys())}"
        )
    
    return component_map[component_type](**kwargs)

def get_available_modules():
    """获取所有可用的测试模块"""
    return {
        'P1': 'UnitTester - 单元测试器',
        'P2': 'IntegrationTester - 集成测试器',
        'P3': 'PerformanceTester - 性能测试器',
        'P4': 'StressTester - 压力测试器',
        'P5': 'BacktestEngine - 回测引擎',
        'P6': 'SimulatedTrader - 模拟交易器',
        'P7': 'ABTester - A/B测试器',
        'P8': 'AutomatedTester - 自动化测试器',
        'P9': 'TestStatusAggregator - 测试状态聚合器'
    }

def get_module_info(module_name: str):
    """
    获取指定模块的详细信息
    
    Args:
        module_name: 模块名称 (P1-P9)
    
    Returns:
        模块信息字典
    
    Raises:
        ValueError: 无效的模块名称
    """
    module_info = {
        'P1': {
            'name': 'UnitTester',
            'description': '完整的单元测试框架，支持断言、Mock对象、测试套件',
            'components': ['AssertionLibrary', 'MockObject', 'TestRunner', 'TestReporter']
        },
        'P2': {
            'name': 'IntegrationTester',
            'description': '集成测试器，支持API、数据库、消息队列集成测试',
            'components': ['APIIntegrationTest', 'DatabaseIntegrationTest', 'MessageQueueIntegrationTest']
        },
        'P3': {
            'name': 'PerformanceTester',
            'description': '性能测试器，支持负载测试、基准测试、实时监控',
            'components': ['PerformanceMetrics', 'PerformanceTester']
        },
        'P4': {
            'name': 'StressTester',
            'description': '压力测试器，支持极限负载、系统资源监控',
            'components': ['SystemMetrics', 'StressTester', 'StressTestManager']
        },
        'P5': {
            'name': 'BacktestEngine',
            'description': '回测引擎，支持交易策略历史数据回测',
            'components': ['PerformanceAnalyzer', 'BacktestEngine', 'SimpleStrategy']
        },
        'P6': {
            'name': 'SimulatedTrader',
            'description': '模拟交易器，支持订单管理、组合管理、风险控制',
            'components': ['Order', 'Trade', 'Position', 'Portfolio', 'SimulatedTrader']
        },
        'P7': {
            'name': 'ABTester',
            'description': 'A/B测试器，支持实验设计、用户分组、统计分析',
            'components': ['ExperimentConfig', 'UserData', 'ExperimentResult', 'ABTestManager']
        },
        'P8': {
            'name': 'AutomatedTester',
            'description': '自动化测试器，支持测试计划、用例库、执行调度',
            'components': ['TestPlan', 'TestEnvironment', 'TestScheduler', 'AutomatedTester']
        },
        'P9': {
            'name': 'TestStatusAggregator',
            'description': '测试状态聚合器，支持状态聚合、监控告警、报告生成',
            'components': ['SystemHealth', 'TestStatusAggregator', 'Alert']
        }
    }
    
    if module_name not in module_info:
        raise ValueError(f"无效的模块名称: {module_name}。支持的模块: P1-P9")
    
    return module_info[module_name]

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 显示可用模块
    print("P区 - 企业级测试系统")
    print("=" * 50)
    
    modules = get_available_modules()
    for code, description in modules.items():
        print(f"{code}: {description}")
    
    print("\n使用示例:")
    print("from P import create_testing_component")
    print("unit_tester = create_testing_component('unit')")
    print("performance_tester = create_testing_component('performance', base_url='http://api.example.com')")