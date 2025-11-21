#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2集成测试器包
提供全面的集成测试功能，包括系统集成、API接口、数据库、消息队列等测试
"""

from .IntegrationTester import (
    # 核心类和枚举
    IntegrationTester,
    IntegrationTestConfig,
    TestCase,
    TestResult,
    TestExecutionResult,
    TestResult as TR,
    Environment,
    
    # 测试组件
    SystemIntegrationTest,
    APIInterfaceTest,
    DatabaseIntegrationTest,
    MessageQueueTest,
    ExternalServiceTest,
    DataFlowTest,
    EnvironmentConfigTest,
    
    # 工具函数
    create_default_config
)

# 包版本
__version__ = "1.0.0"

# 导出的所有公共类和函数
__all__ = [
    # 核心类
    'IntegrationTester',
    'IntegrationTestConfig',
    'TestCase',
    'TestResult',
    'TestExecutionResult',
    'Environment',
    
    # 测试组件
    'SystemIntegrationTest',
    'APIInterfaceTest',
    'DatabaseIntegrationTest',
    'MessageQueueTest',
    'ExternalServiceTest',
    'DataFlowTest',
    'EnvironmentConfigTest',
    
    # 工具函数
    'create_default_config',
    
    # 常量
    'TR'
]

# 便捷的别名
TestStatus = TestResult
TestStatus.PASS = TestResult.PASS
TestStatus.FAIL = TestResult.FAIL
TestStatus.SKIP = TestResult.SKIP
TestStatus.ERROR = TestResult.ERROR

# 环境别名
Env = Environment
Env.DEVELOPMENT = Environment.DEVELOPMENT
Env.STAGING = Environment.STAGING
Env.PRODUCTION = Environment.PRODUCTION
Env.TESTING = Environment.TESTING