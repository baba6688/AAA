"""
J5优化工具模块
提供完整的优化算法集合，包括经典优化、智能优化、约束优化、
多目标优化、贝叶斯优化等功能。

版本: 1.0.0
日期: 2025-11-13
"""

try:
    # 相对导入（在包内部使用）
    from .OptimizationTools import (
        # 异常类
        OptimizationError,
        ConvergenceError,
        ConstraintViolationError,
        
        # 数据结构
        OptimizationResult,
        MultiObjectiveResult,
        
        # 经典优化器
        GradientDescentOptimizer,
        NewtonOptimizer,
        ConjugateGradientOptimizer,
        
        # 智能优化器
        GeneticAlgorithm,
        ParticleSwarmOptimizer,
        SimulatedAnnealingOptimizer,
        
        # 约束优化器
        LagrangeMultiplierMethod,
        PenaltyFunctionMethod,
        BarrierFunctionMethod,
        
        # 多目标优化器
        NSGA2Optimizer,
        
        # 贝叶斯优化器
        BayesianOptimizer,
        
        # 主要工具类
        OptimizationTools,
        ExtendedOptimizationTools,
        FinalOptimizationTools,
        
        # 高级功能
        AdvancedOptimizers,
        OptimizationBenchmark,
        OptimizationUtils,
        OptimizationProfiler,
        OptimizationVisualization,
        
        # 元启发式算法
        MetaheuristicOptimizers,
        
        # 分析工具
        OptimizationAnalysis,
        OptimizationConfig,
        
        # 工具函数
        check_bounds,
        project_to_bounds,
        timer,
        
        # 示例函数
        comprehensive_example,
        final_comprehensive_example
    )
except ImportError:
    # 绝对导入（作为独立模块使用）
    from OptimizationTools import (
        # 异常类
        OptimizationError,
        ConvergenceError,
        ConstraintViolationError,
        
        # 数据结构
        OptimizationResult,
        MultiObjectiveResult,
        
        # 经典优化器
        GradientDescentOptimizer,
        NewtonOptimizer,
        ConjugateGradientOptimizer,
        
        # 智能优化器
        GeneticAlgorithm,
        ParticleSwarmOptimizer,
        SimulatedAnnealingOptimizer,
        
        # 约束优化器
        LagrangeMultiplierMethod,
        PenaltyFunctionMethod,
        BarrierFunctionMethod,
        
        # 多目标优化器
        NSGA2Optimizer,
        
        # 贝叶斯优化器
        BayesianOptimizer,
        
        # 主要工具类
        OptimizationTools,
        ExtendedOptimizationTools,
        FinalOptimizationTools,
        
        # 高级功能
        AdvancedOptimizers,
        OptimizationBenchmark,
        OptimizationUtils,
        OptimizationProfiler,
        OptimizationVisualization,
        
        # 元启发式算法
        MetaheuristicOptimizers,
        
        # 分析工具
        OptimizationAnalysis,
        OptimizationConfig,
        
        # 工具函数
        check_bounds,
        project_to_bounds,
        timer,
        
        # 示例函数
        comprehensive_example,
        final_comprehensive_example
    )

# 定义模块级公开接口
__all__ = [
    # 异常类
    'OptimizationError',
    'ConvergenceError', 
    'ConstraintViolationError',
    
    # 数据结构
    'OptimizationResult',
    'MultiObjectiveResult',
    
    # 核心优化器
    'GradientDescentOptimizer',
    'NewtonOptimizer', 
    'ConjugateGradientOptimizer',
    'GeneticAlgorithm',
    'ParticleSwarmOptimizer',
    'SimulatedAnnealingOptimizer',
    'LagrangeMultiplierMethod',
    'PenaltyFunctionMethod',
    'BarrierFunctionMethod',
    'NSGA2Optimizer',
    'BayesianOptimizer',
    
    # 主要工具类
    'OptimizationTools',
    'ExtendedOptimizationTools',
    'FinalOptimizationTools',
    
    # 高级功能
    'AdvancedOptimizers',
    'OptimizationBenchmark',
    'OptimizationUtils',
    'OptimizationProfiler',
    'OptimizationVisualization',
    
    # 扩展算法
    'MetaheuristicOptimizers',
    'OptimizationAnalysis',
    'OptimizationConfig',
    
    # 工具函数
    'check_bounds',
    'project_to_bounds',
    'timer',
    
    # 示例
    'comprehensive_example',
    'final_comprehensive_example'
]

# 模块版本信息
__version__ = "1.0.0"
__author__ = "J5系统"
__email__ = "support@j5system.com"

# 模块配置
# 设置默认日志级别
import logging
logging.basicConfig(level=logging.INFO)

def get_version():
    """获取模块版本"""
    return __version__

def get_author():
    """获取模块作者"""
    return __author__