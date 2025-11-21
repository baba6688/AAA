"""
U4 优化算法库模块 - Optimization Algorithm Library

这个模块提供了一系列完整的优化算法实现，包括单目标优化、多目标优化和超参数调优功能。

主要功能：
- 单目标优化算法：梯度下降、遗传算法、粒子群优化、模拟退火、蚁群算法、差分进化、贝叶斯优化
- 多目标优化算法：NSGA-II
- 超参数自动调优：支持多种优化算法的自动参数调优
- 问题定义：支持单目标和多目标优化问题定义
- 结果可视化：收敛曲线和帕累托前沿绘制

模块结构：
- OptimizationProblem: 优化问题基类
- MultiObjectiveProblem: 多目标优化问题类
- BaseOptimizer: 优化器基类
- GradientDescentOptimizer: 梯度下降优化器（包括GD、SGD、Momentum、Adam、RMSprop）
- GeneticAlgorithmOptimizer: 遗传算法优化器
- ParticleSwarmOptimizer: 粒子群优化算法
- SimulatedAnnealingOptimizer: 模拟退火算法
- AntColonyOptimizer: 蚁群算法
- DifferentialEvolutionOptimizer: 差分进化算法
- BayesianOptimizer: 贝叶斯优化算法
- NSGA2Optimizer: NSGA-II多目标优化算法
- HyperparameterTuner: 超参数自动调优器
- OptimizationAlgorithmLibrary: 优化算法库主类，提供统一的接口

使用示例：
    from U.U4 import OptimizationProblem, GeneticAlgorithmOptimizer
    
    # 创建优化问题
    def objective(x):
        return sum(xi**2 for xi in x)
    
    bounds = [(-5, 5), (-5, 5)]
    problem = OptimizationProblem(2, bounds, objective)
    
    # 使用遗传算法优化
    optimizer = GeneticAlgorithmOptimizer(problem, population_size=50)
    best_solution, best_fitness = optimizer.optimize(max_iterations=1000)

Author: U4模块开发团队
Date: 2025-11-14
Version: 1.0.0
"""

# 导入所有核心类和优化算法
try:
    # 相对导入（当作为包的一部分时）
    from .OptimizationAlgorithmLibrary import (
        # 优化问题类
        OptimizationProblem,
        MultiObjectiveProblem,
        
        # 优化器基类
        BaseOptimizer,
        
        # 单目标优化算法
        GradientDescentOptimizer,
        GeneticAlgorithmOptimizer,
        ParticleSwarmOptimizer,
        SimulatedAnnealingOptimizer,
        AntColonyOptimizer,
        DifferentialEvolutionOptimizer,
        BayesianOptimizer,
        
        # 多目标优化算法
        NSGA2Optimizer,
        
        # 超参数调优
        HyperparameterTuner,
        
        # 主库类
        OptimizationAlgorithmLibrary
    )
except ImportError:
    # 绝对导入（当直接运行时）
    from OptimizationAlgorithmLibrary import (
        # 优化问题类
        OptimizationProblem,
        MultiObjectiveProblem,
        
        # 优化器基类
        BaseOptimizer,
        
        # 单目标优化算法
        GradientDescentOptimizer,
        GeneticAlgorithmOptimizer,
        ParticleSwarmOptimizer,
        SimulatedAnnealingOptimizer,
        AntColonyOptimizer,
        DifferentialEvolutionOptimizer,
        BayesianOptimizer,
        
        # 多目标优化算法
        NSGA2Optimizer,
        
        # 超参数调优
        HyperparameterTuner,
        
        # 主库类
        OptimizationAlgorithmLibrary
    )

# 定义模块的公共接口
__all__ = [
    # 优化问题类
    'OptimizationProblem',
    'MultiObjectiveProblem',
    
    # 优化器基类
    'BaseOptimizer',
    
    # 单目标优化算法
    'GradientDescentOptimizer',
    'GeneticAlgorithmOptimizer', 
    'ParticleSwarmOptimizer',
    'SimulatedAnnealingOptimizer',
    'AntColonyOptimizer',
    'DifferentialEvolutionOptimizer',
    'BayesianOptimizer',
    
    # 多目标优化算法
    'NSGA2Optimizer',
    
    # 超参数调优
    'HyperparameterTuner',
    
    # 主库类
    'OptimizationAlgorithmLibrary'
]

# 模块版本信息
__version__ = '1.0.0'
__author__ = 'U4模块开发团队'
__email__ = 'u4-team@example.com'

# 模块初始化信息
def __initialize_module_info():
    """初始化模块信息"""
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    U4 优化算法库 v{__version__:<8}                      ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  🚀 单目标优化算法：                                        ║
    ║     • 梯度下降 (GD/SGD/Momentum/Adam/RMSprop)             ║
    ║     • 遗传算法 (GA)                                        ║
    ║     • 粒子群优化 (PSO)                                     ║
    ║     • 模拟退火 (SA)                                        ║
    ║     • 蚁群算法 (ACO)                                       ║
    ║     • 差分进化 (DE)                                        ║
    ║     • 贝叶斯优化 (BO)                                      ║
    ║                                                               ║
    ║  🎯 多目标优化算法：                                        ║
    ║     • NSGA-II                                              ║
    ║                                                               ║
    ║  ⚙️  超参数调优：                                           ║
    ║     • 自动参数优化                                         ║
    ║                                                               ║
    ║  📊 可视化功能：                                            ║
    ║     • 收敛曲线                                             ║
    ║     • 帕累托前沿                                           ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  使用 'from U.U4 import *' 或指定具体类导入              ║
    ║  例如: from U.U4 import GeneticAlgorithmOptimizer          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

# 打印初始化信息
__initialize_module_info()

# 便捷的导入别名（可选）
# GA = GeneticAlgorithmOptimizer
# PSO = ParticleSwarmOptimizer
# SA = SimulatedAnnealingOptimizer
# DE = DifferentialEvolutionOptimizer
# BO = BayesianOptimizer

# 模块功能检查
def _check_dependencies():
    """检查模块依赖"""
    required_modules = ['numpy', 'matplotlib', 'scipy', 'sklearn']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"警告：缺少以下依赖模块：{', '.join(missing_modules)}")
        print("请使用 pip install 安装：")
        print(f"pip install {' '.join(missing_modules)}")
    
    return len(missing_modules) == 0

# 检查依赖
_dependencies_ok = _check_dependencies()

if _dependencies_ok:
    print("✅ 所有依赖模块检查通过")
else:
    print("❌ 依赖模块检查失败，请安装缺失的模块")