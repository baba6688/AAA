"""
U5 统计算法库模块
==================

U5子模块提供完整的统计分析和机器学习算法库，包含各种统计算法和数据分析工具。

功能模块：
1. 描述性统计分析
2. 假设检验和显著性检验  
3. 方差分析(ANOVA)
4. 回归分析
5. 时间序列分析
6. 概率分布拟合
7. 贝叶斯统计
8. 多元统计分析
9. 统计显著性检验

主要类：
- StatisticalAlgorithmLibrary: 统计算法库主类

版本: 1.0.0
作者: U5算法库团队
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "U5算法库团队"

# 导入主类 - 支持相对导入和绝对导入
try:
    # 相对导入（作为包的一部分）
    from .StatisticalAlgorithmLibrary import StatisticalAlgorithmLibrary
except ImportError:
    # 绝对导入（独立测试时）
    from StatisticalAlgorithmLibrary import StatisticalAlgorithmLibrary

# 定义导出列表
__all__ = [
    'StatisticalAlgorithmLibrary'
]

# 模块初始化信息
def __init_module_info():
    """初始化模块信息"""
    print("=" * 60)
    print("U5 统计算法库 v1.0.0")
    print("=" * 60)
    print("已成功加载统计分析和机器学习算法库")
    print("包含9大功能模块的完整统计分析工具")
    print("主要功能：描述性统计、假设检验、回归分析、时间序列等")
    print("=" * 60)

# 自动执行模块初始化（可选，取消注释以启用）
# __init_module_info()

# 模块使用示例
def get_started_example():
    """
    U5统计算法库使用示例
    
    Returns:
        StatisticalAlgorithmLibrary: 统计库实例
    """
    print("创建U5统计算法库实例...")
    stats_lib = StatisticalAlgorithmLibrary()
    
    print("生成示例数据...")
    import numpy as np
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)
    
    print("进行描述性统计分析...")
    desc_stats = stats_lib.descriptive_statistics(data)
    print(f"样本均值: {desc_stats['均值']:.2f}")
    print(f"样本标准差: {desc_stats['标准差']:.2f}")
    print(f"偏度: {desc_stats['偏度']:.3f}")
    print(f"峰度: {desc_stats['峰度']:.3f}")
    
    print("进行单样本t检验...")
    t_test = stats_lib.one_sample_t_test(data, 100)
    print(f"t统计量: {t_test['t统计量']:.4f}")
    print(f"p值: {t_test['p值']:.6f}")
    print(f"决策: {t_test['决策']}")
    
    return stats_lib

# 快速访问统计函数的快捷方式（可选）
def quick_descriptive_stats(data):
    """快速描述性统计"""
    stats_lib = StatisticalAlgorithmLibrary()
    return stats_lib.descriptive_statistics(data)

def quick_t_test(data, population_mean):
    """快速t检验"""
    stats_lib = StatisticalAlgorithmLibrary()
    return stats_lib.one_sample_t_test(data, population_mean)

def quick_regression(x, y):
    """快速线性回归"""
    stats_lib = StatisticalAlgorithmLibrary()
    return stats_lib.linear_regression(x, y)

# 导出快捷函数到__all__
__all__.extend([
    'get_started_example',
    'quick_descriptive_stats', 
    'quick_t_test',
    'quick_regression'
])

# 兼容性检查
def check_dependencies():
    """检查依赖库是否已安装"""
    missing_deps = []
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
        
    try:
        import scipy
    except ImportError:
        missing_deps.append('scipy')
        
    try:
        import sklearn
    except ImportError:
        missing_deps.append('scikit-learn')
    
    if missing_deps:
        print(f"警告: 缺少以下依赖库: {', '.join(missing_deps)}")
        print("请使用以下命令安装: pip install " + " ".join(missing_deps))
        return False
    else:
        print("所有依赖库检查通过!")
        return True

# 在模块导入时检查依赖（可选）
# check_dependencies()