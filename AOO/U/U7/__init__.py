#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图算法库 (Graph Algorithm Library) - U7模块

这是一个完整的图算法库，实现了图论中的核心算法和现代图神经网络技术。
包含图的遍历、搜索、最短路径、最小生成树、聚类、社区检测、图神经网络、
相似性度量、嵌入学习、动态图算法和可视化分析等功能。

主要组件:
- Graph: 图数据结构
- TraversalAlgorithms: 图的遍历和搜索算法
- ShortestPathAlgorithms: 最短路径算法
- MinimumSpanningTree: 最小生成树算法
- CommunityDetection: 图的聚类和社区检测
- GraphNeuralNetwork: 图神经网络
- GraphSimilarity: 图的相似性度量
- GraphEmbedding: 图的嵌入和表示学习
- DynamicGraph: 动态图算法
- GraphVisualization: 图的可视化和分析
- GraphAlgorithmLibrary: 主图算法库类


版本: 1.0.0
日期: 2025-11-05
"""

from .GraphAlgorithmLibrary import (
    # 数据结构
    Edge,
    Node,
    Graph,
    
    # 算法类
    TraversalAlgorithms,
    ShortestPathAlgorithms,
    MinimumSpanningTree,
    CommunityDetection,
    GraphNeuralNetwork,
    GraphSimilarity,
    GraphEmbedding,
    DynamicGraph,
    GraphVisualization,
    
    # 主库类
    GraphAlgorithmLibrary,
    
    # 工具函数
    create_sample_graph,
    demonstrate_algorithms
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

__all__ = [
    # 数据结构
    'Edge',
    'Node', 
    'Graph',
    
    # 算法类
    'TraversalAlgorithms',
    'ShortestPathAlgorithms',
    'MinimumSpanningTree',
    'CommunityDetection',
    'GraphNeuralNetwork',
    'GraphSimilarity',
    'GraphEmbedding',
    'DynamicGraph',
    'GraphVisualization',
    
    # 主库类
    'GraphAlgorithmLibrary',
    
    # 工具函数
    'create_sample_graph',
    'demonstrate_algorithms'
]

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """获取版本字符串"""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"

def get_version_info():
    """获取详细版本信息"""
    return VERSION_INFO.copy()

# 库初始化检查
def _check_dependencies():
    """检查依赖项"""
    try:
        import math
        import random
        import itertools
        import collections
        import json
        import pickle
        import time
        return True
    except ImportError as e:
        print(f"警告: 缺少必要的依赖项: {e}")
        return False

# 初始化标志
_DEPENDENCIES_OK = _check_dependencies()

if _DEPENDENCIES_OK:
    print(f"图算法库 v{get_version()} 初始化成功")
else:
    print("图算法库初始化失败: 依赖项检查失败")