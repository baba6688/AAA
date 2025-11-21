"""
C3抽象学习器模块
实现抽象概念提取、层次结构构建、关系映射和推理功能
"""

from .AbstractLearner import AbstractLearner, Concept, ConceptRelation, ConceptVisualizer

__all__ = [
    'AbstractLearner',
    'Concept', 
    'ConceptRelation',
    'ConceptVisualizer'
]

__version__ = '1.0.0'
__author__ = 'C3抽象学习器开发团队'

# 模块功能说明
"""
C3抽象学习器主要功能：

1. 抽象概念提取和建模
   - 使用无监督学习算法（K-means、DBSCAN、层次聚类）进行概念提取
   - 支持多种聚类方法和参数配置
   - 自动确定最优聚类数

2. 概念层次结构构建
   - 构建多层次的概念体系
   - 建立父子概念关系
   - 支持概念的动态更新

3. 概念关系映射和推理
   - 自动识别概念间的多种关系（is-a, similar-to, has-property等）
   - 基于关系网络进行逻辑推理
   - 支持复杂查询和推理任务

4. 抽象学习算法实现
   - 集成多种无监督学习算法
   - 实现概念的自动发现和分类
   - 支持增量学习和模型更新

5. 概念泛化和特化
   - 支持概念的向上泛化（抽象化）
   - 支持概念的向下特化（具体化）
   - 基于属性和实例进行概念操作

6. 概念相似度计算
   - 多维度相似度计算（属性、层次、实例）
   - 支持相似概念发现和聚类
   - 动态相似度阈值调整

7. 抽象知识应用和验证
   - 将抽象概念应用于新数据
   - 知识有效性验证和评估
   - 支持模型持久化和加载

8. 概念可视化工具
   - 层次结构可视化
   - 相似度矩阵热图
   - 概念关系网络图
   - 统计信息图表

使用示例：
```python
from C.C3 import AbstractLearner

# 初始化学习器
learner = AbstractLearner(
    clustering_method='kmeans',
    similarity_threshold=0.7,
    max_levels=5,
    min_concept_size=3
)

# 提取概念
concepts = learner.extract_concepts(data, feature_names)

# 构建层次结构
hierarchy = learner.build_hierarchy(concepts)

# 映射关系
relations = learner.map_relations()

# 推理
result = learner.reason("找出相似的概念")

# 可视化
learner.visualize_concepts(save_path='concepts.png')
```
"""