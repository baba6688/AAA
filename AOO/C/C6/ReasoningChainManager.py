"""
推理链管理器 (ReasoningChainManager)
实现推理链的构建、管理、优化和验证功能
支持多种推理算法和可视化
"""

import json
import time
import uuid
import networkx as nx
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ReasoningType(Enum):
    """推理类型枚举"""
    DEDUCTIVE = "deductive"  # 演绎推理
    INDUCTIVE = "inductive"  # 归纳推理
    ABDUCTIVE = "abductive"  # 溯因推理
    ANALOGICAL = "analogical"  # 类比推理
    CAUSAL = "causal"  # 因果推理
    HYPOTHETICAL = "hypothetical"  # 假设推理


class NodeType(Enum):
    """节点类型枚举"""
    FACT = "fact"  # 事实
    HYPOTHESIS = "hypothesis"  # 假设
    RULE = "rule"  # 规则
    CONCLUSION = "conclusion"  # 结论
    EVIDENCE = "evidence"  # 证据
    QUESTION = "question"  # 问题


class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_LOW = (0.0, 0.2)
    LOW = (0.2, 0.4)
    MEDIUM = (0.4, 0.6)
    HIGH = (0.6, 0.8)
    VERY_HIGH = (0.8, 1.0)


@dataclass
class ReasoningNode:
    """推理节点"""
    id: str
    content: str
    node_type: NodeType
    confidence: float
    reasoning_type: Optional[ReasoningType] = None
    metadata: Dict[str, Any] = None
    timestamp: str = None
    version: int = 1
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningNode':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class ReasoningEdge:
    """推理边"""
    source: str
    target: str
    weight: float
    reasoning_type: ReasoningType
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReasoningPath:
    """推理路径"""
    nodes: List[str]
    edges: List[str]
    total_confidence: float
    reasoning_types: List[ReasoningType]
    length: int
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""
    build_time: float
    optimization_time: float
    validation_time: float
    node_count: int
    edge_count: int
    path_count: int
    average_confidence: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ReasoningChainManager:
    """推理链管理器"""
    
    def __init__(self, name: str = "推理链管理器"):
        self.name = name
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ReasoningNode] = {}
        self.edges: Dict[str, ReasoningEdge] = {}
        self.paths: Dict[str, ReasoningPath] = {}
        self.versions: Dict[str, Any] = {}
        self.performance_history: List[PerformanceMetrics] = []
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # 推理算法配置
        self.reasoning_algorithms = {
            ReasoningType.DEDUCTIVE: self._deductive_reasoning,
            ReasoningType.INDUCTIVE: self._inductive_reasoning,
            ReasoningType.ABDUCTIVE: self._abductive_reasoning,
            ReasoningType.ANALOGICAL: self._analogical_reasoning,
            ReasoningType.CAUSAL: self._causal_reasoning,
            ReasoningType.HYPOTHETICAL: self._hypothetical_reasoning
        }
        
        # 优化算法配置
        self.optimization_algorithms = {
            'shortest_path': self._shortest_path_optimization,
            'highest_confidence': self._highest_confidence_optimization,
            'balanced': self._balanced_optimization,
            'multi_criteria': self._multi_criteria_optimization
        }
    
    def add_node(self, content: str, node_type: NodeType, 
                confidence: float = 0.5, reasoning_type: Optional[ReasoningType] = None,
                metadata: Dict[str, Any] = None) -> str:
        """添加推理节点"""
        with self._lock:
            node_id = str(uuid.uuid4())
            node = ReasoningNode(
                id=node_id,
                content=content,
                node_type=node_type,
                confidence=confidence,
                reasoning_type=reasoning_type,
                metadata=metadata or {}
            )
            
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.to_dict())
            
            self.logger.info(f"添加节点: {node_id} ({node_type.value})")
            return node_id
    
    def add_edge(self, source_id: str, target_id: str, 
                reasoning_type: ReasoningType, weight: float = 1.0,
                confidence: float = 0.5, metadata: Dict[str, Any] = None) -> bool:
        """添加推理边"""
        with self._lock:
            if source_id not in self.nodes or target_id not in self.nodes:
                raise ValueError("源节点或目标节点不存在")
            
            edge_id = str(uuid.uuid4())
            edge = ReasoningEdge(
                source=source_id,
                target=target_id,
                weight=weight,
                reasoning_type=reasoning_type,
                confidence=confidence,
                metadata=metadata or {}
            )
            
            self.edges[edge_id] = edge
            self.graph.add_edge(source_id, target_id, 
                              edge_id=edge_id, 
                              weight=weight,
                              confidence=confidence)
            
            self.logger.info(f"添加边: {source_id} -> {target_id}")
            return True
    
    def build_reasoning_chain(self, facts: List[Dict[str, Any]], 
                            rules: List[Dict[str, Any]]) -> str:
        """构建推理链"""
        start_time = time.time()
        
        try:
            # 添加事实节点
            fact_node_ids = []
            for fact in facts:
                node_id = self.add_node(
                    content=fact.get('content', ''),
                    node_type=NodeType.FACT,
                    confidence=fact.get('confidence', 0.8),
                    metadata=fact.get('metadata', {})
                )
                fact_node_ids.append(node_id)
            
            # 添加规则节点
            rule_node_ids = []
            for rule in rules:
                node_id = self.add_node(
                    content=rule.get('content', ''),
                    node_type=NodeType.RULE,
                    confidence=rule.get('confidence', 0.7),
                    reasoning_type=ReasoningType.DEDUCTIVE,
                    metadata=rule.get('metadata', {})
                )
                rule_node_ids.append(node_id)
            
            # 建立事实和规则之间的连接
            for fact_id in fact_node_ids:
                for rule_id in rule_node_ids:
                    self.add_edge(fact_id, rule_id, ReasoningType.DEDUCTIVE)
            
            # 执行推理算法
            self._execute_reasoning_algorithms()
            
            # 构建推理路径
            self._build_reasoning_paths()
            
            # 记录性能指标
            build_time = time.time() - start_time
            metrics = PerformanceMetrics(
                build_time=build_time,
                optimization_time=0.0,
                validation_time=0.0,
                node_count=len(self.nodes),
                edge_count=len(self.edges),
                path_count=len(self.paths),
                average_confidence=self._calculate_average_confidence()
            )
            self.performance_history.append(metrics)
            
            self.logger.info(f"推理链构建完成，耗时: {build_time:.2f}秒")
            return "推理链构建成功"
            
        except Exception as e:
            self.logger.error(f"推理链构建失败: {str(e)}")
            raise
    
    def _execute_reasoning_algorithms(self):
        """执行推理算法"""
        for reasoning_type, algorithm in self.reasoning_algorithms.items():
            try:
                algorithm()
            except Exception as e:
                self.logger.warning(f"推理算法 {reasoning_type.value} 执行失败: {str(e)}")
    
    def _deductive_reasoning(self):
        """演绎推理"""
        # 从一般规则推导具体结论
        rule_nodes = [node for node in self.nodes.values() 
                     if node.node_type == NodeType.RULE]
        
        for rule_node in rule_nodes:
            # 查找可应用的事实
            applicable_facts = self._find_applicable_facts(rule_node)
            
            for fact_node in applicable_facts:
                conclusion = self._apply_deductive_rule(rule_node, fact_node)
                if conclusion:
                    conclusion_id = self.add_node(
                        content=conclusion,
                        node_type=NodeType.CONCLUSION,
                        confidence=min(rule_node.confidence, fact_node.confidence),
                        reasoning_type=ReasoningType.DEDUCTIVE
                    )
                    self.add_edge(fact_node.id, conclusion_id, ReasoningType.DEDUCTIVE)
                    self.add_edge(rule_node.id, conclusion_id, ReasoningType.DEDUCTIVE)
    
    def _inductive_reasoning(self):
        """归纳推理"""
        # 从具体事例归纳一般规律
        fact_groups = self._group_related_facts()
        
        for group in fact_groups:
            if len(group) >= 2:  # 至少需要2个事实进行归纳
                hypothesis = self._induce_general_rule(group)
                if hypothesis:
                    hypothesis_id = self.add_node(
                        content=hypothesis,
                        node_type=NodeType.HYPOTHESIS,
                        confidence=min([f.confidence for f in group]) * 0.8,
                        reasoning_type=ReasoningType.INDUCTIVE
                    )
                    
                    for fact in group:
                        self.add_edge(fact.id, hypothesis_id, ReasoningType.INDUCTIVE)
    
    def _abductive_reasoning(self):
        """溯因推理"""
        # 根据观察结果推断最可能的解释
        conclusion_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.CONCLUSION]
        
        for conclusion in conclusion_nodes:
            # 查找可能的解释
            explanations = self._find_explanations(conclusion)
            
            for explanation in explanations:
                explanation_id = self.add_node(
                    content=explanation,
                    node_type=NodeType.HYPOTHESIS,
                    confidence=conclusion.confidence * 0.7,
                    reasoning_type=ReasoningType.ABDUCTIVE
                )
                self.add_edge(conclusion.id, explanation_id, ReasoningType.ABDUCTIVE)
    
    def _analogical_reasoning(self):
        """类比推理"""
        # 基于相似性进行推理
        similar_pairs = self._find_similar_pairs()
        
        for source, target in similar_pairs:
            analogical_conclusion = self._derive_analogical_conclusion(source, target)
            if analogical_conclusion:
                conclusion_id = self.add_node(
                    content=analogical_conclusion,
                    node_type=NodeType.CONCLUSION,
                    confidence=min(source.confidence, target.confidence) * 0.6,
                    reasoning_type=ReasoningType.ANALOGICAL
                )
                self.add_edge(source.id, conclusion_id, ReasoningType.ANALOGICAL)
                self.add_edge(target.id, conclusion_id, ReasoningType.ANALOGICAL)
    
    def _causal_reasoning(self):
        """因果推理"""
        # 识别因果关系
        causal_relations = self._identify_causal_relations()
        
        for cause, effect in causal_relations:
            causal_node_id = self.add_node(
                content=f"{cause.content} 导致 {effect.content}",
                node_type=NodeType.CONCLUSION,
                confidence=cause.confidence * effect.confidence * 0.9,
                reasoning_type=ReasoningType.CAUSAL
            )
            self.add_edge(cause.id, causal_node_id, ReasoningType.CAUSAL)
            self.add_edge(effect.id, causal_node_id, ReasoningType.CAUSAL)
    
    def _hypothetical_reasoning(self):
        """假设推理"""
        # 基于假设进行推理
        hypotheses = [node for node in self.nodes.values() 
                     if node.node_type == NodeType.HYPOTHESIS]
        
        for hypothesis in hypotheses:
            consequences = self._derive_consequences(hypothesis)
            
            for consequence in consequences:
                consequence_id = self.add_node(
                    content=consequence,
                    node_type=NodeType.CONCLUSION,
                    confidence=hypothesis.confidence * 0.8,
                    reasoning_type=ReasoningType.HYPOTHETICAL
                )
                self.add_edge(hypothesis.id, consequence_id, ReasoningType.HYPOTHETICAL)
    
    def _build_reasoning_paths(self):
        """构建推理路径"""
        try:
            # 查找所有从事实到结论的路径
            source_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.FACT]
            target_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.CONCLUSION]
            
            path_count = 0
            
            for source in source_nodes:
                for target in target_nodes:
                    try:
                        # 查找所有简单路径
                        paths = list(nx.all_simple_paths(self.graph, source.id, target.id, cutoff=5))
                        
                        for path in paths:
                            if len(path) > 1:
                                path_id = self._create_path_from_nodes(path)
                                path_count += 1
                    except nx.NetworkXNoPath:
                        continue
            
            self.logger.info(f"构建了 {path_count} 条推理路径")
            
        except Exception as e:
            self.logger.error(f"构建推理路径失败: {str(e)}")
    
    def optimize_reasoning_paths(self, algorithm: str = 'balanced') -> List[str]:
        """优化推理路径"""
        start_time = time.time()
        
        try:
            if algorithm not in self.optimization_algorithms:
                raise ValueError(f"未知的优化算法: {algorithm}")
            
            optimized_paths = self.optimization_algorithms[algorithm]()
            
            optimization_time = time.time() - start_time
            
            # 更新性能指标
            if self.performance_history:
                self.performance_history[-1].optimization_time = optimization_time
            
            self.logger.info(f"推理路径优化完成，使用算法: {algorithm}")
            return optimized_paths
            
        except Exception as e:
            self.logger.error(f"推理路径优化失败: {str(e)}")
            raise
    
    def _shortest_path_optimization(self) -> List[str]:
        """最短路径优化"""
        try:
            # 找到所有最短路径
            source_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.FACT]
            target_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.CONCLUSION]
            
            optimized_paths = []
            
            for source in source_nodes:
                for target in target_nodes:
                    try:
                        path = nx.shortest_path(self.graph, source.id, target.id, weight='weight')
                        if len(path) > 1:  # 确保路径有效
                            path_id = self._create_path_from_nodes(path)
                            optimized_paths.append(path_id)
                    except nx.NetworkXNoPath:
                        continue
            
            return optimized_paths
            
        except Exception as e:
            self.logger.error(f"最短路径优化失败: {str(e)}")
            return []
    
    def _highest_confidence_optimization(self) -> List[str]:
        """最高置信度优化"""
        try:
            # 基于置信度权重计算最优路径
            source_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.FACT]
            target_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.CONCLUSION]
            
            optimized_paths = []
            
            for source in source_nodes:
                for target in target_nodes:
                    try:
                        # 使用负置信度作为权重，寻找最高置信度路径
                        path = nx.shortest_path(self.graph, source.id, target.id, 
                                              weight=lambda u, v, d: 1 - d.get('confidence', 0.5))
                        
                        if len(path) > 1:
                            path_id = self._create_path_from_nodes(path)
                            optimized_paths.append(path_id)
                    except nx.NetworkXNoPath:
                        continue
            
            return optimized_paths
            
        except Exception as e:
            self.logger.error(f"最高置信度优化失败: {str(e)}")
            return []
    
    def _balanced_optimization(self) -> List[str]:
        """平衡优化"""
        try:
            # 综合考虑路径长度和置信度
            source_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.FACT]
            target_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.CONCLUSION]
            
            optimized_paths = []
            path_scores = []
            
            for source in source_nodes:
                for target in target_nodes:
                    try:
                        # 获取所有简单路径
                        paths = list(nx.all_simple_paths(self.graph, source.id, target.id, cutoff=5))
                        
                        for path in paths:
                            if len(path) > 1:
                                # 计算综合得分
                                score = self._calculate_path_score(path)
                                path_scores.append((path, score))
                    except nx.NetworkXNoPath:
                        continue
            
            # 按得分排序，选择最优路径
            path_scores.sort(key=lambda x: x[1], reverse=True)
            
            for path, _ in path_scores[:10]:  # 选择前10个最优路径
                path_id = self._create_path_from_nodes(path)
                optimized_paths.append(path_id)
            
            return optimized_paths
            
        except Exception as e:
            self.logger.error(f"平衡优化失败: {str(e)}")
            return []
    
    def _multi_criteria_optimization(self) -> List[str]:
        """多准则优化"""
        try:
            # 使用多准则决策分析优化路径
            source_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.FACT]
            target_nodes = [node for node in self.nodes.values() 
                          if node.node_type == NodeType.CONCLUSION]
            
            optimized_paths = []
            
            for source in source_nodes:
                for target in target_nodes:
                    try:
                        # 获取所有路径
                        paths = list(nx.all_simple_paths(self.graph, source.id, target.id, cutoff=6))
                        
                        for path in paths:
                            if len(path) > 1:
                                # 多准则评估
                                criteria_score = self._evaluate_multi_criteria(path)
                                
                                if criteria_score > 0.5:  # 阈值筛选
                                    path_id = self._create_path_from_nodes(path)
                                    optimized_paths.append(path_id)
                    except nx.NetworkXNoPath:
                        continue
            
            return optimized_paths
            
        except Exception as e:
            self.logger.error(f"多准则优化失败: {str(e)}")
            return []
    
    def validate_reasoning_chain(self) -> Dict[str, Any]:
        """验证推理链"""
        start_time = time.time()
        
        try:
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'statistics': {},
                'confidence_analysis': {}
            }
            
            # 检查图结构完整性
            structural_issues = self._check_structural_integrity()
            validation_results['errors'].extend(structural_issues['errors'])
            validation_results['warnings'].extend(structural_issues['warnings'])
            
            # 检查推理逻辑一致性
            logical_issues = self._check_logical_consistency()
            validation_results['errors'].extend(logical_issues['errors'])
            validation_results['warnings'].extend(logical_issues['warnings'])
            
            # 检查置信度合理性
            confidence_issues = self._check_confidence_validity()
            validation_results['errors'].extend(confidence_issues['errors'])
            validation_results['warnings'].extend(confidence_issues['warnings'])
            
            # 统计分析
            validation_results['statistics'] = self._generate_statistics()
            
            # 置信度分析
            validation_results['confidence_analysis'] = self._analyze_confidence_distribution()
            
            # 确定整体有效性
            validation_results['is_valid'] = len(validation_results['errors']) == 0
            
            validation_time = time.time() - start_time
            
            # 更新性能指标
            if self.performance_history:
                self.performance_history[-1].validation_time = validation_time
            
            self.logger.info(f"推理链验证完成，有效性: {validation_results['is_valid']}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"推理链验证失败: {str(e)}")
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'statistics': {},
                'confidence_analysis': {}
            }
    
    def visualize_reasoning_chain(self, output_path: str = None, 
                                 figsize: Tuple[int, int] = (15, 10)) -> str:
        """可视化推理链"""
        try:
            plt.figure(figsize=figsize)
            
            # 设置布局
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
            
            # 定义节点颜色映射
            node_colors = {
                NodeType.FACT: '#4CAF50',      # 绿色
                NodeType.HYPOTHESIS: '#FF9800', # 橙色
                NodeType.RULE: '#2196F3',      # 蓝色
                NodeType.CONCLUSION: '#9C27B0', # 紫色
                NodeType.EVIDENCE: '#F44336',   # 红色
                NodeType.QUESTION: '#607D8B'    # 灰蓝色
            }
            
            # 绘制节点
            for node_type, color in node_colors.items():
                node_ids = [node_id for node_id, node in self.nodes.items() 
                          if node.node_type == node_type]
                if node_ids:
                    nx.draw_networkx_nodes(self.graph, pos, 
                                         nodelist=node_ids,
                                         node_color=color,
                                         node_size=1000,
                                         alpha=0.8)
            
            # 绘制边
            edge_colors = {
                ReasoningType.DEDUCTIVE: '#4CAF50',
                ReasoningType.INDUCTIVE: '#FF9800',
                ReasoningType.ABDUCTIVE: '#2196F3',
                ReasoningType.ANALOGICAL: '#9C27B0',
                ReasoningType.CAUSAL: '#F44336',
                ReasoningType.HYPOTHETICAL: '#607D8B'
            }
            
            for reasoning_type, color in edge_colors.items():
                edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                        if d.get('reasoning_type') == reasoning_type]
                if edges:
                    nx.draw_networkx_edges(self.graph, pos,
                                         edgelist=edges,
                                         edge_color=color,
                                         width=2,
                                         alpha=0.6,
                                         arrows=True,
                                         arrowsize=20)
            
            # 添加标签
            labels = {node_id: f"{node.content[:20]}..." if len(node.content) > 20 
                     else node.content for node_id, node in self.nodes.items()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            # 创建图例
            legend_elements = []
            for node_type, color in node_colors.items():
                legend_elements.append(mpatches.Patch(color=color, label=f'节点-{node_type.value}'))
            
            for reasoning_type, color in edge_colors.items():
                legend_elements.append(mpatches.Patch(color=color, label=f'边-{reasoning_type.value}'))
            
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            plt.title(f'推理链可视化 - {self.name}', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # 保存图像
            if output_path is None:
                output_path = f"reasoning_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"推理链可视化保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"推理链可视化失败: {str(e)}")
            raise
    
    def explain_reasoning_chain(self, path_id: str = None) -> Dict[str, Any]:
        """解释推理链"""
        try:
            explanation = {
                'chain_overview': {},
                'reasoning_process': [],
                'confidence_breakdown': {},
                'alternative_paths': [],
                'recommendations': []
            }
            
            # 链概览
            explanation['chain_overview'] = {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'node_types': {node_type.value: count 
                              for node_type, count in self._count_nodes_by_type().items()},
                'reasoning_types': {reasoning_type.value: count 
                                  for reasoning_type, count in self._count_edges_by_type().items()}
            }
            
            # 推理过程
            if path_id and path_id in self.paths:
                path = self.paths[path_id]
                explanation['reasoning_process'] = self._explain_path(path)
            else:
                # 选择最重要的路径进行解释
                important_paths = self._get_most_important_paths()
                for path in important_paths[:3]:  # 最多解释3条路径
                    explanation['reasoning_process'].append(self._explain_path(path))
            
            # 置信度分析
            explanation['confidence_breakdown'] = self._analyze_confidence_breakdown()
            
            # 替代路径
            explanation['alternative_paths'] = self._find_alternative_paths()
            
            # 建议
            explanation['recommendations'] = self._generate_recommendations()
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"推理链解释失败: {str(e)}")
            raise
    
    def create_version(self, description: str = "") -> str:
        """创建版本"""
        try:
            version_id = str(uuid.uuid4())
            version_data = {
                'version_id': version_id,
                'description': description,
                'timestamp': datetime.now().isoformat(),
                'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
                'edges': {edge_id: asdict(edge) for edge_id, edge in self.edges.items()},
                'paths': {path_id: asdict(path) for path_id, path in self.paths.items()},
                'performance_metrics': [asdict(metric) for metric in self.performance_history]
            }
            
            self.versions[version_id] = version_data
            
            self.logger.info(f"创建版本: {version_id}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"创建版本失败: {str(e)}")
            raise
    
    def restore_version(self, version_id: str) -> bool:
        """恢复版本"""
        try:
            if version_id not in self.versions:
                raise ValueError(f"版本不存在: {version_id}")
            
            version_data = self.versions[version_id]
            
            # 恢复数据
            self.nodes = {node_id: ReasoningNode.from_dict(node_data) 
                         for node_id, node_data in version_data['nodes'].items()}
            self.edges = {edge_id: ReasoningEdge(**edge_data) 
                         for edge_id, edge_data in version_data['edges'].items()}
            
            # 重建图结构
            self.graph.clear()
            for node_id in self.nodes:
                self.graph.add_node(node_id, **self.nodes[node_id].to_dict())
            
            for edge_id, edge in self.edges.items():
                self.graph.add_edge(edge.source, edge.target,
                                  edge_id=edge_id,
                                  weight=edge.weight,
                                  confidence=edge.confidence)
            
            # 恢复路径和性能指标
            self.paths = {path_id: ReasoningPath(**path_data) 
                         for path_id, path_data in version_data['paths'].items()}
            
            self.performance_history = [PerformanceMetrics(**metric_data) 
                                      for metric_data in version_data['performance_metrics']]
            
            self.logger.info(f"恢复版本: {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复版本失败: {str(e)}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            if not self.performance_history:
                return {'error': '没有性能数据'}
            
            latest_metrics = self.performance_history[-1]
            
            # 计算趋势
            trends = {}
            if len(self.performance_history) >= 2:
                prev_metrics = self.performance_history[-2]
                trends = {
                    'build_time_trend': latest_metrics.build_time - prev_metrics.build_time,
                    'optimization_time_trend': latest_metrics.optimization_time - prev_metrics.optimization_time,
                    'validation_time_trend': latest_metrics.validation_time - prev_metrics.validation_time,
                    'confidence_trend': latest_metrics.average_confidence - prev_metrics.average_confidence
                }
            
            return {
                'latest_metrics': asdict(latest_metrics),
                'trends': trends,
                'history_length': len(self.performance_history),
                'performance_summary': self._summarize_performance()
            }
            
        except Exception as e:
            self.logger.error(f"获取性能指标失败: {str(e)}")
            return {'error': str(e)}
    
    def export_chain(self, file_path: str) -> bool:
        """导出推理链"""
        try:
            # 自定义JSON编码器处理枚举类型
            def json_serializer(obj):
                if hasattr(obj, 'value'):  # 枚举类型
                    return obj.value
                elif hasattr(obj, '__dict__'):  # 自定义对象
                    result = {}
                    for key, value in obj.__dict__.items():
                        if hasattr(value, 'value'):  # 枚举
                            result[key] = value.value
                        else:
                            result[key] = value
                    return result
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            export_data = {
                'name': self.name,
                'nodes': {node_id: self._serialize_node(node) for node_id, node in self.nodes.items()},
                'edges': {edge_id: self._serialize_edge(edge) for edge_id, edge in self.edges.items()},
                'paths': {path_id: self._serialize_path(path) for path_id, path in self.paths.items()},
                'versions': self._serialize_versions(self.versions),
                'performance_history': [self._serialize_metrics(metric) for metric in self.performance_history],
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"推理链导出至: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"推理链导出失败: {str(e)}")
            return False
    
    def _serialize_node(self, node: ReasoningNode) -> Dict[str, Any]:
        """序列化节点"""
        return {
            'id': node.id,
            'content': node.content,
            'node_type': node.node_type.value,
            'confidence': node.confidence,
            'reasoning_type': node.reasoning_type.value if node.reasoning_type else None,
            'metadata': node.metadata,
            'timestamp': node.timestamp,
            'version': node.version
        }
    
    def _serialize_edge(self, edge: ReasoningEdge) -> Dict[str, Any]:
        """序列化边"""
        return {
            'source': edge.source,
            'target': edge.target,
            'weight': edge.weight,
            'reasoning_type': edge.reasoning_type.value,
            'confidence': edge.confidence,
            'metadata': edge.metadata
        }
    
    def _serialize_path(self, path: ReasoningPath) -> Dict[str, Any]:
        """序列化路径"""
        return {
            'nodes': path.nodes,
            'edges': path.edges,
            'total_confidence': path.total_confidence,
            'reasoning_types': [rt.value for rt in path.reasoning_types],
            'length': path.length,
            'metadata': path.metadata
        }
    
    def _serialize_versions(self, versions: Dict[str, Any]) -> Dict[str, Any]:
        """序列化版本"""
        serialized = {}
        for version_id, version_data in versions.items():
            serialized[version_id] = {
                'version_id': version_data['version_id'],
                'description': version_data['description'],
                'timestamp': version_data['timestamp'],
                'nodes': {node_id: self._serialize_node(ReasoningNode.from_dict(node_data)) 
                         for node_id, node_data in version_data['nodes'].items()},
                'edges': {edge_id: self._serialize_edge(ReasoningEdge(**edge_data)) 
                         for edge_id, edge_data in version_data['edges'].items()},
                'paths': {path_id: self._serialize_path(ReasoningPath(**path_data)) 
                         for path_id, path_data in version_data['paths'].items()},
                'performance_metrics': [self._serialize_metrics(PerformanceMetrics(**metric_data)) 
                                      for metric_data in version_data['performance_metrics']]
            }
        return serialized
    
    def _serialize_metrics(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """序列化性能指标"""
        return {
            'build_time': metrics.build_time,
            'optimization_time': metrics.optimization_time,
            'validation_time': metrics.validation_time,
            'node_count': metrics.node_count,
            'edge_count': metrics.edge_count,
            'path_count': metrics.path_count,
            'average_confidence': metrics.average_confidence,
            'timestamp': metrics.timestamp
        }
    
    def import_chain(self, file_path: str) -> bool:
        """导入推理链"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 恢复基本属性
            self.name = import_data.get('name', self.name)
            
            # 恢复节点
            self.nodes = {}
            for node_id, node_data in import_data['nodes'].items():
                # 转换枚举值
                node_data['node_type'] = NodeType(node_data['node_type'])
                if node_data['reasoning_type']:
                    node_data['reasoning_type'] = ReasoningType(node_data['reasoning_type'])
                
                self.nodes[node_id] = ReasoningNode(**node_data)
            
            # 恢复边
            self.edges = {}
            for edge_id, edge_data in import_data['edges'].items():
                # 转换枚举值
                edge_data['reasoning_type'] = ReasoningType(edge_data['reasoning_type'])
                
                self.edges[edge_id] = ReasoningEdge(**edge_data)
            
            # 重建图结构
            self.graph.clear()
            for node_id in self.nodes:
                self.graph.add_node(node_id, **self.nodes[node_id].to_dict())
            
            for edge_id, edge in self.edges.items():
                self.graph.add_edge(edge.source, edge.target,
                                  edge_id=edge_id,
                                  weight=edge.weight,
                                  confidence=edge.confidence)
            
            # 恢复路径
            self.paths = {}
            for path_id, path_data in import_data['paths'].items():
                # 转换枚举值
                path_data['reasoning_types'] = [ReasoningType(rt) for rt in path_data['reasoning_types']]
                
                self.paths[path_id] = ReasoningPath(**path_data)
            
            # 恢复版本
            self.versions = import_data.get('versions', {})
            
            # 恢复性能历史
            self.performance_history = []
            for metric_data in import_data.get('performance_history', []):
                self.performance_history.append(PerformanceMetrics(**metric_data))
            
            self.logger.info(f"推理链导入成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"推理链导入失败: {str(e)}")
            return False
    
    # 辅助方法
    def _find_applicable_facts(self, rule_node: ReasoningNode) -> List[ReasoningNode]:
        """查找可应用的事实"""
        # 简化的实现，实际应用中需要更复杂的匹配逻辑
        return [node for node in self.nodes.values() 
                if node.node_type == NodeType.FACT and node.confidence > 0.5]
    
    def _apply_deductive_rule(self, rule_node: ReasoningNode, fact_node: ReasoningNode) -> Optional[str]:
        """应用演绎规则"""
        # 简化的规则应用逻辑
        if fact_node.confidence > 0.6 and rule_node.confidence > 0.6:
            return f"基于规则 '{rule_node.content}' 和事实 '{fact_node.content}' 得出结论"
        return None
    
    def _group_related_facts(self) -> List[List[ReasoningNode]]:
        """分组相关事实"""
        # 简化的分组逻辑
        facts = [node for node in self.nodes.values() if node.node_type == NodeType.FACT]
        return [facts[i:i+3] for i in range(0, len(facts), 3)]
    
    def _induce_general_rule(self, facts: List[ReasoningNode]) -> Optional[str]:
        """归纳一般规则"""
        if len(facts) >= 2:
            return f"基于 {len(facts)} 个事实归纳得出一般规律"
        return None
    
    def _find_explanations(self, conclusion: ReasoningNode) -> List[str]:
        """查找解释"""
        return [f"可能的解释: {conclusion.content} 的原因"]
    
    def _find_similar_pairs(self) -> List[Tuple[ReasoningNode, ReasoningNode]]:
        """查找相似对"""
        nodes = list(self.nodes.values())
        pairs = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if abs(nodes[i].confidence - nodes[j].confidence) < 0.2:
                    pairs.append((nodes[i], nodes[j]))
        return pairs[:5]  # 限制数量
    
    def _derive_analogical_conclusion(self, source: ReasoningNode, target: ReasoningNode) -> Optional[str]:
        """推导类比结论"""
        return f"基于类比推理: {source.content} 与 {target.content} 相似"
    
    def _identify_causal_relations(self) -> List[Tuple[ReasoningNode, ReasoningNode]]:
        """识别因果关系"""
        # 简化的因果关系识别
        facts = [node for node in self.nodes.values() if node.node_type == NodeType.FACT]
        conclusions = [node for node in self.nodes.values() if node.node_type == NodeType.CONCLUSION]
        
        relations = []
        for fact in facts:
            for conclusion in conclusions:
                if fact.confidence > 0.7 and conclusion.confidence > 0.6:
                    relations.append((fact, conclusion))
        return relations[:3]  # 限制数量
    
    def _derive_consequences(self, hypothesis: ReasoningNode) -> List[str]:
        """推导后果"""
        return [f"假设 '{hypothesis.content}' 的可能后果"]
    
    def _create_path_from_nodes(self, node_ids: List[str]) -> str:
        """从节点创建路径"""
        path_id = str(uuid.uuid4())
        
        # 计算总置信度
        total_confidence = np.mean([self.nodes[node_id].confidence for node_id in node_ids])
        
        # 收集推理类型
        reasoning_types = []
        for i in range(len(node_ids) - 1):
            source_id, target_id = node_ids[i], node_ids[i+1]
            for edge in self.edges.values():
                if edge.source == source_id and edge.target == target_id:
                    reasoning_types.append(edge.reasoning_type)
                    break
        
        path = ReasoningPath(
            nodes=node_ids,
            edges=[],  # 简化实现
            total_confidence=total_confidence,
            reasoning_types=reasoning_types,
            length=len(node_ids)
        )
        
        self.paths[path_id] = path
        return path_id
    
    def _calculate_path_score(self, path: List[str]) -> float:
        """计算路径得分"""
        if len(path) < 2:
            return 0.0
        
        # 综合考虑置信度和路径长度
        confidence_score = np.mean([self.nodes[node_id].confidence for node_id in path])
        length_score = 1.0 / len(path)  # 路径越短得分越高
        
        return confidence_score * 0.7 + length_score * 0.3
    
    def _evaluate_multi_criteria(self, path: List[str]) -> float:
        """多准则评估"""
        if len(path) < 2:
            return 0.0
        
        criteria_scores = {
            'confidence': np.mean([self.nodes[node_id].confidence for node_id in path]),
            'diversity': len(set(node.node_type for node_id in path 
                               for node in [self.nodes[node_id]])) / len(NodeType),
            'efficiency': 1.0 / len(path),
            'consistency': self._calculate_consistency_score(path)
        }
        
        # 加权平均
        weights = {'confidence': 0.4, 'diversity': 0.2, 'efficiency': 0.2, 'consistency': 0.2}
        
        return sum(score * weights[criterion] for criterion, score in criteria_scores.items())
    
    def _calculate_consistency_score(self, path: List[str]) -> float:
        """计算一致性得分"""
        # 检查路径中节点类型的一致性
        node_types = [self.nodes[node_id].node_type for node_id in path]
        
        # 期望的序列: FACT -> RULE -> CONCLUSION
        expected_sequence = [NodeType.FACT, NodeType.RULE, NodeType.CONCLUSION]
        
        matches = 0
        for i, node_type in enumerate(node_types):
            if i < len(expected_sequence) and node_type == expected_sequence[i]:
                matches += 1
        
        return matches / len(expected_sequence) if expected_sequence else 0.0
    
    def _check_structural_integrity(self) -> Dict[str, List[str]]:
        """检查结构完整性"""
        errors = []
        warnings = []
        
        # 检查孤立节点
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            warnings.append(f"发现 {len(isolated_nodes)} 个孤立节点")
        
        # 检查连通性
        if not nx.is_weakly_connected(self.graph):
            warnings.append("图不是弱连通的")
        
        # 检查是否有环
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                warnings.append(f"发现 {len(cycles)} 个环")
        except:
            pass
        
        return {'errors': errors, 'warnings': warnings}
    
    def _check_logical_consistency(self) -> Dict[str, List[str]]:
        """检查逻辑一致性"""
        errors = []
        warnings = []
        
        # 检查置信度异常值
        confidences = [node.confidence for node in self.nodes.values()]
        if confidences:
            max_conf = max(confidences)
            min_conf = min(confidences)
            
            if max_conf > 1.0 or min_conf < 0.0:
                errors.append("发现异常置信度值")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _check_confidence_validity(self) -> Dict[str, List[str]]:
        """检查置信度有效性"""
        errors = []
        warnings = []
        
        for node_id, node in self.nodes.items():
            if not 0.0 <= node.confidence <= 1.0:
                errors.append(f"节点 {node_id} 置信度超出范围: {node.confidence}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """生成统计信息"""
        return {
            'node_count_by_type': self._count_nodes_by_type(),
            'edge_count_by_type': self._count_edges_by_type(),
            'average_confidence': self._calculate_average_confidence(),
            'graph_density': nx.density(self.graph),
            'path_count': len(self.paths)
        }
    
    def _count_nodes_by_type(self) -> Dict[NodeType, int]:
        """按类型统计节点"""
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node.node_type] += 1
        return counts
    
    def _count_edges_by_type(self) -> Dict[ReasoningType, int]:
        """按类型统计边"""
        counts = defaultdict(int)
        for edge in self.edges.values():
            counts[edge.reasoning_type] += 1
        return counts
    
    def _calculate_average_confidence(self) -> float:
        """计算平均置信度"""
        if not self.nodes:
            return 0.0
        return np.mean([node.confidence for node in self.nodes.values()])
    
    def _analyze_confidence_distribution(self) -> Dict[str, Any]:
        """分析置信度分布"""
        confidences = [node.confidence for node in self.nodes.values()]
        
        if not confidences:
            return {}
        
        return {
            'min': min(confidences),
            'max': max(confidences),
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'distribution': self._categorize_confidence(confidences)
        }
    
    def _categorize_confidence(self, confidences: List[float]) -> Dict[str, int]:
        """分类置信度"""
        categories = defaultdict(int)
        for confidence in confidences:
            for level in ConfidenceLevel:
                if level.value[0] <= confidence < level.value[1]:
                    categories[level.name] += 1
                    break
            else:
                categories['VERY_HIGH'] += 1  # 处理 1.0 的情况
        
        return dict(categories)
    
    def _explain_path(self, path: ReasoningPath) -> Dict[str, Any]:
        """解释路径"""
        explanation = {
            'path_length': path.length,
            'total_confidence': path.total_confidence,
            'reasoning_steps': [],
            'step_details': []
        }
        
        # 解释推理步骤
        for i, node_id in enumerate(path.nodes):
            node = self.nodes[node_id]
            step = {
                'step_number': i + 1,
                'node_type': node.node_type.value,
                'content': node.content,
                'confidence': node.confidence
            }
            explanation['step_details'].append(step)
        
        # 推理类型分析
        reasoning_type_counts = defaultdict(int)
        for reasoning_type in path.reasoning_types:
            reasoning_type_counts[reasoning_type.value] += 1
        explanation['reasoning_types'] = dict(reasoning_type_counts)
        
        return explanation
    
    def _get_most_important_paths(self) -> List[ReasoningPath]:
        """获取最重要的路径"""
        if not self.paths:
            return []
        
        # 按置信度排序
        sorted_paths = sorted(self.paths.values(), 
                            key=lambda p: p.total_confidence, 
                            reverse=True)
        return sorted_paths[:5]
    
    def _analyze_confidence_breakdown(self) -> Dict[str, Any]:
        """分析置信度分解"""
        return {
            'high_confidence_nodes': len([n for n in self.nodes.values() if n.confidence > 0.8]),
            'medium_confidence_nodes': len([n for n in self.nodes.values() if 0.5 <= n.confidence <= 0.8]),
            'low_confidence_nodes': len([n for n in self.nodes.values() if n.confidence < 0.5]),
            'confidence_trends': self._analyze_confidence_trends()
        }
    
    def _analyze_confidence_trends(self) -> Dict[str, float]:
        """分析置信度趋势"""
        if len(self.performance_history) < 2:
            return {}
        
        latest = self.performance_history[-1]
        previous = self.performance_history[-2]
        
        return {
            'confidence_change': latest.average_confidence - previous.average_confidence,
            'node_count_change': latest.node_count - previous.node_count,
            'edge_count_change': latest.edge_count - previous.edge_count
        }
    
    def _find_alternative_paths(self) -> List[Dict[str, Any]]:
        """查找替代路径"""
        alternatives = []
        
        # 查找不同推理类型的路径
        reasoning_type_paths = defaultdict(list)
        for path_id, path in self.paths.items():
            primary_type = path.reasoning_types[0] if path.reasoning_types else None
            if primary_type:
                reasoning_type_paths[primary_type].append((path_id, path))
        
        for reasoning_type, paths in reasoning_type_paths.items():
            if len(paths) > 1:
                alternatives.append({
                    'reasoning_type': reasoning_type.value,
                    'alternative_count': len(paths),
                    'confidence_range': {
                        'min': min(p[1].total_confidence for p in paths),
                        'max': max(p[1].total_confidence for p in paths)
                    }
                })
        
        return alternatives
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于统计数据的建议
        stats = self._generate_statistics()
        
        if stats['graph_density'] < 0.1:
            recommendations.append("图密度较低，建议增加更多连接以提高推理完整性")
        
        if stats['average_confidence'] < 0.6:
            recommendations.append("平均置信度较低，建议验证输入数据的可靠性")
        
        if len(self.paths) == 0:
            recommendations.append("未发现推理路径，建议检查节点和边的连接关系")
        
        # 基于性能的建议
        if self.performance_history:
            latest = self.performance_history[-1]
            if latest.build_time > 5.0:
                recommendations.append("构建时间较长，考虑优化推理算法")
        
        return recommendations
    
    def _summarize_performance(self) -> Dict[str, Any]:
        """总结性能"""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-5:]  # 最近5次
        
        return {
            'avg_build_time': np.mean([m.build_time for m in recent_metrics]),
            'avg_optimization_time': np.mean([m.optimization_time for m in recent_metrics]),
            'avg_validation_time': np.mean([m.validation_time for m in recent_metrics]),
            'avg_confidence': np.mean([m.average_confidence for m in recent_metrics]),
            'performance_stability': np.std([m.build_time for m in recent_metrics])
        }


# 使用示例和测试函数
def example_usage():
    """使用示例"""
    # 创建推理链管理器
    manager = ReasoningChainManager("示例推理链")
    
    # 添加事实
    facts = [
        {'content': '今天下雨', 'confidence': 0.9},
        {'content': '地面湿了', 'confidence': 0.85},
        {'content': '人们带伞', 'confidence': 0.7}
    ]
    
    # 添加规则
    rules = [
        {'content': '下雨导致地面湿', 'confidence': 0.8},
        {'content': '下雨人们会带伞', 'confidence': 0.75}
    ]
    
    # 构建推理链
    manager.build_reasoning_chain(facts, rules)
    
    # 优化路径
    optimized_paths = manager.optimize_reasoning_paths('balanced')
    print(f"优化后的路径数量: {len(optimized_paths)}")
    
    # 验证推理链
    validation_results = manager.validate_reasoning_chain()
    print(f"推理链有效性: {validation_results['is_valid']}")
    
    # 可视化
    viz_path = manager.visualize_reasoning_chain()
    print(f"可视化保存至: {viz_path}")
    
    # 解释推理链
    explanation = manager.explain_reasoning_chain()
    print(f"推理链解释: {json.dumps(explanation, ensure_ascii=False, indent=2)}")
    
    # 创建版本
    version_id = manager.create_version("初始版本")
    print(f"创建版本: {version_id}")
    
    # 获取性能指标
    performance = manager.get_performance_metrics()
    print(f"性能指标: {json.dumps(performance, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行示例
    example_usage()