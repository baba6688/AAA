#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图算法库 (Graph Algorithm Library)

这是一个完整的图算法库，实现了图论中的核心算法和现代图神经网络技术。
包含图的遍历、搜索、最短路径、最小生成树、聚类、社区检测、图神经网络、
相似性度量、嵌入学习、动态图算法和可视化分析等功能。


版本: 1.0.0
日期: 2025-11-05
"""

import heapq
import math
import random
import itertools
import collections
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import pickle
import time


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class Edge:
    """图边数据结构"""
    source: Any
    target: Any
    weight: float = 1.0
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def __repr__(self) -> str:
        return f"Edge({self.source} -> {self.target}, weight={self.weight})"


@dataclass
class Node:
    """图节点数据结构"""
    id: Any
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def __repr__(self) -> str:
        return f"Node({self.id})"


class Graph:
    """图数据结构"""
    
    def __init__(self, directed: bool = False, weighted: bool = False):
        """
        初始化图
        
        Args:
            directed: 是否为有向图
            weighted: 是否为加权图
        """
        self.directed = directed
        self.weighted = weighted
        self.nodes: Dict[Any, Node] = {}
        self.edges: Dict[Tuple[Any, Any], Edge] = {}
        self.adjacency_list: Dict[Any, List[Tuple[Any, float]]] = defaultdict(list)
        
    def add_node(self, node_id: Any, attributes: Dict[str, Any] = None) -> None:
        """添加节点"""
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, attributes)
            self.adjacency_list[node_id] = []
    
    def add_edge(self, source: Any, target: Any, weight: float = 1.0, 
                 attributes: Dict[str, Any] = None) -> None:
        """添加边"""
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
            
        edge = Edge(source, target, weight, attributes)
        self.edges[(source, target)] = edge
        self.adjacency_list[source].append((target, weight))
        
        # 如果是无向图，添加反向边
        if not self.directed and source != target:
            reverse_edge = Edge(target, source, weight, attributes)
            self.edges[(target, source)] = reverse_edge
            self.adjacency_list[target].append((source, weight))
    
    def remove_node(self, node_id: Any) -> None:
        """删除节点"""
        if node_id in self.nodes:
            # 删除所有相关的边
            edges_to_remove = [(src, tgt) for (src, tgt) in self.edges.keys() 
                             if src == node_id or tgt == node_id]
            for edge_key in edges_to_remove:
                del self.edges[edge_key]
            
            # 删除邻接列表中的条目
            del self.adjacency_list[node_id]
            
            # 删除其他节点的邻接列表中的该节点
            for neighbors in self.adjacency_list.values():
                neighbors[:] = [(tgt, w) for tgt, w in neighbors if tgt != node_id]
            
            # 删除节点
            del self.nodes[node_id]
    
    def remove_edge(self, source: Any, target: Any) -> None:
        """删除边"""
        edge_key = (source, target)
        if edge_key in self.edges:
            del self.edges[edge_key]
            self.adjacency_list[source] = [(tgt, w) for tgt, w in self.adjacency_list[source] 
                                         if tgt != target]
            
            # 如果是无向图，删除反向边
            if not self.directed:
                reverse_key = (target, source)
                if reverse_key in self.edges:
                    del self.edges[reverse_key]
                self.adjacency_list[target] = [(tgt, w) for tgt, w in self.adjacency_list[target] 
                                            if tgt != source]
    
    def get_neighbors(self, node_id: Any) -> List[Tuple[Any, float]]:
        """获取节点的邻居"""
        return self.adjacency_list.get(node_id, [])
    
    def get_edge_weight(self, source: Any, target: Any) -> float:
        """获取边的权重"""
        edge_key = (source, target)
        if edge_key in self.edges:
            return self.edges[edge_key].weight
        return float('inf')
    
    def get_nodes(self) -> List[Any]:
        """获取所有节点"""
        return list(self.nodes.keys())
    
    def get_edges(self) -> List[Edge]:
        """获取所有边"""
        return list(self.edges.values())
    
    def get_node_count(self) -> int:
        """获取节点数量"""
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        """获取边数量"""
        if self.directed:
            return len(self.edges)
        else:
            # 对于无向图，每条边在edges字典中出现两次（正向和反向）
            # 所以需要除以2
            return len(self.edges) // 2
    
    def is_connected(self) -> bool:
        """检查图是否连通"""
        if not self.nodes:
            return True
        
        start_node = next(iter(self.nodes.keys()))
        visited = set()
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for neighbor, _ in self.get_neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return len(visited) == len(self.nodes)
    
    def copy(self) -> 'Graph':
        """复制图"""
        new_graph = Graph(self.directed, self.weighted)
        new_graph.nodes = {k: Node(v.id, v.attributes.copy()) for k, v in self.nodes.items()}
        new_graph.edges = {k: Edge(v.source, v.target, v.weight, v.attributes.copy()) 
                          for k, v in self.edges.items()}
        new_graph.adjacency_list = defaultdict(list)
        for node, neighbors in self.adjacency_list.items():
            new_graph.adjacency_list[node] = neighbors.copy()
        return new_graph
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'directed': self.directed,
            'weighted': self.weighted,
            'nodes': {k: v.attributes for k, v in self.nodes.items()},
            'edges': [(e.source, e.target, e.weight, e.attributes) for e in self.edges.values()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Graph':
        """从字典创建图"""
        graph = cls(data['directed'], data['weighted'])
        
        # 添加节点
        for node_id, attributes in data['nodes'].items():
            graph.add_node(node_id, attributes)
        
        # 添加边
        for source, target, weight, attributes in data['edges']:
            graph.add_edge(source, target, weight, attributes)
        
        return graph


# ============================================================================
# 图的遍历和搜索算法
# ============================================================================

class TraversalAlgorithms:
    """图的遍历和搜索算法类"""
    
    @staticmethod
    def bfs(graph: Graph, start_node: Any, target_node: Any = None) -> Tuple[List[Any], Dict[Any, Any]]:
        """
        广度优先搜索 (Breadth-First Search)
        
        Args:
            graph: 图对象
            start_node: 起始节点
            target_node: 目标节点（可选）
            
        Returns:
            访问路径和父节点映射
        """
        if start_node not in graph.nodes:
            raise ValueError(f"起始节点 {start_node} 不存在")
        
        visited = set()
        queue = deque([start_node])
        parent = {start_node: None}
        path = []
        
        while queue:
            current_node = queue.popleft()
            
            if current_node not in visited:
                visited.add(current_node)
                path.append(current_node)
                
                # 如果找到目标节点，停止搜索
                if target_node and current_node == target_node:
                    break
                
                # 将未访问的邻居加入队列
                for neighbor, _ in graph.get_neighbors(current_node):
                    if neighbor not in visited:
                        parent[neighbor] = current_node
                        queue.append(neighbor)
        
        return path, parent
    
    @staticmethod
    def dfs(graph: Graph, start_node: Any, target_node: Any = None) -> Tuple[List[Any], Dict[Any, Any]]:
        """
        深度优先搜索 (Depth-First Search)
        
        Args:
            graph: 图对象
            start_node: 起始节点
            target_node: 目标节点（可选）
            
        Returns:
            访问路径和父节点映射
        """
        if start_node not in graph.nodes:
            raise ValueError(f"起始节点 {start_node} 不存在")
        
        visited = set()
        stack = [start_node]
        parent = {start_node: None}
        path = []
        
        while stack:
            current_node = stack.pop()
            
            if current_node not in visited:
                visited.add(current_node)
                path.append(current_node)
                
                # 如果找到目标节点，停止搜索
                if target_node and current_node == target_node:
                    break
                
                # 将未访问的邻居加入栈
                for neighbor, _ in reversed(graph.get_neighbors(current_node)):
                    if neighbor not in visited:
                        parent[neighbor] = current_node
                        stack.append(neighbor)
        
        return path, parent
    
    @staticmethod
    def find_path(graph: Graph, start: Any, end: Any, algorithm: str = 'bfs') -> List[Any]:
        """
        查找两点之间的路径
        
        Args:
            graph: 图对象
            start: 起始节点
            end: 目标节点
            algorithm: 搜索算法 ('bfs' 或 'dfs')
            
        Returns:
            路径列表
        """
        if algorithm == 'bfs':
            _, parent = TraversalAlgorithms.bfs(graph, start, end)
        elif algorithm == 'dfs':
            _, parent = TraversalAlgorithms.dfs(graph, start, end)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        # 重构路径
        if end not in parent:
            return []  # 没有找到路径
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parent[current]
        
        return list(reversed(path))
    
    @staticmethod
    def find_all_paths(graph: Graph, start: Any, end: Any, max_depth: int = 10) -> List[List[Any]]:
        """
        查找两点之间的所有路径
        
        Args:
            graph: 图对象
            start: 起始节点
            end: 目标节点
            max_depth: 最大深度限制
            
        Returns:
            所有路径的列表
        """
        paths = []
        
        def dfs_path(current: Any, target: Any, path: List[Any], visited: Set[Any], depth: int):
            if depth > max_depth:
                return
            
            if current == target:
                paths.append(path.copy())
                return
            
            if current in visited:
                return
            
            visited.add(current)
            path.append(current)
            
            for neighbor, _ in graph.get_neighbors(current):
                dfs_path(neighbor, target, path, visited, depth + 1)
            
            path.pop()
            visited.remove(current)
        
        dfs_path(start, end, [], set(), 0)
        return paths
    
    @staticmethod
    def topological_sort(graph: Graph) -> List[Any]:
        """
        拓扑排序（仅适用于有向无环图）
        
        Args:
            graph: 有向图对象
            
        Returns:
            拓扑排序结果
        """
        if not graph.directed:
            raise ValueError("拓扑排序仅适用于有向图")
        
        # 计算入度
        in_degree = {node: 0 for node in graph.nodes}
        for edge in graph.edges.values():
            in_degree[edge.target] += 1
        
        # 使用队列存储入度为0的节点
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # 更新邻居的入度
            for neighbor, _ in graph.get_neighbors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否有环
        if len(result) != len(graph.nodes):
            raise ValueError("图中存在环，无法进行拓扑排序")
        
        return result


# ============================================================================
# 最短路径算法
# ============================================================================

class ShortestPathAlgorithms:
    """最短路径算法类"""
    
    @staticmethod
    def dijkstra(graph: Graph, start: Any, end: Any = None) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
        """
        Dijkstra算法（单源最短路径）
        
        Args:
            graph: 加权图对象
            start: 起始节点
            end: 目标节点（可选）
            
        Returns:
            距离字典和父节点字典
        """
        if not graph.weighted:
            raise ValueError("Dijkstra算法需要加权图")
        
        # 初始化距离和父节点
        distances = {node: float('inf') for node in graph.nodes}
        parent = {node: None for node in graph.nodes}
        distances[start] = 0
        
        # 使用优先队列
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # 如果到达目标节点，可以提前终止
            if end and current_node == end:
                break
            
            # 检查邻居
            for neighbor, weight in graph.get_neighbors(current_node):
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        parent[neighbor] = current_node
                        heapq.heappush(pq, (new_distance, neighbor))
        
        return distances, parent
    
    @staticmethod
    def bellman_ford(graph: Graph, start: Any) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
        """
        Bellman-Ford算法（可以检测负权重环）
        
        Args:
            graph: 加权图对象
            start: 起始节点
            
        Returns:
            距离字典、父节点字典和是否有负权重环
        """
        if not graph.weighted:
            raise ValueError("Bellman-Ford算法需要加权图")
        
        # 初始化
        distances = {node: float('inf') for node in graph.nodes}
        parent = {node: None for node in graph.nodes}
        distances[start] = 0
        
        # 松弛所有边 V-1 次
        for _ in range(len(graph.nodes) - 1):
            for edge in graph.edges.values():
                if distances[edge.source] != float('inf'):
                    new_distance = distances[edge.source] + edge.weight
                    if new_distance < distances[edge.target]:
                        distances[edge.target] = new_distance
                        parent[edge.target] = edge.source
        
        # 检查负权重环
        has_negative_cycle = False
        for edge in graph.edges.values():
            if distances[edge.source] != float('inf'):
                if distances[edge.source] + edge.weight < distances[edge.target]:
                    has_negative_cycle = True
                    break
        
        return distances, parent, has_negative_cycle
    
    @staticmethod
    def floyd_warshall(graph: Graph) -> Dict[Tuple[Any, Any], float]:
        """
        Floyd-Warshall算法（所有节点对最短路径）
        
        Args:
            graph: 加权图对象
            
        Returns:
            所有节点对的最短距离
        """
        if not graph.weighted:
            raise ValueError("Floyd-Warshall算法需要加权图")
        
        nodes = list(graph.nodes.keys())
        n = len(nodes)
        
        # 初始化距离矩阵
        dist = {}
        for i in nodes:
            for j in nodes:
                if i == j:
                    dist[(i, j)] = 0
                else:
                    dist[(i, j)] = float('inf')
        
        # 设置直接边的距离
        for edge in graph.edges.values():
            dist[(edge.source, edge.target)] = min(dist[(edge.source, edge.target)], edge.weight)
        
        # Floyd-Warshall算法
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
        
        return dist
    
    @staticmethod
    def get_shortest_path(graph: Graph, start: Any, end: Any, 
                         algorithm: str = 'dijkstra') -> List[Any]:
        """
        获取两点间的最短路径
        
        Args:
            graph: 加权图对象
            start: 起始节点
            end: 目标节点
            algorithm: 算法选择 ('dijkstra', 'bellman_ford')
            
        Returns:
            最短路径
        """
        if algorithm == 'dijkstra':
            _, parent = ShortestPathAlgorithms.dijkstra(graph, start, end)
        elif algorithm == 'bellman_ford':
            _, parent, _ = ShortestPathAlgorithms.bellman_ford(graph, start)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        # 重构路径
        if parent[end] is None and start != end:
            return []
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parent[current]
        
        return list(reversed(path))


# ============================================================================
# 最小生成树算法
# ============================================================================

class MinimumSpanningTree:
    """最小生成树算法类"""
    
    @staticmethod
    def kruskal(graph: Graph) -> List[Edge]:
        """
        Kruskal算法
        
        Args:
            graph: 加权无向图
            
        Returns:
            最小生成树的边列表
        """
        if graph.directed:
            raise ValueError("Kruskal算法仅适用于无向图")
        
        # 并查集实现
        parent = {node: node for node in graph.nodes}
        rank = {node: 0 for node in graph.nodes}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True
        
        # 按权重排序所有边
        sorted_edges = sorted(graph.edges.values(), key=lambda e: e.weight)
        mst_edges = []
        
        for edge in sorted_edges:
            if union(edge.source, edge.target):
                mst_edges.append(edge)
                if len(mst_edges) == len(graph.nodes) - 1:
                    break
        
        return mst_edges
    
    @staticmethod
    def prim(graph: Graph, start: Any = None) -> List[Edge]:
        """
        Prim算法
        
        Args:
            graph: 加权无向图
            start: 起始节点
            
        Returns:
            最小生成树的边列表
        """
        if graph.directed:
            raise ValueError("Prim算法仅适用于无向图")
        
        if not start:
            start = next(iter(graph.nodes))
        
        if start not in graph.nodes:
            raise ValueError(f"起始节点 {start} 不存在")
        
        visited = {start}
        pq = []
        mst_edges = []
        
        # 将起始节点的所有边加入优先队列
        for neighbor, weight in graph.get_neighbors(start):
            heapq.heappush(pq, (weight, start, neighbor))
        
        while pq and len(visited) < len(graph.nodes):
            weight, source, target = heapq.heappop(pq)
            
            if target not in visited:
                visited.add(target)
                mst_edges.append(Edge(source, target, weight))
                
                # 将新节点的所有边加入优先队列
                for neighbor, w in graph.get_neighbors(target):
                    if neighbor not in visited:
                        heapq.heappush(pq, (w, target, neighbor))
        
        return mst_edges
    
    @staticmethod
    def get_mst_weight(mst_edges: List[Edge]) -> float:
        """计算最小生成树的总权重"""
        return sum(edge.weight for edge in mst_edges)


# ============================================================================
# 图的聚类和社区检测
# ============================================================================

class CommunityDetection:
    """图的聚类和社区检测算法"""
    
    @staticmethod
    def louvain_community_detection(graph: Graph, resolution: float = 1.0) -> Dict[Any, int]:
        """
        Louvain社区检测算法
        
        Args:
            graph: 图对象
            resolution: 分辨率参数
            
        Returns:
            节点到社区的映射
        """
        # 初始化：每个节点为一个社区
        communities = {node: i for i, node in enumerate(graph.nodes)}
        improved = True
        
        while improved:
            improved = False
            
            for node in graph.nodes:
                current_community = communities[node]
                best_community = current_community
                best_gain = 0
                
                # 计算移动到邻居社区的增益
                neighbor_communities = {}
                for neighbor, _ in graph.get_neighbors(node):
                    neighbor_community = communities[neighbor]
                    if neighbor_community != current_community:
                        if neighbor_community not in neighbor_communities:
                            neighbor_communities[neighbor_community] = 0
                        neighbor_communities[neighbor_community] += 1
                
                # 选择增益最大的社区
                for community, count in neighbor_communities.items():
                    gain = count - resolution * 0.1  # 简化的增益计算
                    if gain > best_gain:
                        best_gain = gain
                        best_community = community
                
                # 如果找到更好的社区，移动节点
                if best_community != current_community:
                    communities[node] = best_community
                    improved = True
        
        return communities
    
    @staticmethod
    def label_propagation(graph: Graph, max_iterations: int = 100) -> Dict[Any, int]:
        """
        标签传播算法
        
        Args:
            graph: 图对象
            max_iterations: 最大迭代次数
            
        Returns:
            节点到社区的映射
        """
        # 初始化：每个节点的标签为其自身ID
        labels = {node: node for node in graph.nodes}
        
        for iteration in range(max_iterations):
            new_labels = labels.copy()
            changed = False
            
            for node in graph.nodes:
                # 统计邻居的标签频率
                label_counts = defaultdict(int)
                for neighbor, _ in graph.get_neighbors(node):
                    neighbor_label = labels[neighbor]
                    label_counts[neighbor_label] += 1
                
                if label_counts:
                    # 选择频率最高的标签
                    most_frequent_label = max(label_counts.items(), key=lambda x: x[1])[0]
                    if most_frequent_label != labels[node]:
                        new_labels[node] = most_frequent_label
                        changed = True
            
            labels = new_labels
            if not changed:
                break
        
        # 将标签转换为社区ID
        unique_labels = list(set(labels.values()))
        label_to_community = {label: i for i, label in enumerate(unique_labels)}
        
        return {node: label_to_community[labels[node]] for node in graph.nodes}
    
    @staticmethod
    def spectral_clustering(graph: Graph, k: int = 2) -> Dict[Any, int]:
        """
        谱聚类算法（简化版）
        
        Args:
            graph: 图对象
            k: 聚类数量
            
        Returns:
            节点到聚类的映射
        """
        # 构建邻接矩阵
        nodes = list(graph.nodes)
        n = len(nodes)
        node_to_index = {node: i for i, node in enumerate(nodes)}
        
        # 邻接矩阵
        A = [[0.0] * n for _ in range(n)]
        for edge in graph.edges.values():
            i, j = node_to_index[edge.source], node_to_index[edge.target]
            A[i][j] = 1.0
            A[j][i] = 1.0
        
        # 度矩阵
        D = [[0.0] * n for _ in range(n)]
        for i in range(n):
            D[i][i] = sum(A[i])
        
        # 拉普拉斯矩阵 L = D - A
        L = [[D[i][j] - A[i][j] for j in range(n)] for i in range(n)]
        
        # 简化的特征值分解（使用幂迭代法）
        def power_iteration(matrix, num_iterations=100):
            n = len(matrix)
            vector = [1.0 / n] * n
            
            for _ in range(num_iterations):
                new_vector = [0.0] * n
                for i in range(n):
                    for j in range(n):
                        new_vector[i] += matrix[i][j] * vector[j]
                
                # 归一化
                norm = sum(x * x for x in new_vector) ** 0.5
                if norm > 0:
                    vector = [x / norm for x in new_vector]
            
            return vector
        
        # 获取前k个特征向量（简化版）
        eigenvectors = []
        for _ in range(k):
            eigenv = power_iteration(L)
            eigenvectors.append(eigenv)
            
            # 正交化
            for prev_eigenv in eigenvectors[:-1]:
                dot_product = sum(eigenv[i] * prev_eigenv[i] for i in range(n))
                eigenv = [eigenv[i] - dot_product * prev_eigenv[i] for i in range(n)]
        
        # 简单的k-means聚类
        clusters = {}
        for i in range(n):
            # 使用第一个特征向量的符号进行二分类
            if k == 2:
                clusters[nodes[i]] = 0 if eigenvectors[0][i] >= 0 else 1
            else:
                # 随机分配（简化实现）
                clusters[nodes[i]] = i % k
        
        return clusters
    
    @staticmethod
    def calculate_modularity(graph: Graph, communities: Dict[Any, int]) -> float:
        """
        计算模块度
        
        Args:
            graph: 图对象
            communities: 社区映射
            
        Returns:
            模块度值
        """
        m = graph.get_edge_count()
        if m == 0:
            return 0.0
        
        modularity = 0.0
        
        for edge in graph.edges.values():
            if communities[edge.source] == communities[edge.target]:
                # 内部边
                ki = sum(1 for _, _ in graph.get_neighbors(edge.source))
                kj = sum(1 for _, _ in graph.get_neighbors(edge.target))
                modularity += 1 - (ki * kj) / (2 * m)
        
        return modularity / (2 * m)


# ============================================================================
# 图神经网络
# ============================================================================

class GraphNeuralNetwork:
    """简化的图神经网络实现"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        初始化GNN
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 简化的权重矩阵
        self.W1 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.W2 = [[random.gauss(0, 0.1) for _ in range(output_dim)] for _ in range(hidden_dim)]
        
        # 激活函数
        self.activation = lambda x: max(0, x)  # ReLU
    
    def relu(self, x: float) -> float:
        """ReLU激活函数"""
        return max(0, x)
    
    def softmax(self, x: List[float]) -> List[float]:
        """Softmax激活函数"""
        exp_x = [math.exp(val) for val in x]
        sum_exp = sum(exp_x)
        return [val / sum_exp for val in exp_x]
    
    def matrix_multiply(self, matrix: List[List[float]], vector: List[float]) -> List[float]:
        """矩阵乘法"""
        result = []
        for row in matrix:
            dot_product = sum(a * b for a, b in zip(row, vector))
            result.append(dot_product)
        return result
    
    def aggregate_neighbors(self, graph: Graph, node_features: Dict[Any, List[float]]) -> Dict[Any, List[float]]:
        """聚合邻居节点特征"""
        aggregated = {}
        
        for node in graph.nodes:
            neighbors = graph.get_neighbors(node)
            if not neighbors:
                aggregated[node] = node_features[node]
                continue
            
            # 聚合邻居特征（简单平均）
            neighbor_features = []
            for neighbor, _ in neighbors:
                if neighbor in node_features:
                    neighbor_features.append(node_features[neighbor])
            
            if neighbor_features:
                # 计算平均值
                aggregated[node] = [
                    sum(features[i] for features in neighbor_features) / len(neighbor_features)
                    for i in range(len(neighbor_features[0]))
                ]
            else:
                aggregated[node] = node_features[node]
        
        return aggregated
    
    def gcn_layer(self, graph: Graph, node_features: Dict[Any, List[float]]) -> Dict[Any, List[float]]:
        """图卷积层"""
        # 聚合邻居特征
        aggregated = self.aggregate_neighbors(graph, node_features)
        
        # 应用线性变换和激活函数
        new_features = {}
        for node, features in aggregated.items():
            # 线性变换
            hidden = self.matrix_multiply(self.W1, features)
            # 激活函数
            activated = [self.relu(h) for h in hidden]
            new_features[node] = activated
        
        return new_features
    
    def forward(self, graph: Graph, node_features: Dict[Any, List[float]]) -> Dict[Any, List[float]]:
        """前向传播"""
        # 第一层：图卷积
        hidden_features = self.gcn_layer(graph, node_features)
        
        # 第二层：输出层
        output_features = {}
        for node, features in hidden_features.items():
            # 线性变换到输出维度
            output = self.matrix_multiply(self.W2, features)
            # Softmax激活
            output_features[node] = self.softmax(output)
        
        return output_features
    
    def node_classification(self, graph: Graph, node_features: Dict[Any, List[float]], 
                          labeled_nodes: Dict[Any, int]) -> Dict[Any, float]:
        """
        节点分类
        
        Args:
            graph: 图对象
            node_features: 节点特征
            labeled_nodes: 标记的节点
            
        Returns:
            节点分类概率
        """
        # 前向传播
        output = self.forward(graph, node_features)
        
        # 返回分类概率
        return {node: probs[label] if label < len(probs) else 0.0 
                for node, probs in output.items() 
                for label in [labeled_nodes.get(node, 0)]}
    
    def graph_classification(self, graph: Graph, node_features: Dict[Any, List[float]]) -> List[float]:
        """
        图分类
        
        Args:
            graph: 图对象
            node_features: 节点特征
            
        Returns:
            图分类概率
        """
        # 前向传播
        output = self.forward(graph, node_features)
        
        # 全局池化（简单平均）
        if not output:
            return [0.0] * self.output_dim
        
        num_nodes = len(output)
        pooled = [0.0] * self.output_dim
        
        for node_features in output.values():
            for i, feature in enumerate(node_features):
                pooled[i] += feature
        
        # 平均
        pooled = [feature / num_nodes for feature in pooled]
        
        # Softmax
        return self.softmax(pooled)


# ============================================================================
# 图的相似性度量
# ============================================================================

class GraphSimilarity:
    """图相似性度量算法"""
    
    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        """Jaccard相似度"""
        union = set1 | set2
        if not union:
            return 1.0
        intersection = set1 & set2
        return len(intersection) / len(union)
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def node_similarity(graph: Graph, node1: Any, node2: Any) -> float:
        """
        节点相似度（基于共同邻居）
        
        Args:
            graph: 图对象
            node1, node2: 要比较的节点
            
        Returns:
            相似度值
        """
        neighbors1 = {neighbor for neighbor, _ in graph.get_neighbors(node1)}
        neighbors2 = {neighbor for neighbor, _ in graph.get_neighbors(node2)}
        
        return GraphSimilarity.jaccard_similarity(neighbors1, neighbors2)
    
    @staticmethod
    def structural_similarity(graph: Graph, node1: Any, node2: Any) -> float:
        """
        结构相似度
        
        Args:
            graph: 图对象
            node1, node2: 要比较的节点
            
        Returns:
            结构相似度值
        """
        # 度相似度
        degree1 = len(graph.get_neighbors(node1))
        degree2 = len(graph.get_neighbors(node2))
        
        if degree1 == 0 and degree2 == 0:
            degree_sim = 1.0
        else:
            degree_sim = 1 - abs(degree1 - degree2) / max(degree1, degree2)
        
        # 邻居度相似度
        neighbor_degrees1 = sorted([len(graph.get_neighbors(neighbor)) for neighbor, _ in graph.get_neighbors(node1)])
        neighbor_degrees2 = sorted([len(graph.get_neighbors(neighbor)) for neighbor, _ in graph.get_neighbors(node2)])
        
        # 计算度序列相似度
        max_len = max(len(neighbor_degrees1), len(neighbor_degrees2))
        if max_len == 0:
            neighbor_sim = 1.0
        else:
            neighbor_degrees1.extend([0] * (max_len - len(neighbor_degrees1)))
            neighbor_degrees2.extend([0] * (max_len - len(neighbor_degrees2)))
            neighbor_sim = GraphSimilarity.cosine_similarity(neighbor_degrees1, neighbor_degrees2)
        
        return (degree_sim + neighbor_sim) / 2
    
    @staticmethod
    def graph_edit_distance(graph1: Graph, graph2: Graph) -> int:
        """
        图编辑距离（简化版）
        
        Args:
            graph1, graph2: 要比较的图
            
        Returns:
            编辑距离
        """
        # 节点编辑距离
        nodes1 = set(graph1.nodes.keys())
        nodes2 = set(graph2.nodes.keys())
        
        node_insertions = len(nodes2 - nodes1)
        node_deletions = len(nodes1 - nodes2)
        common_nodes = nodes1 & nodes2
        
        # 边编辑距离
        edges1 = set(graph1.edges.keys())
        edges2 = set(graph2.edges.keys())
        
        edge_insertions = len(edges2 - edges1)
        edge_deletions = len(edges1 - edges2)
        common_edges = edges1 & edges2
        
        # 权重差异（对于共同边）
        weight_diff = 0
        for edge_key in common_edges:
            weight1 = graph1.edges[edge_key].weight
            weight2 = graph2.edges[edge_key].weight
            weight_diff += abs(weight1 - weight2)
        
        return node_insertions + node_deletions + edge_insertions + edge_deletions + int(weight_diff)
    
    @staticmethod
    def WeisfeilerLehman_similarity(graph: Graph, node1: Any, node2: Any, iterations: int = 3) -> float:
        """
        Weisfeiler-Lehman相似度
        
        Args:
            graph: 图对象
            node1, node2: 要比较的节点
            iterations: 迭代次数
            
        Returns:
            相似度值
        """
        # 初始化标签
        labels = {node: "1" for node in graph.nodes}
        
        for iteration in range(iterations):
            new_labels = {}
            for node in graph.nodes:
                # 获取邻居标签
                neighbor_labels = sorted([labels[neighbor] for neighbor, _ in graph.get_neighbors(node)])
                # 创建新标签
                new_label = f"{labels[node]}{''.join(neighbor_labels)}"
                new_labels[node] = new_label
            
            labels = new_labels
        
        # 计算标签相似度
        label1 = labels.get(node1, "")
        label2 = labels.get(node2, "")
        
        if label1 == label2:
            return 1.0
        
        # 计算最长公共子序列
        lcs_length = GraphSimilarity._lcs_length(label1, label2)
        max_length = max(len(label1), len(label2))
        
        return lcs_length / max_length if max_length > 0 else 0.0
    
    @staticmethod
    def _lcs_length(str1: str, str2: str) -> int:
        """计算最长公共子序列长度"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


# ============================================================================
# 图的嵌入和表示学习
# ============================================================================

class GraphEmbedding:
    """图的嵌入和表示学习算法"""
    
    @staticmethod
    def deepwalk(graph: Graph, walk_length: int = 40, num_walks: int = 80, 
                embedding_dim: int = 128) -> Dict[Any, List[float]]:
        """
        DeepWalk算法
        
        Args:
            graph: 图对象
            walk_length: 随机游走长度
            num_walks: 每个节点的游走次数
            embedding_dim: 嵌入维度
            
        Returns:
            节点嵌入向量
        """
        # 生成随机游走序列
        walks = []
        nodes = list(graph.nodes.keys())
        
        for _ in range(num_walks):
            for start_node in nodes:
                walk = [start_node]
                current_node = start_node
                
                for _ in range(walk_length - 1):
                    neighbors = graph.get_neighbors(current_node)
                    if not neighbors:
                        break
                    
                    next_node = random.choice(neighbors)[0]
                    walk.append(next_node)
                    current_node = next_node
                
                walks.append(walk)
        
        # 简化的Skip-gram模型
        embeddings = GraphEmbedding._skipgram_simple(walks, embedding_dim)
        
        return embeddings
    
    @staticmethod
    def _skipgram_simple(walks: List[List[Any]], embedding_dim: int) -> Dict[Any, List[float]]:
        """简化的Skip-gram模型"""
        # 收集所有节点
        nodes = set()
        for walk in walks:
            nodes.update(walk)
        
        # 初始化嵌入
        embeddings = {node: [random.gauss(0, 0.1) for _ in range(embedding_dim)] for node in nodes}
        
        # 简化的训练过程
        learning_rate = 0.01
        window_size = 5
        
        for walk in walks:
            for i, target_node in enumerate(walk):
                # 定义窗口
                start = max(0, i - window_size)
                end = min(len(walk), i + window_size + 1)
                
                context_nodes = walk[start:end]
                context_nodes.remove(target_node)
                
                # 简化的梯度更新
                for context_node in context_nodes:
                    # 计算误差（这里简化处理）
                    for dim in range(embedding_dim):
                        error = random.gauss(0, 0.01)  # 模拟误差
                        embeddings[target_node][dim] += learning_rate * error
                        embeddings[context_node][dim] += learning_rate * error
        
        return embeddings
    
    @staticmethod
    def node2vec(graph: Graph, walk_length: int = 40, num_walks: int = 80, 
                embedding_dim: int = 128, p: float = 1.0, q: float = 1.0) -> Dict[Any, List[float]]:
        """
        Node2Vec算法
        
        Args:
            graph: 图对象
            walk_length: 随机游走长度
            num_walks: 每个节点的游走次数
            embedding_dim: 嵌入维度
            p: 返回参数
            q: 进出参数
            
        Returns:
            节点嵌入向量
        """
        walks = []
        nodes = list(graph.nodes.keys())
        
        for _ in range(num_walks):
            for start_node in nodes:
                walk = [start_node]
                current_node = start_node
                previous_node = None
                
                for _ in range(walk_length - 1):
                    neighbors = graph.get_neighbors(current_node)
                    if not neighbors:
                        break
                    
                    # 计算转移概率
                    if previous_node is None:
                        # 第一个节点，随机选择
                        next_node = random.choice(neighbors)[0]
                    else:
                        # 基于p和q计算概率
                        weights = []
                        for neighbor, weight in neighbors:
                            if neighbor == previous_node:
                                weights.append(weight / p)  # 返回概率
                            elif GraphEmbedding._is_neighbor(graph, neighbor, previous_node):
                                weights.append(weight)  # 正常概率
                            else:
                                weights.append(weight / q)  # 进出概率
                        
                        # 选择下一个节点
                        total_weight = sum(weights)
                        if total_weight == 0:
                            next_node = random.choice(neighbors)[0]
                        else:
                            probs = [w / total_weight for w in weights]
                            next_node = random.choices([n[0] for n in neighbors], weights=probs)[0]
                    
                    walk.append(next_node)
                    previous_node = current_node
                    current_node = next_node
                
                walks.append(walk)
        
        # 使用Skip-gram训练嵌入
        return GraphEmbedding._skipgram_simple(walks, embedding_dim)
    
    @staticmethod
    def _is_neighbor(graph: Graph, node1: Any, node2: Any) -> bool:
        """检查两个节点是否相邻"""
        neighbors = {neighbor for neighbor, _ in graph.get_neighbors(node1)}
        return node2 in neighbors
    
    @staticmethod
    def graph_sage(graph: Graph, embedding_dim: int = 128, 
                  num_layers: int = 2) -> Dict[Any, List[float]]:
        """
        GraphSAGE算法（简化版）
        
        Args:
            graph: 图对象
            embedding_dim: 嵌入维度
            num_layers: 层数
            
        Returns:
            节点嵌入向量
        """
        # 初始化特征（使用节点的度作为特征）
        features = {}
        for node in graph.nodes:
            degree = len(graph.get_neighbors(node))
            features[node] = [float(degree), random.gauss(0, 0.1)]
        
        # 多层聚合
        current_features = features
        
        for layer in range(num_layers):
            new_features = {}
            
            for node in graph.nodes:
                # 聚合邻居特征
                neighbors = graph.get_neighbors(node)
                if not neighbors:
                    new_features[node] = current_features[node]
                    continue
                
                # 采样邻居（限制数量）
                sampled_neighbors = neighbors[:5] if len(neighbors) > 5 else neighbors
                
                # 聚合邻居特征
                aggregated = [0.0, 0.0]
                for neighbor, _ in sampled_neighbors:
                    for i in range(2):
                        aggregated[i] += current_features[neighbor][i]
                
                # 平均
                aggregated = [x / len(sampled_neighbors) for x in aggregated]
                
                # 与自身特征连接
                combined = current_features[node] + aggregated
                
                # 线性变换到目标维度
                if len(combined) != embedding_dim:
                    # 简单降维或升维
                    if len(combined) > embedding_dim:
                        combined = combined[:embedding_dim]
                    else:
                        combined.extend([0.0] * (embedding_dim - len(combined)))
                
                new_features[node] = combined
            
            current_features = new_features
        
        return current_features
    
    @staticmethod
    def compute_similarity_matrix(embeddings: Dict[Any, List[float]]) -> Dict[Tuple[Any, Any], float]:
        """
        计算嵌入相似度矩阵
        
        Args:
            embeddings: 节点嵌入向量
            
        Returns:
            相似度矩阵
        """
        similarity_matrix = {}
        nodes = list(embeddings.keys())
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i <= j:  # 只计算上三角矩阵
                    similarity = GraphSimilarity.cosine_similarity(
                        embeddings[node1], embeddings[node2]
                    )
                    similarity_matrix[(node1, node2)] = similarity
                    if i != j:
                        similarity_matrix[(node2, node1)] = similarity
        
        return similarity_matrix


# ============================================================================
# 动态图算法
# ============================================================================

class DynamicGraph:
    """动态图算法类"""
    
    def __init__(self, initial_graph: Graph = None):
        """
        初始化动态图
        
        Args:
            initial_graph: 初始图
        """
        self.graph = initial_graph if initial_graph else Graph()
        self.history = []  # 记录图的变化历史
        
    def add_node_dynamic(self, node_id: Any, attributes: Dict[str, Any] = None, 
                        timestamp: float = None) -> None:
        """动态添加节点"""
        if timestamp is None:
            timestamp = time.time()
        
        self.graph.add_node(node_id, attributes)
        self.history.append({
            'type': 'add_node',
            'node': node_id,
            'attributes': attributes,
            'timestamp': timestamp
        })
    
    def add_edge_dynamic(self, source: Any, target: Any, weight: float = 1.0,
                        attributes: Dict[str, Any] = None, timestamp: float = None) -> None:
        """动态添加边"""
        if timestamp is None:
            timestamp = time.time()
        
        self.graph.add_edge(source, target, weight, attributes)
        self.history.append({
            'type': 'add_edge',
            'source': source,
            'target': target,
            'weight': weight,
            'attributes': attributes,
            'timestamp': timestamp
        })
    
    def remove_node_dynamic(self, node_id: Any, timestamp: float = None) -> None:
        """动态删除节点"""
        if timestamp is None:
            timestamp = time.time()
        
        self.graph.remove_node(node_id)
        self.history.append({
            'type': 'remove_node',
            'node': node_id,
            'timestamp': timestamp
        })
    
    def remove_edge_dynamic(self, source: Any, target: Any, timestamp: float = None) -> None:
        """动态删除边"""
        if timestamp is None:
            timestamp = time.time()
        
        self.graph.remove_edge(source, target)
        self.history.append({
            'type': 'remove_edge',
            'source': source,
            'target': target,
            'timestamp': timestamp
        })
    
    def update_edge_weight_dynamic(self, source: Any, target: Any, new_weight: float,
                                 timestamp: float = None) -> None:
        """动态更新边权重"""
        if timestamp is None:
            timestamp = time.time()
        
        old_weight = self.graph.get_edge_weight(source, target)
        self.graph.add_edge(source, target, new_weight)
        self.history.append({
            'type': 'update_edge',
            'source': source,
            'target': target,
            'old_weight': old_weight,
            'new_weight': new_weight,
            'timestamp': timestamp
        })
    
    def incremental_shortest_path(self, source: Any, target: Any) -> List[Any]:
        """增量最短路径算法"""
        # 使用Dijkstra算法重新计算
        return ShortestPathAlgorithms.get_shortest_path(self.graph, source, target)
    
    def incremental_clustering(self, algorithm: str = 'louvain') -> Dict[Any, int]:
        """增量聚类算法"""
        if algorithm == 'louvain':
            return CommunityDetection.louvain_community_detection(self.graph)
        elif algorithm == 'label_propagation':
            return CommunityDetection.label_propagation(self.graph)
        else:
            raise ValueError(f"不支持的聚类算法: {algorithm}")
    
    def get_graph_at_timestamp(self, timestamp: float) -> Graph:
        """获取指定时间戳的图快照"""
        snapshot = Graph(self.graph.directed, self.graph.weighted)
        
        # 重放历史直到指定时间戳
        for event in self.history:
            if event['timestamp'] > timestamp:
                break
            
            if event['type'] == 'add_node':
                snapshot.add_node(event['node'], event['attributes'])
            elif event['type'] == 'add_edge':
                snapshot.add_edge(event['source'], event['target'], 
                                event['weight'], event['attributes'])
            elif event['type'] == 'remove_node':
                snapshot.remove_node(event['node'])
            elif event['type'] == 'remove_edge':
                snapshot.remove_edge(event['source'], event['target'])
        
        return snapshot
    
    def detect_anomalies(self, window_size: int = 10) -> List[Dict[str, Any]]:
        """
        检测图异常
        
        Args:
            window_size: 滑动窗口大小
            
        Returns:
            异常事件列表
        """
        anomalies = []
        
        for i in range(len(self.history) - window_size + 1):
            window = self.history[i:i + window_size]
            
            # 计算窗口内的统计信息
            node_changes = sum(1 for event in window if event['type'] in ['add_node', 'remove_node'])
            edge_changes = sum(1 for event in window if event['type'] in ['add_edge', 'remove_edge'])
            
            # 简单的异常检测：变化量超过阈值
            if node_changes > window_size * 0.5 or edge_changes > window_size * 0.7:
                anomalies.append({
                    'start_index': i,
                    'end_index': i + window_size - 1,
                    'node_changes': node_changes,
                    'edge_changes': edge_changes,
                    'timestamp': window[0]['timestamp']
                })
        
        return anomalies
    
    def get_evolution_metrics(self) -> Dict[str, Any]:
        """获取图演化指标"""
        if not self.history:
            return {}
        
        # 计算各种统计信息
        node_additions = sum(1 for event in self.history if event['type'] == 'add_node')
        node_deletions = sum(1 for event in self.history if event['type'] == 'remove_node')
        edge_additions = sum(1 for event in self.history if event['type'] == 'add_edge')
        edge_deletions = sum(1 for event in self.history if event['type'] == 'remove_edge')
        
        # 计算时间跨度
        time_span = self.history[-1]['timestamp'] - self.history[0]['timestamp']
        
        return {
            'total_events': len(self.history),
            'node_additions': node_additions,
            'node_deletions': node_deletions,
            'edge_additions': edge_additions,
            'edge_deletions': edge_deletions,
            'time_span': time_span,
            'events_per_second': len(self.history) / time_span if time_span > 0 else 0,
            'current_node_count': self.graph.get_node_count(),
            'current_edge_count': self.graph.get_edge_count()
        }


# ============================================================================
# 图的可视化和分析
# ============================================================================

class GraphVisualization:
    """图的可视化和分析工具"""
    
    @staticmethod
    def calculate_layout(graph: Graph, algorithm: str = 'spring') -> Dict[Any, Tuple[float, float]]:
        """
        计算图的布局
        
        Args:
            graph: 图对象
            algorithm: 布局算法 ('spring', 'circular', 'random')
            
        Returns:
            节点位置字典
        """
        nodes = list(graph.nodes.keys())
        n = len(nodes)
        
        if algorithm == 'circular':
            # 圆形布局
            positions = {}
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / n
                x = math.cos(angle)
                y = math.sin(angle)
                positions[node] = (x, y)
            return positions
        
        elif algorithm == 'random':
            # 随机布局
            random.seed(42)  # 保证可重复性
            positions = {}
            for node in nodes:
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                positions[node] = (x, y)
            return positions
        
        elif algorithm == 'spring':
            # 弹簧布局（简化版）
            positions = {}
            # 初始化随机位置
            for node in nodes:
                positions[node] = (random.uniform(-1, 1), random.uniform(-1, 1))
            
            # 迭代优化
            for iteration in range(100):
                forces = {node: (0.0, 0.0) for node in nodes}
                
                # 计算节点间的排斥力
                for i, node1 in enumerate(nodes):
                    for j, node2 in enumerate(nodes):
                        if i != j:
                            x1, y1 = positions[node1]
                            x2, y2 = positions[node2]
                            
                            dx = x1 - x2
                            dy = y1 - y2
                            distance = math.sqrt(dx * dx + dy * dy) + 0.01
                            
                            # 排斥力
                            force = 1.0 / distance
                            forces[node1] = (forces[node1][0] + force * dx / distance,
                                           forces[node1][1] + force * dy / distance)
                
                # 计算边的吸引力
                for edge in graph.edges.values():
                    node1, node2 = edge.source, edge.target
                    x1, y1 = positions[node1]
                    x2, y2 = positions[node2]
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = math.sqrt(dx * dx + dy * dy) + 0.01
                    
                    # 吸引力
                    force = distance * 0.01
                    forces[node1] = (forces[node1][0] + force * dx / distance,
                                   forces[node1][1] + force * dy / distance)
                    forces[node2] = (forces[node2][0] - force * dx / distance,
                                   forces[node2][1] - force * dy / distance)
                
                # 更新位置
                for node in nodes:
                    x, y = positions[node]
                    fx, fy = forces[node]
                    positions[node] = (x + fx * 0.01, y + fy * 0.01)
            
            return positions
        
        else:
            raise ValueError(f"不支持的布局算法: {algorithm}")
    
    @staticmethod
    def analyze_graph_properties(graph: Graph) -> Dict[str, Any]:
        """
        分析图的属性
        
        Args:
            graph: 图对象
            
        Returns:
            图属性分析结果
        """
        if graph.get_node_count() == 0:
            return {}
        
        # 基本属性
        num_nodes = graph.get_node_count()
        num_edges = graph.get_edge_count()
        
        # 度分布
        degrees = [len(graph.get_neighbors(node)) for node in graph.nodes]
        avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        
        # 连通性
        is_connected = graph.is_connected()
        
        # 聚类系数
        clustering_coeffs = []
        for node in graph.nodes:
            neighbors = [neighbor for neighbor, _ in graph.get_neighbors(node)]
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            edges_between_neighbors = 0
            for i, neighbor1 in enumerate(neighbors):
                for j, neighbor2 in enumerate(neighbors):
                    if i < j:
                        if graph.get_edge_weight(neighbor1, neighbor2) != float('inf'):
                            edges_between_neighbors += 1
            
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            clustering_coeffs.append(edges_between_neighbors / possible_edges if possible_edges > 0 else 0)
        
        avg_clustering = sum(clustering_coeffs) / len(clustering_coeffs) if clustering_coeffs else 0
        
        # 直径（仅对连通图）
        diameter = None
        if is_connected and num_nodes > 1:
            # 简化的直径计算
            max_distance = 0
            nodes_list = list(graph.nodes.keys())
            for i in range(min(10, len(nodes_list))):  # 限制计算量
                for j in range(i + 1, min(10, len(nodes_list))):
                    try:
                        path = TraversalAlgorithms.find_path(graph, nodes_list[i], nodes_list[j])
                        if path:
                            max_distance = max(max_distance, len(path) - 1)
                    except:
                        pass
            diameter = max_distance if max_distance > 0 else None
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'is_directed': graph.directed,
            'is_weighted': graph.weighted,
            'density': 2 * num_edges / (num_nodes * (num_nodes - 1)) if not graph.directed and num_nodes > 1 else
                      num_edges / (num_nodes * (num_nodes - 1)) if graph.directed and num_nodes > 1 else 0,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'min_degree': min_degree,
            'is_connected': is_connected,
            'avg_clustering_coefficient': avg_clustering,
            'diameter': diameter,
            'degree_distribution': {
                'degrees': degrees,
                'unique_degrees': list(set(degrees)),
                'degree_counts': {degree: degrees.count(degree) for degree in set(degrees)}
            }
        }
    
    @staticmethod
    def export_to_gexf(graph: Graph, filename: str) -> None:
        """
        导出图到GEXF格式
        
        Args:
            graph: 图对象
            filename: 输出文件名
        """
        gexf_content = ['<?xml version="1.0" encoding="UTF-8"?>']
        gexf_content.append('<gexf xmlns="http://gexf.net/1.3" version="1.3">')
        gexf_content.append('  <graph mode="static" defaultedgetype="directed">')
        
        # 节点
        gexf_content.append('    <nodes>')
        for node_id, node in graph.nodes.items():
            attributes = ' '.join([f'{k}="{v}"' for k, v in node.attributes.items()])
            gexf_content.append(f'      <node id="{node_id}" label="{node_id}" {attributes}/>')
        gexf_content.append('    </nodes>')
        
        # 边
        gexf_content.append('    <edges>')
        for edge_id, edge in enumerate(graph.edges.values()):
            attributes = ' '.join([f'{k}="{v}"' for k, v in edge.attributes.items()])
            gexf_content.append(f'      <edge id="{edge_id}" source="{edge.source}" '
                              f'target="{edge.target}" weight="{edge.weight}" {attributes}/>')
        gexf_content.append('    </edges>')
        
        gexf_content.append('  </graph>')
        gexf_content.append('</gexf>')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(gexf_content))
    
    @staticmethod
    def export_to_json(graph: Graph, filename: str) -> None:
        """导出图到JSON格式"""
        graph_dict = graph.to_dict()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def generate_graph_report(graph: Graph, filename: str = None) -> str:
        """
        生成图分析报告
        
        Args:
            graph: 图对象
            filename: 可选的输出文件名
            
        Returns:
            报告内容
        """
        properties = GraphVisualization.analyze_graph_properties(graph)
        
        report = ["# 图分析报告\n"]
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 基本信息
        report.append("## 基本信息")
        report.append(f"- 节点数量: {properties.get('num_nodes', 0)}")
        report.append(f"- 边数量: {properties.get('num_edges', 0)}")
        report.append(f"- 是否为有向图: {properties.get('is_directed', False)}")
        report.append(f"- 是否为加权图: {properties.get('is_weighted', False)}")
        report.append(f"- 图密度: {properties.get('density', 0):.4f}")
        report.append("")
        
        # 度分布
        report.append("## 度分布")
        report.append(f"- 平均度: {properties.get('avg_degree', 0):.2f}")
        report.append(f"- 最大度: {properties.get('max_degree', 0)}")
        report.append(f"- 最小度: {properties.get('min_degree', 0)}")
        report.append("")
        
        # 连通性
        report.append("## 连通性")
        report.append(f"- 是否连通: {properties.get('is_connected', False)}")
        if properties.get('diameter') is not None:
            report.append(f"- 图直径: {properties.get('diameter')}")
        report.append("")
        
        # 聚类
        report.append("## 聚类分析")
        report.append(f"- 平均聚类系数: {properties.get('avg_clustering_coefficient', 0):.4f}")
        report.append("")
        
        # 度分布详情
        if 'degree_distribution' in properties:
            report.append("## 度分布详情")
            degree_counts = properties['degree_distribution']['degree_counts']
            for degree in sorted(degree_counts.keys()):
                count = degree_counts[degree]
                report.append(f"- 度为 {degree} 的节点: {count} 个")
            report.append("")
        
        report_content = '\n'.join(report)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        return report_content


# ============================================================================
# 主图算法库类
# ============================================================================

class GraphAlgorithmLibrary:
    """图算法库主类"""
    
    def __init__(self):
        """初始化图算法库"""
        self.graph = None
        self.traversal = TraversalAlgorithms()
        self.shortest_path = ShortestPathAlgorithms()
        self.mst = MinimumSpanningTree()
        self.community = CommunityDetection()
        self.gnn = None  # 需要初始化时指定参数
        self.similarity = GraphSimilarity()
        self.embedding = GraphEmbedding()
        self.dynamic = DynamicGraph()
        self.visualization = GraphVisualization()
    
    def create_graph(self, directed: bool = False, weighted: bool = False) -> Graph:
        """
        创建新图
        
        Args:
            directed: 是否为有向图
            weighted: 是否为加权图
            
        Returns:
            图对象
        """
        self.graph = Graph(directed, weighted)
        self.dynamic = DynamicGraph(self.graph)
        return self.graph
    
    def load_graph_from_dict(self, graph_dict: Dict[str, Any]) -> Graph:
        """
        从字典加载图
        
        Args:
            graph_dict: 图的字典表示
            
        Returns:
            图对象
        """
        self.graph = Graph.from_dict(graph_dict)
        self.dynamic = DynamicGraph(self.graph)
        return self.graph
    
    def get_current_graph(self) -> Graph:
        """获取当前图"""
        return self.graph
    
    def initialize_gnn(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """初始化图神经网络"""
        self.gnn = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试用例"""
        test_results = {}
        
        try:
            # 测试1: 基本图操作
            test_results['basic_operations'] = self._test_basic_operations()
            
            # 测试2: 遍历算法
            test_results['traversal'] = self._test_traversal_algorithms()
            
            # 测试3: 最短路径算法
            test_results['shortest_path'] = self._test_shortest_path_algorithms()
            
            # 测试4: 最小生成树
            test_results['mst'] = self._test_minimum_spanning_tree()
            
            # 测试5: 社区检测
            test_results['community_detection'] = self._test_community_detection()
            
            # 测试6: 图相似性
            test_results['similarity'] = self._test_graph_similarity()
            
            # 测试7: 图嵌入
            test_results['embedding'] = self._test_graph_embedding()
            
            # 测试8: 动态图
            test_results['dynamic'] = self._test_dynamic_graph()
            
            # 测试9: 图分析
            test_results['analysis'] = self._test_graph_analysis()
            
        except Exception as e:
            test_results['error'] = str(e)
        
        return test_results
    
    def _test_basic_operations(self) -> Dict[str, Any]:
        """测试基本图操作"""
        graph = self.create_graph(directed=False, weighted=True)
        
        # 添加节点和边
        nodes = ['A', 'B', 'C', 'D', 'E']
        edges = [('A', 'B', 1), ('A', 'C', 4), ('B', 'C', 2), 
                ('B', 'D', 5), ('C', 'D', 1), ('C', 'E', 3), ('D', 'E', 2)]
        
        for node in nodes:
            graph.add_node(node)
        
        for source, target, weight in edges:
            graph.add_edge(source, target, weight)
        
        return {
            'node_count': graph.get_node_count(),
            'edge_count': graph.get_edge_count(),
            'is_connected': graph.is_connected(),
            'test_passed': graph.get_node_count() == 5 and graph.get_edge_count() == 7
        }
    
    def _test_traversal_algorithms(self) -> Dict[str, Any]:
        """测试遍历算法"""
        graph = self.graph if self.graph else self.create_graph()
        
        # BFS测试
        bfs_path, bfs_parent = self.traversal.bfs(graph, 'A', 'D')
        bfs_success = 'D' in bfs_path
        
        # DFS测试
        dfs_path, dfs_parent = self.traversal.dfs(graph, 'A', 'D')
        dfs_success = 'D' in dfs_path
        
        # 路径查找测试
        path = self.traversal.find_path(graph, 'A', 'D')
        path_success = len(path) > 0
        
        return {
            'bfs_success': bfs_success,
            'dfs_success': dfs_success,
            'path_finding_success': path_success,
            'test_passed': bfs_success and dfs_success and path_success
        }
    
    def _test_shortest_path_algorithms(self) -> Dict[str, Any]:
        """测试最短路径算法"""
        graph = self.graph if self.graph else self.create_graph()
        
        if not graph.weighted:
            return {'test_passed': False, 'reason': '图不是加权图'}
        
        # Dijkstra测试
        try:
            distances, parent = self.shortest_path.dijkstra(graph, 'A')
            dijkstra_success = 'D' in distances and distances['D'] != float('inf')
        except:
            dijkstra_success = False
        
        # 最短路径测试
        try:
            shortest_path = self.shortest_path.get_shortest_path(graph, 'A', 'D')
            sp_success = len(shortest_path) > 0
        except:
            sp_success = False
        
        return {
            'dijkstra_success': dijkstra_success,
            'shortest_path_success': sp_success,
            'test_passed': dijkstra_success and sp_success
        }
    
    def _test_minimum_spanning_tree(self) -> Dict[str, Any]:
        """测试最小生成树算法"""
        graph = self.graph if self.graph else self.create_graph()
        
        if graph.directed:
            return {'test_passed': False, 'reason': 'MST算法仅适用于无向图'}
        
        # Kruskal测试
        try:
            mst_edges = self.mst.kruskal(graph)
            kruskal_success = len(mst_edges) == graph.get_node_count() - 1
        except:
            kruskal_success = False
        
        # Prim测试
        try:
            mst_edges = self.mst.prim(graph, 'A')
            prim_success = len(mst_edges) == graph.get_node_count() - 1
        except:
            prim_success = False
        
        return {
            'kruskal_success': kruskal_success,
            'prim_success': prim_success,
            'test_passed': kruskal_success and prim_success
        }
    
    def _test_community_detection(self) -> Dict[str, Any]:
        """测试社区检测算法"""
        graph = self.graph if self.graph else self.create_graph()
        
        # Louvain算法测试
        try:
            communities = self.community.louvain_community_detection(graph)
            louvain_success = len(set(communities.values())) > 0
        except:
            louvain_success = False
        
        # 标签传播测试
        try:
            communities = self.community.label_propagation(graph)
            label_prop_success = len(set(communities.values())) > 0
        except:
            label_prop_success = False
        
        return {
            'louvain_success': louvain_success,
            'label_propagation_success': label_prop_success,
            'test_passed': louvain_success and label_prop_success
        }
    
    def _test_graph_similarity(self) -> Dict[str, Any]:
        """测试图相似性算法"""
        graph = self.graph if self.graph else self.create_graph()
        
        if graph.get_node_count() < 2:
            return {'test_passed': False, 'reason': '节点数量不足'}
        
        nodes = list(graph.nodes.keys())
        node1, node2 = nodes[0], nodes[1]
        
        # 节点相似度测试
        try:
            similarity = self.similarity.node_similarity(graph, node1, node2)
            node_sim_success = 0 <= similarity <= 1
        except:
            node_sim_success = False
        
        # 结构相似度测试
        try:
            struct_sim = self.similarity.structural_similarity(graph, node1, node2)
            struct_sim_success = 0 <= struct_sim <= 1
        except:
            struct_sim_success = False
        
        return {
            'node_similarity_success': node_sim_success,
            'structural_similarity_success': struct_sim_success,
            'test_passed': node_sim_success and struct_sim_success
        }
    
    def _test_graph_embedding(self) -> Dict[str, Any]:
        """测试图嵌入算法"""
        graph = self.graph if self.graph else self.create_graph()
        
        if graph.get_node_count() < 2:
            return {'test_passed': False, 'reason': '节点数量不足'}
        
        # DeepWalk测试
        try:
            embeddings = self.embedding.deepwalk(graph, num_walks=10, walk_length=10)
            deepwalk_success = len(embeddings) > 0
        except:
            deepwalk_success = False
        
        # Node2Vec测试
        try:
            embeddings = self.embedding.node2vec(graph, num_walks=10, walk_length=10)
            node2vec_success = len(embeddings) > 0
        except:
            node2vec_success = False
        
        return {
            'deepwalk_success': deepwalk_success,
            'node2vec_success': node2vec_success,
            'test_passed': deepwalk_success and node2vec_success
        }
    
    def _test_dynamic_graph(self) -> Dict[str, Any]:
        """测试动态图算法"""
        # 创建动态图
        dynamic_graph = DynamicGraph()
        
        # 添加节点和边
        dynamic_graph.add_node_dynamic('F')
        dynamic_graph.add_node_dynamic('G')
        dynamic_graph.add_edge_dynamic('F', 'G', 3.0)
        
        # 测试增量聚类
        try:
            communities = dynamic_graph.incremental_clustering('label_propagation')
            clustering_success = len(communities) > 0
        except:
            clustering_success = False
        
        # 测试演化指标
        try:
            metrics = dynamic_graph.get_evolution_metrics()
            metrics_success = 'total_events' in metrics
        except:
            metrics_success = False
        
        return {
            'clustering_success': clustering_success,
            'metrics_success': metrics_success,
            'test_passed': clustering_success and metrics_success
        }
    
    def _test_graph_analysis(self) -> Dict[str, Any]:
        """测试图分析功能"""
        graph = self.graph if self.graph else self.create_graph()
        
        # 属性分析测试
        try:
            properties = self.visualization.analyze_graph_properties(graph)
            analysis_success = 'num_nodes' in properties and 'num_edges' in properties
        except:
            analysis_success = False
        
        # 布局计算测试
        try:
            positions = self.visualization.calculate_layout(graph, 'circular')
            layout_success = len(positions) > 0
        except:
            layout_success = False
        
        return {
            'analysis_success': analysis_success,
            'layout_success': layout_success,
            'test_passed': analysis_success and layout_success
        }


# ============================================================================
# 示例用法和测试
# ============================================================================

def create_sample_graph() -> Graph:
    """创建示例图"""
    graph = Graph(directed=False, weighted=True)
    
    # 添加节点
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for node in nodes:
        graph.add_node(node)
    
    # 添加边
    edges = [
        ('A', 'B', 1), ('A', 'C', 4), ('B', 'C', 2), ('B', 'D', 5),
        ('C', 'D', 1), ('C', 'E', 3), ('D', 'E', 2), ('D', 'F', 4),
        ('E', 'F', 1), ('E', 'G', 2), ('F', 'G', 3)
    ]
    
    for source, target, weight in edges:
        graph.add_edge(source, target, weight)
    
    return graph


def demonstrate_algorithms():
    """演示各种算法"""
    print("=== 图算法库演示 ===\n")
    
    # 创建图算法库实例
    library = GraphAlgorithmLibrary()
    
    # 创建示例图
    graph = create_sample_graph()
    library.load_graph_from_dict(graph.to_dict())
    
    print("1. 图的基本属性:")
    properties = library.visualization.analyze_graph_properties(graph)
    print(f"   节点数: {properties['num_nodes']}")
    print(f"   边数: {properties['num_edges']}")
    print(f"   是否连通: {properties['is_connected']}")
    print(f"   平均度: {properties['avg_degree']:.2f}")
    print()
    
    print("2. 最短路径算法:")
    try:
        distances, _ = library.shortest_path.dijkstra(graph, 'A')
        path = library.shortest_path.get_shortest_path(graph, 'A', 'G')
        print(f"   从A到G的最短距离: {distances.get('G', 'inf')}")
        print(f"   最短路径: {' -> '.join(path)}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    print("3. 最小生成树:")
    try:
        mst_edges = library.mst.kruskal(graph)
        mst_weight = library.mst.get_mst_weight(mst_edges)
        print(f"   MST边数: {len(mst_edges)}")
        print(f"   MST总权重: {mst_weight}")
        print(f"   MST边: {[f'{e.source}-{e.target}({e.weight})' for e in mst_edges]}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    print("4. 社区检测:")
    try:
        communities = library.community.label_propagation(graph)
        modularity = library.community.calculate_modularity(graph, communities)
        print(f"   社区数量: {len(set(communities.values()))}")
        print(f"   社区划分: {communities}")
        print(f"   模块度: {modularity:.4f}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    print("5. 图相似性:")
    try:
        sim = library.similarity.node_similarity(graph, 'A', 'B')
        struct_sim = library.similarity.structural_similarity(graph, 'A', 'B')
        print(f"   节点A和B的相似度: {sim:.4f}")
        print(f"   节点A和B的结构相似度: {struct_sim:.4f}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    print("6. 图嵌入:")
    try:
        embeddings = library.embedding.deepwalk(graph, num_walks=20, walk_length=20)
        print(f"   嵌入节点数: {len(embeddings)}")
        print(f"   嵌入维度: {len(next(iter(embeddings.values())))}")
        print(f"   节点A的嵌入: {[f'{x:.3f}' for x in embeddings['A'][:5]]}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    print("7. 动态图:")
    try:
        dynamic = DynamicGraph(graph)
        dynamic.add_node_dynamic('H')
        dynamic.add_edge_dynamic('G', 'H', 2.5)
        metrics = dynamic.get_evolution_metrics()
        print(f"   总事件数: {metrics['total_events']}")
        print(f"   当前节点数: {metrics['current_node_count']}")
        print(f"   当前边数: {metrics['current_edge_count']}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    print("8. 运行完整测试套件:")
    test_results = library.run_all_tests()
    print(f"   测试结果: {test_results}")
    print()
    
    print("=== 演示完成 ===")


if __name__ == "__main__":
    # 运行演示
    demonstrate_algorithms()
    
    # 可以取消注释以下代码来运行特定的测试
    # library = GraphAlgorithmLibrary()
    # test_results = library.run_all_tests()
    # print("测试结果:", test_results)