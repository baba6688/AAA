#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X3分布式缓存管理器

实现分布式缓存的核心功能，包括：
- 分布式存储
- 数据分片
- 缓存同步
- 负载均衡
- 容错处理
- 缓存一致性
- 集群管理
- 性能监控
"""

import hashlib
import json
import threading
import time
import logging
import random
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
from enum import Enum
import copy


class ConsistencyLevel(Enum):
    """一致性级别"""
    ONE = "one"           # 一个节点确认
    QUORUM = "quorum"     # 大多数节点确认
    ALL = "all"           # 所有节点确认


class NodeStatus(Enum):
    """节点状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    SUSPECTED = "suspected"
    DOWN = "down"


@dataclass
class CacheNode:
    """缓存节点"""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: float = 0.0
    load: float = 0.0
    response_time: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        return data


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float
    ttl: float
    version: int = 1
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.timestamp > self.ttl


class ConsistentHash:
    """一致性哈希算法实现"""
    
    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        
        for node in nodes:
            self.add_node(node)
    
    def _hash(self, key: str) -> int:
        """计算哈希值"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: str):
        """添加节点"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node: str):
        """移除节点"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """获取key对应的节点"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # 找到第一个大于等于hash_value的节点
        for ring_key in self.sorted_keys:
            if hash_value <= ring_key:
                return self.ring[ring_key]
        
        # 如果没找到，返回第一个节点（环形）
        return self.ring[self.sorted_keys[0]] if self.sorted_keys else None
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """获取key对应的多个节点（用于复制）"""
        if not self.ring:
            return []
        
        hash_value = self._hash(key)
        nodes = []
        visited = set()
        
        for ring_key in self.sorted_keys:
            if hash_value <= ring_key:
                node = self.ring[ring_key]
                if node not in visited:
                    nodes.append(node)
                    visited.add(node)
                    if len(nodes) >= count:
                        break
        
        # 如果不够，从头开始
        for ring_key in self.sorted_keys:
            node = self.ring[ring_key]
            if node not in visited:
                nodes.append(node)
                visited.add(node)
                if len(nodes) >= count:
                    break
        
        return nodes


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.strategies = {
            'round_robin': self._round_robin,
            'least_connections': self._least_connections,
            'weighted_round_robin': self._weighted_round_robin,
            'random': self._random,
        }
        self.round_robin_index = 0
        self.lock = threading.Lock()
    
    def select_node(self, nodes: List[CacheNode], strategy: str = 'round_robin') -> Optional[CacheNode]:
        """选择节点"""
        healthy_nodes = [node for node in nodes if node.status == NodeStatus.HEALTHY]
        
        if not healthy_nodes:
            return None
        
        if strategy not in self.strategies:
            strategy = 'round_robin'
        
        return self.strategies[strategy](healthy_nodes)
    
    def _round_robin(self, nodes: List[CacheNode]) -> CacheNode:
        """轮询"""
        with self.lock:
            node = nodes[self.round_robin_index % len(nodes)]
            self.round_robin_index += 1
            return node
    
    def _least_connections(self, nodes: List[CacheNode]) -> CacheNode:
        """最少连接"""
        return min(nodes, key=lambda n: n.load)
    
    def _weighted_round_robin(self, nodes: List[CacheNode]) -> CacheNode:
        """加权轮询"""
        # 根据响应时间和负载计算权重
        weights = []
        for node in nodes:
            weight = 1.0 / (node.response_time + 1) * (1.0 / (node.load + 1))
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(nodes)
        
        r = random.uniform(0, total_weight)
        cumsum = 0
        for i, weight in enumerate(weights):
            cumsum += weight
            if r <= cumsum:
                return nodes[i]
        
        return nodes[-1]
    
    def _random(self, nodes: List[CacheNode]) -> CacheNode:
        """随机"""
        return random.choice(nodes)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = defaultdict(dict)
        self.lock = threading.Lock()
    
    def record_operation(self, node_id: str, operation: str, duration: float, success: bool):
        """记录操作指标"""
        with self.lock:
            if node_id not in self.metrics:
                self.metrics[node_id] = {
                    'operations': defaultdict(int),
                    'latency': [],
                    'errors': 0,
                    'total_requests': 0,
                }
            
            node_metrics = self.metrics[node_id]
            node_metrics['operations'][operation] += 1
            node_metrics['total_requests'] += 1
            
            if not success:
                node_metrics['errors'] += 1
            
            node_metrics['latency'].append(duration)
            
            # 保持最近1000个延迟记录
            if len(node_metrics['latency']) > 1000:
                node_metrics['latency'] = node_metrics['latency'][-1000:]
    
    def get_node_stats(self, node_id: str) -> Dict:
        """获取节点统计信息"""
        with self.lock:
            if node_id not in self.metrics:
                return {}
            
            node_metrics = self.metrics[node_id]
            latencies = node_metrics['latency']
            
            return {
                'total_requests': node_metrics['total_requests'],
                'errors': node_metrics['errors'],
                'error_rate': node_metrics['errors'] / max(node_metrics['total_requests'], 1),
                'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
                'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
                'operations': dict(node_metrics['operations']),
            }
    
    def get_cluster_stats(self) -> Dict:
        """获取集群统计信息"""
        with self.lock:
            total_requests = sum(m['total_requests'] for m in self.metrics.values())
            total_errors = sum(m['errors'] for m in self.metrics.values())
            
            all_latencies = []
            for m in self.metrics.values():
                all_latencies.extend(m['latency'])
            
            return {
                'total_requests': total_requests,
                'total_errors': total_errors,
                'cluster_error_rate': total_errors / max(total_requests, 1),
                'avg_cluster_latency': sum(all_latencies) / len(all_latencies) if all_latencies else 0,
                'active_nodes': len(self.metrics),
            }


class DistributedCacheManager:
    """X3分布式缓存管理器"""
    
    def __init__(self, 
                 consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM,
                 load_balance_strategy: str = 'round_robin',
                 replication_factor: int = 2,
                 heartbeat_interval: float = 5.0,
                 failure_detection_timeout: float = 15.0):
        """
        初始化分布式缓存管理器
        
        Args:
            consistency_level: 一致性级别
            load_balance_strategy: 负载均衡策略
            replication_factor: 复制因子
            heartbeat_interval: 心跳间隔
            failure_detection_timeout: 故障检测超时
        """
        self.consistency_level = consistency_level
        self.replication_factor = replication_factor
        self.heartbeat_interval = heartbeat_interval
        self.failure_detection_timeout = failure_detection_timeout
        
        # 存储组件
        self.nodes: Dict[str, CacheNode] = {}
        self.cache_data: Dict[str, Dict[str, CacheEntry]] = defaultdict(dict)
        self.consistent_hash = ConsistentHash([])
        self.load_balancer = LoadBalancer()
        self.monitor = PerformanceMonitor()
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
        }
        
        # 启动后台任务
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        
        self.heartbeat_thread.start()
        self.cleanup_thread.start()
    
    def add_node(self, node_id: str, host: str, port: int) -> bool:
        """添加缓存节点"""
        with self.lock:
            if node_id in self.nodes:
                return False
            
            node = CacheNode(node_id, host, port)
            self.nodes[node_id] = node
            self.consistent_hash.add_node(node_id)
            
            logging.info(f"节点 {node_id} 已添加到集群")
            return True
    
    def remove_node(self, node_id: str) -> bool:
        """移除缓存节点"""
        with self.lock:
            if node_id not in self.nodes:
                return False
            
            # 从一致性哈希环中移除
            self.consistent_hash.remove_node(node_id)
            
            # 重新分配该节点的数据
            self._redistribute_data(node_id)
            
            # 移除节点
            del self.nodes[node_id]
            
            logging.info(f"节点 {node_id} 已从集群移除")
            return True
    
    def _redistribute_data(self, failed_node_id: str):
        """重新分配失效节点的数据"""
        with self.lock:
            # 获取失效节点的数据
            failed_data = self.cache_data.get(failed_node_id, {}).copy()
            
            # 为每个key找到新的节点
            for key, entry in failed_data.items():
                target_nodes = self.consistent_hash.get_nodes(key, self.replication_factor)
                
                # 复制数据到新节点
                for node_id in target_nodes:
                    if node_id != failed_node_id:
                        self.cache_data[node_id][key] = copy.deepcopy(entry)
            
            # 清除失效节点的数据
            if failed_node_id in self.cache_data:
                del self.cache_data[failed_node_id]
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        try:
            with self.lock:
                # 找到存储该key的节点
                target_nodes = self.consistent_hash.get_nodes(key, self.replication_factor)
                
                if not target_nodes:
                    self.stats['misses'] += 1
                    return None
                
                # 尝试从第一个节点获取
                for node_id in target_nodes:
                    if node_id in self.cache_data:
                        entry = self.cache_data[node_id].get(key)
                        if entry and not entry.is_expired():
                            self.stats['hits'] += 1
                            self.monitor.record_operation(node_id, 'get', 
                                                        time.time() - start_time, True)
                            return entry.value
                
                # 所有节点都没有找到有效数据
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            logging.error(f"Get操作失败: {e}")
            self.monitor.record_operation('unknown', 'get', time.time() - start_time, False)
            return None
    
    def set(self, key: str, value: Any, ttl: float = 3600) -> bool:
        """设置缓存值"""
        start_time = time.time()
        
        try:
            with self.lock:
                # 找到存储该key的节点
                target_nodes = self.consistent_hash.get_nodes(key, self.replication_factor)
                
                if not target_nodes:
                    return False
                
                # 创建缓存条目
                entry = CacheEntry(key, value, time.time(), ttl)
                
                # 写入到所有目标节点
                success_count = 0
                for node_id in target_nodes:
                    try:
                        self.cache_data[node_id][key] = copy.deepcopy(entry)
                        success_count += 1
                    except Exception as e:
                        logging.error(f"写入节点 {node_id} 失败: {e}")
                
                # 检查是否满足一致性要求
                required_nodes = self._get_required_nodes_count()
                if success_count >= required_nodes:
                    self.stats['sets'] += 1
                    self.monitor.record_operation('cluster', 'set', 
                                                time.time() - start_time, True)
                    return True
                else:
                    logging.warning(f"一致性检查失败: 成功 {success_count}, 要求 {required_nodes}")
                    return False
                    
        except Exception as e:
            logging.error(f"Set操作失败: {e}")
            self.monitor.record_operation('cluster', 'set', time.time() - start_time, False)
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        start_time = time.time()
        
        try:
            with self.lock:
                # 找到存储该key的节点
                target_nodes = self.consistent_hash.get_nodes(key, self.replication_factor)
                
                if not target_nodes:
                    return False
                
                # 从所有目标节点删除
                success_count = 0
                for node_id in target_nodes:
                    if node_id in self.cache_data and key in self.cache_data[node_id]:
                        del self.cache_data[node_id][key]
                        success_count += 1
                
                # 检查是否满足一致性要求
                required_nodes = self._get_required_nodes_count()
                if success_count >= required_nodes:
                    self.stats['deletes'] += 1
                    self.monitor.record_operation('cluster', 'delete', 
                                                time.time() - start_time, True)
                    return True
                else:
                    return False
                    
        except Exception as e:
            logging.error(f"Delete操作失败: {e}")
            self.monitor.record_operation('cluster', 'delete', time.time() - start_time, False)
            return False
    
    def _get_required_nodes_count(self) -> int:
        """获取一致性要求的最少节点数"""
        total_nodes = len(self.nodes)
        
        if self.consistency_level == ConsistencyLevel.ONE:
            return 1
        elif self.consistency_level == ConsistencyLevel.QUORUM:
            return (total_nodes // 2) + 1
        elif self.consistency_level == ConsistencyLevel.ALL:
            return total_nodes
        
        return max(1, total_nodes // 2 + 1)
    
    def _heartbeat_worker(self):
        """心跳检测工作线程"""
        while self.running:
            try:
                current_time = time.time()
                
                with self.lock:
                    for node_id, node in self.nodes.items():
                        # 检查心跳超时
                        if current_time - node.last_heartbeat > self.failure_detection_timeout:
                            if node.status == NodeStatus.HEALTHY:
                                node.status = NodeStatus.SUSPECTED
                                logging.warning(f"节点 {node_id} 疑似故障")
                            elif node.status == NodeStatus.SUSPECTED:
                                node.status = NodeStatus.DOWN
                                logging.error(f"节点 {node_id} 确认故障")
                                
                                # 重新分配数据
                                self._redistribute_data(node_id)
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logging.error(f"心跳检测错误: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _cleanup_worker(self):
        """清理过期数据工作线程"""
        while self.running:
            try:
                current_time = time.time()
                
                with self.lock:
                    expired_keys = []
                    
                    for node_id, node_data in self.cache_data.items():
                        for key, entry in node_data.items():
                            if entry.is_expired():
                                expired_keys.append((node_id, key))
                    
                    # 删除过期数据
                    for node_id, key in expired_keys:
                        del self.cache_data[node_id][key]
                
                if expired_keys:
                    logging.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
                
                time.sleep(60)  # 每分钟清理一次
                
            except Exception as e:
                logging.error(f"清理过期数据错误: {e}")
                time.sleep(60)
    
    def update_node_heartbeat(self, node_id: str):
        """更新节点心跳"""
        with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.last_heartbeat = time.time()
                
                # 如果节点之前不健康，标记为健康
                if node.status in [NodeStatus.SUSPECTED, NodeStatus.DOWN]:
                    node.status = NodeStatus.HEALTHY
                    logging.info(f"节点 {node_id} 恢复健康")
    
    def get_cluster_info(self) -> Dict:
        """获取集群信息"""
        with self.lock:
            return {
                'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
                'total_nodes': len(self.nodes),
                'healthy_nodes': len([n for n in self.nodes.values() 
                                    if n.status == NodeStatus.HEALTHY]),
                'replication_factor': self.replication_factor,
                'consistency_level': self.consistency_level.value,
            }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            hit_rate = self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1)
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'cluster_stats': self.monitor.get_cluster_stats(),
            }
    
    def get_node_stats(self, node_id: str) -> Dict:
        """获取指定节点统计信息"""
        return self.monitor.get_node_stats(node_id)
    
    def shutdown(self):
        """关闭缓存管理器"""
        logging.info("正在关闭分布式缓存管理器...")
        
        self.running = False
        
        # 等待后台线程结束
        if self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
        
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        logging.info("分布式缓存管理器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建分布式缓存管理器
    cache_manager = DistributedCacheManager(
        consistency_level=ConsistencyLevel.QUORUM,
        load_balance_strategy='round_robin',
        replication_factor=2
    )
    
    try:
        # 添加节点
        cache_manager.add_node("node1", "192.168.1.1", 6379)
        cache_manager.add_node("node2", "192.168.1.2", 6379)
        cache_manager.add_node("node3", "192.168.1.3", 6379)
        
        # 测试缓存操作
        cache_manager.set("user:1", {"name": "张三", "age": 25}, ttl=300)
        cache_manager.set("user:2", {"name": "李四", "age": 30}, ttl=300)
        
        # 获取缓存
        user1 = cache_manager.get("user:1")
        user2 = cache_manager.get("user:2")
        
        print(f"用户1: {user1}")
        print(f"用户2: {user2}")
        
        # 获取统计信息
        stats = cache_manager.get_stats()
        print(f"缓存统计: {stats}")
        
        # 获取集群信息
        cluster_info = cache_manager.get_cluster_info()
        print(f"集群信息: {cluster_info}")
        
    finally:
        cache_manager.shutdown()