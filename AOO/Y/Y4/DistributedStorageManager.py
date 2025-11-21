"""
Y4分布式存储管理器
实现分布式存储、数据分片、存储同步等功能
"""

import json
import hashlib
import threading
import time
import random
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import copy


@dataclass
class StorageNode:
    """存储节点信息"""
    node_id: str
    host: str
    port: int
    capacity: int  # 存储容量(MB)
    used_capacity: int = 0  # 已使用容量(MB)
    status: str = "active"  # active, inactive, failed
    last_heartbeat: float = 0.0
    load_score: float = 0.0  # 负载评分
    response_time: float = 0.0  # 响应时间(ms)


@dataclass
class DataShard:
    """数据分片信息"""
    shard_id: str
    data: bytes
    size: int
    checksum: str
    primary_node: str  # 主节点ID
    replica_nodes: List[str]  # 副本节点ID列表
    created_time: datetime
    last_accessed: datetime
    access_count: int = 0


@dataclass
class SyncOperation:
    """同步操作记录"""
    operation_id: str
    shard_id: str
    source_node: str
    target_node: str
    operation_type: str  # create, update, delete
    timestamp: datetime
    status: str  # pending, success, failed
    retry_count: int = 0


class ConsistentHashing:
    """一致性哈希实现"""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        
    def _hash(self, key: str) -> int:
        """计算哈希值"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node_id: str):
        """添加节点到哈希环"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node_id
            self.sorted_keys.append(hash_value)
        
        self.sorted_keys.sort()
    
    def remove_node(self, node_id: str):
        """从哈希环移除节点"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
                self.sorted_keys.remove(hash_value)
    
    def get_node(self, key: str) -> str:
        """获取key对应的节点"""
        if not self.ring:
            return None
            
        hash_value = self._hash(key)
        
        # 找到第一个大于等于hash_value的节点
        for ring_key in self.sorted_keys:
            if hash_value <= ring_key:
                return self.ring[ring_key]
        
        # 如果没找到，返回第一个节点（环形）
        return self.ring[self.sorted_keys[0]]


class DistributedStorageManager:
    """分布式存储管理器主类"""
    
    def __init__(self, replica_count: int = 3, shard_size_limit: int = 1024 * 1024):
        """
        初始化分布式存储管理器
        
        Args:
            replica_count: 副本数量
            shard_size_limit: 分片大小限制(字节)
        """
        self.replica_count = replica_count
        self.shard_size_limit = shard_size_limit
        
        # 存储节点管理
        self.nodes: Dict[str, StorageNode] = {}
        self.consistent_hash = ConsistentHashing()
        self.node_lock = threading.RLock()
        
        # 数据分片管理
        self.shards: Dict[str, DataShard] = {}
        self.shard_lock = threading.RLock()
        
        # 同步管理
        self.sync_operations: Dict[str, SyncOperation] = {}
        self.sync_queue: List[SyncOperation] = []
        self.sync_lock = threading.Lock()
        
        # 性能监控
        self.metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_response_time': 0.0,
            'node_failures': 0,
            'sync_operations': 0
        }
        self.metrics_lock = threading.Lock()
        
        # 启动后台线程
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.load_balance_thread = threading.Thread(target=self._load_balancer, daemon=True)
        
        self.heartbeat_thread.start()
        self.sync_thread.start()
        self.load_balance_thread.start()
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_node(self, node_id: str, host: str, port: int, capacity: int) -> bool:
        """
        添加存储节点
        
        Args:
            node_id: 节点ID
            host: 主机地址
            port: 端口
            capacity: 容量(MB)
            
        Returns:
            bool: 是否成功添加
        """
        with self.node_lock:
            if node_id in self.nodes:
                self.logger.warning(f"节点 {node_id} 已存在")
                return False
            
            node = StorageNode(
                node_id=node_id,
                host=host,
                port=port,
                capacity=capacity,
                last_heartbeat=time.time()
            )
            
            self.nodes[node_id] = node
            self.consistent_hash.add_node(node_id)
            
            self.logger.info(f"成功添加节点 {node_id} (容量: {capacity}MB)")
            return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        移除存储节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            bool: 是否成功移除
        """
        with self.node_lock:
            if node_id not in self.nodes:
                self.logger.warning(f"节点 {node_id} 不存在")
                return False
            
            # 检查节点上是否有数据
            shards_on_node = []
            for shard_id, shard in self.shards.items():
                if node_id in [shard.primary_node] + shard.replica_nodes:
                    shards_on_node.append(shard_id)
            
            if shards_on_node:
                self.logger.error(f"节点 {node_id} 上还有 {len(shards_on_node)} 个分片，无法移除")
                return False
            
            # 移除节点
            del self.nodes[node_id]
            self.consistent_hash.remove_node(node_id)
            
            self.logger.info(f"成功移除节点 {node_id}")
            return True
    
    def store_data(self, key: str, data: bytes) -> bool:
        """
        存储数据
        
        Args:
            key: 数据键
            data: 数据内容
            
        Returns:
            bool: 是否成功存储
        """
        start_time = time.time()
        
        try:
            # 检查数据大小
            if len(data) > self.shard_size_limit:
                return self._store_large_data(key, data)
            
            # 计算数据哈希
            checksum = hashlib.md5(data).hexdigest()
            
            # 使用一致性哈希选择主节点
            primary_node = self.consistent_hash.get_node(key)
            if not primary_node:
                self.logger.error("没有可用的存储节点")
                return False
            
            # 选择副本节点
            replica_nodes = self._select_replica_nodes(primary_node, self.replica_count - 1)
            
            # 创建分片
            shard_id = self._generate_shard_id(key)
            shard = DataShard(
                shard_id=shard_id,
                data=data,
                size=len(data),
                checksum=checksum,
                primary_node=primary_node,
                replica_nodes=replica_nodes,
                created_time=datetime.now(),
                last_accessed=datetime.now()
            )
            
            with self.shard_lock:
                self.shards[shard_id] = shard
            
            # 存储到各个节点（模拟）
            success_nodes = [primary_node] + replica_nodes
            if self._store_to_nodes(shard, success_nodes):
                self._update_metrics(True, time.time() - start_time)
                self.logger.info(f"成功存储数据 {key} 到节点 {success_nodes}")
                return True
            else:
                self._update_metrics(False, time.time() - start_time)
                return False
                
        except Exception as e:
            self._update_metrics(False, time.time() - start_time)
            self.logger.error(f"存储数据失败: {e}")
            return False
    
    def retrieve_data(self, key: str) -> Optional[bytes]:
        """
        检索数据
        
        Args:
            key: 数据键
            
        Returns:
            Optional[bytes]: 数据内容，失败返回None
        """
        start_time = time.time()
        
        try:
            shard_id = self._generate_shard_id(key)
            
            with self.shard_lock:
                if shard_id not in self.shards:
                    # 检查是否为大数据分片
                    return self._retrieve_large_data(key)
                
                shard = self.shards[shard_id]
                shard.last_accessed = datetime.now()
                shard.access_count += 1
            
            # 尝试从主节点获取数据
            if self._is_node_available(shard.primary_node):
                data = self._retrieve_from_node(shard, shard.primary_node)
                if data:
                    self._update_metrics(True, time.time() - start_time)
                    return data
            
            # 主节点不可用，尝试从副本节点获取
            for replica_node in shard.replica_nodes:
                if self._is_node_available(replica_node):
                    data = self._retrieve_from_node(shard, replica_node)
                    if data:
                        # 触发数据同步，将数据同步回主节点
                        self._schedule_sync(shard_id, replica_node, shard.primary_node)
                        self._update_metrics(True, time.time() - start_time)
                        return data
            
            self._update_metrics(False, time.time() - start_time)
            self.logger.error(f"无法从任何节点获取数据 {key}")
            return None
            
        except Exception as e:
            self._update_metrics(False, time.time() - start_time)
            self.logger.error(f"检索数据失败: {e}")
            return None
    
    def delete_data(self, key: str) -> bool:
        """
        删除数据
        
        Args:
            key: 数据键
            
        Returns:
            bool: 是否成功删除
        """
        start_time = time.time()
        
        try:
            shard_id = self._generate_shard_id(key)
            
            with self.shard_lock:
                if shard_id not in self.shards:
                    # 检查是否为大数据分片
                    return self._delete_large_data(key)
                
                shard = self.shards[shard_id]
            
            # 从所有节点删除数据
            all_nodes = [shard.primary_node] + shard.replica_nodes
            success_nodes = []
            
            for node_id in all_nodes:
                if self._delete_from_node(shard, node_id):
                    success_nodes.append(node_id)
            
            # 从管理器中移除分片
            with self.shard_lock:
                del self.shards[shard_id]
            
            if len(success_nodes) == len(all_nodes):
                self._update_metrics(True, time.time() - start_time)
                self.logger.info(f"成功删除数据 {key} 从节点 {success_nodes}")
                return True
            else:
                self.logger.warning(f"部分节点删除失败: {success_nodes}")
                self._update_metrics(False, time.time() - start_time)
                return False
                
        except Exception as e:
            self._update_metrics(False, time.time() - start_time)
            self.logger.error(f"删除数据失败: {e}")
            return False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """
        获取集群状态
        
        Returns:
            Dict: 集群状态信息
        """
        with self.node_lock:
            active_nodes = sum(1 for node in self.nodes.values() if node.status == "active")
            total_capacity = sum(node.capacity for node in self.nodes.values())
            total_used = sum(node.used_capacity for node in self.nodes.values())
        
        with self.shard_lock:
            total_shards = len(self.shards)
            total_data_size = sum(shard.size for shard in self.shards.values())
        
        return {
            'cluster_info': {
                'total_nodes': len(self.nodes),
                'active_nodes': active_nodes,
                'total_capacity_mb': total_capacity,
                'used_capacity_mb': total_used,
                'capacity_utilization': total_used / total_capacity if total_capacity > 0 else 0,
                'total_shards': total_shards,
                'total_data_size_mb': total_data_size / (1024 * 1024)
            },
            'nodes': {
                node_id: {
                    'status': node.status,
                    'capacity_mb': node.capacity,
                    'used_capacity_mb': node.used_capacity,
                    'load_score': node.load_score,
                    'response_time_ms': node.response_time,
                    'last_heartbeat': node.last_heartbeat
                }
                for node_id, node in self.nodes.items()
            },
            'metrics': copy.deepcopy(self.metrics)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            Dict: 性能指标
        """
        with self.metrics_lock:
            metrics = copy.deepcopy(self.metrics)
        
        # 计算成功率
        if metrics['total_operations'] > 0:
            metrics['success_rate'] = metrics['successful_operations'] / metrics['total_operations']
        else:
            metrics['success_rate'] = 0.0
        
        return metrics
    
    def rebalance_cluster(self) -> bool:
        """
        重新平衡集群负载
        
        Returns:
            bool: 是否成功重新平衡
        """
        try:
            self.logger.info("开始集群负载重新平衡")
            
            # 分析负载分布
            load_analysis = self._analyze_load_distribution()
            
            # 制定重新平衡计划
            rebalance_plan = self._create_rebalance_plan(load_analysis)
            
            # 执行重新平衡
            success = self._execute_rebalance_plan(rebalance_plan)
            
            if success:
                self.logger.info("集群负载重新平衡完成")
            else:
                self.logger.error("集群负载重新平衡失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"重新平衡集群失败: {e}")
            return False
    
    def shutdown(self):
        """关闭分布式存储管理器"""
        self.logger.info("正在关闭分布式存储管理器...")
        
        self.running = False
        
        # 等待后台线程结束
        if self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
        if self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)
        if self.load_balance_thread.is_alive():
            self.load_balance_thread.join(timeout=5)
        
        self.logger.info("分布式存储管理器已关闭")
    
    # 私有方法
    
    def _generate_shard_id(self, key: str) -> str:
        """生成分片ID"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _select_replica_nodes(self, primary_node: str, count: int) -> List[str]:
        """选择副本节点"""
        available_nodes = [node_id for node_id in self.nodes.keys() if node_id != primary_node]
        random.shuffle(available_nodes)
        return available_nodes[:count]
    
    def _store_to_nodes(self, shard: DataShard, node_ids: List[str]) -> bool:
        """存储数据到指定节点（模拟）"""
        try:
            for node_id in node_ids:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    node.used_capacity += shard.size
                    # 模拟网络延迟
                    time.sleep(0.001)
            return True
        except Exception:
            return False
    
    def _retrieve_from_node(self, shard: DataShard, node_id: str) -> Optional[bytes]:
        """从指定节点检索数据（模拟）"""
        try:
            if node_id in self.nodes and self.nodes[node_id].status == "active":
                # 模拟网络延迟
                time.sleep(0.001)
                return shard.data
            return None
        except Exception:
            return None
    
    def _delete_from_node(self, shard: DataShard, node_id: str) -> bool:
        """从指定节点删除数据（模拟）"""
        try:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.used_capacity = max(0, node.used_capacity - shard.size)
                # 模拟网络延迟
                time.sleep(0.001)
            return True
        except Exception:
            return False
    
    def _is_node_available(self, node_id: str) -> bool:
        """检查节点是否可用"""
        return (node_id in self.nodes and 
                self.nodes[node_id].status == "active" and
                time.time() - self.nodes[node_id].last_heartbeat < 30)
    
    def _retrieve_large_data(self, key: str) -> Optional[bytes]:
        """检索大数据分片"""
        try:
            # 查找所有相关分片
            shard_parts = []
            shard_count = 0
            
            while True:
                shard_key = f"{key}_shard_{shard_count}"
                shard_id = self._generate_shard_id(shard_key)
                
                with self.shard_lock:
                    if shard_id not in self.shards:
                        break
                    
                    shard = self.shards[shard_id]
                    shard_parts.append((shard_count, shard))
                
                shard_count += 1
                # 防止无限循环
                if shard_count > 1000:
                    break
            
            if not shard_parts:
                self.logger.warning(f"数据 {key} 不存在")
                return None
            
            # 按分片序号重组数据
            shard_parts.sort(key=lambda x: x[0])  # 按分片序号排序
            reassembled_data = b""
            
            for shard_index, shard in shard_parts:
                # 尝试从主节点获取数据
                data = None
                if self._is_node_available(shard.primary_node):
                    data = self._retrieve_from_node(shard, shard.primary_node)
                else:
                    # 尝试从副本节点获取
                    for replica_node in shard.replica_nodes:
                        if self._is_node_available(replica_node):
                            data = self._retrieve_from_node(shard, replica_node)
                            break
                
                if data is None:
                    self.logger.error(f"无法获取分片 {shard_index} 的数据")
                    return None
                
                reassembled_data += data
            
            return reassembled_data
            
        except Exception as e:
            self.logger.error(f"检索大数据失败: {e}")
            return None
    
    def _delete_large_data(self, key: str) -> bool:
        """删除大数据分片"""
        try:
            # 查找所有相关分片
            shard_count = 0
            deleted_shards = []
            
            while True:
                shard_key = f"{key}_shard_{shard_count}"
                shard_id = self._generate_shard_id(shard_key)
                
                with self.shard_lock:
                    if shard_id not in self.shards:
                        break
                    
                    shard = self.shards[shard_id]
                    deleted_shards.append(shard)
                
                shard_count += 1
                # 防止无限循环
                if shard_count > 1000:
                    break
            
            if not deleted_shards:
                self.logger.warning(f"数据 {key} 不存在")
                return False
            
            # 删除所有分片
            success_count = 0
            for shard in deleted_shards:
                all_nodes = [shard.primary_node] + shard.replica_nodes
                
                shard_success = True
                for node_id in all_nodes:
                    if not self._delete_from_node(shard, node_id):
                        shard_success = False
                        break
                
                if shard_success:
                    with self.shard_lock:
                        if shard.shard_id in self.shards:
                            del self.shards[shard.shard_id]
                    success_count += 1
            
            if success_count == len(deleted_shards):
                self.logger.info(f"成功删除大数据 {key} 的 {success_count} 个分片")
                return True
            else:
                self.logger.warning(f"部分分片删除失败: {success_count}/{len(deleted_shards)}")
                return False
            
        except Exception as e:
            self.logger.error(f"删除大数据失败: {e}")
            return False
    
    def _store_large_data(self, key: str, data: bytes) -> bool:
        """存储大数据的分片处理"""
        # 简单的大数据分片处理
        shard_size = self.shard_size_limit
        offset = 0
        shard_count = 0
        
        while offset < len(data):
            chunk = data[offset:offset + shard_size]
            chunk_key = f"{key}_shard_{shard_count}"
            
            if not self.store_data(chunk_key, chunk):
                return False
            
            offset += shard_size
            shard_count += 1
        
        return True
    
    def _schedule_sync(self, shard_id: str, source_node: str, target_node: str):
        """安排数据同步"""
        sync_op = SyncOperation(
            operation_id=f"sync_{int(time.time())}_{shard_id}",
            shard_id=shard_id,
            source_node=source_node,
            target_node=target_node,
            operation_type="update",
            timestamp=datetime.now(),
            status="pending"
        )
        
        with self.sync_lock:
            self.sync_queue.append(sync_op)
            self.sync_operations[sync_op.operation_id] = sync_op
    
    def _update_metrics(self, success: bool, response_time: float):
        """更新性能指标"""
        with self.metrics_lock:
            self.metrics['total_operations'] += 1
            if success:
                self.metrics['successful_operations'] += 1
            else:
                self.metrics['failed_operations'] += 1
            
            # 更新平均响应时间
            total_ops = self.metrics['total_operations']
            current_avg = self.metrics['average_response_time']
            self.metrics['average_response_time'] = (
                (current_avg * (total_ops - 1) + response_time) / total_ops
            )
    
    def _analyze_load_distribution(self) -> Dict[str, Any]:
        """分析负载分布"""
        load_info = {}
        
        for node_id, node in self.nodes.items():
            utilization = node.used_capacity / node.capacity if node.capacity > 0 else 0
            load_info[node_id] = {
                'utilization': utilization,
                'available_capacity': node.capacity - node.used_capacity,
                'shard_count': sum(1 for shard in self.shards.values() 
                                 if node_id in [shard.primary_node] + shard.replica_nodes)
            }
        
        return load_info
    
    def _create_rebalance_plan(self, load_analysis: Dict[str, Any]) -> List[Dict]:
        """制定重新平衡计划"""
        # 简单的重新平衡策略：高负载节点向低负载节点迁移数据
        plan = []
        
        # 按负载排序
        sorted_nodes = sorted(load_analysis.items(), 
                            key=lambda x: x[1]['utilization'], 
                            reverse=True)
        
        # 找到高负载和低负载节点
        high_load_threshold = 0.8
        low_load_threshold = 0.3
        
        high_load_nodes = [node for node, info in sorted_nodes 
                         if info['utilization'] > high_load_threshold]
        low_load_nodes = [node for node, info in sorted_nodes 
                        if info['utilization'] < low_load_threshold]
        
        # 制定迁移计划
        for high_node in high_load_nodes:
            for low_node in low_load_nodes:
                if load_analysis[high_node]['shard_count'] > 0:
                    plan.append({
                        'action': 'migrate',
                        'source': high_node,
                        'target': low_node,
                        'shard_count': 1  # 每次迁移1个分片
                    })
        
        return plan
    
    def _execute_rebalance_plan(self, plan: List[Dict]) -> bool:
        """执行重新平衡计划"""
        try:
            for operation in plan:
                if operation['action'] == 'migrate':
                    # 模拟数据迁移
                    self.logger.info(f"从节点 {operation['source']} 迁移数据到节点 {operation['target']}")
                    time.sleep(0.1)  # 模拟迁移时间
            return True
        except Exception as e:
            self.logger.error(f"执行重新平衡计划失败: {e}")
            return False
    
    def _heartbeat_monitor(self):
        """心跳监控线程"""
        while self.running:
            try:
                current_time = time.time()
                
                with self.node_lock:
                    for node_id, node in self.nodes.items():
                        if current_time - node.last_heartbeat > 60:  # 60秒超时
                            if node.status != "failed":
                                node.status = "failed"
                                self.logger.warning(f"节点 {node_id} 标记为失败")
                                
                                with self.metrics_lock:
                                    self.metrics['node_failures'] += 1
                
                time.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                self.logger.error(f"心跳监控异常: {e}")
                time.sleep(5)
    
    def _sync_worker(self):
        """同步工作线程"""
        while self.running:
            try:
                with self.sync_lock:
                    if self.sync_queue:
                        operation = self.sync_queue.pop(0)
                    else:
                        operation = None
                
                if operation:
                    self._process_sync_operation(operation)
                
                time.sleep(0.1)  # 避免CPU占用过高
                
            except Exception as e:
                self.logger.error(f"同步工作异常: {e}")
                time.sleep(1)
    
    def _process_sync_operation(self, operation: SyncOperation):
        """处理同步操作"""
        try:
            # 模拟数据同步
            if self._is_node_available(operation.source_node):
                # 同步成功
                operation.status = "success"
                self.logger.info(f"同步操作 {operation.operation_id} 完成")
                
                with self.metrics_lock:
                    self.metrics['sync_operations'] += 1
            else:
                # 同步失败，准备重试
                operation.retry_count += 1
                if operation.retry_count < 3:
                    operation.status = "pending"
                    with self.sync_lock:
                        self.sync_queue.append(operation)
                else:
                    operation.status = "failed"
                    self.logger.error(f"同步操作 {operation.operation_id} 失败")
                    
        except Exception as e:
            self.logger.error(f"处理同步操作异常: {e}")
            operation.status = "failed"
    
    def _load_balancer(self):
        """负载均衡线程"""
        while self.running:
            try:
                # 更新节点负载评分
                with self.node_lock:
                    for node_id, node in self.nodes.items():
                        # 基于利用率、响应时间等计算负载评分
                        utilization = node.used_capacity / node.capacity if node.capacity > 0 else 0
                        node.load_score = utilization * 0.7 + (node.response_time / 1000) * 0.3
                
                time.sleep(30)  # 每30秒更新一次
                
            except Exception as e:
                self.logger.error(f"负载均衡异常: {e}")
                time.sleep(10)