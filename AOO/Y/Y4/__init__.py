"""
Y4分布式存储管理器包
实现分布式存储、数据分片、存储同步等功能
"""

from .DistributedStorageManager import (
    DistributedStorageManager,
    StorageNode,
    DataShard,
    SyncOperation,
    ConsistentHashing
)

__version__ = "1.0.0"
__author__ = "Y4分布式存储团队"
__description__ = "Y4分布式存储管理器"

__all__ = [
    "DistributedStorageManager",
    "StorageNode", 
    "DataShard",
    "SyncOperation",
    "ConsistentHashing"
]