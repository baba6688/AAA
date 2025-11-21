"""
X6缓存清理器 - 主要实现
提供缓存清理的完整功能，包括自动清理、手动清理、多种清理策略等
"""

import os
import time
import json
import threading
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import OrderedDict
import heapq
from concurrent.futures import ThreadPoolExecutor

# 可选依赖
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class CleanStrategy(Enum):
    """清理策略枚举"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    TTL = "ttl"  # 基于TTL过期时间
    FIFO = "fifo"  # 先进先出
    SIZE = "size"  # 基于缓存大小
    HYBRID = "hybrid"  # 混合策略


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    size: int
    access_count: int
    last_access: float
    created_time: float
    ttl: Optional[float] = None
    metadata: Optional[Dict] = None

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_time > self.ttl


class CleanStats:
    """清理统计"""
    
    def __init__(self):
        self.total_cleaned = 0
        self.total_size_cleaned = 0
        self.clean_operations = 0
        self.average_clean_time = 0.0
        self.strategy_stats = {}
        self.start_time = time.time()
        self.last_clean_time = None
        
    def record_clean(self, count: int, size: int, duration: float, strategy: str):
        """记录清理操作"""
        self.total_cleaned += count
        self.total_size_cleaned += size
        self.clean_operations += 1
        self.last_clean_time = time.time()
        
        # 更新平均清理时间
        if self.clean_operations == 1:
            self.average_clean_time = duration
        else:
            self.average_clean_time = (
                (self.average_clean_time * (self.clean_operations - 1) + duration) 
                / self.clean_operations
            )
        
        # 更新策略统计
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {"count": 0, "size": 0}
        self.strategy_stats[strategy]["count"] += count
        self.strategy_stats[strategy]["size"] += size
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        uptime = time.time() - self.start_time
        return {
            "total_cleaned": self.total_cleaned,
            "total_size_cleaned": self.total_size_cleaned,
            "clean_operations": self.clean_operations,
            "average_clean_time": self.average_clean_time,
            "strategy_stats": self.strategy_stats,
            "uptime": uptime,
            "last_clean_time": self.last_clean_time
        }


class CacheCleaner:
    """X6缓存清理器"""
    
    def __init__(self, cache_dir: str = "/tmp/x6_cache", 
                 auto_clean_interval: int = 3600,
                 max_cache_size: int = 1024 * 1024 * 1024,  # 1GB
                 db_path: str = None):
        """
        初始化缓存清理器
        
        Args:
            cache_dir: 缓存目录
            auto_clean_interval: 自动清理间隔（秒）
            max_cache_size: 最大缓存大小（字节）
            db_path: 数据库路径
        """
        self.cache_dir = cache_dir
        self.auto_clean_interval = auto_clean_interval
        self.max_cache_size = max_cache_size
        self.db_path = db_path or os.path.join(cache_dir, "cache_cleaner.db")
        
        # 缓存存储
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # 用于LRU
        self._frequency_heap = []  # 用于LFU
        
        # 统计和监控
        self.stats = CleanStats()
        self._lock = threading.RLock()
        self._auto_clean_thread = None
        self._is_running = False
        
        # 清理回调
        self.clean_callbacks: List[Callable] = []
        
        # 配置日志（必须在其他操作之前）
        self._setup_logging()
        
        # 初始化数据库
        self._init_database()
        
        # 加载现有缓存
        self._load_cache()
        
        # 启动自动清理
        self.start_auto_clean()
    
    def _init_database(self):
        """初始化数据库"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建缓存表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value TEXT,
                size INTEGER,
                access_count INTEGER,
                last_access REAL,
                created_time REAL,
                ttl REAL,
                metadata TEXT
            )
        ''')
        
        # 创建清理日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clean_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                strategy TEXT,
                count INTEGER,
                size INTEGER,
                duration REAL,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.cache_dir, "cache_cleaner.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_cache(self):
        """从数据库加载缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM cache_entries')
        for row in cursor.fetchall():
            key, value_str, size, access_count, last_access, created_time, ttl, metadata_str = row
            
            # 解析JSON数据
            value = json.loads(value_str)
            metadata = json.loads(metadata_str) if metadata_str else None
            
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                access_count=access_count,
                last_access=last_access,
                created_time=created_time,
                ttl=ttl,
                metadata=metadata
            )
            
            self._cache[key] = entry
            self._access_order[key] = entry
            
            # 添加到LFU堆
            heapq.heappush(self._frequency_heap, (entry.access_count, entry.created_time, key))
        
        conn.close()
        self.logger.info(f"已加载 {len(self._cache)} 个缓存条目")
    
    def _save_cache_entry(self, entry: CacheEntry):
        """保存缓存条目到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO cache_entries 
            (key, value, size, access_count, last_access, created_time, ttl, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.key,
            json.dumps(entry.value),
            entry.size,
            entry.access_count,
            entry.last_access,
            entry.created_time,
            entry.ttl,
            json.dumps(entry.metadata) if entry.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def _delete_cache_entry(self, key: str):
        """从数据库删除缓存条目"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
        
        conn.commit()
        conn.close()
    
    def _log_clean_operation(self, strategy: str, count: int, size: int, duration: float, details: str):
        """记录清理操作到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clean_logs (timestamp, strategy, count, size, duration, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (time.time(), strategy, count, size, duration, details))
        
        conn.commit()
        conn.close()
    
    def add_cache_entry(self, key: str, value: Any, ttl: Optional[float] = None, 
                       metadata: Optional[Dict] = None) -> bool:
        """
        添加缓存条目
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒）
            metadata: 元数据
            
        Returns:
            bool: 是否成功添加
        """
        with self._lock:
            try:
                # 计算缓存大小
                value_str = json.dumps(value)
                size = len(value_str.encode('utf-8'))
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size=size,
                    access_count=0,
                    last_access=time.time(),
                    created_time=time.time(),
                    ttl=ttl,
                    metadata=metadata
                )
                
                self._cache[key] = entry
                self._access_order[key] = entry
                heapq.heappush(self._frequency_heap, (entry.access_count, entry.created_time, key))
                
                # 保存到数据库
                self._save_cache_entry(entry)
                
                self.logger.info(f"添加缓存条目: {key}, 大小: {size} 字节")
                return True
                
            except Exception as e:
                self.logger.error(f"添加缓存条目失败: {key}, 错误: {e}")
                return False
    
    def get_cache_entry(self, key: str) -> Optional[Any]:
        """
        获取缓存条目
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值或None
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                self._remove_cache_entry(key)
                return None
            
            # 更新访问统计
            entry.access_count += 1
            entry.last_access = time.time()
            
            # 更新访问顺序
            self._access_order.move_to_end(key)
            
            # 保存更新
            self._save_cache_entry(entry)
            
            return entry.value
    
    def _remove_cache_entry(self, key: str):
        """移除缓存条目"""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            del self._access_order[key]
        self._delete_cache_entry(key)
    
    def get_cache_size(self) -> int:
        """获取当前缓存总大小"""
        with self._lock:
            return sum(entry.size for entry in self._cache.values())
    
    def get_cache_count(self) -> int:
        """获取当前缓存条目数"""
        with self._lock:
            return len(self._cache)
    
    def manual_clean(self, strategy: CleanStrategy = CleanStrategy.LRU, 
                    target_size: Optional[int] = None,
                    target_count: Optional[int] = None) -> Dict:
        """
        手动清理缓存
        
        Args:
            strategy: 清理策略
            target_size: 目标大小（字节）
            target_count: 目标数量
            
        Returns:
            清理结果统计
        """
        start_time = time.time()
        
        with self._lock:
            if not self._cache:
                return {"count": 0, "size": 0, "duration": 0, "strategy": strategy.value}
            
            # 选择清理目标
            if target_size is None and target_count is None:
                # 默认清理30%的缓存
                target_count = max(1, len(self._cache) // 3)
            
            # 获取清理列表
            keys_to_remove = self._get_keys_to_clean(strategy, target_size, target_count)
            
            if not keys_to_remove:
                return {"count": 0, "size": 0, "duration": time.time() - start_time, "strategy": strategy.value}
            
            # 执行清理
            cleaned_size = 0
            for key in keys_to_remove:
                if key in self._cache:
                    cleaned_size += self._cache[key].size
                    self._remove_cache_entry(key)
            
            duration = time.time() - start_time
            
            # 记录统计
            self.stats.record_clean(len(keys_to_remove), cleaned_size, duration, strategy.value)
            
            # 记录到数据库
            self._log_clean_operation(
                strategy.value, 
                len(keys_to_remove), 
                cleaned_size, 
                duration,
                f"手动清理，移除 {len(keys_to_remove)} 个条目"
            )
            
            # 触发清理回调
            self._trigger_clean_callbacks(strategy.value, len(keys_to_remove), cleaned_size)
            
            self.logger.info(f"手动清理完成: {strategy.value}, 清理 {len(keys_to_remove)} 个条目, "
                           f"释放 {cleaned_size} 字节, 耗时 {duration:.3f} 秒")
            
            return {
                "count": len(keys_to_remove),
                "size": cleaned_size,
                "duration": duration,
                "strategy": strategy.value
            }
    
    def _get_keys_to_clean(self, strategy: CleanStrategy, target_size: Optional[int], 
                          target_count: Optional[int]) -> List[str]:
        """根据策略获取要清理的键列表"""
        if not self._cache:
            return []
        
        entries = list(self._cache.values())
        
        if strategy == CleanStrategy.LRU:
            # 最近最少使用
            sorted_entries = sorted(entries, key=lambda x: x.last_access)
        elif strategy == CleanStrategy.LFU:
            # 最少使用频率
            sorted_entries = sorted(entries, key=lambda x: (x.access_count, x.last_access))
        elif strategy == CleanStrategy.TTL:
            # 基于TTL过期时间
            sorted_entries = sorted(entries, key=lambda x: x.created_time + (x.ttl or 0))
        elif strategy == CleanStrategy.FIFO:
            # 先进先出
            sorted_entries = sorted(entries, key=lambda x: x.created_time)
        elif strategy == CleanStrategy.SIZE:
            # 基于缓存大小
            sorted_entries = sorted(entries, key=lambda x: x.size, reverse=True)
        elif strategy == CleanStrategy.HYBRID:
            # 混合策略：综合考虑TTL、访问频率和大小
            def hybrid_score(entry):
                ttl_factor = (time.time() - entry.created_time) / (entry.ttl or 86400)  # 默认1天
                freq_factor = 1.0 / (entry.access_count + 1)
                size_factor = entry.size / (1024 * 1024)  # MB
                return ttl_factor + freq_factor + size_factor
            
            sorted_entries = sorted(entries, key=hybrid_score, reverse=True)
        else:
            sorted_entries = entries
        
        # 确定清理数量
        if target_size is not None:
            # 基于大小清理
            keys_to_remove = []
            total_size = 0
            for entry in sorted_entries:
                if total_size >= target_size:
                    break
                keys_to_remove.append(entry.key)
                total_size += entry.size
        else:
            # 基于数量清理
            keys_to_remove = [entry.key for entry in sorted_entries[:target_count]]
        
        return keys_to_remove
    
    def start_auto_clean(self):
        """启动自动清理"""
        if self._is_running:
            return
        
        self._is_running = True
        self._auto_clean_thread = threading.Thread(target=self._auto_clean_loop, daemon=True)
        self._auto_clean_thread.start()
        self.logger.info("自动清理已启动")
    
    def stop_auto_clean(self):
        """停止自动清理"""
        self._is_running = False
        if self._auto_clean_thread:
            self._auto_clean_thread.join()
        self.logger.info("自动清理已停止")
    
    def _auto_clean_loop(self):
        """自动清理循环"""
        while self._is_running:
            try:
                time.sleep(self.auto_clean_interval)
                
                if not self._is_running:
                    break
                
                # 检查缓存大小
                current_size = self.get_cache_size()
                if current_size > self.max_cache_size * 0.8:  # 超过80%时清理
                    excess_size = current_size - self.max_cache_size * 0.8
                    self.manual_clean(CleanStrategy.HYBRID, target_size=excess_size)
                
                # 清理过期缓存
                expired_keys = []
                current_time = time.time()
                
                with self._lock:
                    for key, entry in self._cache.items():
                        if entry.is_expired():
                            expired_keys.append(key)
                
                # 移除过期缓存
                for key in expired_keys:
                    self._remove_cache_entry(key)
                
                if expired_keys:
                    self.logger.info(f"自动清理过期缓存: {len(expired_keys)} 个条目")
                
            except Exception as e:
                self.logger.error(f"自动清理出错: {e}")
    
    def add_clean_callback(self, callback: Callable):
        """添加清理回调函数"""
        self.clean_callbacks.append(callback)
    
    def _trigger_clean_callbacks(self, strategy: str, count: int, size: int):
        """触发清理回调"""
        for callback in self.clean_callbacks:
            try:
                callback(strategy, count, size)
            except Exception as e:
                self.logger.error(f"清理回调执行失败: {e}")
    
    def get_clean_stats(self) -> Dict:
        """获取清理统计信息"""
        return self.stats.get_stats()
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        try:
            if HAS_PSUTIL:
                process = psutil.Process()
                memory_info = process.memory_info()
                return {
                    "cache_size": self.get_cache_size(),
                    "cache_count": self.get_cache_count(),
                    "memory_usage": memory_info.rss,
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(),
                    "uptime": time.time() - self.stats.start_time
                }
            else:
                # 基础系统信息（不使用psutil）
                return {
                    "cache_size": self.get_cache_size(),
                    "cache_count": self.get_cache_count(),
                    "memory_usage": 0,  # 无法获取
                    "memory_percent": 0,  # 无法获取
                    "cpu_percent": 0,  # 无法获取
                    "uptime": time.time() - self.stats.start_time,
                    "note": "psutil未安装，系统信息受限"
                }
        except Exception as e:
            self.logger.error(f"获取系统信息失败: {e}")
            return {}
    
    def generate_clean_report(self) -> str:
        """生成清理报告"""
        stats = self.get_clean_stats()
        system_info = self.get_system_info()
        
        report = f"""
X6缓存清理器报告
================

清理统计:
- 总清理条目数: {stats['total_cleaned']}
- 总清理大小: {stats['total_size_cleaned'] / 1024 / 1024:.2f} MB
- 清理操作次数: {stats['clean_operations']}
- 平均清理时间: {stats['average_clean_time']:.3f} 秒
- 运行时间: {stats['uptime'] / 3600:.2f} 小时
- 最后清理时间: {datetime.fromtimestamp(stats['last_clean_time']).strftime('%Y-%m-%d %H:%M:%S') if stats['last_clean_time'] else 'N/A'}

策略统计:
"""
        
        for strategy, data in stats['strategy_stats'].items():
            report += f"- {strategy}: {data['count']} 个条目, {data['size'] / 1024 / 1024:.2f} MB\n"
        
        report += f"""
系统信息:
- 当前缓存大小: {system_info.get('cache_size', 0) / 1024 / 1024:.2f} MB
- 当前缓存条目数: {system_info.get('cache_count', 0)}
- 内存使用: {system_info.get('memory_usage', 0) / 1024 / 1024:.2f} MB
- 内存使用率: {system_info.get('memory_percent', 0):.2f}%
- CPU使用率: {system_info.get('cpu_percent', 0):.2f}%

配置信息:
- 缓存目录: {self.cache_dir}
- 最大缓存大小: {self.max_cache_size / 1024 / 1024:.2f} MB
- 自动清理间隔: {self.auto_clean_interval / 60:.1f} 分钟
"""
        
        return report
    
    def optimize_clean_performance(self) -> Dict:
        """清理性能优化"""
        optimization_results = {}
        
        # 1. 重建LFU堆
        self._frequency_heap = []
        for entry in self._cache.values():
            heapq.heappush(self._frequency_heap, (entry.access_count, entry.created_time, entry.key))
        optimization_results['lfu_heap_rebuilt'] = True
        
        # 2. 清理数据库碎片
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('VACUUM')
        conn.close()
        optimization_results['database_optimized'] = True
        
        # 3. 清理日志文件
        log_file = os.path.join(self.cache_dir, "cache_cleaner.log")
        if os.path.exists(log_file):
            # 保留最近1000行
            with open(log_file, 'r') as f:
                lines = f.readlines()
            if len(lines) > 1000:
                with open(log_file, 'w') as f:
                    f.writelines(lines[-1000:])
                optimization_results['log_truncated'] = True
        
        self.logger.info(f"性能优化完成: {optimization_results}")
        return optimization_results
    
    def export_cache_data(self, export_path: str) -> bool:
        """导出缓存数据"""
        try:
            export_data = {
                "timestamp": time.time(),
                "cache_entries": [],
                "stats": self.get_clean_stats(),
                "system_info": self.get_system_info()
            }
            
            with self._lock:
                for key, entry in self._cache.items():
                    export_data["cache_entries"].append({
                        "key": entry.key,
                        "value": entry.value,
                        "size": entry.size,
                        "access_count": entry.access_count,
                        "last_access": entry.last_access,
                        "created_time": entry.created_time,
                        "ttl": entry.ttl,
                        "metadata": entry.metadata
                    })
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"缓存数据已导出到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出缓存数据失败: {e}")
            return False
    
    def import_cache_data(self, import_path: str) -> bool:
        """导入缓存数据"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            with self._lock:
                for entry_data in import_data["cache_entries"]:
                    entry = CacheEntry(**entry_data)
                    self._cache[entry.key] = entry
                    self._save_cache_entry(entry)
            
            self.logger.info(f"已从 {import_path} 导入 {len(import_data['cache_entries'])} 个缓存条目")
            return True
            
        except Exception as e:
            self.logger.error(f"导入缓存数据失败: {e}")
            return False
    
    def shutdown(self):
        """关闭缓存清理器"""
        self.stop_auto_clean()
        
        # 保存所有缓存条目
        with self._lock:
            for entry in self._cache.values():
                self._save_cache_entry(entry)
        
        self.logger.info("缓存清理器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()