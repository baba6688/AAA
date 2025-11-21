#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y5存储优化器
提供全面的存储性能优化、空间优化、访问优化等功能
"""

import os
import sys
import time
import json
import gzip
import pickle
import hashlib
import shutil
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import statistics

# 尝试导入psutil，如果不可用则使用备用实现
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("psutil不可用，将使用备用实现")


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StorageMetrics:
    """存储性能指标"""
    read_speed: float  # 读取速度 (MB/s)
    write_speed: float  # 写入速度 (MB/s)
    space_usage: float  # 空间使用率 (%)
    compression_ratio: float  # 压缩比
    access_time: float  # 平均访问时间 (ms)
    fragmentation_ratio: float  # 碎片化率 (%)
    timestamp: str


@dataclass
class OptimizationReport:
    """优化报告"""
    optimization_type: str
    before_metrics: StorageMetrics
    after_metrics: StorageMetrics
    improvement_percentage: float
    recommendations: List[str]
    timestamp: str
    duration: float  # 优化耗时（秒）


class StorageOptimizer:
    """Y5存储优化器主类"""
    
    def __init__(self, storage_path: str = None, db_path: str = None):
        """
        初始化存储优化器
        
        Args:
            storage_path: 存储路径
            db_path: 数据库路径
        """
        self.storage_path = storage_path or os.path.expanduser("~/Y5_storage")
        self.db_path = db_path or os.path.join(self.storage_path, "optimizer.db")
        self.metrics_history = deque(maxlen=1000)
        self.optimization_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 创建存储目录
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        # 启动监控
        self.start_monitoring()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建指标表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        read_speed REAL,
                        write_speed REAL,
                        space_usage REAL,
                        compression_ratio REAL,
                        access_time REAL,
                        fragmentation_ratio REAL
                    )
                ''')
                
                # 创建优化记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS optimizations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        optimization_type TEXT NOT NULL,
                        before_metrics TEXT,
                        after_metrics TEXT,
                        improvement_percentage REAL,
                        recommendations TEXT,
                        duration REAL
                    )
                ''')
                
                conn.commit()
                logger.info("数据库初始化完成")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
    
    def analyze_storage(self, path: str = None) -> Dict[str, Any]:
        """
        分析存储使用情况
        
        Args:
            path: 要分析的路径
            
        Returns:
            分析结果字典
        """
        target_path = path or self.storage_path
        
        try:
            # 获取磁盘使用情况
            if PSUTIL_AVAILABLE:
                disk_usage = psutil.disk_usage(target_path)
            else:
                # 备用实现：使用shutil计算磁盘使用情况
                total, used, free = shutil.disk_usage(target_path)
                class MockDiskUsage:
                    def __init__(self, total, used, free):
                        self.total = total
                        self.used = used
                        self.free = free
                disk_usage = MockDiskUsage(total, used, free)
            
            # 分析文件类型分布
            file_types = defaultdict(int)
            total_files = 0
            total_size = 0
            
            for root, dirs, files in os.walk(target_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        file_ext = os.path.splitext(file)[1].lower()
                        file_types[file_ext] += 1
                        total_files += 1
                        total_size += file_size
                    except (OSError, IOError):
                        continue
            
            # 计算碎片化程度
            fragmentation_analysis = self._analyze_fragmentation(target_path)
            
            # 分析访问模式
            access_patterns = self._analyze_access_patterns(target_path)
            
            analysis_result = {
                "path": target_path,
                "total_size": total_size,
                "total_files": total_files,
                "disk_usage": {
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "usage_percent": (disk_usage.used / disk_usage.total) * 100
                },
                "file_types": dict(file_types),
                "fragmentation": fragmentation_analysis,
                "access_patterns": access_patterns,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"存储分析完成: {target_path}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"存储分析失败: {e}")
            return {"error": str(e)}
    
    def _analyze_fragmentation(self, path: str) -> Dict[str, Any]:
        """分析碎片化情况"""
        try:
            file_fragments = 0
            total_files = 0
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # 简单的碎片化估算
                        file_size = os.path.getsize(file_path)
                        if file_size > 1024 * 1024:  # 大于1MB的文件
                            # 估算碎片数量（简化算法）
                            estimated_fragments = max(1, file_size // (4 * 1024 * 1024))
                            file_fragments += estimated_fragments - 1
                        total_files += 1
                    except (OSError, IOError):
                        continue
            
            fragmentation_ratio = (file_fragments / max(1, total_files)) * 100 if total_files > 0 else 0
            
            return {
                "fragmentation_ratio": fragmentation_ratio,
                "fragmented_files": file_fragments,
                "total_files": total_files,
                "status": "high" if fragmentation_ratio > 20 else "medium" if fragmentation_ratio > 10 else "low"
            }
            
        except Exception as e:
            logger.error(f"碎片化分析失败: {e}")
            return {"fragmentation_ratio": 0, "error": str(e)}
    
    def _analyze_access_patterns(self, path: str) -> Dict[str, Any]:
        """分析访问模式"""
        try:
            access_counts = defaultdict(int)
            recent_accesses = 0
            old_files = 0
            
            current_time = time.time()
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        stat_info = os.stat(file_path)
                        
                        # 统计访问时间
                        access_time = stat_info.st_atime
                        access_counts[datetime.fromtimestamp(access_time).strftime("%Y-%m-%d")] += 1
                        
                        # 统计最近访问（7天内）
                        if current_time - access_time < 7 * 24 * 3600:
                            recent_accesses += 1
                        
                        # 统计老文件（1年以上未访问）
                        if current_time - access_time > 365 * 24 * 3600:
                            old_files += 1
                            
                    except (OSError, IOError):
                        continue
            
            return {
                "daily_access_counts": dict(access_counts),
                "recent_accesses": recent_accesses,
                "old_files": old_files,
                "hot_files_ratio": recent_accesses / max(1, sum(access_counts.values())),
                "cold_files_ratio": old_files / max(1, sum(access_counts.values()))
            }
            
        except Exception as e:
            logger.error(f"访问模式分析失败: {e}")
            return {"error": str(e)}
    
    def optimize_performance(self, path: str = None, 
                           optimization_level: str = "medium") -> OptimizationReport:
        """
        优化存储性能
        
        Args:
            path: 要优化的路径
            optimization_level: 优化级别 ("low", "medium", "high")
            
        Returns:
            优化报告
        """
        target_path = path or self.storage_path
        start_time = time.time()
        
        logger.info(f"开始性能优化: {target_path} (级别: {optimization_level})")
        
        # 获取优化前指标
        before_metrics = self._measure_performance(target_path)
        
        recommendations = []
        
        try:
            # 1. 清理临时文件
            temp_files_removed = self._clean_temp_files(target_path)
            if temp_files_removed > 0:
                recommendations.append(f"清理了 {temp_files_removed} 个临时文件")
            
            # 2. 优化文件组织
            files_reorganized = self._reorganize_files(target_path, optimization_level)
            if files_reorganized > 0:
                recommendations.append(f"重新组织了 {files_reorganized} 个文件")
            
            # 3. 优化缓存策略
            cache_optimized = self._optimize_cache(target_path, optimization_level)
            if cache_optimized:
                recommendations.append("优化了缓存策略")
            
            # 4. 预加载热文件
            hot_files_preloaded = self._preload_hot_files(target_path)
            if hot_files_preloaded > 0:
                recommendations.append(f"预加载了 {hot_files_preloaded} 个热文件")
            
            # 获取优化后指标
            after_metrics = self._measure_performance(target_path)
            
            # 计算改进百分比
            improvement = self._calculate_improvement(before_metrics, after_metrics)
            
            report = OptimizationReport(
                optimization_type="performance",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat(),
                duration=time.time() - start_time
            )
            
            # 保存优化记录
            self._save_optimization_report(report)
            
            logger.info(f"性能优化完成，改进: {improvement:.2f}%")
            return report
            
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
            raise
    
    def optimize_space(self, path: str = None, 
                      compression_level: int = 6) -> OptimizationReport:
        """
        优化存储空间
        
        Args:
            path: 要优化的路径
            compression_level: 压缩级别 (1-9)
            
        Returns:
            优化报告
        """
        target_path = path or self.storage_path
        start_time = time.time()
        
        logger.info(f"开始空间优化: {target_path} (压缩级别: {compression_level})")
        
        # 获取优化前指标
        before_metrics = self._measure_performance(target_path)
        
        recommendations = []
        space_saved = 0
        
        try:
            # 1. 压缩可压缩文件
            compressed_files, saved_space = self._compress_files(target_path, compression_level)
            space_saved += saved_space
            if compressed_files > 0:
                recommendations.append(f"压缩了 {compressed_files} 个文件，节省 {saved_space / 1024 / 1024:.2f} MB")
            
            # 2. 清理重复文件
            duplicates_removed, duplicate_space = self._remove_duplicates(target_path)
            space_saved += duplicate_space
            if duplicates_removed > 0:
                recommendations.append(f"删除了 {duplicates_removed} 个重复文件，节省 {duplicate_space / 1024 / 1024:.2f} MB")
            
            # 3. 清理空目录
            empty_dirs_removed = self._remove_empty_directories(target_path)
            if empty_dirs_removed > 0:
                recommendations.append(f"清理了 {empty_dirs_removed} 个空目录")
            
            # 4. 移动冷数据到归档
            archived_files, archived_space = self._archive_cold_data(target_path)
            space_saved += archived_space
            if archived_files > 0:
                recommendations.append(f"归档了 {archived_files} 个冷文件，节省 {archived_space / 1024 / 1024:.2f} MB")
            
            # 获取优化后指标
            after_metrics = self._measure_performance(target_path)
            
            # 计算改进百分比
            improvement = self._calculate_improvement(before_metrics, after_metrics)
            
            report = OptimizationReport(
                optimization_type="space",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat(),
                duration=time.time() - start_time
            )
            
            # 保存优化记录
            self._save_optimization_report(report)
            
            logger.info(f"空间优化完成，总节省空间: {space_saved / 1024 / 1024:.2f} MB")
            return report
            
        except Exception as e:
            logger.error(f"空间优化失败: {e}")
            raise
    
    def optimize_access(self, path: str = None) -> OptimizationReport:
        """
        优化存储访问
        
        Args:
            path: 要优化的路径
            
        Returns:
            优化报告
        """
        target_path = path or self.storage_path
        start_time = time.time()
        
        logger.info(f"开始访问优化: {target_path}")
        
        # 获取优化前指标
        before_metrics = self._measure_performance(target_path)
        
        recommendations = []
        
        try:
            # 1. 创建索引文件
            indexes_created = self._create_indexes(target_path)
            if indexes_created > 0:
                recommendations.append(f"创建了 {indexes_created} 个索引文件")
            
            # 2. 优化目录结构
            structure_optimized = self._optimize_directory_structure(target_path)
            if structure_optimized:
                recommendations.append("优化了目录结构")
            
            # 3. 创建访问缓存
            cache_created = self._create_access_cache(target_path)
            if cache_created:
                recommendations.append("创建了访问缓存")
            
            # 4. 预组织文件
            files_organized = self._preorganize_files(target_path)
            if files_organized > 0:
                recommendations.append(f"预组织了 {files_organized} 个文件")
            
            # 获取优化后指标
            after_metrics = self._measure_performance(target_path)
            
            # 计算改进百分比
            improvement = self._calculate_improvement(before_metrics, after_metrics)
            
            report = OptimizationReport(
                optimization_type="access",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat(),
                duration=time.time() - start_time
            )
            
            # 保存优化记录
            self._save_optimization_report(report)
            
            logger.info(f"访问优化完成，改进: {improvement:.2f}%")
            return report
            
        except Exception as e:
            logger.error(f"访问优化失败: {e}")
            raise
    
    def compress_data(self, data: Any, compression_type: str = "gzip") -> bytes:
        """
        压缩数据
        
        Args:
            data: 要压缩的数据
            compression_type: 压缩类型 ("gzip", "bz2", "lzma")
            
        Returns:
            压缩后的字节数据
        """
        try:
            serialized_data = pickle.dumps(data)
            
            if compression_type == "gzip":
                return gzip.compress(serialized_data)
            elif compression_type == "bz2":
                import bz2
                return bz2.compress(serialized_data)
            elif compression_type == "lzma":
                import lzma
                return lzma.compress(serialized_data)
            else:
                raise ValueError(f"不支持的压缩类型: {compression_type}")
                
        except Exception as e:
            logger.error(f"数据压缩失败: {e}")
            raise
    
    def decompress_data(self, compressed_data: bytes, compression_type: str = "gzip") -> Any:
        """
        解压缩数据
        
        Args:
            compressed_data: 压缩的字节数据
            compression_type: 压缩类型
            
        Returns:
            解压缩后的原始数据
        """
        try:
            if compression_type == "gzip":
                decompressed_data = gzip.decompress(compressed_data)
            elif compression_type == "bz2":
                import bz2
                decompressed_data = bz2.decompress(compressed_data)
            elif compression_type == "lzma":
                import lzma
                decompressed_data = lzma.decompress(compressed_data)
            else:
                raise ValueError(f"不支持的压缩类型: {compression_type}")
            
            return pickle.loads(decompressed_data)
            
        except Exception as e:
            logger.error(f"数据解压缩失败: {e}")
            raise
    
    def generate_optimization_suggestions(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        生成优化建议
        
        Args:
            analysis_result: 存储分析结果
            
        Returns:
            优化建议列表
        """
        suggestions = []
        
        try:
            # 基于空间使用率的建议
            usage_percent = analysis_result.get("disk_usage", {}).get("usage_percent", 0)
            if usage_percent > 90:
                suggestions.append("存储空间使用率过高，建议立即进行空间优化")
            elif usage_percent > 80:
                suggestions.append("存储空间使用率较高，建议进行空间优化")
            elif usage_percent < 50:
                suggestions.append("存储空间使用率较低，可以考虑减少存储配额")
            
            # 基于碎片化的建议
            fragmentation = analysis_result.get("fragmentation", {})
            frag_ratio = fragmentation.get("fragmentation_ratio", 0)
            if frag_ratio > 20:
                suggestions.append("文件碎片化严重，建议进行磁盘整理")
            elif frag_ratio > 10:
                suggestions.append("文件存在一定碎片化，建议优化文件组织")
            
            # 基于访问模式的建议
            access_patterns = analysis_result.get("access_patterns", {})
            hot_ratio = access_patterns.get("hot_files_ratio", 0)
            cold_ratio = access_patterns.get("cold_files_ratio", 0)
            
            if cold_ratio > 0.3:
                suggestions.append("存在较多冷数据，建议进行数据归档")
            if hot_ratio < 0.1:
                suggestions.append("热文件比例较低，建议优化缓存策略")
            
            # 基于文件类型的建议
            file_types = analysis_result.get("file_types", {})
            large_files = sum(1 for ext, count in file_types.items() 
                            if ext in ['.avi', '.mkv', '.mp4', '.zip', '.rar'])
            if large_files > 10:
                suggestions.append("发现较多大文件，建议进行压缩优化")
            
            # 基于文件数量的建议
            total_files = analysis_result.get("total_files", 0)
            if total_files > 10000:
                suggestions.append("文件数量较多，建议优化文件组织结构")
            
            if not suggestions:
                suggestions.append("当前存储状态良好，建议定期进行维护优化")
                
        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")
            suggestions.append("无法生成具体建议，请检查存储状态")
        
        return suggestions
    
    def start_monitoring(self, interval: int = 300):
        """
        启动优化监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.is_monitoring:
            logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"启动存储监控，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("停止存储监控")
    
    def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集性能指标
                metrics = self._measure_performance(self.storage_path)
                self.metrics_history.append(metrics)
                
                # 保存到数据库
                self._save_metrics_to_db(metrics)
                
                # 检查是否需要自动优化
                self._check_auto_optimization(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(60)  # 错误时等待1分钟再继续
    
    def _check_auto_optimization(self, metrics: StorageMetrics):
        """检查是否需要自动优化"""
        try:
            # 空间使用率过高时自动优化
            if metrics.space_usage > 85:
                logger.info("检测到空间使用率过高，触发自动空间优化")
                self.optimize_space(self.storage_path)
            
            # 碎片化严重时自动优化
            if metrics.fragmentation_ratio > 25:
                logger.info("检测到碎片化严重，触发自动性能优化")
                self.optimize_performance(self.storage_path)
            
        except Exception as e:
            logger.error(f"自动优化检查失败: {e}")
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取优化历史记录
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            优化历史记录列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM optimizations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                columns = [description[0] for description in cursor.description]
                history = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    # 解析JSON字段
                    if record.get('before_metrics'):
                        record['before_metrics'] = json.loads(record['before_metrics'])
                    if record.get('after_metrics'):
                        record['after_metrics'] = json.loads(record['after_metrics'])
                    if record.get('recommendations'):
                        record['recommendations'] = json.loads(record['recommendations'])
                    
                    history.append(record)
                
                return history
                
        except Exception as e:
            logger.error(f"获取优化历史失败: {e}")
            return []
    
    def export_report(self, output_path: str, format: str = "json") -> bool:
        """
        导出优化报告
        
        Args:
            output_path: 输出文件路径
            format: 导出格式 ("json", "html", "csv")
            
        Returns:
            是否导出成功
        """
        try:
            # 获取历史数据
            history = self.get_optimization_history(50)
            analysis = self.analyze_storage(self.storage_path)
            
            report_data = {
                "export_time": datetime.now().isoformat(),
                "storage_path": self.storage_path,
                "analysis": analysis,
                "optimization_history": history,
                "current_metrics": asdict(self._measure_performance(self.storage_path))
            }
            
            if format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            elif format == "html":
                self._generate_html_report(report_data, output_path)
            
            elif format == "csv":
                self._generate_csv_report(history, output_path)
            
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"报告导出成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"报告导出失败: {e}")
            return False
    
    # 辅助方法实现
    def _measure_performance(self, path: str) -> StorageMetrics:
        """测量存储性能指标"""
        try:
            # 测量读写速度
            read_speed, write_speed = self._measure_io_speed(path)
            
            # 获取空间使用率
            if PSUTIL_AVAILABLE:
                disk_usage = psutil.disk_usage(path)
                space_usage = (disk_usage.used / disk_usage.total) * 100
            else:
                # 备用实现：使用shutil计算磁盘使用情况
                total, used, free = shutil.disk_usage(path)
                space_usage = (used / total) * 100 if total > 0 else 0
            
            # 测量访问时间
            access_time = self._measure_access_time(path)
            
            # 获取碎片化率
            fragmentation = self._analyze_fragmentation(path)
            fragmentation_ratio = fragmentation.get("fragmentation_ratio", 0)
            
            # 估算压缩比
            compression_ratio = self._estimate_compression_ratio(path)
            
            return StorageMetrics(
                read_speed=read_speed,
                write_speed=write_speed,
                space_usage=space_usage,
                compression_ratio=compression_ratio,
                access_time=access_time,
                fragmentation_ratio=fragmentation_ratio,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"性能测量失败: {e}")
            return StorageMetrics(0, 0, 0, 0, 0, 0, datetime.now().isoformat())
    
    def _measure_io_speed(self, path: str) -> Tuple[float, float]:
        """测量IO速度"""
        test_file = os.path.join(path, "speed_test.tmp")
        
        try:
            # 确保目录存在
            os.makedirs(path, exist_ok=True)
            
            # 测试写入速度
            start_time = time.time()
            with open(test_file, 'wb') as f:
                # 写入1MB测试数据（减少测试时间）
                test_data = b'0' * (1024 * 1024)
                f.write(test_data)
            write_time = time.time() - start_time
            write_speed = (1 / write_time) if write_time > 0 else 0
            
            # 测试读取速度
            start_time = time.time()
            with open(test_file, 'rb') as f:
                f.read()
            read_time = time.time() - start_time
            read_speed = (1 / read_time) if read_time > 0 else 0
            
            # 清理测试文件
            try:
                os.remove(test_file)
            except:
                pass
            
            return read_speed, write_speed
            
        except Exception as e:
            logger.error(f"IO速度测量失败: {e}")
            return 0, 0
    
    def _measure_access_time(self, path: str) -> float:
        """测量平均访问时间"""
        try:
            access_times = []
            
            # 选择一些文件进行测试
            test_files = []
            for root, dirs, files in os.walk(path):
                test_files.extend([os.path.join(root, f) for f in files[:5]])  # 每个目录最多5个文件
                if len(test_files) >= 20:  # 最多测试20个文件
                    break
            
            for file_path in test_files[:10]:  # 实际测试10个文件
                try:
                    start_time = time.time()
                    with open(file_path, 'rb') as f:
                        f.read(1024)  # 读取1KB
                    access_time = (time.time() - start_time) * 1000  # 转换为毫秒
                    access_times.append(access_time)
                except (OSError, IOError):
                    continue
            
            return statistics.mean(access_times) if access_times else 0
            
        except Exception as e:
            logger.error(f"访问时间测量失败: {e}")
            return 0
    
    def _estimate_compression_ratio(self, path: str) -> float:
        """估算压缩比"""
        try:
            total_size = 0
            compressed_size = 0
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        
                        # 尝试压缩文件内容的一部分来估算压缩比
                        with open(file_path, 'rb') as f:
                            sample_data = f.read(min(file_size, 1024 * 100))  # 最多读取100KB样本
                            if sample_data:
                                compressed_sample = gzip.compress(sample_data)
                                estimated_compressed = (len(compressed_sample) / len(sample_data)) * file_size
                                compressed_size += estimated_compressed
                                
                    except (OSError, IOError):
                        continue
            
            return (compressed_size / total_size) if total_size > 0 else 1.0
            
        except Exception as e:
            logger.error(f"压缩比估算失败: {e}")
            return 1.0
    
    def _calculate_improvement(self, before: StorageMetrics, after: StorageMetrics) -> float:
        """计算改进百分比"""
        try:
            # 综合多个指标的改进情况
            improvements = []
            
            # 读写速度改进
            if before.read_speed > 0:
                read_improvement = ((after.read_speed - before.read_speed) / before.read_speed) * 100
                improvements.append(read_improvement)
            
            if before.write_speed > 0:
                write_improvement = ((after.write_speed - before.write_speed) / before.write_speed) * 100
                improvements.append(write_improvement)
            
            # 访问时间改进（越小越好）
            if before.access_time > 0:
                access_improvement = ((before.access_time - after.access_time) / before.access_time) * 100
                improvements.append(access_improvement)
            
            # 碎片化改进（越小越好）
            if before.fragmentation_ratio > 0:
                frag_improvement = ((before.fragmentation_ratio - after.fragmentation_ratio) / before.fragmentation_ratio) * 100
                improvements.append(frag_improvement)
            
            return statistics.mean(improvements) if improvements else 0
            
        except Exception as e:
            logger.error(f"改进计算失败: {e}")
            return 0
    
    def _clean_temp_files(self, path: str) -> int:
        """清理临时文件"""
        temp_extensions = ['.tmp', '.temp', '.bak', '.log', '.cache']
        removed_count = 0
        
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in temp_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            removed_count += 1
                        except (OSError, IOError):
                            continue
            
            logger.info(f"清理临时文件: {removed_count} 个")
            return removed_count
            
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")
            return 0
    
    def _reorganize_files(self, path: str, level: str) -> int:
        """重新组织文件"""
        reorganized_count = 0
        
        try:
            # 根据文件类型重新组织
            file_categories = {
                'documents': ['.doc', '.docx', '.pdf', '.txt', '.rtf'],
                'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
                'videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv'],
                'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
                'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
                'code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
            }
            
            for category, extensions in file_categories.items():
                category_path = os.path.join(path, category)
                os.makedirs(category_path, exist_ok=True)
                
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in extensions):
                            src_path = os.path.join(root, file)
                            dst_path = os.path.join(category_path, file)
                            
                            # 避免在同一目录下移动
                            if os.path.dirname(src_path) != category_path:
                                try:
                                    shutil.move(src_path, dst_path)
                                    reorganized_count += 1
                                except (OSError, IOError):
                                    continue
            
            logger.info(f"重新组织文件: {reorganized_count} 个")
            return reorganized_count
            
        except Exception as e:
            logger.error(f"重新组织文件失败: {e}")
            return 0
    
    def _optimize_cache(self, path: str, level: str) -> bool:
        """优化缓存策略"""
        try:
            cache_dir = os.path.join(path, ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 创建缓存配置文件
            cache_config = {
                "cache_size": "100MB",
                "cache_policy": "lru",
                "compression": level != "low",
                "preload_hot_files": level == "high"
            }
            
            config_path = os.path.join(cache_dir, "cache_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(cache_config, f, ensure_ascii=False, indent=2)
            
            logger.info("缓存策略优化完成")
            return True
            
        except Exception as e:
            logger.error(f"缓存优化失败: {e}")
            return False
    
    def _preload_hot_files(self, path: str) -> int:
        """预加载热文件"""
        try:
            hot_files = []
            current_time = time.time()
            
            # 找出最近访问的文件
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        stat_info = os.stat(file_path)
                        # 最近7天访问的文件
                        if current_time - stat_info.st_atime < 7 * 24 * 3600:
                            hot_files.append(file_path)
                    except (OSError, IOError):
                        continue
            
            # 创建热文件索引
            hot_files_index = os.path.join(path, ".cache", "hot_files.json")
            with open(hot_files_index, 'w', encoding='utf-8') as f:
                json.dump(hot_files[:100], f, ensure_ascii=False, indent=2)  # 最多保存100个热文件
            
            logger.info(f"预加载热文件: {len(hot_files)} 个")
            return len(hot_files)
            
        except Exception as e:
            logger.error(f"预加载热文件失败: {e}")
            return 0
    
    def _compress_files(self, path: str, compression_level: int) -> Tuple[int, int]:
        """压缩文件"""
        compressed_count = 0
        saved_space = 0
        
        # 可压缩的文件类型
        compressible_types = ['.txt', '.log', '.json', '.xml', '.csv', '.py', '.js', '.html', '.css']
        
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # 跳过已压缩的文件和缓存文件
                    if (file.endswith('.gz') or 
                        file.endswith('.bz2') or 
                        file.endswith('.xz') or
                        '.cache' in file_path):
                        continue
                    
                    # 只处理可压缩的文件类型
                    if any(file.lower().endswith(ext) for ext in compressible_types):
                        try:
                            original_size = os.path.getsize(file_path)
                            
                            # 读取文件内容并压缩
                            with open(file_path, 'rb') as f:
                                data = f.read()
                            
                            compressed_data = gzip.compress(data, compresslevel=compression_level)
                            compressed_size = len(compressed_data)
                            
                            # 如果压缩有效，保存压缩后的文件
                            if compressed_size < original_size * 0.9:  # 压缩率至少10%
                                compressed_file_path = file_path + '.gz'
                                with open(compressed_file_path, 'wb') as f:
                                    f.write(compressed_data)
                                
                                os.remove(file_path)  # 删除原文件
                                
                                compressed_count += 1
                                saved_space += (original_size - compressed_size)
                                
                        except (OSError, IOError, Exception):
                            continue
            
            logger.info(f"压缩文件: {compressed_count} 个，节省空间: {saved_space / 1024 / 1024:.2f} MB")
            return compressed_count, saved_space
            
        except Exception as e:
            logger.error(f"文件压缩失败: {e}")
            return 0, 0
    
    def _remove_duplicates(self, path: str) -> Tuple[int, int]:
        """删除重复文件"""
        file_hashes = defaultdict(list)
        removed_count = 0
        saved_space = 0
        
        try:
            # 计算文件哈希值
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # 跳过缓存和系统文件
                    if '.cache' in file_path or file.startswith('.'):
                        continue
                    
                    try:
                        # 计算文件哈希
                        hasher = hashlib.md5()
                        with open(file_path, 'rb') as f:
                            # 只读取前1MB来计算哈希（提高速度）
                            chunk = f.read(1024 * 1024)
                            while chunk:
                                hasher.update(chunk)
                                chunk = f.read(1024 * 1024)
                        
                        file_hash = hasher.hexdigest()
                        file_hashes[file_hash].append(file_path)
                        
                    except (OSError, IOError):
                        continue
            
            # 删除重复文件（保留第一个）
            for file_hash, file_list in file_hashes.items():
                if len(file_list) > 1:
                    # 按修改时间排序，保留最新的
                    file_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    
                    for duplicate_file in file_list[1:]:
                        try:
                            file_size = os.path.getsize(duplicate_file)
                            os.remove(duplicate_file)
                            removed_count += 1
                            saved_space += file_size
                        except (OSError, IOError):
                            continue
            
            logger.info(f"删除重复文件: {removed_count} 个，节省空间: {saved_space / 1024 / 1024:.2f} MB")
            return removed_count, saved_space
            
        except Exception as e:
            logger.error(f"删除重复文件失败: {e}")
            return 0, 0
    
    def _remove_empty_directories(self, path: str) -> int:
        """删除空目录"""
        removed_count = 0
        
        try:
            # 从最深层开始删除
            for root, dirs, files in os.walk(path, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    
                    # 检查目录是否为空
                    try:
                        if not os.listdir(dir_path):  # 目录为空
                            os.rmdir(dir_path)
                            removed_count += 1
                    except (OSError, IOError):
                        continue
            
            logger.info(f"删除空目录: {removed_count} 个")
            return removed_count
            
        except Exception as e:
            logger.error(f"删除空目录失败: {e}")
            return 0
    
    def _archive_cold_data(self, path: str) -> Tuple[int, int]:
        """归档冷数据"""
        archived_count = 0
        archived_space = 0
        
        try:
            archive_dir = os.path.join(path, ".archive")
            os.makedirs(archive_dir, exist_ok=True)
            
            current_time = time.time()
            
            # 找出冷数据（1年以上未访问的文件）
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # 跳过已归档和系统文件
                    if '.archive' in file_path or '.cache' in file_path or file.startswith('.'):
                        continue
                    
                    try:
                        stat_info = os.stat(file_path)
                        # 1年以上未访问的文件
                        if current_time - stat_info.st_atime > 365 * 24 * 3600:
                            file_size = os.path.getsize(file_path)
                            
                            # 移动到归档目录
                            archive_file_path = os.path.join(archive_dir, file)
                            shutil.move(file_path, archive_file_path)
                            
                            archived_count += 1
                            archived_space += file_size
                            
                    except (OSError, IOError):
                        continue
            
            logger.info(f"归档冷数据: {archived_count} 个文件，空间: {archived_space / 1024 / 1024:.2f} MB")
            return archived_count, archived_space
            
        except Exception as e:
            logger.error(f"归档冷数据失败: {e}")
            return 0, 0
    
    def _create_indexes(self, path: str) -> int:
        """创建索引文件"""
        indexes_created = 0
        
        try:
            index_dir = os.path.join(path, ".indexes")
            os.makedirs(index_dir, exist_ok=True)
            
            # 创建文件类型索引
            file_type_index = defaultdict(list)
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    file_type_index[file_ext].append(file_path)
            
            # 保存索引文件
            for file_type, file_list in file_type_index.items():
                if file_type:  # 只保存有扩展名的文件类型
                    index_file = os.path.join(index_dir, f"{file_type[1:]}_index.json")
                    with open(index_file, 'w', encoding='utf-8') as f:
                        json.dump(file_list, f, ensure_ascii=False, indent=2)
                    indexes_created += 1
            
            # 创建访问频率索引
            access_index = []
            current_time = time.time()
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        stat_info = os.stat(file_path)
                        access_index.append({
                            "path": file_path,
                            "last_access": stat_info.st_atime,
                            "access_frequency": current_time - stat_info.st_atime
                        })
                    except (OSError, IOError):
                        continue
            
            # 按访问频率排序
            access_index.sort(key=lambda x: x["access_frequency"])
            
            access_index_file = os.path.join(index_dir, "access_frequency.json")
            with open(access_index_file, 'w', encoding='utf-8') as f:
                json.dump(access_index, f, ensure_ascii=False, indent=2)
            indexes_created += 1
            
            logger.info(f"创建索引文件: {indexes_created} 个")
            return indexes_created
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return 0
    
    def _optimize_directory_structure(self, path: str) -> bool:
        """优化目录结构"""
        try:
            # 创建标准的目录结构
            standard_dirs = [
                "documents", "images", "videos", "audio", 
                "archives", "code", "data", "temp"
            ]
            
            for dir_name in standard_dirs:
                dir_path = os.path.join(path, dir_name)
                os.makedirs(dir_path, exist_ok=True)
            
            logger.info("目录结构优化完成")
            return True
            
        except Exception as e:
            logger.error(f"目录结构优化失败: {e}")
            return False
    
    def _create_access_cache(self, path: str) -> bool:
        """创建访问缓存"""
        try:
            cache_dir = os.path.join(path, ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 创建访问缓存
            access_cache = {}
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        stat_info = os.stat(file_path)
                        access_cache[file_path] = {
                            "size": stat_info.st_size,
                            "mtime": stat_info.st_mtime,
                            "atime": stat_info.st_atime
                        }
                    except (OSError, IOError):
                        continue
            
            cache_file = os.path.join(cache_dir, "access_cache.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(access_cache, f, ensure_ascii=False, indent=2)
            
            logger.info("访问缓存创建完成")
            return True
            
        except Exception as e:
            logger.error(f"创建访问缓存失败: {e}")
            return False
    
    def _preorganize_files(self, path: str) -> int:
        """预组织文件"""
        organized_count = 0
        
        try:
            # 按文件大小和访问频率组织文件
            file_info = []
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        stat_info = os.stat(file_path)
                        file_info.append({
                            "path": file_path,
                            "size": stat_info.st_size,
                            "atime": stat_info.st_atime,
                            "mtime": stat_info.st_mtime
                        })
                    except (OSError, IOError):
                        continue
            
            # 按访问时间排序
            file_info.sort(key=lambda x: x["atime"], reverse=True)
            
            # 创建预组织映射
            preorg_file = os.path.join(path, ".cache", "preorganized.json")
            with open(preorg_file, 'w', encoding='utf-8') as f:
                json.dump(file_info, f, ensure_ascii=False, indent=2)
            
            organized_count = len(file_info)
            logger.info(f"预组织文件: {organized_count} 个")
            return organized_count
            
        except Exception as e:
            logger.error(f"预组织文件失败: {e}")
            return 0
    
    def _save_optimization_report(self, report: OptimizationReport):
        """保存优化报告到数据库"""
        try:
            # 确保数据库目录存在
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO optimizations (
                        timestamp, optimization_type, before_metrics, after_metrics,
                        improvement_percentage, recommendations, duration
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    report.timestamp,
                    report.optimization_type,
                    json.dumps(asdict(report.before_metrics)),
                    json.dumps(asdict(report.after_metrics)),
                    report.improvement_percentage,
                    json.dumps(report.recommendations),
                    report.duration
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存优化报告失败: {e}")
    
    def _save_metrics_to_db(self, metrics: StorageMetrics):
        """保存指标到数据库"""
        try:
            # 确保数据库目录存在
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO metrics (
                        timestamp, read_speed, write_speed, space_usage,
                        compression_ratio, access_time, fragmentation_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp,
                    metrics.read_speed,
                    metrics.write_speed,
                    metrics.space_usage,
                    metrics.compression_ratio,
                    metrics.access_time,
                    metrics.fragmentation_ratio
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存指标失败: {e}")
    
    def _generate_html_report(self, data: Dict[str, Any], output_path: str):
        """生成HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Y5存储优化器报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
        .optimization {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #007cba; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Y5存储优化器报告</h1>
        <p>生成时间: {data['export_time']}</p>
        <p>存储路径: {data['storage_path']}</p>
    </div>
    
    <div class="section">
        <h2>存储分析</h2>
        <div class="metric">总文件数: {data['analysis'].get('total_files', 0)}</div>
        <div class="metric">总大小: {data['analysis'].get('total_size', 0) / 1024 / 1024:.2f} MB</div>
        <div class="metric">空间使用率: {data['analysis'].get('disk_usage', {}).get('usage_percent', 0):.2f}%</div>
        <div class="metric">碎片化率: {data['analysis'].get('fragmentation', {}).get('fragmentation_ratio', 0):.2f}%</div>
    </div>
    
    <div class="section">
        <h2>优化历史</h2>
        {self._generate_optimization_history_html(data['optimization_history'])}
    </div>
    
    <div class="section">
        <h2>当前指标</h2>
        <div class="metric">读取速度: {data['current_metrics'].read_speed:.2f} MB/s</div>
        <div class="metric">写入速度: {data['current_metrics'].write_speed:.2f} MB/s</div>
        <div class="metric">访问时间: {data['current_metrics'].access_time:.2f} ms</div>
    </div>
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_optimization_history_html(self, history: List[Dict[str, Any]]) -> str:
        """生成优化历史HTML"""
        html = ""
        for record in history:
            html += f"""
            <div class="optimization">
                <h3>{record['optimization_type']} 优化</h3>
                <p>时间: {record['timestamp']}</p>
                <p>改进: {record['improvement_percentage']:.2f}%</p>
                <p>耗时: {record['duration']:.2f} 秒</p>
                <ul>
            """
            for recommendation in record.get('recommendations', []):
                html += f"<li>{recommendation}</li>"
            html += """
                </ul>
            </div>
            """
        return html
    
    def _generate_csv_report(self, history: List[Dict[str, Any]], output_path: str):
        """生成CSV报告"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'optimization_type', 'improvement_percentage', 'duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in history:
                writer.writerow({
                    'timestamp': record['timestamp'],
                    'optimization_type': record['optimization_type'],
                    'improvement_percentage': record['improvement_percentage'],
                    'duration': record['duration']
                })
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_monitoring') and self.is_monitoring:
            self.stop_monitoring()


# 示例使用函数
def demo_storage_optimizer():
    """存储优化器演示"""
    print("=== Y5存储优化器演示 ===")
    
    # 创建优化器实例
    optimizer = StorageOptimizer()
    
    # 分析存储
    print("\n1. 分析存储...")
    analysis = optimizer.analyze_storage()
    print(f"存储分析结果: {json.dumps(analysis, ensure_ascii=False, indent=2)}")
    
    # 生成优化建议
    print("\n2. 生成优化建议...")
    suggestions = optimizer.generate_optimization_suggestions(analysis)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"建议 {i}: {suggestion}")
    
    # 执行性能优化
    print("\n3. 执行性能优化...")
    performance_report = optimizer.optimize_performance(optimization_level="medium")
    print(f"性能优化结果: 改进 {performance_report.improvement_percentage:.2f}%")
    
    # 执行空间优化
    print("\n4. 执行空间优化...")
    space_report = optimizer.optimize_space(compression_level=6)
    print(f"空间优化结果: 改进 {space_report.improvement_percentage:.2f}%")
    
    # 执行访问优化
    print("\n5. 执行访问优化...")
    access_report = optimizer.optimize_access()
    print(f"访问优化结果: 改进 {access_report.improvement_percentage:.2f}%")
    
    # 数据压缩演示
    print("\n6. 数据压缩演示...")
    test_data = {"message": "Hello, Y5 Storage Optimizer!", "timestamp": time.time()}
    compressed = optimizer.compress_data(test_data, "gzip")
    decompressed = optimizer.decompress_data(compressed, "gzip")
    print(f"原始数据大小: {len(str(test_data))} 字符")
    print(f"压缩后大小: {len(compressed)} 字节")
    print(f"解压缩结果: {decompressed}")
    
    # 导出报告
    print("\n7. 导出报告...")
    optimizer.export_report("storage_optimization_report.json", "json")
    optimizer.export_report("storage_optimization_report.html", "html")
    print("报告导出完成")
    
    # 获取优化历史
    print("\n8. 获取优化历史...")
    history = optimizer.get_optimization_history(5)
    print(f"最近 {len(history)} 次优化记录")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_storage_optimizer()