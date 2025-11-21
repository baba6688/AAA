#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W7带宽管理器
提供全面的网络带宽控制、监控和管理功能
"""

import time
import threading
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import socket


@dataclass
class BandwidthConfig:
    """带宽配置类"""
    max_upload_speed: float = 10.0  # 最大上传速度 (MB/s)
    max_download_speed: float = 50.0  # 最大下载速度 (MB/s)
    daily_data_limit: float = 1024.0  # 日流量限制 (MB)
    monthly_data_limit: float = 30720.0  # 月流量限制 (30GB)
    qos_enabled: bool = True  # QoS是否启用
    optimization_enabled: bool = True  # 网络优化是否启用
    monitoring_interval: float = 1.0  # 监控间隔 (秒)


@dataclass
class NetworkStats:
    """网络统计类"""
    timestamp: float
    upload_speed: float  # 当前上传速度 (MB/s)
    download_speed: float  # 当前下载速度 (MB/s)
    total_upload: float  # 总上传量 (MB)
    total_download: float  # 总下载量 (MB)
    cpu_usage: float  # CPU使用率 (%)
    memory_usage: float  # 内存使用率 (%)


@dataclass
class QoSPolicy:
    """QoS策略类"""
    name: str
    priority: int  # 优先级 (1-10, 10为最高)
    max_bandwidth: float  # 最大带宽 (MB/s)
    protocol: str  # 协议 (TCP, UDP, ANY)
    port_range: Optional[Tuple[int, int]]  # 端口范围
    enabled: bool = True


class BandwidthManager:
    """W7带宽管理器主类"""
    
    def __init__(self, config: Optional[BandwidthConfig] = None):
        """
        初始化带宽管理器
        
        Args:
            config: 带宽配置对象
        """
        self.config = config or BandwidthConfig()
        self.logger = self._setup_logger()
        
        # 统计数据
        self.stats_history: deque = deque(maxlen=3600)  # 保存1小时的统计数据
        self.daily_usage = {'upload': 0.0, 'download': 0.0}
        self.monthly_usage = {'upload': 0.0, 'download': 0.0}
        
        # 监控状态
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_stats: Optional[NetworkStats] = None
        
        # QoS策略
        self.qos_policies: List[QoSPolicy] = []
        
        # 流量限制状态
        self.traffic_limits = {
            'daily_limit_reached': False,
            'monthly_limit_reached': False,
            'current_upload_limit': self.config.max_upload_speed,
            'current_download_limit': self.config.max_download_speed
        }
        
        self.logger.info("W7带宽管理器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('W7BandwidthManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self) -> bool:
        """
        开始带宽监控
        
        Returns:
            bool: 是否成功启动监控
        """
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return False
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("带宽监控已启动")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        停止带宽监控
        
        Returns:
            bool: 是否成功停止监控
        """
        if not self.is_monitoring:
            self.logger.warning("监控未在运行")
            return False
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("带宽监控已停止")
        return True
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                stats = self._collect_network_stats()
                if stats:
                    self._process_stats(stats)
                    self.last_stats = stats
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(5)
    
    def _collect_network_stats(self) -> Optional[NetworkStats]:
        """收集网络统计数据"""
        try:
            # 获取网络IO统计
            net_io = psutil.net_io_counters()
            
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # 计算网络速度
            current_time = time.time()
            
            if self.last_stats:
                time_diff = current_time - self.last_stats.timestamp
                if time_diff > 0:
                    upload_speed = (net_io.bytes_sent - self.last_stats.total_upload * 1024 * 1024) / (time_diff * 1024 * 1024)
                    download_speed = (net_io.bytes_recv - self.last_stats.total_download * 1024 * 1024) / (time_diff * 1024 * 1024)
                else:
                    upload_speed = download_speed = 0.0
            else:
                upload_speed = download_speed = 0.0
            
            stats = NetworkStats(
                timestamp=current_time,
                upload_speed=max(0, upload_speed),
                download_speed=max(0, download_speed),
                total_upload=net_io.bytes_sent / (1024 * 1024),  # 转换为MB
                total_download=net_io.bytes_recv / (1024 * 1024),  # 转换为MB
                cpu_usage=cpu_percent,
                memory_usage=memory.percent
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"收集网络统计失败: {e}")
            return None
    
    def _process_stats(self, stats: NetworkStats):
        """处理统计数据"""
        # 保存到历史记录
        self.stats_history.append(stats)
        
        # 更新使用量统计
        if self.last_stats:
            time_diff = stats.timestamp - self.last_stats.timestamp
            if time_diff > 0:
                upload_delta = (stats.total_upload - self.last_stats.total_upload)
                download_delta = (stats.total_download - self.last_stats.total_download)
                
                self.daily_usage['upload'] += upload_delta
                self.daily_usage['download'] += download_delta
                self.monthly_usage['upload'] += upload_delta
                self.monthly_usage['download'] += download_delta
        
        # 检查流量限制
        self._check_traffic_limits()
        
        # 应用QoS策略
        if self.config.qos_enabled:
            self._apply_qos_policies(stats)
    
    def _check_traffic_limits(self):
        """检查流量限制"""
        # 检查日流量限制
        daily_total = self.daily_usage['upload'] + self.daily_usage['download']
        if daily_total >= self.config.daily_data_limit:
            self.traffic_limits['daily_limit_reached'] = True
            self.logger.warning(f"日流量限制已到达: {daily_total:.2f}MB / {self.config.daily_data_limit:.2f}MB")
        else:
            self.traffic_limits['daily_limit_reached'] = False
        
        # 检查月流量限制
        monthly_total = self.monthly_usage['upload'] + self.monthly_usage['download']
        if monthly_total >= self.config.monthly_data_limit:
            self.traffic_limits['monthly_limit_reached'] = True
            self.logger.warning(f"月流量限制已到达: {monthly_total:.2f}MB / {self.config.monthly_data_limit:.2f}MB")
        else:
            self.traffic_limits['monthly_limit_reached'] = False
    
    def _apply_qos_policies(self, stats: NetworkStats):
        """应用QoS策略"""
        # 这里实现QoS策略逻辑
        # 由于实际的网络带宽控制需要系统级权限，这里只做模拟
        for policy in self.qos_policies:
            if not policy.enabled:
                continue
            
            # 根据策略调整带宽限制
            if stats.upload_speed > policy.max_bandwidth:
                self.traffic_limits['current_upload_limit'] = min(
                    self.traffic_limits['current_upload_limit'],
                    policy.max_bandwidth
                )
            
            if stats.download_speed > policy.max_bandwidth:
                self.traffic_limits['current_download_limit'] = min(
                    self.traffic_limits['current_download_limit'],
                    policy.max_bandwidth
                )
    
    def get_current_stats(self) -> Optional[NetworkStats]:
        """
        获取当前网络统计
        
        Returns:
            NetworkStats: 当前网络统计信息
        """
        return self.last_stats
    
    def get_bandwidth_usage(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """
        获取带宽使用统计
        
        Args:
            duration_minutes: 统计时长（分钟）
            
        Returns:
            Dict: 带宽使用统计信息
        """
        if not self.stats_history:
            return {}
        
        # 筛选指定时间范围内的数据
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_stats = [s for s in self.stats_history if s.timestamp >= cutoff_time]
        
        if not recent_stats:
            return {}
        
        # 计算统计数据
        upload_speeds = [s.upload_speed for s in recent_stats]
        download_speeds = [s.download_speed for s in recent_stats]
        
        return {
            'duration_minutes': duration_minutes,
            'upload': {
                'average': statistics.mean(upload_speeds),
                'max': max(upload_speeds),
                'min': min(upload_speeds),
                'current': recent_stats[-1].upload_speed if recent_stats else 0
            },
            'download': {
                'average': statistics.mean(download_speeds),
                'max': max(download_speeds),
                'min': min(download_speeds),
                'current': recent_stats[-1].download_speed if recent_stats else 0
            },
            'total_usage': {
                'upload': self.daily_usage['upload'],
                'download': self.daily_usage['download'],
                'total': self.daily_usage['upload'] + self.daily_usage['download']
            }
        }
    
    def get_traffic_report(self) -> Dict[str, Any]:
        """
        获取流量报告
        
        Returns:
            Dict: 流量报告信息
        """
        return {
            'daily': {
                'upload': self.daily_usage['upload'],
                'download': self.daily_usage['download'],
                'total': self.daily_usage['upload'] + self.daily_usage['download'],
                'limit': self.config.daily_data_limit,
                'percentage': ((self.daily_usage['upload'] + self.daily_usage['download']) / self.config.daily_data_limit) * 100
            },
            'monthly': {
                'upload': self.monthly_usage['upload'],
                'download': self.monthly_usage['download'],
                'total': self.monthly_usage['upload'] + self.monthly_usage['download'],
                'limit': self.config.monthly_data_limit,
                'percentage': ((self.monthly_usage['upload'] + self.monthly_usage['download']) / self.config.monthly_data_limit) * 100
            },
            'limits': self.traffic_limits
        }
    
    def add_qos_policy(self, policy: QoSPolicy) -> bool:
        """
        添加QoS策略
        
        Args:
            policy: QoS策略对象
            
        Returns:
            bool: 是否成功添加
        """
        try:
            self.qos_policies.append(policy)
            self.logger.info(f"QoS策略已添加: {policy.name}")
            return True
        except Exception as e:
            self.logger.error(f"添加QoS策略失败: {e}")
            return False
    
    def remove_qos_policy(self, policy_name: str) -> bool:
        """
        移除QoS策略
        
        Args:
            policy_name: 策略名称
            
        Returns:
            bool: 是否成功移除
        """
        try:
            original_count = len(self.qos_policies)
            self.qos_policies = [p for p in self.qos_policies if p.name != policy_name]
            
            if len(self.qos_policies) < original_count:
                self.logger.info(f"QoS策略已移除: {policy_name}")
                return True
            else:
                self.logger.warning(f"QoS策略未找到: {policy_name}")
                return False
        except Exception as e:
            self.logger.error(f"移除QoS策略失败: {e}")
            return False
    
    def update_config(self, new_config: BandwidthConfig):
        """
        更新配置
        
        Args:
            new_config: 新的配置对象
        """
        old_config = self.config
        self.config = new_config
        self.logger.info("配置已更新")
        self.logger.info(f"上传速度限制: {old_config.max_upload_speed} -> {new_config.max_upload_speed} MB/s")
        self.logger.info(f"下载速度限制: {old_config.max_download_speed} -> {new_config.max_download_speed} MB/s")
    
    def optimize_network(self) -> Dict[str, Any]:
        """
        网络优化
        
        Returns:
            Dict: 优化结果
        """
        if not self.config.optimization_enabled:
            return {'status': 'disabled', 'message': '网络优化功能未启用'}
        
        optimization_results = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'optimizations': []
        }
        
        try:
            # 获取网络接口信息
            net_interfaces = psutil.net_if_stats()
            active_interfaces = [name for name, stats in net_interfaces.items() if stats.isup]
            
            # TCP连接优化建议
            tcp_connections = len(psutil.net_connections())
            
            optimization_results['optimizations'] = [
                {
                    'type': 'network_interface',
                    'message': f'检测到 {len(active_interfaces)} 个活跃网络接口',
                    'interfaces': active_interfaces
                },
                {
                    'type': 'tcp_connections',
                    'message': f'当前TCP连接数: {tcp_connections}',
                    'recommendation': '如果连接数过多，考虑关闭不必要的连接' if tcp_connections > 1000 else 'TCP连接数正常'
                },
                {
                    'type': 'bandwidth_optimization',
                    'message': '建议启用QoS策略来优化带宽分配',
                    'current_qos_policies': len(self.qos_policies)
                }
            ]
            
            self.logger.info("网络优化完成")
            
        except Exception as e:
            optimization_results['status'] = 'error'
            optimization_results['message'] = str(e)
            self.logger.error(f"网络优化失败: {e}")
        
        return optimization_results
    
    def reset_daily_usage(self):
        """重置日使用量统计"""
        self.daily_usage = {'upload': 0.0, 'download': 0.0}
        self.logger.info("日使用量统计已重置")
    
    def reset_monthly_usage(self):
        """重置月使用量统计"""
        self.monthly_usage = {'upload': 0.0, 'download': 0.0}
        self.logger.info("月使用量统计已重置")
    
    def export_report(self, filepath: str) -> bool:
        """
        导出报告到文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功导出
        """
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'config': asdict(self.config),
                'traffic_report': self.get_traffic_report(),
                'bandwidth_usage_1h': self.get_bandwidth_usage(60),
                'bandwidth_usage_24h': self.get_bandwidth_usage(1440),
                'qos_policies': [asdict(policy) for policy in self.qos_policies],
                'stats_count': len(self.stats_history)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"报告已导出到: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出报告失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取管理器状态
        
        Returns:
            Dict: 状态信息
        """
        return {
            'is_monitoring': self.is_monitoring,
            'config': asdict(self.config),
            'traffic_limits': self.traffic_limits,
            'qos_policies_count': len(self.qos_policies),
            'stats_history_size': len(self.stats_history),
            'last_update': datetime.fromtimestamp(self.last_stats.timestamp).isoformat() if self.last_stats else None,
            'uptime': time.time() - self.stats_history[0].timestamp if self.stats_history else 0
        }


# 便利函数
def create_default_manager() -> BandwidthManager:
    """创建默认配置的带宽管理器"""
    config = BandwidthConfig()
    return BandwidthManager(config)


def create_custom_manager(
    max_upload: float = 10.0,
    max_download: float = 50.0,
    daily_limit: float = 1024.0,
    monthly_limit: float = 30720.0
) -> BandwidthManager:
    """创建自定义配置的带宽管理器"""
    config = BandwidthConfig(
        max_upload_speed=max_upload,
        max_download_speed=max_download,
        daily_data_limit=daily_limit,
        monthly_data_limit=monthly_limit
    )
    return BandwidthManager(config)


if __name__ == "__main__":
    # 示例用法
    print("W7带宽管理器示例")
    
    # 创建管理器
    manager = create_default_manager()
    
    # 启动监控
    manager.start_monitoring()
    
    try:
        # 运行一段时间来收集数据
        time.sleep(10)
        
        # 获取当前状态
        stats = manager.get_current_stats()
        if stats:
            print(f"当前上传速度: {stats.upload_speed:.2f} MB/s")
            print(f"当前下载速度: {stats.download_speed:.2f} MB/s")
        
        # 获取使用统计
        usage = manager.get_bandwidth_usage(5)
        print(f"5分钟平均上传速度: {usage['upload']['average']:.2f} MB/s")
        print(f"5分钟平均下载速度: {usage['download']['average']:.2f} MB/s")
        
        # 获取流量报告
        report = manager.get_traffic_report()
        print(f"日使用量: {report['daily']['total']:.2f} MB")
        
        # 网络优化
        optimization = manager.optimize_network()
        print(f"优化状态: {optimization['status']}")
        
    finally:
        # 停止监控
        manager.stop_monitoring()
        print("示例运行完成")