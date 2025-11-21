"""
Y8存储清理器主实现文件

提供完整的存储清理功能，包括：
- 自动清理和手动清理
- 多种清理策略
- 清理统计和监控
- 配置管理和报告生成
"""

import os
import sys
import time
import json
import shutil
import logging
import threading
import hashlib
import glob
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

try:
    import psutil
except ImportError:
    psutil = None


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/D/AO/AOO/Y/Y8/cleaner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CleanStats:
    """清理统计信息"""
    total_files_scanned: int = 0
    total_files_cleaned: int = 0
    total_space_freed: int = 0  # bytes
    start_time: float = 0.0
    end_time: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    @property
    def duration(self) -> float:
        """清理耗时（秒）"""
        return self.end_time - self.start_time
    
    @property
    def space_freed_mb(self) -> float:
        """释放空间（MB）"""
        return self.total_space_freed / (1024 * 1024)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['duration'] = self.duration
        data['space_freed_mb'] = self.space_freed_mb
        return data


@dataclass
class CleanConfig:
    """清理配置"""
    auto_clean_enabled: bool = True
    auto_clean_interval: int = 3600  # 秒
    max_temp_age: int = 86400  # 24小时
    max_cache_age: int = 604800  # 7天
    max_log_age: int = 2592000  # 30天
    min_file_size: int = 1024  # 1KB
    dry_run: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    backup_before_clean: bool = False
    backup_path: str = ""
    excluded_paths: List[str] = None
    included_extensions: List[str] = None
    
    def __post_init__(self):
        if self.excluded_paths is None:
            self.excluded_paths = [
                '/workspace/D/AO/AOO/Y/Y8/cleaner.log',
                '/workspace/D/AO/AOO/Y/Y8/backup',
                os.path.expanduser('~/.cache')
            ]
        if self.included_extensions is None:
            self.included_extensions = [
                '.tmp', '.temp', '.cache', '.log', '.bak', '.old', '.swp'
            ]


class CleanStrategy(ABC):
    """清理策略基类"""
    
    @abstractmethod
    def should_clean(self, file_path: str, file_stats: Dict) -> bool:
        """判断文件是否应该清理"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """获取清理优先级，数值越小优先级越高"""
        pass


class AgeBasedStrategy(CleanStrategy):
    """基于文件年龄的清理策略"""
    
    def __init__(self, max_age_seconds: int):
        self.max_age_seconds = max_age_seconds
    
    def should_clean(self, file_path: str, file_stats: Dict) -> bool:
        file_age = time.time() - file_stats.get('mtime', 0)
        return file_age > self.max_age_seconds
    
    def get_priority(self) -> int:
        return 1


class SizeBasedStrategy(CleanStrategy):
    """基于文件大小的清理策略"""
    
    def __init__(self, min_size_bytes: int):
        self.min_size_bytes = min_size_bytes
    
    def should_clean(self, file_path: str, file_stats: Dict) -> bool:
        file_size = file_stats.get('size', 0)
        return file_size >= self.min_size_bytes
    
    def get_priority(self) -> int:
        return 2


class ExtensionBasedStrategy(CleanStrategy):
    """基于文件扩展名的清理策略"""
    
    def __init__(self, extensions: List[str]):
        self.extensions = extensions
    
    def should_clean(self, file_path: str, file_stats: Dict) -> bool:
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in self.extensions
    
    def get_priority(self) -> int:
        return 3


class CleanMonitor:
    """清理监控器"""
    
    def __init__(self):
        self.progress = 0.0
        self.status = "idle"
        self.current_file = ""
        self.start_time = 0.0
        self.total_files = 0
        self.processed_files = 0
        self.lock = threading.Lock()
    
    def start_monitoring(self, total_files: int):
        """开始监控"""
        with self.lock:
            self.total_files = total_files
            self.processed_files = 0
            self.progress = 0.0
            self.status = "running"
            self.start_time = time.time()
    
    def update_progress(self, processed: int, current_file: str = ""):
        """更新进度"""
        with self.lock:
            self.processed_files = processed
            self.current_file = current_file
            if self.total_files > 0:
                self.progress = (processed / self.total_files) * 100
    
    def finish_monitoring(self):
        """结束监控"""
        with self.lock:
            self.status = "completed"
    
    def get_status(self) -> Dict:
        """获取当前状态"""
        with self.lock:
            return {
                'progress': self.progress,
                'status': self.status,
                'current_file': self.current_file,
                'processed_files': self.processed_files,
                'total_files': self.total_files,
                'elapsed_time': time.time() - self.start_time
            }


class CleanReporter:
    """清理报告生成器"""
    
    def __init__(self, output_dir: str = "/workspace/D/AO/AOO/Y/Y8/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, stats: CleanStats, config: CleanConfig) -> str:
        """生成清理报告"""
        report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"clean_report_{report_id}.json")
        
        report_data = {
            'report_id': report_id,
            'timestamp': datetime.now().isoformat(),
            'config': asdict(config),
            'statistics': stats.to_dict(),
            'summary': {
                'files_scanned': stats.total_files_scanned,
                'files_cleaned': stats.total_files_cleaned,
                'space_freed_mb': round(stats.space_freed_mb, 2),
                'duration_seconds': round(stats.duration, 2),
                'success_rate': round(
                    (stats.total_files_cleaned / max(stats.total_files_scanned, 1)) * 100, 2
                )
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"清理报告已生成: {report_file}")
        return report_file
    
    def generate_html_report(self, stats: CleanStats, config: CleanConfig) -> str:
        """生成HTML格式报告"""
        report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"clean_report_{report_id}.html")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Y8存储清理器报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .stats {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; min-width: 200px; }}
        .progress-bar {{ width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: #4CAF50; transition: width 0.3s; }}
        .error {{ color: #f44336; }}
        .warning {{ color: #ff9800; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Y8存储清理器报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>扫描文件</h3>
            <p>{stats.total_files_scanned:,}</p>
        </div>
        <div class="stat-card">
            <h3>清理文件</h3>
            <p>{stats.total_files_cleaned:,}</p>
        </div>
        <div class="stat-card">
            <h3>释放空间</h3>
            <p>{stats.space_freed_mb:.2f} MB</p>
        </div>
        <div class="stat-card">
            <h3>清理耗时</h3>
            <p>{stats.duration:.2f} 秒</p>
        </div>
        <div class="stat-card">
            <h3>成功率</h3>
            <p>{round((stats.total_files_cleaned / max(stats.total_files_scanned, 1)) * 100, 2)}%</p>
        </div>
    </div>
    
    <div class="progress-bar">
        <div class="progress-fill" style="width: {round((stats.total_files_cleaned / max(stats.total_files_scanned, 1)) * 100, 2)}%"></div>
    </div>
    
    <h3>配置信息</h3>
    <ul>
        <li>自动清理: {'启用' if config.auto_clean_enabled else '禁用'}</li>
        <li>干运行模式: {'启用' if config.dry_run else '禁用'}</li>
        <li>并行处理: {'启用' if config.parallel_processing else '禁用'}</li>
        <li>备份文件: {'启用' if config.backup_before_clean else '禁用'}</li>
    </ul>
    
    {'<h3>错误信息</h3><ul>' + ''.join(f'<li class="error">{error}</li>' for error in stats.errors) + '</ul>' if stats.errors else ''}
    {'<h3>警告信息</h3><ul>' + ''.join(f'<li class="warning">{warning}</li>' for warning in stats.warnings) + '</ul>' if stats.warnings else ''}
</body>
</html>
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML清理报告已生成: {report_file}")
        return report_file


class FileCleaner:
    """文件清理器"""
    
    def __init__(self, config: CleanConfig, monitor: CleanMonitor):
        self.config = config
        self.monitor = monitor
        self.strategies = self._init_strategies()
    
    def _init_strategies(self) -> List[CleanStrategy]:
        """初始化清理策略"""
        strategies = [
            AgeBasedStrategy(self.config.max_temp_age),
            AgeBasedStrategy(self.config.max_cache_age),
            AgeBasedStrategy(self.config.max_log_age),
            SizeBasedStrategy(self.config.min_file_size),
            ExtensionBasedStrategy(self.config.included_extensions)
        ]
        return sorted(strategies, key=lambda x: x.get_priority())
    
    def scan_files(self, paths: List[str]) -> List[str]:
        """扫描文件"""
        files_to_clean = []
        total_files = 0
        
        for path in paths:
            if os.path.isfile(path):
                total_files += 1
                if self._should_clean_file(path):
                    files_to_clean.append(path)
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    # 跳过排除的路径
                    dirs[:] = [d for d in dirs if not any(
                        excluded in os.path.join(root, d) for excluded in self.config.excluded_paths
                    )]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_files += 1
                        if self._should_clean_file(file_path):
                            files_to_clean.append(file_path)
        
        self.monitor.start_monitoring(total_files)
        return files_to_clean
    
    def _should_clean_file(self, file_path: str) -> bool:
        """判断文件是否应该清理"""
        # 检查排除路径
        if any(excluded in file_path for excluded in self.config.excluded_paths):
            return False
        
        try:
            file_stats = os.stat(file_path)
            file_info = {
                'size': file_stats.st_size,
                'mtime': file_stats.st_mtime,
                'atime': file_stats.st_atime
            }
            
            # 应用清理策略
            for strategy in self.strategies:
                if strategy.should_clean(file_path, file_info):
                    return True
            
            return False
        except (OSError, IOError) as e:
            logger.warning(f"无法访问文件 {file_path}: {e}")
            return False
    
    def clean_files(self, files: List[str], stats: CleanStats) -> bool:
        """清理文件"""
        success = True
        
        for i, file_path in enumerate(files):
            try:
                self.monitor.update_progress(i + 1, file_path)
                
                if self.config.dry_run:
                    logger.info(f"[干运行] 将清理文件: {file_path}")
                    stats.total_files_cleaned += 1
                    stats.total_space_freed += os.path.getsize(file_path)
                    continue
                
                # 备份文件（如果启用）
                if self.config.backup_before_clean and self.config.backup_path:
                    backup_path = os.path.join(
                        self.config.backup_path,
                        os.path.relpath(file_path, '/')
                    )
                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    shutil.copy2(file_path, backup_path)
                
                # 删除文件
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    stats.total_files_cleaned += 1
                    stats.total_space_freed += file_size
                    logger.info(f"已清理文件: {file_path} ({file_size} bytes)")
                
            except Exception as e:
                error_msg = f"清理文件失败 {file_path}: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
                success = False
        
        self.monitor.finish_monitoring()
        return success


class CacheCleaner:
    """缓存清理器"""
    
    def __init__(self, config: CleanConfig):
        self.config = config
        self.cache_paths = [
            os.path.expanduser('~/.cache'),
            '/tmp',
            '/var/cache',
            tempfile.gettempdir()
        ]
    
    def clean_cache(self, stats: CleanStats) -> bool:
        """清理缓存"""
        success = True
        
        for cache_path in self.cache_paths:
            if not os.path.exists(cache_path):
                continue
            
            try:
                for root, dirs, files in os.walk(cache_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # 检查文件年龄
                        file_age = time.time() - os.path.getmtime(file_path)
                        if file_age > self.config.max_cache_age:
                            try:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                stats.total_files_cleaned += 1
                                stats.total_space_freed += file_size
                                logger.info(f"已清理缓存: {file_path}")
                            except Exception as e:
                                error_msg = f"清理缓存失败 {file_path}: {e}"
                                logger.error(error_msg)
                                stats.errors.append(error_msg)
                                success = False
            except Exception as e:
                error_msg = f"扫描缓存目录失败 {cache_path}: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
                success = False
        
        return success


class TempCleaner:
    """临时文件清理器"""
    
    def __init__(self, config: CleanConfig):
        self.config = config
        self.temp_paths = [
            '/tmp',
            '/var/tmp',
            tempfile.gettempdir(),
            os.path.expanduser('~/tmp')
        ]
    
    def clean_temp_files(self, stats: CleanStats) -> bool:
        """清理临时文件"""
        success = True
        
        for temp_path in self.temp_paths:
            if not os.path.exists(temp_path):
                continue
            
            try:
                for root, dirs, files in os.walk(temp_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # 检查文件年龄和扩展名
                        file_age = time.time() - os.path.getmtime(file_path)
                        file_ext = os.path.splitext(file_path)[1].lower()
                        
                        if file_age > self.config.max_temp_age or file_ext in ['.tmp', '.temp']:
                            try:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                stats.total_files_cleaned += 1
                                stats.total_space_freed += file_size
                                logger.info(f"已清理临时文件: {file_path}")
                            except Exception as e:
                                error_msg = f"清理临时文件失败 {file_path}: {e}"
                                logger.error(error_msg)
                                stats.errors.append(error_msg)
                                success = False
            except Exception as e:
                error_msg = f"扫描临时目录失败 {temp_path}: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
                success = False
        
        return success


class LogCleaner:
    """日志文件清理器"""
    
    def __init__(self, config: CleanConfig):
        self.config = config
        self.log_paths = [
            '/var/log',
            os.path.expanduser('~/logs'),
            '/workspace/D/AO/AOO/Y/Y8/logs'
        ]
    
    def clean_log_files(self, stats: CleanStats) -> bool:
        """清理日志文件"""
        success = True
        
        for log_path in self.log_paths:
            if not os.path.exists(log_path):
                continue
            
            try:
                for root, dirs, files in os.walk(log_path):
                    for file in files:
                        if file.endswith('.log') or file.endswith('.log.*'):
                            file_path = os.path.join(root, file)
                            
                            # 检查文件年龄
                            file_age = time.time() - os.path.getmtime(file_path)
                            if file_age > self.config.max_log_age:
                                try:
                                    file_size = os.path.getsize(file_path)
                                    os.remove(file_path)
                                    stats.total_files_cleaned += 1
                                    stats.total_space_freed += file_size
                                    logger.info(f"已清理日志文件: {file_path}")
                                except Exception as e:
                                    error_msg = f"清理日志文件失败 {file_path}: {e}"
                                    logger.error(error_msg)
                                    stats.errors.append(error_msg)
                                    success = False
            except Exception as e:
                error_msg = f"扫描日志目录失败 {log_path}: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
                success = False
        
        return success


class AutoCleaner:
    """自动清理器"""
    
    def __init__(self, storage_cleaner):
        self.storage_cleaner = storage_cleaner
        self.running = False
        self.thread = None
    
    def start(self):
        """启动自动清理"""
        if self.running:
            logger.warning("自动清理已在运行")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._auto_clean_loop, daemon=True)
        self.thread.start()
        logger.info("自动清理已启动")
    
    def stop(self):
        """停止自动清理"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("自动清理已停止")
    
    def _auto_clean_loop(self):
        """自动清理循环"""
        while self.running:
            try:
                # 等待指定间隔
                time.sleep(self.storage_cleaner.config.auto_clean_interval)
                
                if not self.running:
                    break
                
                logger.info("执行自动清理")
                self.storage_cleaner.clean_storage()
                
            except Exception as e:
                logger.error(f"自动清理循环出错: {e}")
                time.sleep(60)  # 出错后等待1分钟再继续


class ManualCleaner:
    """手动清理器"""
    
    def __init__(self, storage_cleaner):
        self.storage_cleaner = storage_cleaner
    
    def clean_now(self) -> CleanStats:
        """立即执行清理"""
        logger.info("开始手动清理")
        return self.storage_cleaner.clean_storage()
    
    def clean_specific_path(self, path: str) -> CleanStats:
        """清理指定路径"""
        logger.info(f"开始清理指定路径: {path}")
        return self.storage_cleaner.clean_storage([path])


class StorageCleaner:
    """主存储清理器类"""
    
    def __init__(self, config: Optional[CleanConfig] = None):
        self.config = config or CleanConfig()
        self.monitor = CleanMonitor()
        self.reporter = CleanReporter()
        
        # 初始化各种清理器
        self.file_cleaner = FileCleaner(self.config, self.monitor)
        self.cache_cleaner = CacheCleaner(self.config)
        self.temp_cleaner = TempCleaner(self.config)
        self.log_cleaner = LogCleaner(self.config)
        
        # 初始化自动和手动清理器
        self.auto_cleaner = AutoCleaner(self)
        self.manual_cleaner = ManualCleaner(self)
        
        logger.info("存储清理器初始化完成")
    
    def start_auto_clean(self):
        """启动自动清理"""
        if not self.config.auto_clean_enabled:
            logger.warning("自动清理未启用")
            return
        
        self.auto_cleaner.start()
    
    def stop_auto_clean(self):
        """停止自动清理"""
        self.auto_cleaner.stop()
    
    def clean_storage(self, paths: Optional[List[str]] = None) -> CleanStats:
        """执行存储清理"""
        stats = CleanStats()
        stats.start_time = time.time()
        
        try:
            # 默认清理路径
            if paths is None:
                paths = [
                    '/tmp',
                    tempfile.gettempdir(),
                    os.path.expanduser('~/.cache'),
                    '/workspace/D/AO/AOO/Y/Y8/temp'
                ]
            
            logger.info(f"开始清理存储，路径: {paths}")
            
            # 1. 清理临时文件
            logger.info("清理临时文件...")
            self.temp_cleaner.clean_temp_files(stats)
            
            # 2. 清理缓存文件
            logger.info("清理缓存文件...")
            self.cache_cleaner.clean_cache(stats)
            
            # 3. 清理日志文件
            logger.info("清理日志文件...")
            self.log_cleaner.clean_log_files(stats)
            
            # 4. 扫描和清理其他文件
            logger.info("扫描和清理其他文件...")
            files_to_clean = self.file_cleaner.scan_files(paths)
            stats.total_files_scanned = self.monitor.total_files
            
            if files_to_clean:
                logger.info(f"找到 {len(files_to_clean)} 个需要清理的文件")
                self.file_cleaner.clean_files(files_to_clean, stats)
            else:
                logger.info("没有找到需要清理的文件")
                self.monitor.finish_monitoring()
            
        except Exception as e:
            error_msg = f"存储清理过程中发生错误: {e}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
        
        finally:
            stats.end_time = time.time()
            
            # 生成报告
            try:
                json_report = self.reporter.generate_report(stats, self.config)
                html_report = self.reporter.generate_html_report(stats, self.config)
                logger.info(f"清理报告已生成: {json_report}, {html_report}")
            except Exception as e:
                logger.error(f"生成报告失败: {e}")
        
        logger.info(f"存储清理完成，耗时: {stats.duration:.2f}秒")
        return stats
    
    def get_clean_status(self) -> Dict:
        """获取清理状态"""
        return self.monitor.get_status()
    
    def get_disk_usage(self) -> Dict:
        """获取磁盘使用情况"""
        if psutil is None:
            logger.warning("psutil未安装，无法获取磁盘使用情况")
            return {}
        
        try:
            usage = psutil.disk_usage('/')
            return {
                'total': usage.total,
                'used': usage.used,
                'free': usage.free,
                'percent': (usage.used / usage.total) * 100
            }
        except Exception as e:
            logger.error(f"获取磁盘使用情况失败: {e}")
            return {}
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"配置已更新: {key} = {value}")
            else:
                logger.warning(f"未知配置项: {key}")
    
    def save_config(self, config_file: str = "/workspace/D/AO/AOO/Y/Y8/cleaner_config.json"):
        """保存配置到文件"""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {config_file}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def load_config(self, config_file: str = "/workspace/D/AO/AOO/Y/Y8/cleaner_config.json"):
        """从文件加载配置"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                logger.info(f"配置已从文件加载: {config_file}")
            else:
                logger.warning(f"配置文件不存在: {config_file}")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")


# 便利函数
def create_cleaner(config: Optional[CleanConfig] = None) -> StorageCleaner:
    """创建存储清理器实例"""
    return StorageCleaner(config)


def quick_clean(dry_run: bool = False) -> CleanStats:
    """快速清理（使用默认配置）"""
    config = CleanConfig(dry_run=dry_run)
    cleaner = StorageCleaner(config)
    return cleaner.clean_storage()


if __name__ == "__main__":
    # 示例使用
    print("Y8存储清理器启动...")
    
    # 创建清理器
    config = CleanConfig(
        auto_clean_enabled=False,
        dry_run=True,  # 演示模式，不实际删除文件
        max_temp_age=3600,  # 1小时
        max_cache_age=86400  # 24小时
    )
    
    cleaner = StorageCleaner(config)
    
    # 执行清理
    stats = cleaner.clean_storage()
    
    # 打印结果
    print(f"清理完成:")
    print(f"- 扫描文件: {stats.total_files_scanned}")
    print(f"- 清理文件: {stats.total_files_cleaned}")
    print(f"- 释放空间: {stats.space_freed_mb:.2f} MB")
    print(f"- 清理耗时: {stats.duration:.2f} 秒")
    
    if stats.errors:
        print(f"- 错误数量: {len(stats.errors)}")
    
    if stats.warnings:
        print(f"- 警告数量: {len(stats.warnings)}")