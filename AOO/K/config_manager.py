"""
配置管理器
负责管理系统配置，支持多环境配置、配置热更新、配置验证和加密配置
提供统一的配置访问接口和配置变更监听
"""

import os
import json
import logging
import threading
import time
import re
from typing import Dict, Any, Optional, List, Set, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import toml
from threading import RLock
from collections import defaultdict, OrderedDict
import hashlib
import secrets
from cryptography.fernet import Fernet

class ConfigFormat(Enum):
    """配置格式枚举"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"

class ConfigSource(Enum):
    """配置源枚举"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    MEMORY = "memory"
    DEFAULT = "default"

class ConfigEncryption(Enum):
    """配置加密枚举"""
    NONE = "none"
    FERNET = "fernet"
    AES = "aes"

@dataclass
class ConfigMetadata:
    """配置元数据数据类"""
    key: str
    source: ConfigSource
    format: ConfigFormat
    last_modified: float
    checksum: str
    encrypted: bool
    version: str
    description: str

@dataclass
class ConfigChangeEvent:
    """配置变更事件数据类"""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float
    source: str
    user: str

class ConfigManager:
    """配置管理器 - 生产环境级别实现"""
    
    def __init__(self, config_path: str = None, environment: str = None):
        self.config_path = Path(config_path) if config_path else None
        self.environment = environment or os.getenv('AOO_ENVIRONMENT', 'development')
        
        # 配置存储
        self._config_data = OrderedDict()  # 配置键 -> 配置值
        self._config_metadata = OrderedDict()  # 配置键 -> ConfigMetadata
        self._config_sources = defaultdict(OrderedDict)  # 配置源 -> 配置键 -> 配置值
        
        # 默认配置
        self._default_config = {
            'system_name': 'AOO智能量化交易系统',
            'environment': self.environment,
            'auto_discovery': True,
            'modules': {
                'scanner': {
                    'deep_scan': True,
                    'ignore_dirs': ['__pycache__', '.git', 'config', 'logs', 'tests']
                },
                'factory': {
                    'auto_wire': True,
                    'singleton': True
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        # 线程安全
        self._config_lock = RLock()
        self._file_watcher_lock = RLock()
        
        # 配置验证
        self._config_validators = {}
        self._required_keys = set()
        
        # 配置加密
        self._encryption_key = None
        self._fernet = None
        self._encryption_enabled = False
        
        # 配置监听
        self._config_listeners = defaultdict(list)  # 配置键 -> 监听器列表
        self._global_listeners = []  # 全局监听器
        
        # 文件监控
        self._file_watchers = {}
        self._file_watcher_thread = None
        self._file_watcher_interval = 5.0
        
        # 性能统计
        self._config_stats = {
            'total_reads': 0,
            'total_writes': 0,
            'total_updates': 0,
            'total_deletes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'encryption_operations': 0,
            'start_time': time.time()
        }
        
        # 缓存
        self._config_cache = {}
        self._cache_enabled = True
        self._cache_ttl = 300  # 5分钟
        
        # 环境变量
        self._env_vars = {}
        
        # 日志
        self.logger = logging.getLogger('AOO.Config')
        
        # 初始化
        self._initialize_config()
    
    def _initialize_config(self):
        """初始化配置"""
        # 加载环境变量文件
        self._load_env_file()
        
        # 加载默认配置
        self._load_default_config()
        
        # 尝试加载配置文件
        if self.config_path and self.config_path.exists():
            self.load_from_file(self.config_path)
        else:
            self.logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
        
        # 加载环境变量
        self._load_environment_variables()
        
        # 启动文件监控（如果启用）
        if self.get('config.file_watcher.enabled', True):
            self._start_file_watcher()
    
    def _load_env_file(self):
        """加载.env文件"""
        # 尝试多个可能的.env文件位置
        possible_paths = [
            Path(__file__).parent.parent / '.env',  # 项目根目录
            Path(__file__).parent.parent / 'config' / '.env',  # config目录
            Path('.env'),  # 当前工作目录
        ]
        
        env_loaded = False
        for env_path in possible_paths:
            if env_path.exists():
                try:
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                # 移除值中的引号
                                if (value.startswith('"') and value.endswith('"')) or \
                                   (value.startswith("'") and value.endswith("'")):
                                    value = value[1:-1]
                                
                                self._env_vars[key] = value
                                # 同时设置到系统环境变量
                                os.environ[key] = value
                    
                    self.logger.info(f"已加载环境变量文件: {env_path}")
                    env_loaded = True
                    break
                except Exception as e:
                    self.logger.warning(f"加载环境变量文件失败 {env_path}: {e}")
        
        if not env_loaded:
            self.logger.info("未找到环境变量文件，使用系统环境变量")
        
        # 同时从系统环境变量加载
        for key, value in os.environ.items():
            if key not in self._env_vars:
                self._env_vars[key] = value
    
    def _load_default_config(self):
        """加载默认配置"""
        with self._config_lock:
            for key, value in self._default_config.items():
                self._set_config_value(key, value, ConfigSource.DEFAULT, ConfigFormat.JSON)
    
    def _load_environment_variables(self):
        """加载环境变量"""
        env_prefix = self.get('config.env_prefix', 'AOO_')
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                # 转换环境变量名为配置键
                config_key = env_key[len(env_prefix):].lower().replace('_', '.')
                
                # 转换环境变量值
                try:
                    # 尝试解析JSON值
                    if env_value.startswith('{') or env_value.startswith('['):
                        parsed_value = json.loads(env_value)
                    else:
                        parsed_value = env_value
                    
                    self._set_config_value(config_key, parsed_value, ConfigSource.ENVIRONMENT, ConfigFormat.ENV)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，使用原始字符串值
                    self._set_config_value(config_key, env_value, ConfigSource.ENVIRONMENT, ConfigFormat.ENV)
    
    def load_from_file(self, file_path: Path, format: ConfigFormat = None):
        """从文件加载配置"""
        if not file_path.exists():
            self.logger.error(f"配置文件不存在: {file_path}")
            return False
        
        try:
            # 自动检测文件格式
            if format is None:
                suffix = file_path.suffix.lower()
                if suffix == '.json':
                    format = ConfigFormat.JSON
                elif suffix in ['.yaml', '.yml']:
                    format = ConfigFormat.YAML
                elif suffix == '.toml':
                    format = ConfigFormat.TOML
                elif suffix == '.ini':
                    format = ConfigFormat.INI
                else:
                    format = ConfigFormat.JSON
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                if format == ConfigFormat.JSON:
                    config_data = json.load(f)
                elif format == ConfigFormat.YAML:
                    config_data = yaml.safe_load(f)
                elif format == ConfigFormat.TOML:
                    config_data = toml.load(f)
                elif format == ConfigFormat.INI:
                    # 简化INI处理
                    import configparser
                    parser = configparser.ConfigParser()
                    parser.read(file_path)
                    config_data = {}
                    for section in parser.sections():
                        config_data[section] = dict(parser.items(section))
                else:
                    self.logger.error(f"不支持的配置格式: {format}")
                    return False
            
            # 处理配置数据 - 添加环境变量替换
            config_data = self._replace_env_vars(config_data)
            self._process_config_data(config_data, ConfigSource.FILE, format, str(file_path))
            
            self.logger.info(f"配置文件加载成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"配置文件加载失败 {file_path}: {e}")
            return False
    
    def _replace_env_vars(self, config: Any) -> Any:
        """递归替换配置中的环境变量"""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._replace_single_env_var(config)
        else:
            return config
    
    def _replace_single_env_var(self, value: str) -> str:
        """替换单个字符串中的环境变量"""
        # 匹配 ${VAR_NAME} 或 $VAR_NAME 格式
        pattern = r'\$\{([^}]+)\}|\$([a-zA-Z_][a-zA-Z0-9_]*)'
        
        def replace_match(match):
            # 匹配 ${VAR} 或 $VAR 格式
            var_name = match.group(1) or match.group(2)
            if not var_name:
                return match.group(0)
            
            # 首先查找自定义环境变量，然后查找系统环境变量
            env_value = self._env_vars.get(var_name, os.environ.get(var_name, match.group(0)))
            
            # 如果环境变量不存在，保持原样并记录警告
            if env_value == match.group(0):
                self.logger.debug(f"环境变量未设置: {var_name}，使用默认值")
                return match.group(0)
            
            return env_value
        
        return re.sub(pattern, replace_match, value)
    
    def _process_config_data(self, config_data: Dict[str, Any], source: ConfigSource, format: ConfigFormat, source_name: str):
        """处理配置数据"""
        def flatten_config(data: Dict[str, Any], prefix: str = ''):
            """扁平化配置数据"""
            items = []
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    items.extend(flatten_config(value, full_key).items())
                else:
                    items.append((full_key, value))
            return dict(items)
        
        # 扁平化配置
        flat_config = flatten_config(config_data)
        
        # 设置配置值
        with self._config_lock:
            for key, value in flat_config.items():
                self._set_config_value(key, value, source, format, source_name)
    
    def _set_config_value(self, key: str, value: Any, source: ConfigSource, format: ConfigFormat, source_name: str = None):
        """设置配置值"""
        # 检查配置键是否已存在（环境变量优先级低于文件配置）
        if key in self._config_data and source == ConfigSource.ENVIRONMENT:
            self.logger.debug(f"配置键已存在，跳过环境变量: {key}")
            return
        
        # 验证配置值
        if not self._validate_config_value(key, value):
            self.logger.warning(f"配置值验证失败: {key} = {value}")
            return
        
        # 加密敏感配置
        if self._is_sensitive_key(key):
            value = self._encrypt_value(key, value)
        
        # 获取旧值
        old_value = self._config_data.get(key)
        
        # 设置新值
        self._config_data[key] = value
        
        # 更新元数据
        checksum = self._calculate_checksum(value)
        metadata = ConfigMetadata(
            key=key,
            source=source,
            format=format,
            last_modified=time.time(),
            checksum=checksum,
            encrypted=self._is_sensitive_key(key),
            version="1.0",
            description=f"From {source.value}" + (f" ({source_name})" if source_name else "")
        )
        self._config_metadata[key] = metadata
        
        # 更新配置源
        self._config_sources[source][key] = value
        
        # 触发配置变更事件
        if old_value != value:
            self._trigger_config_change(key, old_value, value, source.value)
        
        # 更新统计
        if key in self._config_data:
            self._config_stats['total_updates'] += 1
        else:
            self._config_stats['total_writes'] += 1
        
        self.logger.debug(f"配置设置: {key} = [REDACTED]" if self._is_sensitive_key(key) else f"配置设置: {key} = {value}")
    
    def _validate_config_value(self, key: str, value: Any) -> bool:
        """验证配置值"""
        # 检查必需键
        if key in self._required_keys and value is None:
            self.logger.error(f"必需配置为空: {key}")
            return False
        
        # 调用自定义验证器
        if key in self._config_validators:
            validator = self._config_validators[key]
            try:
                if not validator(value):
                    self.logger.error(f"配置值验证失败: {key} = {value}")
                    self._config_stats['validation_errors'] += 1
                    return False
            except Exception as e:
                self.logger.error(f"配置验证器执行失败 {key}: {e}")
                return False
        
        return True
    
    def _is_sensitive_key(self, key: str) -> bool:
        """检查是否为敏感配置键"""
        sensitive_patterns = [
            'password', 'secret', 'key', 'token', 'credential',
            'api_key', 'access_key', 'private_key'
        ]
        return any(pattern in key.lower() for pattern in sensitive_patterns)
    
    def _encrypt_value(self, key: str, value: Any) -> Any:
        """加密配置值"""
        if not self._encryption_enabled or not self._fernet:
            return value
        
        try:
            if isinstance(value, str):
                encrypted_value = self._fernet.encrypt(value.encode()).decode()
                self._config_stats['encryption_operations'] += 1
                return encrypted_value
            else:
                self.logger.warning(f"无法加密非字符串配置: {key}")
                return value
        except Exception as e:
            self.logger.error(f"配置加密失败 {key}: {e}")
            return value
    
    def _decrypt_value(self, key: str, value: Any) -> Any:
        """解密配置值"""
        if not self._encryption_enabled or not self._fernet:
            return value
        
        try:
            if isinstance(value, str) and value.startswith('gAAAAA'):
                decrypted_value = self._fernet.decrypt(value.encode()).decode()
                self._config_stats['encryption_operations'] += 1
                return decrypted_value
            else:
                return value
        except Exception as e:
            self.logger.error(f"配置解密失败 {key}: {e}")
            return value
    
    def _calculate_checksum(self, value: Any) -> str:
        """计算配置值的校验和"""
        value_str = str(value)
        return hashlib.md5(value_str.encode()).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        with self._config_lock:
            # 检查缓存
            cache_key = f"{key}_{self.environment}"
            if (self._cache_enabled and 
                cache_key in self._config_cache and
                time.time() - self._config_cache[cache_key]['timestamp'] < self._cache_ttl):
                self._config_stats['cache_hits'] += 1
                return self._config_cache[cache_key]['value']
            
            self._config_stats['cache_misses'] += 1
            self._config_stats['total_reads'] += 1
            
            # 获取配置值
            value = self._config_data.get(key, default)
            
            # 解密敏感配置
            if self._is_sensitive_key(key):
                value = self._decrypt_value(key, value)
            
            # 更新缓存
            if self._cache_enabled:
                self._config_cache[cache_key] = {
                    'value': value,
                    'timestamp': time.time()
                }
            
            return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置章节"""
        result = {}
        prefix = f"{section}."
        
        with self._config_lock:
            for key, value in self._config_data.items():
                if key.startswith(prefix):
                    # 解密敏感配置
                    if self._is_sensitive_key(key):
                        value = self._decrypt_value(key, value)
                    
                    # 提取子键
                    sub_key = key[len(prefix):]
                    result[sub_key] = value
                elif key == section:
                    # 直接匹配章节名
                    if self._is_sensitive_key(key):
                        value = self._decrypt_value(key, value)
                    result = value
                    break
        
        return result
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._config_lock:
            all_config = {}
            for key, value in self._config_data.items():
                # 解密敏感配置
                if self._is_sensitive_key(key):
                    value = self._decrypt_value(key, value)
                all_config[key] = value
            
            return all_config
    
    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.MEMORY) -> bool:
        """设置配置值"""
        with self._config_lock:
            return self._set_config_value(key, value, source, ConfigFormat.JSON, "manual")
    
    def update(self, updates: Dict[str, Any], source: ConfigSource = ConfigSource.MEMORY) -> bool:
        """批量更新配置"""
        with self._config_lock:
            success = True
            for key, value in updates.items():
                if not self._set_config_value(key, value, source, ConfigFormat.JSON, "batch_update"):
                    success = False
            
            return success
    
    def delete(self, key: str) -> bool:
        """删除配置"""
        with self._config_lock:
            if key not in self._config_data:
                return False
            
            # 获取旧值
            old_value = self._config_data[key]
            
            # 删除配置
            del self._config_data[key]
            
            # 删除元数据
            if key in self._config_metadata:
                del self._config_metadata[key]
            
            # 从配置源中删除
            for source_config in self._config_sources.values():
                if key in source_config:
                    del source_config[key]
            
            # 触发配置删除事件
            self._trigger_config_change(key, old_value, None, "delete")
            
            # 更新统计
            self._config_stats['total_deletes'] += 1
            
            self.logger.info(f"配置删除: {key}")
            return True
    
    def exists(self, key: str) -> bool:
        """检查配置是否存在"""
        with self._config_lock:
            return key in self._config_data
    
    def reload(self) -> bool:
        """重新加载配置"""
        self.logger.info("重新加载配置...")
        
        # 清空当前配置
        with self._config_lock:
            self._config_data.clear()
            self._config_metadata.clear()
            self._config_sources.clear()
            self._config_cache.clear()
        
        # 重新初始化配置
        self._initialize_config()
        
        self.logger.info("配置重新加载完成")
        return True
    
    def save_to_file(self, file_path: Path, format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """保存配置到文件"""
        try:
            # 准备配置数据
            config_data = self.get_all_config()
            
            # 创建目录
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config_data, f, ensure_ascii=False, indent=4)
                elif format == ConfigFormat.YAML:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
                elif format == ConfigFormat.TOML:
                    toml.dump(config_data, f)
                else:
                    self.logger.error(f"不支持的配置格式: {format}")
                    return False
            
            self.logger.info(f"配置保存成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"配置保存失败 {file_path}: {e}")
            return False
    
    def enable_encryption(self, encryption_key: str = None):
        """启用配置加密"""
        if not encryption_key:
            # 生成随机密钥
            encryption_key = Fernet.generate_key()
        
        self._encryption_key = encryption_key
        self._fernet = Fernet(encryption_key)
        self._encryption_enabled = True
        
        self.logger.info("配置加密已启用")
    
    def disable_encryption(self):
        """禁用配置加密"""
        self._encryption_enabled = False
        self._fernet = None
        self._encryption_key = None
        
        self.logger.info("配置加密已禁用")
    
    def add_validator(self, key: str, validator: Callable[[Any], bool]):
        """添加配置验证器"""
        self._config_validators[key] = validator
        self.logger.debug(f"配置验证器已添加: {key}")
    
    def mark_required(self, key: str):
        """标记配置为必需"""
        self._required_keys.add(key)
        self.logger.debug(f"配置标记为必需: {key}")
    
    def add_listener(self, key: str, listener: Callable[[ConfigChangeEvent], None]):
        """添加配置监听器"""
        with self._config_lock:
            self._config_listeners[key].append(listener)
            self.logger.debug(f"配置监听器已添加: {key}")
    
    def add_global_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """添加全局配置监听器"""
        with self._config_lock:
            self._global_listeners.append(listener)
            self.logger.debug("全局配置监听器已添加")
    
    def _trigger_config_change(self, key: str, old_value: Any, new_value: Any, source: str):
        """触发配置变更事件"""
        event = ConfigChangeEvent(
            key=key,
            old_value=old_value,
            new_value=new_value,
            timestamp=time.time(),
            source=source,
            user="system"
        )
        
        # 触发特定键的监听器
        listeners = self._config_listeners.get(key, [])
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"配置监听器执行失败 {key}: {e}")
        
        # 触发全局监听器
        for listener in self._global_listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"全局配置监听器执行失败: {e}")
    
    def _start_file_watcher(self):
        """启动文件监控"""
        if self._file_watcher_thread is not None and self._file_watcher_thread.is_alive():
            return
        
        self._file_watcher_thread = threading.Thread(
            target=self._file_watcher_worker,
            daemon=True,
            name="ConfigFileWatcher"
        )
        self._file_watcher_thread.start()
        self.logger.info("配置文件监控已启动")
    
    def _file_watcher_worker(self):
        """文件监控工作线程"""
        last_modified = {}
        
        while True:
            try:
                # 检查配置文件变化
                if self.config_path and self.config_path.exists():
                    current_modified = self.config_path.stat().st_mtime
                    
                    if self.config_path not in last_modified:
                        last_modified[self.config_path] = current_modified
                    elif last_modified[self.config_path] != current_modified:
                        self.logger.info(f"检测到配置文件变化: {self.config_path}")
                        self.reload()
                        last_modified[self.config_path] = current_modified
                
                # 等待下一次检查
                time.sleep(self._file_watcher_interval)
                
            except Exception as e:
                self.logger.error(f"文件监控工作线程异常: {e}")
                time.sleep(self._file_watcher_interval)
    
    def get_metadata(self, key: str) -> Optional[ConfigMetadata]:
        """获取配置元数据"""
        with self._config_lock:
            return self._config_metadata.get(key)
    
    def get_all_metadata(self) -> Dict[str, ConfigMetadata]:
        """获取所有配置元数据"""
        with self._config_lock:
            return dict(self._config_metadata)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = time.time() - self._config_stats['start_time']
        
        return {
            'config': {
                'total_keys': len(self._config_data),
                'sources': {source.value: len(keys) for source, keys in self._config_sources.items()},
                'environment': self.environment,
                'encryption_enabled': self._encryption_enabled
            },
            'performance': {
                'uptime': uptime,
                'reads_per_second': self._config_stats['total_reads'] / uptime if uptime > 0 else 0,
                'writes_per_second': self._config_stats['total_writes'] / uptime if uptime > 0 else 0,
                'updates_per_second': self._config_stats['total_updates'] / uptime if uptime > 0 else 0,
                'cache_hit_rate': self._config_stats['cache_hits'] / (self._config_stats['cache_hits'] + self._config_stats['cache_misses']) if (self._config_stats['cache_hits'] + self._config_stats['cache_misses']) > 0 else 0
            },
            'operations': self._config_stats.copy(),
            'validation': {
                'validators_registered': len(self._config_validators),
                'required_keys': len(self._required_keys),
                'validation_errors': self._config_stats['validation_errors']
            },
            'security': {
                'encryption_operations': self._config_stats['encryption_operations'],
                'sensitive_keys': len([k for k in self._config_data.keys() if self._is_sensitive_key(k)])
            }
        }
    
    def get_env_var(self, key: str, default: str = "") -> str:
        """获取环境变量"""
        return self._env_vars.get(key, default)
    
    def set_env_var(self, key: str, value: str):
        """设置环境变量"""
        self._env_vars[key] = value
        os.environ[key] = value
        self.logger.debug(f"设置环境变量: {key}")
    
    def validate_config(self) -> Dict[str, Any]:
        """验证配置完整性并返回结果"""
        missing_vars = []
        warnings = []
        
        # 检查必需的环境变量
        required_env_vars = [
            'ENCRYPTION_KEY'
        ]
        
        # 根据配置检查交易所API密钥
        if self.get('exchanges.binance.enabled', False):
            required_env_vars.extend(['BINANCE_API_KEY', 'BINANCE_API_SECRET'])
        
        if self.get('exchanges.okx.enabled', False):
            required_env_vars.extend(['OKX_API_KEY', 'OKX_API_SECRET', 'OKX_PASSPHRASE'])
        
        for var in required_env_vars:
            if not self.get_env_var(var):
                missing_vars.append(var)
        
        # 检查配置完整性
        required_sections = ['scanner', 'registry', 'factory']
        for section in required_sections:
            if not self.get_section(section):
                warnings.append(f"缺少配置章节: {section}")
        
        # 检查日志配置
        if not self.get('logging.level'):
            warnings.append("未设置日志级别，使用默认值INFO")
        
        return {
            'valid': len(missing_vars) == 0,
            'missing_environment_variables': missing_vars,
            'warnings': warnings,
            'environment_variables_count': len(self._env_vars),
            'config_sections_count': len(self._config_data),
            'pending_updates': 0
        }
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        validation = self.validate_config()
        
        return {
            'config_path': str(self.config_path) if self.config_path else None,
            'environment': self.environment,
            'system_name': self.get('system_name', 'unknown'),
            'auto_discovery': self.get('auto_discovery', False),
            'validation': validation,
            'config_size': len(str(self._config_data)),
            'env_vars_count': len(self._env_vars)
        }


# 全局配置管理器实例
_global_config_manager = None

def get_global_config_manager(config_path: str = None) -> ConfigManager:
    """获取全局配置管理器实例"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_path)
    return _global_config_manager

def create_default_config(config_path: str) -> bool:
    """创建默认配置文件"""
    try:
        config_manager = get_global_config_manager(config_path)
        return config_manager.save_to_file(Path(config_path))
    except Exception as e:
        logging.error(f"创建默认配置文件失败: {e}")
        return False


# 配置管理器构建器
class ConfigManagerBuilder:
    """配置管理器构建器"""
    
    def __init__(self):
        self._config_path = None
        self._initial_config = {}
    
    def set_config_path(self, path: str) -> 'ConfigManagerBuilder':
        """设置配置文件路径"""
        self._config_path = path
        return self
    
    def set_initial_config(self, config: Dict[str, Any]) -> 'ConfigManagerBuilder':
        """设置初始配置"""
        self._initial_config = config
        return self
    
    def build(self) -> ConfigManager:
        """构建配置管理器实例"""
        config_manager = get_global_config_manager(self._config_path)
        
        # 应用初始配置
        for key, value in self._initial_config.items():
            config_manager.set(key, value)
        
        return config_manager