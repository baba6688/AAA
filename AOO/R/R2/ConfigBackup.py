"""
R2配置备份器 - 主要实现文件
提供完整的配置备份、恢复、版本管理功能
"""

import os
import json
import shutil
import hashlib
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import base64
import re
import configparser

# 可选依赖
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
from dataclasses import dataclass, asdict
from enum import Enum


class ConfigType(Enum):
    """配置类型枚举"""
    SYSTEM = "system"
    USER = "user"
    APPLICATION = "application"
    TEMPLATE = "template"


class BackupStatus(Enum):
    """备份状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ConfigMetadata:
    """配置元数据"""
    name: str
    config_type: ConfigType
    version: str
    created_at: str
    description: str = ""
    tags: List[str] = None
    checksum: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class BackupRecord:
    """备份记录"""
    id: str
    metadata: ConfigMetadata
    backup_path: str
    size: int
    status: BackupStatus
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        metadata_dict = asdict(self.metadata)
        metadata_dict['config_type'] = self.metadata.config_type.value
        return {
            'id': self.id,
            'metadata': metadata_dict,
            'backup_path': self.backup_path,
            'size': self.size,
            'status': self.status.value,
            'error_message': self.error_message
        }


class SensitiveDataProtector:
    """敏感信息保护器"""
    
    def __init__(self, password: Optional[str] = None):
        self.password = password or "default_r2_backup_key"
        self._setup_encryption()
    
    def _setup_encryption(self):
        """设置加密"""
        if CRYPTO_AVAILABLE:
            password_bytes = self.password.encode()
            salt = b'r2_backup_salt_2025'  # 固定盐值，生产环境应随机生成
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            self.cipher = Fernet(key)
            self.use_real_encryption = True
        else:
            # 后备方案：简单的base64编码（仅用于演示，生产环境不推荐）
            self.cipher = None
            self.use_real_encryption = False
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """加密敏感数据"""
        sensitive_patterns = [
            r'password', r'passwd', r'pwd',
            r'secret', r'key', r'token',
            r'api_key', r'access_key', r'secret_key',
            r'private_key', r'credential'
        ]
        
        encrypted_data = data.copy()
        
        def is_sensitive_key_or_value(key, value):
            """检查键名或值是否包含敏感信息"""
            if isinstance(value, str):
                # 检查键名
                for pattern in sensitive_patterns:
                    if re.search(pattern, key, re.IGNORECASE):
                        return True
                # 检查值内容
                for pattern in sensitive_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return True
            return False
        
        def encrypt_value(value):
            return value  # 这个函数不再需要，由encrypt_dict处理
        
        def encrypt_dict(obj, parent_key=""):
            if isinstance(obj, dict):
                return {k: encrypt_dict(v, f"{parent_key}.{k}" if parent_key else k) 
                       for k, v in obj.items()}
            elif isinstance(obj, list):
                return [encrypt_dict(item, parent_key) for item in obj]
            else:
                # 检查是否需要加密
                if is_sensitive_key_or_value(parent_key, obj):
                    try:
                        if self.use_real_encryption:
                            return self.cipher.encrypt(str(obj).encode()).decode()
                        else:
                            # 后备方案：简单的base64编码
                            encoded = base64.b64encode(str(obj).encode()).decode()
                            return f"ENC_B64_{encoded}"
                    except:
                        return obj
                return obj
        
        return encrypt_dict(encrypted_data)
    
    def decrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解密敏感数据"""
        decrypted_data = data.copy()
        
        def is_encrypted_value(value):
            """检查值是否被加密"""
            if isinstance(value, str):
                if self.use_real_encryption and value.startswith('gAAAAA'):
                    return True
                elif value.startswith('ENC_B64_'):
                    return True
            return False
        
        def decrypt_value(value):
            return value  # 这个函数不再需要，由decrypt_dict处理
        
        def decrypt_dict(obj, parent_key=""):
            if isinstance(obj, dict):
                return {k: decrypt_dict(v, f"{parent_key}.{k}" if parent_key else k) 
                       for k, v in obj.items()}
            elif isinstance(obj, list):
                return [decrypt_dict(item, parent_key) for item in obj]
            else:
                # 检查是否需要解密
                if is_encrypted_value(obj):
                    try:
                        if self.use_real_encryption and obj.startswith('gAAAAA'):
                            return self.cipher.decrypt(obj.encode()).decode()
                        elif obj.startswith('ENC_B64_'):
                            # 后备方案：base64解码
                            encoded = obj[8:]  # 移除前缀
                            return base64.b64decode(encoded).decode()
                    except:
                        return obj
                return obj
        
        return decrypt_dict(decrypted_data)


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any], config_type: ConfigType) -> Tuple[bool, List[str]]:
        """验证配置有效性"""
        errors = []
        
        # 基础验证
        if not isinstance(config, dict):
            errors.append("配置必须是字典格式")
            return False, errors
        
        # 类型特定验证
        if config_type == ConfigType.SYSTEM:
            errors.extend(ConfigValidator._validate_system_config(config))
        elif config_type == ConfigType.USER:
            errors.extend(ConfigValidator._validate_user_config(config))
        elif config_type == ConfigType.APPLICATION:
            errors.extend(ConfigValidator._validate_app_config(config))
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_system_config(config: Dict[str, Any]) -> List[str]:
        """验证系统配置"""
        errors = []
        required_fields = ['hostname', 'network', 'security']
        
        for field in required_fields:
            if field not in config:
                errors.append(f"系统配置缺少必需字段: {field}")
        
        return errors
    
    @staticmethod
    def _validate_user_config(config: Dict[str, Any]) -> List[str]:
        """验证用户配置"""
        errors = []
        if 'preferences' not in config:
            errors.append("用户配置缺少preferences字段")
        
        return errors
    
    @staticmethod
    def _validate_app_config(config: Dict[str, Any]) -> List[str]:
        """验证应用配置"""
        errors = []
        required_fields = ['app_name', 'version', 'database']
        
        for field in required_fields:
            if field not in config:
                errors.append(f"应用配置缺少必需字段: {field}")
        
        return errors
    
    @staticmethod
    def calculate_checksum(data: Union[str, Dict[str, Any]]) -> str:
        """计算配置数据校验和"""
        if isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        return hashlib.sha256(content.encode()).hexdigest()


class BackupManager:
    """备份管理器"""
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.protector = SensitiveDataProtector()
        self.validator = ConfigValidator()
        self._load_metadata()
    
    def _load_metadata(self):
        """加载备份元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {'backups': []}
        else:
            self.metadata = {'backups': []}
    
    def _save_metadata(self):
        """保存备份元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def create_backup(self, config_name: str, config_data: Dict[str, Any], 
                     config_type: ConfigType, description: str = "", 
                     tags: List[str] = None) -> BackupRecord:
        """创建备份"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"{config_name}_{timestamp}"
        
        # 验证配置
        is_valid, errors = self.validator.validate_config(config_data, config_type)
        if not is_valid:
            return BackupRecord(
                id=backup_id,
                metadata=ConfigMetadata(
                    name=config_name,
                    config_type=config_type,
                    version="1.0.0",
                    created_at=timestamp,
                    description=description,
                    tags=tags or []
                ),
                backup_path="",
                size=0,
                status=BackupStatus.FAILED,
                error_message=f"配置验证失败: {'; '.join(errors)}"
            )
        
        # 加密敏感数据
        encrypted_data = self.protector.encrypt_sensitive_data(config_data)
        
        # 计算校验和
        checksum = self.validator.calculate_checksum(encrypted_data)
        
        # 创建元数据
        metadata = ConfigMetadata(
            name=config_name,
            config_type=config_type,
            version="1.0.0",
            created_at=timestamp,
            description=description,
            tags=tags or [],
            checksum=checksum
        )
        
        # 保存备份文件
        backup_file = self.backup_dir / f"{backup_id}.json"
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(encrypted_data, f, indent=2, ensure_ascii=False)
            
            size = backup_file.stat().st_size
            
            # 创建备份记录
            record = BackupRecord(
                id=backup_id,
                metadata=metadata,
                backup_path=str(backup_file),
                size=size,
                status=BackupStatus.SUCCESS
            )
            
            # 保存到元数据
            self.metadata['backups'].append(record.to_dict())
            self._save_metadata()
            
            return record
            
        except Exception as e:
            return BackupRecord(
                id=backup_id,
                metadata=metadata,
                backup_path="",
                size=0,
                status=BackupStatus.FAILED,
                error_message=str(e)
            )
    
    def restore_backup(self, backup_id: str) -> Tuple[bool, Dict[str, Any], str]:
        """恢复备份"""
        # 查找备份记录
        backup_record = None
        for record_dict in self.metadata['backups']:
            if record_dict['id'] == backup_id:
                backup_record = record_dict
                break
        
        if not backup_record:
            return False, {}, "备份记录不存在"
        
        try:
            # 读取备份文件
            backup_file = Path(backup_record['backup_path'])
            if not backup_file.exists():
                return False, {}, "备份文件不存在"
            
            with open(backup_file, 'r', encoding='utf-8') as f:
                encrypted_data = json.load(f)
            
            # 解密数据
            decrypted_data = self.protector.decrypt_sensitive_data(encrypted_data)
            
            # 验证完整性
            current_checksum = self.validator.calculate_checksum(encrypted_data)
            if current_checksum != backup_record['metadata']['checksum']:
                return False, {}, "备份文件校验和不匹配，可能已损坏"
            
            return True, decrypted_data, "恢复成功"
            
        except Exception as e:
            return False, {}, f"恢复失败: {str(e)}"
    
    def list_backups(self, config_type: Optional[ConfigType] = None) -> List[Dict[str, Any]]:
        """列出备份"""
        backups = self.metadata['backups']
        
        if config_type:
            backups = [b for b in backups if b['metadata']['config_type'] == config_type.value]
        
        # 按创建时间排序
        backups.sort(key=lambda x: x['metadata']['created_at'], reverse=True)
        
        return backups
    
    def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        backup_record = None
        for i, record_dict in enumerate(self.metadata['backups']):
            if record_dict['id'] == backup_id:
                backup_record = self.metadata['backups'].pop(i)
                break
        
        if not backup_record:
            return False
        
        # 删除备份文件
        backup_file = Path(backup_record['backup_path'])
        if backup_file.exists():
            backup_file.unlink()
        
        self._save_metadata()
        return True
    
    def compare_backups(self, backup_id1: str, backup_id2: str) -> Dict[str, Any]:
        """比较两个备份"""
        success1, data1, _ = self.restore_backup(backup_id1)
        success2, data2, _ = self.restore_backup(backup_id2)
        
        if not success1 or not success2:
            return {"error": "无法读取备份数据"}
        
        # 简单比较
        diff = {
            "added": [],
            "removed": [],
            "modified": []
        }
        
        # 找出新增的键
        all_keys = set(data1.keys()) | set(data2.keys())
        for key in all_keys:
            if key not in data1:
                diff["added"].append(key)
            elif key not in data2:
                diff["removed"].append(key)
            elif data1[key] != data2[key]:
                diff["modified"].append({
                    "key": key,
                    "old_value": data1[key],
                    "new_value": data2[key]
                })
        
        return diff


class ConfigBackup:
    """配置备份器主类"""
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_manager = BackupManager(backup_dir)
        self.validator = ConfigValidator()
        self.protector = SensitiveDataProtector()
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    # 配置备份方法
    def backup_system_config(self, config_name: str, config_data: Dict[str, Any], 
                           description: str = "", tags: List[str] = None) -> BackupRecord:
        """备份系统配置"""
        return self.backup_manager.create_backup(
            config_name, config_data, ConfigType.SYSTEM, description, tags
        )
    
    def backup_user_config(self, config_name: str, config_data: Dict[str, Any], 
                          description: str = "", tags: List[str] = None) -> BackupRecord:
        """备份用户配置"""
        return self.backup_manager.create_backup(
            config_name, config_data, ConfigType.USER, description, tags
        )
    
    def backup_app_config(self, config_name: str, config_data: Dict[str, Any], 
                         description: str = "", tags: List[str] = None) -> BackupRecord:
        """备份应用配置"""
        return self.backup_manager.create_backup(
            config_name, config_data, ConfigType.APPLICATION, description, tags
        )
    
    # 配置恢复方法
    def restore_config(self, backup_id: str) -> Tuple[bool, Dict[str, Any], str]:
        """恢复配置"""
        return self.backup_manager.restore_backup(backup_id)
    
    def quick_restore(self, config_name: str, config_type: ConfigType) -> Tuple[bool, Dict[str, Any], str]:
        """快速恢复最新配置"""
        backups = self.backup_manager.list_backups(config_type)
        config_backups = [b for b in backups if b['metadata']['name'] == config_name]
        
        if not config_backups:
            return False, {}, f"未找到 {config_name} 的备份"
        
        # 返回最新的备份
        latest_backup = config_backups[0]
        return self.restore_config(latest_backup['id'])
    
    # 版本管理方法
    def list_config_versions(self, config_name: str) -> List[Dict[str, Any]]:
        """列出配置的所有版本"""
        all_backups = self.backup_manager.list_backups()
        return [b for b in all_backups if b['metadata']['name'] == config_name]
    
    def compare_config_versions(self, backup_id1: str, backup_id2: str) -> Dict[str, Any]:
        """比较配置版本"""
        return self.backup_manager.compare_backups(backup_id1, backup_id2)
    
    # 批量操作方法
    def batch_backup(self, configs: List[Tuple[str, Dict[str, Any], ConfigType, str]]) -> List[BackupRecord]:
        """批量备份配置"""
        results = []
        for config_name, config_data, config_type, description in configs:
            record = self.backup_manager.create_backup(config_name, config_data, config_type, description)
            results.append(record)
        return results
    
    def batch_restore(self, backup_ids: List[str]) -> List[Tuple[bool, Dict[str, Any], str]]:
        """批量恢复配置"""
        results = []
        for backup_id in backup_ids:
            result = self.restore_config(backup_id)
            results.append(result)
        return results
    
    # 配置模板方法
    def create_template(self, template_name: str, template_data: Dict[str, Any], 
                       description: str = "") -> BackupRecord:
        """创建配置模板"""
        return self.backup_manager.create_backup(
            template_name, template_data, ConfigType.TEMPLATE, description
        )
    
    def apply_template(self, template_name: str, custom_data: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any], str]:
        """应用配置模板"""
        success, template_data, message = self.quick_restore(template_name, ConfigType.TEMPLATE)
        if not success:
            return False, {}, message
        
        if custom_data:
            # 合并自定义数据
            template_data.update(custom_data)
        
        return True, template_data, "模板应用成功"
    
    # 变更追踪方法
    def get_config_history(self, config_name: str) -> List[Dict[str, Any]]:
        """获取配置变更历史"""
        return self.list_config_versions(config_name)
    
    def track_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """追踪配置变更"""
        return self.validator.compare_configs(old_config, new_config)
    
    # 验证方法
    def validate_config(self, config: Dict[str, Any], config_type: ConfigType) -> Tuple[bool, List[str]]:
        """验证配置"""
        return self.validator.validate_config(config, config_type)
    
    # 工具方法
    def list_all_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份"""
        return self.backup_manager.list_backups()
    
    def get_backup_info(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """获取备份信息"""
        backups = self.backup_manager.list_backups()
        for backup in backups:
            if backup['id'] == backup_id:
                return backup
        return None
    
    def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        return self.backup_manager.delete_backup(backup_id)
    
    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """清理旧备份，保留最新N个"""
        all_backups = self.backup_manager.list_backups()
        deleted_count = 0
        
        if len(all_backups) > keep_count:
            # 按创建时间排序，保留最新的
            backups_to_delete = all_backups[keep_count:]
            for backup in backups_to_delete:
                if self.delete_backup(backup['id']):
                    deleted_count += 1
        
        return deleted_count


# 扩展ConfigValidator类，添加缺失的方法
def compare_configs(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    """比较两个配置"""
    diff = {
        "added": [],
        "removed": [],
        "modified": [],
        "unchanged": []
    }
    
    all_keys = set(old_config.keys()) | set(new_config.keys())
    
    for key in all_keys:
        if key not in old_config:
            diff["added"].append(key)
        elif key not in new_config:
            diff["removed"].append(key)
        elif old_config[key] == new_config[key]:
            diff["unchanged"].append(key)
        else:
            diff["modified"].append({
                "key": key,
                "old_value": old_config[key],
                "new_value": new_config[key]
            })
    
    return diff

# 动态添加方法到ConfigValidator类
ConfigValidator.compare_configs = compare_configs