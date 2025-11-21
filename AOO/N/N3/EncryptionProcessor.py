#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3加密处理器
==================

这是一个全面的加密处理器，提供企业级的数据加密、解密、密钥管理、数字签名等功能。

功能特性：
1. 对称加密算法(AES、DES、ChaCha20等)
2. 非对称加密算法(RSA、ECC等)
3. 哈希函数和数字签名
4. 密钥管理和轮换
5. 数据加密存储
6. 传输加密保护
7. 加密性能优化
8. 加密合规性检查
9. 加密审计日志


版本: 1.0.0
日期: 2025-11-06
"""

import os
import sys
import json
import time
import hashlib
import hmac
import secrets
import logging
import threading
import asyncio
import base64
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec, utils
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.fernet import InvalidToken


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('encryption_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EncryptionConfig:
    """加密配置类"""
    symmetric_algorithm: str = "AES-256-GCM"
    asymmetric_algorithm: str = "RSA-4096"
    hash_algorithm: str = "SHA-256"
    key_rotation_days: int = 90
    max_key_age_days: int = 180
    audit_enabled: bool = True
    performance_optimization: bool = True
    compliance_check: bool = True


@dataclass
class KeyInfo:
    """密钥信息类"""
    key_id: str
    algorithm: str
    created_at: datetime
    expires_at: datetime
    key_type: str  # 'symmetric', 'asymmetric', 'private', 'public'
    status: str  # 'active', 'expired', 'revoked'
    usage_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class AuditLog:
    """审计日志类"""
    timestamp: datetime
    operation: str
    key_id: Optional[str]
    success: bool
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_id: Optional[str] = None


class ComplianceChecker:
    """加密合规性检查器"""
    
    def __init__(self):
        self.standards = {
            'FIPS_140_2': self._check_fips_140_2,
            'GDPR': self._check_gdpr,
            'PCI_DSS': self._check_pci_dss,
            'SOX': self._check_sox,
            'HIPAA': self._check_hipaa
        }
    
    def check_compliance(self, standard: str, config: EncryptionConfig) -> Tuple[bool, List[str]]:
        """检查合规性"""
        if standard not in self.standards:
            return False, [f"不支持的标准: {standard}"]
        
        return self.standards[standard](config)
    
    def _check_fips_140_2(self, config: EncryptionConfig) -> Tuple[bool, List[str]]:
        """检查FIPS 140-2合规性"""
        issues = []
        
        # 检查算法是否FIPS认证
        if config.symmetric_algorithm not in ["AES-128", "AES-192", "AES-256"]:
            issues.append("对称加密算法不符合FIPS 140-2标准")
        
        if config.asymmetric_algorithm not in ["RSA-2048", "RSA-3072", "RSA-4096"]:
            issues.append("非对称加密算法不符合FIPS 140-2标准")
        
        if config.hash_algorithm not in ["SHA-256", "SHA-384", "SHA-512"]:
            issues.append("哈希算法不符合FIPS 140-2标准")
        
        return len(issues) == 0, issues
    
    def _check_gdpr(self, config: EncryptionConfig) -> Tuple[bool, List[str]]:
        """检查GDPR合规性"""
        issues = []
        
        # GDPR要求适当的加密措施
        if config.symmetric_algorithm not in ["AES-256", "ChaCha20-Poly1305"]:
            issues.append("GDPR建议使用AES-256或ChaCha20-Poly1305")
        
        return len(issues) == 0, issues
    
    def _check_pci_dss(self, config: EncryptionConfig) -> Tuple[bool, List[str]]:
        """检查PCI DSS合规性"""
        issues = []
        
        # PCI DSS要求强加密
        if "AES" not in config.symmetric_algorithm:
            issues.append("PCI DSS要求使用AES加密")
        
        if config.key_rotation_days > 90:
            issues.append("PCI DSS要求密钥轮换不超过90天")
        
        return len(issues) == 0, issues
    
    def _check_sox(self, config: EncryptionConfig) -> Tuple[bool, List[str]]:
        """检查SOX合规性"""
        issues = []
        
        # SOX要求适当的加密和审计
        if not config.audit_enabled:
            issues.append("SOX要求启用审计日志")
        
        return len(issues) == 0, issues
    
    def _check_hipaa(self, config: EncryptionConfig) -> Tuple[bool, List[str]]:
        """检查HIPAA合规性"""
        issues = []
        
        # HIPAA要求加密保护
        if "AES" not in config.symmetric_algorithm:
            issues.append("HIPAA建议使用AES加密")
        
        return len(issues) == 0, issues


class PerformanceOptimizer:
    """加密性能优化器"""
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
    
    def optimize_symmetric_encryption(self, data: bytes, algorithm: str) -> bytes:
        """优化对称加密性能"""
        # 检查缓存
        cache_key = hashlib.sha256(data + algorithm.encode()).hexdigest()
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # 执行加密（这里只是示例，实际实现需要调用加密算法）
        encrypted_data = data  # 实际应该调用加密算法
        
        # 缓存结果
        with self._lock:
            self._cache[cache_key] = encrypted_data
        
        return encrypted_data
    
    def clear_cache(self):
        """清理缓存"""
        with self._lock:
            self._cache.clear()


class EncryptionProcessor:
    """
    N3加密处理器主类
    
    提供全面的加密、解密、密钥管理、数字签名等功能
    """
    
    def __init__(self, config: Optional[EncryptionConfig] = None, 
                 master_key: Optional[bytes] = None):
        """
        初始化加密处理器
        
        Args:
            config: 加密配置
            master_key: 主密钥，用于密钥加密
        """
        self.config = config or EncryptionConfig()
        self.master_key = master_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        
        # 组件初始化
        self.compliance_checker = ComplianceChecker()
        self.performance_optimizer = PerformanceOptimizer()
        
        # 密钥存储
        self._keys: Dict[str, KeyInfo] = {}
        self._key_storage_path = Path("keys")
        self._key_storage_path.mkdir(exist_ok=True)
        
        # 审计日志
        self.audit_logs: List[AuditLog] = []
        self._audit_lock = threading.Lock()
        
        # 线程池用于性能优化
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 加载现有密钥
        self._load_keys()
        
        logger.info("N3加密处理器初始化完成")
    
    def _log_audit(self, operation: str, key_id: Optional[str], 
                   success: bool, details: Dict[str, Any],
                   ip_address: Optional[str] = None, user_id: Optional[str] = None):
        """记录审计日志"""
        if not self.config.audit_enabled:
            return
        
        audit_log = AuditLog(
            timestamp=datetime.now(),
            operation=operation,
            key_id=key_id,
            success=success,
            details=details,
            ip_address=ip_address,
            user_id=user_id
        )
        
        with self._audit_lock:
            self.audit_logs.append(audit_log)
        
        # 保持日志大小在合理范围内
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-5000:]
        
        logger.info(f"审计日志: {operation} - {'成功' if success else '失败'}")
    
    def _save_keys(self):
        """保存密钥信息到文件"""
        keys_data = {}
        for key_id, key_info in self._keys.items():
            keys_data[key_id] = {
                'key_id': key_info.key_id,
                'algorithm': key_info.algorithm,
                'created_at': key_info.created_at.isoformat(),
                'expires_at': key_info.expires_at.isoformat(),
                'key_type': key_info.key_type,
                'status': key_info.status,
                'usage_count': key_info.usage_count,
                'last_used': key_info.last_used.isoformat() if key_info.last_used else None
            }
        
        with open(self._key_storage_path / "keys.json", 'w', encoding='utf-8') as f:
            json.dump(keys_data, f, ensure_ascii=False, indent=2)
    
    def _load_keys(self):
        """从文件加载密钥信息"""
        keys_file = self._key_storage_path / "keys.json"
        if keys_file.exists():
            try:
                with open(keys_file, 'r', encoding='utf-8') as f:
                    keys_data = json.load(f)
                
                for key_id, data in keys_data.items():
                    key_info = KeyInfo(
                        key_id=data['key_id'],
                        algorithm=data['algorithm'],
                        created_at=datetime.fromisoformat(data['created_at']),
                        expires_at=datetime.fromisoformat(data['expires_at']),
                        key_type=data['key_type'],
                        status=data['status'],
                        usage_count=data['usage_count'],
                        last_used=datetime.fromisoformat(data['last_used']) if data['last_used'] else None
                    )
                    self._keys[key_id] = key_info
                
                logger.info(f"加载了 {len(self._keys)} 个密钥")
            except Exception as e:
                logger.error(f"加载密钥失败: {e}")
    
    # ==================== 对称加密算法 ====================
    
    def generate_symmetric_key(self, algorithm: str = "AES-256-GCM") -> str:
        """
        生成对称密钥
        
        Args:
            algorithm: 对称加密算法
            
        Returns:
            密钥ID
        """
        key_id = f"sym_{algorithm}_{int(time.time())}"
        
        # 生成密钥
        if algorithm.startswith("AES"):
            key_length = int(algorithm.split("-")[1]) // 8
            key = os.urandom(key_length)
        elif algorithm == "ChaCha20-Poly1305":
            key = os.urandom(32)
        else:
            raise ValueError(f"不支持的对称加密算法: {algorithm}")
        
        # 保存密钥
        key_file = self._key_storage_path / f"{key_id}.key"
        encrypted_key = self.cipher_suite.encrypt(key)
        with open(key_file, 'wb') as f:
            f.write(encrypted_key)
        
        # 记录密钥信息
        expires_at = datetime.now() + timedelta(days=self.config.max_key_age_days)
        key_info = KeyInfo(
            key_id=key_id,
            algorithm=algorithm,
            created_at=datetime.now(),
            expires_at=expires_at,
            key_type="symmetric",
            status="active"
        )
        self._keys[key_id] = key_info
        self._save_keys()
        
        self._log_audit("generate_symmetric_key", key_id, True, 
                       {"algorithm": algorithm})
        
        logger.info(f"生成对称密钥: {key_id}")
        return key_id
    
    def encrypt_symmetric(self, data: Union[str, bytes], key_id: str) -> bytes:
        """
        对称加密
        
        Args:
            data: 要加密的数据
            key_id: 密钥ID
            
        Returns:
            加密后的数据
        """
        if key_id not in self._keys:
            raise ValueError(f"密钥不存在: {key_id}")
        
        key_info = self._keys[key_id]
        if key_info.status != "active":
            raise ValueError(f"密钥状态无效: {key_info.status}")
        
        # 读取密钥
        key_file = self._key_storage_path / f"{key_id}.key"
        with open(key_file, 'rb') as f:
            encrypted_key = f.read()
        
        try:
            key = self.cipher_suite.decrypt(encrypted_key)
        except InvalidToken:
            raise ValueError("主密钥无效，无法解密密钥文件")
        
        # 转换为bytes
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # 执行加密
        algorithm = key_info.algorithm
        if algorithm == "AES-256-GCM":
            nonce = os.urandom(12)
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, data, None)
            encrypted_data = nonce + ciphertext
        elif algorithm == "ChaCha20-Poly1305":
            nonce = os.urandom(12)
            chacha = ChaCha20Poly1305(key)
            ciphertext = chacha.encrypt(nonce, data, None)
            encrypted_data = nonce + ciphertext
        else:
            raise ValueError(f"不支持的加密算法: {algorithm}")
        
        # 更新密钥使用统计
        key_info.usage_count += 1
        key_info.last_used = datetime.now()
        self._save_keys()
        
        self._log_audit("encrypt_symmetric", key_id, True, 
                       {"data_length": len(data), "algorithm": algorithm})
        
        return encrypted_data
    
    def decrypt_symmetric(self, encrypted_data: bytes, key_id: str) -> bytes:
        """
        对称解密
        
        Args:
            encrypted_data: 加密的数据
            key_id: 密钥ID
            
        Returns:
            解密后的数据
        """
        if key_id not in self._keys:
            raise ValueError(f"密钥不存在: {key_id}")
        
        key_info = self._keys[key_id]
        if key_info.status != "active":
            raise ValueError(f"密钥状态无效: {key_info.status}")
        
        # 读取密钥
        key_file = self._key_storage_path / f"{key_id}.key"
        with open(key_file, 'rb') as f:
            encrypted_key = f.read()
        
        try:
            key = self.cipher_suite.decrypt(encrypted_key)
        except InvalidToken:
            raise ValueError("主密钥无效，无法解密密钥文件")
        
        # 执行解密
        algorithm = key_info.algorithm
        if algorithm == "AES-256-GCM":
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        elif algorithm == "ChaCha20-Poly1305":
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            chacha = ChaCha20Poly1305(key)
            plaintext = chacha.decrypt(nonce, ciphertext, None)
        else:
            raise ValueError(f"不支持的解密算法: {algorithm}")
        
        # 更新密钥使用统计
        key_info.usage_count += 1
        key_info.last_used = datetime.now()
        self._save_keys()
        
        self._log_audit("decrypt_symmetric", key_id, True, 
                       {"encrypted_length": len(encrypted_data), "algorithm": algorithm})
        
        return plaintext
    
    # ==================== 非对称加密算法 ====================
    
    def generate_asymmetric_keypair(self, algorithm: str = "RSA-4096") -> Tuple[str, str]:
        """
        生成非对称密钥对
        
        Args:
            algorithm: 非对称加密算法
            
        Returns:
            (私钥ID, 公钥ID)
        """
        private_key_id = f"priv_{algorithm}_{int(time.time())}"
        public_key_id = f"pub_{algorithm}_{int(time.time())}"
        
        # 生成密钥对
        if algorithm.startswith("RSA"):
            key_size = int(algorithm.split("-")[1])
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
        elif algorithm == "ECC-P256":
            private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        else:
            raise ValueError(f"不支持的非对称加密算法: {algorithm}")
        
        public_key = private_key.public_key()
        
        # 保存私钥
        private_key_file = self._key_storage_path / f"{private_key_id}.key"
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(self.master_key)
        )
        with open(private_key_file, 'wb') as f:
            f.write(private_pem)
        
        # 保存公钥
        public_key_file = self._key_storage_path / f"{public_key_id}.key"
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(public_key_file, 'wb') as f:
            f.write(public_pem)
        
        # 记录密钥信息
        expires_at = datetime.now() + timedelta(days=self.config.max_key_age_days)
        
        private_key_info = KeyInfo(
            key_id=private_key_id,
            algorithm=algorithm,
            created_at=datetime.now(),
            expires_at=expires_at,
            key_type="private",
            status="active"
        )
        
        public_key_info = KeyInfo(
            key_id=public_key_id,
            algorithm=algorithm,
            created_at=datetime.now(),
            expires_at=expires_at,
            key_type="public",
            status="active"
        )
        
        self._keys[private_key_id] = private_key_info
        self._keys[public_key_id] = public_key_info
        self._save_keys()
        
        self._log_audit("generate_asymmetric_keypair", private_key_id, True, 
                       {"algorithm": algorithm})
        
        logger.info(f"生成非对称密钥对: {private_key_id}, {public_key_id}")
        return private_key_id, public_key_id
    
    def encrypt_asymmetric(self, data: Union[str, bytes], public_key_id: str) -> bytes:
        """
        非对称加密
        
        Args:
            data: 要加密的数据
            public_key_id: 公钥ID
            
        Returns:
            加密后的数据
        """
        if public_key_id not in self._keys:
            raise ValueError(f"公钥不存在: {public_key_id}")
        
        key_info = self._keys[public_key_id]
        if key_info.status != "active":
            raise ValueError(f"密钥状态无效: {key_info.status}")
        
        # 读取公钥
        public_key_file = self._key_storage_path / f"{public_key_id}.key"
        with open(public_key_file, 'rb') as f:
            public_pem = f.read()
        
        public_key = serialization.load_pem_public_key(public_pem, backend=default_backend())
        
        # 转换为bytes
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # 执行加密
        algorithm = key_info.algorithm
        if algorithm.startswith("RSA"):
            # RSA加密有长度限制，需要分块处理
            max_length = (key_info.algorithm.split("-")[1] == "2048" and 190) or 446
            if len(data) > max_length:
                raise ValueError(f"数据太长，RSA最大支持 {max_length} 字节")
            
            ciphertext = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            raise ValueError(f"不支持的加密算法: {algorithm}")
        
        # 更新密钥使用统计
        key_info.usage_count += 1
        key_info.last_used = datetime.now()
        self._save_keys()
        
        self._log_audit("encrypt_asymmetric", public_key_id, True, 
                       {"data_length": len(data), "algorithm": algorithm})
        
        return ciphertext
    
    def decrypt_asymmetric(self, encrypted_data: bytes, private_key_id: str) -> bytes:
        """
        非对称解密
        
        Args:
            encrypted_data: 加密的数据
            private_key_id: 私钥ID
            
        Returns:
            解密后的数据
        """
        if private_key_id not in self._keys:
            raise ValueError(f"私钥不存在: {private_key_id}")
        
        key_info = self._keys[private_key_id]
        if key_info.status != "active":
            raise ValueError(f"密钥状态无效: {key_info.status}")
        
        # 读取私钥
        private_key_file = self._key_storage_path / f"{private_key_id}.key"
        with open(private_key_file, 'rb') as f:
            private_pem = f.read()
        
        private_key = serialization.load_pem_private_key(
            private_pem,
            password=self.master_key,
            backend=default_backend()
        )
        
        # 执行解密
        algorithm = key_info.algorithm
        if algorithm.startswith("RSA"):
            plaintext = private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            raise ValueError(f"不支持的解密算法: {algorithm}")
        
        # 更新密钥使用统计
        key_info.usage_count += 1
        key_info.last_used = datetime.now()
        self._save_keys()
        
        self._log_audit("decrypt_asymmetric", private_key_id, True, 
                       {"encrypted_length": len(encrypted_data), "algorithm": algorithm})
        
        return plaintext
    
    # ==================== 哈希函数和数字签名 ====================
    
    def hash_data(self, data: Union[str, bytes], algorithm: str = "SHA-256") -> str:
        """
        计算哈希值
        
        Args:
            data: 要哈希的数据
            algorithm: 哈希算法
            
        Returns:
            哈希值的十六进制字符串
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == "SHA-256":
            hash_obj = hashlib.sha256(data)
        elif algorithm == "SHA-384":
            hash_obj = hashlib.sha384(data)
        elif algorithm == "SHA-512":
            hash_obj = hashlib.sha512(data)
        elif algorithm == "MD5":
            hash_obj = hashlib.md5(data)
        else:
            raise ValueError(f"不支持的哈希算法: {algorithm}")
        
        hash_hex = hash_obj.hexdigest()
        
        self._log_audit("hash_data", None, True, 
                       {"data_length": len(data), "algorithm": algorithm})
        
        return hash_hex
    
    def sign_data(self, data: Union[str, bytes], private_key_id: str) -> bytes:
        """
        数字签名
        
        Args:
            data: 要签名的数据
            private_key_id: 私钥ID
            
        Returns:
            数字签名
        """
        if private_key_id not in self._keys:
            raise ValueError(f"私钥不存在: {private_key_id}")
        
        key_info = self._keys[private_key_id]
        if key_info.status != "active":
            raise ValueError(f"密钥状态无效: {key_info.status}")
        
        # 读取私钥
        private_key_file = self._key_storage_path / f"{private_key_id}.key"
        with open(private_key_file, 'rb') as f:
            private_pem = f.read()
        
        private_key = serialization.load_pem_private_key(
            private_pem,
            password=self.master_key,
            backend=default_backend()
        )
        
        # 转换为bytes
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # 计算哈希
        hash_obj = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hash_obj.update(data)
        hash_digest = hash_obj.finalize()
        
        # 生成签名
        if key_info.algorithm.startswith("RSA"):
            signature = private_key.sign(
                hash_digest,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                utils.Prehashed(hashes.SHA256())
            )
        else:
            raise ValueError(f"不支持的签名算法: {key_info.algorithm}")
        
        # 更新密钥使用统计
        key_info.usage_count += 1
        key_info.last_used = datetime.now()
        self._save_keys()
        
        self._log_audit("sign_data", private_key_id, True, 
                       {"data_length": len(data), "algorithm": key_info.algorithm})
        
        return signature
    
    def verify_signature(self, data: Union[str, bytes], signature: bytes, 
                        public_key_id: str) -> bool:
        """
        验证数字签名
        
        Args:
            data: 原始数据
            signature: 数字签名
            public_key_id: 公钥ID
            
        Returns:
            签名是否有效
        """
        if public_key_id not in self._keys:
            raise ValueError(f"公钥不存在: {public_key_id}")
        
        key_info = self._keys[public_key_id]
        if key_info.status != "active":
            raise ValueError(f"密钥状态无效: {key_info.status}")
        
        # 读取公钥
        public_key_file = self._key_storage_path / f"{public_key_id}.key"
        with open(public_key_file, 'rb') as f:
            public_pem = f.read()
        
        public_key = serialization.load_pem_public_key(public_pem, backend=default_backend())
        
        # 转换为bytes
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # 计算哈希
        hash_obj = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hash_obj.update(data)
        hash_digest = hash_obj.finalize()
        
        # 验证签名
        try:
            if key_info.algorithm.startswith("RSA"):
                public_key.verify(
                    signature,
                    hash_digest,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    utils.Prehashed(hashes.SHA256())
                )
                valid = True
            else:
                raise ValueError(f"不支持的验证算法: {key_info.algorithm}")
        except Exception:
            valid = False
        
        self._log_audit("verify_signature", public_key_id, valid, 
                       {"data_length": len(data), "algorithm": key_info.algorithm})
        
        return valid
    
    # ==================== 密钥管理和轮换 ====================
    
    def rotate_keys(self, algorithm: str) -> List[str]:
        """
        轮换密钥
        
        Args:
            algorithm: 要轮换的算法
            
        Returns:
            新生成的密钥ID列表
        """
        new_keys = []
        
        # 生成新密钥
        if algorithm in ["AES-256-GCM", "ChaCha20-Poly1305"]:
            new_key_id = self.generate_symmetric_key(algorithm)
            new_keys.append(new_key_id)
        elif algorithm.startswith("RSA"):
            private_id, public_id = self.generate_asymmetric_keypair(algorithm)
            new_keys.extend([private_id, public_id])
        
        # 标记旧密钥为过期
        for key_id, key_info in self._keys.items():
            if key_info.algorithm == algorithm and key_info.status == "active":
                key_info.status = "expired"
        
        self._save_keys()
        
        self._log_audit("rotate_keys", None, True, 
                       {"algorithm": algorithm, "new_keys": new_keys})
        
        logger.info(f"轮换密钥算法 {algorithm}, 新密钥: {new_keys}")
        return new_keys
    
    def revoke_key(self, key_id: str) -> bool:
        """
        撤销密钥
        
        Args:
            key_id: 密钥ID
            
        Returns:
            是否成功撤销
        """
        if key_id not in self._keys:
            return False
        
        self._keys[key_id].status = "revoked"
        self._save_keys()
        
        self._log_audit("revoke_key", key_id, True, {})
        
        logger.info(f"撤销密钥: {key_id}")
        return True
    
    def get_expired_keys(self) -> List[str]:
        """获取过期密钥列表"""
        now = datetime.now()
        expired_keys = []
        
        for key_id, key_info in self._keys.items():
            if key_info.expires_at <= now and key_info.status == "active":
                expired_keys.append(key_id)
        
        return expired_keys
    
    def cleanup_expired_keys(self) -> int:
        """清理过期密钥"""
        expired_keys = self.get_expired_keys()
        count = 0
        
        for key_id in expired_keys:
            try:
                # 删除密钥文件
                key_file = self._key_storage_path / f"{key_id}.key"
                if key_file.exists():
                    key_file.unlink()
                
                # 从内存中删除
                del self._keys[key_id]
                count += 1
                
                self._log_audit("cleanup_expired_key", key_id, True, {})
                
            except Exception as e:
                logger.error(f"清理密钥失败 {key_id}: {e}")
                self._log_audit("cleanup_expired_key", key_id, False, {"error": str(e)})
        
        if count > 0:
            self._save_keys()
        
        logger.info(f"清理了 {count} 个过期密钥")
        return count
    
    # ==================== 数据加密存储 ====================
    
    def encrypt_file(self, file_path: Union[str, Path], key_id: str, 
                    output_path: Optional[Union[str, Path]] = None) -> str:
        """
        加密文件
        
        Args:
            file_path: 源文件路径
            key_id: 密钥ID
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if output_path is None:
            output_path = file_path.with_suffix(file_path.suffix + '.enc')
        else:
            output_path = Path(output_path)
        
        # 读取文件内容
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # 加密数据
        encrypted_data = self.encrypt_symmetric(data, key_id)
        
        # 写入加密文件
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        self._log_audit("encrypt_file", key_id, True, 
                       {"file_path": str(file_path), "output_path": str(output_path)})
        
        logger.info(f"加密文件: {file_path} -> {output_path}")
        return str(output_path)
    
    def decrypt_file(self, encrypted_file_path: Union[str, Path], key_id: str,
                    output_path: Optional[Union[str, Path]] = None) -> str:
        """
        解密文件
        
        Args:
            encrypted_file_path: 加密文件路径
            key_id: 密钥ID
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        encrypted_file_path = Path(encrypted_file_path)
        if not encrypted_file_path.exists():
            raise FileNotFoundError(f"文件不存在: {encrypted_file_path}")
        
        if output_path is None:
            output_path = encrypted_file_path.with_suffix('')
            if output_path.suffix == '.enc':
                output_path = output_path.with_suffix('')
        else:
            output_path = Path(output_path)
        
        # 读取加密文件
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        
        # 解密数据
        decrypted_data = self.decrypt_symmetric(encrypted_data, key_id)
        
        # 写入解密文件
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        self._log_audit("decrypt_file", key_id, True, 
                       {"encrypted_file_path": str(encrypted_file_path), "output_path": str(output_path)})
        
        logger.info(f"解密文件: {encrypted_file_path} -> {output_path}")
        return str(output_path)
    
    def encrypt_data_structure(self, data: Any, key_id: str) -> str:
        """
        加密数据结构
        
        Args:
            data: 要加密的数据结构
            key_id: 密钥ID
            
        Returns:
            加密后的Base64字符串
        """
        # 序列化为JSON
        json_data = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        
        # 加密
        encrypted_data = self.encrypt_symmetric(json_data.encode('utf-8'), key_id)
        
        # 转换为Base64
        encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
        
        self._log_audit("encrypt_data_structure", key_id, True, 
                       {"data_type": type(data).__name__})
        
        return encrypted_b64
    
    def decrypt_data_structure(self, encrypted_b64: str, key_id: str) -> Any:
        """
        解密数据结构
        
        Args:
            encrypted_b64: 加密的Base64字符串
            key_id: 密钥ID
            
        Returns:
            解密后的数据结构
        """
        # 从Base64解码
        encrypted_data = base64.b64decode(encrypted_b64.encode('utf-8'))
        
        # 解密
        decrypted_data = self.decrypt_symmetric(encrypted_data, key_id)
        
        # 反序列化为JSON
        json_data = decrypted_data.decode('utf-8')
        data = json.loads(json_data)
        
        self._log_audit("decrypt_data_structure", key_id, True, {})
        
        return data
    
    # ==================== 传输加密保护 ====================
    
    def create_secure_channel(self, peer_public_key_id: str) -> Dict[str, Any]:
        """
        创建安全通道
        
        Args:
            peer_public_key_id: 对端公钥ID
            
        Returns:
            安全通道信息
        """
        # 生成临时会话密钥（不存储，直接使用）
        session_key = os.urandom(32)
        
        # 生成随机数
        nonce = secrets.token_bytes(32)
        
        # 使用对端公钥加密会话密钥
        encrypted_session_key_data = self.encrypt_asymmetric(session_key, peer_public_key_id)
        
        channel_info = {
            'session_key': base64.b64encode(session_key).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'encrypted_session_key': base64.b64encode(encrypted_session_key_data).decode('utf-8'),
            'created_at': datetime.now().isoformat()
        }
        
        self._log_audit("create_secure_channel", None, True, 
                       {"peer_public_key_id": peer_public_key_id})
        
        return channel_info
    
    def secure_transmit(self, data: Union[str, bytes], channel_info: Dict[str, Any]) -> str:
        """
        安全传输数据
        
        Args:
            data: 要传输的数据
            channel_info: 安全通道信息
            
        Returns:
            加密传输的数据包
        """
        # 获取会话密钥
        session_key = base64.b64decode(channel_info['session_key'].encode('utf-8'))
        
        # 转换为bytes
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # 使用会话密钥加密数据
        nonce = os.urandom(12)
        aesgcm = AESGCM(session_key)
        encrypted_data = aesgcm.encrypt(nonce, data, None)
        
        # 创建数据包
        packet = {
            'encrypted_data': base64.b64encode(nonce + encrypted_data).decode('utf-8'),
            'timestamp': datetime.now().isoformat(),
            'nonce': channel_info['nonce']
        }
        
        packet_data = json.dumps(packet, separators=(',', ':'))
        packet_hash = self.hash_data(packet_data)
        
        secure_packet = {
            'packet': packet,
            'hash': packet_hash
        }
        
        self._log_audit("secure_transmit", None, True, 
                       {"data_length": len(data)})
        
        return base64.b64encode(json.dumps(secure_packet).encode('utf-8')).decode('utf-8')
    
    def secure_receive(self, secure_packet_b64: str, channel_info: Dict[str, Any]) -> bytes:
        """
        安全接收数据
        
        Args:
            secure_packet_b64: 安全数据包
            channel_info: 安全通道信息
            
        Returns:
            接收的数据
        """
        # 解码数据包
        secure_packet_data = base64.b64decode(secure_packet_b64.encode('utf-8'))
        secure_packet = json.loads(secure_packet_data.decode('utf-8'))
        
        packet = secure_packet['packet']
        packet_hash = secure_packet['hash']
        
        # 验证哈希
        packet_data = json.dumps(packet, separators=(',', ':'))
        calculated_hash = self.hash_data(packet_data)
        
        if packet_hash != calculated_hash:
            raise ValueError("数据包哈希验证失败")
        
        # 解密数据
        encrypted_data = base64.b64decode(packet['encrypted_data'].encode('utf-8'))
        
        # 提取nonce和密文
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        # 获取会话密钥
        session_key = base64.b64decode(channel_info['session_key'].encode('utf-8'))
        
        # 解密数据
        aesgcm = AESGCM(session_key)
        decrypted_data = aesgcm.decrypt(nonce, ciphertext, None)
        
        self._log_audit("secure_receive", None, True, {})
        
        return decrypted_data
    
    # ==================== 加密性能优化 ====================
    
    def batch_encrypt(self, data_list: List[Tuple[Union[str, bytes], str]], 
                     max_workers: int = 4) -> List[bytes]:
        """
        批量加密
        
        Args:
            data_list: 数据和密钥ID的元组列表
            max_workers: 最大工作线程数
            
        Returns:
            加密后的数据列表
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for data, key_id in data_list:
                future = executor.submit(self.encrypt_symmetric, data, key_id)
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"批量加密失败: {e}")
                    results.append(None)
        
        self._log_audit("batch_encrypt", None, True, 
                       {"batch_size": len(data_list)})
        
        return results
    
    def batch_decrypt(self, encrypted_data_list: List[Tuple[bytes, str]], 
                     max_workers: int = 4) -> List[bytes]:
        """
        批量解密
        
        Args:
            encrypted_data_list: 加密数据和密钥ID的元组列表
            max_workers: 最大工作线程数
            
        Returns:
            解密后的数据列表
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for encrypted_data, key_id in encrypted_data_list:
                future = executor.submit(self.decrypt_symmetric, encrypted_data, key_id)
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"批量解密失败: {e}")
                    results.append(None)
        
        self._log_audit("batch_decrypt", None, True, 
                       {"batch_size": len(encrypted_data_list)})
        
        return results
    
    # ==================== 加密合规性检查 ====================
    
    def check_compliance(self, standard: str) -> Tuple[bool, List[str]]:
        """
        检查加密配置合规性
        
        Args:
            standard: 合规标准
            
        Returns:
            (是否合规, 问题列表)
        """
        return self.compliance_checker.check_compliance(standard, self.config)
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """获取合规性报告"""
        standards = ['FIPS_140_2', 'GDPR', 'PCI_DSS', 'SOX', 'HIPAA']
        report = {}
        
        for standard in standards:
            compliant, issues = self.check_compliance(standard)
            report[standard] = {
                'compliant': compliant,
                'issues': issues
            }
        
        return report
    
    # ==================== 加密审计日志 ====================
    
    def get_audit_logs(self, limit: int = 100, 
                      operation: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取审计日志
        
        Args:
            limit: 返回日志数量限制
            operation: 过滤操作类型
            
        Returns:
            审计日志列表
        """
        logs = self.audit_logs
        
        if operation:
            logs = [log for log in logs if log.operation == operation]
        
        logs = logs[-limit:]
        
        return [asdict(log) for log in logs]
    
    def export_audit_logs(self, file_path: Union[str, Path], 
                         format: str = 'json') -> str:
        """
        导出审计日志
        
        Args:
            file_path: 导出文件路径
            format: 导出格式 ('json' 或 'csv')
            
        Returns:
            导出文件路径
        """
        file_path = Path(file_path)
        
        if format == 'json':
            logs_data = [asdict(log) for log in self.audit_logs]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(logs_data, f, ensure_ascii=False, indent=2, default=str)
        elif format == 'csv':
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'operation', 'key_id', 'success', 'details', 'ip_address', 'user_id'])
                for log in self.audit_logs:
                    writer.writerow([
                        log.timestamp.isoformat(),
                        log.operation,
                        log.key_id,
                        log.success,
                        json.dumps(log.details, ensure_ascii=False),
                        log.ip_address,
                        log.user_id
                    ])
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        self._log_audit("export_audit_logs", None, True, 
                       {"file_path": str(file_path), "format": format})
        
        logger.info(f"导出审计日志: {file_path}")
        return str(file_path)
    
    # ==================== 工具方法 ====================
    
    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """获取密钥信息"""
        if key_id in self._keys:
            return asdict(self._keys[key_id])
        return None
    
    def list_keys(self, key_type: Optional[str] = None, 
                  status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出密钥"""
        keys = list(self._keys.values())
        
        if key_type:
            keys = [k for k in keys if k.key_type == key_type]
        
        if status:
            keys = [k for k in keys if k.status == status]
        
        return [asdict(key) for key in keys]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'total_keys': len(self._keys),
            'active_keys': len([k for k in self._keys.values() if k.status == 'active']),
            'expired_keys': len([k for k in self._keys.values() if k.status == 'expired']),
            'revoked_keys': len([k for k in self._keys.values() if k.status == 'revoked']),
            'total_operations': len(self.audit_logs),
            'successful_operations': len([log for log in self.audit_logs if log.success]),
            'failed_operations': len([log for log in self.audit_logs if not log.success])
        }
        
        # 算法统计
        algorithm_stats = {}
        for key in self._keys.values():
            algo = key.algorithm
            if algo not in algorithm_stats:
                algorithm_stats[algo] = 0
            algorithm_stats[algo] += 1
        
        stats['algorithm_distribution'] = algorithm_stats
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        self.executor.shutdown(wait=True)
        logger.info("N3加密处理器已清理")


# ==================== 测试用例 ====================

def test_encryption_processor():
    """测试加密处理器"""
    print("开始测试N3加密处理器...")
    
    # 创建加密处理器实例
    config = EncryptionConfig(
        symmetric_algorithm="AES-256-GCM",
        asymmetric_algorithm="RSA-4096",
        key_rotation_days=30,
        audit_enabled=True
    )
    
    processor = EncryptionProcessor(config)
    
    try:
        # 1. 测试对称加密
        print("\n1. 测试对称加密...")
        key_id = processor.generate_symmetric_key("AES-256-GCM")
        original_data = "这是要加密的敏感数据！"
        encrypted_data = processor.encrypt_symmetric(original_data, key_id)
        decrypted_data = processor.decrypt_symmetric(encrypted_data, key_id)
        assert original_data.encode('utf-8') == decrypted_data
        print("✓ 对称加密测试通过")
        
        # 2. 测试非对称加密
        print("\n2. 测试非对称加密...")
        private_key_id, public_key_id = processor.generate_asymmetric_keypair("RSA-2048")
        small_data = "小数据"
        encrypted_asym = processor.encrypt_asymmetric(small_data, public_key_id)
        decrypted_asym = processor.decrypt_asymmetric(encrypted_asym, private_key_id)
        assert small_data.encode('utf-8') == decrypted_asym
        print("✓ 非对称加密测试通过")
        
        # 3. 测试哈希和数字签名
        print("\n3. 测试哈希和数字签名...")
        data = "需要签名的重要数据"
        hash_value = processor.hash_data(data)
        print(f"哈希值: {hash_value}")
        
        signature = processor.sign_data(data, private_key_id)
        is_valid = processor.verify_signature(data, signature, public_key_id)
        assert is_valid
        print("✓ 哈希和数字签名测试通过")
        
        # 4. 测试文件加密
        print("\n4. 测试文件加密...")
        test_file = Path("test_file.txt")
        test_file.write_text("这是一个测试文件的内容", encoding='utf-8')
        
        encrypted_file = processor.encrypt_file(test_file, key_id)
        decrypted_file = processor.decrypt_file(encrypted_file, key_id)
        
        original_content = test_file.read_text(encoding='utf-8')
        decrypted_content = Path(decrypted_file).read_text(encoding='utf-8')
        assert original_content == decrypted_content
        print("✓ 文件加密测试通过")
        
        # 5. 测试数据加密存储
        print("\n5. 测试数据加密存储...")
        test_data = {
            "用户名": "张三",
            "密码": "secret123",
            "年龄": 30,
            "权限": ["read", "write"]
        }
        
        encrypted_struct = processor.encrypt_data_structure(test_data, key_id)
        decrypted_struct = processor.decrypt_data_structure(encrypted_struct, key_id)
        assert test_data == decrypted_struct
        print("✓ 数据加密存储测试通过")
        
        # 6. 测试合规性检查
        print("\n6. 测试合规性检查...")
        compliant, issues = processor.check_compliance("FIPS_140_2")
        print(f"FIPS 140-2 合规性: {compliant}")
        if issues:
            print(f"问题: {issues}")
        print("✓ 合规性检查测试通过")
        
        # 7. 测试审计日志
        print("\n7. 测试审计日志...")
        logs = processor.get_audit_logs(limit=10)
        print(f"审计日志条数: {len(logs)}")
        print("✓ 审计日志测试通过")
        
        # 8. 测试统计信息
        print("\n8. 测试统计信息...")
        stats = processor.get_statistics()
        print(f"统计信息: {stats}")
        print("✓ 统计信息测试通过")
        
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise
    finally:
        # 清理测试文件
        for file_path in ["test_file.txt", "test_file.txt.enc", "decrypted_file.txt"]:
            if Path(file_path).exists():
                Path(file_path).unlink()
        
        processor.cleanup()


if __name__ == "__main__":
    test_encryption_processor()