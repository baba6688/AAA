"""
N1身份验证器

一个完整的身份验证解决方案，提供企业级的安全认证功能。
包含用户管理、多因子认证、JWT令牌、安全策略等核心功能。

主要功能：
- 用户注册和登录
- 多因子认证(MFA)
- JWT令牌管理
- 密码安全策略
- 会话管理
- OAuth集成
- 生物识别支持
- 身份验证日志
- 安全事件处理


版本: 1.0.0
日期: 2025-11-06
"""

import hashlib
import hmac
import secrets
import json
import logging
import sqlite3
import time
import base64
import qrcode
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import jwt
import bcrypt
from cryptography.fernet import Fernet
import pyotp
import phonenumbers
from phonenumbers import parse, is_valid_number


class AuthLevel(Enum):
    """认证级别枚举"""
    BASIC = "basic"           # 基础认证
    STRONG = "strong"         # 强认证
    ENTERPRISE = "enterprise" # 企业级认证


class SecurityEventType(Enum):
    """安全事件类型"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_REVOKED = "token_revoked"


@dataclass
class User:
    """用户数据模型"""
    user_id: str
    username: str
    email: str
    phone: Optional[str] = None
    password_hash: Optional[str] = None
    salt: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    failed_login_attempts: int = 0
    last_login: Optional[datetime] = None
    account_locked_until: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    biometric_data: Optional[str] = None
    roles: List[str] = None
    permissions: List[str] = None
    oauth_providers: Dict[str, str] = None  # provider -> provider_user_id
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.roles is None:
            self.roles = ["user"]
        if self.permissions is None:
            self.permissions = []
        if self.oauth_providers is None:
            self.oauth_providers = {}


@dataclass
class SecurityEvent:
    """安全事件数据模型"""
    event_id: str
    user_id: str
    event_type: SecurityEventType
    timestamp: datetime
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    severity: str = "info"  # info, warning, error, critical
    
    def __post_init__(self):
        if self.event_id is None:
            self.event_id = secrets.token_hex(16)


@dataclass
class Session:
    """会话数据模型"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    last_activity: datetime = None
    
    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = self.created_at


class PasswordPolicy:
    """密码安全策略"""
    
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL_CHARS = True
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    @classmethod
    def validate_password(cls, password: str) -> Tuple[bool, List[str]]:
        """
        验证密码是否符合安全策略
        
        Args:
            password: 要验证的密码
            
        Returns:
            Tuple[bool, List[str]]: (是否符合策略, 错误信息列表)
        """
        errors = []
        
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"密码长度至少需要{cls.MIN_LENGTH}个字符")
        elif len(password) > cls.MAX_LENGTH:
            errors.append(f"密码长度不能超过{cls.MAX_LENGTH}个字符")
        
        if cls.REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("密码必须包含至少一个大写字母")
        
        if cls.REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            errors.append("密码必须包含至少一个小写字母")
        
        if cls.REQUIRE_DIGITS and not any(c.isdigit() for c in password):
            errors.append("密码必须包含至少一个数字")
        
        if cls.REQUIRE_SPECIAL_CHARS and not any(c in cls.SPECIAL_CHARS for c in password):
            errors.append(f"密码必须包含至少一个特殊字符: {cls.SPECIAL_CHARS}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def check_password_strength(cls, password: str) -> int:
        """
        计算密码强度分数 (0-100)
        
        Args:
            password: 要评估的密码
            
        Returns:
            int: 密码强度分数
        """
        score = 0
        length = len(password)
        
        # 长度评分
        if length >= 12:
            score += 25
        elif length >= 8:
            score += 15
        
        # 字符类型评分
        if any(c.isupper() for c in password):
            score += 10
        if any(c.islower() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(c in cls.SPECIAL_CHARS for c in password):
            score += 15
        
        # 唯一性评分
        unique_chars = len(set(password))
        score += min(20, unique_chars)
        
        # 模式检测
        if not cls._has_common_patterns(password):
            score += 10
        
        return min(100, score)
    
    @classmethod
    def _has_common_patterns(cls, password: str) -> bool:
        """检测是否包含常见密码模式"""
        common_patterns = [
            "123", "abc", "password", "qwerty", "111", "000",
            "123456", "654321", "admin", "test", "user"
        ]
        password_lower = password.lower()
        return any(pattern in password_lower for pattern in common_patterns)


class BiometricAuthenticator:
    """生物识别认证器"""
    
    def __init__(self):
        self.supported_methods = ["fingerprint", "face", "voice"]
    
    def generate_biometric_template(self, biometric_data: str) -> str:
        """
        生成生物识别模板
        
        Args:
            biometric_data: 原始生物识别数据
            
        Returns:
            str: 处理后的生物识别模板
        """
        # 使用SHA-256生成生物识别模板
        template_hash = hashlib.sha256(biometric_data.encode()).hexdigest()
        return base64.b64encode(template_hash.encode()).decode()
    
    def verify_biometric(self, template: str, biometric_data: str) -> bool:
        """
        验证生物识别数据
        
        Args:
            template: 存储的生物识别模板
            biometric_data: 新的生物识别数据
            
        Returns:
            bool: 验证结果
        """
        new_template = self.generate_biometric_template(biometric_data)
        return hmac.compare_digest(template, new_template)


class OAuthProvider(Enum):
    """支持的OAuth提供商"""
    GOOGLE = "google"
    GITHUB = "github"
    FACEBOOK = "facebook"
    MICROSOFT = "microsoft"
    APPLE = "apple"


class Authenticator:
    """
    N1身份验证器主类
    
    提供完整的身份验证解决方案，包括用户管理、多因子认证、
    JWT令牌管理、会话管理等企业级安全功能。
    """
    
    def __init__(self, db_path: str = "authenticator.db", jwt_secret: str = None):
        """
        初始化身份验证器
        
        Args:
            db_path: 数据库文件路径
            jwt_secret: JWT签名密钥，如果为None则自动生成
        """
        self.db_path = db_path
        self.jwt_secret = jwt_secret or secrets.token_hex(32)
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # 初始化组件
        self.biometric_auth = BiometricAuthenticator()
        self._setup_logging()
        self._init_database()
        
        # 安全配置
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_timeout = timedelta(hours=24)
        self.token_expiry = timedelta(hours=1)
        self.refresh_token_expiry = timedelta(days=30)
    
    def _setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('authenticator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    password_hash TEXT,
                    salt TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    is_verified BOOLEAN DEFAULT 0,
                    failed_login_attempts INTEGER DEFAULT 0,
                    last_login TIMESTAMP,
                    account_locked_until TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mfa_enabled BOOLEAN DEFAULT 0,
                    mfa_secret TEXT,
                    biometric_data TEXT,
                    roles TEXT,
                    permissions TEXT,
                    oauth_providers TEXT
                )
            ''')
            
            # 创建会话表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # 创建安全事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    event_type TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    details TEXT,
                    severity TEXT DEFAULT 'info',
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # 创建令牌表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tokens (
                    token_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_type TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_revoked BOOLEAN DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
    
    def _encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def _decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def _hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """
        安全地哈希密码
        
        Args:
            password: 原始密码
            salt: 盐值，如果为None则自动生成
            
        Returns:
            Tuple[str, str]: (哈希密码, 盐值)
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # 使用bcrypt进行密码哈希
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        return password_hash, salt
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _generate_token(self, user_id: str, token_type: str = "access") -> str:
        """
        生成JWT令牌
        
        Args:
            user_id: 用户ID
            token_type: 令牌类型 (access/refresh)
            
        Returns:
            str: JWT令牌
        """
        now = datetime.utcnow()
        
        if token_type == "access":
            exp = now + self.token_expiry
        else:
            exp = now + self.refresh_token_expiry
        
        payload = {
            "user_id": user_id,
            "type": token_type,
            "iat": now,
            "exp": exp
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        验证JWT令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            Optional[Dict[str, Any]]: 令牌载荷，如果无效则返回None
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("令牌已过期")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("无效令牌")
            return None
    
    def _log_security_event(self, user_id: str, event_type: SecurityEventType, 
                          ip_address: str, user_agent: str, details: Dict[str, Any] = None,
                          severity: str = "info"):
        """记录安全事件"""
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            user_id=user_id,
            event_type=event_type,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            severity=severity
        )
        
        # 保存到数据库
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO security_events 
                (event_id, user_id, event_type, timestamp, ip_address, user_agent, details, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.user_id,
                event.event_type.value,
                event.timestamp,
                event.ip_address,
                event.user_agent,
                json.dumps(event.details),
                event.severity
            ))
            conn.commit()
        
        # 记录日志
        self.logger.info(f"安全事件: {event_type.value} - 用户: {user_id} - 详情: {details}")
    
    def register_user(self, username: str, email: str, password: str, 
                     phone: str = None, ip_address: str = "127.0.0.1", 
                     user_agent: str = "Unknown") -> Dict[str, Any]:
        """
        用户注册
        
        Args:
            username: 用户名
            email: 邮箱
            password: 密码
            phone: 手机号
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            Dict[str, Any]: 注册结果
        """
        try:
            # 验证密码策略
            is_valid, errors = PasswordPolicy.validate_password(password)
            if not is_valid:
                return {
                    "success": False,
                    "error": "密码不符合安全策略",
                    "details": errors
                }
            
            # 检查用户名和邮箱是否已存在
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT user_id FROM users WHERE username = ? OR email = ?', 
                             (username, email))
                if cursor.fetchone():
                    return {
                        "success": False,
                        "error": "用户名或邮箱已存在"
                    }
            
            # 创建用户
            user_id = secrets.token_hex(16)
            password_hash, salt = self._hash_password(password)
            
            # 验证手机号
            if phone:
                try:
                    # 尝试解析手机号，先尝试国际格式，再尝试默认地区
                    try:
                        parsed_phone = parse(phone, None)
                    except:
                        # 如果解析失败，尝试添加默认地区代码
                        if not phone.startswith('+'):
                            phone_with_country = f"+1{phone}"  # 默认为美国
                            parsed_phone = parse(phone_with_country, None)
                        else:
                            raise ValueError("无法解析手机号")
                    
                    if not is_valid_number(parsed_phone):
                        return {
                            "success": False,
                            "error": "无效的手机号格式"
                        }
                    phone = format(parsed_phone, 'E.164')
                except Exception as e:
                    # 手机号验证失败时，记录警告但不阻止注册
                    self.logger.warning(f"手机号验证失败: {phone}, 错误: {str(e)}")
                    phone = None  # 将手机号设为None，继续注册流程
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                phone=phone,
                password_hash=password_hash,
                salt=salt
            )
            
            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users 
                    (user_id, username, email, phone, password_hash, salt, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user.user_id, user.username, user.email, user.phone,
                    user.password_hash, user.salt, user.created_at, user.updated_at
                ))
                conn.commit()
            
            # 记录安全事件
            self._log_security_event(
                user_id, SecurityEventType.LOGIN_SUCCESS, ip_address, user_agent,
                {"action": "user_registered", "username": username, "email": email}
            )
            
            return {
                "success": True,
                "message": "用户注册成功",
                "user_id": user_id,
                "username": username,
                "email": email
            }
            
        except Exception as e:
            self.logger.error(f"用户注册失败: {str(e)}")
            return {
                "success": False,
                "error": f"注册失败: {str(e)}"
            }
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = "127.0.0.1", 
                         user_agent: str = "Unknown",
                         require_mfa: bool = False) -> Dict[str, Any]:
        """
        用户认证
        
        Args:
            username: 用户名或邮箱
            password: 密码
            ip_address: IP地址
            user_agent: 用户代理
            require_mfa: 是否需要MFA验证
            
        Returns:
            Dict[str, Any]: 认证结果
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, username, email, password_hash, salt, is_active, 
                           failed_login_attempts, account_locked_until, mfa_enabled, mfa_secret
                    FROM users WHERE username = ? OR email = ?
                ''', (username, username))
                user_row = cursor.fetchone()
                
                if not user_row:
                    self._log_security_event(
                        "unknown", SecurityEventType.LOGIN_FAILURE, ip_address, user_agent,
                        {"action": "invalid_username", "username": username}
                    )
                    return {
                        "success": False,
                        "error": "用户名或密码错误",
                        "requires_mfa": False
                    }
                
                (user_id, db_username, email, password_hash, salt, is_active,
                 failed_attempts, locked_until, mfa_enabled, mfa_secret) = user_row
                
                # 检查账户是否被锁定
                if locked_until:
                    lock_time = datetime.fromisoformat(locked_until)
                    if datetime.now() < lock_time:
                        remaining_time = (lock_time - datetime.now()).total_seconds() / 60
                        return {
                            "success": False,
                            "error": f"账户已被锁定，剩余时间: {remaining_time:.1f}分钟",
                            "requires_mfa": False
                        }
                
                # 检查账户是否激活
                if not is_active:
                    self._log_security_event(
                        user_id, SecurityEventType.LOGIN_FAILURE, ip_address, user_agent,
                        {"action": "account_inactive", "username": username}
                    )
                    return {
                        "success": False,
                        "error": "账户已被禁用",
                        "requires_mfa": False
                    }
                
                # 验证密码
                if not self._verify_password(password, password_hash):
                    # 增加失败尝试次数
                    new_failed_attempts = failed_attempts + 1
                    lock_user = False
                    
                    if new_failed_attempts >= self.max_failed_attempts:
                        lock_user = True
                        lock_until = datetime.now() + self.lockout_duration
                    
                    cursor.execute('''
                        UPDATE users 
                        SET failed_login_attempts = ?, account_locked_until = ?
                        WHERE user_id = ?
                    ''', (new_failed_attempts, lock_until.isoformat() if lock_user else None, user_id))
                    conn.commit()
                    
                    # 记录安全事件
                    self._log_security_event(
                        user_id, SecurityEventType.LOGIN_FAILURE, ip_address, user_agent,
                        {
                            "action": "invalid_password", 
                            "failed_attempts": new_failed_attempts,
                            "account_locked": lock_user
                        },
                        "warning" if lock_user else "info"
                    )
                    
                    return {
                        "success": False,
                        "error": "用户名或密码错误",
                        "requires_mfa": False,
                        "failed_attempts": new_failed_attempts,
                        "account_locked": lock_user
                    }
                
                # 密码验证成功，重置失败尝试次数
                cursor.execute('''
                    UPDATE users 
                    SET failed_login_attempts = 0, account_locked_until = NULL, last_login = ?
                    WHERE user_id = ?
                ''', (datetime.now().isoformat(), user_id))
                conn.commit()
                
                # 检查MFA
                if mfa_enabled and require_mfa:
                    # 生成MFA令牌
                    mfa_token = secrets.token_hex(32)
                    cursor.execute('''
                        INSERT INTO tokens (token_id, user_id, token_type, token_hash, expires_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        mfa_token, user_id, "mfa",
                        hashlib.sha256(mfa_token.encode()).hexdigest(),
                        (datetime.now() + timedelta(minutes=5)).isoformat()
                    ))
                    conn.commit()
                    
                    return {
                        "success": False,
                        "error": "需要MFA验证",
                        "requires_mfa": True,
                        "mfa_token": mfa_token
                    }
                
                # 创建会话
                session_id = secrets.token_hex(32)
                session = Session(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + self.session_timeout,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                cursor.execute('''
                    INSERT INTO sessions 
                    (session_id, user_id, created_at, expires_at, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id, session.user_id, session.created_at.isoformat(),
                    session.expires_at.isoformat(), session.ip_address, session.user_agent
                ))
                conn.commit()
                
                # 生成访问令牌和刷新令牌
                access_token = self._generate_token(user_id, "access")
                refresh_token = self._generate_token(user_id, "refresh")
                
                # 记录成功登录
                self._log_security_event(
                    user_id, SecurityEventType.LOGIN_SUCCESS, ip_address, user_agent,
                    {"action": "login_success", "username": username}
                )
                
                return {
                    "success": True,
                    "message": "登录成功",
                    "session_id": session_id,
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "user_id": user_id,
                    "username": db_username,
                    "email": email,
                    "requires_mfa": False
                }
                
        except Exception as e:
            self.logger.error(f"用户认证失败: {str(e)}")
            return {
                "success": False,
                "error": f"认证失败: {str(e)}",
                "requires_mfa": False
            }
    
    def verify_mfa(self, mfa_token: str, mfa_code: str) -> Dict[str, Any]:
        """
        验证MFA代码
        
        Args:
            mfa_token: MFA令牌
            mfa_code: MFA代码
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 验证MFA令牌
                mfa_token_hash = hashlib.sha256(mfa_token.encode()).hexdigest()
                cursor.execute('''
                    SELECT user_id, expires_at FROM tokens 
                    WHERE token_id = ? AND token_type = 'mfa' AND is_revoked = 0
                ''', (mfa_token_hash,))
                token_row = cursor.fetchone()
                
                if not token_row:
                    return {
                        "success": False,
                        "error": "无效的MFA令牌"
                    }
                
                user_id, expires_at = token_row
                if datetime.now() > datetime.fromisoformat(expires_at):
                    return {
                        "success": False,
                        "error": "MFA令牌已过期"
                    }
                
                # 获取用户的MFA密钥
                cursor.execute('SELECT mfa_secret FROM users WHERE user_id = ?', (user_id,))
                mfa_secret = cursor.fetchone()[0]
                
                if not mfa_secret:
                    return {
                        "success": False,
                        "error": "用户未设置MFA"
                    }
                
                # 验证MFA代码
                totp = pyotp.TOTP(mfa_secret)
                if not totp.verify(mfa_code, valid_window=1):
                    return {
                        "success": False,
                        "error": "MFA代码无效"
                    }
                
                # 标记MFA令牌为已使用
                cursor.execute('UPDATE tokens SET is_revoked = 1 WHERE token_id = ?', (mfa_token_hash,))
                
                # 生成访问令牌和刷新令牌
                access_token = self._generate_token(user_id, "access")
                refresh_token = self._generate_token(user_id, "refresh")
                
                return {
                    "success": True,
                    "message": "MFA验证成功",
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "user_id": user_id
                }
                
        except Exception as e:
            self.logger.error(f"MFA验证失败: {str(e)}")
            return {
                "success": False,
                "error": f"MFA验证失败: {str(e)}"
            }
    
    def setup_mfa(self, user_id: str) -> Dict[str, Any]:
        """
        为用户设置MFA
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 设置结果
        """
        try:
            # 生成MFA密钥
            mfa_secret = pyotp.random_base32()
            
            # 生成TOTP URI
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT username, email FROM users WHERE user_id = ?', (user_id,))
                user_row = cursor.fetchone()
                
                if not user_row:
                    return {
                        "success": False,
                        "error": "用户不存在"
                    }
                
                username, email = user_row
                
                # 更新用户MFA设置
                cursor.execute('''
                    UPDATE users 
                    SET mfa_enabled = 1, mfa_secret = ?, updated_at = ?
                    WHERE user_id = ?
                ''', (mfa_secret, datetime.now().isoformat(), user_id))
                conn.commit()
            
            # 生成QR码
            totp_uri = pyotp.totp.TOTP(mfa_secret).provisioning_uri(
                name=email,
                issuer_name="N1 Authenticator"
            )
            
            qr = qrcode.QRCode(version=1, box_size=10, border=4)
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            qr_image = qr.make_image(fill_color="black", back_color="white")
            qr_image_path = f"mfa_qr_{user_id}.png"
            qr_image.save(qr_image_path)
            
            return {
                "success": True,
                "message": "MFA设置成功",
                "mfa_secret": mfa_secret,
                "qr_code_path": qr_image_path,
                "otpauth_url": totp_uri
            }
            
        except Exception as e:
            self.logger.error(f"MFA设置失败: {str(e)}")
            return {
                "success": False,
                "error": f"MFA设置失败: {str(e)}"
            }
    
    def change_password(self, user_id: str, old_password: str, new_password: str,
                       ip_address: str = "127.0.0.1", user_agent: str = "Unknown") -> Dict[str, Any]:
        """
        修改密码
        
        Args:
            user_id: 用户ID
            old_password: 旧密码
            new_password: 新密码
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            Dict[str, Any]: 修改结果
        """
        try:
            # 验证新密码策略
            is_valid, errors = PasswordPolicy.validate_password(new_password)
            if not is_valid:
                return {
                    "success": False,
                    "error": "新密码不符合安全策略",
                    "details": errors
                }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT password_hash FROM users WHERE user_id = ?', (user_id,))
                user_row = cursor.fetchone()
                
                if not user_row:
                    return {
                        "success": False,
                        "error": "用户不存在"
                    }
                
                stored_password_hash = user_row[0]
                
                # 验证旧密码
                if not self._verify_password(old_password, stored_password_hash):
                    self._log_security_event(
                        user_id, SecurityEventType.SUSPICIOUS_ACTIVITY, ip_address, user_agent,
                        {"action": "password_change_failed", "reason": "invalid_old_password"}
                    )
                    return {
                        "success": False,
                        "error": "旧密码错误"
                    }
                
                # 检查新密码是否与旧密码相同
                new_password_hash, _ = self._hash_password(new_password)
                if hmac.compare_digest(stored_password_hash, new_password_hash):
                    return {
                        "success": False,
                        "error": "新密码不能与旧密码相同"
                    }
                
                # 更新密码
                cursor.execute('''
                    UPDATE users 
                    SET password_hash = ?, updated_at = ?
                    WHERE user_id = ?
                ''', (new_password_hash, datetime.now().isoformat(), user_id))
                conn.commit()
                
                # 记录安全事件
                self._log_security_event(
                    user_id, SecurityEventType.PASSWORD_CHANGE, ip_address, user_agent,
                    {"action": "password_changed"}
                )
                
                return {
                    "success": True,
                    "message": "密码修改成功"
                }
                
        except Exception as e:
            self.logger.error(f"密码修改失败: {str(e)}")
            return {
                "success": False,
                "error": f"密码修改失败: {str(e)}"
            }
    
    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        刷新访问令牌
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            Dict[str, Any]: 刷新结果
        """
        try:
            # 验证刷新令牌
            payload = self._verify_token(refresh_token)
            if not payload or payload.get("type") != "refresh":
                return {
                    "success": False,
                    "error": "无效的刷新令牌"
                }
            
            user_id = payload.get("user_id")
            
            # 生成新的访问令牌
            new_access_token = self._generate_token(user_id, "access")
            
            # 记录安全事件
            self._log_security_event(
                user_id, SecurityEventType.TOKEN_REFRESHED, "127.0.0.1", "Unknown",
                {"action": "token_refreshed"}
            )
            
            return {
                "success": True,
                "access_token": new_access_token,
                "user_id": user_id
            }
            
        except Exception as e:
            self.logger.error(f"令牌刷新失败: {str(e)}")
            return {
                "success": False,
                "error": f"令牌刷新失败: {str(e)}"
            }
    
    def logout(self, user_id: str, session_id: str = None, token: str = None,
              ip_address: str = "127.0.0.1", user_agent: str = "Unknown") -> Dict[str, Any]:
        """
        用户登出
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            token: 访问令牌
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            Dict[str, Any]: 登出结果
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 禁用会话
                if session_id:
                    cursor.execute('UPDATE sessions SET is_active = 0 WHERE session_id = ?', (session_id,))
                
                # 撤销令牌
                if token:
                    token_hash = hashlib.sha256(token.encode()).hexdigest()
                    cursor.execute('UPDATE tokens SET is_revoked = 1 WHERE token_hash = ?', (token_hash,))
                
                conn.commit()
                
                # 记录安全事件
                self._log_security_event(
                    user_id, SecurityEventType.LOGOUT, ip_address, user_agent,
                    {"action": "logout", "session_id": session_id}
                )
                
                return {
                    "success": True,
                    "message": "登出成功"
                }
                
        except Exception as e:
            self.logger.error(f"登出失败: {str(e)}")
            return {
                "success": False,
                "error": f"登出失败: {str(e)}"
            }
    
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 用户信息
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, username, email, phone, is_active, is_verified,
                           last_login, created_at, mfa_enabled, roles, permissions
                    FROM users WHERE user_id = ?
                ''', (user_id,))
                user_row = cursor.fetchone()
                
                if not user_row:
                    return {
                        "success": False,
                        "error": "用户不存在"
                    }
                
                (db_user_id, username, email, phone, is_active, is_verified,
                 last_login, created_at, mfa_enabled, roles_str, permissions_str) = user_row
                
                return {
                    "success": True,
                    "user": {
                        "user_id": db_user_id,
                        "username": username,
                        "email": email,
                        "phone": phone,
                        "is_active": bool(is_active),
                        "is_verified": bool(is_verified),
                        "last_login": last_login,
                        "created_at": created_at,
                        "mfa_enabled": bool(mfa_enabled),
                        "roles": json.loads(roles_str) if roles_str else [],
                        "permissions": json.loads(permissions_str) if permissions_str else []
                    }
                }
                
        except Exception as e:
            self.logger.error(f"获取用户信息失败: {str(e)}")
            return {
                "success": False,
                "error": f"获取用户信息失败: {str(e)}"
            }
    
    def get_security_events(self, user_id: str = None, limit: int = 100) -> Dict[str, Any]:
        """
        获取安全事件日志
        
        Args:
            user_id: 用户ID，如果为None则获取所有用户的事件
            limit: 返回记录数量限制
            
        Returns:
            Dict[str, Any]: 安全事件列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if user_id:
                    cursor.execute('''
                        SELECT event_id, user_id, event_type, timestamp, ip_address, 
                               user_agent, details, severity
                        FROM security_events 
                        WHERE user_id = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (user_id, limit))
                else:
                    cursor.execute('''
                        SELECT event_id, user_id, event_type, timestamp, ip_address, 
                               user_agent, details, severity
                        FROM security_events 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (limit,))
                
                events = []
                for row in cursor.fetchall():
                    (event_id, event_user_id, event_type, timestamp, ip_address,
                     user_agent, details_str, severity) = row
                    
                    events.append({
                        "event_id": event_id,
                        "user_id": event_user_id,
                        "event_type": event_type,
                        "timestamp": timestamp,
                        "ip_address": ip_address,
                        "user_agent": user_agent,
                        "details": json.loads(details_str) if details_str else {},
                        "severity": severity
                    })
                
                return {
                    "success": True,
                    "events": events
                }
                
        except Exception as e:
            self.logger.error(f"获取安全事件失败: {str(e)}")
            return {
                "success": False,
                "error": f"获取安全事件失败: {str(e)}"
            }
    
    def verify_biometric(self, user_id: str, biometric_data: str) -> Dict[str, Any]:
        """
        验证生物识别数据
        
        Args:
            user_id: 用户ID
            biometric_data: 生物识别数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT biometric_data FROM users WHERE user_id = ?', (user_id,))
                user_row = cursor.fetchone()
                
                if not user_row:
                    return {
                        "success": False,
                        "error": "用户不存在"
                    }
                
                stored_biometric = user_row[0]
                if not stored_biometric:
                    return {
                        "success": False,
                        "error": "用户未设置生物识别"
                    }
                
                # 验证生物识别
                is_valid = self.biometric_auth.verify_biometric(stored_biometric, biometric_data)
                
                if is_valid:
                    # 生成访问令牌
                    access_token = self._generate_token(user_id, "access")
                    refresh_token = self._generate_token(user_id, "refresh")
                    
                    return {
                        "success": True,
                        "message": "生物识别验证成功",
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "user_id": user_id
                    }
                else:
                    return {
                        "success": False,
                        "error": "生物识别验证失败"
                    }
                    
        except Exception as e:
            self.logger.error(f"生物识别验证失败: {str(e)}")
            return {
                "success": False,
                "error": f"生物识别验证失败: {str(e)}"
            }
    
    def setup_biometric(self, user_id: str, biometric_data: str) -> Dict[str, Any]:
        """
        为用户设置生物识别
        
        Args:
            user_id: 用户ID
            biometric_data: 生物识别数据
            
        Returns:
            Dict[str, Any]: 设置结果
        """
        try:
            # 生成生物识别模板
            biometric_template = self.biometric_auth.generate_biometric_template(biometric_data)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users 
                    SET biometric_data = ?, updated_at = ?
                    WHERE user_id = ?
                ''', (biometric_template, datetime.now().isoformat(), user_id))
                conn.commit()
                
                if cursor.rowcount == 0:
                    return {
                        "success": False,
                        "error": "用户不存在"
                    }
            
            return {
                "success": True,
                "message": "生物识别设置成功"
            }
            
        except Exception as e:
            self.logger.error(f"生物识别设置失败: {str(e)}")
            return {
                "success": False,
                "error": f"生物识别设置失败: {str(e)}"
            }
    
    def get_password_strength(self, password: str) -> Dict[str, Any]:
        """
        获取密码强度评估
        
        Args:
            password: 密码
            
        Returns:
            Dict[str, Any]: 密码强度评估结果
        """
        try:
            score = PasswordPolicy.check_password_strength(password)
            is_valid, errors = PasswordPolicy.validate_password(password)
            
            # 强度等级
            if score >= 80:
                strength = "很强"
            elif score >= 60:
                strength = "强"
            elif score >= 40:
                strength = "中等"
            elif score >= 20:
                strength = "弱"
            else:
                strength = "很弱"
            
            return {
                "success": True,
                "score": score,
                "strength": strength,
                "is_valid": is_valid,
                "errors": errors,
                "recommendations": self._get_password_recommendations(password, score)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"密码强度评估失败: {str(e)}"
            }
    
    def _get_password_recommendations(self, password: str, score: int) -> List[str]:
        """获取密码改进建议"""
        recommendations = []
        
        if len(password) < 12:
            recommendations.append("建议使用至少12个字符的密码")
        
        if not any(c.isupper() for c in password):
            recommendations.append("添加大写字母可以提高安全性")
        
        if not any(c.islower() for c in password):
            recommendations.append("添加小写字母可以提高安全性")
        
        if not any(c.isdigit() for c in password):
            recommendations.append("添加数字可以提高安全性")
        
        if not any(c in PasswordPolicy.SPECIAL_CHARS for c in password):
            recommendations.append("添加特殊字符可以提高安全性")
        
        if len(set(password)) < len(password) * 0.6:
            recommendations.append("使用更多不同的字符可以提高安全性")
        
        if score < 60:
            recommendations.append("建议使用更复杂的密码组合")
        
        return recommendations
    
    def cleanup_expired_sessions(self) -> int:
        """
        清理过期的会话和令牌
        
        Returns:
            int: 清理的记录数量
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # 清理过期会话
                cursor.execute('DELETE FROM sessions WHERE expires_at < ?', (now,))
                expired_sessions = cursor.rowcount
                
                # 清理过期令牌
                cursor.execute('DELETE FROM tokens WHERE expires_at < ?', (now,))
                expired_tokens = cursor.rowcount
                
                conn.commit()
                
                total_cleaned = expired_sessions + expired_tokens
                self.logger.info(f"清理了 {total_cleaned} 条过期记录")
                
                return total_cleaned
                
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {str(e)}")
            return 0


# 使用示例和测试函数
def demo_authenticator():
    """演示身份验证器的使用"""
    print("=== N1身份验证器演示 ===\n")
    
    # 初始化身份验证器
    auth = Authenticator()
    
    # 1. 用户注册
    print("1. 用户注册")
    register_result = auth.register_user(
        username="testuser",
        email="test@example.com",
        password="SecurePass123!",
        phone="+1234567890"
    )
    print(f"注册结果: {register_result}\n")
    
    if register_result["success"]:
        user_id = register_result["user_id"]
        
        # 2. 设置MFA
        print("2. 设置MFA")
        mfa_result = auth.setup_mfa(user_id)
        print(f"MFA设置结果: {mfa_result}\n")
        
        # 3. 用户登录
        print("3. 用户登录")
        login_result = auth.authenticate_user(
            username="testuser",
            password="SecurePass123!",
            require_mfa=True
        )
        print(f"登录结果: {login_result}\n")
        
        # 4. 密码强度评估
        print("4. 密码强度评估")
        strength_result = auth.get_password_strength("WeakPass")
        print(f"弱密码评估: {strength_result}\n")
        
        strength_result = auth.get_password_strength("VeryStrongPassword123!")
        print(f"强密码评估: {strength_result}\n")
        
        # 5. 获取用户信息
        print("5. 获取用户信息")
        user_info = auth.get_user_info(user_id)
        print(f"用户信息: {user_info}\n")
        
        # 6. 获取安全事件
        print("6. 获取安全事件")
        events = auth.get_security_events(user_id, limit=10)
        print(f"安全事件: {events}\n")
    
    # 7. 清理过期数据
    print("7. 清理过期数据")
    cleaned = auth.cleanup_expired_sessions()
    print(f"清理了 {cleaned} 条过期记录\n")


if __name__ == "__main__":
    demo_authenticator()