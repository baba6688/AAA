#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N1身份验证器包

一个完整的企业级身份验证系统，提供用户认证、多因子认证、JWT令牌管理等核心功能。

主要模块:
- Authenticator: 核心身份验证器类
- User: 用户数据类
- Session: 会话管理类
- PasswordPolicy: 密码策略类
- BiometricAuthenticator: 生物识别认证器
- OAuthProvider: OAuth提供者枚举

使用示例:
    from N1 import Authenticator, User, Session
    
    auth = Authenticator()
    user = User(username="john", password="secure_password")
    session = auth.authenticate_user(user.username, user.password)

版本: 1.0.0
日期: 2025-11-06
"""

from .Authenticator import (
    AuthLevel,
    SecurityEventType,
    User,
    SecurityEvent,
    Session,
    PasswordPolicy,
    BiometricAuthenticator,
    OAuthProvider,
    Authenticator
)

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "security@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # 核心类
    "Authenticator",
    
    # 用户管理
    "User",
    "Session",
    
    # 安全策略
    "AuthLevel",
    "SecurityEventType",
    "SecurityEvent",
    "PasswordPolicy",
    
    # 认证扩展
    "BiometricAuthenticator",
    "OAuthProvider",
]

# 包初始化信息
def get_version():
    """获取版本信息"""
    return __version__

def get_author():
    """获取作者信息"""
    return __author__

def get_license():
    """获取许可证信息"""
    return __license__

# 便捷函数
def create_authenticator(config_file=None):
    """创建身份验证器实例的便捷函数
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        Authenticator实例
    """
    return Authenticator(config_file)

def create_user(username, password, email=None, **kwargs):
    """创建用户实例的便捷函数
    
    Args:
        username: 用户名
        password: 密码
        email: 邮箱地址，可选
        **kwargs: 其他用户属性
        
    Returns:
        User实例
    """
    return User(username=username, password=password, email=email, **kwargs)

# 使用示例
EXAMPLE_USAGE = """
使用示例:

1. 基本身份验证:
   from N1 import Authenticator, User
   
   auth = Authenticator()
   result = auth.authenticate_user("username", "password")
   if result:
       print(f"用户 {result.username} 登录成功")

2. 创建用户:
   from N1 import create_user
   
   user = create_user("john", "secure_password", email="john@example.com")
   print(f"用户 {user.username} 创建成功")

3. 多因子认证:
   from N1 import BiometricAuthenticator
   
   bio_auth = BiometricAuthenticator()
   mfa_result = bio_auth.verify_biometric(user_id, biometric_data)

4. 会话管理:
   from N1 import Session, AuthLevel
   
   session = Session(user_id="user123", auth_level=AuthLevel.ADMIN)
   if auth.validate_session(session.session_id):
       print("会话有效")
"""

if __name__ == "__main__":
    print("N1身份验证器包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)