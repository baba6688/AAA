"""
S3邮件服务模块
==================

完整的邮件发送、接收、管理和统计功能模块。

功能特性：
- 邮件发送与接收（SMTP、POP3、IMAP）
- 邮件模板系统
- 附件处理
- 邮件队列管理
- 垃圾邮件过滤
- 内容加密签名
- 邮件统计监控

版本信息
---------
"""

from typing import Union, List

# 版本信息
__version__ = "1.0.0"
__author__ = "S3邮件服务团队"
__email__ = "support@s3-email.com"
__description__ = "S3邮件服务 - 完整的邮件处理解决方案"

# 导入所有核心类
from .EmailService import (
    EmailService,          # 主要邮件服务类
    EmailConfig,           # 邮件配置类
    EmailMessage,          # 邮件消息类
    EmailTemplate,         # 邮件模板类
    EmailAttachment,       # 邮件附件类
    EmailQueue,            # 邮件队列类
    EmailFilter,           # 邮件过滤器类
    EmailSecurity,         # 邮件安全类
    EmailStats            # 邮件统计类
)

# 默认配置
DEFAULT_SMTP_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'smtp_username': '',
    'smtp_password': '',
    'smtp_use_tls': True,
    'smtp_use_ssl': False
}

DEFAULT_POP3_CONFIG = {
    'pop3_server': 'pop.gmail.com',
    'pop3_port': 995,
    'pop3_username': '',
    'pop3_password': '',
    'pop3_use_ssl': True
}

DEFAULT_IMAP_CONFIG = {
    'imap_server': 'imap.gmail.com',
    'imap_port': 993,
    'imap_username': '',
    'imap_password': '',
    'imap_use_ssl': True
}

# 常用邮箱服务配置
EMAIL_PROVIDERS = {
    'gmail': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'smtp_use_tls': True,
        'pop3_server': 'pop.gmail.com',
        'pop3_port': 995,
        'imap_server': 'imap.gmail.com',
        'imap_port': 993
    },
    'qq': {
        'smtp_server': 'smtp.qq.com',
        'smtp_port': 587,
        'smtp_use_tls': True,
        'pop3_server': 'pop.qq.com',
        'pop3_port': 995,
        'imap_server': 'imap.qq.com',
        'imap_port': 993
    },
    '163': {
        'smtp_server': 'smtp.163.com',
        'smtp_port': 25,
        'smtp_use_tls': False,
        'smtp_use_ssl': True,
        'pop3_server': 'pop.163.com',
        'pop3_port': 995,
        'imap_server': 'imap.163.com',
        'imap_port': 993
    },
    'outlook': {
        'smtp_server': 'smtp-mail.outlook.com',
        'smtp_port': 587,
        'smtp_use_tls': True,
        'pop3_server': 'outlook.office365.com',
        'pop3_port': 995,
        'imap_server': 'outlook.office365.com',
        'imap_port': 993
    }
}

# 常量定义
class EmailConstants:
    """邮件相关常量"""
    
    # 优先级
    PRIORITY_HIGH = "high"
    PRIORITY_NORMAL = "normal"
    PRIORITY_LOW = "low"
    
    # 内容类型
    CONTENT_TYPE_TEXT = "text"
    CONTENT_TYPE_HTML = "html"
    
    # 附件类型
    ATTACHMENT_TYPE_PDF = "pdf"
    ATTACHMENT_TYPE_IMAGE = "image"
    ATTACHMENT_TYPE_DOCUMENT = "document"
    ATTACHMENT_TYPE_AUDIO = "audio"
    ATTACHMENT_TYPE_VIDEO = "video"
    
    # 安全级别
    SECURITY_NONE = "none"
    SECURITY_SIGNATURE = "signature"
    SECURITY_ENCRYPTION = "encryption"
    SECURITY_BOTH = "both"
    
    # 队列状态
    QUEUE_RUNNING = "running"
    QUEUE_STOPPED = "stopped"
    QUEUE_PAUSED = "paused"

# 便利函数
def create_gmail_config(username: str, password: str, encryption_key: str = "") -> EmailConfig:
    """创建Gmail配置
    
    Args:
        username: Gmail用户名（完整邮箱地址）
        password: Gmail密码或应用专用密码
        encryption_key: 加密密钥
    
    Returns:
        EmailConfig: 邮件配置对象
    """
    return EmailConfig(
        smtp_server=EMAIL_PROVIDERS['gmail']['smtp_server'],
        smtp_port=EMAIL_PROVIDERS['gmail']['smtp_port'],
        smtp_username=username,
        smtp_password=password,
        smtp_use_tls=EMAIL_PROVIDERS['gmail']['smtp_use_tls'],
        smtp_use_ssl=False,
        pop3_server=EMAIL_PROVIDERS['gmail']['pop3_server'],
        pop3_port=EMAIL_PROVIDERS['gmail']['pop3_port'],
        pop3_username=username,
        pop3_password=password,
        pop3_use_ssl=True,
        imap_server=EMAIL_PROVIDERS['gmail']['imap_server'],
        imap_port=EMAIL_PROVIDERS['gmail']['imap_port'],
        imap_username=username,
        imap_password=password,
        imap_use_ssl=True,
        encryption_key=encryption_key
    )

def create_qq_config(username: str, password: str, encryption_key: str = "") -> EmailConfig:
    """创建QQ邮箱配置"""
    return EmailConfig(
        smtp_server=EMAIL_PROVIDERS['qq']['smtp_server'],
        smtp_port=EMAIL_PROVIDERS['qq']['smtp_port'],
        smtp_username=username,
        smtp_password=password,
        smtp_use_tls=EMAIL_PROVIDERS['qq']['smtp_use_tls'],
        pop3_server=EMAIL_PROVIDERS['qq']['pop3_server'],
        pop3_port=EMAIL_PROVIDERS['qq']['pop3_port'],
        pop3_username=username,
        pop3_password=password,
        pop3_use_ssl=True,
        imap_server=EMAIL_PROVIDERS['qq']['imap_server'],
        imap_port=EMAIL_PROVIDERS['qq']['imap_port'],
        imap_username=username,
        imap_password=password,
        imap_use_ssl=True,
        encryption_key=encryption_key
    )

def create_simple_message(to: str, subject: str, body: str, 
                         from_email: str = "", body_type: str = "text") -> EmailMessage:
    """创建简单邮件消息
    
    Args:
        to: 收件人
        subject: 邮件主题
        body: 邮件正文
        from_email: 发件人邮箱
        body_type: 内容类型（text或html）
    
    Returns:
        EmailMessage: 邮件消息对象
    """
    return EmailMessage(
        to=to,
        subject=subject,
        body=body,
        body_type=body_type,
        from_email=from_email
    )

def create_html_message(to: str, subject: str, html_content: str, 
                       from_email: str = "") -> EmailMessage:
    """创建HTML格式邮件消息"""
    return create_simple_message(to, subject, html_content, from_email, "html")

def create_attachment(filename: str, content: Union[bytes, str], 
                     content_type: str = "", content_id: str = "") -> EmailAttachment:
    """创建邮件附件
    
    Args:
        filename: 附件文件名
        content: 附件内容
        content_type: 内容类型
        content_id: 内容ID（用于内嵌图片）
    
    Returns:
        EmailAttachment: 附件对象
    """
    return EmailAttachment(
        filename=filename,
        content=content,
        content_type=content_type,
        content_id=content_id
    )

def create_template(name: str, subject_template: str, body_template: str,
                   template_type: str = "text", variables: List[str] = None) -> EmailTemplate:
    """创建邮件模板
    
    Args:
        name: 模板名称
        subject_template: 主题模板
        body_template: 正文模板
        template_type: 模板类型（text或html）
        variables: 模板变量列表
    
    Returns:
        EmailTemplate: 邮件模板对象
    """
    return EmailTemplate(
        name=name,
        subject_template=subject_template,
        body_template=body_template,
        template_type=template_type,
        variables=variables or []
    )

def quick_start(username: str, password: str, provider: str = "gmail") -> EmailService:
    """快速初始化邮件服务
    
    Args:
        username: 邮箱用户名
        password: 邮箱密码
        provider: 邮箱提供商（gmail, qq, 163, outlook）
    
    Returns:
        EmailService: 配置好的邮件服务实例
    
    Examples:
        # Gmail快速启动
        email_service = quick_start("user@gmail.com", "app_password", "gmail")
        
        # 发送简单邮件
        message = create_simple_message(
            "recipient@example.com", 
            "测试邮件", 
            "这是一封测试邮件"
        )
        email_service.send_email(message)
    """
    if provider not in EMAIL_PROVIDERS:
        raise ValueError(f"不支持的邮箱提供商: {provider}")
    
    if provider == "gmail":
        config = create_gmail_config(username, password)
    elif provider == "qq":
        config = create_qq_config(username, password)
    else:
        # 通用配置
        provider_config = EMAIL_PROVIDERS[provider]
        config = EmailConfig(
            smtp_server=provider_config['smtp_server'],
            smtp_port=provider_config['smtp_port'],
            smtp_username=username,
            smtp_password=password,
            smtp_use_tls=provider_config.get('smtp_use_tls', True),
            smtp_use_ssl=provider_config.get('smtp_use_ssl', False),
            encryption_key=""
        )
    
    return EmailService(config)

# 快速入门示例
QUICK_START_EXAMPLES = """
快速入门示例
=============

1. 基础使用
-----------

```python
from S3 import quick_start, create_simple_message

# 快速初始化Gmail服务
email_service = quick_start("your_email@gmail.com", "your_app_password", "gmail")

# 创建并发送邮件
message = create_simple_message(
    to="recipient@example.com",
    subject="测试邮件",
    body="这是一封测试邮件"
)

# 发送邮件
success = email_service.send_email(message)
print(f"邮件发送{'成功' if success else '失败'}")
```

2. HTML邮件
-----------

```python
from S3 import quick_start, create_html_message

# 创建HTML邮件
html_message = create_html_message(
    to="recipient@example.com",
    subject="HTML邮件示例",
    html_content="<html><body><h1>Welcome</h1><p>HTML email</p></body></html>"
)

email_service.send_email(html_message)
```

3. 附件邮件
-----------

```python
from S3 import quick_start, create_simple_message, create_attachment

# 创建带附件的邮件
message = create_simple_message(
    to="recipient@example.com",
    subject="带附件的邮件",
    body="请查看附件"
)

# 添加附件
with open("document.pdf", "rb") as f:
    attachment = create_attachment("document.pdf", f.read(), "application/pdf")
    message.attachments.append(attachment)

email_service.send_email(message)
```

4. 模板邮件
-----------

```python
from S3 import quick_start, EmailTemplate

# 添加模板
template = EmailTemplate(
    name="welcome_email",
    subject_template="Welcome to {company}, {username}!",
    body_template="Dear {username},\\n\\nWelcome to {company}!\\n\\nYour account information:\\n- Username: {username}\\n- Email: {email}\\n- Registration time: {register_time}\\n\\nIf you have any questions, please contact customer service.\\n\\nBest regards!",
    variables=["username", "email", "company", "register_time"]
)

email_service.add_template(template)

# 发送模板邮件
email_service.send_template_email(
    template_name="welcome_email",
    recipients=["user@example.com"],
    variables={
        "username": "张三",
        "email": "zhangsan@example.com",
        "company": "科技公司",
        "register_time": "2025-11-13"
    }
)
```

5. 邮件统计
-----------

```python
# 获取统计信息
stats = email_service.get_stats()
print(email_service.get_stats_summary())

# 查看队列状态
queue_stats = email_service.queue.get_stats()
print(f"队列状态: {queue_stats}")
```

6. 垃圾邮件过滤
--------------

```python
# 添加黑名单
email_service.add_to_blacklist("spam@spam.com")

# 添加白名单
email_service.add_to_whitelist("important@company.com")

# 手动过滤邮件
filter_result = email_service.filter.is_spam(message)
print(f"邮件过滤结果: {'垃圾邮件' if filter_result else '正常邮件'}")
```
"""

# 导出声明
__all__ = [
    # 核心类
    "EmailService",
    "EmailConfig",
    "EmailMessage", 
    "EmailTemplate",
    "EmailAttachment",
    "EmailQueue",
    "EmailFilter",
    "EmailSecurity",
    "EmailStats",
    
    # 常量类
    "EmailConstants",
    
    # 便利函数
    "create_gmail_config",
    "create_qq_config", 
    "create_simple_message",
    "create_html_message",
    "create_attachment",
    "create_template",
    "quick_start",
    
    # 预定义配置
    "DEFAULT_SMTP_CONFIG",
    "DEFAULT_POP3_CONFIG", 
    "DEFAULT_IMAP_CONFIG",
    "EMAIL_PROVIDERS",
    
    # 示例代码
    "QUICK_START_EXAMPLES",
    
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]

# 模块初始化日志
logger = __import__('logging').getLogger(__name__)
logger.info(f"S3邮件服务模块已加载 v{__version__}")

# 便捷访问
S3Email = EmailService  # 别名

# 文档生成标识
__doc__ = f"""
S3邮件服务模块 v{__version__}

{__description__}

主要功能：
- 邮件发送与接收
- 模板系统
- 附件处理  
- 队列管理
- 垃圾过滤
- 加密签名
- 统计监控

快速开始：使用quick_start()函数初始化服务
详细文档：请参考QUICK_START_EXAMPLES
"""

# 验证依赖
try:
    import smtplib
    import poplib
    import imaplib
    import email
    print("✓ 核心邮件依赖已就绪")
except ImportError as e:
    print(f"⚠ 缺少邮件依赖: {e}")

# 可选依赖检查
try:
    from cryptography.fernet import Fernet
    print("✓ 加密功能可用")
except ImportError:
    print("⚠ 加密功能不可用，请安装: pip install cryptography")

# 使用示例提示
if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*50)
    print("快速测试:")
    print("="*50)
    print(QUICK_START_EXAMPLES[:500] + "...")