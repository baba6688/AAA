"""
S2通知服务模块
提供完整的通知服务解决方案，包括邮件、短信和推送通知功能

作者: S区开发团队
版本: 1.0.0
许可证: MIT

快速开始:
    from S2 import NotificationService, create_notification_service, NotificationType
    
    # 方式1: 使用便利函数
    config = {
        'email': {'smtp_server': 'smtp.gmail.com', 'smtp_port': 587},
        'sms': {'api_url': 'https://api.example.com', 'api_key': 'your_key'}
    }
    service = create_notification_service(config)
    service.start()
    
    # 发送通知
    service.send_email('user@example.com', '欢迎', '欢迎使用我们的服务')
    service.send_sms('13800138000', '您的验证码是1234')
    
    service.stop()

详细文档: 请参考 NotificationService.py 文件
"""

__version__ = "1.0.0"
__author__ = "S区开发团队"
__email__ = "dev-team@s-region.com"
__license__ = "MIT"

# 导入所有核心类
from .NotificationService import (
    NotificationType,
    NotificationStatus,
    NotificationTemplate,
    Notification,
    EmailService,
    SMSService,
    PushService,
    NotificationQueue,
    NotificationHistory,
    NotificationService
)

# 导入便利函数
from .NotificationService import (
    DEFAULT_CONFIG,
    create_notification_service
)

# 版本信息
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'email': __email__,
    'license': __license__,
    'description': 'S2通知服务模块 - 提供邮件、短信和推送通知功能'
}

# 默认配置 - 扩展版本
DEFAULT_EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': '',
    'password': '',
    'from_email': '',
    'use_tls': True,
    'use_ssl': False,
    'timeout': 30
}

DEFAULT_SMS_CONFIG = {
    'api_url': '',
    'api_key': '',
    'timeout': 30,
    'retry_times': 3,
    'retry_interval': 1
}

DEFAULT_PUSH_CONFIG = {
    'fcm_url': 'https://fcm.googleapis.com/fcm/send',
    'server_key': '',
    'timeout': 30,
    'retry_times': 3,
    'retry_interval': 1
}

DEFAULT_SERVICE_CONFIG = {
    'email': DEFAULT_EMAIL_CONFIG,
    'sms': DEFAULT_SMS_CONFIG,
    'push': DEFAULT_PUSH_CONFIG,
    'queue_size': 1000,
    'db_path': 'notification_history.db',
    'auto_start': False,
    'worker_threads': 1,
    'batch_size': 100,
    'retry_failed_interval': 3600,  # 1小时
    'max_retry_times': 3,
    'enable_history': True,
    'enable_statistics': True
}

# 常量定义
DEFAULT_TIMEOUT = 30
DEFAULT_QUEUE_SIZE = 1000
DEFAULT_BATCH_SIZE = 100
DEFAULT_RETRY_TIMES = 3
DEFAULT_WORKER_THREADS = 1

# 通知优先级常量
PRIORITY_LOWEST = 1
PRIORITY_LOW = 2
PRIORITY_NORMAL = 3
PRIORITY_HIGH = 4
PRIORITY_HIGHEST = 5

# 重试间隔常量（秒）
RETRY_INTERVAL_IMMEDIATE = 1
RETRY_INTERVAL_SHORT = 5
RETRY_INTERVAL_MEDIUM = 30
RETRY_INTERVAL_LONG = 300
RETRY_INTERVAL_HOUR = 3600

# 数据库配置常量
DB_DEFAULT_PATH = 'notification_history.db'
DB_BACKUP_SUFFIX = '.backup'
DB_COMMIT_FREQUENCY = 100  # 每100条记录提交一次

# 日志配置常量
LOG_DEFAULT_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 邮件配置常量
EMAIL_DEFAULT_ENCODING = 'utf-8'
EMAIL_DEFAULT_MIME_TYPE = 'plain'
EMAIL_DEFAULT_CHARSET = 'utf-8'
EMAIL_ATTACHMENT_MAX_SIZE = 10 * 1024 * 1024  # 10MB

# 短信配置常量
SMS_DEFAULT_ENCODING = 'utf-8'
SMS_MAX_LENGTH = 160
SMS_CONTENT_PREFIX = '[通知]'
SMS_TEMPLATE_PLACEHOLDER_START = '{'
SMS_TEMPLATE_PLACEHOLDER_END = '}'

# 推送配置常量
PUSH_DEFAULT_ICON = 'default'
PUSH_DEFAULT_SOUND = 'default'
PUSH_DEFAULT_BADGE = 0
PUSH_NOTIFICATION_TIMEOUT = 5
PUSH_DATA_MESSAGE_TIMEOUT = 10

# 验证常量
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
PHONE_REGEX = r'^1[3-9]\d{9}$'
PUSH_TOKEN_REGEX = r'^[a-zA-Z0-9]{10,100}$'

# 错误代码常量
ERROR_CODE_INVALID_EMAIL = 'INVALID_EMAIL'
ERROR_CODE_INVALID_PHONE = 'INVALID_PHONE'
ERROR_CODE_INVALID_PUSH_TOKEN = 'INVALID_PUSH_TOKEN'
ERROR_CODE_SMTP_CONNECTION_FAILED = 'SMTP_CONNECTION_FAILED'
ERROR_CODE_SMS_API_ERROR = 'SMS_API_ERROR'
ERROR_CODE_PUSH_API_ERROR = 'PUSH_API_ERROR'
ERROR_CODE_QUEUE_FULL = 'QUEUE_FULL'
ERROR_CODE_DATABASE_ERROR = 'DATABASE_ERROR'
ERROR_CODE_TEMPLATE_NOT_FOUND = 'TEMPLATE_NOT_FOUND'
ERROR_CODE_VALIDATION_FAILED = 'VALIDATION_FAILED'

# 状态消息
STATUS_MESSAGES = {
    NotificationStatus.PENDING: '等待处理',
    NotificationStatus.SENDING: '正在发送',
    NotificationStatus.SENT: '发送成功',
    NotificationStatus.FAILED: '发送失败',
    NotificationStatus.RETRY: '重试中'
}

# 类型消息
TYPE_MESSAGES = {
    NotificationType.EMAIL: '邮件通知',
    NotificationType.SMS: '短信通知',
    NotificationType.PUSH: '推送通知'
}

def get_version_info():
    """获取版本信息"""
    return VERSION_INFO.copy()

def get_default_config(service_type='all'):
    """获取默认配置
    
    Args:
        service_type: 配置类型，可选值：'all', 'email', 'sms', 'push', 'service'
    
    Returns:
        dict: 相应的配置字典
    """
    configs = {
        'all': DEFAULT_SERVICE_CONFIG,
        'email': DEFAULT_EMAIL_CONFIG,
        'sms': DEFAULT_SMS_CONFIG,
        'push': DEFAULT_PUSH_CONFIG,
        'service': {
            'queue_size': DEFAULT_QUEUE_SIZE,
            'db_path': DB_DEFAULT_PATH,
            'auto_start': False,
            'worker_threads': DEFAULT_WORKER_THREADS,
            'batch_size': DEFAULT_BATCH_SIZE,
            'enable_history': True,
            'enable_statistics': True
        }
    }
    return configs.get(service_type, DEFAULT_SERVICE_CONFIG).copy()

def create_email_config(smtp_server, username, password, from_email, **kwargs):
    """创建邮件配置
    
    Args:
        smtp_server: SMTP服务器地址
        username: 用户名
        password: 密码或应用专用密码
        from_email: 发送者邮箱
        **kwargs: 其他可选配置参数
    
    Returns:
        dict: 邮件配置字典
    """
    config = DEFAULT_EMAIL_CONFIG.copy()
    config.update({
        'smtp_server': smtp_server,
        'username': username,
        'password': password,
        'from_email': from_email
    })
    config.update(kwargs)
    return config

def create_sms_config(api_url, api_key, **kwargs):
    """创建短信配置
    
    Args:
        api_url: 短信API地址
        api_key: API密钥
        **kwargs: 其他可选配置参数
    
    Returns:
        dict: 短信配置字典
    """
    config = DEFAULT_SMS_CONFIG.copy()
    config.update({
        'api_url': api_url,
        'api_key': api_key
    })
    config.update(kwargs)
    return config

def create_push_config(server_key, **kwargs):
    """创建推送配置
    
    Args:
        server_key: 推送服务密钥（如FCM服务器密钥）
        **kwargs: 其他可选配置参数
    
    Returns:
        dict: 推送配置字典
    """
    config = DEFAULT_PUSH_CONFIG.copy()
    config.update({'server_key': server_key})
    config.update(kwargs)
    return config

def create_quick_config(email_config=None, sms_config=None, push_config=None, **kwargs):
    """创建快速配置
    
    Args:
        email_config: 邮件配置字典，None时使用默认配置
        sms_config: 短信配置字典，None时使用默认配置
        push_config: 推送配置字典，None时使用默认配置
        **kwargs: 其他服务配置参数
    
    Returns:
        dict: 完整的通知服务配置
    """
    config = get_default_config('all')
    
    if email_config:
        config['email'].update(email_config)
    if sms_config:
        config['sms'].update(sms_config)
    if push_config:
        config['push'].update(push_config)
    
    config.update(kwargs)
    return config

def validate_email(email):
    """验证邮箱地址格式
    
    Args:
        email: 邮箱地址字符串
    
    Returns:
        bool: 验证结果
    """
    import re
    pattern = re.compile(EMAIL_REGEX)
    return bool(pattern.match(email))

def validate_phone(phone):
    """验证手机号码格式（中国大陆）
    
    Args:
        phone: 手机号码字符串
    
    Returns:
        bool: 验证结果
    """
    import re
    pattern = re.compile(PHONE_REGEX)
    return bool(pattern.match(phone))

def validate_push_token(token):
    """验证推送Token格式
    
    Args:
        token: 推送Token字符串
    
    Returns:
        bool: 验证结果
    """
    import re
    pattern = re.compile(PUSH_TOKEN_REGEX)
    return bool(pattern.match(token))

def validate_recipient(notification_type, recipient):
    """验证接收者信息格式
    
    Args:
        notification_type: 通知类型
        recipient: 接收者信息
    
    Returns:
        bool: 验证结果
    """
    if notification_type == NotificationType.EMAIL:
        return validate_email(recipient)
    elif notification_type == NotificationType.SMS:
        return validate_phone(recipient)
    elif notification_type == NotificationType.PUSH:
        return validate_push_token(recipient)
    return False

def create_quick_service(email_smtp=None, email_username=None, email_password=None, 
                        email_from=None, sms_api_url=None, sms_api_key=None,
                        push_server_key=None, **kwargs):
    """创建快速通知服务实例
    
    Args:
        email_smtp: SMTP服务器地址
        email_username: 邮箱用户名
        email_password: 邮箱密码
        email_from: 发送者邮箱
        sms_api_url: 短信API地址
        sms_api_key: 短信API密钥
        push_server_key: 推送服务密钥
        **kwargs: 其他配置参数
    
    Returns:
        NotificationService: 通知服务实例
    """
    config = get_default_config('all')
    
    # 设置邮件配置
    if email_smtp and email_username and email_password and email_from:
        config['email'] = create_email_config(
            email_smtp, email_username, email_password, email_from
        )
    
    # 设置短信配置
    if sms_api_url and sms_api_key:
        config['sms'] = create_sms_config(sms_api_url, sms_api_key)
    
    # 设置推送配置
    if push_server_key:
        config['push'] = create_push_config(push_server_key)
    
    # 更新其他配置
    config.update(kwargs)
    
    # 创建服务
    service = create_notification_service(config)
    
    # 自动启动（如果配置了）
    if config.get('auto_start', False):
        service.start()
    
    return service

def get_status_message(status):
    """获取状态描述消息
    
    Args:
        status: NotificationStatus枚举值
    
    Returns:
        str: 状态描述消息
    """
    return STATUS_MESSAGES.get(status, '未知状态')

def get_type_message(notification_type):
    """获取类型描述消息
    
    Args:
        notification_type: NotificationType枚举值
    
    Returns:
        str: 类型描述消息
    """
    return TYPE_MESSAGES.get(notification_type, '未知类型')

def create_notification_id():
    """创建通知ID
    
    Returns:
        str: 通知ID字符串
    """
    import uuid
    return str(uuid.uuid4())

def format_notification_log(notification):
    """格式化通知日志信息
    
    Args:
        notification: Notification对象
    
    Returns:
        str: 格式化的日志字符串
    """
    status_msg = get_status_message(notification.status)
    type_msg = get_type_message(notification.type)
    return (f"[{type_msg}] ID:{notification.id} "
            f"收件人:{notification.recipient} "
            f"标题:{notification.title} "
            f"状态:{status_msg} "
            f"重试:{notification.retry_count}/{notification.max_retries}")

# 导出所有公共接口
__all__ = [
    # 版本信息
    '__version__',
    '__author__', 
    '__email__',
    '__license__',
    
    # 核心类
    'NotificationType',
    'NotificationStatus', 
    'NotificationTemplate',
    'Notification',
    'EmailService',
    'SMSService',
    'PushService',
    'NotificationQueue',
    'NotificationHistory',
    'NotificationService',
    
    # 便利函数
    'create_notification_service',
    'create_quick_service',
    
    # 配置
    'DEFAULT_CONFIG',
    'DEFAULT_SERVICE_CONFIG',
    'DEFAULT_EMAIL_CONFIG',
    'DEFAULT_SMS_CONFIG',
    'DEFAULT_PUSH_CONFIG',
    'get_default_config',
    'create_email_config',
    'create_sms_config',
    'create_push_config',
    'create_quick_config',
    
    # 工具函数
    'validate_email',
    'validate_phone',
    'validate_push_token',
    'validate_recipient',
    'create_notification_id',
    'format_notification_log',
    'get_status_message',
    'get_type_message',
    'get_version_info',
    
    # 常量
    'DEFAULT_TIMEOUT',
    'DEFAULT_QUEUE_SIZE',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_RETRY_TIMES',
    'DEFAULT_WORKER_THREADS',
    'PRIORITY_LOWEST',
    'PRIORITY_LOW',
    'PRIORITY_NORMAL',
    'PRIORITY_HIGH',
    'PRIORITY_HIGHEST',
    'RETRY_INTERVAL_IMMEDIATE',
    'RETRY_INTERVAL_SHORT',
    'RETRY_INTERVAL_MEDIUM',
    'RETRY_INTERVAL_LONG',
    'RETRY_INTERVAL_HOUR',
    'DB_DEFAULT_PATH',
    'DB_BACKUP_SUFFIX',
    'DB_COMMIT_FREQUENCY',
    'LOG_DEFAULT_LEVEL',
    'LOG_FORMAT',
    'LOG_DATE_FORMAT',
    'EMAIL_DEFAULT_ENCODING',
    'EMAIL_DEFAULT_MIME_TYPE',
    'EMAIL_DEFAULT_CHARSET',
    'EMAIL_ATTACHMENT_MAX_SIZE',
    'SMS_DEFAULT_ENCODING',
    'SMS_MAX_LENGTH',
    'SMS_CONTENT_PREFIX',
    'SMS_TEMPLATE_PLACEHOLDER_START',
    'SMS_TEMPLATE_PLACEHOLDER_END',
    'PUSH_DEFAULT_ICON',
    'PUSH_DEFAULT_SOUND',
    'PUSH_DEFAULT_BADGE',
    'PUSH_NOTIFICATION_TIMEOUT',
    'PUSH_DATA_MESSAGE_TIMEOUT',
    
    # 错误代码
    'ERROR_CODE_INVALID_EMAIL',
    'ERROR_CODE_INVALID_PHONE',
    'ERROR_CODE_INVALID_PUSH_TOKEN',
    'ERROR_CODE_SMTP_CONNECTION_FAILED',
    'ERROR_CODE_SMS_API_ERROR',
    'ERROR_CODE_PUSH_API_ERROR',
    'ERROR_CODE_QUEUE_FULL',
    'ERROR_CODE_DATABASE_ERROR',
    'ERROR_CODE_TEMPLATE_NOT_FOUND',
    'ERROR_CODE_VALIDATION_FAILED'
]

# 模块初始化日志
def _init_module():
    """模块初始化"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"S2通知服务模块 v{__version__} 已加载")
    logger.info(f"作者: {__author__} | 许可证: {__license__}")

# 自动初始化
_init_module()