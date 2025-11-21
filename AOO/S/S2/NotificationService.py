"""
S2通知服务模块
提供邮件通知、短信通知、推送通知等功能的综合通知服务
"""

import smtplib
import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import logging
import os

# 可选依赖
try:
    import requests
except ImportError:
    requests = None


class NotificationType(Enum):
    """通知类型枚举"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"


class NotificationStatus(Enum):
    """通知状态枚举"""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class NotificationTemplate:
    """通知模板"""
    id: str
    name: str
    type: NotificationType
    subject: str
    content: str
    variables: List[str]
    created_at: str
    updated_at: str


@dataclass
class Notification:
    """通知对象"""
    id: str
    type: NotificationType
    recipient: str
    title: str
    content: str
    template_id: Optional[str] = None
    priority: int = 1  # 1-5, 5为最高优先级
    status: NotificationStatus = NotificationStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: str = ""
    sent_at: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def __lt__(self, other):
        """支持优先级队列比较"""
        if not isinstance(other, Notification):
            return NotImplemented
        return self.priority < other.priority
    
    def __le__(self, other):
        """支持优先级队列比较"""
        if not isinstance(other, Notification):
            return NotImplemented
        return self.priority <= other.priority
    
    def __gt__(self, other):
        """支持优先级队列比较"""
        if not isinstance(other, Notification):
            return NotImplemented
        return self.priority > other.priority
    
    def __ge__(self, other):
        """支持优先级队列比较"""
        if not isinstance(other, Notification):
            return NotImplemented
        return self.priority >= other.priority


class EmailService:
    """邮件服务"""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
        self.logger = logging.getLogger(__name__)
    
    def send_email(self, notification: Notification) -> bool:
        """发送邮件"""
        try:
            # 创建邮件内容
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = notification.recipient
            msg['Subject'] = notification.title
            
            # 添加邮件正文
            msg.attach(MIMEText(notification.content, 'plain', 'utf-8'))
            
            # 添加附件（如果有）
            if notification.metadata and 'attachments' in notification.metadata:
                for attachment in notification.metadata['attachments']:
                    self._add_attachment(msg, attachment)
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port']) as server:
                if self.smtp_config.get('use_tls', True):
                    server.starttls()
                if self.smtp_config.get('username') and self.smtp_config.get('password'):
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            self.logger.info(f"邮件发送成功: {notification.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"邮件发送失败: {notification.id}, 错误: {str(e)}")
            return False
    
    def _add_attachment(self, msg: MIMEMultipart, attachment: Dict[str, Any]):
        """添加附件"""
        try:
            with open(attachment['file_path'], 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment["file_path"])}'
                )
                msg.attach(part)
        except Exception as e:
            self.logger.warning(f"附件添加失败: {str(e)}")


class SMSService:
    """短信服务"""
    
    def __init__(self, sms_config: Dict[str, Any]):
        self.sms_config = sms_config
        self.logger = logging.getLogger(__name__)
    
    def send_sms(self, notification: Notification) -> bool:
        """发送短信"""
        try:
            # 构建短信内容
            message = f"{notification.title}\n{notification.content}"
            
            # 调用短信网关API
            payload = {
                'phone': notification.recipient,
                'message': message,
                'template_id': notification.template_id,
                'variables': notification.metadata.get('variables', {}) if notification.metadata else {}
            }
            
            if requests is None:
                raise ImportError("requests module is not installed")
            
            headers = {
                'Authorization': f"Bearer {self.sms_config['api_key']}",
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.sms_config['api_url'],
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info(f"短信发送成功: {notification.id}")
                return True
            else:
                self.logger.error(f"短信发送失败: {notification.id}, 状态码: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"短信发送异常: {notification.id}, 错误: {str(e)}")
            return False


class PushService:
    """推送服务"""
    
    def __init__(self, push_config: Dict[str, Any]):
        self.push_config = push_config
        self.logger = logging.getLogger(__name__)
    
    def send_push(self, notification: Notification) -> bool:
        """发送推送通知"""
        try:
            # 构建推送payload
            payload = {
                'to': notification.recipient,
                'notification': {
                    'title': notification.title,
                    'body': notification.content,
                    'icon': notification.metadata.get('icon', 'default') if notification.metadata else 'default',
                    'sound': notification.metadata.get('sound', 'default') if notification.metadata else 'default'
                },
                'data': notification.metadata or {}
            }
            
            if requests is None:
                raise ImportError("requests module is not installed")
            
            headers = {
                'Authorization': f"key={self.push_config['server_key']}",
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.push_config['fcm_url'],
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info(f"推送发送成功: {notification.id}")
                return True
            else:
                self.logger.error(f"推送发送失败: {notification.id}, 状态码: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"推送发送异常: {notification.id}, 错误: {str(e)}")
            return False


class NotificationQueue:
    """通知队列"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.lock = threading.Lock()
    
    def put(self, notification: Notification):
        """添加通知到队列（按优先级排序）"""
        # 优先级队列：负数优先级，数字越小优先级越高
        priority = -notification.priority
        self.queue.put((priority, notification))
    
    def get(self, timeout: Optional[float] = None) -> Notification:
        """从队列获取通知"""
        _, notification = self.queue.get(timeout=timeout)
        return notification
    
    def qsize(self) -> int:
        """队列大小"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """队列是否为空"""
        return self.queue.empty()


class NotificationHistory:
    """通知历史记录"""
    
    def __init__(self, db_path: str = "notification_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                recipient TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                template_id TEXT,
                priority INTEGER DEFAULT 1,
                status TEXT DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                created_at TEXT NOT NULL,
                sent_at TEXT,
                error_message TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                subject TEXT NOT NULL,
                content TEXT NOT NULL,
                variables TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_notification(self, notification: Notification):
        """保存通知记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO notifications 
            (id, type, recipient, title, content, template_id, priority, status, 
             retry_count, max_retries, created_at, sent_at, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            notification.id, notification.type.value, notification.recipient,
            notification.title, notification.content, notification.template_id,
            notification.priority, notification.status.value, notification.retry_count,
            notification.max_retries, notification.created_at, notification.sent_at,
            notification.error_message, json.dumps(notification.metadata) if notification.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_notification_history(self, recipient: Optional[str] = None, 
                               type: Optional[NotificationType] = None,
                               status: Optional[NotificationStatus] = None,
                               limit: int = 100) -> List[Dict]:
        """获取通知历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM notifications WHERE 1=1"
        params = []
        
        if recipient:
            query += " AND recipient = ?"
            params.append(recipient)
        
        if type:
            query += " AND type = ?"
            params.append(type.value)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        result = []
        for row in rows:
            notification_dict = dict(zip(columns, row))
            if notification_dict.get('metadata'):
                try:
                    notification_dict['metadata'] = json.loads(notification_dict['metadata'])
                except:
                    notification_dict['metadata'] = None
            result.append(notification_dict)
        
        conn.close()
        return result
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取通知统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # 总发送量
        cursor.execute("SELECT COUNT(*) FROM notifications WHERE created_at >= ?", (start_date,))
        total_sent = cursor.fetchone()[0]
        
        # 成功率
        cursor.execute("SELECT COUNT(*) FROM notifications WHERE created_at >= ? AND status = 'sent'", (start_date,))
        successful_sent = cursor.fetchone()[0]
        
        # 失败率
        cursor.execute("SELECT COUNT(*) FROM notifications WHERE created_at >= ? AND status = 'failed'", (start_date,))
        failed_sent = cursor.fetchone()[0]
        
        # 按类型统计
        cursor.execute('''
            SELECT type, COUNT(*) FROM notifications 
            WHERE created_at >= ? 
            GROUP BY type
        ''', (start_date,))
        type_stats = dict(cursor.fetchall())
        
        # 按状态统计
        cursor.execute('''
            SELECT status, COUNT(*) FROM notifications 
            WHERE created_at >= ? 
            GROUP BY status
        ''', (start_date,))
        status_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_sent': total_sent,
            'successful_sent': successful_sent,
            'failed_sent': failed_sent,
            'success_rate': successful_sent / total_sent if total_sent > 0 else 0,
            'failure_rate': failed_sent / total_sent if total_sent > 0 else 0,
            'type_statistics': type_stats,
            'status_statistics': status_stats,
            'period_days': days
        }


class NotificationService:
    """通知服务主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        
        # 初始化各个服务
        self.email_service = EmailService(config.get('email', {}))
        self.sms_service = SMSService(config.get('sms', {}))
        self.push_service = PushService(config.get('push', {}))
        
        # 初始化队列和历史记录
        self.notification_queue = NotificationQueue(config.get('queue_size', 1000))
        self.history = NotificationHistory(config.get('db_path', 'notification_history.db'))
        
        # 模板存储
        self.templates: Dict[str, NotificationTemplate] = {}
        
        # 工作线程
        self.worker_thread = None
        self.is_running = False
        
        # 加载模板
        self._load_templates()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('NotificationService')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_templates(self):
        """加载模板"""
        # 默认模板
        default_templates = [
            NotificationTemplate(
                id="welcome_email",
                name="欢迎邮件",
                type=NotificationType.EMAIL,
                subject="欢迎使用我们的服务",
                content="亲爱的用户，欢迎您使用我们的服务！\n\n您的账户已成功创建。",
                variables=["username"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            ),
            NotificationTemplate(
                id="verification_sms",
                name="验证码短信",
                type=NotificationType.SMS,
                subject="",
                content="您的验证码是：{code}，5分钟内有效。",
                variables=["code"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            ),
            NotificationTemplate(
                id="push_notification",
                name="推送通知",
                type=NotificationType.PUSH,
                subject="新消息",
                content="您有新的消息：{message}",
                variables=["message"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    def start(self):
        """启动通知服务"""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            self.logger.info("通知服务已启动")
    
    def stop(self):
        """停止通知服务"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.logger.info("通知服务已停止")
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.is_running:
            try:
                if not self.notification_queue.empty():
                    notification = self.notification_queue.get(timeout=1)
                    self._process_notification(notification)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"工作线程错误: {str(e)}")
                time.sleep(1)
    
    def _process_notification(self, notification: Notification):
        """处理单个通知"""
        notification.status = NotificationStatus.SENDING
        self.history.save_notification(notification)
        
        success = False
        
        try:
            if notification.type == NotificationType.EMAIL:
                success = self.email_service.send_email(notification)
            elif notification.type == NotificationType.SMS:
                success = self.sms_service.send_sms(notification)
            elif notification.type == NotificationType.PUSH:
                success = self.push_service.send_push(notification)
            
            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now().isoformat()
            else:
                notification.retry_count += 1
                if notification.retry_count < notification.max_retries:
                    notification.status = NotificationStatus.RETRY
                    # 重新加入队列
                    self.notification_queue.put(notification)
                else:
                    notification.status = NotificationStatus.FAILED
                    notification.error_message = "达到最大重试次数"
        
        except Exception as e:
            notification.status = NotificationStatus.FAILED
            notification.error_message = str(e)
            self.logger.error(f"通知处理异常: {notification.id}, 错误: {str(e)}")
        
        self.history.save_notification(notification)
    
    def send_notification(self, type: NotificationType, recipient: str, 
                         title: str, content: str, template_id: Optional[str] = None,
                         priority: int = 1, metadata: Optional[Dict] = None) -> str:
        """发送通知"""
        notification = Notification(
            id=str(uuid.uuid4()),
            type=type,
            recipient=recipient,
            title=title,
            content=content,
            template_id=template_id,
            priority=priority,
            metadata=metadata
        )
        
        # 如果有模板ID，尝试使用模板
        if template_id and template_id in self.templates:
            template = self.templates[template_id]
            if template.type == type:
                # 替换模板变量
                if metadata and 'variables' in metadata:
                    for var_name, var_value in metadata['variables'].items():
                        template.content = template.content.replace(f"{{{var_name}}}", str(var_value))
                        template.subject = template.subject.replace(f"{{{var_name}}}", str(var_value))
                
                notification.title = template.subject
                notification.content = template.content
        
        # 保存到历史记录
        self.history.save_notification(notification)
        
        # 加入队列
        self.notification_queue.put(notification)
        
        self.logger.info(f"通知已加入队列: {notification.id}")
        return notification.id
    
    def send_email(self, recipient: str, subject: str, content: str, 
                   template_id: Optional[str] = None, priority: int = 1,
                   metadata: Optional[Dict] = None) -> str:
        """发送邮件通知"""
        return self.send_notification(
            NotificationType.EMAIL, recipient, subject, content, 
            template_id, priority, metadata
        )
    
    def send_sms(self, recipient: str, content: str, template_id: Optional[str] = None,
                priority: int = 1, metadata: Optional[Dict] = None) -> str:
        """发送短信通知"""
        return self.send_notification(
            NotificationType.SMS, recipient, "", content,
            template_id, priority, metadata
        )
    
    def send_push(self, recipient: str, title: str, content: str,
                 template_id: Optional[str] = None, priority: int = 1,
                 metadata: Optional[Dict] = None) -> str:
        """发送推送通知"""
        return self.send_notification(
            NotificationType.PUSH, recipient, title, content,
            template_id, priority, metadata
        )
    
    def add_template(self, template: NotificationTemplate):
        """添加通知模板"""
        self.templates[template.id] = template
        
        # 保存到数据库
        conn = sqlite3.connect(self.history.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO templates 
            (id, name, type, subject, content, variables, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            template.id, template.name, template.type.value,
            template.subject, template.content, json.dumps(template.variables),
            template.created_at, template.updated_at
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"模板已添加: {template.id}")
    
    def get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """获取通知模板"""
        return self.templates.get(template_id)
    
    def get_all_templates(self) -> List[NotificationTemplate]:
        """获取所有通知模板"""
        return list(self.templates.values())
    
    def get_notification_status(self, notification_id: str) -> Optional[Dict]:
        """获取通知状态"""
        history = self.history.get_notification_history(limit=1)
        for item in history:
            if item['id'] == notification_id:
                return item
        return None
    
    def get_notification_history(self, recipient: Optional[str] = None,
                               type: Optional[NotificationType] = None,
                               status: Optional[NotificationStatus] = None,
                               limit: int = 100) -> List[Dict]:
        """获取通知历史"""
        return self.history.get_notification_history(recipient, type, status, limit)
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取通知统计"""
        return self.history.get_statistics(days)
    
    def retry_failed_notifications(self):
        """重试失败的通知"""
        failed_notifications = self.history.get_notification_history(
            status=NotificationStatus.FAILED, limit=1000
        )
        
        for notif_data in failed_notifications:
            if notif_data['retry_count'] < notif_data['max_retries']:
                notification = Notification(
                    id=notif_data['id'],
                    type=NotificationType(notif_data['type']),
                    recipient=notif_data['recipient'],
                    title=notif_data['title'],
                    content=notif_data['content'],
                    template_id=notif_data['template_id'],
                    priority=notif_data['priority'],
                    status=NotificationStatus.PENDING,
                    retry_count=notif_data['retry_count'],
                    max_retries=notif_data['max_retries'],
                    created_at=notif_data['created_at'],
                    metadata=json.loads(notif_data['metadata']) if notif_data['metadata'] else None
                )
                self.notification_queue.put(notification)
        
        self.logger.info(f"已重试 {len(failed_notifications)} 个失败通知")


# 配置示例
DEFAULT_CONFIG = {
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_app_password',
        'from_email': 'your_email@gmail.com',
        'use_tls': True
    },
    'sms': {
        'api_url': 'https://api.sms-provider.com/send',
        'api_key': 'your_sms_api_key'
    },
    'push': {
        'fcm_url': 'https://fcm.googleapis.com/fcm/send',
        'server_key': 'your_fcm_server_key'
    },
    'queue_size': 1000,
    'db_path': 'notification_history.db'
}


def create_notification_service(config: Optional[Dict[str, Any]] = None) -> NotificationService:
    """创建通知服务实例"""
    if config is None:
        config = DEFAULT_CONFIG
    
    return NotificationService(config)