"""
S3邮件服务核心实现
提供完整的邮件发送、接收、管理等功能
"""

import smtplib
import poplib
import imaplib
import email
import email.mime.text
import email.mime.multipart
import email.mime.base
import email.encoders
import email.utils
import json
import os
import hashlib
import base64
import mimetypes
import threading
import queue
import time
import ssl
import socket
from datetime import datetime, timedelta
from email.header import decode_header
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

# 可选依赖
CRYPTO_AVAILABLE = False
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    pass


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """邮件配置类"""
    smtp_server: str
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    
    pop3_server: str = ""
    pop3_port: int = 110
    pop3_username: str = ""
    pop3_password: str = ""
    pop3_use_ssl: bool = False
    
    imap_server: str = ""
    imap_port: int = 143
    imap_username: str = ""
    imap_password: str = ""
    imap_use_ssl: bool = False
    
    encryption_key: str = ""
    max_attachment_size: int = 10 * 1024 * 1024  # 10MB
    retry_attempts: int = 3
    timeout: int = 30


@dataclass
class EmailAttachment:
    """邮件附件类"""
    filename: str
    content: Union[bytes, str]
    content_type: str = ""
    content_id: str = ""
    
    def __post_init__(self):
        if not self.content_type:
            self.content_type = mimetypes.guess_type(self.filename)[0] or 'application/octet-stream'


@dataclass
class EmailMessage:
    """邮件消息类"""
    to: Union[str, List[str]]
    subject: str
    body: str
    body_type: str = "text"  # "text" or "html"
    from_email: str = ""
    cc: List[str] = None
    bcc: List[str] = None
    attachments: List[EmailAttachment] = None
    priority: str = "normal"  # "high", "normal", "low"
    
    def __post_init__(self):
        if self.cc is None:
            self.cc = []
        if self.bcc is None:
            self.bcc = []
        if self.attachments is None:
            self.attachments = []


@dataclass
class EmailTemplate:
    """邮件模板类"""
    name: str
    subject_template: str
    body_template: str
    variables: List[str] = None
    template_type: str = "text"  # "text" or "html"
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []


class EmailQueue:
    """邮件发送队列"""
    
    def __init__(self, max_workers: int = 5):
        self.queue = queue.Queue()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.stats = {
            'sent': 0,
            'failed': 0,
            'total': 0,
            'start_time': None,
            'last_activity': None
        }
    
    def start(self):
        """启动队列"""
        if not self.is_running:
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            logger.info("邮件队列已启动")
    
    def stop(self):
        """停止队列"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("邮件队列已停止")
    
    def add_email(self, email_service: 'EmailService', message: EmailMessage, callback: Callable = None):
        """添加邮件到队列"""
        self.queue.put((email_service, message, callback))
        self.stats['total'] += 1
        self._process_queue()
    
    def _process_queue(self):
        """处理队列"""
        if not self.is_running:
            return
        
        def process_email():
            try:
                email_service, message, callback = self.queue.get(timeout=1)
                email_service._send_email_internal(message)
                self.stats['sent'] += 1
                self.stats['last_activity'] = datetime.now()
                if callback:
                    callback(True, None)
            except queue.Empty:
                pass
            except Exception as e:
                self.stats['failed'] += 1
                self.stats['last_activity'] = datetime.now()
                logger.error(f"邮件发送失败: {e}")
                if callback:
                    callback(False, str(e))
            finally:
                if self.is_running:
                    self.executor.submit(process_email)
        
        self.executor.submit(process_email)
    
    def get_stats(self) -> Dict:
        """获取队列统计"""
        return self.stats.copy()


class EmailFilter:
    """邮件过滤器"""
    
    def __init__(self):
        self.blacklist = set()
        self.whitelist = set()
        self.spam_keywords = [
            'spam', '广告', '推广', '免费', '优惠', '特价',
            'click here', 'buy now', 'urgent', 'act now'
        ]
        self.max_body_length = 10000
    
    def add_to_blacklist(self, email: str):
        """添加到黑名单"""
        self.blacklist.add(email.lower())
    
    def add_to_whitelist(self, email: str):
        """添加到白名单"""
        self.whitelist.add(email.lower())
    
    def is_spam(self, message: EmailMessage) -> bool:
        """检查是否为垃圾邮件"""
        # 检查发件人
        from_email = message.from_email.lower()
        if from_email in self.blacklist:
            return True
        if self.whitelist and from_email not in self.whitelist:
            return True
        
        # 检查主题和内容
        content = (message.subject + " " + message.body).lower()
        for keyword in self.spam_keywords:
            if keyword in content:
                return True
        
        # 检查邮件长度
        if len(message.body) > self.max_body_length:
            return True
        
        return False


class EmailSecurity:
    """邮件安全类"""
    
    def __init__(self, encryption_key: str = ""):
        self.encryption_key = encryption_key
        self.cipher = None
        if encryption_key:
            self._setup_encryption(encryption_key)
    
    def _setup_encryption(self, key: str):
        """设置加密"""
        if not CRYPTO_AVAILABLE:
            logger.warning("cryptography模块不可用，跳过加密设置")
            return
        
        # 使用PBKDF2生成密钥
        salt = b's3_email_salt'  # 在实际应用中应该使用随机盐
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        self.cipher = Fernet(key)
    
    def encrypt_content(self, content: str) -> str:
        """加密内容"""
        if not self.cipher:
            return content
        return self.cipher.encrypt(content.encode()).decode()
    
    def decrypt_content(self, encrypted_content: str) -> str:
        """解密内容"""
        if not self.cipher:
            return encrypted_content
        return self.cipher.decrypt(encrypted_content.encode()).decode()
    
    def generate_signature(self, content: str) -> str:
        """生成邮件签名"""
        hash_object = hashlib.sha256(content.encode())
        return hash_object.hexdigest()
    
    def verify_signature(self, content: str, signature: str) -> bool:
        """验证邮件签名"""
        expected_signature = self.generate_signature(content)
        return expected_signature == signature


class EmailStats:
    """邮件统计类"""
    
    def __init__(self):
        self.stats = {
            'total_sent': 0,
            'total_received': 0,
            'failed_sends': 0,
            'success_rate': 0.0,
            'daily_stats': {},
            'hourly_stats': {},
            'top_recipients': {},
            'attachment_stats': {'total_size': 0, 'count': 0}
        }
    
    def record_sent(self, recipient: str, attachment_size: int = 0):
        """记录发送邮件"""
        self.stats['total_sent'] += 1
        now = datetime.now()
        date_key = now.strftime('%Y-%m-%d')
        hour_key = now.strftime('%H')
        
        # 每日统计
        if date_key not in self.stats['daily_stats']:
            self.stats['daily_stats'][date_key] = 0
        self.stats['daily_stats'][date_key] += 1
        
        # 每小时统计
        if hour_key not in self.stats['hourly_stats']:
            self.stats['hourly_stats'][hour_key] = 0
        self.stats['hourly_stats'][hour_key] += 1
        
        # 收件人统计
        if recipient not in self.stats['top_recipients']:
            self.stats['top_recipients'][recipient] = 0
        self.stats['top_recipients'][recipient] += 1
        
        # 附件统计
        if attachment_size > 0:
            self.stats['attachment_stats']['total_size'] += attachment_size
            self.stats['attachment_stats']['count'] += 1
        
        self._update_success_rate()
    
    def record_failed(self):
        """记录发送失败"""
        self.stats['failed_sends'] += 1
        self._update_success_rate()
    
    def record_received(self):
        """记录接收邮件"""
        self.stats['total_received'] += 1
    
    def _update_success_rate(self):
        """更新成功率"""
        total = self.stats['total_sent'] + self.stats['failed_sends']
        if total > 0:
            self.stats['success_rate'] = (self.stats['total_sent'] / total) * 100
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def get_summary(self) -> str:
        """获取统计摘要"""
        stats = self.get_stats()
        return f"""
邮件统计摘要:
- 总发送: {stats['total_sent']}
- 总接收: {stats['total_received']}
- 失败发送: {stats['failed_sends']}
- 成功率: {stats['success_rate']:.2f}%
- 附件总数: {stats['attachment_stats']['count']}
- 附件总大小: {stats['attachment_stats']['total_size'] / 1024 / 1024:.2f} MB
        """.strip()


class EmailService:
    """邮件服务主类"""
    
    def __init__(self, config: EmailConfig):
        self.config = config
        self.queue = EmailQueue()
        self.filter = EmailFilter()
        self.security = EmailSecurity(config.encryption_key)
        self.stats = EmailStats()
        self.templates = {}
        self.is_running = False
        
        # 启动队列
        self.queue.start()
        self.is_running = True
        
        logger.info("邮件服务初始化完成")
    
    def add_template(self, template: EmailTemplate):
        """添加邮件模板"""
        self.templates[template.name] = template
        logger.info(f"邮件模板已添加: {template.name}")
    
    def send_email(self, message: EmailMessage, use_queue: bool = True, callback: Callable = None) -> bool:
        """发送邮件"""
        try:
            # 垃圾邮件过滤
            if self.filter.is_spam(message):
                logger.warning("邮件被垃圾邮件过滤器拦截")
                if callback:
                    callback(False, "邮件被垃圾邮件过滤器拦截")
                return False
            
            # 使用队列发送
            if use_queue:
                self.queue.add_email(self, message, callback)
                logger.info("邮件已添加到发送队列")
                return True
            else:
                # 直接发送
                result = self._send_email_internal(message)
                if callback:
                    callback(result, None if result else "发送失败")
                return result
                
        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
            self.stats.record_failed()
            if callback:
                callback(False, str(e))
            return False
    
    def _send_email_internal(self, message: EmailMessage) -> bool:
        """内部邮件发送方法"""
        try:
            # 创建邮件对象
            msg = email.mime.multipart.MIMEMultipart()
            msg['From'] = message.from_email or self.config.smtp_username
            msg['To'] = ', '.join(message.to) if isinstance(message.to, list) else message.to
            msg['Subject'] = message.subject
            msg['Date'] = email.utils.formatdate()
            
            # 添加CC和BCC
            if message.cc:
                msg['Cc'] = ', '.join(message.cc)
            if message.bcc:
                msg['Bcc'] = ', '.join(message.bcc)
            
            # 设置优先级
            if message.priority == "high":
                msg['X-Priority'] = "1"
                msg['X-MSMail-Priority'] = "High"
            elif message.priority == "low":
                msg['X-Priority'] = "5"
                msg['X-MSMail-Priority'] = "Low"
            
            # 添加邮件正文
            if message.body_type == "html":
                msg.attach(email.mime.text.MIMEText(message.body, 'html', 'utf-8'))
            else:
                msg.attach(email.mime.text.MIMEText(message.body, 'plain', 'utf-8'))
            
            # 添加附件
            total_attachment_size = 0
            for attachment in message.attachments:
                if len(attachment.content) > self.config.max_attachment_size:
                    raise ValueError(f"附件 {attachment.filename} 超过大小限制")
                
                total_attachment_size += len(attachment.content)
                
                if isinstance(attachment.content, str):
                    attachment.content = attachment.content.encode('utf-8')
                
                part = email.mime.base.MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.content)
                email.encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment.filename}'
                )
                msg.attach(part)
            
            # 发送邮件
            if self.config.smtp_use_ssl:
                server = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port, timeout=self.config.timeout)
            else:
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port, timeout=self.config.timeout)
                if self.config.smtp_use_tls:
                    server.starttls()
            
            if self.config.smtp_username and self.config.smtp_password:
                server.login(self.config.smtp_username, self.config.smtp_password)
            
            # 收件人列表
            all_recipients = []
            if isinstance(message.to, str):
                all_recipients = [message.to]
            else:
                all_recipients = message.to
            all_recipients.extend(message.cc)
            all_recipients.extend(message.bcc)
            
            server.send_message(msg)
            server.quit()
            
            # 记录统计
            for recipient in all_recipients:
                self.stats.record_sent(recipient, total_attachment_size)
            
            logger.info(f"邮件发送成功: {message.subject}")
            return True
            
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            self.stats.record_failed()
            raise
    
    def receive_emails_pop3(self, limit: int = 10) -> List[Dict]:
        """通过POP3接收邮件"""
        if not self.config.pop3_server:
            raise ValueError("POP3服务器未配置")
        
        try:
            if self.config.pop3_use_ssl:
                server = poplib.POP3_SSL(self.config.pop3_server, self.config.pop3_port, timeout=self.config.timeout)
            else:
                server = poplib.POP3(self.config.pop3_server, self.config.pop3_port, timeout=self.config.timeout)
            
            server.user(self.config.pop3_username)
            server.pass_(self.config.pop3_password)
            
            messages = []
            num_messages = len(server.list()[1])
            
            for i in range(min(limit, num_messages)):
                try:
                    response, msg_lines, octets = server.retr(i + 1)
                    msg_content = b'\n'.join(msg_lines)
                    msg = email.message_from_bytes(msg_content)
                    
                    # 解析邮件信息
                    message_info = self._parse_email_message(msg)
                    messages.append(message_info)
                    self.stats.record_received()
                    
                except Exception as e:
                    logger.error(f"解析邮件 {i + 1} 失败: {e}")
                    continue
            
            server.quit()
            logger.info(f"成功接收 {len(messages)} 封邮件")
            return messages
            
        except Exception as e:
            logger.error(f"POP3接收邮件失败: {e}")
            raise
    
    def receive_emails_imap(self, folder: str = "INBOX", limit: int = 10) -> List[Dict]:
        """通过IMAP接收邮件"""
        if not self.config.imap_server:
            raise ValueError("IMAP服务器未配置")
        
        try:
            if self.config.imap_use_ssl:
                server = imaplib.IMAP4_SSL(self.config.imap_server, self.config.imap_port)
            else:
                server = imaplib.IMAP4(self.config.imap_server, self.config.imap_port)
            
            server.login(self.config.imap_username, self.config.imap_password)
            server.select(folder)
            
            messages = []
            typ, msg_nums = server.search(None, 'ALL')
            
            if msg_nums[0]:
                msg_ids = msg_nums[0].split()
                for msg_id in msg_ids[-limit:]:
                    try:
                        typ, msg_data = server.fetch(msg_id, '(RFC822)')
                        msg_content = msg_data[0][1]
                        msg = email.message_from_bytes(msg_content)
                        
                        message_info = self._parse_email_message(msg)
                        messages.append(message_info)
                        self.stats.record_received()
                        
                    except Exception as e:
                        logger.error(f"解析邮件 {msg_id.decode()} 失败: {e}")
                        continue
            
            server.logout()
            logger.info(f"成功接收 {len(messages)} 封邮件")
            return messages
            
        except Exception as e:
            logger.error(f"IMAP接收邮件失败: {e}")
            raise
    
    def _parse_email_message(self, msg: email.message.Message) -> Dict:
        """解析邮件消息"""
        def decode_header_value(value):
            """解码邮件头"""
            if value is None:
                return ""
            decoded_fragments = decode_header(value)
            decoded_string = ""
            for fragment, encoding in decoded_fragments:
                if isinstance(fragment, bytes):
                    try:
                        decoded_string += fragment.decode(encoding or 'utf-8')
                    except (UnicodeDecodeError, LookupError):
                        decoded_string += fragment.decode('utf-8', errors='replace')
                else:
                    decoded_string += fragment
            return decoded_string
        
        # 提取基本信息
        subject = decode_header_value(msg.get('Subject', ''))
        from_email = decode_header_value(msg.get('From', ''))
        to = decode_header_value(msg.get('To', ''))
        date = msg.get('Date', '')
        
        # 提取正文
        body = ""
        html_body = ""
        attachments = []
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                # 提取正文
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    except:
                        body = part.get_payload()
                elif content_type == "text/html" and "attachment" not in content_disposition:
                    try:
                        html_body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    except:
                        html_body = part.get_payload()
                
                # 提取附件
                elif "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        filename = decode_header_value(filename)
                        content = part.get_payload(decode=True)
                        attachments.append({
                            'filename': filename,
                            'content': content,
                            'content_type': content_type
                        })
        else:
            # 单部分邮件
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
            elif content_type == "text/html":
                html_body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
        
        return {
            'subject': subject,
            'from': from_email,
            'to': to,
            'date': date,
            'body': body,
            'html_body': html_body,
            'attachments': attachments,
            'message_id': msg.get('Message-ID', ''),
            'in_reply_to': msg.get('In-Reply-To', ''),
            'references': msg.get('References', '')
        }
    
    def send_template_email(self, template_name: str, recipients: Union[str, List[str]], 
                          variables: Dict[str, Any], **kwargs) -> bool:
        """发送模板邮件"""
        if template_name not in self.templates:
            raise ValueError(f"模板 {template_name} 不存在")
        
        template = self.templates[template_name]
        
        # 替换模板变量
        try:
            subject = template.subject_template.format(**variables)
            body = template.body_template.format(**variables)
        except KeyError as e:
            raise ValueError(f"模板变量 {e} 未提供")
        
        # 创建邮件消息
        message = EmailMessage(
            to=recipients,
            subject=subject,
            body=body,
            body_type=template.template_type,
            **kwargs
        )
        
        return self.send_email(message)
    
    def download_attachment(self, attachment: Dict, save_path: str) -> bool:
        """下载附件"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(attachment['content'])
            logger.info(f"附件已下载: {save_path}")
            return True
        except Exception as e:
            logger.error(f"下载附件失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取邮件统计"""
        return {
            'email_stats': self.stats.get_stats(),
            'queue_stats': self.queue.get_stats(),
            'filter_stats': {
                'blacklist_count': len(self.filter.blacklist),
                'whitelist_count': len(self.filter.whitelist)
            }
        }
    
    def get_stats_summary(self) -> str:
        """获取统计摘要"""
        return self.stats.get_summary()
    
    def add_to_blacklist(self, email: str):
        """添加到黑名单"""
        self.filter.add_to_blacklist(email)
    
    def add_to_whitelist(self, email: str):
        """添加白名单"""
        self.filter.add_to_whitelist(email)
    
    def stop(self):
        """停止邮件服务"""
        self.queue.stop()
        self.is_running = False
        logger.info("邮件服务已停止")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()