"""
S5报告服务核心实现

提供完整的报告生成、调度、发送、存储等功能
"""

import os
import json
import datetime
import smtplib
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Any, Union
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

# 可选依赖
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    schedule = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportStatus(Enum):
    """报告状态枚举"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    SENT = "sent"


class ReportType(Enum):
    """报告类型枚举"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """报告配置类"""
    name: str
    description: str
    report_type: ReportType
    template_id: str
    recipients: List[str]
    schedule_cron: str
    enabled: bool = True
    created_at: Optional[datetime.datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()


@dataclass
class ReportTemplate:
    """报告模板类"""
    id: str
    name: str
    description: str
    content: str
    variables: List[str]
    format_type: str = "html"  # html, pdf, excel
    created_at: Optional[datetime.datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()


class ReportTemplateManager:
    """报告模板管理器"""
    
    def __init__(self, storage_path: str = "templates"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.templates: Dict[str, ReportTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """从文件加载模板"""
        template_file = self.storage_path / "templates.json"
        if template_file.exists():
            with open(template_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for template_data in data:
                    template = ReportTemplate(**template_data)
                    self.templates[template.id] = template
    
    def _save_templates(self):
        """保存模板到文件"""
        template_file = self.storage_path / "templates.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(template) for template in self.templates.values()], 
                     f, ensure_ascii=False, indent=2, default=str)
    
    def create_template(self, template: ReportTemplate) -> bool:
        """创建新模板"""
        try:
            self.templates[template.id] = template
            self._save_templates()
            logger.info(f"模板创建成功: {template.name}")
            return True
        except Exception as e:
            logger.error(f"模板创建失败: {e}")
            return False
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """获取模板"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[ReportTemplate]:
        """列出所有模板"""
        return list(self.templates.values())
    
    def delete_template(self, template_id: str) -> bool:
        """删除模板"""
        try:
            if template_id in self.templates:
                del self.templates[template_id]
                self._save_templates()
                logger.info(f"模板删除成功: {template_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"模板删除失败: {e}")
            return False


class ReportSchedule:
    """报告调度器"""
    
    def __init__(self):
        self.schedules: Dict[str, ReportConfig] = {}
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def add_schedule(self, config: ReportConfig):
        """添加调度配置"""
        self.schedules[config.name] = config
        if config.enabled:
            self._setup_schedule(config)
        logger.info(f"调度配置添加成功: {config.name}")
    
    def _setup_schedule(self, config: ReportConfig):
        """设置调度任务"""
        if not SCHEDULE_AVAILABLE:
            logger.warning("schedule模块不可用，跳过调度设置")
            return
            
        try:
            if config.report_type == ReportType.DAILY:
                schedule.every().day.at("09:00").do(self._run_report, config).tag(config.name)
            elif config.report_type == ReportType.WEEKLY:
                schedule.every().monday.at("09:00").do(self._run_report, config).tag(config.name)
            elif config.report_type == ReportType.MONTHLY:
                schedule.every().month.do(self._run_report, config).tag(config.name)
            else:
                # 自定义cron表达式
                schedule.every().minute.do(self._run_report, config).tag(config.name)
        except Exception as e:
            logger.error(f"调度设置失败: {e}")
    
    def _run_report(self, config: ReportConfig):
        """执行报告生成"""
        logger.info(f"开始执行报告: {config.name}")
        # 这里会调用ReportService的生成方法
        pass
    
    def start(self):
        """启动调度器"""
        if not SCHEDULE_AVAILABLE:
            logger.warning("schedule模块不可用，无法启动调度器")
            return
            
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.thread.start()
            logger.info("报告调度器已启动")
    
    def stop(self):
        """停止调度器"""
        self.running = False
        if self.thread:
            self.thread.join()
        if SCHEDULE_AVAILABLE:
            schedule.clear()
        logger.info("报告调度器已停止")
    
    def _run_scheduler(self):
        """运行调度器主循环"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)


class ReportSender:
    """报告发送器"""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
        self.smtp_server = smtp_config.get('server', 'localhost')
        self.smtp_port = smtp_config.get('port', 587)
        self.username = smtp_config.get('username', '')
        self.password = smtp_config.get('password', '')
    
    def send_report(self, report_path: str, recipients: List[str], subject: str, 
                   body: str = "") -> bool:
        """发送报告邮件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # 添加邮件正文
            if body:
                msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 添加报告附件
            if os.path.exists(report_path):
                with open(report_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(report_path)}'
                )
                msg.attach(part)
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"报告发送成功: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"报告发送失败: {e}")
            return False


class ReportStorage:
    """报告存储管理器"""
    
    def __init__(self, storage_path: str = "reports"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_report(self, report_name: str, content: Union[str, bytes], 
                   format_type: str = "html") -> str:
        """保存报告"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_name}_{timestamp}.{format_type}"
        filepath = self.storage_path / filename
        
        if isinstance(content, str):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(filepath, 'wb') as f:
                f.write(content)
        
        logger.info(f"报告保存成功: {filepath}")
        return str(filepath)
    
    def get_reports(self, report_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取报告列表"""
        reports = []
        for file_path in self.storage_path.glob("*.html"):
            stat = file_path.stat()
            report_info = {
                "name": file_path.stem,
                "path": str(file_path),
                "size": stat.st_size,
                "created_at": datetime.datetime.fromtimestamp(stat.st_ctime),
                "modified_at": datetime.datetime.fromtimestamp(stat.st_mtime)
            }
            
            if report_name is None or report_name in file_path.stem:
                reports.append(report_info)
        
        return sorted(reports, key=lambda x: x['created_at'], reverse=True)
    
    def delete_report(self, report_path: str) -> bool:
        """删除报告"""
        try:
            if os.path.exists(report_path):
                os.remove(report_path)
                logger.info(f"报告删除成功: {report_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"报告删除失败: {e}")
            return False


class ReportPermission:
    """报告权限管理器"""
    
    def __init__(self, db_path: str = "permissions.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化权限数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                report_name TEXT NOT NULL,
                permission_type TEXT NOT NULL,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                granted_by TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def grant_permission(self, user_id: str, report_name: str, 
                        permission_type: str, granted_by: str = "system") -> bool:
        """授予权限"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO permissions (user_id, report_name, permission_type, granted_by)
                VALUES (?, ?, ?, ?)
            ''', (user_id, report_name, permission_type, granted_by))
            
            conn.commit()
            conn.close()
            logger.info(f"权限授予成功: {user_id} -> {report_name}")
            return True
        except Exception as e:
            logger.error(f"权限授予失败: {e}")
            return False
    
    def revoke_permission(self, user_id: str, report_name: str) -> bool:
        """撤销权限"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM permissions 
                WHERE user_id = ? AND report_name = ?
            ''', (user_id, report_name))
            
            conn.commit()
            conn.close()
            logger.info(f"权限撤销成功: {user_id} -> {report_name}")
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"权限撤销失败: {e}")
            return False
    
    def check_permission(self, user_id: str, report_name: str, 
                        required_permission: str = "read") -> bool:
        """检查权限"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT permission_type FROM permissions 
                WHERE user_id = ? AND report_name = ?
            ''', (user_id, report_name))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                # 简单的权限检查逻辑
                user_permission = result[0]
                return user_permission in [required_permission, "admin", "full"]
            
            return False
        except Exception as e:
            logger.error(f"权限检查失败: {e}")
            return False
    
    def get_user_permissions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户权限"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT report_name, permission_type, granted_at, granted_by
                FROM permissions WHERE user_id = ?
            ''', (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            permissions = []
            for row in results:
                permissions.append({
                    "report_name": row[0],
                    "permission_type": row[1],
                    "granted_at": row[2],
                    "granted_by": row[3]
                })
            
            return permissions
        except Exception as e:
            logger.error(f"获取用户权限失败: {e}")
            return []


class ReportVersion:
    """报告版本管理器"""
    
    def __init__(self, storage_path: str = "versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.versions_db = self.storage_path / "versions.db"
        self._init_database()
    
    def _init_database(self):
        """初始化版本数据库"""
        conn = sqlite3.connect(str(self.versions_db))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_name TEXT NOT NULL,
                version_number TEXT NOT NULL,
                file_path TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_version(self, report_name: str, content: Union[str, bytes], 
                      version_number: str, description: str = "", 
                      created_by: str = "system") -> str:
        """创建新版本"""
        try:
            # 保存文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_name}_v{version_number}_{timestamp}.html"
            filepath = self.storage_path / filename
            
            if isinstance(content, str):
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                with open(filepath, 'wb') as f:
                    f.write(content)
            
            # 保存版本信息
            conn = sqlite3.connect(str(self.versions_db))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO versions (report_name, version_number, file_path, description, created_by)
                VALUES (?, ?, ?, ?, ?)
            ''', (report_name, version_number, str(filepath), description, created_by))
            
            conn.commit()
            conn.close()
            
            logger.info(f"报告版本创建成功: {report_name} v{version_number}")
            return str(filepath)
        except Exception as e:
            logger.error(f"报告版本创建失败: {e}")
            raise
    
    def get_versions(self, report_name: str) -> List[Dict[str, Any]]:
        """获取报告版本列表"""
        try:
            conn = sqlite3.connect(str(self.versions_db))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT version_number, file_path, description, created_at, created_by
                FROM versions WHERE report_name = ?
                ORDER BY created_at DESC
            ''', (report_name,))
            
            results = cursor.fetchall()
            conn.close()
            
            versions = []
            for row in results:
                versions.append({
                    "version_number": row[0],
                    "file_path": row[1],
                    "description": row[2],
                    "created_at": row[3],
                    "created_by": row[4]
                })
            
            return versions
        except Exception as e:
            logger.error(f"获取报告版本失败: {e}")
            return []
    
    def get_version_content(self, report_name: str, version_number: str) -> Optional[str]:
        """获取特定版本内容"""
        try:
            conn = sqlite3.connect(str(self.versions_db))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_path FROM versions 
                WHERE report_name = ? AND version_number = ?
            ''', (report_name, version_number))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                file_path = result[0]
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
            
            return None
        except Exception as e:
            logger.error(f"获取版本内容失败: {e}")
            return None


class ReportStatistics:
    """报告统计管理器"""
    
    def __init__(self, db_path: str = "statistics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化统计数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_name TEXT NOT NULL,
                report_type TEXT NOT NULL,
                status TEXT NOT NULL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                execution_time REAL,
                file_size INTEGER,
                recipient_count INTEGER,
                error_message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_generation(self, report_name: str, report_type: str, status: ReportStatus,
                         execution_time: float = 0, file_size: int = 0, 
                         recipient_count: int = 0, error_message: str = ""):
        """记录报告生成统计"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO statistics 
                (report_name, report_type, status, execution_time, file_size, recipient_count, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (report_name, report_type, status.value, execution_time, file_size, 
                  recipient_count, error_message))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"记录统计信息失败: {e}")
    
    def get_statistics(self, start_date: Optional[datetime.datetime] = None,
                      end_date: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT 
                    COUNT(*) as total_reports,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_reports,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_reports,
                    AVG(execution_time) as avg_execution_time,
                    SUM(file_size) as total_file_size,
                    SUM(recipient_count) as total_recipients
                FROM statistics
            '''
            params = []
            
            if start_date:
                query += " WHERE generated_at >= ?"
                params.append(start_date)
                if end_date:
                    query += " AND generated_at <= ?"
                    params.append(end_date)
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            # 获取报告类型分布
            cursor.execute('''
                SELECT report_type, COUNT(*) as count
                FROM statistics
                GROUP BY report_type
            ''')
            type_distribution = dict(cursor.fetchall())
            
            # 获取状态分布
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM statistics
                GROUP BY status
            ''')
            status_distribution = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_reports": result[0] or 0,
                "successful_reports": result[1] or 0,
                "failed_reports": result[2] or 0,
                "success_rate": (result[1] or 0) / (result[0] or 1) * 100,
                "avg_execution_time": result[3] or 0,
                "total_file_size": result[4] or 0,
                "total_recipients": result[5] or 0,
                "type_distribution": type_distribution,
                "status_distribution": status_distribution
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近活动"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT report_name, report_type, status, generated_at, execution_time
                FROM statistics
                ORDER BY generated_at DESC
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            activities = []
            for row in results:
                activities.append({
                    "report_name": row[0],
                    "report_type": row[1],
                    "status": row[2],
                    "generated_at": row[3],
                    "execution_time": row[4]
                })
            
            return activities
        except Exception as e:
            logger.error(f"获取最近活动失败: {e}")
            return []


class ReportService:
    """S5报告服务主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.template_manager = ReportTemplateManager(
            config.get('template_path', 'templates')
        )
        self.schedule = ReportSchedule()
        self.sender = ReportSender(config.get('smtp', {}))
        self.storage = ReportStorage(config.get('storage_path', 'reports'))
        self.permission = ReportPermission(config.get('permission_db', 'permissions.db'))
        self.version = ReportVersion(config.get('version_path', 'versions'))
        self.statistics = ReportStatistics(config.get('statistics_db', 'statistics.db'))
        
        # 初始化默认模板
        self._init_default_templates()
    
    def _init_default_templates(self):
        """初始化默认模板"""
        default_templates = [
            ReportTemplate(
                id="daily_summary",
                name="日报模板",
                description="每日汇总报告模板",
                content="""
                <html>
                <head><title>日报 - {{date}}</title></head>
                <body>
                    <h1>日报 - {{date}}</h1>
                    <h2>业务概览</h2>
                    <p>今日业务数据：{{business_data}}</p>
                    <h2>关键指标</h2>
                    <ul>
                        <li>用户数：{{user_count}}</li>
                        <li>订单数：{{order_count}}</li>
                        <li>收入：{{revenue}}</li>
                    </ul>
                    <h2>重要事件</h2>
                    <p>{{events}}</p>
                </body>
                </html>
                """,
                variables=["date", "business_data", "user_count", "order_count", "revenue", "events"]
            ),
            ReportTemplate(
                id="weekly_analysis",
                name="周报模板",
                description="每周分析报告模板",
                content="""
                <html>
                <head><title>周报 - {{week}}</title></head>
                <body>
                    <h1>周报 - {{week}}</h1>
                    <h2>本周总结</h2>
                    <p>{{summary}}</p>
                    <h2>数据分析</h2>
                    <table border="1">
                        <tr><th>指标</th><th>本周</th><th>上周</th><th>变化</th></tr>
                        <tr><td>用户增长</td><td>{{current_users}}</td><td>{{previous_users}}</td><td>{{user_growth}}</td></tr>
                        <tr><td>订单量</td><td>{{current_orders}}</td><td>{{previous_orders}}</td><td>{{order_growth}}</td></tr>
                    </table>
                    <h2>下周计划</h2>
                    <p>{{next_week_plan}}</p>
                </body>
                </html>
                """,
                variables=["week", "summary", "current_users", "previous_users", "user_growth",
                          "current_orders", "previous_orders", "order_growth", "next_week_plan"]
            )
        ]
        
        for template in default_templates:
            if not self.template_manager.get_template(template.id):
                self.template_manager.create_template(template)
    
    def generate_report(self, config: ReportConfig, data: Dict[str, Any]) -> str:
        """生成报告"""
        start_time = time.time()
        report_path = ""
        
        try:
            # 获取模板
            template = self.template_manager.get_template(config.template_id)
            if not template:
                raise ValueError(f"模板不存在: {config.template_id}")
            
            # 替换模板变量
            content = template.content
            for var in template.variables:
                placeholder = f"{{{{{var}}}}}"
                value = data.get(var, f"[{var}未定义]")
                content = content.replace(placeholder, str(value))
            
            # 添加生成时间
            content = content.replace("{{generated_at}}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # 保存报告
            report_path = self.storage.save_report(
                config.name, content, template.format_type
            )
            
            # 创建版本
            version_number = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.version.create_version(
                config.name, content, version_number, 
                f"自动生成版本 {version_number}"
            )
            
            # 记录统计
            execution_time = time.time() - start_time
            file_size = os.path.getsize(report_path) if os.path.exists(report_path) else 0
            self.statistics.record_generation(
                config.name, config.report_type.value, ReportStatus.COMPLETED,
                execution_time, file_size, len(config.recipients)
            )
            
            logger.info(f"报告生成成功: {config.name}")
            return report_path
            
        except Exception as e:
            # 记录错误统计
            execution_time = time.time() - start_time
            self.statistics.record_generation(
                config.name, config.report_type.value, ReportStatus.FAILED,
                execution_time, 0, len(config.recipients), str(e)
            )
            logger.error(f"报告生成失败: {e}")
            raise
    
    def send_report(self, report_path: str, config: ReportConfig, 
                   subject: str = "", body: str = "") -> bool:
        """发送报告"""
        try:
            if not subject:
                subject = f"{config.name} - {datetime.datetime.now().strftime('%Y-%m-%d')}"
            
            if not body:
                body = f"请查看附件中的{config.name}报告。"
            
            success = self.sender.send_report(
                report_path, config.recipients, subject, body
            )
            
            if success:
                # 记录发送统计
                self.statistics.record_generation(
                    config.name, config.report_type.value, ReportStatus.SENT,
                    file_size=os.path.getsize(report_path) if os.path.exists(report_path) else 0,
                    recipient_count=len(config.recipients)
                )
            
            return success
        except Exception as e:
            logger.error(f"报告发送失败: {e}")
            return False
    
    def add_schedule(self, config: ReportConfig):
        """添加报告调度"""
        self.schedule.add_schedule(config)
    
    def start_scheduler(self):
        """启动调度器"""
        self.schedule.start()
    
    def stop_scheduler(self):
        """停止调度器"""
        self.schedule.stop()
    
    def get_statistics(self, start_date: Optional[datetime.datetime] = None,
                      end_date: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """获取统计信息"""
        return self.statistics.get_statistics(start_date, end_date)
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近活动"""
        return self.statistics.get_recent_activity(limit)
    
    def check_permission(self, user_id: str, report_name: str) -> bool:
        """检查用户权限"""
        return self.permission.check_permission(user_id, report_name)
    
    def grant_permission(self, user_id: str, report_name: str, 
                        permission_type: str = "read") -> bool:
        """授予用户权限"""
        return self.permission.grant_permission(user_id, report_name, permission_type)
    
    def get_report_versions(self, report_name: str) -> List[Dict[str, Any]]:
        """获取报告版本"""
        return self.version.get_versions(report_name)
    
    def get_template_content(self, template_id: str) -> Optional[str]:
        """获取模板内容"""
        template = self.template_manager.get_template(template_id)
        return template.content if template else None
    
    def list_templates(self) -> List[ReportTemplate]:
        """列出所有模板"""
        return self.template_manager.list_templates()