#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q9文档状态聚合器

该模块提供完整的文档状态聚合、分析、监控和报告功能。
支持从多个文档模块收集状态信息，生成综合报告和可视化仪表板。
"""

import os
import json
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import sqlite3
import threading
from collections import defaultdict, deque
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentStatus(Enum):
    """文档状态枚举"""
    UNKNOWN = "unknown"
    ACTIVE = "active"
    OUTDATED = "outdated"
    MISSING = "missing"
    CORRUPTED = "corrupted"
    ARCHIVED = "archived"
    DRAFT = "draft"
    PUBLISHED = "published"


class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DocumentInfo:
    """文档信息数据结构"""
    doc_id: str
    name: str
    path: str
    status: DocumentStatus
    last_modified: float
    size: int
    checksum: str
    module: str
    version: str
    metadata: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int


@dataclass
class StatusReport:
    """状态报告数据结构"""
    timestamp: float
    total_documents: int
    active_documents: int
    outdated_documents: int
    missing_documents: int
    corrupted_documents: int
    status_distribution: Dict[str, int]
    module_status: Dict[str, Dict[str, Any]]
    trends: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class Alert:
    """预警信息数据结构"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    affected_documents: List[str]
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None


class StatusCollector:
    """状态收集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.document_paths = config.get('document_paths', [])
        self.module_configs = config.get('module_configs', {})
        self.scan_interval = config.get('scan_interval', 300)  # 5分钟
        
    def collect_document_status(self, module_name: str, module_path: str) -> List[DocumentInfo]:
        """收集指定模块的文档状态"""
        documents = []
        
        try:
            for root, dirs, files in os.walk(module_path):
                for file in files:
                    if self._is_document_file(file):
                        file_path = os.path.join(root, file)
                        doc_info = self._analyze_document(file_path, module_name)
                        if doc_info:
                            documents.append(doc_info)
                            
        except Exception as e:
            logger.error(f"收集模块 {module_name} 状态时出错: {e}")
            
        return documents
    
    def _is_document_file(self, filename: str) -> bool:
        """判断是否为文档文件"""
        doc_extensions = {'.md', '.txt', '.doc', '.docx', '.pdf', '.rst', '.html'}
        return any(filename.lower().endswith(ext) for ext in doc_extensions)
    
    def _analyze_document(self, file_path: str, module_name: str) -> Optional[DocumentInfo]:
        """分析单个文档"""
        try:
            stat = os.stat(file_path)
            checksum = self._calculate_checksum(file_path)
            
            # 确定文档状态
            status = self._determine_status(stat)
            
            # 获取文档元数据
            metadata = self._extract_metadata(file_path)
            
            doc_info = DocumentInfo(
                doc_id=self._generate_doc_id(file_path),
                name=os.path.basename(file_path),
                path=file_path,
                status=status,
                last_modified=stat.st_mtime,
                size=stat.st_size,
                checksum=checksum,
                module=module_name,
                version=metadata.get('version', '1.0'),
                metadata=metadata,
                created_at=stat.st_ctime,
                last_accessed=time.time(),
                access_count=0
            )
            
            return doc_info
            
        except Exception as e:
            logger.error(f"分析文档 {file_path} 时出错: {e}")
            return None
    
    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    def _determine_status(self, stat) -> DocumentStatus:
        """确定文档状态"""
        current_time = time.time()
        days_since_modified = (current_time - stat.st_mtime) / (24 * 3600)
        
        if days_since_modified > 365:
            return DocumentStatus.ARCHIVED
        elif days_since_modified > 90:
            return DocumentStatus.OUTDATED
        elif stat.st_size == 0:
            return DocumentStatus.CORRUPTED
        else:
            return DocumentStatus.ACTIVE
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取文档元数据"""
        metadata = {}
        
        try:
            if file_path.lower().endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 简单的元数据提取
                    lines = content.split('\n')
                    for line in lines[:20]:  # 只检查前20行
                        if line.startswith('#'):
                            metadata['title'] = line.lstrip('#').strip()
                        elif line.startswith('```yaml') or line.startswith('---'):
                            # 提取YAML front matter
                            break
        except:
            pass
            
        return metadata
    
    def _generate_doc_id(self, file_path: str) -> str:
        """生成文档唯一ID"""
        return hashlib.md5(file_path.encode()).hexdigest()


class DataAggregator:
    """数据聚合器"""
    
    def __init__(self):
        self.aggregation_rules = {
            'status_distribution': self._aggregate_status_distribution,
            'module_summary': self._aggregate_module_summary,
            'trend_analysis': self._analyze_trends,
            'quality_metrics': self._calculate_quality_metrics
        }
    
    def aggregate(self, documents: List[DocumentInfo]) -> Dict[str, Any]:
        """聚合文档数据"""
        aggregated_data = {}
        
        for rule_name, rule_func in self.aggregation_rules.items():
            try:
                aggregated_data[rule_name] = rule_func(documents)
            except Exception as e:
                logger.error(f"执行聚合规则 {rule_name} 时出错: {e}")
                aggregated_data[rule_name] = {}
        
        return aggregated_data
    
    def _aggregate_status_distribution(self, documents: List[DocumentInfo]) -> Dict[str, int]:
        """聚合状态分布"""
        distribution = defaultdict(int)
        for doc in documents:
            distribution[doc.status.value] += 1
        return dict(distribution)
    
    def _aggregate_module_summary(self, documents: List[DocumentInfo]) -> Dict[str, Dict[str, Any]]:
        """聚合模块摘要"""
        module_summary = defaultdict(lambda: {
            'total': 0, 'active': 0, 'outdated': 0, 'missing': 0, 'corrupted': 0
        })
        
        for doc in documents:
            summary = module_summary[doc.module]
            summary['total'] += 1
            if doc.status == DocumentStatus.ACTIVE:
                summary['active'] += 1
            elif doc.status == DocumentStatus.OUTDATED:
                summary['outdated'] += 1
            elif doc.status == DocumentStatus.MISSING:
                summary['missing'] += 1
            elif doc.status == DocumentStatus.CORRUPTED:
                summary['corrupted'] += 1
        
        return dict(module_summary)
    
    def _analyze_trends(self, documents: List[DocumentInfo]) -> Dict[str, Any]:
        """分析趋势"""
        # 简化的趋势分析
        current_time = time.time()
        recent_docs = [doc for doc in documents 
                      if current_time - doc.last_modified < 7 * 24 * 3600]  # 最近7天
        
        return {
            'recent_changes': len(recent_docs),
            'average_age_days': sum((current_time - doc.last_modified) / (24 * 3600) 
                                  for doc in documents) / max(len(documents), 1),
            'most_active_module': max(set(doc.module for doc in documents), 
                                    key=lambda x: sum(1 for d in documents if d.module == x))
        }
    
    def _calculate_quality_metrics(self, documents: List[DocumentInfo]) -> Dict[str, float]:
        """计算质量指标"""
        if not documents:
            return {}
        
        total = len(documents)
        active = sum(1 for doc in documents if doc.status == DocumentStatus.ACTIVE)
        corrupted = sum(1 for doc in documents if doc.status == DocumentStatus.CORRUPTED)
        outdated = sum(1 for doc in documents if doc.status == DocumentStatus.OUTDATED)
        
        return {
            'completeness': active / total,
            'corruption_rate': corrupted / total,
            'outdated_rate': outdated / total,
            'overall_health': (active - corrupted - outdated) / total
        }


class StatusAnalyzer:
    """状态分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get('thresholds', {})
    
    def analyze(self, aggregated_data: Dict[str, Any], 
                historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析文档状态"""
        analysis = {
            'health_score': self._calculate_health_score(aggregated_data),
            'risk_assessment': self._assess_risks(aggregated_data),
            'trends': self._analyze_historical_trends(historical_data),
            'recommendations': self._generate_recommendations(aggregated_data)
        }
        
        return analysis
    
    def _calculate_health_score(self, data: Dict[str, Any]) -> float:
        """计算健康分数"""
        quality_metrics = data.get('quality_metrics', {})
        
        if not quality_metrics:
            return 0.0
        
        completeness = quality_metrics.get('completeness', 0)
        corruption_rate = quality_metrics.get('corruption_rate', 0)
        outdated_rate = quality_metrics.get('outdated_rate', 0)
        
        # 健康分数计算 (0-100)
        health_score = (completeness * 100) - (corruption_rate * 100) - (outdated_rate * 50)
        return max(0, min(100, health_score))
    
    def _assess_risks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估风险"""
        risks = []
        
        quality_metrics = data.get('quality_metrics', {})
        status_dist = data.get('status_distribution', {})
        
        # 检查高风险指标
        if quality_metrics.get('corruption_rate', 0) > 0.1:
            risks.append({
                'type': 'high_corruption_rate',
                'level': 'high',
                'description': '文档损坏率过高',
                'value': quality_metrics.get('corruption_rate', 0)
            })
        
        if quality_metrics.get('outdated_rate', 0) > 0.3:
            risks.append({
                'type': 'high_outdated_rate',
                'level': 'medium',
                'description': '文档过期率较高',
                'value': quality_metrics.get('outdated_rate', 0)
            })
        
        if status_dist.get('missing', 0) > 0:
            risks.append({
                'type': 'missing_documents',
                'level': 'high',
                'description': f'发现 {status_dist.get("missing", 0)} 个缺失文档',
                'value': status_dist.get('missing', 0)
            })
        
        return {
            'risk_level': 'high' if any(r['level'] == 'high' for r in risks) else 'medium' if risks else 'low',
            'risks': risks
        }
    
    def _analyze_historical_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析历史趋势"""
        if len(historical_data) < 2:
            return {'trend': 'insufficient_data'}
        
        # 简化的趋势分析
        recent_health = [d.get('health_score', 0) for d in historical_data[-5:]]
        
        if len(recent_health) >= 2:
            if recent_health[-1] > recent_health[-2]:
                trend = 'improving'
            elif recent_health[-1] < recent_health[-2]:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
        
        return {
            'trend': trend,
            'health_progression': recent_health,
            'volatility': self._calculate_volatility(recent_health)
        }
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """计算波动性"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        quality_metrics = data.get('quality_metrics', {})
        status_dist = data.get('status_distribution', {})
        
        if quality_metrics.get('corruption_rate', 0) > 0.05:
            recommendations.append("建议检查和修复损坏的文档")
        
        if quality_metrics.get('outdated_rate', 0) > 0.2:
            recommendations.append("建议更新过期的文档内容")
        
        if status_dist.get('missing', 0) > 0:
            recommendations.append("建议找回或重新创建缺失的文档")
        
        if quality_metrics.get('completeness', 1) < 0.8:
            recommendations.append("建议完善文档覆盖范围")
        
        return recommendations


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, report_data: StatusReport) -> str:
        """生成状态报告"""
        timestamp = datetime.datetime.fromtimestamp(report_data.timestamp)
        report_file = os.path.join(self.output_dir, f"status_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json")
        
        report_dict = asdict(report_data)
        
        # 处理AlertLevel枚举
        if 'alerts' in report_dict:
            for alert in report_dict['alerts']:
                if 'level' in alert and hasattr(alert['level'], 'value'):
                    alert['level'] = alert['level'].value
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        # 生成HTML报告
        html_report = self._generate_html_report(report_data)
        html_file = os.path.join(self.output_dir, f"status_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html")
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"报告已生成: {report_file} 和 {html_file}")
        return report_file
    
    def _generate_html_report(self, report: StatusReport) -> str:
        """生成HTML格式报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>文档状态报告</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background-color: #e8f4fd; border-radius: 5px; text-align: center; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .alert-info {{ background-color: #d1ecf1; color: #0c5460; }}
                .alert-warning {{ background-color: #fff3cd; color: #856404; }}
                .alert-error {{ background-color: #f8d7da; color: #721c24; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>文档状态报告</h1>
                <p>生成时间: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>总体统计</h2>
                <div class="metric">
                    <h3>{total_documents}</h3>
                    <p>总文档数</p>
                </div>
                <div class="metric">
                    <h3>{active_documents}</h3>
                    <p>活跃文档</p>
                </div>
                <div class="metric">
                    <h3>{outdated_documents}</h3>
                    <p>过期文档</p>
                </div>
                <div class="metric">
                    <h3>{missing_documents}</h3>
                    <p>缺失文档</p>
                </div>
            </div>
            
            <div class="section">
                <h2>状态分布</h2>
                <table>
                    <tr><th>状态</th><th>数量</th><th>占比</th></tr>
                    {status_distribution_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>模块状态</h2>
                <table>
                    <tr><th>模块</th><th>总文档</th><th>活跃</th><th>过期</th><th>缺失</th></tr>
                    {module_status_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>预警信息</h2>
                {alerts_html}
            </div>
            
            <div class="section">
                <h2>建议</h2>
                <ul>
                    {recommendations_html}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # 填充模板数据
        timestamp_str = datetime.datetime.fromtimestamp(report.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        status_rows = []
        total = report.total_documents
        for status, count in report.status_distribution.items():
            percentage = (count / total * 100) if total > 0 else 0
            status_rows.append(f"<tr><td>{status}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>")
        
        module_rows = []
        for module, status in report.module_status.items():
            module_rows.append(f"<tr><td>{module}</td><td>{status.get('total', 0)}</td>"
                             f"<td>{status.get('active', 0)}</td><td>{status.get('outdated', 0)}</td>"
                             f"<td>{status.get('missing', 0)}</td></tr>")
        
        alerts_html = ""
        for alert in report.alerts:
            level_class = f"alert-{alert.get('level', 'info')}"
            alerts_html += f'<div class="alert {level_class}">{alert.get("message", "")}</div>'
        
        recommendations_html = "".join(f"<li>{rec}</li>" for rec in getattr(report, 'recommendations', []))
        
        return html_template.format(
            timestamp=timestamp_str,
            total_documents=report.total_documents,
            active_documents=report.active_documents,
            outdated_documents=report.outdated_documents,
            missing_documents=report.missing_documents,
            status_distribution_rows="".join(status_rows),
            module_status_rows="".join(module_rows),
            alerts_html=alerts_html,
            recommendations_html=recommendations_html
        )


class AlertManager:
    """预警管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = config.get('alert_rules', {})
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
    
    def check_alerts(self, report_data: StatusReport) -> List[Alert]:
        """检查并生成预警"""
        alerts = []
        
        # 检查各种预警条件
        alerts.extend(self._check_missing_documents(report_data))
        alerts.extend(self._check_corrupted_documents(report_data))
        alerts.extend(self._check_outdated_documents(report_data))
        alerts.extend(self._check_health_threshold(report_data))
        
        # 更新活跃预警
        for alert in alerts:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
        
        return alerts
    
    def _check_missing_documents(self, report: StatusReport) -> List[Alert]:
        """检查缺失文档"""
        alerts = []
        if report.missing_documents > 0:
            alert = Alert(
                alert_id=f"missing_docs_{int(time.time())}",
                level=AlertLevel.ERROR,
                title="发现缺失文档",
                message=f"发现 {report.missing_documents} 个缺失文档",
                affected_documents=[],
                timestamp=time.time()
            )
            alerts.append(alert)
        return alerts
    
    def _check_corrupted_documents(self, report: StatusReport) -> List[Alert]:
        """检查损坏文档"""
        alerts = []
        if report.corrupted_documents > 0:
            alert = Alert(
                alert_id=f"corrupted_docs_{int(time.time())}",
                level=AlertLevel.ERROR,
                title="发现损坏文档",
                message=f"发现 {report.corrupted_documents} 个损坏文档",
                affected_documents=[],
                timestamp=time.time()
            )
            alerts.append(alert)
        return alerts
    
    def _check_outdated_documents(self, report: StatusReport) -> List[Alert]:
        """检查过期文档"""
        alerts = []
        if report.outdated_documents > 5:  # 超过5个过期文档时预警
            alert = Alert(
                alert_id=f"outdated_docs_{int(time.time())}",
                level=AlertLevel.WARNING,
                title="过期文档过多",
                message=f"发现 {report.outdated_documents} 个过期文档，建议更新",
                affected_documents=[],
                timestamp=time.time()
            )
            alerts.append(alert)
        return alerts
    
    def _check_health_threshold(self, report: StatusReport) -> List[Alert]:
        """检查健康阈值"""
        alerts = []
        
        # 计算健康分数
        if report.total_documents > 0:
            health_score = (report.active_documents / report.total_documents) * 100
            if health_score < 50:
                alert = Alert(
                    alert_id=f"low_health_{int(time.time())}",
                    level=AlertLevel.CRITICAL,
                    title="文档健康状况不佳",
                    message=f"文档健康分数仅为 {health_score:.1f}%，需要立即关注",
                    affected_documents=[],
                    timestamp=time.time()
                )
                alerts.append(alert)
        
        return alerts


class HistoryManager:
    """历史记录管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS document_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    doc_id TEXT,
                    status TEXT,
                    module TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS report_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    report_data TEXT
                )
            ''')
    
    def save_document_history(self, documents: List[DocumentInfo]):
        """保存文档历史记录"""
        with sqlite3.connect(self.db_path) as conn:
            for doc in documents:
                conn.execute('''
                    INSERT INTO document_history 
                    (timestamp, doc_id, status, module, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (time.time(), doc.doc_id, doc.status.value, doc.module, 
                      json.dumps(doc.metadata)))
    
    def save_report_history(self, report: StatusReport):
        """保存报告历史记录"""
        # 转换枚举值为字符串以便JSON序列化
        report_dict = asdict(report)
        
        # 处理AlertLevel枚举
        if 'alerts' in report_dict:
            for alert in report_dict['alerts']:
                if 'level' in alert and hasattr(alert['level'], 'value'):
                    alert['level'] = alert['level'].value
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO report_history (timestamp, report_data)
                VALUES (?, ?)
            ''', (report.timestamp, json.dumps(report_dict)))
    
    def get_historical_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取历史数据"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT timestamp, report_data FROM report_history
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            historical_data = []
            for row in cursor.fetchall():
                try:
                    report_data = json.loads(row[1])
                    historical_data.append(report_data)
                except:
                    continue
            
            return historical_data


class DashboardGenerator:
    """仪表板生成器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_dashboard(self, current_data: StatusReport, 
                          historical_data: List[Dict[str, Any]]) -> str:
        """生成可视化仪表板"""
        dashboard_file = os.path.join(self.output_dir, "dashboard.html")
        
        dashboard_html = self._create_dashboard_html(current_data, historical_data)
        
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        logger.info(f"仪表板已生成: {dashboard_file}")
        return dashboard_file
    
    def _create_dashboard_html(self, current_data: StatusReport, 
                              historical_data: List[Dict[str, Any]]) -> str:
        """创建仪表板HTML"""
        # 简化的仪表板HTML（实际应用中可以使用Chart.js等库）
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>文档状态仪表板</title>
            <meta charset="UTF-8">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                               gap: 20px; margin-bottom: 30px; }}
                .metric-card {{ background: white; padding: 25px; border-radius: 10px; 
                              box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
                .metric-value {{ font-size: 2.5em; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ color: #666; font-size: 0.9em; }}
                .chart-container {{ background: white; padding: 25px; border-radius: 10px; 
                                  box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
                .status-good {{ color: #28a745; }}
                .status-warning {{ color: #ffc107; }}
                .status-error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>📊 文档状态仪表板</h1>
                    <p>实时监控文档健康状况 - {timestamp}</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{total_docs}</div>
                        <div class="metric-label">总文档数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value status-good">{active_docs}</div>
                        <div class="metric-label">活跃文档</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value status-warning">{outdated_docs}</div>
                        <div class="metric-label">过期文档</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value status-error">{missing_docs}</div>
                        <div class="metric-label">缺失文档</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>文档状态分布</h3>
                    <canvas id="statusChart" width="400" height="200"></canvas>
                </div>
                
                <div class="chart-container">
                    <h3>模块状态概览</h3>
                    <canvas id="moduleChart" width="400" height="200"></canvas>
                </div>
            </div>
            
            <script>
                // 状态分布图表
                const statusCtx = document.getElementById('statusChart').getContext('2d');
                new Chart(statusCtx, {{
                    type: 'doughnut',
                    data: {{
                        labels: {status_labels},
                        datasets: [{{
                            data: {status_data},
                            backgroundColor: [
                                '#28a745', '#ffc107', '#dc3545', '#6c757d', '#17a2b8'
                            ]
                        }}]
                    }}
                }});
                
                // 模块状态图表
                const moduleCtx = document.getElementById('moduleChart').getContext('2d');
                new Chart(moduleCtx, {{
                    type: 'bar',
                    data: {{
                        labels: {module_labels},
                        datasets: [{{
                            label: '活跃文档',
                            data: {module_active_data},
                            backgroundColor: '#28a745'
                        }}, {{
                            label: '过期文档',
                            data: {module_outdated_data},
                            backgroundColor: '#ffc107'
                        }}]
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        # 准备数据
        timestamp = datetime.datetime.fromtimestamp(current_data.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        status_labels = list(current_data.status_distribution.keys())
        status_data = list(current_data.status_distribution.values())
        
        module_labels = list(current_data.module_status.keys())
        module_active_data = [status.get('active', 0) for status in current_data.module_status.values()]
        module_outdated_data = [status.get('outdated', 0) for status in current_data.module_status.values()]
        
        return html_template.format(
            timestamp=timestamp,
            total_docs=current_data.total_documents,
            active_docs=current_data.active_documents,
            outdated_docs=current_data.outdated_documents,
            missing_docs=current_data.missing_documents,
            status_labels=status_labels,
            status_data=status_data,
            module_labels=module_labels,
            module_active_data=module_active_data,
            module_outdated_data=module_outdated_data
        )


class DocumentStatusAggregator:
    """文档状态聚合器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collector = StatusCollector(config)
        self.aggregator = DataAggregator()
        self.analyzer = StatusAnalyzer(config)
        self.report_generator = ReportGenerator(config.get('output_dir', './reports'))
        self.alert_manager = AlertManager(config)
        self.history_manager = HistoryManager(config.get('history_db', './history.db'))
        self.dashboard_generator = DashboardGenerator(config.get('dashboard_dir', './dashboard'))
        
        self.monitoring_active = False
        self.monitor_thread = None
        
    def collect_status(self) -> List[DocumentInfo]:
        """收集所有文档状态"""
        all_documents = []
        
        for module_name, module_path in self.config.get('document_paths', {}).items():
            if os.path.exists(module_path):
                documents = self.collector.collect_document_status(module_name, module_path)
                all_documents.extend(documents)
            else:
                logger.warning(f"模块路径不存在: {module_path}")
        
        return all_documents
    
    def generate_status_report(self) -> StatusReport:
        """生成状态报告"""
        # 收集文档状态
        documents = self.collect_status()
        
        # 数据聚合
        aggregated_data = self.aggregator.aggregate(documents)
        
        # 获取历史数据
        historical_data = self.history_manager.get_historical_data()
        
        # 状态分析
        analysis = self.analyzer.analyze(aggregated_data, historical_data)
        
        # 生成报告
        status_dist = aggregated_data.get('status_distribution', {})
        module_status = aggregated_data.get('module_summary', {})
        
        report = StatusReport(
            timestamp=time.time(),
            total_documents=len(documents),
            active_documents=status_dist.get('active', 0),
            outdated_documents=status_dist.get('outdated', 0),
            missing_documents=status_dist.get('missing', 0),
            corrupted_documents=status_dist.get('corrupted', 0),
            status_distribution=status_dist,
            module_status=module_status,
            trends=aggregated_data.get('trend_analysis', {}),
            alerts=[],  # 将在下一步填充
            recommendations=analysis.get('recommendations', [])
        )
        
        # 检查预警
        alerts = self.alert_manager.check_alerts(report)
        report.alerts = [asdict(alert) for alert in alerts]
        
        # 保存历史记录
        self.history_manager.save_document_history(documents)
        self.history_manager.save_report_history(report)
        
        return report
    
    def generate_reports(self) -> Dict[str, str]:
        """生成所有报告"""
        report = self.generate_status_report()
        
        # 生成JSON报告
        json_report_file = self.report_generator.generate_report(report)
        
        # 生成仪表板
        historical_data = self.history_manager.get_historical_data()
        dashboard_file = self.dashboard_generator.generate_dashboard(report, historical_data)
        
        return {
            'json_report': json_report_file,
            'dashboard': dashboard_file
        }
    
    def start_monitoring(self, interval: int = 300):
        """启动实时监控"""
        if self.monitoring_active:
            logger.warning("监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"文档状态监控已启动，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("文档状态监控已停止")
    
    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                report = self.generate_status_report()
                
                # 检查是否有严重预警
                critical_alerts = [alert for alert in report.alerts 
                                 if alert.get('level') == AlertLevel.CRITICAL.value]
                
                if critical_alerts:
                    logger.critical(f"发现严重预警: {critical_alerts}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"监控过程中出错: {e}")
                time.sleep(interval)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要"""
        report = self.generate_status_report()
        
        # 计算健康分数
        if report.total_documents > 0:
            health_score = (report.active_documents / report.total_documents) * 100
        else:
            health_score = 0
        
        # 确定健康状态
        if health_score >= 80:
            health_status = "优秀"
        elif health_score >= 60:
            health_status = "良好"
        elif health_score >= 40:
            health_status = "一般"
        else:
            health_status = "需要关注"
        
        return {
            'health_score': health_score,
            'health_status': health_status,
            'total_documents': report.total_documents,
            'active_documents': report.active_documents,
            'issues_count': report.missing_documents + report.corrupted_documents + report.outdated_documents,
            'last_updated': report.timestamp
        }


# 使用示例和配置
def create_sample_config() -> Dict[str, Any]:
    """创建示例配置"""
    return {
        'document_paths': {
            'core': '/workspace/D/AO/AOO/Q/Q9/docs/core',
            'api': '/workspace/D/AO/AOO/Q/Q9/docs/api',
            'user_guide': '/workspace/D/AO/AOO/Q/Q9/docs/user_guide'
        },
        'scan_interval': 300,
        'output_dir': '/workspace/D/AO/AOO/Q/Q9/reports',
        'dashboard_dir': '/workspace/D/AO/AOO/Q/Q9/dashboard',
        'history_db': '/workspace/D/AO/AOO/Q/Q9/history.db',
        'thresholds': {
            'min_health_score': 60,
            'max_corruption_rate': 0.05,
            'max_outdated_rate': 0.3
        },
        'alert_rules': {
            'missing_docs_threshold': 0,
            'corrupted_docs_threshold': 0,
            'outdated_docs_threshold': 5
        }
    }


if __name__ == "__main__":
    # 示例使用
    config = create_sample_config()
    aggregator = DocumentStatusAggregator(config)
    
    # 生成报告
    reports = aggregator.generate_reports()
    print("报告生成完成:")
    for report_type, file_path in reports.items():
        print(f"  {report_type}: {file_path}")
    
    # 获取健康摘要
    health = aggregator.get_health_summary()
    print(f"\n文档健康状况: {health['health_status']} (分数: {health['health_score']:.1f})")