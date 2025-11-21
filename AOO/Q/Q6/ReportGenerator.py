"""
Q6报告生成器主模块

提供完整的数据处理、模板渲染、图表生成和报告输出功能。
"""

import os
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import schedule
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import jinja2
from pathlib import Path
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataProcessor:
    """数据处理器 - 负责数据清洗、转换和聚合"""
    
    def __init__(self):
        self.data_sources = {}
        self.processed_data = {}
    
    def load_data(self, source_type, source_path, **kwargs):
        """
        加载数据源
        
        Args:
            source_type: 数据源类型 ('csv', 'excel', 'json', 'database')
            source_path: 数据源路径
            **kwargs: 额外参数
        """
        try:
            if source_type == 'csv':
                data = pd.read_csv(source_path, **kwargs)
            elif source_type == 'excel':
                data = pd.read_excel(source_path, **kwargs)
            elif source_type == 'json':
                with open(source_path, 'r', encoding='utf-8') as f:
                    data = pd.DataFrame(json.load(f))
            else:
                raise ValueError(f"不支持的数据源类型: {source_type}")
            
            self.data_sources[source_path] = data
            return data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def clean_data(self, data, cleaning_rules=None):
        """
        数据清洗
        
        Args:
            data: 原始数据
            cleaning_rules: 清洗规则字典
        """
        if cleaning_rules is None:
            cleaning_rules = {
                'remove_duplicates': True,
                'handle_missing': 'drop',  # 'drop', 'fill_mean', 'fill_median'
                'remove_outliers': False,
                'normalize': False
            }
        
        cleaned_data = data.copy()
        
        # 移除重复值
        if cleaning_rules.get('remove_duplicates', True):
            cleaned_data = cleaned_data.drop_duplicates()
        
        # 处理缺失值
        missing_strategy = cleaning_rules.get('handle_missing', 'drop')
        if missing_strategy == 'drop':
            cleaned_data = cleaned_data.dropna()
        elif missing_strategy == 'fill_mean':
            cleaned_data = cleaned_data.fillna(cleaned_data.mean(numeric_only=True))
        elif missing_strategy == 'fill_median':
            cleaned_data = cleaned_data.fillna(cleaned_data.median(numeric_only=True))
        
        # 移除异常值
        if cleaning_rules.get('remove_outliers', False):
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_data = cleaned_data[
                    (cleaned_data[col] >= lower_bound) & 
                    (cleaned_data[col] <= upper_bound)
                ]
        
        # 数据标准化
        if cleaning_rules.get('normalize', False):
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[numeric_columns] = (
                cleaned_data[numeric_columns] - cleaned_data[numeric_columns].min()
            ) / (cleaned_data[numeric_columns].max() - cleaned_data[numeric_columns].min())
        
        return cleaned_data
    
    def aggregate_data(self, data, group_by, aggregations):
        """
        数据聚合
        
        Args:
            data: 数据
            group_by: 分组字段
            aggregations: 聚合函数字典 {'字段名': '函数名'}
        """
        return data.groupby(group_by).agg(aggregations).reset_index()
    
    def transform_data(self, data, transformations):
        """
        数据转换
        
        Args:
            data: 数据
            transformations: 转换规则字典
        """
        transformed_data = data.copy()
        
        for column, transform_type in transformations.items():
            if transform_type == 'log':
                transformed_data[column] = np.log(transformed_data[column] + 1)
            elif transform_type == 'sqrt':
                transformed_data[column] = np.sqrt(transformed_data[column])
            elif transform_type == 'normalize':
                transformed_data[column] = (
                    transformed_data[column] - transformed_data[column].min()
                ) / (transformed_data[column].max() - transformed_data[column].min())
            elif transform_type == 'standardize':
                transformed_data[column] = (
                    transformed_data[column] - transformed_data[column].mean()
                ) / transformed_data[column].std()
        
        return transformed_data


class TemplateManager:
    """模板管理器 - 管理报告模板"""
    
    def __init__(self, template_dir="templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """加载默认模板"""
        # 创建默认模板文件
        default_templates = {
            'basic_report.html': """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
        .content { margin: 20px 0; }
        .chart { text-align: center; margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>生成时间: {{ generate_time }}</p>
    </div>
    <div class="content">
        {{ content }}
    </div>
</body>
</html>
            """,
            'executive_summary.html': """
<!DOCTYPE html>
<html>
<head>
    <title>执行摘要 - {{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }
        .summary { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .kpi { display: inline-block; margin: 10px; padding: 20px; background-color: #007bff; color: white; border-radius: 5px; text-align: center; }
        .insights { margin: 20px 0; }
        .chart { text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="summary">
        <h1>执行摘要</h1>
        <h2>{{ title }}</h2>
        <div class="kpis">
            {{ kpis }}
        </div>
        <div class="insights">
            {{ insights }}
        </div>
        <div class="charts">
            {{ charts }}
        </div>
    </div>
</body>
</html>
            """
        }
        
        for template_name, template_content in default_templates.items():
            template_path = self.template_dir / template_name
            if not template_path.exists():
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(template_content)
    
    def add_template(self, name, template_content):
        """添加自定义模板"""
        template_path = self.template_dir / f"{name}.html"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
    
    def get_template(self, name):
        """获取模板"""
        try:
            return self.jinja_env.get_template(f"{name}.html")
        except:
            return None
    
    def render_template(self, template_name, **kwargs):
        """渲染模板"""
        template = self.get_template(template_name)
        if template:
            return template.render(**kwargs)
        return None


class ChartGenerator:
    """图表生成器 - 生成各种类型的图表"""
    
    def __init__(self, output_dir="charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.chart_styles = ['default', 'seaborn', 'ggplot', 'bmh']
        
    def _save_chart(self, fig, filename):
        """保存图表"""
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return str(filepath)
    
    def _base64_encode_chart(self, fig):
        """将图表编码为base64字符串"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64
    
    def create_bar_chart(self, data, x_col, y_col, title="柱状图", 
                        filename="bar_chart.png", to_base64=False):
        """创建柱状图"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(data[x_col], data[y_col], color='skyblue', alpha=0.8)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if to_base64:
            return self._base64_encode_chart(fig)
        else:
            return self._save_chart(fig, filename)
    
    def create_line_chart(self, data, x_col, y_col, title="线图", 
                         filename="line_chart.png", to_base64=False):
        """创建线图"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(data[x_col], data[y_col], marker='o', linewidth=2, markersize=6)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if to_base64:
            return self._base64_encode_chart(fig)
        else:
            return self._save_chart(fig, filename)
    
    def create_pie_chart(self, data, label_col, value_col, title="饼图", 
                        filename="pie_chart.png", to_base64=False):
        """创建饼图"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(8, 8))
        
        wedges, texts, autotexts = ax.pie(
            data[value_col], 
            labels=data[label_col], 
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(data)))
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if to_base64:
            return self._base64_encode_chart(fig)
        else:
            return self._save_chart(fig, filename)
    
    def create_scatter_plot(self, data, x_col, y_col, title="散点图", 
                           filename="scatter_plot.png", to_base64=False):
        """创建散点图"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(data[x_col], data[y_col], alpha=0.6, s=50)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if to_base64:
            return self._base64_encode_chart(fig)
        else:
            return self._save_chart(fig, filename)
    
    def create_heatmap(self, data, title="热力图", filename="heatmap.png", 
                      to_base64=False):
        """创建热力图"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if to_base64:
            return self._base64_encode_chart(fig)
        else:
            return self._save_chart(fig, filename)
    
    def create_dashboard(self, data, chart_configs, title="仪表板", 
                        filename="dashboard.png", to_base64=False):
        """创建仪表板"""
        n_charts = len(chart_configs)
        cols = min(2, n_charts)
        rows = (n_charts + cols - 1) // cols
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_charts == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten() if n_charts > 1 else axes
        
        for i, config in enumerate(chart_configs):
            ax = axes[i]
            chart_type = config['type']
            
            if chart_type == 'bar':
                ax.bar(data[config['x']], data[config['y']], color='skyblue', alpha=0.8)
            elif chart_type == 'line':
                ax.plot(data[config['x']], data[config['y']], marker='o')
            elif chart_type == 'scatter':
                ax.scatter(data[config['x']], data[config['y']], alpha=0.6)
            
            ax.set_title(config.get('title', f'图表 {i+1}'))
            ax.tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for i in range(n_charts, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if to_base64:
            return self._base64_encode_chart(fig)
        else:
            return self._save_chart(fig, filename)


class ReportScheduler:
    """报告调度器 - 定时生成和发送报告"""
    
    def __init__(self, report_generator):
        self.report_generator = report_generator
        self.scheduled_reports = {}
    
    def schedule_report(self, name, schedule_type, report_config, email_config=None):
        """
        调度报告
        
        Args:
            name: 报告名称
            schedule_type: 调度类型 ('daily', 'weekly', 'monthly')
            report_config: 报告配置
            email_config: 邮件配置
        """
        self.scheduled_reports[name] = {
            'schedule_type': schedule_type,
            'report_config': report_config,
            'email_config': email_config
        }
    
    def _run_scheduled_report(self, name):
        """运行调度的报告"""
        config = self.scheduled_reports[name]
        print(f"正在生成定时报告: {name}")
        
        try:
            # 生成报告
            report_path = self.report_generator.generate_report(**config['report_config'])
            
            # 发送邮件
            if config['email_config']:
                self._send_email_report(report_path, config['email_config'])
            
            print(f"定时报告 {name} 生成完成: {report_path}")
        except Exception as e:
            print(f"定时报告 {name} 生成失败: {e}")
    
    def _send_email_report(self, report_path, email_config):
        """发送邮件报告"""
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['sender']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = email_config['subject']
            
            # 添加邮件正文
            body = email_config.get('body', '请查收附件中的报告。')
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 添加附件
            if os.path.exists(report_path):
                with open(report_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(report_path)}',
                )
                msg.attach(part)
            
            # 发送邮件
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"邮件发送成功: {email_config['recipients']}")
        except Exception as e:
            print(f"邮件发送失败: {e}")
    
    def start_scheduler(self):
        """启动调度器"""
        for name, config in self.scheduled_reports.items():
            schedule_type = config['schedule_type']
            if schedule_type == 'daily':
                schedule.every().day.at("09:00").do(self._run_scheduled_report, name)
            elif schedule_type == 'weekly':
                schedule.every().monday.at("09:00").do(self._run_scheduled_report, name)
            elif schedule_type == 'monthly':
                schedule.every().month.do(self._run_scheduled_report, name)
        
        print("报告调度器已启动")
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次


class ReportGenerator:
    """主报告生成器类"""
    
    def __init__(self, output_dir="reports", template_dir="templates", chart_dir="charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_processor = DataProcessor()
        self.template_manager = TemplateManager(template_dir)
        self.chart_generator = ChartGenerator(chart_dir)
        self.scheduler = ReportScheduler(self)
        
        # 报告配置
        self.report_configs = {}
    
    def configure_report(self, name, config):
        """配置报告模板"""
        self.report_configs[name] = config
    
    def generate_data_summary(self, data):
        """生成数据摘要"""
        summary = {
            'total_records': len(data),
            'columns': list(data.columns),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object']).columns),
            'missing_values': data.isnull().sum().to_dict(),
            'basic_stats': data.describe().to_dict() if not data.empty else {}
        }
        return summary
    
    def create_standard_charts(self, data, chart_configs=None):
        """创建标准图表"""
        if chart_configs is None:
            chart_configs = []
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # 为数值列创建直方图
            for col in numeric_cols[:3]:  # 最多3个数值列
                chart_configs.append({
                    'type': 'histogram',
                    'column': col,
                    'title': f'{col}分布图'
                })
            
            # 为分类列创建饼图
            for col in categorical_cols[:2]:  # 最多2个分类列
                if data[col].nunique() <= 10:  # 只处理类别数不超过10的列
                    chart_configs.append({
                        'type': 'pie',
                        'column': col,
                        'title': f'{col}分布饼图'
                    })
        
        generated_charts = []
        for config in chart_configs:
            try:
                chart_type = config['type']
                if chart_type == 'histogram':
                    plt.figure(figsize=(10, 6))
                    data[config['column']].hist(bins=20, alpha=0.7)
                    plt.title(config['title'])
                    plt.xlabel(config['column'])
                    plt.ylabel('频次')
                    chart_path = self.chart_generator._save_chart(
                        plt.gcf(), f"{config['column']}_histogram.png"
                    )
                    generated_charts.append({
                        'type': 'histogram',
                        'path': chart_path,
                        'column': config['column'],
                        'title': config['title']
                    })
                
                elif chart_type == 'pie':
                    value_counts = data[config['column']].value_counts().head(10)
                    chart_base64 = self.chart_generator.create_pie_chart(
                        pd.DataFrame({
                            'label': value_counts.index,
                            'value': value_counts.values
                        }),
                        'label', 'value', config['title'], to_base64=True
                    )
                    generated_charts.append({
                        'type': 'pie',
                        'base64': chart_base64,
                        'column': config['column'],
                        'title': config['title']
                    })
                
            except Exception as e:
                print(f"图表生成失败 {config}: {e}")
        
        return generated_charts
    
    def generate_report(self, data=None, data_source=None, template_name="basic_report", 
                       output_format="html", title="数据报告", 
                       include_charts=True, chart_configs=None, **kwargs):
        """
        生成报告
        
        Args:
            data: 数据DataFrame
            data_source: 数据源路径
            template_name: 模板名称
            output_format: 输出格式 ('html', 'pdf', 'excel')
            title: 报告标题
            include_charts: 是否包含图表
            chart_configs: 图表配置
            **kwargs: 额外参数
        """
        try:
            # 加载数据
            if data is None and data_source:
                data = self.data_processor.load_data(
                    kwargs.get('source_type', 'csv'), data_source,
                    **kwargs.get('load_params', {})
                )
            
            if data is None or data.empty:
                raise ValueError("没有有效的数据用于生成报告")
            
            # 数据处理
            cleaned_data = self.data_processor.clean_data(
                data, kwargs.get('cleaning_rules', {})
            )
            
            # 生成数据摘要
            data_summary = self.generate_data_summary(cleaned_data)
            
            # 生成图表
            charts = []
            if include_charts:
                charts = self.create_standard_charts(cleaned_data, chart_configs)
            
            # 渲染模板
            template_data = {
                'title': title,
                'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_summary': data_summary,
                'charts': charts,
                'data_table': cleaned_data.head(10).to_html(classes='table table-striped') if output_format == 'html' else None,
                **kwargs
            }
            
            # 添加内容
            content_parts = []
            
            # 数据摘要部分
            content_parts.append("<h2>数据摘要</h2>")
            content_parts.append(f"<p>总记录数: {data_summary['total_records']}</p>")
            content_parts.append(f"<p>列数: {len(data_summary['columns'])}</p>")
            
            # 图表部分
            if charts:
                content_parts.append("<h2>数据可视化</h2>")
                for chart in charts:
                    if chart['type'] == 'histogram':
                        content_parts.append(f"<div class='chart'><img src='{chart['path']}' alt='{chart['title']}'></div>")
                    elif chart['type'] == 'pie':
                        content_parts.append(f"<div class='chart'><img src='data:image/png;base64,{chart['base64']}' alt='{chart['title']}'></div>")
            
            # 数据表部分
            if output_format == 'html' and 'data_table' in template_data:
                content_parts.append("<h2>数据预览</h2>")
                content_parts.append(template_data['data_table'])
            
            template_data['content'] = '\n'.join(content_parts)
            
            # 生成报告内容
            report_content = self.template_manager.render_template(template_name, **template_data)
            
            if not report_content:
                raise ValueError(f"模板渲染失败: {template_name}")
            
            # 保存报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if output_format == 'html':
                filename = f"{title}_{timestamp}.html"
                filepath = self.output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                return str(filepath)
            
            elif output_format == 'excel':
                filename = f"{title}_{timestamp}.xlsx"
                filepath = self.output_dir / filename
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    cleaned_data.to_excel(writer, sheet_name='数据', index=False)
                    # 保存图表到单独的工作表
                    if charts:
                        chart_df = pd.DataFrame([
                            {'图表类型': c['type'], '图表标题': c['title'], '文件路径': c.get('path', '')}
                            for c in charts
                        ])
                        chart_df.to_excel(writer, sheet_name='图表信息', index=False)
                return str(filepath)
            
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")
        
        except Exception as e:
            print(f"报告生成失败: {e}")
            return None
    
    def create_interactive_report(self, data, title="交互式报告", output_file=None):
        """创建交互式HTML报告"""
        try:
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = self.output_dir / f"interactive_{title}_{timestamp}.html"
            
            # 生成交互式HTML
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .chart-container {{ margin: 20px 0; }}
        .controls {{ margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }}
        select, button {{ margin: 5px; padding: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="controls">
            <label>选择图表类型: </label>
            <select id="chartType">
                <option value="scatter">散点图</option>
                <option value="line">线图</option>
                <option value="bar">柱状图</option>
                <option value="histogram">直方图</option>
            </select>
            <label>X轴: </label>
            <select id="xAxis"></select>
            <label>Y轴: </label>
            <select id="yAxis"></select>
            <button onclick="updateChart()">更新图表</button>
        </div>
        <div id="chart" class="chart-container"></div>
        <div id="dataTable"></div>
    </div>
    
    <script>
        const data = {data.to_json(orient='records')};
        const columns = {list(data.columns)};
        
        // 初始化选择框
        const xSelect = document.getElementById('xAxis');
        const ySelect = document.getElementById('yAxis');
        
        columns.forEach(col => {{
            xSelect.innerHTML += `<option value="${{col}}">${{col}}</option>`;
            ySelect.innerHTML += `<option value="${{col}}">${{col}}</option>`;
        }});
        
        function updateChart() {{
            const chartType = document.getElementById('chartType').value;
            const xAxis = document.getElementById('xAxis').value;
            const yAxis = document.getElementById('yAxis').value;
            
            let trace;
            if (chartType === 'histogram') {{
                trace = {{
                    x: data.map(d => d[xAxis]),
                    type: 'histogram'
                }};
            }} else {{
                trace = {{
                    x: data.map(d => d[xAxis]),
                    y: data.map(d => d[yAxis]),
                    type: chartType
                }};
            }}
            
            const layout = {{
                title: `${{chartType}} - ${{xAxis}} vs ${{yAxis}}`,
                xaxis: {{ title: xAxis }},
                yaxis: {{ title: yAxis }}
            }};
            
            Plotly.newPlot('chart', [trace], layout);
        }}
        
        // 初始图表
        updateChart();
        
        // 显示数据表
        const tableHTML = '<table border="1">' +
            '<thead><tr>' + columns.map(col => `<th>${{col}}</th>`).join('') + '</tr></thead>' +
            '<tbody>' +
            data.slice(0, 20).map(row => '<tr>' + columns.map(col => `<td>${{row[col]}}</td>`).join('') + '</tr>').join('') +
            '</tbody></table>';
        document.getElementById('dataTable').innerHTML = tableHTML;
    </script>
</body>
</html>
            """
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(output_file)
        
        except Exception as e:
            print(f"交互式报告创建失败: {e}")
            return None
    
    def batch_generate_reports(self, report_configs):
        """批量生成报告"""
        results = []
        for config in report_configs:
            try:
                result = self.generate_report(**config)
                results.append({
                    'config': config,
                    'result': result,
                    'status': 'success' if result else 'failed'
                })
            except Exception as e:
                results.append({
                    'config': config,
                    'result': str(e),
                    'status': 'error'
                })
        return results
    
    def export_to_multiple_formats(self, data, base_filename, formats=['html', 'excel'], **kwargs):
        """导出多种格式"""
        results = {}
        for fmt in formats:
            try:
                result = self.generate_report(
                    data=data,
                    title=base_filename,
                    output_format=fmt,
                    **kwargs
                )
                results[fmt] = result
            except Exception as e:
                results[fmt] = f"导出失败: {e}"
        return results


# 示例使用函数
def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        '日期': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        '销售额': np.random.normal(10000, 2000, n_samples),
        '订单数': np.random.poisson(50, n_samples),
        '地区': np.random.choice(['北京', '上海', '广州', '深圳'], n_samples),
        '产品类别': np.random.choice(['电子产品', '服装', '食品', '图书'], n_samples),
        '客户满意度': np.random.uniform(1, 5, n_samples)
    })
    
    # 添加一些趋势
    data['销售额'] += np.arange(n_samples) * 10
    data['客户满意度'] += np.random.normal(0, 0.2, n_samples)
    data['客户满意度'] = np.clip(data['客户满意度'], 1, 5)
    
    return data


if __name__ == "__main__":
    # 示例使用
    generator = ReportGenerator()
    
    # 创建示例数据
    sample_data = create_sample_data()
    
    # 生成HTML报告
    html_report = generator.generate_report(
        data=sample_data,
        title="销售数据分析报告",
        template_name="basic_report",
        output_format="html"
    )
    print(f"HTML报告已生成: {html_report}")
    
    # 生成Excel报告
    excel_report = generator.generate_report(
        data=sample_data,
        title="销售数据分析报告",
        output_format="excel"
    )
    print(f"Excel报告已生成: {excel_report}")
    
    # 生成交互式报告
    interactive_report = generator.create_interactive_report(
        data=sample_data,
        title="销售数据交互式报告"
    )
    print(f"交互式报告已生成: {interactive_report}")