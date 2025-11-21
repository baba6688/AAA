"""
Q4策略文档生成器

一个用于生成策略文档的完整解决方案，支持生成策略描述、参数说明、风险分析、
绩效分析、使用示例、适用场景、优化建议和版本管理等功能。
"""

import json
import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class StrategyParameter:
    """策略参数定义"""
    name: str
    description: str
    type: str
    default_value: Any
    range: Optional[str] = None
    required: bool = True
    notes: Optional[str] = None


@dataclass
class RiskFactor:
    """风险因素定义"""
    factor: str
    description: str
    impact_level: str  # 低、中、高
    probability: str  # 低、中、高
    mitigation: Optional[str] = None


@dataclass
class PerformanceMetric:
    """绩效指标定义"""
    metric: str
    value: Union[str, float]
    period: str
    benchmark: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class UsageExample:
    """使用示例定义"""
    title: str
    description: str
    code: str
    expected_result: str


@dataclass
class VersionInfo:
    """版本信息定义"""
    version: str
    date: str
    changes: List[str]
    author: str


class StrategyDocGenerator:
    """策略文档生成器主类"""
    
    def __init__(self, strategy_name: str, strategy_type: str = "通用"):
        """
        初始化策略文档生成器
        
        Args:
            strategy_name: 策略名称
            strategy_type: 策略类型
        """
        self.strategy_name = strategy_name
        self.strategy_type = strategy_type
        self.creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 策略基本信息
        self.description = ""
        self.principle = ""
        self.logic = ""
        
        # 策略参数
        self.parameters: List[StrategyParameter] = []
        
        # 风险分析
        self.risks: List[RiskFactor] = []
        
        # 绩效分析
        self.performance: List[PerformanceMetric] = []
        
        # 使用示例
        self.examples: List[UsageExample] = []
        
        # 适用场景
        self.suitable_scenarios = []
        self.limitations = []
        
        # 优化建议
        self.optimization_suggestions = []
        
        # 版本管理
        self.versions: List[VersionInfo] = []
    
    def set_strategy_description(self, description: str, principle: str, logic: str) -> None:
        """
        设置策略描述信息
        
        Args:
            description: 策略描述
            principle: 策略原理
            logic: 策略逻辑
        """
        self.description = description
        self.principle = principle
        self.logic = logic
    
    def add_parameter(self, parameter: StrategyParameter) -> None:
        """添加策略参数"""
        self.parameters.append(parameter)
    
    def add_risk(self, risk: RiskFactor) -> None:
        """添加风险因素"""
        self.risks.append(risk)
    
    def add_performance(self, performance: PerformanceMetric) -> None:
        """添加绩效指标"""
        self.performance.append(performance)
    
    def add_example(self, example: UsageExample) -> None:
        """添加使用示例"""
        self.examples.append(example)
    
    def set_scenarios(self, suitable_scenarios: List[str], limitations: List[str]) -> None:
        """
        设置适用场景
        
        Args:
            suitable_scenarios: 适用场景列表
            limitations: 限制条件列表
        """
        self.suitable_scenarios = suitable_scenarios
        self.limitations = limitations
    
    def add_optimization_suggestion(self, suggestion: str) -> None:
        """添加优化建议"""
        self.optimization_suggestions.append(suggestion)
    
    def add_version(self, version: VersionInfo) -> None:
        """添加版本信息"""
        self.versions.append(version)
    
    def generate_markdown_doc(self) -> str:
        """生成Markdown格式的策略文档"""
        doc = f"""# {self.strategy_name} 策略文档

## 基本信息

- **策略名称**: {self.strategy_name}
- **策略类型**: {self.strategy_type}
- **创建日期**: {self.creation_date}
- **文档版本**: {self.versions[-1].version if self.versions else "1.0.0"}

---

## 1. 策略描述

### 1.1 策略概述
{self.description}

### 1.2 策略原理
{self.principle}

### 1.3 策略逻辑
{self.logic}

---

## 2. 参数文档

"""
        
        if self.parameters:
            doc += "| 参数名称 | 类型 | 默认值 | 范围 | 必需 | 说明 |\n"
            doc += "|---------|------|--------|------|------|------|\n"
            
            for param in self.parameters:
                range_str = param.range or "无限制"
                required_str = "是" if param.required else "否"
                doc += f"| {param.name} | {param.type} | {param.default_value} | {range_str} | {required_str} | {param.description} |\n"
            
            doc += "\n### 参数详细说明\n\n"
            for param in self.parameters:
                doc += f"**{param.name}**\n"
                doc += f"- 类型: {param.type}\n"
                doc += f"- 默认值: {param.default_value}\n"
                if param.range:
                    doc += f"- 取值范围: {param.range}\n"
                doc += f"- 是否必需: {'是' if param.required else '否'}\n"
                doc += f"- 说明: {param.description}\n"
                if param.notes:
                    doc += f"- 备注: {param.notes}\n"
                doc += "\n"
        else:
            doc += "暂无参数配置。\n\n"
        
        # 风险分析
        doc += "## 3. 风险分析\n\n"
        if self.risks:
            for risk in self.risks:
                doc += f"### 3.{self.risks.index(risk) + 1} {risk.factor}\n\n"
                doc += f"**风险描述**: {risk.description}\n\n"
                doc += f"**影响程度**: {risk.impact_level}\n\n"
                doc += f"**发生概率**: {risk.probability}\n\n"
                if risk.mitigation:
                    doc += f"**缓解措施**: {risk.mitigation}\n\n"
                doc += "---\n\n"
        else:
            doc += "暂无风险分析。\n\n"
        
        # 绩效分析
        doc += "## 4. 绩效分析\n\n"
        if self.performance:
            doc += "| 指标 | 数值 | 期间 | 基准 | 备注 |\n"
            doc += "|------|------|------|------|------|\n"
            
            for perf in self.performance:
                benchmark = perf.benchmark or "无"
                notes = perf.notes or ""
                doc += f"| {perf.metric} | {perf.value} | {perf.period} | {benchmark} | {notes} |\n"
        else:
            doc += "暂无绩效数据。\n\n"
        
        # 使用示例
        doc += "## 5. 使用示例\n\n"
        if self.examples:
            for i, example in enumerate(self.examples, 1):
                doc += f"### 5.{i} {example.title}\n\n"
                doc += f"**示例描述**: {example.description}\n\n"
                doc += f"**代码示例**:\n```python\n{example.code}\n```\n\n"
                doc += f"**预期结果**: {example.expected_result}\n\n"
                doc += "---\n\n"
        else:
            doc += "暂无使用示例。\n\n"
        
        # 适用场景
        doc += "## 6. 适用场景\n\n"
        doc += "### 6.1 适用条件\n\n"
        if self.suitable_scenarios:
            for scenario in self.suitable_scenarios:
                doc += f"- {scenario}\n"
        else:
            doc += "暂无适用条件。\n"
        
        doc += "\n### 6.2 限制条件\n\n"
        if self.limitations:
            for limitation in self.limitations:
                doc += f"- {limitation}\n"
        else:
            doc += "暂无限制条件。\n"
        
        doc += "\n---\n\n"
        
        # 优化建议
        doc += "## 7. 优化建议\n\n"
        if self.optimization_suggestions:
            for i, suggestion in enumerate(self.optimization_suggestions, 1):
                doc += f"### 7.{i} 优化建议\n\n{suggestion}\n\n"
        else:
            doc += "暂无优化建议。\n\n"
        
        doc += "---\n\n"
        
        # 版本管理
        doc += "## 8. 版本管理\n\n"
        if self.versions:
            doc += "| 版本 | 日期 | 变更内容 | 作者 |\n"
            doc += "|------|------|----------|------|\n"
            
            for version in self.versions:
                changes_str = "<br>".join(version.changes)
                doc += f"| {version.version} | {version.date} | {changes_str} | {version.author} |\n"
        else:
            doc += "暂无版本记录。\n"
        
        doc += f"\n---\n\n*文档生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return doc
    
    def generate_html_doc(self) -> str:
        """生成HTML格式的策略文档"""
        markdown_doc = self.generate_markdown_doc()
        
        # 简单的Markdown到HTML转换
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.strategy_name} - 策略文档</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #2c3e50;
            margin-top: 25px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        {markdown_doc.replace(chr(10), '<br>').replace('```python', '<pre><code>').replace('```', '</code></pre>')}
    </div>
</body>
</html>"""
        return html
    
    def save_doc(self, output_path: str, format: str = "markdown") -> None:
        """
        保存策略文档到文件
        
        Args:
            output_path: 输出文件路径
            format: 文档格式，支持 'markdown' 和 'html'
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "markdown":
            content = self.generate_markdown_doc()
            if not output_file.suffix:
                output_file = output_file.with_suffix('.md')
        elif format.lower() == "html":
            content = self.generate_html_doc()
            if not output_file.suffix:
                output_file = output_file.with_suffix('.html')
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def to_dict(self) -> Dict[str, Any]:
        """将策略文档转换为字典格式"""
        return {
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "creation_date": self.creation_date,
            "description": self.description,
            "principle": self.principle,
            "logic": self.logic,
            "parameters": [asdict(param) for param in self.parameters],
            "risks": [asdict(risk) for risk in self.risks],
            "performance": [asdict(perf) for perf in self.performance],
            "examples": [asdict(example) for example in self.examples],
            "suitable_scenarios": self.suitable_scenarios,
            "limitations": self.limitations,
            "optimization_suggestions": self.optimization_suggestions,
            "versions": [asdict(version) for version in self.versions]
        }
    
    def to_json(self) -> str:
        """将策略文档转换为JSON格式"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyDocGenerator':
        """从字典创建策略文档生成器实例"""
        generator = cls(data["strategy_name"], data.get("strategy_type", "通用"))
        generator.description = data.get("description", "")
        generator.principle = data.get("principle", "")
        generator.logic = data.get("logic", "")
        
        # 重建参数列表
        for param_data in data.get("parameters", []):
            param = StrategyParameter(**param_data)
            generator.parameters.append(param)
        
        # 重建风险列表
        for risk_data in data.get("risks", []):
            risk = RiskFactor(**risk_data)
            generator.risks.append(risk)
        
        # 重建绩效列表
        for perf_data in data.get("performance", []):
            perf = PerformanceMetric(**perf_data)
            generator.performance.append(perf)
        
        # 重建示例列表
        for example_data in data.get("examples", []):
            example = UsageExample(**example_data)
            generator.examples.append(example)
        
        generator.suitable_scenarios = data.get("suitable_scenarios", [])
        generator.limitations = data.get("limitations", [])
        generator.optimization_suggestions = data.get("optimization_suggestions", [])
        
        # 重建版本列表
        for version_data in data.get("versions", []):
            version = VersionInfo(**version_data)
            generator.versions.append(version)
        
        return generator
    
    def get_summary(self) -> Dict[str, Any]:
        """获取策略文档摘要信息"""
        return {
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "creation_date": self.creation_date,
            "parameter_count": len(self.parameters),
            "risk_count": len(self.risks),
            "performance_count": len(self.performance),
            "example_count": len(self.examples),
            "version_count": len(self.versions),
            "last_updated": self.versions[-1].date if self.versions else self.creation_date
        }