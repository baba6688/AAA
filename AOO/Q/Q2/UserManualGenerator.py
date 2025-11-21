"""
Q2用户手册生成器 - 主要实现

提供完整的用户手册生成解决方案，包括内容结构化、模板系统、
多媒体支持、分步指导、FAQ管理、用户反馈、多语言支持和版本控制。
"""

import json
import os
import re
import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid


class TemplateType(Enum):
    """模板类型枚举"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    QUICK_START = "quick_start"
    TROUBLESHOOTING = "troubleshooting"


class OutputFormat(Enum):
    """输出格式枚举"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"


@dataclass
class ContentSection:
    """内容章节数据结构"""
    id: str
    title: str
    content: str
    level: int  # 标题级别 1-6
    order: int
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class MultimediaItem:
    """多媒体项目数据结构"""
    id: str
    type: str  # image, video, chart, diagram
    path: str
    caption: str
    alt_text: str
    description: str
    tags: List[str] = None


@dataclass
class StepGuide:
    """分步指导数据结构"""
    id: str
    title: str
    description: str
    steps: List[Dict[str, Any]]
    prerequisites: List[str] = None
    estimated_time: str = None
    difficulty_level: str = None


@dataclass
class FAQItem:
    """FAQ项目数据结构"""
    id: str
    question: str
    answer: str
    category: str
    tags: List[str] = None
    related_sections: List[str] = None


@dataclass
class UserFeedback:
    """用户反馈数据结构"""
    id: str
    section_id: str
    feedback_type: str  # suggestion, bug, question, praise
    content: str
    rating: Optional[int] = None
    contact_info: Optional[str] = None
    timestamp: str = None
    status: str = "pending"  # pending, reviewed, resolved


class ContentStructure:
    """内容结构管理器"""
    
    def __init__(self):
        self.sections: Dict[str, ContentSection] = {}
        self.section_order: List[str] = []
    
    def add_section(self, section: ContentSection) -> None:
        """添加章节"""
        self.sections[section.id] = section
        if section.id not in self.section_order:
            self.section_order.append(section.id)
    
    def remove_section(self, section_id: str) -> bool:
        """删除章节"""
        if section_id in self.sections:
            # 移除子章节
            children = [sid for sid, s in self.sections.items() if s.parent_id == section_id]
            for child_id in children:
                self.remove_section(child_id)
            
            del self.sections[section_id]
            if section_id in self.section_order:
                self.section_order.remove(section_id)
            return True
        return False
    
    def move_section(self, section_id: str, new_parent_id: Optional[str], new_order: int) -> bool:
        """移动章节"""
        if section_id not in self.sections:
            return False
        
        section = self.sections[section_id]
        section.parent_id = new_parent_id
        
        # 重新排序
        self.section_order.remove(section_id)
        self.section_order.insert(new_order, section_id)
        return True
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """获取树形结构"""
        def build_tree(parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
            result = []
            for section_id in self.section_order:
                section = self.sections[section_id]
                if section.parent_id == parent_id:
                    node = {
                        'id': section.id,
                        'title': section.title,
                        'level': section.level,
                        'content': section.content,
                        'metadata': section.metadata or {},
                        'children': build_tree(section_id)
                    }
                    result.append(node)
            return result
        
        return build_tree()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'sections': {sid: asdict(section) for sid, section in self.sections.items()},
            'section_order': self.section_order
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典格式加载"""
        self.sections = {sid: ContentSection(**section_data) for sid, section_data in data['sections'].items()}
        self.section_order = data['section_order']


class TemplateManager:
    """模板管理器"""
    
    def __init__(self):
        self.templates: Dict[TemplateType, Dict[str, str]] = {
            TemplateType.SIMPLE: {
                'header': '# {title}\n\n{introduction}\n\n',
                'section': '## {title}\n\n{content}\n\n',
                'footer': '---\n\n*最后更新：{last_update}*\n'
            },
            TemplateType.DETAILED: {
                'header': '# {title}\n\n**版本：** {version} | **最后更新：** {last_update}\n\n{introduction}\n\n## 目录\n\n{toc}\n\n',
                'section': '## {title}\n\n{content}\n\n### 相关资源\n{resources}\n\n',
                'footer': '---\n\n## 反馈与支持\n\n如果您在使用过程中遇到问题或有改进建议，请通过以下方式联系我们：\n\n{feedback_info}\n\n*文档版本：{version} | 生成时间：{generate_time}*\n'
            },
            TemplateType.TECHNICAL: {
                'header': '# {title} - 技术文档\n\n**API版本：** {api_version} | **兼容性：** {compatibility}\n\n{introduction}\n\n## 快速导航\n\n{navigation}\n\n',
                'section': '## {title}\n\n### 概述\n{overview}\n\n### 详细说明\n{details}\n\n### 示例代码\n{code_example}\n\n### 参数说明\n{parameters}\n\n',
                'footer': '---\n\n## 技术支持\n\n**API文档：** {api_docs}\n**更新日志：** {changelog}\n\n*技术文档版本：{version}*\n'
            },
            TemplateType.QUICK_START: {
                'header': '# {title} - 快速开始\n\n🎯 **目标：** {goal}\n⏱️ **预计时间：** {estimated_time}\n\n{introduction}\n\n',
                'section': '## 步骤 {step_number}: {title}\n\n{content}\n\n{media_content}\n\n### 验证步骤\n{verification}\n\n',
                'footer': '🎉 **完成！** 您已经成功完成快速开始指南。\n\n下一步：{next_steps}\n'
            },
            TemplateType.TROUBLESHOOTING: {
                'header': '# {title} - 故障排除指南\n\n{introduction}\n\n## 快速诊断\n\n{quick_diagnosis}\n\n',
                'section': '## 问题：{title}\n\n**症状：** {symptoms}\n\n**原因：** {causes}\n\n**解决方案：**\n{solutions}\n\n**预防措施：** {prevention}\n\n',
                'footer': '---\n\n## 需要更多帮助？\n\n如果以上解决方案都无法解决您的问题，请：\n\n{support_info}\n'
            }
        }
    
    def get_template(self, template_type: TemplateType, template_key: str) -> str:
        """获取模板"""
        return self.templates.get(template_type, {}).get(template_key, '')
    
    def render_template(self, template_type: TemplateType, template_key: str, **kwargs) -> str:
        """渲染模板"""
        template = self.get_template(template_type, template_key)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"模板参数缺失: {e}")
    
    def add_custom_template(self, template_type: TemplateType, template_key: str, template_content: str) -> None:
        """添加自定义模板"""
        if template_type not in self.templates:
            self.templates[template_type] = {}
        self.templates[template_type][template_key] = template_content


class MultimediaHandler:
    """多媒体处理器"""
    
    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        self.media_items: Dict[str, MultimediaItem] = {}
        self.supported_formats = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'],
            'video': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm'],
            'document': ['.pdf', '.doc', '.docx', '.txt'],
            'chart': ['.png', '.svg', '.pdf']
        }
    
    def add_media(self, media_item: MultimediaItem) -> bool:
        """添加多媒体项目"""
        if self._validate_media_format(media_item.path, media_item.type):
            self.media_items[media_item.id] = media_item
            return True
        return False
    
    def remove_media(self, media_id: str) -> bool:
        """删除多媒体项目"""
        if media_id in self.media_items:
            del self.media_items[media_id]
            return True
        return False
    
    def _validate_media_format(self, file_path: str, media_type: str) -> bool:
        """验证媒体格式"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats.get(media_type, [])
    
    def generate_markdown_media(self, media_id: str) -> str:
        """生成媒体markdown代码"""
        if media_id not in self.media_items:
            return ""
        
        media = self.media_items[media_id]
        if media.type == 'image':
            return f"![{media.alt_text}]({media.path})\n\n*{media.caption}*\n\n"
        elif media.type == 'video':
            return f"[{media.caption}]({media.path})\n\n*{media.description}*\n\n"
        else:
            return f"[{media.caption}]({media.path})\n\n"
    
    def get_media_by_tag(self, tag: str) -> List[MultimediaItem]:
        """根据标签获取媒体"""
        return [item for item in self.media_items.values() if tag in (item.tags or [])]
    
    def optimize_media_path(self, original_path: str) -> str:
        """优化媒体路径"""
        if self.base_path and not original_path.startswith(('http://', 'https://')):
            return os.path.join(self.base_path, original_path)
        return original_path


class StepByStepGuide:
    """分步指导生成器"""
    
    def __init__(self):
        self.guides: Dict[str, StepGuide] = {}
    
    def create_guide(self, guide: StepGuide) -> None:
        """创建分步指导"""
        self.guides[guide.id] = guide
    
    def update_guide(self, guide_id: str, **kwargs) -> bool:
        """更新分步指导"""
        if guide_id in self.guides:
            guide = self.guides[guide_id]
            for key, value in kwargs.items():
                if hasattr(guide, key):
                    setattr(guide, key, value)
            return True
        return False
    
    def generate_markdown_guide(self, guide_id: str) -> str:
        """生成分步指导的markdown"""
        if guide_id not in self.guides:
            return ""
        
        guide = self.guides[guide_id]
        result = f"# {guide.title}\n\n"
        result += f"{guide.description}\n\n"
        
        if guide.prerequisites:
            result += "## 前置条件\n\n"
            for prereq in guide.prerequisites:
                result += f"- {prereq}\n"
            result += "\n"
        
        if guide.estimated_time:
            result += f"**预计时间：** {guide.estimated_time}\n\n"
        
        if guide.difficulty_level:
            result += f"**难度级别：** {guide.difficulty_level}\n\n"
        
        result += "## 操作步骤\n\n"
        
        for i, step in enumerate(guide.steps, 1):
            result += f"### 步骤 {i}\n\n"
            result += f"{step.get('description', '')}\n\n"
            
            if 'media' in step:
                result += f"![步骤{i}]({step['media']})\n\n"
            
            if 'code' in step:
                result += f"```\n{step['code']}\n```\n\n"
            
            if 'verification' in step:
                result += f"**验证：** {step['verification']}\n\n"
        
        return result
    
    def add_step(self, guide_id: str, step: Dict[str, Any]) -> bool:
        """添加步骤"""
        if guide_id in self.guides:
            self.guides[guide_id].steps.append(step)
            return True
        return False


class FAQManager:
    """FAQ管理器"""
    
    def __init__(self):
        self.faqs: Dict[str, FAQItem] = {}
        self.categories: Dict[str, List[str]] = {}  # category -> faq_ids
    
    def add_faq(self, faq: FAQItem) -> None:
        """添加FAQ"""
        self.faqs[faq.id] = faq
        if faq.category not in self.categories:
            self.categories[faq.category] = []
        if faq.id not in self.categories[faq.category]:
            self.categories[faq.category].append(faq.id)
    
    def remove_faq(self, faq_id: str) -> bool:
        """删除FAQ"""
        if faq_id in self.faqs:
            faq = self.faqs[faq_id]
            if faq.category in self.categories and faq_id in self.categories[faq.category]:
                self.categories[faq.category].remove(faq_id)
            del self.faqs[faq_id]
            return True
        return False
    
    def get_faqs_by_category(self, category: str) -> List[FAQItem]:
        """根据分类获取FAQ"""
        if category not in self.categories:
            return []
        return [self.faqs[faq_id] for faq_id in self.categories[category] if faq_id in self.faqs]
    
    def search_faqs(self, query: str) -> List[FAQItem]:
        """搜索FAQ"""
        results = []
        query_lower = query.lower()
        
        for faq in self.faqs.values():
            if (query_lower in faq.question.lower() or 
                query_lower in faq.answer.lower() or
                any(query_lower in tag.lower() for tag in (faq.tags or []))):
                results.append(faq)
        
        return results
    
    def generate_markdown_faq(self, category: str = None) -> str:
        """生成FAQ的markdown"""
        if category:
            faqs = self.get_faqs_by_category(category)
            title = f"常见问题 - {category}"
        else:
            faqs = list(self.faqs.values())
            title = "常见问题"
        
        result = f"# {title}\n\n"
        
        if category is None:
            # 按分类组织
            for cat, faq_ids in self.categories.items():
                result += f"## {cat}\n\n"
                for faq_id in faq_ids:
                    if faq_id in self.faqs:
                        faq = self.faqs[faq_id]
                        result += f"### {faq.question}\n\n"
                        result += f"{faq.answer}\n\n"
                        if faq.tags:
                            result += f"*标签：{', '.join(faq.tags)}*\n\n"
                result += "\n"
        else:
            # 单个分类
            for faq in faqs:
                result += f"### {faq.question}\n\n"
                result += f"{faq.answer}\n\n"
                if faq.tags:
                    result += f"*标签：{', '.join(faq.tags)}*\n\n"
        
        return result


class FeedbackHandler:
    """用户反馈处理器"""
    
    def __init__(self):
        self.feedbacks: Dict[str, UserFeedback] = {}
        self.feedback_types = ['suggestion', 'bug', 'question', 'praise']
    
    def add_feedback(self, feedback: UserFeedback) -> None:
        """添加反馈"""
        if not feedback.timestamp:
            feedback.timestamp = datetime.datetime.now().isoformat()
        self.feedbacks[feedback.id] = feedback
    
    def update_feedback_status(self, feedback_id: str, status: str) -> bool:
        """更新反馈状态"""
        if feedback_id in self.feedbacks:
            self.feedbacks[feedback_id].status = status
            return True
        return False
    
    def get_feedbacks_by_section(self, section_id: str) -> List[UserFeedback]:
        """获取指定章节的反馈"""
        return [fb for fb in self.feedbacks.values() if fb.section_id == section_id]
    
    def get_feedbacks_by_status(self, status: str) -> List[UserFeedback]:
        """获取指定状态的反馈"""
        return [fb for fb in self.feedbacks.values() if fb.status == status]
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """获取反馈统计"""
        total = len(self.feedbacks)
        by_status = {}
        by_type = {}
        
        for feedback in self.feedbacks.values():
            by_status[feedback.status] = by_status.get(feedback.status, 0) + 1
            by_type[feedback.feedback_type] = by_type.get(feedback.feedback_type, 0) + 1
        
        avg_rating = 0
        ratings = [fb.rating for fb in self.feedbacks.values() if fb.rating is not None]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
        
        return {
            'total': total,
            'by_status': by_status,
            'by_type': by_type,
            'average_rating': round(avg_rating, 2)
        }
    
    def generate_feedback_report(self) -> str:
        """生成反馈报告"""
        stats = self.get_feedback_statistics()
        
        result = "# 用户反馈报告\n\n"
        result += f"**生成时间：** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        result += "## 总体统计\n\n"
        result += f"- 总反馈数：{stats['total']}\n"
        result += f"- 平均评分：{stats['average_rating']}/5\n\n"
        
        result += "## 按状态分类\n\n"
        for status, count in stats['by_status'].items():
            result += f"- {status}：{count}\n"
        result += "\n"
        
        result += "## 按类型分类\n\n"
        for ftype, count in stats['by_type'].items():
            result += f"- {ftype}：{count}\n"
        result += "\n"
        
        # 待处理的反馈
        pending_feedbacks = self.get_feedbacks_by_status('pending')
        if pending_feedbacks:
            result += "## 待处理反馈\n\n"
            for feedback in pending_feedbacks[:10]:  # 只显示前10条
                result += f"### {feedback.feedback_type.title()}\n\n"
                result += f"**章节：** {feedback.section_id}\n"
                result += f"**内容：** {feedback.content}\n"
                if feedback.rating:
                    result += f"**评分：** {feedback.rating}/5\n"
                result += f"**时间：** {feedback.timestamp}\n\n"
        
        return result


class MultiLanguageSupport:
    """多语言支持"""
    
    def __init__(self):
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = 'zh-CN'
        self.supported_languages = ['zh-CN', 'en-US', 'ja-JP']
    
    def add_translation(self, language: str, key: str, value: str) -> None:
        """添加翻译"""
        if language not in self.translations:
            self.translations[language] = {}
        self.translations[language][key] = value
    
    def set_language(self, language: str) -> bool:
        """设置当前语言"""
        if language in self.supported_languages:
            self.current_language = language
            return True
        return False
    
    def translate(self, key: str, language: str = None) -> str:
        """翻译"""
        lang = language or self.current_language
        return self.translations.get(lang, {}).get(key, key)
    
    def translate_content(self, content: str, target_language: str) -> str:
        """翻译内容"""
        # 简单的占位符替换翻译
        # 在实际应用中，这里可以集成翻译API
        translations_map = {
            'en-US': {
                '介绍': 'Introduction',
                '快速开始': 'Quick Start',
                '用户指南': 'User Guide',
                '常见问题': 'FAQ',
                '联系我们': 'Contact'
            },
            'zh-CN': {
                'Introduction': '介绍',
                'Quick Start': '快速开始',
                'User Guide': '用户指南',
                'FAQ': '常见问题',
                'Contact': '联系我们'
            }
        }
        
        lang_map = translations_map.get(target_language, {})
        result = content
        for source, target in lang_map.items():
            result = result.replace(source, target)
        
        return result
    
    def export_translations(self, language: str) -> str:
        """导出翻译文件"""
        if language not in self.translations:
            return ""
        
        return json.dumps(self.translations[language], ensure_ascii=False, indent=2)


class VersionControl:
    """版本控制"""
    
    def __init__(self):
        self.versions: List[Dict[str, Any]] = []
        self.current_version = "1.0.0"
    
    def create_version(self, version: str, description: str, author: str) -> str:
        """创建新版本"""
        version_info = {
            'version': version,
            'description': description,
            'author': author,
            'timestamp': datetime.datetime.now().isoformat(),
            'changes': []
        }
        
        self.versions.append(version_info)
        self.current_version = version
        return version
    
    def add_change(self, version: str, change_type: str, description: str) -> bool:
        """添加变更记录"""
        for version_info in self.versions:
            if version_info['version'] == version:
                version_info['changes'].append({
                    'type': change_type,
                    'description': description,
                    'timestamp': datetime.datetime.now().isoformat()
                })
                return True
        return False
    
    def get_version_info(self, version: str = None) -> Optional[Dict[str, Any]]:
        """获取版本信息"""
        target_version = version or self.current_version
        for version_info in self.versions:
            if version_info['version'] == target_version:
                return version_info
        return None
    
    def get_change_log(self) -> str:
        """生成变更日志"""
        if not self.versions:
            return "# 变更日志\n\n暂无变更记录。\n"
        
        result = "# 变更日志\n\n"
        
        for version_info in reversed(self.versions):
            result += f"## 版本 {version_info['version']}\n\n"
            result += f"**发布日期：** {version_info['timestamp'][:10]}\n"
            result += f"**作者：** {version_info['author']}\n\n"
            result += f"{version_info['description']}\n\n"
            
            if version_info['changes']:
                result += "### 详细变更\n\n"
                for change in version_info['changes']:
                    result += f"- **{change['type']}：** {change['description']}\n"
                result += "\n"
            
            result += "---\n\n"
        
        return result


class UserManualGenerator:
    """用户手册生成器主类"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化各个组件
        self.content_structure = ContentStructure()
        self.template_manager = TemplateManager()
        self.multimedia_handler = MultimediaHandler()
        self.step_guide = StepByStepGuide()
        self.faq_manager = FAQManager()
        self.feedback_handler = FeedbackHandler()
        self.multi_language = MultiLanguageSupport()
        self.version_control = VersionControl()
        
        # 手册基本信息
        self.manual_info = {
            'title': '用户手册',
            'version': '1.0.0',
            'author': 'Q2开发团队',
            'description': '',
            'introduction': ''
        }
    
    def set_manual_info(self, **kwargs) -> None:
        """设置手册基本信息"""
        self.manual_info.update(kwargs)
    
    def add_section(self, title: str, content: str, level: int = 1, 
                   parent_id: Optional[str] = None, order: Optional[int] = None) -> str:
        """添加章节"""
        section_id = str(uuid.uuid4())
        if order is None:
            order = len(self.content_structure.section_order)
        
        section = ContentSection(
            id=section_id,
            title=title,
            content=content,
            level=level,
            order=order,
            parent_id=parent_id
        )
        
        self.content_structure.add_section(section)
        return section_id
    
    def add_multimedia(self, media_type: str, path: str, caption: str, 
                      alt_text: str, description: str, tags: List[str] = None) -> str:
        """添加多媒体项目"""
        media_id = str(uuid.uuid4())
        media_item = MultimediaItem(
            id=media_id,
            type=media_type,
            path=path,
            caption=caption,
            alt_text=alt_text,
            description=description,
            tags=tags or []
        )
        
        self.multimedia_handler.add_media(media_item)
        return media_id
    
    def create_step_guide(self, title: str, description: str, 
                         steps: List[Dict[str, Any]]) -> str:
        """创建分步指导"""
        guide_id = str(uuid.uuid4())
        guide = StepGuide(
            id=guide_id,
            title=title,
            description=description,
            steps=steps
        )
        
        self.step_guide.create_guide(guide)
        return guide_id
    
    def add_faq(self, question: str, answer: str, category: str, 
               tags: List[str] = None) -> str:
        """添加FAQ"""
        faq_id = str(uuid.uuid4())
        faq = FAQItem(
            id=faq_id,
            question=question,
            answer=answer,
            category=category,
            tags=tags or []
        )
        
        self.faq_manager.add_faq(faq)
        return faq_id
    
    def add_feedback(self, section_id: str, feedback_type: str, content: str,
                    rating: Optional[int] = None, contact_info: Optional[str] = None) -> str:
        """添加用户反馈"""
        feedback_id = str(uuid.uuid4())
        feedback = UserFeedback(
            id=feedback_id,
            section_id=section_id,
            feedback_type=feedback_type,
            content=content,
            rating=rating,
            contact_info=contact_info
        )
        
        self.feedback_handler.add_feedback(feedback)
        return feedback_id
    
    def generate_manual(self, template_type: TemplateType = TemplateType.DETAILED,
                       output_format: OutputFormat = OutputFormat.MARKDOWN,
                       language: str = 'zh-CN') -> str:
        """生成用户手册"""
        
        # 设置语言
        self.multi_language.set_language(language)
        
        # 生成目录
        toc = self._generate_table_of_contents()
        
        # 生成各章节内容
        content = self._generate_content(template_type, toc)
        
        # 生成FAQ
        faq_content = self.faq_manager.generate_markdown_faq()
        
        # 生成反馈信息
        feedback_info = self._generate_feedback_info()
        
        # 生成完整手册
        full_content = self._assemble_manual(content, faq_content, feedback_info, 
                                           template_type, language)
        
        # 保存文件
        output_file = self._save_manual(full_content, output_format, language)
        
        return output_file
    
    def _generate_table_of_contents(self) -> str:
        """生成目录"""
        toc = []
        tree = self.content_structure.get_tree_structure()
        
        def add_toc_items(items, level=0):
            for item in items:
                indent = "  " * level
                toc.append(f"{indent}- [{item['title']}](#{item['id']})")
                if item['children']:
                    add_toc_items(item['children'], level + 1)
        
        add_toc_items(tree)
        return "\n".join(toc)
    
    def _generate_content(self, template_type: TemplateType, toc: str) -> str:
        """生成内容"""
        content = ""
        tree = self.content_structure.get_tree_structure()
        
        def render_section(item):
            section_content = self.template_manager.render_template(
                template_type, 'section',
                title=item['title'],
                content=item['content'],
                resources="",  # 可以后续添加相关资源
                overview=item.get('metadata', {}).get('overview', ''),
                details=item.get('metadata', {}).get('details', ''),
                code_example=item.get('metadata', {}).get('code_example', ''),
                parameters=item.get('metadata', {}).get('parameters', '')
            )
            
            # 递归渲染子章节
            if item['children']:
                for child in item['children']:
                    section_content += render_section(child)
            
            return section_content
        
        for item in tree:
            content += render_section(item)
        
        return content
    
    def _generate_feedback_info(self) -> str:
        """生成反馈信息"""
        stats = self.feedback_handler.get_feedback_statistics()
        return f"当前版本：{self.manual_info['version']} | 总反馈数：{stats['total']} | 平均评分：{stats['average_rating']}/5"
    
    def _assemble_manual(self, content: str, faq_content: str, feedback_info: str,
                        template_type: TemplateType, language: str) -> str:
        """组装完整手册"""
        
        # 生成头部
        header = self.template_manager.render_template(
            template_type, 'header',
            title=self.manual_info['title'],
            version=self.manual_info['version'],
            last_update=datetime.datetime.now().strftime('%Y-%m-%d'),
            introduction=self.manual_info['introduction'],
            toc=self._generate_table_of_contents(),
            goal=self.manual_info.get('goal', ''),
            estimated_time=self.manual_info.get('estimated_time', ''),
            api_version=self.manual_info.get('api_version', ''),
            compatibility=self.manual_info.get('compatibility', ''),
            navigation="",  # 可以后续添加导航
            quick_diagnosis=""  # 可以后续添加快速诊断
        )
        
        # 生成尾部
        footer = self.template_manager.render_template(
            template_type, 'footer',
            last_update=datetime.datetime.now().strftime('%Y-%m-%d'),
            feedback_info=feedback_info,
            version=self.manual_info['version'],
            generate_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            api_docs=self.manual_info.get('api_docs', ''),
            changelog=self.version_control.get_change_log(),
            next_steps=self.manual_info.get('next_steps', ''),
            support_info=self.manual_info.get('support_info', '')
        )
        
        # 组装完整内容
        manual_content = header + content
        
        # 添加FAQ（如果存在）
        if faq_content.strip():
            manual_content += "\n\n" + faq_content
        
        manual_content += "\n\n" + footer
        
        return manual_content
    
    def _save_manual(self, content: str, output_format: OutputFormat, language: str) -> str:
        """保存手册"""
        filename = f"{self.manual_info['title']}_{language}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if output_format == OutputFormat.MARKDOWN:
            filename += ".md"
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            # 其他格式可以后续添加转换逻辑
            filename += ".md"  # 暂时保存为markdown
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return str(filepath)
    
    def export_data(self, filepath: str) -> None:
        """导出数据"""
        data = {
            'manual_info': self.manual_info,
            'content_structure': self.content_structure.to_dict(),
            'multimedia': {mid: asdict(item) for mid, item in self.multimedia_handler.media_items.items()},
            'guides': {gid: asdict(guide) for gid, guide in self.step_guide.guides.items()},
            'faqs': {fid: asdict(faq) for fid, faq in self.faq_manager.faqs.items()},
            'feedbacks': {fid: asdict(fb) for fid, fb in self.feedback_handler.feedbacks.items()},
            'translations': self.multi_language.translations,
            'versions': self.version_control.versions
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def import_data(self, filepath: str) -> None:
        """导入数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.manual_info = data['manual_info']
        self.content_structure.from_dict(data['content_structure'])
        
        # 恢复多媒体
        self.multimedia_handler.media_items = {
            mid: MultimediaItem(**item_data) 
            for mid, item_data in data['multimedia'].items()
        }
        
        # 恢复分步指导
        self.step_guide.guides = {
            gid: StepGuide(**guide_data) 
            for gid, guide_data in data['guides'].items()
        }
        
        # 恢复FAQ
        self.faq_manager.faqs = {
            fid: FAQItem(**faq_data) 
            for fid, faq_data in data['faqs'].items()
        }
        
        # 恢复反馈
        self.feedback_handler.feedbacks = {
            fid: UserFeedback(**fb_data) 
            for fid, fb_data in data['feedbacks'].items()
        }
        
        # 恢复翻译
        self.multi_language.translations = data['translations']
        
        # 恢复版本
        self.version_control.versions = data['versions']
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'sections_count': len(self.content_structure.sections),
            'multimedia_count': len(self.multimedia_handler.media_items),
            'guides_count': len(self.step_guide.guides),
            'faqs_count': len(self.faq_manager.faqs),
            'feedbacks_count': len(self.feedback_handler.feedbacks),
            'supported_languages': self.multi_language.supported_languages,
            'current_language': self.multi_language.current_language,
            'current_version': self.version_control.current_version
        }