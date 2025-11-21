#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q7教程生成器
一个功能完整的教程生成器，支持内容结构化、步骤指导、代码示例、互动元素等功能
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class DifficultyLevel(Enum):
    """难度级别枚举"""
    BEGINNER = "初级"
    INTERMEDIATE = "中级"
    ADVANCED = "高级"


class ContentType(Enum):
    """内容类型枚举"""
    TEXT = "文本"
    CODE = "代码"
    IMAGE = "图片"
    VIDEO = "视频"
    ANIMATION = "动画"
    QUIZ = "测验"
    EXERCISE = "练习"


@dataclass
class CodeExample:
    """代码示例数据类"""
    language: str
    code: str
    description: str
    output: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class QuizQuestion:
    """测验题目数据类"""
    question: str
    options: List[str]
    correct_answer: int
    explanation: str
    difficulty: DifficultyLevel


@dataclass
class Exercise:
    """练习题数据类"""
    title: str
    description: str
    requirements: List[str]
    hints: List[str]
    solution: Optional[str] = None
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER


@dataclass
class ContentBlock:
    """内容块数据类"""
    id: str
    type: ContentType
    title: str
    content: str
    code_example: Optional[CodeExample] = None
    quiz_question: Optional[QuizQuestion] = None
    exercise: Optional[Exercise] = None
    multimedia_files: List[str] = None
    order: int = 0

    def __post_init__(self):
        if self.multimedia_files is None:
            self.multimedia_files = []


@dataclass
class Chapter:
    """章节数据类"""
    id: str
    title: str
    description: str
    content_blocks: List[ContentBlock]
    estimated_time: int  # 预计学习时间（分钟）
    difficulty: DifficultyLevel
    prerequisites: List[str] = None

    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class ProgressRecord:
    """进度记录数据类"""
    user_id: str
    tutorial_id: str
    chapter_id: str
    content_block_id: str
    completed: bool
    score: Optional[float] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class TutorialGenerator:
    """教程生成器主类"""
    
    def __init__(self, tutorial_id: str, title: str, description: str, 
                 difficulty: DifficultyLevel = DifficultyLevel.BEGINNER):
        """
        初始化教程生成器
        
        Args:
            tutorial_id: 教程唯一标识符
            title: 教程标题
            description: 教程描述
            difficulty: 难度级别
        """
        self.tutorial_id = tutorial_id
        self.title = title
        self.description = description
        self.difficulty = difficulty
        self.chapters: List[Chapter] = []
        self.progress_records: List[ProgressRecord] = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "author": "Q7教程生成器"
        }
    
    def add_chapter(self, chapter: Chapter) -> None:
        """
        添加章节
        
        Args:
            chapter: 章节对象
        """
        self.chapters.append(chapter)
        self.metadata["updated_at"] = datetime.now().isoformat()
    
    def create_chapter(self, title: str, description: str, 
                      estimated_time: int, difficulty: DifficultyLevel,
                      prerequisites: List[str] = None) -> Chapter:
        """
        创建章节
        
        Args:
            title: 章节标题
            description: 章节描述
            estimated_time: 预计学习时间（分钟）
            difficulty: 难度级别
            prerequisites: 前置条件
            
        Returns:
            Chapter: 创建的章节对象
        """
        chapter_id = f"chapter_{len(self.chapters) + 1}"
        chapter = Chapter(
            id=chapter_id,
            title=title,
            description=description,
            content_blocks=[],
            estimated_time=estimated_time,
            difficulty=difficulty,
            prerequisites=prerequisites or []
        )
        self.add_chapter(chapter)
        return chapter
    
    def add_content_block(self, chapter_id: str, content_block: ContentBlock) -> None:
        """
        添加内容块到指定章节
        
        Args:
            chapter_id: 章节ID
            content_block: 内容块对象
        """
        chapter = self._find_chapter(chapter_id)
        if chapter:
            content_block.order = len(chapter.content_blocks)
            chapter.content_blocks.append(content_block)
            self.metadata["updated_at"] = datetime.now().isoformat()
    
    def create_text_content(self, chapter_id: str, title: str, content: str) -> ContentBlock:
        """
        创建文本内容块
        
        Args:
            chapter_id: 章节ID
            title: 内容标题
            content: 文本内容
            
        Returns:
            ContentBlock: 创建的内容块
        """
        content_block_id = f"content_{len(self.chapters) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        content_block = ContentBlock(
            id=content_block_id,
            type=ContentType.TEXT,
            title=title,
            content=content
        )
        self.add_content_block(chapter_id, content_block)
        return content_block
    
    def create_code_example(self, chapter_id: str, title: str, 
                          language: str, code: str, description: str,
                          output: str = None, explanation: str = None) -> ContentBlock:
        """
        创建代码示例内容块
        
        Args:
            chapter_id: 章节ID
            title: 内容标题
            language: 编程语言
            code: 代码
            description: 描述
            output: 输出结果
            explanation: 解释
            
        Returns:
            ContentBlock: 创建的内容块
        """
        content_block_id = f"code_{len(self.chapters) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        code_example = CodeExample(
            language=language,
            code=code,
            description=description,
            output=output,
            explanation=explanation
        )
        content_block = ContentBlock(
            id=content_block_id,
            type=ContentType.CODE,
            title=title,
            content=description,
            code_example=code_example
        )
        self.add_content_block(chapter_id, content_block)
        return content_block
    
    def create_quiz(self, chapter_id: str, title: str, 
                   question: str, options: List[str], correct_answer: int,
                   explanation: str, difficulty: DifficultyLevel = DifficultyLevel.BEGINNER) -> ContentBlock:
        """
        创建测验内容块
        
        Args:
            chapter_id: 章节ID
            title: 内容标题
            question: 问题
            options: 选项列表
            correct_answer: 正确答案索引
            explanation: 解释
            difficulty: 难度级别
            
        Returns:
            ContentBlock: 创建的内容块
        """
        content_block_id = f"quiz_{len(self.chapters) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        quiz_question = QuizQuestion(
            question=question,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            difficulty=difficulty
        )
        content_block = ContentBlock(
            id=content_block_id,
            type=ContentType.QUIZ,
            title=title,
            content=question,
            quiz_question=quiz_question
        )
        self.add_content_block(chapter_id, content_block)
        return content_block
    
    def create_exercise(self, chapter_id: str, title: str, description: str,
                       requirements: List[str], hints: List[str],
                       solution: str = None, difficulty: DifficultyLevel = DifficultyLevel.BEGINNER) -> ContentBlock:
        """
        创建练习题内容块
        
        Args:
            chapter_id: 章节ID
            title: 练习标题
            description: 练习描述
            requirements: 要求列表
            hints: 提示列表
            solution: 解决方案
            difficulty: 难度级别
            
        Returns:
            ContentBlock: 创建的内容块
        """
        content_block_id = f"exercise_{len(self.chapters) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exercise = Exercise(
            title=title,
            description=description,
            requirements=requirements,
            hints=hints,
            solution=solution,
            difficulty=difficulty
        )
        content_block = ContentBlock(
            id=content_block_id,
            type=ContentType.EXERCISE,
            title=title,
            content=description,
            exercise=exercise
        )
        self.add_content_block(chapter_id, content_block)
        return content_block
    
    def add_multimedia(self, content_block_id: str, file_path: str) -> None:
        """
        添加多媒体文件到内容块
        
        Args:
            content_block_id: 内容块ID
            file_path: 文件路径
        """
        content_block = self._find_content_block(content_block_id)
        if content_block:
            content_block.multimedia_files.append(file_path)
    
    def generate_step_by_step_guide(self, chapter_id: str, steps: List[str]) -> ContentBlock:
        """
        生成逐步指导内容
        
        Args:
            chapter_id: 章节ID
            steps: 步骤列表
            
        Returns:
            ContentBlock: 创建的内容块
        """
        content = "## 步骤指导\n\n"
        for i, step in enumerate(steps, 1):
            content += f"{i}. {step}\n\n"
        
        guide_id = f"guide_{len(self.chapters) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        content_block = ContentBlock(
            id=guide_id,
            type=ContentType.TEXT,
            title="逐步操作指导",
            content=content
        )
        self.add_content_block(chapter_id, content_block)
        return content_block
    
    def track_progress(self, user_id: str, chapter_id: str, 
                      content_block_id: str, completed: bool, score: float = None) -> None:
        """
        跟踪学习进度
        
        Args:
            user_id: 用户ID
            chapter_id: 章节ID
            content_block_id: 内容块ID
            completed: 是否完成
            score: 得分
        """
        progress_record = ProgressRecord(
            user_id=user_id,
            tutorial_id=self.tutorial_id,
            chapter_id=chapter_id,
            content_block_id=content_block_id,
            completed=completed,
            score=score
        )
        self.progress_records.append(progress_record)
    
    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户学习进度
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict: 进度信息
        """
        user_records = [r for r in self.progress_records if r.user_id == user_id]
        
        total_blocks = sum(len(chapter.content_blocks) for chapter in self.chapters)
        completed_blocks = len([r for r in user_records if r.completed])
        
        progress_percentage = (completed_blocks / total_blocks * 100) if total_blocks > 0 else 0
        
        return {
            "user_id": user_id,
            "tutorial_id": self.tutorial_id,
            "total_content_blocks": total_blocks,
            "completed_blocks": completed_blocks,
            "progress_percentage": round(progress_percentage, 2),
            "completed_chapters": len(set(r.chapter_id for r in user_records if r.completed)),
            "total_chapters": len(self.chapters),
            "average_score": self._calculate_average_score(user_records),
            "last_activity": max([r.timestamp for r in user_records], default=None)
        }
    
    def generate_tutorial_html(self, user_id: str = None) -> str:
        """
        生成HTML格式的教程
        
        Args:
            user_id: 用户ID（用于个性化显示）
            
        Returns:
            str: HTML内容
        """
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .chapter {{ margin-bottom: 30px; border: 1px solid #ddd; border-radius: 8px; }}
        .chapter-header {{ background: #007bff; color: white; padding: 15px; border-radius: 8px 8px 0 0; }}
        .content-block {{ padding: 15px; border-bottom: 1px solid #eee; }}
        .content-block:last-child {{ border-bottom: none; }}
        .code-block {{ background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; }}
        .quiz {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .exercise {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .progress-bar {{ width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: #28a745; transition: width 0.3s; }}
        .difficulty {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; }}
        .difficulty.beginner {{ background: #d4edda; color: #155724; }}
        .difficulty.intermediate {{ background: #fff3cd; color: #856404; }}
        .difficulty.advanced {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p>{self.description}</p>
        <div class="difficulty {self.difficulty.value.lower()}">{self.difficulty.value}</div>
"""
        
        # 添加进度条（如果提供了用户ID）
        if user_id:
            progress = self.get_user_progress(user_id)
            html += f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress['progress_percentage']}%"></div>
        </div>
        <p>学习进度: {progress['completed_blocks']}/{progress['total_content_blocks']} ({progress['progress_percentage']}%)</p>
"""
        
        html += "    </div>\n"
        
        # 生成章节内容
        for chapter in self.chapters:
            html += f"""
    <div class="chapter">
        <div class="chapter-header">
            <h2>{chapter.title}</h2>
            <p>{chapter.description}</p>
            <small>预计学习时间: {chapter.estimated_time}分钟</small>
            <div class="difficulty {chapter.difficulty.value.lower()}">{chapter.difficulty.value}</div>
        </div>
"""
            
            for content_block in chapter.content_blocks:
                html += f"""
        <div class="content-block">
            <h3>{content_block.title}</h3>
"""
                
                if content_block.type == ContentType.TEXT:
                    html += f"            <p>{content_block.content}</p>\n"
                
                elif content_block.type == ContentType.CODE and content_block.code_example:
                    html += f"""
            <div class="code-block">
                <h4>{content_block.code_example.language} 示例</h4>
                <pre><code>{content_block.code_example.code}</code></pre>
"""
                    if content_block.code_example.output:
                        html += f"                <p><strong>输出:</strong></p>\n                <pre><code>{content_block.code_example.output}</code></pre>\n"
                    if content_block.code_example.explanation:
                        html += f"                <p><em>{content_block.code_example.explanation}</em></p>\n"
                    html += "            </div>\n"
                
                elif content_block.type == ContentType.QUIZ and content_block.quiz_question:
                    quiz = content_block.quiz_question
                    html += f"""
            <div class="quiz">
                <h4>测验</h4>
                <p><strong>{quiz.question}</strong></p>
                <ul>
"""
                    for i, option in enumerate(quiz.options):
                        html += f"                    <li>{chr(65 + i)}. {option}</li>\n"
                    html += "                </ul>\n                <p><em>选择答案后查看解释</em></p>\n            </div>\n"
                
                elif content_block.type == ContentType.EXERCISE and content_block.exercise:
                    exercise = content_block.exercise
                    html += f"""
            <div class="exercise">
                <h4>练习: {exercise.title}</h4>
                <p>{exercise.description}</p>
                <h5>要求:</h5>
                <ul>
"""
                    for req in exercise.requirements:
                        html += f"                    <li>{req}</li>\n"
                    html += "                </ul>\n                <h5>提示:</h5>\n                <ul>\n"
                    for hint in exercise.hints:
                        html += f"                    <li>{hint}</li>\n"
                    html += "                </ul>\n            </div>\n"
                
                html += "        </div>\n"
            
            html += "    </div>\n"
        
        html += """
</body>
</html>"""
        return html
    
    def export_to_json(self, file_path: str) -> None:
        """
        导出教程为JSON格式
        
        Args:
            file_path: 输出文件路径
        """
        tutorial_data = {
            "metadata": self.metadata,
            "tutorial_info": {
                "id": self.tutorial_id,
                "title": self.title,
                "description": self.description,
                "difficulty": self.difficulty.value
            },
            "chapters": [asdict(chapter) for chapter in self.chapters],
            "progress_records": [asdict(record) for record in self.progress_records]
        }
        
        # 处理枚举类型转换
        for chapter_data in tutorial_data["chapters"]:
            chapter_data["difficulty"] = chapter_data["difficulty"].value if hasattr(chapter_data["difficulty"], 'value') else chapter_data["difficulty"]
            for block in chapter_data["content_blocks"]:
                block["type"] = block["type"].value if hasattr(block["type"], 'value') else block["type"]
                if block.get("quiz_question"):
                    block["quiz_question"]["difficulty"] = block["quiz_question"]["difficulty"].value if hasattr(block["quiz_question"]["difficulty"], 'value') else block["quiz_question"]["difficulty"]
                if block.get("exercise"):
                    block["exercise"]["difficulty"] = block["exercise"]["difficulty"].value if hasattr(block["exercise"]["difficulty"], 'value') else block["exercise"]["difficulty"]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(tutorial_data, f, ensure_ascii=False, indent=2)
    
    def _find_chapter(self, chapter_id: str) -> Optional[Chapter]:
        """查找章节"""
        for chapter in self.chapters:
            if chapter.id == chapter_id:
                return chapter
        return None
    
    def _find_content_block(self, content_block_id: str) -> Optional[ContentBlock]:
        """查找内容块"""
        for chapter in self.chapters:
            for content_block in chapter.content_blocks:
                if content_block.id == content_block_id:
                    return content_block
        return None
    
    def _calculate_average_score(self, records: List[ProgressRecord]) -> float:
        """计算平均分"""
        scores = [r.score for r in records if r.score is not None]
        return round(sum(scores) / len(scores), 2) if scores else 0.0
    
    def get_knowledge_check_results(self, user_id: str) -> Dict[str, Any]:
        """
        获取知识检查结果
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict: 知识检查结果
        """
        user_records = [r for r in self.progress_records if r.user_id == user_id]
        
        quiz_records = [r for r in user_records if "quiz_" in r.content_block_id]
        exercise_records = [r for r in user_records if "exercise_" in r.content_block_id]
        
        quiz_scores = [r.score for r in quiz_records if r.score is not None]
        exercise_scores = [r.score for r in exercise_records if r.score is not None]
        
        return {
            "quiz_results": {
                "total_attempts": len(quiz_records),
                "average_score": round(sum(quiz_scores) / len(quiz_scores), 2) if quiz_scores else 0.0,
                "passed_count": len([s for s in quiz_scores if s >= 60]),
                "pass_rate": round(len([s for s in quiz_scores if s >= 60]) / len(quiz_scores) * 100, 2) if quiz_scores else 0.0
            },
            "exercise_results": {
                "total_attempts": len(exercise_records),
                "average_score": round(sum(exercise_scores) / len(exercise_scores), 2) if exercise_scores else 0.0,
                "completed_count": len([r for r in exercise_records if r.completed]),
                "completion_rate": round(len([r for r in exercise_records if r.completed]) / len(exercise_records) * 100, 2) if exercise_records else 0.0
            }
        }


def create_sample_tutorial() -> TutorialGenerator:
    """
    创建示例教程
    
    Returns:
        TutorialGenerator: 示例教程生成器
    """
    # 创建教程
    tutorial = TutorialGenerator(
        tutorial_id="python_basics_001",
        title="Python基础教程",
        description="从零开始学习Python编程语言的基础知识",
        difficulty=DifficultyLevel.BEGINNER
    )
    
    # 创建第一章
    chapter1 = tutorial.create_chapter(
        title="Python简介与环境配置",
        description="了解Python语言的基本概念和开发环境搭建",
        estimated_time=30,
        difficulty=DifficultyLevel.BEGINNER
    )
    
    # 添加文本内容
    tutorial.create_text_content(
        chapter1.id,
        "什么是Python",
        "Python是一种高级编程语言，具有简洁易读的语法和强大的功能。它被广泛应用于Web开发、数据科学、人工智能等领域。"
    )
    
    # 添加代码示例
    tutorial.create_code_example(
        chapter1.id,
        "第一个Python程序",
        "python",
        "print('Hello, World!')",
        "经典的Hello World程序",
        "Hello, World!",
        "这是最简单的Python程序，用于输出文本到控制台"
    )
    
    # 添加测验
    tutorial.create_quiz(
        chapter1.id,
        "Python特点测验",
        "Python语言的主要特点是什么？",
        ["编译型语言", "解释型语言", "汇编语言", "机器语言"],
        1,
        "Python是解释型语言，这意味着代码不需要编译就可以直接运行。",
        DifficultyLevel.BEGINNER
    )
    
    # 添加练习
    tutorial.create_exercise(
        chapter1.id,
        "输出练习",
        "编写一个Python程序，输出你的姓名和年龄",
        ["使用print()函数", "输出格式为：姓名: XXX, 年龄: XX"],
        ["使用字符串格式化", "可以尝试f-string语法"],
        'name = "张三"\nage = 20\nprint(f"姓名: {name}, 年龄: {age}")',
        DifficultyLevel.BEGINNER
    )
    
    # 添加逐步指导
    tutorial.generate_step_by_step_guide(
        chapter1.id,
        [
            "下载并安装Python解释器",
            "配置环境变量",
            "安装代码编辑器（如VS Code）",
            "创建第一个Python文件",
            "运行程序并查看结果"
        ]
    )
    
    return tutorial


if __name__ == "__main__":
    # 创建并测试示例教程
    tutorial = create_sample_tutorial()
    
    # 生成HTML
    html_content = tutorial.generate_tutorial_html("user_001")
    
    # 保存HTML文件
    with open("sample_tutorial.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # 模拟用户进度
    tutorial.track_progress("user_001", "chapter_1", "content_1", True, 100)
    tutorial.track_progress("user_001", "chapter_1", "code_1", True, 95)
    tutorial.track_progress("user_001", "chapter_1", "quiz_1", True, 80)
    
    # 获取进度信息
    progress = tutorial.get_user_progress("user_001")
    print("用户进度:", progress)
    
    # 获取知识检查结果
    knowledge_check = tutorial.get_knowledge_check_results("user_001")
    print("知识检查结果:", knowledge_check)
    
    # 导出JSON
    tutorial.export_to_json("sample_tutorial.json")
    
    print("示例教程已生成完成！")