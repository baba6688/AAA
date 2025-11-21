# -*- coding: utf-8 -*-
"""
智能量化交易系统 - 完整闭环版
Intelligent Quantitative Trading System - Complete Closed Loop Version

系统架构：8层智能架构
1. 外部环境层 (A区) - 环境监控、数据采集
2. 感知与认知层 (B区) - 市场感知、模式识别  
3. 知识与推理层 (C区) - 知识图谱、因果推理
4. 自我意识层 (D区) - 自我认知、能力评估
5. 创新与生成层 (E区) - 策略生成、创意引擎
6. 多层次学习 (F区) - 5层学习体系
7. 决策与执行 (G区) - 智能决策、交易执行
8. 反馈与进化 (H区) - 反思进化、系统升级


日期: 2025-11-05
"""

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "agent@minimax.com"

# 核心模块导入
from . import A  # 外部环境层
from . import B  # 感知与认知层
from . import C  # 知识与推理层
from . import D  # 自我意识层
from . import E  # 创新与生成层
from . import F  # 多层次学习
from . import G  # 决策与执行
from . import H  # 反馈与进化

# 系统配置
SYSTEM_CONFIG = {
    "name": "智能量化交易系统",
    "version": __version__,
    "architecture": "8层智能架构",
    "closed_loop": True,
    "self_optimizing": True,
    "self_evolving": True,
    "layers": {
        "A": "外部环境层",
        "B": "感知与认知层", 
        "C": "知识与推理层",
        "D": "自我意识层",
        "E": "创新与生成层",
        "F": "多层次学习",
        "G": "决策与执行",
        "H": "反馈与进化"
    }
}

# 系统状态
SYSTEM_STATUS = {
    "initialized": False,
    "running": False,
    "learning": False,
    "evolving": False,
    "current_layer": None,
    "performance_metrics": {},
    "last_update": None
}