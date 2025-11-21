#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C5规则引擎系统
实现完整的规则引擎功能，包括规则定义、匹配、执行、冲突解决、学习优化等
"""

import json
import uuid
import time
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import re
import copy


class RuleType(Enum):
    """规则类型枚举"""
    IF_THEN = "if_then"           # IF-THEN规则
    DECISION_TABLE = "decision_table"  # 决策表
    DECISION_TREE = "decision_tree"    # 决策树
    FUZZY_RULE = "fuzzy_rule"     # 模糊规则
    TEMPORAL_RULE = "temporal_rule"    # 时序规则


class ConflictResolutionStrategy(Enum):
    """冲突解决策略"""
    FIRST_MATCH = "first_match"       # 第一个匹配
    PRIORITY_BASED = "priority_based" # 基于优先级
    WEIGHT_BASED = "weight_based"     # 基于权重
    RECENCY_BASED = "recency_based"   # 基于最近使用
    LEARNING_BASED = "learning_based" # 基于学习


@dataclass
class RuleMetadata:
    """规则元数据"""
    id: str
    name: str
    description: str
    version: str
    created_time: datetime
    updated_time: datetime
    author: str
    priority: int = 1
    weight: float = 1.0
    category: str = "default"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class RuleExecutionResult:
    """规则执行结果"""
    rule_id: str
    rule_name: str
    success: bool
    result: Any
    execution_time: float
    confidence: float
    explanation: str
    timestamp: datetime
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    last_execution: Optional[datetime] = None
    usage_frequency: float = 0.0


class Rule(ABC):
    """规则抽象基类"""
    
    def __init__(self, metadata: RuleMetadata, rule_type: RuleType):
        self.metadata = metadata
        self.rule_type = rule_type
        self.performance = PerformanceMetrics()
        self.is_active = True
        self.conditions = []
        self.actions = []
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """评估规则条件"""
        pass
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> RuleExecutionResult:
        """执行规则"""
        pass
    
    def get_explanation(self, context: Dict[str, Any]) -> str:
        """获取规则解释"""
        return f"规则 '{self.metadata.name}' (类型: {self.rule_type.value})"
    
    def update_performance(self, execution_result: RuleExecutionResult):
        """更新性能指标"""
        self.performance.total_executions += 1
        if execution_result.success:
            self.performance.successful_executions += 1
        else:
            self.performance.failed_executions += 1
        
        # 更新平均执行时间
        total_time = self.performance.average_execution_time * (self.performance.total_executions - 1)
        self.performance.average_execution_time = (total_time + execution_result.execution_time) / self.performance.total_executions
        
        # 更新成功率
        self.performance.success_rate = self.performance.successful_executions / self.performance.total_executions
        self.performance.last_execution = execution_result.timestamp


class IfThenRule(Rule):
    """IF-THEN规则实现"""
    
    def __init__(self, metadata: RuleMetadata, conditions: List[str], actions: List[str]):
        super().__init__(metadata, RuleType.IF_THEN)
        self.conditions = conditions
        self.actions = actions
        self.condition_functions = []
        self.action_functions = []
        self._compile_conditions()
        self._compile_actions()
    
    def _compile_conditions(self):
        """编译条件表达式"""
        for condition in self.conditions:
            # 简单的条件编译，实际应用中可能需要更复杂的解析器
            func = self._create_condition_function(condition)
            self.condition_functions.append(func)
    
    def _compile_actions(self):
        """编译动作表达式"""
        for action in self.actions:
            func = self._create_action_function(action)
            self.action_functions.append(func)
    
    def _create_condition_function(self, condition: str) -> Callable:
        """创建条件函数"""
        def condition_func(context: Dict[str, Any]) -> bool:
            try:
                # 简单的条件评估，实际应用中需要更安全的表达式解析
                return eval(condition, {"__builtins__": {}}, context)
            except Exception as e:
                logging.warning(f"条件评估错误: {condition}, 错误: {e}")
                return False
        return condition_func
    
    def _create_action_function(self, action: str) -> Callable:
        """创建动作函数"""
        def action_func(context: Dict[str, Any]) -> Any:
            try:
                # 简单的动作执行，实际应用中需要更安全的表达式执行
                result = eval(action, {"__builtins__": {}}, context)
                return result
            except Exception as e:
                logging.warning(f"动作执行错误: {action}, 错误: {e}")
                return None
        return action_func
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """评估IF-THEN规则"""
        try:
            # 评估所有条件
            for condition_func in self.condition_functions:
                if not condition_func(context):
                    return False, None, "条件不满足"
            
            # 执行动作
            results = []
            for action_func in self.action_functions:
                result = action_func(context)
                results.append(result)
            
            return True, results, "所有条件满足，动作执行成功"
        except Exception as e:
            return False, None, f"规则执行错误: {str(e)}"
    
    def execute(self, context: Dict[str, Any]) -> RuleExecutionResult:
        """执行IF-THEN规则"""
        start_time = time.time()
        
        try:
            success, result, explanation = self.evaluate(context)
            execution_time = time.time() - start_time
            
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=success,
                result=result,
                execution_time=execution_time,
                confidence=1.0 if success else 0.0,
                explanation=explanation,
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=False,
                result=None,
                execution_time=execution_time,
                confidence=0.0,
                explanation=f"执行异常: {str(e)}",
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result


class DecisionTableRule(Rule):
    """决策表规则实现"""
    
    def __init__(self, metadata: RuleMetadata, decision_table: pd.DataFrame):
        super().__init__(metadata, RuleType.DECISION_TABLE)
        self.decision_table = decision_table
        self.input_columns = []
        self.output_columns = []
        self._analyze_table()
    
    def _analyze_table(self):
        """分析决策表结构"""
        if not self.decision_table.empty:
            # 假设最后一列为输出列，其他为输入列
            self.input_columns = list(self.decision_table.columns[:-1])
            self.output_columns = [self.decision_table.columns[-1]]
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """评估决策表规则"""
        try:
            # 匹配输入条件
            matched_rows = self.decision_table.copy()
            
            for col in self.input_columns:
                if col in context:
                    matched_rows = matched_rows[matched_rows[col] == context[col]]
            
            if matched_rows.empty:
                return False, None, "没有匹配的决策表行"
            
            # 返回第一个匹配的结果
            result = matched_rows.iloc[0][self.output_columns[0]]
            return True, result, f"决策表匹配成功: {result}"
            
        except Exception as e:
            return False, None, f"决策表评估错误: {str(e)}"
    
    def execute(self, context: Dict[str, Any]) -> RuleExecutionResult:
        """执行决策表规则"""
        start_time = time.time()
        
        try:
            success, result, explanation = self.evaluate(context)
            execution_time = time.time() - start_time
            
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=success,
                result=result,
                execution_time=execution_time,
                confidence=1.0 if success else 0.0,
                explanation=explanation,
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=False,
                result=None,
                execution_time=execution_time,
                confidence=0.0,
                explanation=f"执行异常: {str(e)}",
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result


class DecisionTreeRule(Rule):
    """决策树规则实现"""
    
    def __init__(self, metadata: RuleMetadata, tree_structure: Dict[str, Any]):
        super().__init__(metadata, RuleType.DECISION_TREE)
        self.tree_structure = tree_structure
    
    def _evaluate_node(self, node: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """评估决策树节点"""
        if 'value' in node:  # 叶子节点
            return True, node['value'], f"决策树叶子节点: {node['value']}"
        
        if 'condition' in node and 'children' in node:
            # 评估条件
            condition = node['condition']
            try:
                condition_met = eval(condition, {"__builtins__": {}}, context)
                
                if condition_met and 'true' in node['children']:
                    return self._evaluate_node(node['children']['true'], context)
                elif not condition_met and 'false' in node['children']:
                    return self._evaluate_node(node['children']['false'], context)
                else:
                    return False, None, "决策树条件不匹配"
            except Exception as e:
                return False, None, f"决策树条件评估错误: {str(e)}"
        
        return False, None, "无效的决策树节点"
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """评估决策树规则"""
        return self._evaluate_node(self.tree_structure, context)
    
    def execute(self, context: Dict[str, Any]) -> RuleExecutionResult:
        """执行决策树规则"""
        start_time = time.time()
        
        try:
            success, result, explanation = self.evaluate(context)
            execution_time = time.time() - start_time
            
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=success,
                result=result,
                execution_time=execution_time,
                confidence=1.0 if success else 0.0,
                explanation=explanation,
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=False,
                result=None,
                execution_time=execution_time,
                confidence=0.0,
                explanation=f"执行异常: {str(e)}",
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result


class FuzzyRule(Rule):
    """模糊规则实现"""
    
    def __init__(self, metadata: RuleMetadata, fuzzy_conditions: Dict[str, Dict], fuzzy_actions: Dict[str, Any]):
        super().__init__(metadata, RuleType.FUZZY_RULE)
        self.fuzzy_conditions = fuzzy_conditions
        self.fuzzy_actions = fuzzy_actions
        self.membership_functions = {}
        self._initialize_membership_functions()
    
    def _initialize_membership_functions(self):
        """初始化隶属度函数"""
        # 简单的三角形隶属度函数
        self.membership_functions = {
            'low': lambda x: max(0, min(1, (0.3 - x) / 0.3)) if x <= 0.3 else 0,
            'medium': lambda x: max(0, min(1, 1 - abs(x - 0.5) / 0.3)) if 0.2 <= x <= 0.8 else 0,
            'high': lambda x: max(0, min(1, (x - 0.7) / 0.3)) if x >= 0.7 else 0
        }
    
    def _calculate_membership(self, variable: str, value: float) -> Dict[str, float]:
        """计算隶属度"""
        memberships = {}
        for fuzzy_set, func in self.membership_functions.items():
            memberships[fuzzy_set] = func(value)
        return memberships
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """评估模糊规则"""
        try:
            # 计算每个条件的隶属度
            condition_scores = []
            
            for var, fuzzy_info in self.fuzzy_conditions.items():
                if var in context:
                    value = context[var]
                    memberships = self._calculate_membership(var, value)
                    
                    # 计算条件满足度
                    condition_score = 0
                    for fuzzy_set, expected_membership in fuzzy_info.items():
                        if fuzzy_set in memberships:
                            condition_score = max(condition_score, 
                                                min(expected_membership, memberships[fuzzy_set]))
                    
                    condition_scores.append(condition_score)
            
            if not condition_scores:
                return False, None, "没有有效的模糊条件"
            
            # 使用最小值作为规则激活度
            activation_degree = min(condition_scores)
            
            if activation_degree > 0.5:  # 阈值
                # 执行模糊动作
                result = self._defuzzify(activation_degree)
                return True, result, f"模糊规则激活度: {activation_degree:.2f}"
            else:
                return False, None, f"模糊规则激活度不足: {activation_degree:.2f}"
                
        except Exception as e:
            return False, None, f"模糊规则评估错误: {str(e)}"
    
    def _defuzzify(self, activation_degree: float) -> Any:
        """去模糊化"""
        # 简单的去模糊化实现
        result = {}
        for action_var, action_info in self.fuzzy_actions.items():
            if isinstance(action_info, dict):
                # 计算加权平均值
                weighted_sum = 0
                weight_sum = 0
                for fuzzy_set, weight in action_info.items():
                    if fuzzy_set in self.membership_functions:
                        # 假设模糊集的中心值
                        center_value = {'low': 0.2, 'medium': 0.5, 'high': 0.8}.get(fuzzy_set, 0.5)
                        weighted_sum += center_value * weight * activation_degree
                        weight_sum += weight * activation_degree
                
                if weight_sum > 0:
                    result[action_var] = weighted_sum / weight_sum
                else:
                    result[action_var] = 0.5
            else:
                result[action_var] = action_info * activation_degree
        
        return result
    
    def execute(self, context: Dict[str, Any]) -> RuleExecutionResult:
        """执行模糊规则"""
        start_time = time.time()
        
        try:
            success, result, explanation = self.evaluate(context)
            execution_time = time.time() - start_time
            
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=success,
                result=result,
                execution_time=execution_time,
                confidence=1.0 if success else 0.0,
                explanation=explanation,
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=False,
                result=None,
                execution_time=execution_time,
                confidence=0.0,
                explanation=f"执行异常: {str(e)}",
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result


class TemporalRule(Rule):
    """时序规则实现"""
    
    def __init__(self, metadata: RuleMetadata, temporal_conditions: List[Dict], actions: List[str]):
        super().__init__(metadata, RuleType.TEMPORAL_RULE)
        self.temporal_conditions = temporal_conditions
        self.actions = actions
        self.event_history = deque(maxlen=1000)  # 事件历史
    
    def add_event(self, event: Dict[str, Any]):
        """添加事件到历史"""
        event['timestamp'] = datetime.now()
        self.event_history.append(event)
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """评估时序规则"""
        try:
            # 检查时序条件
            for condition in self.temporal_conditions:
                if not self._check_temporal_condition(condition):
                    return False, None, f"时序条件不满足: {condition}"
            
            # 执行动作
            results = []
            for action in self.actions:
                try:
                    result = eval(action, {"__builtins__": {}}, context)
                    results.append(result)
                except Exception as e:
                    logging.warning(f"时序规则动作执行错误: {action}, 错误: {e}")
            
            return True, results, "时序条件满足，动作执行成功"
            
        except Exception as e:
            return False, None, f"时序规则评估错误: {str(e)}"
    
    def _check_temporal_condition(self, condition: Dict[str, Any]) -> bool:
        """检查时序条件"""
        condition_type = condition.get('type')
        
        if condition_type == 'after':
            # 检查事件是否在指定时间之后发生
            event_type = condition.get('event')
            time_offset = condition.get('time_offset', 0)
            target_time = datetime.now() - timedelta(seconds=time_offset)
            
            for event in reversed(self.event_history):
                if event.get('type') == event_type and event['timestamp'] > target_time:
                    return True
            return False
        
        elif condition_type == 'within':
            # 检查事件是否在指定时间窗口内发生
            event_type = condition.get('event')
            time_window = condition.get('time_window', 0)
            target_time = datetime.now() - timedelta(seconds=time_window)
            
            recent_events = [e for e in self.event_history if e['timestamp'] > target_time]
            return any(e.get('type') == event_type for e in recent_events)
        
        elif condition_type == 'sequence':
            # 检查事件序列
            event_sequence = condition.get('sequence', [])
            if len(event_sequence) > len(self.event_history):
                return False
            
            # 检查最近的事件序列
            for i in range(len(event_sequence)):
                expected_event = event_sequence[-(i+1)]
                actual_event = self.event_history[-(i+1)]
                if actual_event.get('type') != expected_event:
                    return False
            return True
        
        return False
    
    def execute(self, context: Dict[str, Any]) -> RuleExecutionResult:
        """执行时序规则"""
        start_time = time.time()
        
        try:
            success, result, explanation = self.evaluate(context)
            execution_time = time.time() - start_time
            
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=success,
                result=result,
                execution_time=execution_time,
                confidence=1.0 if success else 0.0,
                explanation=explanation,
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_result = RuleExecutionResult(
                rule_id=self.metadata.id,
                rule_name=self.metadata.name,
                success=False,
                result=None,
                execution_time=execution_time,
                confidence=0.0,
                explanation=f"执行异常: {str(e)}",
                timestamp=datetime.now(),
                context=context
            )
            
            self.update_performance(execution_result)
            return execution_result


class RuleConflictDetector:
    """规则冲突检测器"""
    
    def __init__(self):
        self.conflict_rules = defaultdict(list)
    
    def detect_conflicts(self, rules: List[Rule]) -> List[Dict[str, Any]]:
        """检测规则冲突"""
        conflicts = []
        
        # 检测相同条件的不同结果冲突
        condition_groups = defaultdict(list)
        for rule in rules:
            if isinstance(rule, IfThenRule):
                condition_key = tuple(sorted(rule.conditions))
                condition_groups[condition_key].append(rule)
        
        for condition_key, rule_list in condition_groups.items():
            if len(rule_list) > 1:
                conflicts.append({
                    'type': 'condition_conflict',
                    'condition': condition_key,
                    'rules': [rule.metadata.id for rule in rule_list],
                    'description': f"条件 {condition_key} 有 {len(rule_list)} 个规则"
                })
        
        # 检测优先级冲突
        priority_groups = defaultdict(list)
        for rule in rules:
            priority_groups[rule.metadata.priority].append(rule)
        
        for priority, rule_list in priority_groups.items():
            if len(rule_list) > 1:
                conflicts.append({
                    'type': 'priority_conflict',
                    'priority': priority,
                    'rules': [rule.metadata.id for rule in rule_list],
                    'description': f"优先级 {priority} 有 {len(rule_list)} 个规则"
                })
        
        return conflicts
    
    def suggest_resolution(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """建议冲突解决方案"""
        suggestions = []
        
        for conflict in conflicts:
            if conflict['type'] == 'condition_conflict':
                suggestions.append({
                    'conflict_id': conflict,
                    'strategy': 'priority_based',
                    'description': '使用优先级解决条件冲突',
                    'action': '设置不同的优先级值'
                })
            elif conflict['type'] == 'priority_conflict':
                suggestions.append({
                    'conflict_id': conflict,
                    'strategy': 'weight_based',
                    'description': '使用权重解决优先级冲突',
                    'action': '设置不同的权重值'
                })
        
        return suggestions


class RuleOptimizer:
    """规则优化器"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_rules(self, rules: List[Rule]) -> List[Rule]:
        """优化规则"""
        optimized_rules = []
        
        for rule in rules:
            optimized_rule = self._optimize_single_rule(rule)
            optimized_rules.append(optimized_rule)
        
        return optimized_rules
    
    def _optimize_single_rule(self, rule: Rule) -> Rule:
        """优化单个规则"""
        # 基于性能数据优化
        if rule.performance.total_executions > 10:
            # 如果成功率太低，考虑修改
            if rule.performance.success_rate < 0.5:
                logging.info(f"规则 {rule.metadata.name} 成功率较低，考虑优化")
            
            # 如果执行时间太长，考虑优化
            if rule.performance.average_execution_time > 1.0:
                logging.info(f"规则 {rule.metadata.name} 执行时间较长，考虑优化")
        
        return rule
    
    def suggest_improvements(self, rules: List[Rule]) -> List[Dict[str, Any]]:
        """建议改进"""
        suggestions = []
        
        for rule in rules:
            if rule.performance.success_rate < 0.7:
                suggestions.append({
                    'rule_id': rule.metadata.id,
                    'rule_name': rule.metadata.name,
                    'issue': 'success_rate_low',
                    'suggestion': '提高规则条件的准确性',
                    'current_success_rate': rule.performance.success_rate
                })
            
            if rule.performance.average_execution_time > 0.5:
                suggestions.append({
                    'rule_id': rule.metadata.id,
                    'rule_name': rule.metadata.name,
                    'issue': 'execution_time_high',
                    'suggestion': '简化规则条件或优化条件评估',
                    'current_execution_time': rule.performance.average_execution_time
                })
        
        return suggestions


class RuleValidator:
    """规则验证器"""
    
    def __init__(self):
        self.validation_rules = []
    
    def validate_rule(self, rule: Rule) -> Tuple[bool, List[str]]:
        """验证单个规则"""
        errors = []
        
        # 检查基本属性
        if not rule.metadata.name:
            errors.append("规则名称不能为空")
        
        if not rule.metadata.id:
            errors.append("规则ID不能为空")
        
        # 检查规则类型特定的条件
        if isinstance(rule, IfThenRule):
            if not rule.conditions:
                errors.append("IF-THEN规则必须有条件")
            if not rule.actions:
                errors.append("IF-THEN规则必须有动作")
        
        elif isinstance(rule, DecisionTableRule):
            if rule.decision_table.empty:
                errors.append("决策表不能为空")
        
        elif isinstance(rule, DecisionTreeRule):
            if not rule.tree_structure:
                errors.append("决策树结构不能为空")
        
        return len(errors) == 0, errors
    
    def validate_rule_set(self, rules: List[Rule]) -> Dict[str, Any]:
        """验证规则集"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'rule_count': len(rules)
        }
        
        for rule in rules:
            rule_valid, rule_errors = self.validate_rule(rule)
            if not rule_valid:
                validation_result['valid'] = False
                validation_result['errors'].extend([f"规则 {rule.metadata.name}: {error}" for error in rule_errors])
        
        # 检查规则集一致性
        rule_names = [rule.metadata.name for rule in rules]
        if len(rule_names) != len(set(rule_names)):
            validation_result['warnings'].append("存在重复的规则名称")
        
        return validation_result


class RuleVersionManager:
    """规则版本管理器"""
    
    def __init__(self):
        self.rule_versions = defaultdict(list)
        self.current_versions = {}
    
    def create_version(self, rule: Rule) -> str:
        """创建规则版本"""
        version_id = str(uuid.uuid4())
        version_data = {
            'version_id': version_id,
            'rule': copy.deepcopy(rule),
            'timestamp': datetime.now(),
            'hash': self._calculate_rule_hash(rule)
        }
        
        self.rule_versions[rule.metadata.id].append(version_data)
        self.current_versions[rule.metadata.id] = version_id
        
        return version_id
    
    def get_version(self, rule_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        """获取指定版本"""
        if rule_id in self.rule_versions:
            for version in self.rule_versions[rule_id]:
                if version['version_id'] == version_id:
                    return version
        return None
    
    def get_current_version(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """获取当前版本"""
        if rule_id in self.current_versions:
            return self.get_version(rule_id, self.current_versions[rule_id])
        return None
    
    def list_versions(self, rule_id: str) -> List[Dict[str, Any]]:
        """列出所有版本"""
        return self.rule_versions.get(rule_id, [])
    
    def rollback(self, rule_id: str, version_id: str) -> bool:
        """回滚到指定版本"""
        version_data = self.get_version(rule_id, version_id)
        if version_data:
            self.current_versions[rule_id] = version_id
            return True
        return False
    
    def _calculate_rule_hash(self, rule: Rule) -> str:
        """计算规则哈希值"""
        rule_data = {
            'name': rule.metadata.name,
            'conditions': getattr(rule, 'conditions', []),
            'actions': getattr(rule, 'actions', []),
            'tree_structure': getattr(rule, 'tree_structure', None),
            'decision_table': getattr(rule, 'decision_table', None).to_dict() if hasattr(rule, 'decision_table') and not rule.decision_table.empty else None
        }
        
        rule_string = json.dumps(rule_data, sort_keys=True, default=str)
        return hashlib.md5(rule_string.encode()).hexdigest()


class RuleEngine:
    """C5规则引擎主类"""
    
    def __init__(self, 
                 conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.PRIORITY_BASED):
        self.rules: Dict[str, Rule] = {}
        self.conflict_detector = RuleConflictDetector()
        self.optimizer = RuleOptimizer()
        self.validator = RuleValidator()
        self.version_manager = RuleVersionManager()
        self.conflict_resolution = conflict_resolution
        self.execution_history = deque(maxlen=10000)
        self.performance_metrics = PerformanceMetrics()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, rule: Rule) -> bool:
        """添加规则"""
        try:
            # 验证规则
            is_valid, errors = self.validator.validate_rule(rule)
            if not is_valid:
                self.logger.error(f"规则验证失败: {errors}")
                return False
            
            # 创建版本
            version_id = self.version_manager.create_version(rule)
            
            # 添加规则
            self.rules[rule.metadata.id] = rule
            
            self.logger.info(f"规则添加成功: {rule.metadata.name}, 版本: {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加规则失败: {str(e)}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """删除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"规则删除成功: {rule_id}")
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """列出所有规则"""
        rule_list = []
        for rule_id, rule in self.rules.items():
            rule_info = {
                'id': rule_id,
                'name': rule.metadata.name,
                'type': rule.rule_type.value,
                'priority': rule.metadata.priority,
                'weight': rule.metadata.weight,
                'active': rule.is_active,
                'performance': asdict(rule.performance)
            }
            rule_list.append(rule_info)
        return rule_list
    
    def execute_rules(self, context: Dict[str, Any]) -> List[RuleExecutionResult]:
        """执行规则"""
        results = []
        matched_rules = []
        
        # 匹配规则
        for rule in self.rules.values():
            if rule.is_active:
                try:
                    success, _, explanation = rule.evaluate(context)
                    if success:
                        matched_rules.append((rule, explanation))
                except Exception as e:
                    self.logger.warning(f"规则 {rule.metadata.name} 评估失败: {str(e)}")
        
        if not matched_rules:
            self.logger.info("没有匹配的规则")
            return results
        
        # 解决冲突并执行
        execution_order = self._resolve_conflicts(matched_rules)
        
        for rule, explanation in execution_order:
            try:
                result = rule.execute(context)
                result.explanation = f"{explanation} -> {result.explanation}"
                results.append(result)
                self.execution_history.append(result)
                
                # 更新引擎性能指标
                self._update_engine_performance(result)
                
            except Exception as e:
                self.logger.error(f"规则执行失败: {rule.metadata.name}, 错误: {str(e)}")
        
        return results
    
    def _resolve_conflicts(self, matched_rules: List[Tuple[Rule, str]]) -> List[Tuple[Rule, str]]:
        """解决规则冲突"""
        if self.conflict_resolution == ConflictResolutionStrategy.FIRST_MATCH:
            return matched_rules[:1]
        
        elif self.conflict_resolution == ConflictResolutionStrategy.PRIORITY_BASED:
            return sorted(matched_rules, key=lambda x: x[0].metadata.priority, reverse=True)
        
        elif self.conflict_resolution == ConflictResolutionStrategy.WEIGHT_BASED:
            return sorted(matched_rules, key=lambda x: x[0].metadata.weight, reverse=True)
        
        elif self.conflict_resolution == ConflictResolutionStrategy.RECENCY_BASED:
            return sorted(matched_rules, key=lambda x: x[0].performance.last_execution or datetime.min, reverse=True)
        
        elif self.conflict_resolution == ConflictResolutionStrategy.LEARNING_BASED:
            return sorted(matched_rules, key=lambda x: x[0].performance.success_rate, reverse=True)
        
        return matched_rules
    
    def _update_engine_performance(self, result: RuleExecutionResult):
        """更新引擎性能指标"""
        self.performance_metrics.total_executions += 1
        if result.success:
            self.performance_metrics.successful_executions += 1
        else:
            self.performance_metrics.failed_executions += 1
        
        # 更新平均执行时间
        total_time = self.performance_metrics.average_execution_time * (self.performance_metrics.total_executions - 1)
        self.performance_metrics.average_execution_time = (total_time + result.execution_time) / self.performance_metrics.total_executions
        
        # 更新成功率
        self.performance_metrics.success_rate = self.performance_metrics.successful_executions / self.performance_metrics.total_executions
        self.performance_metrics.last_execution = result.timestamp
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """检测规则冲突"""
        return self.conflict_detector.detect_conflicts(list(self.rules.values()))
    
    def optimize_rules(self) -> List[Rule]:
        """优化规则"""
        return self.optimizer.optimize_rules(list(self.rules.values()))
    
    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """建议改进"""
        return self.optimizer.suggest_improvements(list(self.rules.values()))
    
    def validate_rules(self) -> Dict[str, Any]:
        """验证规则"""
        return self.validator.validate_rule_set(list(self.rules.values()))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        rule_performances = {}
        for rule_id, rule in self.rules.items():
            rule_performances[rule_id] = {
                'name': rule.metadata.name,
                'performance': asdict(rule.performance)
            }
        
        return {
            'engine_performance': asdict(self.performance_metrics),
            'rule_performances': rule_performances,
            'total_rules': len(self.rules),
            'active_rules': sum(1 for rule in self.rules.values() if rule.is_active)
        }
    
    def explain_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """解释决策过程"""
        explanation = {
            'context': context,
            'matched_rules': [],
            'execution_order': [],
            'final_decision': None
        }
        
        # 分析匹配的规则
        for rule_id, rule in self.rules.items():
            if rule.is_active:
                try:
                    success, result, eval_explanation = rule.evaluate(context)
                    if success:
                        rule_explanation = {
                            'rule_id': rule_id,
                            'rule_name': rule.metadata.name,
                            'rule_type': rule.rule_type.value,
                            'explanation': rule.get_explanation(context),
                            'evaluation': eval_explanation,
                            'priority': rule.metadata.priority,
                            'weight': rule.metadata.weight
                        }
                        explanation['matched_rules'].append(rule_explanation)
                except Exception as e:
                    self.logger.warning(f"规则解释失败: {rule.metadata.name}, 错误: {str(e)}")
        
        # 排序执行顺序
        matched_rules = [(self.rules[r['rule_id']], r['explanation']) for r in explanation['matched_rules']]
        execution_order = self._resolve_conflicts(matched_rules)
        
        explanation['execution_order'] = [
            {
                'rule_name': rule.metadata.name,
                'rule_type': rule.rule_type.value,
                'explanation': exp
            }
            for rule, exp in execution_order
        ]
        
        # 执行规则获取最终决策
        results = self.execute_rules(context)
        if results:
            explanation['final_decision'] = {
                'success': results[0].success,
                'result': results[0].result,
                'explanation': results[0].explanation,
                'confidence': results[0].confidence
            }
        
        return explanation
    
    def export_rules(self, file_path: str):
        """导出规则"""
        export_data = {
            'rules': [],
            'version_info': {
                'export_time': datetime.now().isoformat(),
                'rule_count': len(self.rules)
            }
        }
        
        for rule_id, rule in self.rules.items():
            rule_data = {
                'metadata': asdict(rule.metadata),
                'rule_type': rule.rule_type.value,
                'conditions': getattr(rule, 'conditions', []),
                'actions': getattr(rule, 'actions', []),
                'tree_structure': getattr(rule, 'tree_structure', None),
                'decision_table': getattr(rule, 'decision_table', None).to_dict() if hasattr(rule, 'decision_table') and not rule.decision_table.empty else None,
                'fuzzy_conditions': getattr(rule, 'fuzzy_conditions', {}),
                'fuzzy_actions': getattr(rule, 'fuzzy_actions', {}),
                'temporal_conditions': getattr(rule, 'temporal_conditions', [])
            }
            export_data['rules'].append(rule_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"规则导出成功: {file_path}")
    
    def import_rules(self, file_path: str) -> bool:
        """导入规则"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            for rule_data in import_data['rules']:
                # 重建规则元数据
                metadata = RuleMetadata(**rule_data['metadata'])
                
                # 根据规则类型重建规则
                rule_type = RuleType(rule_data['rule_type'])
                
                if rule_type == RuleType.IF_THEN:
                    rule = IfThenRule(metadata, rule_data['conditions'], rule_data['actions'])
                elif rule_type == RuleType.DECISION_TABLE:
                    df = pd.DataFrame(rule_data['decision_table'])
                    rule = DecisionTableRule(metadata, df)
                elif rule_type == RuleType.DECISION_TREE:
                    rule = DecisionTreeRule(metadata, rule_data['tree_structure'])
                elif rule_type == RuleType.FUZZY_RULE:
                    rule = FuzzyRule(metadata, rule_data['fuzzy_conditions'], rule_data['fuzzy_actions'])
                elif rule_type == RuleType.TEMPORAL_RULE:
                    rule = TemporalRule(metadata, rule_data['temporal_conditions'], rule_data['actions'])
                else:
                    continue
                
                self.add_rule(rule)
            
            self.logger.info(f"规则导入成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"规则导入失败: {str(e)}")
            return False


# 示例使用函数
def create_sample_rules():
    """创建示例规则"""
    # 创建IF-THEN规则
    metadata1 = RuleMetadata(
        id=str(uuid.uuid4()),
        name="价格预警规则",
        description="当价格超过阈值时发出预警",
        version="1.0",
        created_time=datetime.now(),
        updated_time=datetime.now(),
        author="System",
        priority=1,
        weight=1.0
    )
    
    if_then_rule = IfThenRule(
        metadata1,
        conditions=["price > threshold", "volume > min_volume"],
        actions=["alert('价格预警')", "context['signal'] = 'BUY'"]
    )
    
    # 创建决策表规则
    metadata2 = RuleMetadata(
        id=str(uuid.uuid4()),
        name="交易决策表",
        description="基于价格和成交量的交易决策",
        version="1.0",
        created_time=datetime.now(),
        updated_time=datetime.now(),
        author="System",
        priority=2,
        weight=0.8
    )
    
    decision_data = {
        'price_level': ['low', 'medium', 'high', 'low', 'high'],
        'volume_level': ['high', 'low', 'medium', 'medium', 'high'],
        'action': ['BUY', 'HOLD', 'SELL', 'BUY', 'SELL']
    }
    decision_df = pd.DataFrame(decision_data)
    
    decision_rule = DecisionTableRule(metadata2, decision_df)
    
    # 创建决策树规则
    metadata3 = RuleMetadata(
        id=str(uuid.uuid4()),
        name="风险评估决策树",
        description="基于多个因素的风险评估",
        version="1.0",
        created_time=datetime.now(),
        updated_time=datetime.now(),
        author="System",
        priority=3,
        weight=0.9
    )
    
    tree_structure = {
        'condition': 'price_change > 0.05',
        'children': {
            'true': {
                'condition': 'volume > average_volume',
                'children': {
                    'true': {'value': 'HIGH_RISK'},
                    'false': {'value': 'MEDIUM_RISK'}
                }
            },
            'false': {
                'value': 'LOW_RISK'
            }
        }
    }
    
    tree_rule = DecisionTreeRule(metadata3, tree_structure)
    
    return [if_then_rule, decision_rule, tree_rule]


def demo_rule_engine():
    """演示规则引擎功能"""
    print("=== C5规则引擎演示 ===\n")
    
    # 创建规则引擎
    engine = RuleEngine(ConflictResolutionStrategy.PRIORITY_BASED)
    
    # 创建示例规则
    sample_rules = create_sample_rules()
    
    # 添加规则
    print("1. 添加规则:")
    for rule in sample_rules:
        success = engine.add_rule(rule)
        print(f"   添加规则 '{rule.metadata.name}': {'成功' if success else '失败'}")
    
    # 列出规则
    print("\n2. 规则列表:")
    rules = engine.list_rules()
    for rule in rules:
        print(f"   - {rule['name']} (类型: {rule['type']}, 优先级: {rule['priority']})")
    
    # 执行规则
    print("\n3. 执行规则:")
    test_contexts = [
        {'price': 100, 'threshold': 90, 'volume': 1000, 'min_volume': 500, 'price_change': 0.06, 'average_volume': 800},
        {'price': 80, 'threshold': 90, 'volume': 300, 'min_volume': 500, 'price_change': 0.02, 'average_volume': 800}
    ]
    
    for i, context in enumerate(test_contexts, 1):
        print(f"   测试场景 {i}: {context}")
        results = engine.execute_rules(context)
        for result in results:
            print(f"     规则 '{result.rule_name}': {'成功' if result.success else '失败'}")
            print(f"     结果: {result.result}")
            print(f"     解释: {result.explanation}")
    
    # 冲突检测
    print("\n4. 冲突检测:")
    conflicts = engine.detect_conflicts()
    if conflicts:
        for conflict in conflicts:
            print(f"   冲突类型: {conflict['type']}")
            print(f"   描述: {conflict['description']}")
    else:
        print("   没有检测到冲突")
    
    # 性能报告
    print("\n5. 性能报告:")
    performance = engine.get_performance_report()
    print(f"   总执行次数: {performance['engine_performance']['total_executions']}")
    print(f"   成功率: {performance['engine_performance']['success_rate']:.2%}")
    print(f"   平均执行时间: {performance['engine_performance']['average_execution_time']:.4f}秒")
    
    # 决策解释
    print("\n6. 决策解释:")
    explanation = engine.explain_decision(test_contexts[0])
    print(f"   匹配的规则数量: {len(explanation['matched_rules'])}")
    for rule_info in explanation['matched_rules']:
        print(f"     - {rule_info['rule_name']}: {rule_info['evaluation']}")
    
    # 优化建议
    print("\n7. 优化建议:")
    suggestions = engine.suggest_improvements()
    if suggestions:
        for suggestion in suggestions:
            print(f"   规则 '{suggestion['rule_name']}': {suggestion['suggestion']}")
    else:
        print("   暂无优化建议")
    
    # 规则验证
    print("\n8. 规则验证:")
    validation = engine.validate_rules()
    print(f"   验证结果: {'通过' if validation['valid'] else '失败'}")
    if validation['errors']:
        print("   错误:")
        for error in validation['errors']:
            print(f"     - {error}")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_rule_engine()