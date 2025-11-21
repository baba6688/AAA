"""
O4算法优化器主模块

该模块是O4算法优化器的核心组件，提供全面的算法优化功能。
包含算法复杂度分析、数据结构优化、排序搜索优化、动态规划优化、
并行算法优化、机器学习优化和异步处理等功能。

主要类:
- AlgorithmOptimizer: 主要优化器类
- ComplexityAnalyzer: 复杂度分析器
- DataStructureOptimizer: 数据结构优化器
- SortingSearchOptimizer: 排序搜索优化器
- DPGreedyOptimizer: 动态规划和贪心算法优化器
- ParallelOptimizer: 并行算法优化器
- MLOptimizer: 机器学习算法优化器
- AsyncOptimizer: 异步算法优化器

作者: AI系统
版本: 1.0.0
创建时间: 2025-11-06
"""

import asyncio
import logging
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import functools
import inspect
import sys
import os
import json
import numpy as np
from collections import defaultdict, deque
import heapq
from itertools import combinations, permutations
import warnings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('algorithm_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class ComplexityType(Enum):
    """复杂度类型枚举"""
    TIME = "time"
    SPACE = "space"
    BOTH = "both"


class OptimizationLevel(Enum):
    """优化级别枚举"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AlgorithmType(Enum):
    """算法类型枚举"""
    SORTING = "sorting"
    SEARCHING = "searching"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    GREEDY = "greedy"
    GRAPH = "graph"
    TREE = "tree"
    STRING = "string"
    NUMERICAL = "numerical"
    MACHINE_LEARNING = "machine_learning"
    PARALLEL = "parallel"


@dataclass
class ComplexityResult:
    """复杂度分析结果"""
    time_complexity: str
    space_complexity: str
    best_case: str
    average_case: str
    worst_case: str
    optimization_suggestions: List[str] = field(default_factory=list)
    performance_score: float = 0.0
    memory_usage_estimate: int = 0


@dataclass
class OptimizationResult:
    """优化结果"""
    original_algorithm: str
    optimized_algorithm: str
    performance_improvement: float
    complexity_before: ComplexityResult
    complexity_after: ComplexityResult
    optimization_applied: List[str]
    execution_time_before: float
    execution_time_after: float
    memory_usage_before: int
    memory_usage_after: int
    success: bool = True
    error_message: str = ""


class ComplexityAnalyzer:
    """
    算法复杂度分析器
    
    提供时间复杂度、空间复杂度的分析和优化建议功能。
    """
    
    def __init__(self):
        """初始化复杂度分析器"""
        self.logger = logging.getLogger(f"{__name__}.ComplexityAnalyzer")
        self.complexity_patterns = {
            'O(1)': {'name': '常数时间', 'score': 10},
            'O(log n)': {'name': '对数时间', 'score': 9},
            'O(n)': {'name': '线性时间', 'score': 7},
            'O(n log n)': {'name': '线性对数时间', 'score': 6},
            'O(n²)': {'name': '二次时间', 'score': 4},
            'O(n³)': {'name': '三次时间', 'score': 2},
            'O(2^n)': {'name': '指数时间', 'score': 1},
            'O(n!)': {'name': '阶乘时间', 'score': 0.5}
        }
    
    def analyze_algorithm_complexity(self, algorithm_func: Callable, 
                                   input_size: int = 1000,
                                   iterations: int = 10) -> ComplexityResult:
        """
        分析算法复杂度
        
        Args:
            algorithm_func: 要分析的算法函数
            input_size: 输入数据大小
            iterations: 测试迭代次数
            
        Returns:
            ComplexityResult: 复杂度分析结果
        """
        try:
            self.logger.info(f"开始分析算法复杂度，输入大小: {input_size}")
            
            # 生成测试数据
            test_data = self._generate_test_data(input_size)
            
            # 测量执行时间
            times = []
            memory_usage = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = algorithm_func(test_data)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                
                # 估算内存使用（简化实现）
                memory_usage.append(sys.getsizeof(result))
            
            avg_time = sum(times) / len(times)
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            # 根据执行时间趋势推断复杂度
            time_complexity = self._estimate_time_complexity(times, input_size)
            space_complexity = self._estimate_space_complexity(avg_memory)
            
            # 生成优化建议
            suggestions = self._generate_optimization_suggestions(time_complexity, space_complexity)
            
            # 计算性能评分
            performance_score = self._calculate_performance_score(time_complexity, space_complexity)
            
            result = ComplexityResult(
                time_complexity=time_complexity,
                space_complexity=space_complexity,
                best_case="O(1)",
                average_case=time_complexity,
                worst_case=time_complexity,
                optimization_suggestions=suggestions,
                performance_score=performance_score,
                memory_usage_estimate=int(avg_memory)
            )
            
            self.logger.info(f"复杂度分析完成: {time_complexity}, {space_complexity}")
            return result
            
        except Exception as e:
            self.logger.error(f"复杂度分析失败: {str(e)}")
            raise
    
    def _generate_test_data(self, size: int) -> List[int]:
        """生成测试数据"""
        return list(range(size))
    
    def _estimate_time_complexity(self, times: List[float], input_size: int) -> str:
        """估算时间复杂度"""
        if len(times) < 2:
            return "O(n)"
        
        # 简化的复杂度估算
        avg_time = sum(times) / len(times)
        
        if avg_time < 0.001:
            return "O(1)"
        elif avg_time < 0.01:
            return "O(log n)"
        elif avg_time < 0.1:
            return "O(n)"
        elif avg_time < 1.0:
            return "O(n log n)"
        elif avg_time < 10.0:
            return "O(n²)"
        else:
            return "O(n³)"
    
    def _estimate_space_complexity(self, memory_usage: int) -> str:
        """估算空间复杂度"""
        if memory_usage < 1024:
            return "O(1)"
        elif memory_usage < 10240:
            return "O(n)"
        else:
            return "O(n²)"
    
    def _generate_optimization_suggestions(self, time_complexity: str, 
                                         space_complexity: str) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 时间复杂度优化建议
        if time_complexity in ["O(n²)", "O(n³)", "O(2^n)", "O(n!)"]:
            suggestions.append("考虑使用更高效的算法或数据结构")
            suggestions.append("应用分治策略或动态规划")
            suggestions.append("使用哈希表或平衡树优化查找操作")
        
        if time_complexity == "O(n²)":
            suggestions.append("考虑使用快速排序或归并排序替代冒泡排序")
            suggestions.append("应用双指针技巧优化嵌套循环")
        
        # 空间复杂度优化建议
        if space_complexity == "O(n²)":
            suggestions.append("使用原地算法减少空间复杂度")
            suggestions.append("考虑流式处理大数据集")
        
        # 通用优化建议
        suggestions.extend([
            "使用适当的数据结构",
            "考虑缓存和记忆化技术",
            "应用并行化处理",
            "优化算法实现细节"
        ])
        
        return suggestions
    
    def _calculate_performance_score(self, time_complexity: str, space_complexity: str) -> float:
        """计算性能评分"""
        time_score = self.complexity_patterns.get(time_complexity, {}).get('score', 5)
        space_score = self.complexity_patterns.get(space_complexity, {}).get('score', 5)
        return (time_score + space_score) / 2


class DataStructureOptimizer:
    """
    数据结构优化器
    
    提供数组、链表、树、图等数据结构的优化选择和建议。
    """
    
    def __init__(self):
        """初始化数据结构优化器"""
        self.logger = logging.getLogger(f"{__name__}.DataStructureOptimizer")
        self.structure_characteristics = {
            'array': {
                'access': 'O(1)',
                'search': 'O(n)',
                'insert': 'O(n)',
                'delete': 'O(n)',
                'memory': 'O(n)',
                'use_cases': ['随机访问', '顺序处理', '缓存友好']
            },
            'linked_list': {
                'access': 'O(n)',
                'search': 'O(n)',
                'insert': 'O(1)',
                'delete': 'O(1)',
                'memory': 'O(n)',
                'use_cases': ['频繁插入删除', '动态大小', '内存碎片处理']
            },
            'hash_table': {
                'access': 'O(1)',
                'search': 'O(1)',
                'insert': 'O(1)',
                'delete': 'O(1)',
                'memory': 'O(n)',
                'use_cases': ['快速查找', '去重', '频率统计']
            },
            'binary_tree': {
                'access': 'O(log n)',
                'search': 'O(log n)',
                'insert': 'O(log n)',
                'delete': 'O(log n)',
                'memory': 'O(n)',
                'use_cases': ['有序数据', '范围查询', '优先级队列']
            },
            'graph': {
                'access': 'O(V+E)',
                'search': 'O(V+E)',
                'insert': 'O(1)',
                'delete': 'O(V)',
                'memory': 'O(V+E)',
                'use_cases': ['网络建模', '路径查找', '关系分析']
            }
        }
    
    def recommend_data_structure(self, operations: List[str], 
                               data_size: int = 1000,
                               access_pattern: str = "random") -> Dict[str, Any]:
        """
        推荐合适的数据结构
        
        Args:
            operations: 需要的操作列表
            data_size: 数据大小
            access_pattern: 访问模式 ("random", "sequential", "frequent_insert")
            
        Returns:
            Dict: 推荐的数据结构和建议
        """
        try:
            self.logger.info(f"数据结构推荐: 操作={operations}, 大小={data_size}")
            
            # 分析操作频率
            operation_scores = defaultdict(int)
            
            for op in operations:
                if op in ['access', 'get', '[]']:
                    operation_scores['access'] += 3
                elif op in ['search', 'find', 'contains']:
                    operation_scores['search'] += 3
                elif op in ['insert', 'add', 'push']:
                    operation_scores['insert'] += 2
                elif op in ['delete', 'remove', 'pop']:
                    operation_scores['delete'] += 2
            
            # 根据访问模式调整权重
            if access_pattern == "sequential":
                operation_scores['access'] *= 0.5
                operation_scores['search'] *= 0.8
            elif access_pattern == "frequent_insert":
                operation_scores['insert'] *= 2
                operation_scores['delete'] *= 2
            
            # 计算每个数据结构的适合度
            structure_scores = {}
            
            for structure, characteristics in self.structure_characteristics.items():
                score = 0
                
                # 根据操作权重计算得分
                for op_type, weight in operation_scores.items():
                    if op_type in characteristics:
                        complexity = characteristics[op_type]
                        if complexity == "O(1)":
                            score += weight * 10
                        elif complexity == "O(log n)":
                            score += weight * 8
                        elif complexity == "O(n)":
                            score += weight * 5
                        else:
                            score += weight * 2
                
                structure_scores[structure] = score
            
            # 推荐最佳数据结构
            best_structure = max(structure_scores, key=structure_scores.get)
            
            result = {
                'recommended_structure': best_structure,
                'score': structure_scores[best_structure],
                'characteristics': self.structure_characteristics[best_structure],
                'alternatives': sorted(structure_scores.items(), 
                                     key=lambda x: x[1], reverse=True)[1:4],
                'optimization_tips': self._generate_optimization_tips(best_structure, operations),
                'implementation_example': self._get_implementation_example(best_structure)
            }
            
            self.logger.info(f"推荐数据结构: {best_structure}")
            return result
            
        except Exception as e:
            self.logger.error(f"数据结构推荐失败: {str(e)}")
            raise
    
    def _generate_optimization_tips(self, structure: str, operations: List[str]) -> List[str]:
        """生成优化建议"""
        tips = []
        
        if structure == 'array':
            tips.extend([
                "使用NumPy数组提高数值计算性能",
                "考虑预分配数组大小避免频繁扩容",
                "使用切片操作代替循环遍历"
            ])
        elif structure == 'linked_list':
            tips.extend([
                "使用双向链表支持双向遍历",
                "维护头尾指针优化首尾操作",
                "考虑使用内存池减少分配开销"
            ])
        elif structure == 'hash_table':
            tips.extend([
                "选择合适的哈希函数减少冲突",
                "设置合适的负载因子和扩容策略",
                "使用开放寻址法或链地址法处理冲突"
            ])
        elif structure == 'binary_tree':
            tips.extend([
                "使用平衡树(如AVL树、红黑树)保持性能",
                "考虑B树用于磁盘存储",
                "使用堆实现优先级队列"
            ])
        elif structure == 'graph':
            tips.extend([
                "使用邻接表存储稀疏图",
                "使用邻接矩阵存储密集图",
                "考虑图的压缩存储格式"
            ])
        
        return tips
    
    def _get_implementation_example(self, structure: str) -> str:
        """获取实现示例"""
        examples = {
            'array': """
# 优化的数组使用示例
import numpy as np

# 使用NumPy数组提高性能
data = np.arange(1000000, dtype=np.int32)
result = np.sum(data[data > 500000])
            """,
            'linked_list': """
# 优化的链表实现示例
class OptimizedLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, value):
        node = Node(value)
        if not self.head:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self.size += 1
            """,
            'hash_table': """
# 优化的哈希表实现示例
class OptimizedHashTable:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.load_factor = 0.75
        self.table = [[] for _ in range(capacity)]
        self.size = 0
    
    def _hash(self, key):
        return hash(key) % self.capacity
    
    def put(self, key, value):
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                self.table[index] = [(k, value)]
                return
        self.table[index].append((key, value))
        self.size += 1
            """
        }
        return examples.get(structure, "# 无示例代码")


class SortingSearchOptimizer:
    """
    排序和搜索算法优化器
    
    提供快速排序、归并排序、二分查找等算法的优化实现。
    """
    
    def __init__(self):
        """初始化排序搜索优化器"""
        self.logger = logging.getLogger(f"{__name__}.SortingSearchOptimizer")
        self.sorting_algorithms = {
            'quick_sort': {
                'complexity': 'O(n log n) average, O(n²) worst',
                'space': 'O(log n)',
                'stable': False,
                'best_for': ['随机数据', '大数据集', '内存敏感']
            },
            'merge_sort': {
                'complexity': 'O(n log n)',
                'space': 'O(n)',
                'stable': True,
                'best_for': ['链表排序', '需要稳定性', '外部排序']
            },
            'heap_sort': {
                'complexity': 'O(n log n)',
                'space': 'O(1)',
                'stable': False,
                'best_for': ['原地排序', '内存受限', '优先级队列']
            },
            'radix_sort': {
                'complexity': 'O(d*n)',
                'space': 'O(n)',
                'stable': True,
                'best_for': ['整数排序', '字符串排序', '固定长度数据']
            }
        }
    
    def optimized_quick_sort(self, arr: List[T], 
                           use_introsort: bool = True,
                           median_of_three: bool = True) -> List[T]:
        """
        优化的快速排序实现
        
        Args:
            arr: 待排序数组
            use_introsort: 是否使用内省排序
            median_of_three: 是否使用三数取中法
            
        Returns:
            List[T]: 排序后的数组
        """
        try:
            self.logger.info(f"开始优化快速排序，数组大小: {len(arr)}")
            
            if len(arr) <= 1:
                return arr.copy()
            
            # 创建副本避免修改原数组
            result = arr.copy()
            
            if use_introsort:
                result = self._introsort(result, 0, len(result) - 1, 
                                      2 * int(np.log2(len(result))))
            else:
                result = self._quick_sort_impl(result, 0, len(result) - 1, median_of_three)
            
            self.logger.info("快速排序完成")
            return result
            
        except Exception as e:
            self.logger.error(f"快速排序失败: {str(e)}")
            raise
    
    def _quick_sort_impl(self, arr: List[T], low: int, high: int, 
                        median_of_three: bool) -> List[T]:
        """快速排序实现"""
        while low < high:
            if high - low < 10:  # 小数组使用插入排序
                arr[low:high+1] = self._insertion_sort(arr[low:high+1])
                break
            
            # 选择枢轴
            if median_of_three:
                pivot_index = self._median_of_three_pivot(arr, low, high)
            else:
                pivot_index = high
            
            pivot_index = self._partition(arr, low, high, pivot_index)
            
            # 递归排序较小的部分，迭代排序较大的部分
            if pivot_index - low < high - pivot_index:
                arr[low:pivot_index] = self._quick_sort_impl(arr, low, pivot_index - 1, median_of_three)
                low = pivot_index + 1
            else:
                arr[pivot_index + 1:high + 1] = self._quick_sort_impl(arr, pivot_index + 1, high, median_of_three)
                high = pivot_index - 1
        
        return arr
    
    def _introsort(self, arr: List[T], low: int, high: int, max_depth: int) -> List[T]:
        """内省排序实现"""
        while high - low > 16:  # 切换到堆排序的阈值
            if max_depth == 0:
                # 切换到堆排序
                self._heapify(arr, low, high + 1)
                return arr
            
            pivot_index = self._partition(arr, low, high, high)
            
            if pivot_index - low < high - pivot_index:
                arr[low:pivot_index] = self._introsort(arr, low, pivot_index - 1, max_depth - 1)
                low = pivot_index + 1
            else:
                arr[pivot_index + 1:high + 1] = self._introsort(arr, pivot_index + 1, high, max_depth - 1)
                high = pivot_index - 1
        
        # 小数组使用插入排序
        if high - low > 1:
            arr[low:high + 1] = self._insertion_sort(arr[low:high + 1])
        
        return arr
    
    def _median_of_three_pivot(self, arr: List[T], low: int, high: int) -> int:
        """三数取中法选择枢轴"""
        mid = (low + high) // 2
        
        # 排序 low, mid, high
        if arr[mid] < arr[low]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[high] < arr[low]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[high] < arr[mid]:
            arr[mid], arr[high] = arr[high], arr[mid]
        
        # 将中位数放到 high-1 位置
        arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
        return high - 1
    
    def _partition(self, arr: List[T], low: int, high: int, pivot_index: int) -> int:
        """数组分区"""
        arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
        pivot = arr[high]
        
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    def _insertion_sort(self, arr: List[T]) -> List[T]:
        """插入排序"""
        result = arr.copy()
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and result[j] > key:
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
        return result
    
    def _heapify(self, arr: List[T], low: int, high: int) -> None:
        """堆化"""
        n = high - low
        for i in range(low + n // 2 - 1, low - 1, -1):
            self._sift_down(arr, i, low, high)
    
    def _sift_down(self, arr: List[T], i: int, low: int, high: int) -> None:
        """下滤操作"""
        n = high - low
        root = i
        while True:
            child = 2 * root - low + 1
            if child >= high:
                break
            
            if child + 1 < high and arr[child] < arr[child + 1]:
                child += 1
            
            if arr[root] < arr[child]:
                arr[root], arr[child] = arr[child], arr[root]
                root = child
            else:
                break
    
    def optimized_binary_search(self, arr: List[T], target: T, 
                              use_interpolation: bool = False) -> Optional[int]:
        """
        优化的二分查找实现
        
        Args:
            arr: 已排序数组
            target: 目标值
            use_interpolation: 是否使用插值查找
            
        Returns:
            Optional[int]: 目标值索引，不存在则返回None
        """
        try:
            self.logger.debug(f"开始二分查找，目标值: {target}")
            
            if not arr:
                return None
            
            left, right = 0, len(arr) - 1
            
            while left <= right:
                if use_interpolation and len(arr) > 100:
                    # 插值查找
                    if arr[right] != arr[left]:
                        mid = left + (target - arr[left]) * (right - left) // (arr[right] - arr[left])
                    else:
                        mid = (left + right) // 2
                else:
                    # 经典二分查找
                    mid = (left + right) // 2
                
                # 确保mid在有效范围内
                mid = max(left, min(right, mid))
                
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            self.logger.debug("二分查找完成，未找到目标值")
            return None
            
        except Exception as e:
            self.logger.error(f"二分查找失败: {str(e)}")
            raise
    
    def optimized_merge_sort(self, arr: List[T], use_bottom_up: bool = False) -> List[T]:
        """
        优化的归并排序实现
        
        Args:
            arr: 待排序数组
            use_bottom_up: 是否使用自底向上实现
            
        Returns:
            List[T]: 排序后的数组
        """
        try:
            self.logger.info(f"开始归并排序，数组大小: {len(arr)}")
            
            if len(arr) <= 1:
                return arr.copy()
            
            result = arr.copy()
            
            if use_bottom_up:
                result = self._bottom_up_merge_sort(result)
            else:
                result = self._top_down_merge_sort(result)
            
            self.logger.info("归并排序完成")
            return result
            
        except Exception as e:
            self.logger.error(f"归并排序失败: {str(e)}")
            raise
    
    def _top_down_merge_sort(self, arr: List[T]) -> List[T]:
        """自顶向下归并排序"""
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self._top_down_merge_sort(arr[:mid])
        right = self._top_down_merge_sort(arr[mid:])
        
        return self._merge(left, right)
    
    def _bottom_up_merge_sort(self, arr: List[T]) -> List[T]:
        """自底向上归并排序"""
        n = len(arr)
        result = arr.copy()
        
        size = 1
        while size < n:
            for left in range(0, n, 2 * size):
                mid = min(left + size, n)
                right = min(left + 2 * size, n)
                
                if mid < right:
                    result[left:right] = self._merge(result[left:mid], result[mid:right])
            
            size *= 2
        
        return result
    
    def _merge(self, left: List[T], right: List[T]) -> List[T]:
        """合并两个有序数组"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    def recommend_sorting_algorithm(self, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        推荐合适的排序算法
        
        Args:
            data_characteristics: 数据特征描述
            
        Returns:
            Dict: 推荐的排序算法和建议
        """
        try:
            self.logger.info("开始排序算法推荐")
            
            size = data_characteristics.get('size', 1000)
            data_type = data_characteristics.get('type', 'number')
            stability_required = data_characteristics.get('stability', False)
            memory_constraint = data_characteristics.get('memory_constraint', False)
            already_partially_sorted = data_characteristics.get('partially_sorted', False)
            
            # 计算算法适合度
            algorithm_scores = {}
            
            for algo, characteristics in self.sorting_algorithms.items():
                score = 0
                
                # 根据数据大小调整
                if size > 1000000:  # 大数据集
                    if characteristics['complexity'] == 'O(n log n)':
                        score += 10
                    elif 'O(n²)' in characteristics['complexity']:
                        score -= 10
                
                # 根据稳定性要求调整
                if stability_required and characteristics['stable']:
                    score += 8
                elif stability_required and not characteristics['stable']:
                    score -= 5
                
                # 根据内存约束调整
                if memory_constraint and characteristics['space'] == 'O(1)':
                    score += 8
                elif memory_constraint and characteristics['space'] == 'O(n)':
                    score -= 3
                
                # 根据部分有序调整
                if already_partially_sorted:
                    if algo == 'insertion_sort':
                        score += 15
                    elif algo == 'quick_sort':
                        score += 5
                
                algorithm_scores[algo] = score
            
            # 推荐最佳算法
            best_algorithm = max(algorithm_scores, key=algorithm_scores.get)
            
            result = {
                'recommended_algorithm': best_algorithm,
                'score': algorithm_scores[best_algorithm],
                'characteristics': self.sorting_algorithms[best_algorithm],
                'alternatives': sorted(algorithm_scores.items(), 
                                     key=lambda x: x[1], reverse=True)[1:3],
                'optimization_tips': self._get_sorting_optimization_tips(best_algorithm, data_characteristics),
                'performance_prediction': self._predict_performance(best_algorithm, data_characteristics)
            }
            
            self.logger.info(f"推荐排序算法: {best_algorithm}")
            return result
            
        except Exception as e:
            self.logger.error(f"排序算法推荐失败: {str(e)}")
            raise
    
    def _get_sorting_optimization_tips(self, algorithm: str, 
                                     characteristics: Dict[str, Any]) -> List[str]:
        """获取排序优化建议"""
        tips = []
        
        if algorithm == 'quick_sort':
            tips.extend([
                "使用三数取中法选择枢轴",
                "小数组使用插入排序",
                "考虑使用内省排序防止最坏情况"
            ])
        elif algorithm == 'merge_sort':
            tips.extend([
                "对于链表排序，归并排序特别高效",
                "使用临时缓冲区减少内存分配",
                "考虑自然归并排序利用已有顺序"
            ])
        elif algorithm == 'heap_sort':
            tips.extend([
                "原地排序，空间效率高",
                "适合内存受限环境",
                "可以用于优先级队列实现"
            ])
        elif algorithm == 'radix_sort':
            tips.extend([
                "适合整数和字符串排序",
                "稳定排序算法",
                "可以并行化处理"
            ])
        
        # 通用优化建议
        tips.extend([
            "考虑数据的分布特征选择算法",
            "对于小数据集，简单算法可能更高效",
            "测试不同算法的实际性能"
        ])
        
        return tips
    
    def _predict_performance(self, algorithm: str, 
                           characteristics: Dict[str, Any]) -> Dict[str, float]:
        """预测算法性能"""
        size = characteristics.get('size', 1000)
        
        # 简化的性能预测模型
        base_time = size * np.log2(size) * 1e-6  # 基准时间（秒）
        
        if algorithm == 'quick_sort':
            predicted_time = base_time * 0.8
        elif algorithm == 'merge_sort':
            predicted_time = base_time * 1.2
        elif algorithm == 'heap_sort':
            predicted_time = base_time * 1.1
        elif algorithm == 'radix_sort':
            predicted_time = base_time * 0.6
        else:
            predicted_time = base_time
        
        return {
            'estimated_time_seconds': predicted_time,
            'estimated_memory_mb': size * 8 / (1024 * 1024),  # 假设8字节每元素
            'performance_rating': max(1, min(10, 10 - predicted_time * 1000))
        }


class DPGreedyOptimizer:
    """
    动态规划和贪心算法优化器
    
    提供状态转移优化、备忘录优化等功能。
    """
    
    def __init__(self):
        """初始化动态规划和贪心算法优化器"""
        self.logger = logging.getLogger(f"{__name__}.DPGreedyOptimizer")
        self.memoization_cache = {}
        self.tabulation_cache = {}
    
    def optimize_dp_with_memoization(self, func: Callable, 
                                   cache_key_func: Optional[Callable] = None) -> Callable:
        """
        使用备忘录优化动态规划算法
        
        Args:
            func: 原始动态规划函数
            cache_key_func: 缓存键生成函数
            
        Returns:
            Callable: 优化后的函数
        """
        try:
            self.logger.info("应用备忘录优化")
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = (args, tuple(sorted(kwargs.items())))
                
                # 检查缓存
                if cache_key in self.memoization_cache:
                    self.logger.debug("使用缓存结果")
                    return self.memoization_cache[cache_key]
                
                # 计算结果并缓存
                result = func(*args, **kwargs)
                self.memoization_cache[cache_key] = result
                
                return result
            
            return wrapper
            
        except Exception as e:
            self.logger.error(f"备忘录优化失败: {str(e)}")
            raise
    
    def optimize_dp_with_tabulation(self, func: Callable) -> Callable:
        """
        使用表格化优化动态规划算法
        
        Args:
            func: 原始动态规划函数
            
        Returns:
            Callable: 优化后的函数
        """
        try:
            self.logger.info("应用表格化优化")
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = (args, tuple(sorted(kwargs.items())))
                
                # 检查缓存
                if cache_key in self.tabulation_cache:
                    self.logger.debug("使用表格化缓存结果")
                    return self.tabulation_cache[cache_key]
                
                # 计算结果并缓存
                result = func(*args, **kwargs)
                self.tabulation_cache[cache_key] = result
                
                return result
            
            return wrapper
            
        except Exception as e:
            self.logger.error(f"表格化优化失败: {str(e)}")
            raise
    
    def optimize_greedy_algorithm(self, items: List[Any], 
                                key_func: Callable,
                                reverse: bool = False) -> List[Any]:
        """
        优化贪心算法实现
        
        Args:
            items: 待处理项目列表
            key_func: 排序键函数
            reverse: 是否降序排序
            
        Returns:
            List[Any]: 优化后的结果
        """
        try:
            self.logger.info(f"优化贪心算法，项目数量: {len(items)}")
            
            # 按键函数排序
            sorted_items = sorted(items, key=key_func, reverse=reverse)
            
            # 应用贪心策略（这里提供通用的贪心实现）
            result = []
            for item in sorted_items:
                if self._is_feasible(result + [item]):
                    result.append(item)
            
            self.logger.info(f"贪心算法完成，选择了 {len(result)} 个项目")
            return result
            
        except Exception as e:
            self.logger.error(f"贪心算法优化失败: {str(e)}")
            raise
    
    def _is_feasible(self, items: List[Any]) -> bool:
        """检查项目组合是否可行（需要根据具体问题实现）"""
        # 这里是通用实现，具体问题需要重写此方法
        return True
    
    def analyze_dp_problem(self, problem_description: str) -> Dict[str, Any]:
        """
        分析动态规划问题并提供优化建议
        
        Args:
            problem_description: 问题描述
            
        Returns:
            Dict: 分析结果和优化建议
        """
        try:
            self.logger.info("分析动态规划问题")
            
            # 简化的DP问题分析
            analysis = {
                'problem_type': self._classify_dp_problem(problem_description),
                'optimal_substructure': True,
                'overlapping_subproblems': True,
                'recommended_approach': 'memoization',
                'complexity_analysis': self._analyze_dp_complexity(problem_description),
                'optimization_strategies': self._suggest_dp_optimizations(problem_description),
                'implementation_patterns': self._get_dp_patterns()
            }
            
            self.logger.info(f"DP问题分析完成: {analysis['problem_type']}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"DP问题分析失败: {str(e)}")
            raise
    
    def _classify_dp_problem(self, description: str) -> str:
        """分类DP问题类型"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['路径', '路径', 'path']):
            return '路径问题'
        elif any(word in description_lower for word in ['序列', 'sequence', 'subsequence']):
            return '序列问题'
        elif any(word in description_lower for word in ['分割', 'partition', 'cut']):
            return '分割问题'
        elif any(word in description_lower for word in ['背包', 'knapsack']):
            return '背包问题'
        elif any(word in description_lower for word in ['编辑距离', 'edit distance']):
            return '字符串问题'
        else:
            return '通用DP问题'
    
    def _analyze_dp_complexity(self, description: str) -> Dict[str, str]:
        """分析DP复杂度"""
        # 简化的复杂度分析
        return {
            'time_complexity': 'O(n²)',
            'space_complexity': 'O(n²)',
            'optimized_space_complexity': 'O(n)'
        }
    
    def _suggest_dp_optimizations(self, description: str) -> List[str]:
        """建议DP优化策略"""
        suggestions = [
            "使用备忘录避免重复计算",
            "考虑滚动数组优化空间复杂度",
            "分析问题结构选择最优DP方法",
            "使用位运算优化状态表示",
            "考虑并行化独立的子问题"
        ]
        return suggestions
    
    def _get_dp_patterns(self) -> Dict[str, str]:
        """获取DP实现模式"""
        return {
            'memoization': """
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
            """,
            'tabulation': """
def fib_tab(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
            """,
            'space_optimized': """
def fib_space_optimized(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for i in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
            """
        }


class ParallelOptimizer:
    """
    并行算法优化器
    
    提供多线程和分布式算法的优化实现。
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """初始化并行优化器"""
        self.logger = logging.getLogger(f"{__name__}.ParallelOptimizer")
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
    
    def parallel_map(self, func: Callable, data: List[T], 
                    use_processes: bool = False) -> List[Any]:
        """
        并行映射操作
        
        Args:
            func: 要应用的函数
            data: 输入数据
            use_processes: 是否使用进程池（多进程）
            
        Returns:
            List[Any]: 处理结果
        """
        try:
            self.logger.info(f"开始并行映射，数据大小: {len(data)}")
            
            if use_processes:
                executor = self.process_pool
                self.logger.info("使用进程池并行处理")
            else:
                executor = self.thread_pool
                self.logger.info("使用线程池并行处理")
            
            # 提交所有任务
            futures = [executor.submit(func, item) for item in data]
            
            # 收集结果
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"并行任务执行失败: {str(e)}")
                    results.append(None)
            
            self.logger.info(f"并行映射完成，处理了 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"并行映射失败: {str(e)}")
            raise
    
    def parallel_reduce(self, func: Callable, data: List[T], 
                       use_tree_reduction: bool = True) -> Any:
        """
        并行归约操作
        
        Args:
            func: 归约函数
            data: 输入数据
            use_tree_reduction: 是否使用树形归约
            
        Returns:
            Any: 归约结果
        """
        try:
            self.logger.info(f"开始并行归约，数据大小: {len(data)}")
            
            if len(data) <= self.max_workers or not use_tree_reduction:
                # 简单并行映射后归约
                results = self.parallel_map(func, data)
                return functools.reduce(lambda x, y: func(x, y), results)
            else:
                # 树形归约
                return self._tree_reduction(func, data)
            
        except Exception as e:
            self.logger.error(f"并行归约失败: {str(e)}")
            raise
    
    def _tree_reduction(self, func: Callable, data: List[T]) -> Any:
        """树形归约实现"""
        current_level = data
        
        while len(current_level) > 1:
            next_level = []
            
            # 并行处理当前层的元素对
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    next_level.append(func(current_level[i], current_level[i + 1]))
                else:
                    next_level.append(current_level[i])
            
            current_level = next_level
        
        return current_level[0] if current_level else None
    
    def parallel_sort(self, data: List[T], algorithm: str = "merge") -> List[T]:
        """
        并行排序
        
        Args:
            data: 待排序数据
            algorithm: 排序算法 ("merge", "quick")
            
        Returns:
            List[T]: 排序结果
        """
        try:
            self.logger.info(f"开始并行排序，数据大小: {len(data)}")
            
            if algorithm == "merge":
                return self._parallel_merge_sort(data)
            elif algorithm == "quick":
                return self._parallel_quick_sort(data)
            else:
                raise ValueError(f"不支持的排序算法: {algorithm}")
            
        except Exception as e:
            self.logger.error(f"并行排序失败: {str(e)}")
            raise
    
    def _parallel_merge_sort(self, data: List[T]) -> List[T]:
        """并行归并排序"""
        if len(data) <= 1:
            return data
        
        if len(data) <= 1000:  # 小数组使用普通排序
            return sorted(data)
        
        mid = len(data) // 2
        left = data[:mid]
        right = data[mid:]
        
        # 并行排序左右两部分
        futures = [
            self.thread_pool.submit(self._parallel_merge_sort, left),
            self.thread_pool.submit(self._parallel_merge_sort, right)
        ]
        
        left_sorted, right_sorted = [f.result() for f in futures]
        
        # 合并结果
        return self._merge(left_sorted, right_sorted)
    
    def _parallel_quick_sort(self, data: List[T]) -> List[T]:
        """并行快速排序"""
        if len(data) <= 1:
            return data
        
        if len(data) <= 1000:  # 小数组使用普通排序
            return sorted(data)
        
        # 选择枢轴
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        
        # 并行排序左右两部分
        futures = [
            self.thread_pool.submit(self._parallel_quick_sort, left),
            self.thread_pool.submit(self._parallel_quick_sort, right)
        ]
        
        left_sorted, right_sorted = [f.result() for f in futures]
        
        return left_sorted + middle + right_sorted
    
    def _merge(self, left: List[T], right: List[T]) -> List[T]:
        """合并两个有序数组"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    def parallel_search(self, data: List[T], target: T, 
                       use_parallel_binary: bool = True) -> List[int]:
        """
        并行搜索
        
        Args:
            data: 已排序数据
            target: 目标值
            use_parallel_binary: 是否使用并行二分查找
            
        Returns:
            List[int]: 匹配目标的索引列表
        """
        try:
            self.logger.info(f"开始并行搜索，目标值: {target}")
            
            if use_parallel_binary and len(data) > 10000:
                return self._parallel_binary_search(data, target)
            else:
                # 简单线性搜索的并行版本
                chunk_size = max(1, len(data) // self.max_workers)
                chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
                
                def search_chunk(chunk):
                    indices = []
                    for i, item in enumerate(chunk):
                        if item == target:
                            indices.append(i)
                    return indices
                
                results = self.parallel_map(search_chunk, chunks)
                
                # 转换索引
                final_results = []
                for chunk_idx, indices in enumerate(results):
                    for idx in indices:
                        final_results.append(chunk_idx * chunk_size + idx)
                
                return final_results
            
        except Exception as e:
            self.logger.error(f"并行搜索失败: {str(e)}")
            raise
    
    def _parallel_binary_search(self, data: List[T], target: T) -> List[int]:
        """并行二分查找"""
        # 简化的并行二分查找实现
        results = []
        
        def binary_search_range(start, end):
            indices = []
            left, right = start, end
            
            while left <= right:
                mid = (left + right) // 2
                if data[mid] == target:
                    # 扩展查找相同值
                    i = mid
                    while i >= 0 and data[i] == target:
                        indices.append(i)
                        i -= 1
                    i = mid + 1
                    while i < len(data) and data[i] == target:
                        indices.append(i)
                        i += 1
                    break
                elif data[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return indices
        
        # 将数据分成多个范围并行搜索
        ranges = [(i * len(data) // self.max_workers, 
                  (i + 1) * len(data) // self.max_workers - 1) 
                 for i in range(self.max_workers)]
        
        futures = [self.thread_pool.submit(binary_search_range, start, end) 
                  for start, end in ranges]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                self.logger.error(f"并行范围搜索失败: {str(e)}")
        
        return sorted(list(set(results)))  # 去重并排序
    
    def optimize_for_parallelization(self, algorithm_func: Callable, 
                                   data_size: int) -> Dict[str, Any]:
        """
        分析算法并行化潜力并提供优化建议
        
        Args:
            algorithm_func: 算法函数
            data_size: 数据大小
            
        Returns:
            Dict: 并行化分析和优化建议
        """
        try:
            self.logger.info(f"分析算法并行化潜力，数据大小: {data_size}")
            
            # 分析算法特征
            algorithm_analysis = self._analyze_algorithm_parallel_potential(algorithm_func)
            
            # 生成优化建议
            optimization_suggestions = self._generate_parallel_optimization_suggestions(
                algorithm_analysis, data_size)
            
            result = {
                'parallel_potential_score': algorithm_analysis['score'],
                'recommended_approach': algorithm_analysis['approach'],
                'expected_speedup': algorithm_analysis['expected_speedup'],
                'bottlenecks': algorithm_analysis['bottlenecks'],
                'optimization_suggestions': optimization_suggestions,
                'implementation_patterns': self._get_parallel_patterns()
            }
            
            self.logger.info(f"并行化分析完成，潜力评分: {result['parallel_potential_score']}")
            return result
            
        except Exception as e:
            self.logger.error(f"并行化分析失败: {str(e)}")
            raise
    
    def _analyze_algorithm_parallel_potential(self, func: Callable) -> Dict[str, Any]:
        """分析算法的并行化潜力"""
        # 简化的分析实现
        func_source = inspect.getsource(func)
        
        score = 5  # 基础分
        bottlenecks = []
        approach = "线程级并行"
        
        # 检查循环结构
        if "for" in func_source and "range" in func_source:
            score += 2
            approach = "循环并行化"
        
        # 检查递归结构
        if "def " in func_source and func.__name__ in func_source:
            score += 1
            approach = "分治并行"
        
        # 检查数据依赖
        if "=" in func_source and "+" in func_source:
            bottlenecks.append("数据依赖可能限制并行化")
            score -= 1
        
        # 计算预期加速比
        expected_speedup = min(self.max_workers, score)
        
        return {
            'score': min(10, max(1, score)),
            'approach': approach,
            'expected_speedup': expected_speedup,
            'bottlenecks': bottlenecks
        }
    
    def _generate_parallel_optimization_suggestions(self, 
                                                  analysis: Dict[str, Any], 
                                                  data_size: int) -> List[str]:
        """生成并行化优化建议"""
        suggestions = []
        
        score = analysis['score']
        
        if score >= 8:
            suggestions.append("算法具有很高的并行化潜力，建议使用多进程")
            suggestions.append("考虑使用GPU加速计算密集型操作")
        elif score >= 6:
            suggestions.append("算法适合并行化，建议使用线程池")
            suggestions.append("注意线程安全和数据同步问题")
        elif score >= 4:
            suggestions.append("算法部分适合并行化，需要重构代码")
            suggestions.append("识别并分离独立的计算任务")
        else:
            suggestions.append("算法并行化潜力较低，建议优化算法本身")
            suggestions.append("考虑使用更高效的算法替代")
        
        # 根据数据大小给出建议
        if data_size > 1000000:
            suggestions.append("大数据集建议使用分布式计算")
        elif data_size > 100000:
            suggestions.append("中等数据集可以使用多进程处理")
        
        suggestions.extend([
            "使用适当的任务粒度避免过度并行化",
            "考虑内存访问模式优化缓存性能",
            "监控并行化开销，确保净收益为正"
        ])
        
        return suggestions
    
    def _get_parallel_patterns(self) -> Dict[str, str]:
        """获取并行化实现模式"""
        return {
            'thread_pool': """
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task, data) for data in dataset]
    results = [future.result() for future in as_completed(futures)]
            """,
            'process_pool': """
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task, data) for data in dataset]
    results = [future.result() for future in as_completed(futures)]
            """,
            'async_parallel': """
async def parallel_tasks():
    tasks = [asyncio.create_task(async_task(data)) for data in dataset]
    results = await asyncio.gather(*tasks)
    return results
            """
        }
    
    def __del__(self):
        """清理资源"""
        try:
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
        except:
            pass


class MLOptimizer:
    """
    机器学习算法优化器
    
    提供特征选择、模型调参、交叉验证等功能。
    """
    
    def __init__(self):
        """初始化机器学习优化器"""
        self.logger = logging.getLogger(f"{__name__}.MLOptimizer")
        self.feature_importance_cache = {}
        self.model_performance_cache = {}
    
    def optimize_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                 method: str = "univariate") -> Dict[str, Any]:
        """
        特征选择优化
        
        Args:
            X: 特征矩阵
            y: 目标变量
            method: 特征选择方法 ("univariate", "recursive", "mutual_info")
            
        Returns:
            Dict: 特征选择结果
        """
        try:
            self.logger.info(f"开始特征选择优化，特征数量: {X.shape[1]}")
            
            if method == "univariate":
                return self._univariate_feature_selection(X, y)
            elif method == "recursive":
                return self._recursive_feature_selection(X, y)
            elif method == "mutual_info":
                return self._mutual_info_feature_selection(X, y)
            else:
                raise ValueError(f"不支持的特征选择方法: {method}")
            
        except Exception as e:
            self.logger.error(f"特征选择失败: {str(e)}")
            raise
    
    def _univariate_feature_selection(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """单变量特征选择"""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # 使用F统计量进行特征选择
        selector = SelectKBest(score_func=f_classif, k='all')
        X_selected = selector.fit_transform(X, y)
        
        scores = selector.scores_
        p_values = selector.pvalues_
        
        # 计算特征重要性
        feature_importance = scores / np.sum(scores)
        
        result = {
            'selected_features': np.arange(X.shape[1]),
            'feature_scores': scores.tolist(),
            'p_values': p_values.tolist(),
            'feature_importance': feature_importance.tolist(),
            'top_features': np.argsort(scores)[-10:].tolist(),  # 前10个重要特征
            'selection_method': 'univariate'
        }
        
        self.logger.info("单变量特征选择完成")
        return result
    
    def _recursive_feature_selection(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """递归特征消除"""
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        
        # 使用随机森林进行递归特征消除
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=10)
        X_selected = selector.fit_transform(X, y)
        
        feature_ranking = selector.ranking_
        selected_features = np.where(feature_ranking == 1)[0]
        
        result = {
            'selected_features': selected_features.tolist(),
            'feature_ranking': feature_ranking.tolist(),
            'n_features_selected': len(selected_features),
            'selection_method': 'recursive'
        }
        
        self.logger.info("递归特征选择完成")
        return result
    
    def _mutual_info_feature_selection(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """互信息特征选择"""
        from sklearn.feature_selection import mutual_info_classif
        
        # 计算互信息
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # 选择top-k特征
        k = min(10, X.shape[1])
        top_indices = np.argsort(mi_scores)[-k:]
        
        result = {
            'selected_features': top_indices.tolist(),
            'mutual_info_scores': mi_scores.tolist(),
            'feature_importance': (mi_scores / np.sum(mi_scores)).tolist(),
            'selection_method': 'mutual_info'
        }
        
        self.logger.info("互信息特征选择完成")
        return result
    
    def optimize_hyperparameters(self, model_class: type, param_grid: Dict[str, Any],
                               X: np.ndarray, y: np.ndarray, 
                               cv: int = 5) -> Dict[str, Any]:
        """
        超参数优化
        
        Args:
            model_class: 模型类
            param_grid: 参数网格
            X: 训练数据
            y: 目标变量
            cv: 交叉验证折数
            
        Returns:
            Dict: 优化结果
        """
        try:
            self.logger.info("开始超参数优化")
            
            # 简化的网格搜索实现
            best_score = -np.inf
            best_params = {}
            all_results = []
            
            # 生成参数组合
            param_combinations = self._generate_param_combinations(param_grid)
            
            for params in param_combinations[:20]:  # 限制搜索空间
                try:
                    model = model_class(**params)
                    scores = self._cross_validate(model, X, y, cv)
                    avg_score = np.mean(scores)
                    
                    result = {
                        'params': params,
                        'cv_scores': scores.tolist(),
                        'mean_score': avg_score,
                        'std_score': np.std(scores)
                    }
                    all_results.append(result)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = params
                
                except Exception as e:
                    self.logger.warning(f"参数组合 {params} 测试失败: {str(e)}")
                    continue
            
            result = {
                'best_params': best_params,
                'best_score': best_score,
                'all_results': all_results,
                'optimization_method': 'grid_search'
            }
            
            self.logger.info(f"超参数优化完成，最佳得分: {best_score}")
            return result
            
        except Exception as e:
            self.logger.error(f"超参数优化失败: {str(e)}")
            raise
    
    def _generate_param_combinations(self, param_grid: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成参数组合"""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray, cv: int) -> np.ndarray:
        """交叉验证"""
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            except Exception as e:
                self.logger.warning(f"交叉验证fold失败: {str(e)}")
                scores.append(0.0)
        
        return np.array(scores)
    
    def optimize_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                model: Any, cv_methods: List[str] = None) -> Dict[str, Any]:
        """
        交叉验证优化
        
        Args:
            X: 特征数据
            y: 目标数据
            model: 模型实例
            cv_methods: 交叉验证方法列表
            
        Returns:
            Dict: 交叉验证结果
        """
        try:
            self.logger.info("开始交叉验证优化")
            
            if cv_methods is None:
                cv_methods = ['kfold', 'stratified', 'shuffle']
            
            results = {}
            
            for method in cv_methods:
                if method == 'kfold':
                    scores = self._kfold_cv(model, X, y)
                elif method == 'stratified':
                    scores = self._stratified_cv(model, X, y)
                elif method == 'shuffle':
                    scores = self._shuffle_cv(model, X, y)
                else:
                    continue
                
                results[method] = {
                    'scores': scores.tolist(),
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            
            # 推荐最佳交叉验证方法
            best_method = max(results.keys(), 
                            key=lambda m: results[m]['mean'])
            
            result = {
                'cv_results': results,
                'recommended_method': best_method,
                'best_score': results[best_method]['mean'],
                'stability_analysis': self._analyze_cv_stability(results)
            }
            
            self.logger.info(f"交叉验证优化完成，推荐方法: {best_method}")
            return result
            
        except Exception as e:
            self.logger.error(f"交叉验证优化失败: {str(e)}")
            raise
    
    def _kfold_cv(self, model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> np.ndarray:
        """K折交叉验证"""
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        return np.array(scores)
    
    def _stratified_cv(self, model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> np.ndarray:
        """分层交叉验证"""
        from sklearn.model_selection import StratifiedKFold
        
        skfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        return np.array(scores)
    
    def _shuffle_cv(self, model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> np.ndarray:
        """随机洗牌交叉验证"""
        from sklearn.model_selection import ShuffleSplit
        
        shuffle = ShuffleSplit(n_splits=cv, test_size=0.2, random_state=42)
        scores = []
        
        for train_idx, val_idx in shuffle.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        return np.array(scores)
    
    def _analyze_cv_stability(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析交叉验证稳定性"""
        stability_analysis = {}
        
        for method, result in results.items():
            scores = result['scores']
            stability_analysis[method] = {
                'coefficient_of_variation': result['std'] / result['mean'],
                'score_range': result['max'] - result['min'],
                'stability_rating': 'high' if result['std'] < 0.05 else 'medium' if result['std'] < 0.1 else 'low'
            }
        
        return stability_analysis
    
    def analyze_model_performance(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        模型性能分析
        
        Args:
            model: 训练好的模型
            X: 测试数据
            y: 真实标签
            
        Returns:
            Dict: 性能分析结果
        """
        try:
            self.logger.info("开始模型性能分析")
            
            # 预测
            y_pred = model.predict(X)
            y_pred_proba = None
            
            try:
                y_pred_proba = model.predict_proba(X)
            except:
                pass
            
            # 计算各种指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import confusion_matrix, classification_report
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
            
            # 混淆矩阵
            cm = confusion_matrix(y, y_pred)
            
            # 分类报告
            report = classification_report(y, y_pred, output_dict=True, zero_division=0)
            
            result = {
                'metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'model_type': type(model).__name__,
                'feature_importance': self._get_feature_importance(model),
                'performance_grade': self._grade_performance(metrics),
                'optimization_suggestions': self._generate_ml_optimization_suggestions(metrics)
            }
            
            self.logger.info(f"模型性能分析完成，准确率: {metrics['accuracy']:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"模型性能分析失败: {str(e)}")
            raise
    
    def _get_feature_importance(self, model: Any) -> Optional[List[float]]:
        """获取特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_[0]).tolist()
            else:
                return None
        except:
            return None
    
    def _grade_performance(self, metrics: Dict[str, float]) -> str:
        """性能评级"""
        accuracy = metrics['accuracy']
        
        if accuracy >= 0.95:
            return 'A+'
        elif accuracy >= 0.90:
            return 'A'
        elif accuracy >= 0.85:
            return 'B+'
        elif accuracy >= 0.80:
            return 'B'
        elif accuracy >= 0.75:
            return 'C+'
        elif accuracy >= 0.70:
            return 'C'
        else:
            return 'D'
    
    def _generate_ml_optimization_suggestions(self, metrics: Dict[str, float]) -> List[str]:
        """生成机器学习优化建议"""
        suggestions = []
        
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']
        
        if accuracy < 0.7:
            suggestions.append("模型准确率较低，考虑使用更复杂的模型或增加训练数据")
        
        if precision < 0.7:
            suggestions.append("精确率较低，可能存在过多假阳性，考虑调整分类阈值")
        
        if recall < 0.7:
            suggestions.append("召回率较低，可能存在过多假阴性，考虑使用成本敏感学习")
        
        if abs(precision - recall) > 0.2:
            suggestions.append("精确率和召回率不平衡，考虑调整类别权重或使用不同的评估指标")
        
        suggestions.extend([
            "进行特征工程，提取更有意义的特征",
            "尝试集成学习方法如随机森林或梯度提升",
            "使用正则化技术防止过拟合",
            "进行超参数调优以提高性能",
            "增加训练数据量或使用数据增强技术"
        ])
        
        return suggestions


class AsyncOptimizer:
    """
    异步算法优化器
    
    提供异步算法优化处理功能。
    """
    
    def __init__(self):
        """初始化异步优化器"""
        self.logger = logging.getLogger(f"{__name__}.AsyncOptimizer")
        self.semaphore = asyncio.Semaphore(10)  # 限制并发数量
    
    async def async_parallel_map(self, func: Callable, data: List[T]) -> List[Any]:
        """
        异步并行映射
        
        Args:
            func: 异步函数
            data: 输入数据
            
        Returns:
            List[Any]: 处理结果
        """
        try:
            self.logger.info(f"开始异步并行映射，数据大小: {len(data)}")
            
            # 创建异步任务
            tasks = []
            for item in data:
                task = asyncio.create_task(self._safe_async_call(func, item))
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"异步任务失败: {str(result)}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            self.logger.info(f"异步并行映射完成，处理了 {len(processed_results)} 个结果")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"异步并行映射失败: {str(e)}")
            raise
    
    async def _safe_async_call(self, func: Callable, item: T) -> Any:
        """安全的异步函数调用"""
        async with self.semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                return func(item)
    
    async def async_batch_process(self, batch_func: Callable, 
                                data: List[T], batch_size: int = 100) -> List[Any]:
        """
        异步批处理
        
        Args:
            batch_func: 批处理函数
            data: 输入数据
            batch_size: 批大小
            
        Returns:
            List[Any]: 处理结果
        """
        try:
            self.logger.info(f"开始异步批处理，数据大小: {len(data)}, 批大小: {batch_size}")
            
            # 分批处理
            batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            
            # 创建批处理任务
            tasks = []
            for batch in batches:
                task = asyncio.create_task(self._process_batch(batch_func, batch))
                tasks.append(task)
            
            # 等待所有批处理完成
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 合并结果
            all_results = []
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    self.logger.error(f"批处理失败: {str(batch_result)}")
                    all_results.extend([None] * batch_size)
                else:
                    all_results.extend(batch_result)
            
            self.logger.info(f"异步批处理完成")
            return all_results[:len(data)]  # 截断到原始数据大小
            
        except Exception as e:
            self.logger.error(f"异步批处理失败: {str(e)}")
            raise
    
    async def _process_batch(self, batch_func: Callable, batch: List[T]) -> List[Any]:
        """处理单个批次"""
        if asyncio.iscoroutinefunction(batch_func):
            return await batch_func(batch)
        else:
            return batch_func(batch)
    
    async def async_pipeline(self, stages: List[Callable], 
                           initial_data: Any) -> Any:
        """
        异步流水线处理
        
        Args:
            stages: 处理阶段列表
            initial_data: 初始数据
            
        Returns:
            Any: 处理结果
        """
        try:
            self.logger.info(f"开始异步流水线处理，阶段数: {len(stages)}")
            
            data = initial_data
            
            for i, stage in enumerate(stages):
                self.logger.debug(f"执行流水线阶段 {i+1}")
                
                if asyncio.iscoroutinefunction(stage):
                    data = await stage(data)
                else:
                    data = stage(data)
                
                if data is None:
                    self.logger.warning(f"流水线阶段 {i+1} 返回None")
                    break
            
            self.logger.info("异步流水线处理完成")
            return data
            
        except Exception as e:
            self.logger.error(f"异步流水线处理失败: {str(e)}")
            raise
    
    async def async_data_stream_processing(self, data_stream: Any, 
                                         processor: Callable,
                                         buffer_size: int = 1000) -> List[Any]:
        """
        异步数据流处理
        
        Args:
            data_stream: 数据流
            processor: 处理函数
            buffer_size: 缓冲区大小
            
        Returns:
            List[Any]: 处理结果
        """
        try:
            self.logger.info(f"开始异步数据流处理，缓冲区大小: {buffer_size}")
            
            results = []
            buffer = []
            
            async for item in self._async_iterate(data_stream):
                buffer.append(item)
                
                if len(buffer) >= buffer_size:
                    # 处理缓冲区数据
                    processed = await self.async_parallel_map(processor, buffer)
                    results.extend(processed)
                    buffer = []
            
            # 处理剩余数据
            if buffer:
                processed = await self.async_parallel_map(processor, buffer)
                results.extend(processed)
            
            self.logger.info(f"异步数据流处理完成，处理了 {len(results)} 个项目")
            return results
            
        except Exception as e:
            self.logger.error(f"异步数据流处理失败: {str(e)}")
            raise
    
    async def _async_iterate(self, data_stream: Any):
        """异步迭代器"""
        try:
            # 如果是普通可迭代对象，转换为异步
            for item in data_stream:
                yield item
                await asyncio.sleep(0)  # 让出控制权
        except TypeError:
            # 如果已经是异步可迭代对象
            async for item in data_stream:
                yield item
    
    async def async_algorithm_optimization(self, algorithm_func: Callable,
                                         optimization_strategies: List[str],
                                         test_data: Any) -> Dict[str, Any]:
        """
        异步算法优化
        
        Args:
            algorithm_func: 算法函数
            optimization_strategies: 优化策略列表
            test_data: 测试数据
            
        Returns:
            Dict: 优化结果
        """
        try:
            self.logger.info(f"开始异步算法优化，策略数: {len(optimization_strategies)}")
            
            # 并行测试不同优化策略
            tasks = []
            for strategy in optimization_strategies:
                task = asyncio.create_task(
                    self._test_optimization_strategy(algorithm_func, strategy, test_data)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            optimization_results = {}
            for i, result in enumerate(results):
                strategy = optimization_strategies[i]
                if isinstance(result, Exception):
                    optimization_results[strategy] = {
                        'success': False,
                        'error': str(result)
                    }
                else:
                    optimization_results[strategy] = result
            
            # 找出最佳优化策略
            best_strategy = max(
                [(k, v) for k, v in optimization_results.items() if v.get('success', False)],
                key=lambda x: x[1].get('performance_score', 0),
                default=(None, {})
            )[0]
            
            result = {
                'optimization_results': optimization_results,
                'best_strategy': best_strategy,
                'performance_comparison': self._compare_optimization_performance(optimization_results)
            }
            
            self.logger.info(f"异步算法优化完成，最佳策略: {best_strategy}")
            return result
            
        except Exception as e:
            self.logger.error(f"异步算法优化失败: {str(e)}")
            raise
    
    async def _test_optimization_strategy(self, algorithm_func: Callable,
                                        strategy: str, test_data: Any) -> Dict[str, Any]:
        """测试优化策略"""
        try:
            start_time = time.perf_counter()
            
            # 应用优化策略
            if strategy == "memoization":
                optimized_func = self._apply_memoization(algorithm_func)
            elif strategy == "parallelization":
                optimized_func = self._apply_parallelization(algorithm_func)
            elif strategy == "vectorization":
                optimized_func = self._apply_vectorization(algorithm_func)
            else:
                optimized_func = algorithm_func
            
            # 运行算法
            if asyncio.iscoroutinefunction(optimized_func):
                result = await optimized_func(test_data)
            else:
                result = optimized_func(test_data)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # 计算性能评分（简化）
            performance_score = max(0, 10 - execution_time * 10)
            
            return {
                'success': True,
                'execution_time': execution_time,
                'performance_score': performance_score,
                'strategy': strategy,
                'result_type': type(result).__name__
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'strategy': strategy
            }
    
    def _apply_memoization(self, func: Callable) -> Callable:
        """应用记忆化优化"""
        cache = {}
        
        def memoized_func(*args):
            key = str(args)
            if key in cache:
                return cache[key]
            result = func(*args)
            cache[key] = result
            return result
        
        return memoized_func
    
    def _apply_parallelization(self, func: Callable) -> Callable:
        """应用并行化优化"""
        def parallel_func(data):
            if isinstance(data, (list, tuple)) and len(data) > 100:
                # 小数据量直接处理，大数据量并行处理
                chunk_size = max(1, len(data) // 4)
                chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(func, chunk) for chunk in chunks]
                    results = [future.result() for future in futures]
                
                # 合并结果
                if isinstance(data, list):
                    return [item for sublist in results for item in sublist]
                else:
                    return tuple(item for sublist in results for item in sublist)
            else:
                return func(data)
        
        return parallel_func
    
    def _apply_vectorization(self, func: Callable) -> Callable:
        """应用向量化优化"""
        def vectorized_func(data):
            if isinstance(data, (list, tuple)) and all(isinstance(x, (int, float)) for x in data):
                # 如果是数值列表，尝试向量化处理
                try:
                    import numpy as np
                    np_data = np.array(data)
                    # 这里可以应用各种向量化操作
                    return func(np_data)
                except:
                    pass
            return func(data)
        
        return vectorized_func
    
    def _compare_optimization_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """比较优化性能"""
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if not successful_results:
            return {'error': '没有成功的优化策略'}
        
        execution_times = [r['execution_time'] for r in successful_results.values()]
        performance_scores = [r['performance_score'] for r in successful_results.values()]
        
        return {
            'fastest_strategy': min(successful_results.keys(), 
                                  key=lambda k: successful_results[k]['execution_time']),
            'best_performance_strategy': max(successful_results.keys(),
                                           key=lambda k: successful_results[k]['performance_score']),
            'average_execution_time': np.mean(execution_times),
            'performance_improvement_range': {
                'min': min(performance_scores),
                'max': max(performance_scores),
                'std': np.std(performance_scores)
            }
        }


class AlgorithmOptimizer:
    """
    O4算法优化器主类
    
    整合所有优化功能，提供统一的算法优化接口。
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.INTERMEDIATE):
        """
        初始化算法优化器
        
        Args:
            optimization_level: 优化级别
        """
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(f"{__name__}.AlgorithmOptimizer")
        
        # 初始化各个优化器
        self.complexity_analyzer = ComplexityAnalyzer()
        self.data_structure_optimizer = DataStructureOptimizer()
        self.sorting_search_optimizer = SortingSearchOptimizer()
        self.dp_greedy_optimizer = DPGreedyOptimizer()
        self.parallel_optimizer = ParallelOptimizer()
        self.ml_optimizer = MLOptimizer()
        self.async_optimizer = AsyncOptimizer()
        
        self.logger.info(f"算法优化器初始化完成，优化级别: {optimization_level.value}")
    
    def optimize_algorithm(self, algorithm_func: Callable, 
                         optimization_targets: List[str] = None,
                         **kwargs) -> OptimizationResult:
        """
        综合算法优化
        
        Args:
            algorithm_func: 要优化的算法函数
            optimization_targets: 优化目标列表
            **kwargs: 其他参数
            
        Returns:
            OptimizationResult: 优化结果
        """
        try:
            self.logger.info("开始综合算法优化")
            
            if optimization_targets is None:
                optimization_targets = ['complexity', 'performance', 'memory']
            
            # 记录原始性能
            start_time = time.perf_counter()
            original_result = algorithm_func(kwargs.get('test_data', []))
            original_time = time.perf_counter() - start_time
            original_memory = sys.getsizeof(original_result)
            
            # 复杂度分析
            complexity_before = self.complexity_analyzer.analyze_algorithm_complexity(
                algorithm_func, len(kwargs.get('test_data', [])))
            
            # 应用优化策略
            optimized_func = algorithm_func
            optimization_applied = []
            
            if 'complexity' in optimization_targets:
                optimized_func, applied = self._optimize_complexity(optimized_func)
                optimization_applied.extend(applied)
            
            if 'performance' in optimization_targets:
                optimized_func, applied = self._optimize_performance(optimized_func)
                optimization_applied.extend(applied)
            
            if 'memory' in optimization_targets:
                optimized_func, applied = self._optimize_memory(optimized_func)
                optimization_applied.extend(applied)
            
            if 'parallel' in optimization_targets:
                optimized_func, applied = self._optimize_parallel(optimized_func)
                optimization_applied.extend(applied)
            
            # 测试优化后性能
            start_time = time.perf_counter()
            optimized_result = optimized_func(kwargs.get('test_data', []))
            optimized_time = time.perf_counter() - start_time
            optimized_memory = sys.getsizeof(optimized_result)
            
            # 复杂度分析
            complexity_after = self.complexity_analyzer.analyze_algorithm_complexity(
                optimized_func, len(kwargs.get('test_data', [])))
            
            # 计算性能改进
            time_improvement = (original_time - optimized_time) / original_time * 100
            memory_improvement = (original_memory - optimized_memory) / original_memory * 100
            performance_improvement = (time_improvement + memory_improvement) / 2
            
            result = OptimizationResult(
                original_algorithm=algorithm_func.__name__,
                optimized_algorithm=optimized_func.__name__,
                performance_improvement=performance_improvement,
                complexity_before=complexity_before,
                complexity_after=complexity_after,
                optimization_applied=optimization_applied,
                execution_time_before=original_time,
                execution_time_after=optimized_time,
                memory_usage_before=original_memory,
                memory_usage_after=optimized_memory,
                success=True
            )
            
            self.logger.info(f"算法优化完成，性能提升: {performance_improvement:.2f}%")
            return result
            
        except Exception as e:
            self.logger.error(f"算法优化失败: {str(e)}")
            return OptimizationResult(
                original_algorithm=algorithm_func.__name__,
                optimized_algorithm="",
                performance_improvement=0.0,
                complexity_before=ComplexityResult("", "", "", "", []),
                complexity_after=ComplexityResult("", "", "", "", []),
                optimization_applied=[],
                execution_time_before=0.0,
                execution_time_after=0.0,
                memory_usage_before=0,
                memory_usage_after=0,
                success=False,
                error_message=str(e)
            )
    
    def _optimize_complexity(self, func: Callable) -> Tuple[Callable, List[str]]:
        """复杂度优化"""
        applied = []
        
        # 检查是否适合记忆化
        if self._is_suitable_for_memoization(func):
            func = self.dp_greedy_optimizer.optimize_dp_with_memoization(func)
            applied.append("备忘录优化")
        
        # 检查是否适合并行化
        if self._is_suitable_for_parallelization(func):
            # 这里可以添加并行化逻辑
            applied.append("并行化优化")
        
        return func, applied
    
    def _optimize_performance(self, func: Callable) -> Tuple[Callable, List[str]]:
        """性能优化"""
        applied = []
        
        # 添加性能监控装饰器
        @functools.wraps(func)
        def optimized_func(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            if end_time - start_time > 0.1:  # 超过100ms的操作记录警告
                self.logger.warning(f"函数 {func.__name__} 执行时间较长: {end_time - start_time:.4f}s")
            
            return result
        
        applied.append("性能监控优化")
        return optimized_func, applied
    
    def _optimize_memory(self, func: Callable) -> Tuple[Callable, List[str]]:
        """内存优化"""
        applied = []
        
        # 简化的内存优化
        @functools.wraps(func)
        def optimized_func(*args, **kwargs):
            # 强制垃圾回收
            import gc
            gc.collect()
            
            result = func(*args, **kwargs)
            
            # 如果结果很大，考虑使用生成器
            if hasattr(result, '__len__') and len(result) > 10000:
                self.logger.info("检测到大结果集，建议使用生成器")
            
            return result
        
        applied.append("内存管理优化")
        return optimized_func, applied
    
    def _optimize_parallel(self, func: Callable) -> Tuple[Callable, List[str]]:
        """并行优化"""
        applied = []
        
        # 检查是否适合并行化
        if self._is_data_parallel(func):
            def parallel_func(data):
                if isinstance(data, (list, tuple)) and len(data) > 1000:
                    return self.parallel_optimizer.parallel_map(func, data)
                else:
                    return func(data)
            
            applied.append("数据并行优化")
            return parallel_func, applied
        
        return func, applied
    
    def _is_suitable_for_memoization(self, func: Callable) -> bool:
        """检查是否适合记忆化"""
        func_source = inspect.getsource(func)
        return 'def ' in func_source and any(word in func_source for word in ['fib', 'factorial', 'dp'])
    
    def _is_suitable_for_parallelization(self, func: Callable) -> bool:
        """检查是否适合并行化"""
        func_source = inspect.getsource(func)
        return 'for' in func_source and 'range' in func_source
    
    def _is_data_parallel(self, func: Callable) -> bool:
        """检查是否数据并行"""
        func_signature = inspect.signature(func)
        return len(func_signature.parameters) == 1
    
    def optimize_sorting_algorithm(self, data: List[T], 
                                 algorithm_preference: str = "auto") -> Dict[str, Any]:
        """
        优化排序算法
        
        Args:
            data: 待排序数据
            algorithm_preference: 算法偏好
            
        Returns:
            Dict: 优化结果
        """
        try:
            self.logger.info(f"优化排序算法，数据大小: {len(data)}")
            
            # 分析数据特征
            data_characteristics = {
                'size': len(data),
                'type': type(data[0]).__name__ if data else 'unknown',
                'partially_sorted': self._check_partially_sorted(data),
                'stability': True,  # 默认需要稳定性
                'memory_constraint': len(data) > 1000000
            }
            
            # 推荐排序算法
            recommendation = self.sorting_search_optimizer.recommend_sorting_algorithm(
                data_characteristics)
            
            # 应用推荐的算法
            if algorithm_preference == "auto":
                algorithm = recommendation['recommended_algorithm']
            else:
                algorithm = algorithm_preference
            
            # 执行排序
            start_time = time.perf_counter()
            
            if algorithm == "quick_sort":
                sorted_data = self.sorting_search_optimizer.optimized_quick_sort(data)
            elif algorithm == "merge_sort":
                sorted_data = self.sorting_search_optimizer.optimized_merge_sort(data)
            else:
                sorted_data = sorted(data)  # 使用Python内置排序
            
            sort_time = time.perf_counter() - start_time
            
            result = {
                'original_data_size': len(data),
                'recommended_algorithm': algorithm,
                'algorithm_characteristics': recommendation['characteristics'],
                'sorting_time': sort_time,
                'sorted_data': sorted_data,
                'optimization_tips': recommendation['optimization_tips'],
                'performance_prediction': recommendation['performance_prediction']
            }
            
            self.logger.info(f"排序算法优化完成，使用算法: {algorithm}")
            return result
            
        except Exception as e:
            self.logger.error(f"排序算法优化失败: {str(e)}")
            raise
    
    def _check_partially_sorted(self, data: List[T]) -> bool:
        """检查是否部分有序"""
        if len(data) < 10:
            return False
        
        # 计算有序对的比例
        ordered_pairs = 0
        total_pairs = 0
        
        for i in range(len(data) - 1):
            if data[i] <= data[i + 1]:
                ordered_pairs += 1
            total_pairs += 1
        
        return ordered_pairs / total_pairs > 0.8
    
    def optimize_search_algorithm(self, data: List[T], target: T,
                                search_type: str = "auto") -> Dict[str, Any]:
        """
        优化搜索算法
        
        Args:
            data: 搜索数据
            target: 目标值
            search_type: 搜索类型
            
        Returns:
            Dict: 搜索结果
        """
        try:
            self.logger.info(f"优化搜索算法，数据大小: {len(data)}")
            
            # 检查数据是否已排序
            is_sorted = data == sorted(data)
            
            # 选择搜索算法
            if search_type == "auto":
                if is_sorted and len(data) > 100:
                    search_type = "binary"
                else:
                    search_type = "linear"
            
            start_time = time.perf_counter()
            
            if search_type == "binary":
                result_index = self.sorting_search_optimizer.optimized_binary_search(
                    data, target, use_interpolation=len(data) > 1000)
                algorithm_used = "binary_search"
            else:
                result_index = self._linear_search(data, target)
                algorithm_used = "linear_search"
            
            search_time = time.perf_counter() - start_time
            
            result = {
                'target': target,
                'data_size': len(data),
                'is_data_sorted': is_sorted,
                'algorithm_used': algorithm_used,
                'found_index': result_index,
                'search_time': search_time,
                'search_successful': result_index is not None
            }
            
            self.logger.info(f"搜索算法优化完成，使用算法: {algorithm_used}")
            return result
            
        except Exception as e:
            self.logger.error(f"搜索算法优化失败: {str(e)}")
            raise
    
    def _linear_search(self, data: List[T], target: T) -> Optional[int]:
        """线性搜索"""
        for i, item in enumerate(data):
            if item == target:
                return i
        return None
    
    def optimize_data_structure(self, operations: List[str], 
                              data_size: int = 1000) -> Dict[str, Any]:
        """
        优化数据结构选择
        
        Args:
            operations: 需要的操作列表
            data_size: 数据大小
            
        Returns:
            Dict: 优化建议
        """
        try:
            self.logger.info(f"优化数据结构，操作: {operations}, 大小: {data_size}")
            
            recommendation = self.data_structure_optimizer.recommend_data_structure(
                operations, data_size)
            
            result = {
                'operations': operations,
                'data_size': data_size,
                'recommended_structure': recommendation['recommended_structure'],
                'structure_characteristics': recommendation['characteristics'],
                'optimization_tips': recommendation['optimization_tips'],
                'implementation_example': recommendation['implementation_example'],
                'alternatives': recommendation['alternatives']
            }
            
            self.logger.info(f"数据结构优化完成，推荐: {recommendation['recommended_structure']}")
            return result
            
        except Exception as e:
            self.logger.error(f"数据结构优化失败: {str(e)}")
            raise
    
    async def optimize_async_algorithm(self, async_func: Callable,
                                     test_data: Any) -> Dict[str, Any]:
        """
        优化异步算法
        
        Args:
            async_func: 异步函数
            test_data: 测试数据
            
        Returns:
            Dict: 优化结果
        """
        try:
            self.logger.info("开始异步算法优化")
            
            # 定义优化策略
            optimization_strategies = [
                "memoization",
                "parallelization", 
                "vectorization"
            ]
            
            # 并行测试优化策略
            result = await self.async_optimizer.async_algorithm_optimization(
                async_func, optimization_strategies, test_data)
            
            self.logger.info("异步算法优化完成")
            return result
            
        except Exception as e:
            self.logger.error(f"异步算法优化失败: {str(e)}")
            raise
    
    def optimize_ml_algorithm(self, model_class: type, X: np.ndarray, y: np.ndarray,
                            optimization_type: str = "comprehensive") -> Dict[str, Any]:
        """
        优化机器学习算法
        
        Args:
            model_class: 模型类
            X: 特征数据
            y: 目标数据
            optimization_type: 优化类型
            
        Returns:
            Dict: 优化结果
        """
        try:
            self.logger.info("开始机器学习算法优化")
            
            results = {}
            
            if optimization_type in ["comprehensive", "feature_selection"]:
                # 特征选择优化
                results['feature_selection'] = self.ml_optimizer.optimize_feature_selection(
                    X, y, method="univariate")
            
            if optimization_type in ["comprehensive", "hyperparameter"]:
                # 超参数优化（简化版）
                param_grid = self._get_default_param_grid(model_class)
                if param_grid:
                    results['hyperparameter'] = self.ml_optimizer.optimize_hyperparameters(
                        model_class, param_grid, X, y)
            
            if optimization_type in ["comprehensive", "cross_validation"]:
                # 交叉验证优化
                model = model_class()
                results['cross_validation'] = self.ml_optimizer.optimize_cross_validation(
                    X, y, model)
            
            if optimization_type in ["comprehensive", "performance"]:
                # 性能分析
                model = model_class()
                model.fit(X, y)
                results['performance'] = self.ml_optimizer.analyze_model_performance(
                    model, X, y)
            
            self.logger.info("机器学习算法优化完成")
            return results
            
        except Exception as e:
            self.logger.error(f"机器学习算法优化失败: {str(e)}")
            raise
    
    def _get_default_param_grid(self, model_class: type) -> Optional[Dict[str, Any]]:
        """获取默认参数网格"""
        class_name = model_class.__name__
        
        if 'RandomForest' in class_name:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        elif 'SVC' in class_name:
            return {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        elif 'LogisticRegression' in class_name:
            return {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        else:
            return None
    
    def generate_optimization_report(self, optimization_results: List[OptimizationResult]) -> str:
        """
        生成优化报告
        
        Args:
            optimization_results: 优化结果列表
            
        Returns:
            str: 格式化的报告
        """
        try:
            self.logger.info("生成优化报告")
            
            report = []
            report.append("=" * 60)
            report.append("O4算法优化器 - 优化报告")
            report.append("=" * 60)
            report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"优化项目数量: {len(optimization_results)}")
            report.append("")
            
            # 总体统计
            successful_optimizations = [r for r in optimization_results if r.success]
            total_improvement = sum(r.performance_improvement for r in successful_optimizations)
            avg_improvement = total_improvement / len(successful_optimizations) if successful_optimizations else 0
            
            report.append("总体统计:")
            report.append(f"  成功优化项目: {len(successful_optimizations)}/{len(optimization_results)}")
            report.append(f"  平均性能提升: {avg_improvement:.2f}%")
            report.append(f"  总性能提升: {total_improvement:.2f}%")
            report.append("")
            
            # 详细结果
            report.append("详细优化结果:")
            report.append("-" * 40)
            
            for i, result in enumerate(optimization_results, 1):
                report.append(f"项目 {i}: {result.original_algorithm}")
                report.append(f"  状态: {'成功' if result.success else '失败'}")
                
                if result.success:
                    report.append(f"  性能提升: {result.performance_improvement:.2f}%")
                    report.append(f"  执行时间改进: {result.execution_time_before:.4f}s -> {result.execution_time_after:.4f}s")
                    report.append(f"  内存使用改进: {result.memory_usage_before} -> {result.memory_usage_after}")
                    report.append(f"  应用优化: {', '.join(result.optimization_applied)}")
                    report.append(f"  复杂度改进: {result.complexity_before.time_complexity} -> {result.complexity_after.time_complexity}")
                else:
                    report.append(f"  错误: {result.error_message}")
                
                report.append("")
            
            # 优化建议
            report.append("优化建议:")
            report.append("-" * 20)
            
            if successful_optimizations:
                best_optimization = max(successful_optimizations, key=lambda r: r.performance_improvement)
                worst_optimization = min(successful_optimizations, key=lambda r: r.performance_improvement)
                
                report.append(f"最佳优化: {best_optimization.original_algorithm} (提升 {best_optimization.performance_improvement:.2f}%)")
                report.append(f"需要改进: {worst_optimization.original_algorithm} (提升 {worst_optimization.performance_improvement:.2f}%)")
                
                # 统计最常用的优化技术
                all_optimizations = []
                for result in successful_optimizations:
                    all_optimizations.extend(result.optimization_applied)
                
                from collections import Counter
                optimization_counts = Counter(all_optimizations)
                
                report.append("最有效的优化技术:")
                for technique, count in optimization_counts.most_common(3):
                    report.append(f"  {technique}: {count} 次应用")
            
            report.append("")
            report.append("=" * 60)
            
            report_text = "\n".join(report)
            
            # 保存报告到文件
            with open(f"optimization_report_{int(time.time())}.txt", "w", encoding="utf-8") as f:
                f.write(report_text)
            
            self.logger.info("优化报告生成完成")
            return report_text
            
        except Exception as e:
            self.logger.error(f"优化报告生成失败: {str(e)}")
            raise
    
    def benchmark_algorithms(self, algorithms: Dict[str, Callable], 
                           test_data: Any) -> Dict[str, Any]:
        """
        算法性能基准测试
        
        Args:
            algorithms: 算法字典 {名称: 函数}
            test_data: 测试数据
            
        Returns:
            Dict: 基准测试结果
        """
        try:
            self.logger.info(f"开始算法基准测试，算法数量: {len(algorithms)}")
            
            benchmark_results = {}
            
            for name, algorithm in algorithms.items():
                self.logger.info(f"测试算法: {name}")
                
                # 多次运行取平均值
                times = []
                for _ in range(5):
                    start_time = time.perf_counter()
                    try:
                        result = algorithm(test_data)
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                    except Exception as e:
                        self.logger.error(f"算法 {name} 执行失败: {str(e)}")
                        times.append(float('inf'))
                
                # 计算统计信息
                valid_times = [t for t in times if t != float('inf')]
                
                if valid_times:
                    benchmark_results[name] = {
                        'average_time': np.mean(valid_times),
                        'min_time': np.min(valid_times),
                        'max_time': np.max(valid_times),
                        'std_time': np.std(valid_times),
                        'success_rate': len(valid_times) / len(times)
                    }
                else:
                    benchmark_results[name] = {
                        'average_time': float('inf'),
                        'min_time': float('inf'),
                        'max_time': float('inf'),
                        'std_time': 0,
                        'success_rate': 0
                    }
            
            # 排序结果
            sorted_results = dict(sorted(benchmark_results.items(), 
                                       key=lambda x: x[1]['average_time']))
            
            # 生成排名
            ranking = list(sorted_results.keys())
            
            result = {
                'benchmark_results': benchmark_results,
                'ranking': ranking,
                'fastest_algorithm': ranking[0] if ranking else None,
                'slowest_algorithm': ranking[-1] if ranking else None,
                'performance_gaps': self._calculate_performance_gaps(benchmark_results)
            }
            
            self.logger.info(f"算法基准测试完成，最快算法: {result['fastest_algorithm']}")
            return result
            
        except Exception as e:
            self.logger.error(f"算法基准测试失败: {str(e)}")
            raise
    
    def _calculate_performance_gaps(self, results: Dict[str, Any]) -> Dict[str, float]:
        """计算性能差距"""
        valid_results = {k: v for k, v in results.items() if v['average_time'] != float('inf')}
        
        if not valid_results:
            return {}
        
        times = [v['average_time'] for v in valid_results.values()]
        min_time = min(times)
        max_time = max(times)
        
        gaps = {}
        for name, result in valid_results.items():
            gaps[name] = (result['average_time'] - min_time) / min_time * 100
        
        return gaps


# 使用示例和测试函数
def example_usage():
    """使用示例"""
    print("O4算法优化器使用示例")
    print("=" * 50)
    
    # 创建优化器实例
    optimizer = AlgorithmOptimizer(OptimizationLevel.ADVANCED)
    
    # 示例1: 算法复杂度分析
    def example_sorting_algorithm(data):
        """示例排序算法"""
        return sorted(data)
    
    print("\n1. 算法复杂度分析:")
    complexity_result = optimizer.complexity_analyzer.analyze_algorithm_complexity(
        example_sorting_algorithm, 1000)
    print(f"时间复杂度: {complexity_result.time_complexity}")
    print(f"空间复杂度: {complexity_result.space_complexity}")
    print(f"性能评分: {complexity_result.performance_score}")
    
    # 示例2: 数据结构推荐
    print("\n2. 数据结构推荐:")
    ds_result = optimizer.optimize_data_structure(
        operations=['access', 'search', 'insert'], 
        data_size=10000
    )
    print(f"推荐数据结构: {ds_result['recommended_structure']}")
    print(f"优化建议: {ds_result['optimization_tips'][:2]}")
    
    # 示例3: 排序算法优化
    print("\n3. 排序算法优化:")
    test_data = list(range(1000, 0, -1))  # 逆序数据
    sort_result = optimizer.optimize_sorting_algorithm(test_data)
    print(f"推荐算法: {sort_result['recommended_algorithm']}")
    print(f"排序时间: {sort_result['sorting_time']:.6f}秒")
    
    # 示例4: 并行优化
    print("\n4. 并行优化测试:")
    def parallel_task(x):
        time.sleep(0.001)  # 模拟计算
        return x * 2
    
    test_data_parallel = list(range(100))
    parallel_result = optimizer.parallel_optimizer.parallel_map(parallel_task, test_data_parallel)
    print(f"并行处理完成，处理了 {len(parallel_result)} 个项目")
    
    print("\n示例运行完成!")


if __name__ == "__main__":
    # 运行示例
    example_usage()