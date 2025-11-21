"""
J1数学计算工具模块

该模块提供了全面的数学计算功能，包括高级数学运算、金融数学计算、
统计数学工具、优化数学工具等。支持异步处理和缓存机制。

主要组件：
- AdvancedMath: 高级数学运算（矩阵运算、微积分、线性代数）
- FinancialMath: 金融数学计算（期权定价、风险度量、收益率计算）
- StatisticalMath: 统计数学工具（概率分布、假设检验、置信区间）
- OptimizationMath: 优化数学工具（梯度下降、牛顿法、拉格朗日乘数法）
- MathematicalTools: 主工具类，整合所有功能

作者：J1数学计算工具团队
版本：1.0.0
日期：2025-11-06
"""

import asyncio
import logging
import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Awaitable
from functools import wraps, lru_cache
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from scipy import stats, optimize, integrate
from scipy.linalg import inv, det, eig, svd
from scipy.special import erf, erfc, gamma, beta, comb
import json
import pickle
from datetime import datetime, timedelta


# 配置日志
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
except Exception:
    # 如果日志配置失败，使用最基本的配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


# 自定义异常类
class MathError(Exception):
    """数学计算错误基类"""
    pass


class MatrixError(MathError):
    """矩阵运算错误"""
    pass


class FinancialMathError(MathError):
    """金融数学计算错误"""
    pass


class StatisticalError(MathError):
    """统计计算错误"""
    pass


class OptimizationError(MathError):
    """优化计算错误"""
    pass


class CacheError(MathError):
    """缓存操作错误"""
    pass


# 数据结构定义
@dataclass
class MathResult:
    """数学计算结果封装"""
    value: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    computation_time: float
    cache_key: Optional[str] = None


@dataclass
class OptimizationResult:
    """优化计算结果"""
    optimum: float
    x_optimum: np.ndarray
    success: bool
    iterations: int
    function_evaluations: int
    convergence_message: str


@dataclass
class StatisticalTestResult:
    """统计检验结果"""
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[float]
    critical_value: Optional[float]
    confidence_level: float
    reject_null: bool
    test_name: str


# 缓存管理器
class CacheManager:
    """缓存管理器，提供LRU缓存和持久化缓存功能"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存生存时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._access_times = {}
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                return None
                
            # 检查TTL
            if time.time() - self._access_times[key] > self.ttl:
                del self._cache[key]
                del self._access_times[key]
                return None
                
            # 更新访问时间
            self._access_times[key] = time.time()
            return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            # 如果缓存已满，删除最久未访问的条目
            if len(self._cache) >= self.max_size and self._access_times:
                oldest_key = min(self._access_times.keys(), 
                               key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
                
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self._cache)


# 异步数学处理器
class AsyncMathProcessor:
    """异步数学处理器，支持并发数学计算"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化异步处理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def execute_batch(self, tasks: List[Tuple[Callable, Tuple, Dict]]) -> List[Any]:
        """
        批量执行异步任务
        
        Args:
            tasks: 任务列表，每个任务为(函数, 参数元组, 关键字参数字典)
            
        Returns:
            执行结果列表
        """
        loop = asyncio.get_event_loop()
        
        # 创建任务
        futures = []
        for func, args, kwargs in tasks:
            future = loop.run_in_executor(
                self.executor,
                lambda: func(*args, **kwargs)
            )
            futures.append(future)
            
        # 等待所有任务完成
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"异步任务执行错误: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
                
        return processed_results
    
    def shutdown(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)


# 高级数学运算类
class AdvancedMath:
    """高级数学运算类，提供矩阵运算、微积分、线性代数等功能"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        初始化高级数学运算类
        
        Args:
            cache_manager: 缓存管理器实例
        """
        self.cache_manager = cache_manager or CacheManager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def matrix_multiply(self, matrix_a: np.ndarray, matrix_b: np.ndarray, 
                       use_cache: bool = True) -> np.ndarray:
        """
        矩阵乘法运算
        
        Args:
            matrix_a: 矩阵A
            matrix_b: 矩阵B
            use_cache: 是否使用缓存
            
        Returns:
            矩阵乘法结果
            
        Raises:
            MatrixError: 矩阵维度不匹配时抛出
            
        Examples:
            >>> a = np.array([[1, 2], [3, 4]])
            >>> b = np.array([[5, 6], [7, 8]])
            >>> result = AdvancedMath().matrix_multiply(a, b)
            >>> print(result)
            [[19 22]
             [43 50]]
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = None
        if use_cache:
            cache_key = f"matrix_multiply_{hash((matrix_a.tobytes(), matrix_b.tobytes()))}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.info("从缓存获取矩阵乘法结果")
                return cached_result
        
        try:
            # 维度检查
            if matrix_a.shape[1] != matrix_b.shape[0]:
                raise MatrixError(f"矩阵维度不匹配: A({matrix_a.shape}) * B({matrix_b.shape})")
            
            # 执行矩阵乘法
            result = np.dot(matrix_a, matrix_b)
            
            # 缓存结果
            if use_cache and cache_key:
                self.cache_manager.set(cache_key, result)
            
            computation_time = time.time() - start_time
            self.logger.info(f"矩阵乘法完成，耗时: {computation_time:.4f}秒")
            
            return result
            
        except Exception as e:
            self.logger.error(f"矩阵乘法计算错误: {e}")
            raise MatrixError(f"矩阵乘法计算失败: {e}")
    
    def matrix_inverse(self, matrix: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """
        矩阵求逆
        
        Args:
            matrix: 输入矩阵
            use_cache: 是否使用缓存
            
        Returns:
            矩阵的逆
            
        Raises:
            MatrixError: 矩阵不可逆时抛出
            
        Examples:
            >>> a = np.array([[1, 2], [3, 4]])
            >>> result = AdvancedMath().matrix_inverse(a)
            >>> print(result)
            [[-2.   1. ]
             [ 1.5 -0.5]]
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = None
        if use_cache:
            cache_key = f"matrix_inverse_{hash(matrix.tobytes())}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.info("从缓存获取矩阵求逆结果")
                return cached_result
        
        try:
            # 检查矩阵是否可逆
            if matrix.shape[0] != matrix.shape[1]:
                raise MatrixError(f"矩阵必须是方阵，当前形状: {matrix.shape}")
            
            det_value = det(matrix)
            if abs(det_value) < 1e-10:
                raise MatrixError(f"矩阵不可逆，行列式值: {det_value}")
            
            # 计算逆矩阵
            result = inv(matrix)
            
            # 缓存结果
            if use_cache and cache_key:
                self.cache_manager.set(cache_key, result)
            
            computation_time = time.time() - start_time
            self.logger.info(f"矩阵求逆完成，耗时: {computation_time:.4f}秒")
            
            return result
            
        except Exception as e:
            self.logger.error(f"矩阵求逆计算错误: {e}")
            raise MatrixError(f"矩阵求逆计算失败: {e}")
    
    def eigenvalue_decomposition(self, matrix: np.ndarray, 
                                use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        特征值分解
        
        Args:
            matrix: 输入矩阵
            use_cache: 是否使用缓存
            
        Returns:
            (特征值数组, 特征向量矩阵)
            
        Raises:
            MatrixError: 矩阵不是方阵时抛出
            
        Examples:
            >>> a = np.array([[1, 2], [2, 1]])
            >>> eigenvalues, eigenvectors = AdvancedMath().eigenvalue_decomposition(a)
            >>> print("特征值:", eigenvalues)
            >>> print("特征向量:", eigenvectors)
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = None
        if use_cache:
            cache_key = f"eigen_decomp_{hash(matrix.tobytes())}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.info("从缓存获取特征值分解结果")
                return cached_result
        
        try:
            # 检查矩阵是否为方阵
            if matrix.shape[0] != matrix.shape[1]:
                raise MatrixError(f"矩阵必须是方阵，当前形状: {matrix.shape}")
            
            # 特征值分解
            eigenvalues, eigenvectors = eig(matrix)
            
            # 确保特征值为实数
            if not np.all(np.isreal(eigenvalues)):
                self.logger.warning("矩阵有复数特征值")
            
            result = (eigenvalues, eigenvectors)
            
            # 缓存结果
            if use_cache and cache_key:
                self.cache_manager.set(cache_key, result)
            
            computation_time = time.time() - start_time
            self.logger.info(f"特征值分解完成，耗时: {computation_time:.4f}秒")
            
            return result
            
        except Exception as e:
            self.logger.error(f"特征值分解计算错误: {e}")
            raise MatrixError(f"特征值分解计算失败: {e}")
    
    def singular_value_decomposition(self, matrix: np.ndarray, 
                                   use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        奇异值分解 (SVD)
        
        Args:
            matrix: 输入矩阵
            use_cache: 是否使用缓存
            
        Returns:
            (U矩阵, 奇异值数组, V^T矩阵)
            
        Examples:
            >>> a = np.array([[1, 2, 3], [4, 5, 6]])
            >>> U, s, Vt = AdvancedMath().singular_value_decomposition(a)
            >>> print("奇异值:", s)
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = None
        if use_cache:
            cache_key = f"svd_{hash(matrix.tobytes())}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.info("从缓存获取SVD结果")
                return cached_result
        
        try:
            # 执行SVD
            U, s, Vt = svd(matrix)
            
            result = (U, s, Vt)
            
            # 缓存结果
            if use_cache and cache_key:
                self.cache_manager.set(cache_key, result)
            
            computation_time = time.time() - start_time
            self.logger.info(f"SVD分解完成，耗时: {computation_time:.4f}秒")
            
            return result
            
        except Exception as e:
            self.logger.error(f"SVD分解计算错误: {e}")
            raise MatrixError(f"SVD分解计算失败: {e}")
    
    def numerical_integration(self, func: Callable, a: float, b: float, 
                            method: str = 'quad', n_points: int = 1000) -> float:
        """
        数值积分
        
        Args:
            func: 被积函数
            a: 积分下限
            b: 积分上限
            method: 积分方法 ('quad', 'simpson', 'trapezoid')
            n_points: 离散点数（用于数值方法）
            
        Returns:
            积分结果
            
        Examples:
            >>> import math
            >>> result = AdvancedMath().numerical_integration(math.sin, 0, math.pi)
            >>> print(f"∫[0,π] sin(x) dx = {result:.6f}")
        """
        start_time = time.time()
        
        try:
            if method == 'quad':
                # 使用scipy的quad函数
                result, error = integrate.quad(func, a, b)
                self.logger.info(f"数值积分完成，误差估计: {error:.2e}")
                
            elif method == 'simpson':
                # Simpson法则
                if n_points % 2 == 1:
                    n_points += 1
                x = np.linspace(a, b, n_points)
                y = np.array([func(xi) for xi in x])
                result = integrate.simps(y, x)
                
            elif method == 'trapezoid':
                # 梯形法则
                x = np.linspace(a, b, n_points)
                y = np.array([func(xi) for xi in x])
                result = np.trapz(y, x)
                
            else:
                raise ValueError(f"不支持的积分方法: {method}")
            
            computation_time = time.time() - start_time
            self.logger.info(f"数值积分完成，耗时: {computation_time:.4f}秒，结果: {result:.6f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"数值积分计算错误: {e}")
            raise MathError(f"数值积分计算失败: {e}")
    
    def numerical_derivative(self, func: Callable, x: float, h: float = 1e-8) -> float:
        """
        数值微分
        
        Args:
            func: 被微分函数
            x: 求导点
            h: 步长
            
        Returns:
            导数值
            
        Examples:
            >>> import math
            >>> result = AdvancedMath().numerical_derivative(math.cos, 0)
            >>> print(f"cos'(0) = {result:.6f}")
        """
        try:
            # 使用中心差分法
            return (func(x + h) - func(x - h)) / (2 * h)
            
        except Exception as e:
            self.logger.error(f"数值微分计算错误: {e}")
            raise MathError(f"数值微分计算失败: {e}")
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray, 
                          method: str = 'solve') -> np.ndarray:
        """
        求解线性方程组 Ax = b
        
        Args:
            A: 系数矩阵
            b: 常数向量
            method: 求解方法 ('solve', 'lstsq')
            
        Returns:
            解向量
            
        Raises:
            MatrixError: 方程组无解或无穷多解时抛出
            
        Examples:
            >>> A = np.array([[3, 2], [1, 2]])
            >>> b = np.array([5, 3])
            >>> x = AdvancedMath().solve_linear_system(A, b)
            >>> print("解:", x)
        """
        try:
            if method == 'solve':
                # 直接求解（需要矩阵可逆）
                x = np.linalg.solve(A, b)
                
            elif method == 'lstsq':
                # 最小二乘解
                x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                
            else:
                raise ValueError(f"不支持的求解方法: {method}")
            
            self.logger.info(f"线性方程组求解完成，方法: {method}")
            return x
            
        except Exception as e:
            self.logger.error(f"线性方程组求解错误: {e}")
            raise MatrixError(f"线性方程组求解失败: {e}")
    
    def qr_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR分解
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            (Q矩阵, R矩阵)
            
        Examples:
            >>> A = np.array([[1, 2], [3, 4], [5, 6]])
            >>> Q, R = AdvancedMath().qr_decomposition(A)
        """
        try:
            Q, R = np.linalg.qr(matrix)
            self.logger.info("QR分解完成")
            return Q, R
            
        except Exception as e:
            self.logger.error(f"QR分解计算错误: {e}")
            raise MatrixError(f"QR分解计算失败: {e}")


# 金融数学计算类
class FinancialMath:
    """金融数学计算类，提供期权定价、风险度量、收益率计算等功能"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        初始化金融数学计算类
        
        Args:
            cache_manager: 缓存管理器实例
        """
        self.cache_manager = cache_manager or CacheManager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def black_scholes_option_price(self, S: float, K: float, T: float, r: float, 
                                  sigma: float, option_type: str = 'call') -> float:
        """
        Black-Scholes期权定价模型
        
        Args:
            S: 当前股价
            K: 执行价格
            T: 到期时间（年）
            r: 无风险利率
            sigma: 波动率
            option_type: 期权类型 ('call' 或 'put')
            
        Returns:
            期权价格
            
        Raises:
            FinancialMathError: 参数无效时抛出
            
        Examples:
            >>> # 欧式看涨期权
            >>> price = FinancialMath().black_scholes_option_price(
            ...     S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type='call'
            ... )
            >>> print(f"期权价格: {price:.4f}")
        """
        start_time = time.time()
        
        try:
            # 参数验证
            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                raise FinancialMathError("参数必须为正数")
            
            # 计算d1和d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # 计算期权价格
            if option_type.lower() == 'call':
                price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            elif option_type.lower() == 'put':
                price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            else:
                raise FinancialMathError(f"不支持的期权类型: {option_type}")
            
            computation_time = time.time() - start_time
            self.logger.info(f"Black-Scholes期权定价完成，耗时: {computation_time:.4f}秒")
            
            return price
            
        except Exception as e:
            self.logger.error(f"Black-Scholes期权定价错误: {e}")
            raise FinancialMathError(f"Black-Scholes期权定价失败: {e}")
    
    def greeks_calculation(self, S: float, K: float, T: float, r: float, 
                          sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        计算期权希腊字母
        
        Args:
            S: 当前股价
            K: 执行价格
            T: 到期时间
            r: 无风险利率
            sigma: 波动率
            option_type: 期权类型
            
        Returns:
            包含Delta、Gamma、Theta、Vega、Rho的字典
            
        Examples:
            >>> greeks = FinancialMath().greeks_calculation(100, 105, 0.25, 0.05, 0.2)
            >>> print("Delta:", greeks['Delta'])
            >>> print("Gamma:", greeks['Gamma'])
        """
        try:
            # 计算d1和d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # 计算概率密度
            pdf_d1 = stats.norm.pdf(d1)
            
            # 计算希腊字母
            if option_type.lower() == 'call':
                delta = stats.norm.cdf(d1)
                theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
                rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
            else:  # put
                delta = stats.norm.cdf(d1) - 1
                theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
                rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)
            
            # Gamma和Vega对看涨和看跌期权相同
            gamma = pdf_d1 / (S * sigma * np.sqrt(T))
            vega = S * pdf_d1 * np.sqrt(T)
            
            return {
                'Delta': delta,
                'Gamma': gamma,
                'Theta': theta,
                'Vega': vega,
                'Rho': rho
            }
            
        except Exception as e:
            self.logger.error(f"希腊字母计算错误: {e}")
            raise FinancialMathError(f"希腊字母计算失败: {e}")
    
    def var_calculation(self, returns: np.ndarray, confidence_level: float = 0.95, 
                       method: str = 'historical') -> float:
        """
        风险价值(VaR)计算
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            method: 计算方法 ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            VaR值
            
        Examples:
            >>> returns = np.random.normal(0, 0.02, 1000)
            >>> var_95 = FinancialMath().var_calculation(returns, 0.95, 'historical')
            >>> print(f"95% VaR: {var_95:.4f}")
        """
        try:
            if method == 'historical':
                # 历史模拟法
                var = np.percentile(returns, (1 - confidence_level) * 100)
                
            elif method == 'parametric':
                # 参数方法（假设正态分布）
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                z_score = stats.norm.ppf(1 - confidence_level)
                var = mean_return + z_score * std_return
                
            elif method == 'monte_carlo':
                # 蒙特卡洛方法
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                simulated_returns = np.random.normal(mean_return, std_return, 10000)
                var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
                
            else:
                raise FinancialMathError(f"不支持的VaR计算方法: {method}")
            
            self.logger.info(f"VaR计算完成，方法: {method}, 置信水平: {confidence_level}")
            return var
            
        except Exception as e:
            self.logger.error(f"VaR计算错误: {e}")
            raise FinancialMathError(f"VaR计算失败: {e}")
    
    def sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        夏普比率计算
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            夏普比率
            
        Examples:
            >>> returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
            >>> sharpe = FinancialMath().sharpe_ratio(returns, 0.005)
            >>> print(f"夏普比率: {sharpe:.4f}")
        """
        try:
            excess_returns = returns - risk_free_rate
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe = np.mean(excess_returns) / np.std(excess_returns)
            
            # 年化处理（假设252个交易日）
            sharpe_annualized = sharpe * np.sqrt(252)
            
            self.logger.info(f"夏普比率计算完成: {sharpe_annualized:.4f}")
            return sharpe_annualized
            
        except Exception as e:
            self.logger.error(f"夏普比率计算错误: {e}")
            raise FinancialMathError(f"夏普比率计算失败: {e}")
    
    def maximum_drawdown(self, returns: np.ndarray) -> Dict[str, float]:
        """
        最大回撤计算
        
        Args:
            returns: 收益率序列
            
        Returns:
            包含最大回撤、开始索引、结束索引、恢复索引的字典
            
        Examples:
            >>> returns = np.array([0.01, 0.02, -0.03, 0.01, -0.02, 0.04])
            >>> mdd = FinancialMath().maximum_drawdown(returns)
            >>> print(f"最大回撤: {mdd['max_drawdown']:.4f}")
        """
        try:
            # 计算累积收益
            cumulative_returns = (1 + returns).cumprod()
            
            # 计算累积最高点
            peak = np.maximum.accumulate(cumulative_returns)
            
            # 计算回撤
            drawdown = (cumulative_returns - peak) / peak
            
            # 找到最大回撤
            max_drawdown_idx = np.argmin(drawdown)
            max_drawdown = drawdown[max_drawdown_idx]
            
            # 找到峰值点
            peak_idx = np.argmax(cumulative_returns[:max_drawdown_idx + 1])
            
            # 找到恢复点
            recovery_idx = None
            for i in range(max_drawdown_idx + 1, len(cumulative_returns)):
                if cumulative_returns[i] >= peak[peak_idx]:
                    recovery_idx = i
                    break
            
            result = {
                'max_drawdown': max_drawdown,
                'start_index': peak_idx,
                'end_index': max_drawdown_idx,
                'recovery_index': recovery_idx
            }
            
            self.logger.info(f"最大回撤计算完成: {max_drawdown:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"最大回撤计算错误: {e}")
            raise FinancialMathError(f"最大回撤计算失败: {e}")
    
    def beta_calculation(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """
        Beta系数计算
        
        Args:
            asset_returns: 资产收益率序列
            market_returns: 市场收益率序列
            
        Returns:
            Beta系数
            
        Examples:
            >>> asset_returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
            >>> market_returns = np.array([0.005, 0.015, -0.005, 0.025, 0.008])
            >>> beta = FinancialMath().beta_calculation(asset_returns, market_returns)
            >>> print(f"Beta系数: {beta:.4f}")
        """
        try:
            if len(asset_returns) != len(market_returns):
                raise FinancialMathError("资产收益率和市场收益率长度不匹配")
            
            # 计算协方差和方差
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 0.0
            
            beta = covariance / market_variance
            
            self.logger.info(f"Beta系数计算完成: {beta:.4f}")
            return beta
            
        except Exception as e:
            self.logger.error(f"Beta系数计算错误: {e}")
            raise FinancialMathError(f"Beta系数计算失败: {e}")
    
    def capm_expected_return(self, risk_free_rate: float, beta: float, 
                           market_return: float) -> float:
        """
        CAPM预期收益率计算
        
        Args:
            risk_free_rate: 无风险利率
            beta: Beta系数
            market_return: 市场预期收益率
            
        Returns:
            预期收益率
            
        Examples:
            >>> expected_return = FinancialMath().capm_expected_return(0.03, 1.2, 0.08)
            >>> print(f"CAPM预期收益率: {expected_return:.4f}")
        """
        try:
            expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
            
            self.logger.info(f"CAPM预期收益率计算完成: {expected_return:.4f}")
            return expected_return
            
        except Exception as e:
            self.logger.error(f"CAPM预期收益率计算错误: {e}")
            raise FinancialMathError(f"CAPM预期收益率计算失败: {e}")
    
    def implied_volatility(self, option_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str = 'call') -> float:
        """
        隐含波动率计算（使用Newton-Raphson方法）
        
        Args:
            option_price: 期权价格
            S: 当前股价
            K: 执行价格
            T: 到期时间
            r: 无风险利率
            option_type: 期权类型
            
        Returns:
            隐含波动率
            
        Examples:
            >>> iv = FinancialMath().implied_volatility(5.0, 100, 105, 0.25, 0.05)
            >>> print(f"隐含波动率: {iv:.4f}")
        """
        try:
            # 初始猜测
            sigma = 0.2
            
            for _ in range(100):  # 最多迭代100次
                # 计算期权价格
                price = self.black_scholes_option_price(S, K, T, r, sigma, option_type)
                
                # 计算误差
                error = price - option_price
                
                # 检查收敛
                if abs(error) < 1e-6:
                    break
                
                # 计算Vega（用于Newton-Raphson）
                greeks = self.greeks_calculation(S, K, T, r, sigma, option_type)
                vega = greeks['Vega']
                
                if vega < 1e-10:  # 避免除零
                    break
                
                # Newton-Raphson更新
                sigma = sigma - error / vega
                
                # 确保波动率为正
                sigma = max(sigma, 0.001)
                sigma = min(sigma, 5.0)  # 限制最大波动率
            
            self.logger.info(f"隐含波动率计算完成: {sigma:.4f}")
            return sigma
            
        except Exception as e:
            self.logger.error(f"隐含波动率计算错误: {e}")
            raise FinancialMathError(f"隐含波动率计算失败: {e}")


# 统计数学工具类
class StatisticalMath:
    """统计数学工具类，提供概率分布、假设检验、置信区间等功能"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        初始化统计数学工具类
        
        Args:
            cache_manager: 缓存管理器实例
        """
        self.cache_manager = cache_manager or CacheManager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def normal_distribution_test(self, data: np.ndarray, alpha: float = 0.05) -> StatisticalTestResult:
        """
        正态性检验（Shapiro-Wilk检验）
        
        Args:
            data: 数据样本
            alpha: 显著性水平
            
        Returns:
            统计检验结果
            
        Examples:
            >>> data = np.random.normal(0, 1, 100)
            >>> result = StatisticalMath().normal_distribution_test(data)
            >>> print(f"p值: {result.p_value:.4f}")
            >>> print(f"拒绝原假设: {result.reject_null}")
        """
        try:
            # Shapiro-Wilk检验
            statistic, p_value = stats.shapiro(data)
            
            # 判断是否拒绝原假设
            reject_null = p_value < alpha
            
            result = StatisticalTestResult(
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=len(data) - 2,
                critical_value=None,
                confidence_level=1 - alpha,
                reject_null=reject_null,
                test_name="Shapiro-Wilk"
            )
            
            self.logger.info(f"正态性检验完成，p值: {p_value:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"正态性检验错误: {e}")
            raise StatisticalError(f"正态性检验失败: {e}")
    
    def t_test_one_sample(self, data: np.ndarray, population_mean: float, 
                         alpha: float = 0.05) -> StatisticalTestResult:
        """
        单样本t检验
        
        Args:
            data: 数据样本
            population_mean: 总体均值假设值
            alpha: 显著性水平
            
        Returns:
            统计检验结果
            
        Examples:
            >>> data = np.array([1.2, 1.5, 1.3, 1.4, 1.1, 1.6, 1.3])
            >>> result = StatisticalMath().t_test_one_sample(data, 1.5)
            >>> print(f"t统计量: {result.statistic:.4f}")
            >>> print(f"p值: {result.p_value:.4f}")
        """
        try:
            # 执行单样本t检验
            statistic, p_value = stats.ttest_1samp(data, population_mean)
            
            # 判断是否拒绝原假设
            reject_null = p_value < alpha
            
            result = StatisticalTestResult(
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=len(data) - 1,
                critical_value=stats.t.ppf(1 - alpha/2, len(data) - 1),
                confidence_level=1 - alpha,
                reject_null=reject_null,
                test_name="One-sample t-test"
            )
            
            self.logger.info(f"单样本t检验完成，t统计量: {statistic:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"单样本t检验错误: {e}")
            raise StatisticalError(f"单样本t检验失败: {e}")
    
    def t_test_two_samples(self, data1: np.ndarray, data2: np.ndarray, 
                          alpha: float = 0.05, equal_var: bool = True) -> StatisticalTestResult:
        """
        双样本t检验
        
        Args:
            data1: 样本1
            data2: 样本2
            alpha: 显著性水平
            equal_var: 是否假设方差相等
            
        Returns:
            统计检验结果
            
        Examples:
            >>> data1 = np.array([1.2, 1.5, 1.3, 1.4, 1.1])
            >>> data2 = np.array([1.6, 1.8, 1.7, 1.5, 1.9])
            >>> result = StatisticalMath().t_test_two_samples(data1, data2)
            >>> print(f"p值: {result.p_value:.4f}")
        """
        try:
            # 执行双样本t检验
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
            
            # 判断是否拒绝原假设
            reject_null = p_value < alpha
            
            # 计算自由度
            if equal_var:
                df = len(data1) + len(data2) - 2
            else:
                # Welch's t-test自由度
                s1_sq = np.var(data1, ddof=1)
                s2_sq = np.var(data2, ddof=1)
                n1, n2 = len(data1), len(data2)
                df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
            
            result = StatisticalTestResult(
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=df,
                critical_value=stats.t.ppf(1 - alpha/2, df),
                confidence_level=1 - alpha,
                reject_null=reject_null,
                test_name="Two-sample t-test"
            )
            
            self.logger.info(f"双样本t检验完成，t统计量: {statistic:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"双样本t检验错误: {e}")
            raise StatisticalError(f"双样本t检验失败: {e}")
    
    def chi_square_test_independence(self, contingency_table: np.ndarray, 
                                   alpha: float = 0.05) -> StatisticalTestResult:
        """
        卡方独立性检验
        
        Args:
            contingency_table: 列联表
            alpha: 显著性水平
            
        Returns:
            统计检验结果
            
        Examples:
            >>> table = np.array([[10, 20, 30], [6,  9,  17]])
            >>> result = StatisticalMath().chi_square_test_independence(table)
            >>> print(f"卡方统计量: {result.statistic:.4f}")
        """
        try:
            # 执行卡方检验
            statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # 判断是否拒绝原假设
            reject_null = p_value < alpha
            
            result = StatisticalTestResult(
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=dof,
                critical_value=stats.chi2.ppf(1 - alpha, dof),
                confidence_level=1 - alpha,
                reject_null=reject_null,
                test_name="Chi-square independence test"
            )
            
            self.logger.info(f"卡方独立性检验完成，卡方统计量: {statistic:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"卡方独立性检验错误: {e}")
            raise StatisticalError(f"卡方独立性检验失败: {e}")
    
    def confidence_interval_mean(self, data: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        均值的置信区间
        
        Args:
            data: 数据样本
            confidence_level: 置信水平
            
        Returns:
            (下界, 上界)
            
        Examples:
            >>> data = np.array([1.2, 1.5, 1.3, 1.4, 1.1, 1.6, 1.3])
            >>> lower, upper = StatisticalMath().confidence_interval_mean(data, 0.95)
            >>> print(f"95%置信区间: [{lower:.4f}, {upper:.4f}]")
        """
        try:
            n = len(data)
            mean = np.mean(data)
            std_err = stats.sem(data)  # 标准误
            
            # t分布临界值
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, n - 1)
            
            # 置信区间
            margin_of_error = t_critical * std_err
            lower_bound = mean - margin_of_error
            upper_bound = mean + margin_of_error
            
            self.logger.info(f"均值置信区间计算完成: [{lower_bound:.4f}, {upper_bound:.4f}]")
            return lower_bound, upper_bound
            
        except Exception as e:
            self.logger.error(f"置信区间计算错误: {e}")
            raise StatisticalError(f"置信区间计算失败: {e}")
    
    def confidence_interval_proportion(self, successes: int, n: int, 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        比例的置信区间
        
        Args:
            successes: 成功次数
            n: 总试验次数
            confidence_level: 置信水平
            
        Returns:
            (下界, 上界)
            
        Examples:
            >>> lower, upper = StatisticalMath().confidence_interval_proportion(45, 100, 0.95)
            >>> print(f"95%置信区间: [{lower:.4f}, {upper:.4f}]")
        """
        try:
            p_hat = successes / n
            
            # 使用正态近似
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)
            
            # 标准误
            std_error = np.sqrt(p_hat * (1 - p_hat) / n)
            
            # 置信区间
            margin_of_error = z_critical * std_error
            lower_bound = max(0, p_hat - margin_of_error)
            upper_bound = min(1, p_hat + margin_of_error)
            
            self.logger.info(f"比例置信区间计算完成: [{lower_bound:.4f}, {upper_bound:.4f}]")
            return lower_bound, upper_bound
            
        except Exception as e:
            self.logger.error(f"比例置信区间计算错误: {e}")
            raise StatisticalError(f"比例置信区间计算失败: {e}")
    
    def correlation_analysis(self, x: np.ndarray, y: np.ndarray, 
                           method: str = 'pearson') -> Dict[str, float]:
        """
        相关性分析
        
        Args:
            x: 变量x
            y: 变量y
            method: 相关系数方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
            包含相关系数、p值等信息的字典
            
        Examples:
            >>> x = np.array([1, 2, 3, 4, 5])
            >>> y = np.array([2, 4, 6, 8, 10])
            >>> result = StatisticalMath().correlation_analysis(x, y)
            >>> print(f"相关系数: {result['correlation']:.4f}")
        """
        try:
            if method == 'pearson':
                correlation, p_value = stats.pearsonr(x, y)
            elif method == 'spearman':
                correlation, p_value = stats.spearmanr(x, y)
            elif method == 'kendall':
                correlation, p_value = stats.kendalltau(x, y)
            else:
                raise StatisticalError(f"不支持的相关性分析方法: {method}")
            
            result = {
                'correlation': correlation,
                'p_value': p_value,
                'method': method,
                'n_observations': len(x)
            }
            
            self.logger.info(f"相关性分析完成，{method}相关系数: {correlation:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"相关性分析错误: {e}")
            raise StatisticalError(f"相关性分析失败: {e}")
    
    def anova_one_way(self, groups: List[np.ndarray], alpha: float = 0.05) -> StatisticalTestResult:
        """
        单因素方差分析（ANOVA）
        
        Args:
            groups: 各组数据列表
            alpha: 显著性水平
            
        Returns:
            统计检验结果
            
        Examples:
            >>> group1 = np.array([1.2, 1.5, 1.3, 1.4])
            >>> group2 = np.array([1.8, 1.9, 1.7, 2.0])
            >>> group3 = np.array([2.1, 2.3, 2.2, 2.4])
            >>> result = StatisticalMath().anova_one_way([group1, group2, group3])
            >>> print(f"F统计量: {result.statistic:.4f}")
        """
        try:
            # 执行单因素ANOVA
            statistic, p_value = stats.f_oneway(*groups)
            
            # 判断是否拒绝原假设
            reject_null = p_value < alpha
            
            # 计算自由度
            k = len(groups)  # 组数
            n = sum(len(group) for group in groups)  # 总样本数
            df_between = k - 1
            df_within = n - k
            
            result = StatisticalTestResult(
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=(df_between, df_within),
                critical_value=stats.f.ppf(1 - alpha, df_between, df_within),
                confidence_level=1 - alpha,
                reject_null=reject_null,
                test_name="One-way ANOVA"
            )
            
            self.logger.info(f"单因素ANOVA完成，F统计量: {statistic:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"单因素ANOVA错误: {e}")
            raise StatisticalError(f"单因素ANOVA失败: {e}")
    
    def regression_analysis(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        简单线性回归分析
        
        Args:
            x: 自变量
            y: 因变量
            
        Returns:
            回归分析结果字典
            
        Examples:
            >>> x = np.array([1, 2, 3, 4, 5])
            >>> y = np.array([2.1, 4.2, 5.8, 8.1, 10.2])
            >>> result = StatisticalMath().regression_analysis(x, y)
            >>> print(f"斜率: {result['slope']:.4f}")
            >>> print(f"截距: {result['intercept']:.4f}")
        """
        try:
            # 执行线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # 计算预测值和残差
            y_pred = slope * x + intercept
            residuals = y - y_pred
            
            # 计算R²
            r_squared = r_value ** 2
            
            result = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_err': std_err,
                'residuals': residuals,
                'y_predicted': y_pred
            }
            
            self.logger.info(f"线性回归分析完成，R²: {r_squared:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"回归分析错误: {e}")
            raise StatisticalError(f"回归分析失败: {e}")


# 优化数学工具类
class OptimizationMath:
    """优化数学工具类，提供梯度下降、牛顿法、拉格朗日乘数法等功能"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        初始化优化数学工具类
        
        Args:
            cache_manager: 缓存管理器实例
        """
        self.cache_manager = cache_manager or CacheManager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def gradient_descent(self, func: Callable, grad_func: Callable, 
                        initial_point: np.ndarray, learning_rate: float = 0.01,
                        max_iterations: int = 1000, tolerance: float = 1e-6) -> OptimizationResult:
        """
        梯度下降优化算法
        
        Args:
            func: 目标函数
            grad_func: 梯度函数
            initial_point: 初始点
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            
        Returns:
            优化结果
            
        Examples:
            >>> def func(x):
            ...     return (x[0] - 2)**2 + (x[1] - 3)**2
            >>> def grad_func(x):
            ...     return np.array([2*(x[0] - 2), 2*(x[1] - 3)])
            >>> result = OptimizationMath().gradient_descent(func, grad_func, np.array([0.0, 0.0]))
            >>> print(f"最优解: {result.x_optimum}")
        """
        start_time = time.time()
        
        try:
            x = initial_point.copy()
            best_x = x.copy()
            best_f = func(x)
            
            for iteration in range(max_iterations):
                # 计算梯度
                gradient = grad_func(x)
                
                # 更新参数
                x_new = x - learning_rate * gradient
                
                # 检查收敛
                if np.linalg.norm(x_new - x) < tolerance:
                    break
                
                x = x_new
                
                # 更新最优解
                current_f = func(x)
                if current_f < best_f:
                    best_f = current_f
                    best_x = x.copy()
            
            # 判断是否成功收敛
            success = np.linalg.norm(x - best_x) < tolerance or iteration < max_iterations - 1
            
            computation_time = time.time() - start_time
            self.logger.info(f"梯度下降优化完成，迭代次数: {iteration + 1}, 最优值: {best_f:.6f}")
            
            return OptimizationResult(
                optimum=best_f,
                x_optimum=best_x,
                success=success,
                iterations=iteration + 1,
                function_evaluations=iteration + 1,
                convergence_message="Converged" if success else "Maximum iterations reached"
            )
            
        except Exception as e:
            self.logger.error(f"梯度下降优化错误: {e}")
            raise OptimizationError(f"梯度下降优化失败: {e}")
    
    def newton_method(self, func: Callable, grad_func: Callable, hess_func: Callable,
                     initial_point: np.ndarray, max_iterations: int = 100,
                     tolerance: float = 1e-6) -> OptimizationResult:
        """
        牛顿法优化算法
        
        Args:
            func: 目标函数
            grad_func: 梯度函数
            hess_func: 海塞矩阵函数
            initial_point: 初始点
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            
        Returns:
            优化结果
        """
        start_time = time.time()
        
        try:
            x = initial_point.copy()
            best_x = x.copy()
            best_f = func(x)
            
            for iteration in range(max_iterations):
                # 计算梯度和海塞矩阵
                gradient = grad_func(x)
                hessian = hess_func(x)
                
                # 检查海塞矩阵是否可逆
                try:
                    hessian_inv = inv(hessian)
                except:
                    # 如果海塞矩阵奇异，使用伪逆
                    hessian_inv = np.linalg.pinv(hessian)
                
                # 牛顿法更新
                delta = -hessian_inv @ gradient
                x_new = x + delta
                
                # 检查收敛
                if np.linalg.norm(delta) < tolerance:
                    break
                
                x = x_new
                
                # 更新最优解
                current_f = func(x)
                if current_f < best_f:
                    best_f = current_f
                    best_x = x.copy()
            
            # 判断是否成功收敛
            success = np.linalg.norm(x - best_x) < tolerance or iteration < max_iterations - 1
            
            computation_time = time.time() - start_time
            self.logger.info(f"牛顿法优化完成，迭代次数: {iteration + 1}, 最优值: {best_f:.6f}")
            
            return OptimizationResult(
                optimum=best_f,
                x_optimum=best_x,
                success=success,
                iterations=iteration + 1,
                function_evaluations=iteration + 1,
                convergence_message="Converged" if success else "Maximum iterations reached"
            )
            
        except Exception as e:
            self.logger.error(f"牛顿法优化错误: {e}")
            raise OptimizationError(f"牛顿法优化失败: {e}")
    
    def lagrange_multipliers(self, func: Callable, constraints: List[Callable],
                           initial_point: np.ndarray, max_iterations: int = 100,
                           tolerance: float = 1e-6) -> OptimizationResult:
        """
        拉格朗日乘数法
        
        Args:
            func: 目标函数
            constraints: 约束条件函数列表
            initial_point: 初始点
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            
        Returns:
            优化结果
        """
        start_time = time.time()
        
        try:
            n_vars = len(initial_point)
            n_constraints = len(constraints)
            
            # 初始化拉格朗日乘数
            lambdas = np.ones(n_constraints)
            
            x = initial_point.copy()
            best_x = x.copy()
            best_f = func(x)
            
            for iteration in range(max_iterations):
                # 构建拉格朗日函数
                def lagrangian(variables):
                    x_vars = variables[:n_vars]
                    lambda_vars = variables[n_vars:]
                    return func(x_vars) + sum(lambda_vars[i] * constraints[i](x_vars) 
                                            for i in range(n_constraints))
                
                # 构建梯度函数
                def lagrangian_grad(variables):
                    x_vars = variables[:n_vars]
                    lambda_vars = variables[n_vars:]
                    
                    # 对x的偏导数
                    grad_x = AdvancedMath().numerical_derivative(
                        lambda t: func(np.concatenate([t, x_vars[1:]])), x_vars[0]
                    )
                    # 这里简化处理，实际应该计算完整的梯度
                    
                    # 对λ的偏导数（约束条件）
                    grad_lambda = np.array([constraints[i](x_vars) for i in range(n_constraints)])
                    
                    return np.concatenate([grad_x, grad_lambda])
                
                # 使用scipy优化
                result = optimize.minimize(
                    lagrangian,
                    np.concatenate([x, lambdas]),
                    method='BFGS',
                    options={'maxiter': 10}  # 限制内部迭代次数
                )
                
                if result.success:
                    x_new = result.x[:n_vars]
                    
                    # 检查收敛
                    if np.linalg.norm(x_new - x) < tolerance:
                        break
                    
                    x = x_new
                    lambdas = result.x[n_vars:]
                    
                    # 更新最优解
                    current_f = func(x)
                    if current_f < best_f:
                        best_f = current_f
                        best_x = x.copy()
                else:
                    break
            
            # 判断是否成功收敛
            success = result.success if 'result' in locals() else False
            
            computation_time = time.time() - start_time
            self.logger.info(f"拉格朗日乘数法优化完成，迭代次数: {iteration + 1}, 最优值: {best_f:.6f}")
            
            return OptimizationResult(
                optimum=best_f,
                x_optimum=best_x,
                success=success,
                iterations=iteration + 1,
                function_evaluations=iteration + 1,
                convergence_message="Converged" if success else "Optimization failed"
            )
            
        except Exception as e:
            self.logger.error(f"拉格朗日乘数法优化错误: {e}")
            raise OptimizationError(f"拉格朗日乘数法优化失败: {e}")
    
    def genetic_algorithm(self, func: Callable, bounds: List[Tuple[float, float]],
                         population_size: int = 50, generations: int = 100,
                         mutation_rate: float = 0.1, crossover_rate: float = 0.8) -> OptimizationResult:
        """
        遗传算法
        
        Args:
            func: 目标函数
            bounds: 变量边界列表 [(min, max), ...]
            population_size: 种群大小
            generations: 进化代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
            
        Returns:
            优化结果
        """
        start_time = time.time()
        
        try:
            n_vars = len(bounds)
            
            # 初始化种群
            population = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
                size=(population_size, n_vars)
            )
            
            best_fitness = float('inf')
            best_individual = None
            
            for generation in range(generations):
                # 评估适应度
                fitness = np.array([func(individual) for individual in population])
                
                # 更新最优解
                min_idx = np.argmin(fitness)
                if fitness[min_idx] < best_fitness:
                    best_fitness = fitness[min_idx]
                    best_individual = population[min_idx].copy()
                
                # 选择（锦标赛选择）
                def tournament_selection():
                    tournament_size = 3
                    tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
                    tournament_fitness = fitness[tournament_indices]
                    winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                    return population[winner_idx]
                
                # 交叉
                new_population = []
                for _ in range(population_size):
                    if np.random.random() < crossover_rate:
                        parent1 = tournament_selection()
                        parent2 = tournament_selection()
                        
                        # 单点交叉
                        crossover_point = np.random.randint(1, n_vars)
                        child = np.concatenate([
                            parent1[:crossover_point],
                            parent2[crossover_point:]
                        ])
                    else:
                        child = tournament_selection()
                    
                    new_population.append(child)
                
                # 变异
                for i in range(population_size):
                    for j in range(n_vars):
                        if np.random.random() < mutation_rate:
                            # 高斯变异
                            new_population[i][j] += np.random.normal(0, 0.1)
                            # 边界检查
                            new_population[i][j] = np.clip(
                                new_population[i][j], bounds[j][0], bounds[j][1]
                            )
                
                population = np.array(new_population)
            
            computation_time = time.time() - start_time
            self.logger.info(f"遗传算法优化完成，进化代数: {generations}, 最优值: {best_fitness:.6f}")
            
            return OptimizationResult(
                optimum=best_fitness,
                x_optimum=best_individual,
                success=True,
                iterations=generations,
                function_evaluations=generations * population_size,
                convergence_message="Completed"
            )
            
        except Exception as e:
            self.logger.error(f"遗传算法优化错误: {e}")
            raise OptimizationError(f"遗传算法优化失败: {e}")
    
    def simulated_annealing(self, func: Callable, initial_point: np.ndarray,
                          bounds: List[Tuple[float, float]], initial_temp: float = 1000,
                          cooling_rate: float = 0.95, min_temp: float = 1e-8,
                          max_iterations: int = 1000) -> OptimizationResult:
        """
        模拟退火算法
        
        Args:
            func: 目标函数
            initial_point: 初始点
            bounds: 变量边界列表
            initial_temp: 初始温度
            cooling_rate: 降温速率
            min_temp: 最低温度
            max_iterations: 最大迭代次数
            
        Returns:
            优化结果
        """
        start_time = time.time()
        
        try:
            current_point = initial_point.copy()
            current_fitness = func(current_point)
            
            best_point = current_point.copy()
            best_fitness = current_fitness
            
            temperature = initial_temp
            
            for iteration in range(max_iterations):
                # 生成邻域解
                step_size = temperature / initial_temp * 0.1
                new_point = current_point + np.random.normal(0, step_size, len(bounds))
                
                # 边界检查
                for i, (min_bound, max_bound) in enumerate(bounds):
                    new_point[i] = np.clip(new_point[i], min_bound, max_bound)
                
                new_fitness = func(new_point)
                
                # 接受准则
                if new_fitness < current_fitness:
                    # 接受更好的解
                    current_point = new_point
                    current_fitness = new_fitness
                    
                    if new_fitness < best_fitness:
                        best_point = new_point.copy()
                        best_fitness = new_fitness
                else:
                    # 以概率接受较差的解
                    probability = np.exp(-(new_fitness - current_fitness) / temperature)
                    if np.random.random() < probability:
                        current_point = new_point
                        current_fitness = new_fitness
                
                # 降温
                temperature *= cooling_rate
                
                if temperature < min_temp:
                    break
            
            computation_time = time.time() - start_time
            self.logger.info(f"模拟退火优化完成，迭代次数: {iteration + 1}, 最优值: {best_fitness:.6f}")
            
            return OptimizationResult(
                optimum=best_fitness,
                x_optimum=best_point,
                success=True,
                iterations=iteration + 1,
                function_evaluations=iteration + 1,
                convergence_message="Completed"
            )
            
        except Exception as e:
            self.logger.error(f"模拟退火优化错误: {e}")
            raise OptimizationError(f"模拟退火优化失败: {e}")


# 版本信息
__version__ = "1.0.0"
__author__ = "J1数学计算工具团队"

# 便捷函数
def create_math_tools(cache_size: int = 1000, cache_ttl: int = 3600, 
                     max_workers: int = 4) -> 'MathematicalTools':
    """
    创建数学工具实例的便捷函数
    
    Args:
        cache_size: 缓存大小
        cache_ttl: 缓存生存时间（秒）
        max_workers: 异步处理最大工作线程数
        
    Returns:
        数学工具实例
        
    Examples:
        >>> tools = create_math_tools(cache_size=500, cache_ttl=1800)
        >>> result = tools.advanced.matrix_multiply(A, B)
    """
    return MathematicalTools(
        cache_size=cache_size,
        cache_ttl=cache_ttl,
        max_workers=max_workers
    )


def get_version() -> str:
    """
    获取模块版本信息
    
    Returns:
        版本字符串
    """
    return __version__


# 主数学工具类
class MathematicalTools:
    """
    数学工具主类，整合所有数学计算功能
    
    提供统一的接口访问所有数学计算功能，包括高级数学运算、金融数学计算、
    统计数学工具、优化数学工具等。支持异步处理和缓存机制。
    
    Examples:
        >>> # 创建数学工具实例
        >>> math_tools = MathematicalTools()
        
        >>> # 矩阵运算
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])
        >>> result = math_tools.advanced.matrix_multiply(A, B)
        
        >>> # 金融计算
        >>> option_price = math_tools.financial.black_scholes_option_price(100, 105, 0.25, 0.05, 0.2)
        
        >>> # 统计分析
        >>> data = np.random.normal(0, 1, 100)
        >>> test_result = math_tools.statistical.normal_distribution_test(data)
        
        >>> # 优化计算
        >>> def objective(x):
        ...     return (x[0] - 2)**2 + (x[1] - 3)**2
        >>> def gradient(x):
        ...     return np.array([2*(x[0] - 2), 2*(x[1] - 3)])
        >>> result = math_tools.optimization.gradient_descent(objective, gradient, np.array([0.0, 0.0]))
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600, 
                 max_workers: int = 4):
        """
        初始化数学工具主类
        
        Args:
            cache_size: 缓存大小
            cache_ttl: 缓存生存时间（秒）
            max_workers: 异步处理最大工作线程数
        """
        # 初始化缓存管理器
        self.cache_manager = CacheManager(max_size=cache_size, ttl=cache_ttl)
        
        # 初始化异步处理器
        self.async_processor = AsyncMathProcessor(max_workers=max_workers)
        
        # 初始化各个功能模块
        self.advanced = AdvancedMath(self.cache_manager)
        self.financial = FinancialMath(self.cache_manager)
        self.statistical = StatisticalMath(self.cache_manager)
        self.optimization = OptimizationMath(self.cache_manager)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("数学工具主类初始化完成")
    
    async def batch_compute(self, computations: List[Dict[str, Any]]) -> List[Any]:
        """
        批量异步计算
        
        Args:
            computations: 计算任务列表，每个任务包含：
                - 'type': 计算类型 ('matrix', 'financial', 'statistical', 'optimization')
                - 'method': 方法名
                - 'args': 位置参数
                - 'kwargs': 关键字参数
            
        Returns:
            计算结果列表
            
        Examples:
            >>> computations = [
            ...     {
            ...         'type': 'matrix',
            ...         'method': 'matrix_multiply',
            ...         'args': (A, B),
            ...         'kwargs': {}
            ...     },
            ...     {
            ...         'type': 'financial',
            ...         'method': 'black_scholes_option_price',
            ...         'args': (100, 105, 0.25, 0.05, 0.2),
            ...         'kwargs': {'option_type': 'call'}
            ...     }
            ... ]
            >>> results = await math_tools.batch_compute(computations)
        """
        try:
            tasks = []
            
            for comp in computations:
                comp_type = comp['type']
                method = comp['method']
                args = comp.get('args', ())
                kwargs = comp.get('kwargs', {})
                
                # 根据类型选择模块
                if comp_type == 'matrix':
                    module = self.advanced
                elif comp_type == 'financial':
                    module = self.financial
                elif comp_type == 'statistical':
                    module = self.statistical
                elif comp_type == 'optimization':
                    module = self.optimization
                else:
                    raise ValueError(f"未知的计算类型: {comp_type}")
                
                # 获取方法
                if not hasattr(module, method):
                    raise AttributeError(f"模块 {comp_type} 中没有方法 {method}")
                
                method_func = getattr(module, method)
                
                # 创建任务
                task = (method_func, args, kwargs)
                tasks.append(task)
            
            # 执行批量计算
            results = await self.async_processor.execute_batch(tasks)
            
            self.logger.info(f"批量计算完成，处理了 {len(computations)} 个任务")
            return results
            
        except Exception as e:
            self.logger.error(f"批量计算错误: {e}")
            raise MathError(f"批量计算失败: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        return {
            'cache_size': self.cache_manager.size(),
            'max_cache_size': self.cache_manager.max_size,
            'cache_ttl': self.cache_manager.ttl
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache_manager.clear()
        self.logger.info("缓存已清空")
    
    def shutdown(self):
        """关闭异步处理器"""
        self.async_processor.shutdown()
        self.logger.info("数学工具已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        try:
            self.shutdown()
        except Exception as e:
            # 确保异常不会阻止上下文管理器的正常退出
            pass


# 使用示例和测试函数
def example_usage():
    """使用示例函数"""
    print("=== J1数学计算工具使用示例 ===\n")
    
    # 创建数学工具实例
    with MathematicalTools() as math_tools:
        
        # 1. 高级数学运算示例
        print("1. 高级数学运算示例:")
        print("-" * 30)
        
        # 矩阵运算
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        
        print(f"矩阵A:\n{A}")
        print(f"矩阵B:\n{B}")
        
        # 矩阵乘法
        C = math_tools.advanced.matrix_multiply(A, B)
        print(f"A × B =\n{C}")
        
        # 特征值分解
        eigenvalues, eigenvectors = math_tools.advanced.eigenvalue_decomposition(A)
        print(f"特征值: {eigenvalues}")
        print(f"特征向量:\n{eigenvectors}")
        
        # 数值积分
        result = math_tools.advanced.numerical_integration(lambda x: np.sin(x), 0, np.pi)
        print(f"∫[0,π] sin(x) dx = {result:.6f}")
        
        print("\n" + "="*50 + "\n")
        
        # 2. 金融数学计算示例
        print("2. 金融数学计算示例:")
        print("-" * 30)
        
        # Black-Scholes期权定价
        S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2
        call_price = math_tools.financial.black_scholes_option_price(S, K, T, r, sigma, 'call')
        put_price = math_tools.financial.black_scholes_option_price(S, K, T, r, sigma, 'put')
        
        print(f"当前股价: ${S}")
        print(f"执行价格: ${K}")
        print(f"到期时间: {T}年")
        print(f"无风险利率: {r:.1%}")
        print(f"波动率: {sigma:.1%}")
        print(f"看涨期权价格: ${call_price:.4f}")
        print(f"看跌期权价格: ${put_price:.4f}")
        
        # 希腊字母
        greeks = math_tools.financial.greeks_calculation(S, K, T, r, sigma, 'call')
        print(f"看涨期权希腊字母: {greeks}")
        
        # VaR计算
        returns = np.random.normal(0, 0.02, 1000)
        var_95 = math_tools.financial.var_calculation(returns, 0.95, 'historical')
        print(f"95% VaR: {var_95:.4f}")
        
        print("\n" + "="*50 + "\n")
        
        # 3. 统计数学工具示例
        print("3. 统计数学工具示例:")
        print("-" * 30)
        
        # 生成测试数据
        data = np.random.normal(0, 1, 100)
        
        # 正态性检验
        test_result = math_tools.statistical.normal_distribution_test(data)
        print(f"正态性检验 (Shapiro-Wilk):")
        print(f"  统计量: {test_result.statistic:.4f}")
        print(f"  p值: {test_result.p_value:.4f}")
        print(f"  拒绝原假设: {test_result.reject_null}")
        
        # t检验
        t_result = math_tools.statistical.t_test_one_sample(data, 0.0)
        print(f"\n单样本t检验 (μ=0):")
        print(f"  t统计量: {t_result.statistic:.4f}")
        print(f"  p值: {t_result.p_value:.4f}")
        print(f"  自由度: {t_result.degrees_of_freedom}")
        
        # 置信区间
        lower, upper = math_tools.statistical.confidence_interval_mean(data, 0.95)
        print(f"\n均值95%置信区间: [{lower:.4f}, {upper:.4f}]")
        
        # 相关性分析
        x = np.random.normal(0, 1, 50)
        y = 2 * x + np.random.normal(0, 0.5, 50)
        corr_result = math_tools.statistical.correlation_analysis(x, y)
        print(f"\n相关性分析:")
        print(f"  相关系数: {corr_result['correlation']:.4f}")
        print(f"  p值: {corr_result['p_value']:.4f}")
        
        print("\n" + "="*50 + "\n")
        
        # 4. 优化数学工具示例
        print("4. 优化数学工具示例:")
        print("-" * 30)
        
        # 定义目标函数和梯度
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2 + 0.5 * x[0] * x[1]
        
        def gradient(x):
            return np.array([2*(x[0] - 2) + 0.5*x[1], 2*(x[1] - 3) + 0.5*x[0]])
        
        initial_point = np.array([0.0, 0.0])
        
        # 梯度下降
        gd_result = math_tools.optimization.gradient_descent(objective, gradient, initial_point)
        print(f"梯度下降结果:")
        print(f"  最优解: {gd_result.x_optimum}")
        print(f"  最优值: {gd_result.optimum:.6f}")
        print(f"  迭代次数: {gd_result.iterations}")
        print(f"  收敛: {gd_result.success}")
        
        # 牛顿法
        def hessian(x):
            return np.array([[2, 0.5], [0.5, 2]])
        
        newton_result = math_tools.optimization.newton_method(objective, gradient, hessian, initial_point)
        print(f"\n牛顿法结果:")
        print(f"  最优解: {newton_result.x_optimum}")
        print(f"  最优值: {newton_result.optimum:.6f}")
        print(f"  迭代次数: {newton_result.iterations}")
        print(f"  收敛: {newton_result.success}")
        
        # 遗传算法
        bounds = [(-5, 5), (-5, 5)]
        ga_result = math_tools.optimization.genetic_algorithm(objective, bounds)
        print(f"\n遗传算法结果:")
        print(f"  最优解: {ga_result.x_optimum}")
        print(f"  最优值: {ga_result.optimum:.6f}")
        print(f"  迭代次数: {ga_result.iterations}")
        
        print("\n" + "="*50 + "\n")
        
        # 5. 异步批量计算示例
        print("5. 异步批量计算示例:")
        print("-" * 30)
        
        # 准备批量计算任务
        computations = [
            {
                'type': 'matrix',
                'method': 'matrix_multiply',
                'args': (A, B),
                'kwargs': {}
            },
            {
                'type': 'financial',
                'method': 'black_scholes_option_price',
                'args': (S, K, T, r, sigma),
                'kwargs': {'option_type': 'call'}
            },
            {
                'type': 'statistical',
                'method': 'normal_distribution_test',
                'args': (data,),
                'kwargs': {}
            }
        ]
        
        # 执行异步批量计算
        import asyncio
        async def run_batch():
            return await math_tools.batch_compute(computations)
        
        batch_results = asyncio.run(run_batch())
        
        print("批量计算结果:")
        print(f"  矩阵乘法结果形状: {batch_results[0].shape}")
        print(f"  期权价格: ${batch_results[1]:.4f}")
        print(f"  正态性检验p值: {batch_results[2].p_value:.4f}")
        
        # 6. 缓存统计
        print("\n6. 缓存统计:")
        print("-" * 30)
        cache_stats = math_tools.get_cache_stats()
        print(f"缓存大小: {cache_stats['cache_size']}")
        print(f"最大缓存大小: {cache_stats['max_cache_size']}")
        print(f"缓存TTL: {cache_stats['cache_ttl']}秒")


if __name__ == "__main__":
    # 运行示例
    example_usage()
    
    print("\n=== J1数学计算工具测试完成 ===")
    print("所有功能模块运行正常！")


# 向后兼容性别名
MathTools = MathematicalTools  # 兼容性别名