"""
J4统计工具模块

这个模块提供了全面的统计分析和计算功能，包括描述性统计、推断统计、
概率分布、贝叶斯统计、非参数统计和多变量统计分析等。

作者: J4统计团队
版本: 1.0.0
日期: 2025-11-06
"""

import asyncio
import logging
import math
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize, fsolve
from scipy.linalg import det, inv, eig, svd
import pandas as pd
from collections import defaultdict
import functools
import time
import threading
from enum import Enum

# 条件导入statsmodels，避免在没有安装的情况下出错
try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalTest(Enum):
    """统计检验类型枚举"""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    MANN_WHITNEY = "mann_whitney"
    KS_TEST = "ks_test"
    WILCOXON = "wilcoxon"


class DistributionType(Enum):
    """分布类型枚举"""
    NORMAL = "normal"
    T = "t"
    CHI_SQUARE = "chi_square"
    F = "f"
    BETA = "beta"
    GAMMA = "gamma"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    BINOMIAL = "binomial"


@dataclass
class StatisticalResult:
    """统计结果数据类"""
    statistic: float
    p_value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    degrees_of_freedom: Optional[int] = None
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    test_name: str = ""
    interpretation: str = ""
    
    def __post_init__(self):
        """后处理方法，设置解释"""
        if self.p_value < 0.001:
            self.interpretation = "极显著 (p < 0.001)"
        elif self.p_value < 0.01:
            self.interpretation = "高度显著 (p < 0.01)"
        elif self.p_value < 0.05:
            self.interpretation = "显著 (p < 0.05)"
        elif self.p_value < 0.1:
            self.interpretation = "边缘显著 (p < 0.1)"
        else:
            self.interpretation = "不显著 (p >= 0.1)"


class AsyncStatsProcessor:
    """异步统计处理器"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_batch(self, functions: List[Tuple[Callable, List[Any]]]) -> List[Any]:
        """批量异步处理统计函数"""
        tasks = []
        for func, args in functions:
            task = asyncio.create_task(
                asyncio.to_thread(func, *args)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def __del__(self):
        """析构方法，清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class DescriptiveStatistics:
    """描述性统计工具类"""
    
    @staticmethod
    def mean(data: Union[List[float], np.ndarray], weights: Optional[Union[List[float], np.ndarray]] = None) -> float:
        """
        计算加权平均值
        
        Args:
            data: 数据数组
            weights: 权重数组
            
        Returns:
            加权平均值
            
        Raises:
            ValueError: 当数据为空或权重不匹配时
        """
        try:
            data = np.array(data)
            if len(data) == 0:
                raise ValueError("数据不能为空")
            
            if weights is None:
                return float(np.mean(data))
            
            weights = np.array(weights)
            if len(weights) != len(data):
                raise ValueError("权重数组长度必须与数据长度相同")
            
            return float(np.average(data, weights=weights))
            
        except Exception as e:
            logger.error(f"计算平均值时出错: {e}")
            raise
    
    @staticmethod
    def variance(data: Union[List[float], np.ndarray], ddof: int = 0, weights: Optional[Union[List[float], np.ndarray]] = None) -> float:
        """
        计算方差
        
        Args:
            data: 数据数组
            ddof: 自由度修正 (0为总体方差，1为样本方差)
            weights: 权重数组
            
        Returns:
            方差值
        """
        try:
            data = np.array(data)
            if len(data) == 0:
                raise ValueError("数据不能为空")
            
            if weights is None:
                return float(np.var(data, ddof=ddof))
            
            weights = np.array(weights)
            weighted_mean = DescriptiveStatistics.mean(data, weights)
            return float(np.average((data - weighted_mean)**2, weights=weights))
            
        except Exception as e:
            logger.error(f"计算方差时出错: {e}")
            raise
    
    @staticmethod
    def skewness(data: Union[List[float], np.ndarray], bias: bool = True) -> float:
        """
        计算偏度
        
        Args:
            data: 数据数组
            bias: 是否使用偏差修正
            
        Returns:
            偏度值
        """
        try:
            data = np.array(data)
            if len(data) < 3:
                raise ValueError("计算偏度需要至少3个数据点")
            
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=0 if bias else 1)
            
            if std_val == 0:
                return 0.0
            
            skewness = np.mean(((data - mean_val) / std_val) ** 3)
            
            if not bias and len(data) > 2:
                # 样本偏度的偏差修正
                n = len(data)
                skewness *= math.sqrt(n * (n - 1)) / (n - 2)
            
            return float(skewness)
            
        except Exception as e:
            logger.error(f"计算偏度时出错: {e}")
            raise
    
    @staticmethod
    def kurtosis(data: Union[List[float], np.ndarray], bias: bool = True) -> float:
        """
        计算峰度
        
        Args:
            data: 数据数组
            bias: 是否使用偏差修正
            
        Returns:
            峰度值
        """
        try:
            data = np.array(data)
            if len(data) < 4:
                raise ValueError("计算峰度需要至少4个数据点")
            
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=0 if bias else 1)
            
            if std_val == 0:
                return -3.0  # 退化分布的峰度
            
            kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3
            
            if not bias and len(data) > 3:
                # 样本峰度的偏差修正
                n = len(data)
                kurtosis *= ((n + 1) * n * (n - 1)) / ((n - 2) * (n - 3) * (n - 2)) + 3
            
            return float(kurtosis)
            
        except Exception as e:
            logger.error(f"计算峰度时出错: {e}")
            raise
    
    @staticmethod
    def quantile(data: Union[List[float], np.ndarray], q: float, method: str = 'linear') -> float:
        """
        计算分位数
        
        Args:
            data: 数据数组
            q: 分位数 (0-1之间)
            method: 插值方法 ('linear', 'lower', 'higher', 'nearest', 'midpoint')
            
        Returns:
            分位数值
        """
        try:
            data = np.array(data)
            if len(data) == 0:
                raise ValueError("数据不能为空")
            
            if not 0 <= q <= 1:
                raise ValueError("分位数必须在0到1之间")
            
            return float(np.quantile(data, q, method=method))
            
        except Exception as e:
            logger.error(f"计算分位数时出错: {e}")
            raise
    
    @staticmethod
    def percentile(data: Union[List[float], np.ndarray], p: float) -> float:
        """
        计算百分位数
        
        Args:
            data: 数据数组
            p: 百分位数 (0-100之间)
            
        Returns:
            百分位数值
        """
        return DescriptiveStatistics.quantile(data, p / 100.0)
    
    @staticmethod
    def descriptive_stats(data: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """
        计算完整的描述性统计摘要
        
        Args:
            data: 数据数组
            
        Returns:
            包含所有描述性统计量的字典
        """
        try:
            data = np.array(data)
            if len(data) == 0:
                raise ValueError("数据不能为空")
            
            stats_dict = {
                'count': len(data),
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'mode': float(stats.mode(data, keepdims=True)[0][0]),
                'std': float(np.std(data, ddof=1)),
                'variance': float(np.var(data, ddof=1)),
                'skewness': DescriptiveStatistics.skewness(data, bias=False),
                'kurtosis': DescriptiveStatistics.kurtosis(data, bias=False),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'range': float(np.max(data) - np.min(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75)),
                'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
                'coefficient_of_variation': float(np.std(data, ddof=1) / np.mean(data)) if np.mean(data) != 0 else 0
            }
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"计算描述性统计时出错: {e}")
            raise


class InferentialStatistics:
    """推断统计工具类"""
    
    @staticmethod
    def t_test_one_sample(data: Union[List[float], np.ndarray], 
                         population_mean: float, 
                         alternative: str = 'two-sided',
                         alpha: float = 0.05) -> StatisticalResult:
        """
        单样本t检验
        
        Args:
            data: 样本数据
            population_mean: 总体均值假设值
            alternative: 备择假设 ('two-sided', 'greater', 'less')
            alpha: 显著性水平
            
        Returns:
            统计检验结果
        """
        try:
            data = np.array(data)
            n = len(data)
            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)
            
            if sample_std == 0:
                raise ValueError("样本标准差不能为0")
            
            # 计算t统计量
            t_stat = (sample_mean - population_mean) / (sample_std / math.sqrt(n))
            
            # 计算自由度
            df = n - 1
            
            # 计算p值
            if alternative == 'two-sided':
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            elif alternative == 'greater':
                p_value = 1 - stats.t.cdf(t_stat, df)
            elif alternative == 'less':
                p_value = stats.t.cdf(t_stat, df)
            else:
                raise ValueError("alternative参数必须是 'two-sided', 'greater', 或 'less'")
            
            # 计算置信区间
            t_critical = stats.t.ppf(1 - alpha/2, df)
            margin_error = t_critical * (sample_std / math.sqrt(n))
            ci_lower = sample_mean - margin_error
            ci_upper = sample_mean + margin_error
            
            # 效应量 (Cohen's d)
            cohens_d = (sample_mean - population_mean) / sample_std
            
            result = StatisticalResult(
                statistic=t_stat,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                degrees_of_freedom=df,
                effect_size=cohens_d,
                sample_size=n,
                test_name="单样本t检验"
            )
            
            logger.info(f"单样本t检验完成: t = {t_stat:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"单样本t检验时出错: {e}")
            raise
    
    @staticmethod
    def t_test_two_samples(data1: Union[List[float], np.ndarray],
                          data2: Union[List[float], np.ndarray],
                          equal_var: bool = False,
                          alternative: str = 'two-sided',
                          alpha: float = 0.05) -> StatisticalResult:
        """
        双样本t检验
        
        Args:
            data1: 第一组样本数据
            data2: 第二组样本数据
            equal_var: 是否假设等方差
            alternative: 备择假设 ('two-sided', 'greater', 'less')
            alpha: 显著性水平
            
        Returns:
            统计检验结果
        """
        try:
            data1 = np.array(data1)
            data2 = np.array(data2)
            
            n1, n2 = len(data1), len(data2)
            mean1, mean2 = np.mean(data1), np.mean(data2)
            std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
            
            if equal_var:
                # 假设等方差的t检验
                pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                t_stat = (mean1 - mean2) / (pooled_std * math.sqrt(1/n1 + 1/n2))
                df = n1 + n2 - 2
            else:
                # Welch's t检验 (不假设等方差)
                t_stat = (mean1 - mean2) / math.sqrt(std1**2/n1 + std2**2/n2)
                df = (std1**2/n1 + std2**2/n2)**2 / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
            
            # 计算p值
            if alternative == 'two-sided':
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            elif alternative == 'greater':
                p_value = 1 - stats.t.cdf(t_stat, df)
            elif alternative == 'less':
                p_value = stats.t.cdf(t_stat, df)
            else:
                raise ValueError("alternative参数必须是 'two-sided', 'greater', 或 'less'")
            
            # 效应量 (Cohen's d)
            pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std
            
            result = StatisticalResult(
                statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=int(df),
                effect_size=cohens_d,
                sample_size=n1 + n2,
                test_name="双样本t检验"
            )
            
            logger.info(f"双样本t检验完成: t = {t_stat:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"双样本t检验时出错: {e}")
            raise
    
    @staticmethod
    def chi_square_test(observed: np.ndarray, 
                       expected: Optional[np.ndarray] = None,
                       correction: bool = True) -> StatisticalResult:
        """
        卡方检验
        
        Args:
            observed: 观察频数矩阵
            expected: 期望频数矩阵 (如果为None，则计算独立性检验)
            correction: 是否使用Yates连续性修正
            
        Returns:
            统计检验结果
        """
        try:
            observed = np.array(observed)
            
            if expected is None:
                # 独立性检验
                chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(observed, correction=correction)
                test_name = "卡方独立性检验"
            else:
                # 拟合优度检验
                expected = np.array(expected)
                if observed.shape != expected.shape:
                    raise ValueError("观察频数和期望频数的形状必须相同")
                
                if correction and observed.size > 1:
                    # Yates修正
                    chi2_stat = np.sum((np.abs(observed - expected) - 0.5)**2 / expected)
                else:
                    chi2_stat = np.sum((observed - expected)**2 / expected)
                
                dof = observed.size - 1
                p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
                test_name = "卡方拟合优度检验"
            
            # 效应量 (Cramér's V)
            n = np.sum(observed)
            cramers_v = math.sqrt(chi2_stat / (n * (min(observed.shape) - 1)))
            
            result = StatisticalResult(
                statistic=chi2_stat,
                p_value=p_value,
                degrees_of_freedom=dof,
                effect_size=cramers_v,
                sample_size=int(n),
                test_name=test_name
            )
            
            logger.info(f"卡方检验完成: χ² = {chi2_stat:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"卡方检验时出错: {e}")
            raise
    
    @staticmethod
    def anova_one_way(groups: List[Union[List[float], np.ndarray]]) -> StatisticalResult:
        """
        单因素方差分析
        
        Args:
            groups: 各组数据列表
            
        Returns:
            统计检验结果
        """
        try:
            if len(groups) < 2:
                raise ValueError("至少需要两组数据")
            
            groups = [np.array(group) for group in groups]
            group_sizes = [len(group) for group in groups]
            total_n = sum(group_sizes)
            
            if any(size < 2 for size in group_sizes):
                raise ValueError("每组至少需要2个观测值")
            
            # 计算总体均值
            all_data = np.concatenate(groups)
            grand_mean = np.mean(all_data)
            
            # 计算组间平方和 (SSB)
            ssb = sum(n * (np.mean(group) - grand_mean)**2 for group, n in zip(groups, group_sizes))
            
            # 计算组内平方和 (SSW)
            ssw = sum(np.sum((group - np.mean(group))**2) for group in groups)
            
            # 计算自由度
            df_between = len(groups) - 1
            df_within = total_n - len(groups)
            df_total = total_n - 1
            
            # 计算均方
            msb = ssb / df_between
            msw = ssw / df_within
            
            # 计算F统计量
            f_stat = msb / msw
            
            # 计算p值
            p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)
            
            # 效应量 (η²)
            eta_squared = ssb / (ssb + ssw)
            
            result = StatisticalResult(
                statistic=f_stat,
                p_value=p_value,
                degrees_of_freedom=df_between,
                effect_size=eta_squared,
                sample_size=total_n,
                test_name="单因素方差分析"
            )
            
            logger.info(f"单因素方差分析完成: F = {f_stat:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"单因素方差分析时出错: {e}")
            raise
    
    @staticmethod
    def _calculate_aic(data: np.ndarray, pdf_func: Callable, params: Tuple) -> float:
        """计算AIC信息准则"""
        try:
            # 计算对数似然
            log_likelihood = np.sum(np.log(pdf_func(data, *params) + 1e-10))  # 添加小值避免log(0)
            # 计算参数数量
            n_params = len(params)
            # 计算AIC
            aic = 2 * n_params - 2 * log_likelihood
            return float(aic)
        except Exception as e:
            logger.warning(f"计算AIC时出错: {e}")
            return float('inf')
    
    @staticmethod
    def _calculate_bic(data: np.ndarray, pdf_func: Callable, params: Tuple) -> float:
        """计算BIC信息准则"""
        try:
            # 计算对数似然
            log_likelihood = np.sum(np.log(pdf_func(data, *params) + 1e-10))  # 添加小值避免log(0)
            # 计算参数数量和数据点数
            n_params = len(params)
            n = len(data)
            # 计算BIC
            bic = n_params * np.log(n) - 2 * log_likelihood
            return float(bic)
        except Exception as e:
            logger.warning(f"计算BIC时出错: {e}")
            return float('inf')

    @staticmethod
    def correlation_analysis(x: Union[List[float], np.ndarray],
                           y: Union[List[float], np.ndarray],
                           method: str = 'pearson',
                           alpha: float = 0.05) -> StatisticalResult:
        """
        相关性分析
        
        Args:
            x: 第一个变量数据
            y: 第二个变量数据
            method: 相关方法 ('pearson', 'spearman', 'kendall')
            alpha: 显著性水平
            
        Returns:
            统计检验结果
        """
        try:
            x = np.array(x)
            y = np.array(y)
            
            if len(x) != len(y):
                raise ValueError("两个变量的数据长度必须相同")
            
            if len(x) < 3:
                raise ValueError("至少需要3对数据点")
            
            if method == 'pearson':
                corr_coeff, p_value = stats.pearsonr(x, y)
                test_name = "Pearson相关分析"
            elif method == 'spearman':
                corr_coeff, p_value = stats.spearmanr(x, y)
                test_name = "Spearman相关分析"
            elif method == 'kendall':
                corr_coeff, p_value = stats.kendalltau(x, y)
                test_name = "Kendall相关分析"
            else:
                raise ValueError("method参数必须是 'pearson', 'spearman', 或 'kendall'")
            
            # 计算置信区间
            n = len(x)
            if method == 'pearson' and abs(corr_coeff) < 1:
                z_r = 0.5 * math.log((1 + corr_coeff) / (1 - corr_coeff))
                se = 1 / math.sqrt(n - 3)
                z_critical = stats.norm.ppf(1 - alpha/2)
                ci_lower = math.tanh(z_r - z_critical * se)
                ci_upper = math.tanh(z_r + z_critical * se)
                confidence_interval = (ci_lower, ci_upper)
            else:
                confidence_interval = None
            
            result = StatisticalResult(
                statistic=corr_coeff,
                p_value=p_value,
                confidence_interval=confidence_interval,
                sample_size=n,
                test_name=test_name
            )
            
            logger.info(f"相关性分析完成: r = {corr_coeff:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"相关性分析时出错: {e}")
            raise


class ProbabilityDistributions:
    """概率分布工具类"""
    
    @staticmethod
    def normal_distribution(mean: float = 0, std: float = 1) -> Dict[str, Callable]:
        """
        正态分布工具
        
        Args:
            mean: 均值
            std: 标准差
            
        Returns:
            包含概率密度函数、累积分布函数、分位数函数的字典
        """
        def pdf(x: float) -> float:
            """概率密度函数"""
            return stats.norm.pdf(x, loc=mean, scale=std)
        
        def cdf(x: float) -> float:
            """累积分布函数"""
            return stats.norm.cdf(x, loc=mean, scale=std)
        
        def quantile(p: float) -> float:
            """分位数函数"""
            return stats.norm.ppf(p, loc=mean, scale=std)
        
        def random_sample(size: int = 1) -> np.ndarray:
            """随机抽样"""
            return stats.norm.rvs(loc=mean, scale=std, size=size)
        
        return {
            'pdf': pdf,
            'cdf': cdf,
            'quantile': quantile,
            'random_sample': random_sample,
            'mean': mean,
            'std': std
        }
    
    @staticmethod
    def t_distribution(df: float) -> Dict[str, Callable]:
        """
        t分布工具
        
        Args:
            df: 自由度
            
        Returns:
            包含概率密度函数、累积分布函数、分位数函数的字典
        """
        def pdf(x: float) -> float:
            """概率密度函数"""
            return stats.t.pdf(x, df=df)
        
        def cdf(x: float) -> float:
            """累积分布函数"""
            return stats.t.cdf(x, df=df)
        
        def quantile(p: float) -> float:
            """分位数函数"""
            return stats.t.ppf(p, df=df)
        
        def random_sample(size: int = 1) -> np.ndarray:
            """随机抽样"""
            return stats.t.rvs(df=df, size=size)
        
        return {
            'pdf': pdf,
            'cdf': cdf,
            'quantile': quantile,
            'random_sample': random_sample,
            'df': df
        }
    
    @staticmethod
    def chi_square_distribution(df: float) -> Dict[str, Callable]:
        """
        卡方分布工具
        
        Args:
            df: 自由度
            
        Returns:
            包含概率密度函数、累积分布函数、分位数函数的字典
        """
        def pdf(x: float) -> float:
            """概率密度函数"""
            return stats.chi2.pdf(x, df=df)
        
        def cdf(x: float) -> float:
            """累积分布函数"""
            return stats.chi2.cdf(x, df=df)
        
        def quantile(p: float) -> float:
            """分位数函数"""
            return stats.chi2.ppf(p, df=df)
        
        def random_sample(size: int = 1) -> np.ndarray:
            """随机抽样"""
            return stats.chi2.rvs(df=df, size=size)
        
        return {
            'pdf': pdf,
            'cdf': cdf,
            'quantile': quantile,
            'random_sample': random_sample,
            'df': df
        }
    
    @staticmethod
    def f_distribution(df1: float, df2: float) -> Dict[str, Callable]:
        """
        F分布工具
        
        Args:
            df1: 分子自由度
            df2: 分母自由度
            
        Returns:
            包含概率密度函数、累积分布函数、分位数函数的字典
        """
        def pdf(x: float) -> float:
            """概率密度函数"""
            return stats.f.pdf(x, dfn=df1, dfd=df2)
        
        def cdf(x: float) -> float:
            """累积分布函数"""
            return stats.f.cdf(x, dfn=df1, dfd=df2)
        
        def quantile(p: float) -> float:
            """分位数函数"""
            return stats.f.ppf(p, dfn=df1, dfd=df2)
        
        def random_sample(size: int = 1) -> np.ndarray:
            """随机抽样"""
            return stats.f.rvs(dfn=df1, dfd=df2, size=size)
        
        return {
            'pdf': pdf,
            'cdf': cdf,
            'quantile': quantile,
            'random_sample': random_sample,
            'df1': df1,
            'df2': df2
        }
    
    @staticmethod
    def fit_distribution(data: Union[List[float], np.ndarray],
                        distribution: str = 'normal') -> Dict[str, float]:
        """
        拟合分布参数
        
        Args:
            data: 数据
            distribution: 分布类型
            
        Returns:
            拟合参数字典
        """
        try:
            data = np.array(data)
            
            if distribution == 'normal':
                params = stats.norm.fit(data)
                return {
                    'distribution': 'normal',
                    'loc': params[0],  # 均值
                    'scale': params[1],  # 标准差
                    'aic': InferentialStatistics._calculate_aic(data, stats.norm.pdf, params),
                    'bic': InferentialStatistics._calculate_bic(data, stats.norm.pdf, params)
                }
            
            elif distribution == 't':
                params = stats.t.fit(data)
                return {
                    'distribution': 't',
                    'df': params[0],  # 自由度
                    'loc': params[1],  # 位置参数
                    'scale': params[2],  # 尺度参数
                    'aic': InferentialStatistics._calculate_aic(data, stats.t.pdf, params),
                    'bic': InferentialStatistics._calculate_bic(data, stats.t.pdf, params)
                }
            
            elif distribution == 'chi_square':
                params = stats.chi2.fit(data)
                return {
                    'distribution': 'chi_square',
                    'df': params[0],  # 自由度
                    'loc': params[1],  # 位置参数
                    'scale': params[2],  # 尺度参数
                    'aic': InferentialStatistics._calculate_aic(data, stats.chi2.pdf, params),
                    'bic': InferentialStatistics._calculate_bic(data, stats.chi2.pdf, params)
                }
            
            elif distribution == 'gamma':
                params = stats.gamma.fit(data)
                return {
                    'distribution': 'gamma',
                    'a': params[0],  # 形状参数
                    'loc': params[1],  # 位置参数
                    'scale': params[2],  # 尺度参数
                    'aic': InferentialStatistics._calculate_aic(data, stats.gamma.pdf, params),
                    'bic': InferentialStatistics._calculate_bic(data, stats.gamma.pdf, params)
                }
            
            else:
                raise ValueError(f"不支持的分布类型: {distribution}")
                
        except Exception as e:
            logger.error(f"拟合分布时出错: {e}")
            raise
    
    @staticmethod
    def goodness_of_fit_test(data: Union[List[float], np.ndarray],
                           distribution: str = 'normal',
                           alpha: float = 0.05) -> StatisticalResult:
        """
        拟合优度检验
        
        Args:
            data: 数据
            distribution: 分布类型
            alpha: 显著性水平
            
        Returns:
            统计检验结果
        """
        try:
            data = np.array(data)
            
            if distribution == 'normal':
                # Shapiro-Wilk检验 (正态性检验)
                if len(data) >= 3 and len(data) <= 5000:
                    stat, p_value = stats.shapiro(data)
                    test_name = "Shapiro-Wilk正态性检验"
                else:
                    # Kolmogorov-Smirnov检验
                    params = stats.norm.fit(data)
                    stat, p_value = stats.kstest(data, lambda x: stats.norm.cdf(x, *params))
                    test_name = "Kolmogorov-Smirnov正态性检验"
            
            elif distribution == 'exponential':
                # 指数分布检验
                params = stats.expon.fit(data)
                stat, p_value = stats.kstest(data, lambda x: stats.expon.cdf(x, *params))
                test_name = "指数分布拟合优度检验"
            
            elif distribution == 'uniform':
                # 均匀分布检验
                params = stats.uniform.fit(data)
                stat, p_value = stats.kstest(data, lambda x: stats.uniform.cdf(x, *params))
                test_name = "均匀分布拟合优度检验"
            
            else:
                # 通用K-S检验
                if distribution == 'chi_square':
                    params = stats.chi2.fit(data)
                    cdf_func = lambda x: stats.chi2.cdf(x, *params)
                elif distribution == 'gamma':
                    params = stats.gamma.fit(data)
                    cdf_func = lambda x: stats.gamma.cdf(x, *params)
                else:
                    raise ValueError(f"不支持的分布类型: {distribution}")
                
                stat, p_value = stats.kstest(data, cdf_func)
                test_name = f"{distribution}分布拟合优度检验"
            
            result = StatisticalResult(
                statistic=stat,
                p_value=p_value,
                sample_size=len(data),
                test_name=test_name
            )
            
            logger.info(f"拟合优度检验完成: {test_name}, 统计量 = {stat:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"拟合优度检验时出错: {e}")
            raise


class BayesianStatistics:
    """贝叶斯统计工具类"""
    
    def __init__(self):
        self.priors = {}
        self.posteriors = {}
    
    def set_prior(self, parameter: str, distribution: str, **params):
        """
        设置先验分布
        
        Args:
            parameter: 参数名称
            distribution: 分布类型
            **params: 分布参数
        """
        try:
            if distribution == 'normal':
                # 正态先验 N(μ₀, σ₀²)
                self.priors[parameter] = {
                    'type': 'normal',
                    'mean': params.get('mean', 0),
                    'std': params.get('std', 1)
                }
            elif distribution == 'beta':
                # Beta先验 Beta(α, β)
                self.priors[parameter] = {
                    'type': 'beta',
                    'alpha': params.get('alpha', 1),
                    'beta': params.get('beta', 1)
                }
            elif distribution == 'gamma':
                # Gamma先验 Gamma(α, β)
                self.priors[parameter] = {
                    'type': 'gamma',
                    'alpha': params.get('alpha', 1),
                    'beta': params.get('beta', 1)
                }
            elif distribution == 'uniform':
                # 均匀先验 U(a, b)
                self.priors[parameter] = {
                    'type': 'uniform',
                    'low': params.get('low', 0),
                    'high': params.get('high', 1)
                }
            else:
                raise ValueError(f"不支持的先验分布类型: {distribution}")
            
            logger.info(f"设置参数 {parameter} 的先验分布: {distribution}")
            
        except Exception as e:
            logger.error(f"设置先验分布时出错: {e}")
            raise
    
    def bayesian_normal_mean(self, data: Union[List[float], np.ndarray],
                           prior_mean: float = 0,
                           prior_std: float = 1,
                           likelihood_std: Optional[float] = None) -> Dict[str, float]:
        """
        正态分布均值的贝叶斯推断
        
        Args:
            data: 观测数据
            prior_mean: 先验均值
            prior_std: 先验标准差
            likelihood_std: 似然标准差 (如果为None，使用样本标准差)
            
        Returns:
            后验分布参数
        """
        try:
            data = np.array(data)
            n = len(data)
            sample_mean = np.mean(data)
            
            if likelihood_std is None:
                likelihood_std = np.std(data, ddof=1)
            
            # 计算后验参数
            prior_precision = 1 / (prior_std ** 2)
            likelihood_precision = 1 / (likelihood_std ** 2)
            
            # 后验精度
            posterior_precision = prior_precision + n * likelihood_precision
            
            # 后验均值
            posterior_mean = (prior_precision * prior_mean + n * likelihood_precision * sample_mean) / posterior_precision
            
            # 后验标准差
            posterior_std = 1 / math.sqrt(posterior_precision)
            
            result = {
                'prior_mean': prior_mean,
                'prior_std': prior_std,
                'sample_mean': sample_mean,
                'sample_size': n,
                'posterior_mean': posterior_mean,
                'posterior_std': posterior_std,
                'posterior_precision': posterior_precision
            }
            
            logger.info(f"贝叶斯正态均值推断完成: 后验均值 = {posterior_mean:.4f}, 后验标准差 = {posterior_std:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"贝叶斯正态均值推断时出错: {e}")
            raise
    
    def bayesian_normal_variance(self, data: Union[List[float], np.ndarray],
                               prior_alpha: float = 1,
                               prior_beta: float = 1) -> Dict[str, float]:
        """
        正态分布方差的贝叶斯推断
        
        Args:
            data: 观测数据
            prior_alpha: 先验Gamma分布的形状参数
            prior_beta: 先验Gamma分布的尺度参数
            
        Returns:
            后验分布参数
        """
        try:
            data = np.array(data)
            n = len(data)
            sample_var = np.var(data, ddof=1)
            
            # 后验参数
            posterior_alpha = prior_alpha + n / 2
            posterior_beta = prior_beta + n * sample_var / 2
            
            # 后验分布为Gamma(α + n/2, β + n*s²/2)
            result = {
                'prior_alpha': prior_alpha,
                'prior_beta': prior_beta,
                'sample_variance': sample_var,
                'sample_size': n,
                'posterior_alpha': posterior_alpha,
                'posterior_beta': posterior_beta,
                'posterior_mean_variance': posterior_beta / (posterior_alpha - 1) if posterior_alpha > 1 else float('inf'),
                'posterior_mode_variance': posterior_beta / (posterior_alpha + 1)
            }
            
            logger.info(f"贝叶斯正态方差推断完成: 后验α = {posterior_alpha:.4f}, 后验β = {posterior_beta:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"贝叶斯正态方差推断时出错: {e}")
            raise
    
    def bayesian_proportion(self, data: Union[List[int], np.ndarray],
                          prior_alpha: float = 1,
                          prior_beta: float = 1) -> Dict[str, float]:
        """
        比例参数的贝叶斯推断
        
        Args:
            data: 二项数据 (0或1)
            prior_alpha: 先验Beta分布的α参数
            prior_beta: 先验Beta分布的β参数
            
        Returns:
            后验分布参数
        """
        try:
            data = np.array(data)
            n = len(data)
            successes = np.sum(data)
            failures = n - successes
            
            # 后验参数
            posterior_alpha = prior_alpha + successes
            posterior_beta = prior_beta + failures
            
            # 后验分布为Beta(α + x, β + n - x)
            posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
            posterior_mode = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2) if posterior_alpha > 1 and posterior_beta > 1 else None
            
            result = {
                'prior_alpha': prior_alpha,
                'prior_beta': prior_beta,
                'successes': successes,
                'failures': failures,
                'sample_size': n,
                'posterior_alpha': posterior_alpha,
                'posterior_beta': posterior_beta,
                'posterior_mean': posterior_mean,
                'posterior_mode': posterior_mode,
                'credible_interval_95': self._beta_credible_interval(posterior_alpha, posterior_beta, 0.95)
            }
            
            logger.info(f"贝叶斯比例推断完成: 后验均值 = {posterior_mean:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"贝叶斯比例推断时出错: {e}")
            raise
    
    def model_comparison_bayes_factor(self, data: Union[List[float], np.ndarray],
                                    model1_params: Dict[str, Any],
                                    model2_params: Dict[str, Any]) -> Dict[str, float]:
        """
        模型比较的贝叶斯因子计算
        
        Args:
            data: 观测数据
            model1_params: 模型1的参数
            model2_params: 模型2的参数
            
        Returns:
            贝叶斯因子和相关统计量
        """
        try:
            data = np.array(data)
            
            # 计算边际似然 (使用拉普拉斯近似)
            log_marginal1 = self._compute_log_marginal_likelihood(data, model1_params)
            log_marginal2 = self._compute_log_marginal_likelihood(data, model2_params)
            
            # 贝叶斯因子
            log_bayes_factor = log_marginal1 - log_marginal2
            bayes_factor = math.exp(log_bayes_factor)
            
            # 解释贝叶斯因子
            if bayes_factor > 100:
                interpretation = "模型1有极强的证据支持"
            elif bayes_factor > 30:
                interpretation = "模型1有非常强的证据支持"
            elif bayes_factor > 10:
                interpretation = "模型1有强证据支持"
            elif bayes_factor > 3:
                interpretation = "模型1有中等证据支持"
            elif bayes_factor > 1/3:
                interpretation = "证据不足"
            elif bayes_factor > 1/10:
                interpretation = "模型2有中等证据支持"
            elif bayes_factor > 1/30:
                interpretation = "模型2有强证据支持"
            elif bayes_factor > 1/100:
                interpretation = "模型2有非常强的证据支持"
            else:
                interpretation = "模型2有极强的证据支持"
            
            result = {
                'log_marginal_likelihood_model1': log_marginal1,
                'log_marginal_likelihood_model2': log_marginal2,
                'bayes_factor': bayes_factor,
                'log_bayes_factor': log_bayes_factor,
                'interpretation': interpretation
            }
            
            logger.info(f"贝叶斯模型比较完成: 贝叶斯因子 = {bayes_factor:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"贝叶斯模型比较时出错: {e}")
            raise
    
    def _beta_credible_interval(self, alpha: float, beta: float, confidence: float) -> Tuple[float, float]:
        """计算Beta分布的可信区间"""
        alpha_level = (1 - confidence) / 2
        lower = stats.beta.ppf(alpha_level, alpha, beta)
        upper = stats.beta.ppf(1 - alpha_level, alpha, beta)
        return (lower, upper)
    
    def _compute_log_marginal_likelihood(self, data: np.ndarray, params: Dict[str, Any]) -> float:
        """计算边际对数似然 (简化版本)"""
        # 这是一个简化的实现，实际应用中可能需要更复杂的数值方法
        try:
            if 'normal' in params.get('model_type', ''):
                # 正态模型的边际似然 (共轭先验)
                n = len(data)
                sample_mean = np.mean(data)
                sample_var = np.var(data, ddof=1)
                
                prior_mean = params.get('prior_mean', 0)
                prior_std = params.get('prior_std', 1)
                likelihood_std = params.get('likelihood_std', math.sqrt(sample_var))
                
                # 简化的边际似然计算
                log_likelihood = -n/2 * math.log(2 * math.pi * likelihood_std**2) - n * sample_var / (2 * likelihood_std**2)
                log_prior = -0.5 * math.log(2 * math.pi * prior_std**2) - (sample_mean - prior_mean)**2 / (2 * prior_std**2)
                
                return log_likelihood + log_prior
            else:
                # 其他模型的简化实现
                return -len(data)  # 占位符
                
        except Exception as e:
            logger.warning(f"计算边际似然时出错，使用近似值: {e}")
            return -len(data)


class NonParametricStatistics:
    """非参数统计工具类"""
    
    @staticmethod
    def mann_whitney_test(data1: Union[List[float], np.ndarray],
                         data2: Union[List[float], np.ndarray],
                         alternative: str = 'two-sided') -> StatisticalResult:
        """
        Mann-Whitney U检验 (Wilcoxon秩和检验)
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            alternative: 备择假设 ('two-sided', 'greater', 'less')
            
        Returns:
            统计检验结果
        """
        try:
            data1 = np.array(data1)
            data2 = np.array(data2)
            
            if len(data1) == 0 or len(data2) == 0:
                raise ValueError("两组数据都不能为空")
            
            # 执行Mann-Whitney U检验
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
            
            # 计算效应量 (r = Z / sqrt(N))
            n1, n2 = len(data1), len(data2)
            z_score = stats.norm.ppf(1 - p_value/2) if alternative == 'two-sided' else stats.norm.ppf(1 - p_value)
            effect_size = abs(z_score) / math.sqrt(n1 + n2)
            
            result = StatisticalResult(
                statistic=statistic,
                p_value=p_value,
                sample_size=n1 + n2,
                effect_size=effect_size,
                test_name="Mann-Whitney U检验"
            )
            
            logger.info(f"Mann-Whitney U检验完成: U = {statistic:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Mann-Whitney U检验时出错: {e}")
            raise
    
    @staticmethod
    def wilcoxon_signed_rank_test(data1: Union[List[float], np.ndarray],
                                 data2: Union[List[float], np.ndarray],
                                 alternative: str = 'two-sided') -> StatisticalResult:
        """
        Wilcoxon符号秩检验
        
        Args:
            data1: 第一组数据 (配对数据)
            data2: 第二组数据 (配对数据)
            alternative: 备择假设 ('two-sided', 'greater', 'less')
            
        Returns:
            统计检验结果
        """
        try:
            data1 = np.array(data1)
            data2 = np.array(data2)
            
            if len(data1) != len(data2):
                raise ValueError("两组配对数据的长度必须相同")
            
            if len(data1) == 0:
                raise ValueError("数据不能为空")
            
            # 执行Wilcoxon符号秩检验
            statistic, p_value = stats.wilcoxon(data1, data2, alternative=alternative, zero_method='wilcox')
            
            # 计算效应量 (r = Z / sqrt(N))
            n = len(data1)
            z_score = stats.norm.ppf(1 - p_value/2) if alternative == 'two-sided' else stats.norm.ppf(1 - p_value)
            effect_size = abs(z_score) / math.sqrt(n)
            
            result = StatisticalResult(
                statistic=statistic,
                p_value=p_value,
                sample_size=n,
                effect_size=effect_size,
                test_name="Wilcoxon符号秩检验"
            )
            
            logger.info(f"Wilcoxon符号秩检验完成: W = {statistic:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Wilcoxon符号秩检验时出错: {e}")
            raise
    
    @staticmethod
    def kolmogorov_smirnov_test(data1: Union[List[float], np.ndarray],
                               data2: Union[List[float], np.ndarray],
                               alternative: str = 'two-sided') -> StatisticalResult:
        """
        Kolmogorov-Smirnov检验
        
        Args:
            data1: 第一组数据
            data2: 第二组数据或分布名称 ('norm', 'uniform', 'expon'等)
            alternative: 备择假设 ('two-sided', 'less', 'greater')
            
        Returns:
            统计检验结果
        """
        try:
            data1 = np.array(data1)
            
            if isinstance(data2, (list, np.ndarray)):
                # 两样本KS检验
                data2 = np.array(data2)
                statistic, p_value = stats.ks_2samp(data1, data2, alternative=alternative)
                test_name = "两样本Kolmogorov-Smirnov检验"
                sample_size = len(data1) + len(data2)
            else:
                # 单样本KS检验
                if data2 == 'norm':
                    # 与正态分布比较
                    params = stats.norm.fit(data1)
                    statistic, p_value = stats.kstest(data1, lambda x: stats.norm.cdf(x, *params))
                elif data2 == 'uniform':
                    # 与均匀分布比较
                    params = stats.uniform.fit(data1)
                    statistic, p_value = stats.kstest(data1, lambda x: stats.uniform.cdf(x, *params))
                elif data2 == 'expon':
                    # 与指数分布比较
                    params = stats.expon.fit(data1)
                    statistic, p_value = stats.kstest(data1, lambda x: stats.expon.cdf(x, *params))
                else:
                    raise ValueError(f"不支持的分布: {data2}")
                
                test_name = f"单样本Kolmogorov-Smirnov检验 ({data2})"
                sample_size = len(data1)
            
            result = StatisticalResult(
                statistic=statistic,
                p_value=p_value,
                sample_size=sample_size,
                test_name=test_name
            )
            
            logger.info(f"Kolmogorov-Smirnov检验完成: D = {statistic:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Kolmogorov-Smirnov检验时出错: {e}")
            raise
    
    @staticmethod
    def kruskal_wallis_test(groups: List[Union[List[float], np.ndarray]]) -> StatisticalResult:
        """
        Kruskal-Wallis H检验 (非参数单因素方差分析)
        
        Args:
            groups: 各组数据列表
            
        Returns:
            统计检验结果
        """
        try:
            if len(groups) < 2:
                raise ValueError("至少需要两组数据")
            
            groups = [np.array(group) for group in groups]
            
            # 执行Kruskal-Wallis检验
            statistic, p_value = stats.kruskal(*groups)
            
            # 计算效应量 (η²)
            total_n = sum(len(group) for group in groups)
            k = len(groups)
            effect_size = (statistic - k + 1) / (total_n - k)
            
            result = StatisticalResult(
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=k - 1,
                effect_size=effect_size,
                sample_size=total_n,
                test_name="Kruskal-Wallis H检验"
            )
            
            logger.info(f"Kruskal-Wallis H检验完成: H = {statistic:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Kruskal-Wallis H检验时出错: {e}")
            raise
    
    @staticmethod
    def friedman_test(groups: List[Union[List[float], np.ndarray]]) -> StatisticalResult:
        """
        Friedman检验 (非参数重复测量方差分析)
        
        Args:
            groups: 各组数据列表 (每组应该包含相同数量的观测值)
            
        Returns:
            统计检验结果
        """
        try:
            if len(groups) < 3:
                raise ValueError("至少需要三组数据")
            
            groups = [np.array(group) for group in groups]
            
            # 检查每组数据长度是否相同
            group_sizes = [len(group) for group in groups]
            if len(set(group_sizes)) > 1:
                raise ValueError("所有组的数据长度必须相同")
            
            # 执行Friedman检验
            statistic, p_value = stats.friedmanchisquare(*groups)
            
            # 计算效应量 (Kendall's W)
            k = len(groups)
            n = group_sizes[0]
            effect_size = statistic / (k * (k - 1) * n)
            
            result = StatisticalResult(
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=k - 1,
                effect_size=effect_size,
                sample_size=k * n,
                test_name="Friedman检验"
            )
            
            logger.info(f"Friedman检验完成: χ² = {statistic:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Friedman检验时出错: {e}")
            raise
    
    @staticmethod
    def spearman_correlation(x: Union[List[float], np.ndarray],
                           y: Union[List[float], np.ndarray],
                           alpha: float = 0.05) -> StatisticalResult:
        """
        Spearman等级相关分析
        
        Args:
            x: 第一个变量数据
            y: 第二个变量数据
            alpha: 显著性水平
            
        Returns:
            统计检验结果
        """
        try:
            correlation_coeff, p_value = NonParametricStatistics.spearman_correlation_raw(x, y)
            
            # 计算置信区间 (近似)
            n = len(x)
            if abs(correlation_coeff) < 1:
                z_r = 0.5 * math.log((1 + correlation_coeff) / (1 - correlation_coeff))
                se = 1 / math.sqrt(n - 3)
                z_critical = stats.norm.ppf(1 - alpha/2)
                ci_lower = math.tanh(z_r - z_critical * se)
                ci_upper = math.tanh(z_r + z_critical * se)
                confidence_interval = (ci_lower, ci_upper)
            else:
                confidence_interval = None
            
            result = StatisticalResult(
                statistic=correlation_coeff,
                p_value=p_value,
                confidence_interval=confidence_interval,
                sample_size=n,
                test_name="Spearman等级相关"
            )
            
            logger.info(f"Spearman相关分析完成: ρ = {correlation_coeff:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Spearman相关分析时出错: {e}")
            raise
    
    @staticmethod
    def spearman_correlation_raw(x: Union[List[float], np.ndarray],
                               y: Union[List[float], np.ndarray]) -> Tuple[float, float]:
        """Spearman相关系数和p值的原始计算"""
        x = np.array(x)
        y = np.array(y)
        
        if len(x) != len(y):
            raise ValueError("两个变量的数据长度必须相同")
        
        # 计算Spearman相关系数
        correlation_coeff, p_value = stats.spearmanr(x, y)
        
        return correlation_coeff, p_value


class MultivariateStatistics:
    """多变量统计分析工具类"""
    
    @staticmethod
    def principal_component_analysis(data: np.ndarray,
                                   n_components: Optional[int] = None,
                                   standardize: bool = True) -> Dict[str, Any]:
        """
        主成分分析
        
        Args:
            data: 数据矩阵 (n_samples × n_features)
            n_components: 主成分数量 (如果为None，保留所有成分)
            standardize: 是否标准化数据
            
        Returns:
            包含主成分分析结果的字典
        """
        try:
            data = np.array(data)
            
            if data.ndim != 2:
                raise ValueError("数据必须是二维矩阵")
            
            n_samples, n_features = data.shape
            
            # 标准化数据
            if standardize:
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0, ddof=1)
                # 避免除零
                std[std == 0] = 1
                data_standardized = (data - mean) / std
            else:
                data_standardized = data
                mean = None
                std = None
            
            # 计算协方差矩阵
            cov_matrix = np.cov(data_standardized.T)
            
            # 特征值分解
            eigenvalues, eigenvectors = eig(cov_matrix)
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            
            # 排序 (从大到小)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # 计算解释方差比例
            total_variance = np.sum(eigenvalues)
            explained_variance_ratio = eigenvalues / total_variance
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            # 确定主成分数量
            if n_components is None:
                # 保留解释95%方差的主成分
                n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
            
            # 计算主成分得分
            components = eigenvectors[:, :n_components]
            scores = np.dot(data_standardized, components)
            
            # 计算重构数据
            reconstructed = np.dot(scores, components.T)
            if standardize:
                reconstructed = reconstructed * std + mean
            
            # 计算重构误差
            mse = np.mean((data - reconstructed) ** 2)
            
            result = {
                'components': components,
                'eigenvalues': eigenvalues[:n_components],
                'explained_variance_ratio': explained_variance_ratio[:n_components],
                'cumulative_variance_ratio': cumulative_variance_ratio[:n_components],
                'scores': scores,
                'mean': mean,
                'std': std,
                'reconstructed_data': reconstructed,
                'reconstruction_error': mse,
                'n_components': n_components,
                'total_variance': total_variance
            }
            
            logger.info(f"主成分分析完成: 保留{n_components}个主成分，解释方差比例 = {cumulative_variance_ratio[n_components-1]:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"主成分分析时出错: {e}")
            raise
    
    @staticmethod
    def factor_analysis(data: np.ndarray,
                       n_factors: int,
                       rotation: str = 'varimax') -> Dict[str, Any]:
        """
        因子分析
        
        Args:
            data: 数据矩阵
            n_factors: 因子数量
            rotation: 旋转方法 ('varimax', 'promax', 'quartimax')
            
        Returns:
            包含因子分析结果的字典
        """
        try:
            data = np.array(data)
            n_samples, n_variables = data.shape
            
            # 标准化数据
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0, ddof=1)
            std[std == 0] = 1
            data_standardized = (data - mean) / std
            
            # 初始因子载荷矩阵 (随机初始化)
            np.random.seed(42)
            loadings = np.random.randn(n_variables, n_factors)
            
            # 迭代优化 (简化版最大似然估计)
            max_iterations = 100
            tolerance = 1e-6
            
            for iteration in range(max_iterations):
                # 计算因子得分
                factor_scores = np.dot(data_standardized, loadings)
                
                # 更新载荷矩阵
                covariance = np.cov(data_standardized.T)
                loadings_new = np.linalg.solve(
                    np.eye(n_factors) + np.dot(loadings.T, loadings),
                    np.dot(loadings.T, covariance)
                ).T
                
                # 检查收敛
                if np.max(np.abs(loadings - loadings_new)) < tolerance:
                    loadings = loadings_new
                    break
                
                loadings = loadings_new
            
            # 计算特殊方差
            communalities = np.sum(loadings**2, axis=1)
            specific_variances = np.diag(covariance) - communalities
            specific_variances = np.maximum(specific_variances, 1e-6)  # 确保为正
            
            # 旋转载荷矩阵
            if rotation == 'varimax':
                loadings = MultivariateStatistics._varimax_rotation(loadings)
            elif rotation == 'promax':
                loadings = MultivariateStatistics._promax_rotation(loadings)
            
            # 计算因子得分
            factor_scores = np.dot(data_standardized, loadings)
            
            # 计算模型拟合指标
            model_covariance = np.dot(loadings, loadings.T) + np.diag(specific_variances)
            fit_statistic = np.sum((covariance - model_covariance)**2)
            
            result = {
                'loadings': loadings,
                'factor_scores': factor_scores,
                'communalities': communalities,
                'specific_variances': specific_variances,
                'rotation': rotation,
                'n_factors': n_factors,
                'fit_statistic': fit_statistic,
                'mean': mean,
                'std': std
            }
            
            logger.info(f"因子分析完成: 提取{n_factors}个因子，拟合统计量 = {fit_statistic:.6f}")
            return result
            
        except Exception as e:
            logger.error(f"因子分析时出错: {e}")
            raise
    
    @staticmethod
    def discriminant_analysis(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        判别分析
        
        Args:
            X: 特征矩阵
            y: 类别标签
            
        Returns:
            包含判别分析结果的字典
        """
        try:
            X = np.array(X)
            y = np.array(y)
            
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            n_features = X.shape[1]
            
            # 标准化特征
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0, ddof=1)
            std[std == 0] = 1
            X_standardized = (X - mean) / std
            
            # 计算类内和类间散布矩阵
            overall_mean = np.mean(X_standardized, axis=0)
            
            # 类内散布矩阵
            S_w = np.zeros((n_features, n_features))
            # 类间散布矩阵
            S_b = np.zeros((n_features, n_features))
            
            class_means = {}
            class_counts = {}
            
            for class_label in unique_classes:
                class_mask = (y == class_label)
                class_data = X_standardized[class_mask]
                class_mean = np.mean(class_data, axis=0)
                class_count = len(class_data)
                
                class_means[class_label] = class_mean
                class_counts[class_label] = class_count
                
                # 类内散布矩阵
                centered_data = class_data - class_mean
                S_w += np.dot(centered_data.T, centered_data)
                
                # 类间散布矩阵
                mean_diff = class_mean - overall_mean
                S_b += class_count * np.outer(mean_diff, mean_diff)
            
            # 计算判别函数
            if n_classes > 2:
                # 多类情况：计算特征值和特征向量
                S_w_inv = np.linalg.pinv(S_w)
                matrix_to_eigen = np.dot(S_w_inv, S_b)
                eigenvalues, eigenvectors = eig(matrix_to_eigen)
                
                # 排序
                idx = np.argsort(np.real(eigenvalues))[::-1]
                eigenvalues = np.real(eigenvalues[idx])
                eigenvectors = np.real(eigenvectors[:, idx])
                
                # 保留前n_classes-1个判别函数
                n_discriminant = n_classes - 1
                discriminant_functions = eigenvectors[:, :n_discriminant]
                
            else:
                # 两类情况
                class1_mean = class_means[unique_classes[0]]
                class2_mean = class_means[unique_classes[1]]
                discriminant_functions = np.array([class2_mean - class1_mean]).T
                eigenvalues = None
            
            # 计算判别得分
            discriminant_scores = np.dot(X_standardized, discriminant_functions)
            
            # 计算分类准确率
            predicted_classes = []
            for i, score in enumerate(discriminant_scores):
                if n_classes > 2:
                    # 多类情况：选择得分最高的类
                    predicted_class = unique_classes[np.argmax(score)]
                else:
                    # 两类情况：基于阈值分类
                    threshold = 0
                    predicted_class = unique_classes[0] if score[0] > threshold else unique_classes[1]
                predicted_classes.append(predicted_class)
            
            accuracy = np.mean(np.array(predicted_classes) == y)
            
            result = {
                'discriminant_functions': discriminant_functions,
                'discriminant_scores': discriminant_scores,
                'class_means': class_means,
                'class_counts': class_counts,
                'eigenvalues': eigenvalues,
                'accuracy': accuracy,
                'predicted_classes': np.array(predicted_classes),
                'mean': mean,
                'std': std,
                'n_classes': n_classes
            }
            
            logger.info(f"判别分析完成: 准确率 = {accuracy:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"判别分析时出错: {e}")
            raise
    
    @staticmethod
    def cluster_analysis(X: np.ndarray,
                        n_clusters: int,
                        method: str = 'kmeans',
                        random_state: int = 42) -> Dict[str, Any]:
        """
        聚类分析
        
        Args:
            X: 特征矩阵
            n_clusters: 聚类数量
            method: 聚类方法 ('kmeans', 'hierarchical')
            random_state: 随机种子
            
        Returns:
            包含聚类分析结果的字典
        """
        try:
            X = np.array(X)
            
            if method == 'kmeans':
                result = MultivariateStatistics._kmeans_clustering(X, n_clusters, random_state)
            elif method == 'hierarchical':
                result = MultivariateStatistics._hierarchical_clustering(X, n_clusters)
            else:
                raise ValueError(f"不支持的聚类方法: {method}")
            
            logger.info(f"{method}聚类分析完成: {n_clusters}个聚类，轮廓系数 = {result['silhouette_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"聚类分析时出错: {e}")
            raise
    
    @staticmethod
    def canonical_correlation_analysis(X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """
        典型相关分析
        
        Args:
            X: 第一组变量矩阵
            Y: 第二组变量矩阵
            
        Returns:
            包含典型相关分析结果的字典
        """
        try:
            X = np.array(X)
            Y = np.array(Y)
            
            if X.shape[0] != Y.shape[0]:
                raise ValueError("两组变量的样本数量必须相同")
            
            n_samples = X.shape[0]
            
            # 标准化数据
            X_mean = np.mean(X, axis=0)
            Y_mean = np.mean(Y, axis=0)
            X_std = np.std(X, axis=0, ddof=1)
            Y_std = np.std(Y, axis=0, ddof=1)
            
            X_std[X_std == 0] = 1
            Y_std[Y_std == 0] = 1
            
            X_standardized = (X - X_mean) / X_std
            Y_standardized = (Y - Y_mean) / Y_std
            
            # 计算协方差矩阵
            S_xx = np.cov(X_standardized.T)
            S_yy = np.cov(Y_standardized.T)
            S_xy = np.cov(X_standardized.T, Y_standardized.T)[:X.shape[1], X.shape[1]:]
            S_yx = S_xy.T
            
            # 计算典型相关
            try:
                # 使用SVD求解典型相关
                U, s, Vt = svd(np.dot(np.linalg.pinv(np.sqrt(S_xx)), np.dot(S_xy, np.linalg.pinv(np.sqrt(S_yy)))))
                
                canonical_correlations = s
                n_components = len(s)
                
                # 计算典型向量
                canonical_weights_x = np.dot(np.linalg.pinv(np.sqrt(S_xx)), U)
                canonical_weights_y = np.dot(np.linalg.pinv(np.sqrt(S_yy)), Vt.T)
                
                # 计算典型得分
                canonical_scores_x = np.dot(X_standardized, canonical_weights_x)
                canonical_scores_y = np.dot(Y_standardized, canonical_weights_y)
                
                result = {
                    'canonical_correlations': canonical_correlations,
                    'canonical_weights_x': canonical_weights_x,
                    'canonical_weights_y': canonical_weights_y,
                    'canonical_scores_x': canonical_scores_x,
                    'canonical_scores_y': canonical_scores_y,
                    'n_components': n_components,
                    'explained_variance': canonical_correlations**2,
                    'cumulative_variance': np.cumsum(canonical_correlations**2)
                }
                
                logger.info(f"典型相关分析完成: {n_components}对典型变量")
                return result
                
            except np.linalg.LinAlgError:
                # 如果SVD失败，使用特征值分解方法
                matrix_to_eigen = np.dot(np.linalg.pinv(S_xx), np.dot(S_xy, np.dot(np.linalg.pinv(S_yy), S_yx)))
                eigenvalues, eigenvectors = eig(matrix_to_eigen)
                
                # 排序
                idx = np.argsort(np.real(eigenvalues))[::-1]
                eigenvalues = np.real(eigenvalues[idx])
                eigenvectors = np.real(eigenvectors[:, idx])
                
                canonical_correlations = np.sqrt(np.maximum(eigenvalues, 0))
                n_components = len(canonical_correlations)
                
                result = {
                    'canonical_correlations': canonical_correlations,
                    'eigenvalues': eigenvalues,
                    'n_components': n_components,
                    'explained_variance': canonical_correlations**2,
                    'cumulative_variance': np.cumsum(canonical_correlations**2)
                }
                
                logger.info(f"典型相关分析完成: {n_components}对典型变量")
                return result
            
        except Exception as e:
            logger.error(f"典型相关分析时出错: {e}")
            raise
    
    @staticmethod
    def _varimax_rotation(loadings: np.ndarray) -> np.ndarray:
        """Varimax旋转"""
        # 简化的Varimax旋转实现
        n_variables, n_factors = loadings.shape
        rotated_loadings = loadings.copy()
        
        for i in range(n_factors):
            # 提取当前因子
            factor_loadings = rotated_loadings[:, i]
            
            # 计算权重
            weights = factor_loadings**2 - np.mean(factor_loadings**2)
            
            # 更新载荷
            rotated_loadings[:, i] = factor_loadings + 0.1 * weights
        
        return rotated_loadings
    
    @staticmethod
    def _promax_rotation(loadings: np.ndarray) -> np.ndarray:
        """Promax旋转 (简化版)"""
        # 简化的Promax旋转实现
        n_variables, n_factors = loadings.shape
        rotated_loadings = loadings.copy()
        
        # 简化的倾斜旋转
        for i in range(n_factors):
            factor_loadings = rotated_loadings[:, i]
            # 简单的非线性变换
            rotated_loadings[:, i] = np.sign(factor_loadings) * np.abs(factor_loadings)**0.8
        
        return rotated_loadings
    
    @staticmethod
    def _kmeans_clustering(X: np.ndarray, n_clusters: int, random_state: int) -> Dict[str, Any]:
        """K-means聚类"""
        np.random.seed(random_state)
        n_samples, n_features = X.shape
        
        # 随机初始化聚类中心
        centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
        
        max_iterations = 100
        tolerance = 1e-4
        
        for iteration in range(max_iterations):
            # 分配样本到最近的聚类中心
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # 更新聚类中心
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
            
            # 检查收敛
            if np.allclose(centroids, new_centroids, atol=tolerance):
                break
            
            centroids = new_centroids
        
        # 计算聚类内平方和
        wcss = 0
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[k])**2)
        
        # 计算轮廓系数
        silhouette_score = MultivariateStatistics._calculate_silhouette_score(X, labels)
        
        return {
            'labels': labels,
            'centroids': centroids,
            'wcss': wcss,
            'silhouette_score': silhouette_score,
            'method': 'kmeans',
            'n_clusters': n_clusters
        }
    
    @staticmethod
    def _hierarchical_clustering(X: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """层次聚类 (简化版)"""
        n_samples = X.shape[0]
        
        # 初始化每个样本为一个聚类
        clusters = [[i] for i in range(n_samples)]
        distances = MultivariateStatistics._calculate_distance_matrix(X)
        
        # 层次聚类过程
        while len(clusters) > n_clusters:
            min_distance = float('inf')
            merge_indices = None
            
            # 找到距离最近的两个聚类
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster_distance = MultivariateStatistics._cluster_distance(clusters[i], clusters[j], distances)
                    if cluster_distance < min_distance:
                        min_distance = cluster_distance
                        merge_indices = (i, j)
            
            # 合并聚类
            i, j = merge_indices
            clusters[i].extend(clusters[j])
            del clusters[j]
        
        # 生成聚类标签
        labels = np.full(n_samples, -1)
        for cluster_id, cluster in enumerate(clusters):
            labels[cluster] = cluster_id
        
        # 计算聚类内平方和
        wcss = 0
        for cluster_id in range(n_clusters):
            cluster_points = X[labels == cluster_id]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                wcss += np.sum((cluster_points - centroid)**2)
        
        # 计算轮廓系数
        silhouette_score = MultivariateStatistics._calculate_silhouette_score(X, labels)
        
        return {
            'labels': labels,
            'wcss': wcss,
            'silhouette_score': silhouette_score,
            'method': 'hierarchical',
            'n_clusters': n_clusters
        }
    
    @staticmethod
    def _calculate_distance_matrix(X: np.ndarray) -> np.ndarray:
        """计算距离矩阵"""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j])**2))
                distances[i, j] = distances[j, i] = dist
        
        return distances
    
    @staticmethod
    def _cluster_distance(cluster1: List[int], cluster2: List[int], distances: np.ndarray) -> float:
        """计算两个聚类之间的距离 (平均连接)"""
        total_distance = 0
        count = 0
        
        for i in cluster1:
            for j in cluster2:
                total_distance += distances[i, j]
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    @staticmethod
    def _calculate_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
        """计算轮廓系数"""
        n_samples = X.shape[0]
        unique_labels = np.unique(labels)
        silhouette_scores = []
        
        for i in range(n_samples):
            # 计算a(i): 同聚类内平均距离
            same_cluster = X[labels == labels[i]]
            if len(same_cluster) > 1:
                a_i = np.mean([np.linalg.norm(X[i] - point) for point in same_cluster if not np.array_equal(point, X[i])])
            else:
                a_i = 0
            
            # 计算b(i): 最近聚类平均距离
            b_i = float('inf')
            for label in unique_labels:
                if label != labels[i]:
                    other_cluster = X[labels == label]
                    if len(other_cluster) > 0:
                        avg_distance = np.mean([np.linalg.norm(X[i] - point) for point in other_cluster])
                        b_i = min(b_i, avg_distance)
            
            # 轮廓系数
            if b_i == float('inf'):
                s_i = 0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
            
            silhouette_scores.append(s_i)
        
        return np.mean(silhouette_scores)


class AsyncStatisticalProcessor:
    """异步统计处理器"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def batch_descriptive_stats(self, datasets: List[Union[List[float], np.ndarray]]) -> List[Dict[str, float]]:
        """批量异步计算描述性统计"""
        functions = [(DescriptiveStatistics.descriptive_stats, [data]) for data in datasets]
        processor = AsyncStatsProcessor()
        return await processor.process_batch(functions)
    
    async def batch_t_tests(self, test_configs: List[Dict[str, Any]]) -> List[StatisticalResult]:
        """批量异步执行t检验"""
        functions = []
        for config in test_configs:
            if config['type'] == 'one_sample':
                func = InferentialStatistics.t_test_one_sample
                args = [config['data'], config['population_mean']]
            elif config['type'] == 'two_sample':
                func = InferentialStatistics.t_test_two_samples
                args = [config['data1'], config['data2']]
            else:
                raise ValueError(f"不支持的t检验类型: {config['type']}")
            
            functions.append((func, args))
        
        processor = AsyncStatsProcessor()
        return await processor.process_batch(functions)
    
    async def batch_correlation_analysis(self, data_pairs: List[Tuple[Union[List[float], np.ndarray], 
                                                                    Union[List[float], np.ndarray]]],
                                       method: str = 'pearson') -> List[StatisticalResult]:
        """批量异步相关性分析"""
        functions = [(InferentialStatistics.correlation_analysis, [x, y, method]) 
                    for x, y in data_pairs]
        processor = AsyncStatsProcessor()
        return await processor.process_batch(functions)
    
    async def batch_distribution_fitting(self, datasets: List[Union[List[float], np.ndarray]],
                                       distributions: List[str]) -> List[Dict[str, float]]:
        """批量异步分布拟合"""
        functions = [(ProbabilityDistributions.fit_distribution, [data, dist]) 
                    for data, dist in zip(datasets, distributions)]
        processor = AsyncStatsProcessor()
        return await processor.process_batch(functions)
    
    async def batch_cluster_analysis(self, datasets: List[np.ndarray],
                                   n_clusters_list: List[int],
                                   methods: List[str] = None) -> List[Dict[str, Any]]:
        """批量异步聚类分析"""
        if methods is None:
            methods = ['kmeans'] * len(datasets)
        
        functions = [(MultivariateStatistics.cluster_analysis, [data, n_clusters, method]) 
                    for data, n_clusters, method in zip(datasets, n_clusters_list, methods)]
        processor = AsyncStatsProcessor()
        return await processor.process_batch(functions)
    
    def __del__(self):
        """析构方法，清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class StatisticalTools:
    """J4统计工具主类"""
    
    def __init__(self, enable_logging: bool = True, log_level: str = 'INFO'):
        """
        初始化统计工具
        
        Args:
            enable_logging: 是否启用日志记录
            log_level: 日志级别
        """
        self.descriptive = DescriptiveStatistics()
        self.inferential = InferentialStatistics()
        self.distributions = ProbabilityDistributions()
        self.bayesian = BayesianStatistics()
        self.nonparametric = NonParametricStatistics()
        self.multivariate = MultivariateStatistics()
        self.async_processor = AsyncStatsProcessor()
        
        if enable_logging:
            logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        
        logger.info("J4统计工具初始化完成")
    
    def comprehensive_analysis(self, data: Union[List[float], np.ndarray],
                             analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        综合统计分析
        
        Args:
            data: 分析数据
            analysis_types: 分析类型列表，如果为None则执行所有分析
            
        Returns:
            包含所有分析结果的字典
        """
        try:
            if analysis_types is None:
                analysis_types = ['descriptive', 'normality', 'outliers', 'distribution']
            
            results = {}
            
            if 'descriptive' in analysis_types:
                logger.info("执行描述性统计分析")
                results['descriptive_stats'] = self.descriptive.descriptive_stats(data)
            
            if 'normality' in analysis_types:
                logger.info("执行正态性检验")
                results['normality_test'] = self.distributions.goodness_of_fit_test(data, 'normal')
            
            if 'outliers' in analysis_types:
                logger.info("检测异常值")
                results['outliers'] = self._detect_outliers(data)
            
            if 'distribution' in analysis_types:
                logger.info("拟合分布")
                results['distribution_fit'] = self._fit_multiple_distributions(data)
            
            logger.info("综合统计分析完成")
            return results
            
        except Exception as e:
            logger.error(f"综合统计分析时出错: {e}")
            raise
    
    def statistical_power_analysis(self, effect_size: float,
                                 alpha: float = 0.05,
                                 power: float = 0.8,
                                 test_type: str = 't_test') -> Dict[str, float]:
        """
        统计功效分析
        
        Args:
            effect_size: 效应量
            alpha: 显著性水平
            power: 统计功效
            test_type: 检验类型
            
        Returns:
            功效分析结果
        """
        try:
            if test_type == 't_test':
                # t检验的功效分析
                from scipy.stats import norm
                
                # 计算临界值
                z_alpha = norm.ppf(1 - alpha/2)
                z_beta = norm.ppf(power)
                
                # 计算所需样本量
                n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
                
                result = {
                    'effect_size': effect_size,
                    'alpha': alpha,
                    'power': power,
                    'required_sample_size': math.ceil(n),
                    'critical_value': z_alpha,
                    'test_type': 't_test'
                }
                
            elif test_type == 'correlation':
                # 相关性检验的功效分析
                from scipy.stats import norm
                
                # Fisher's z变换
                z_effect = 0.5 * math.log((1 + effect_size) / (1 - effect_size))
                z_alpha = norm.ppf(1 - alpha/2)
                z_beta = norm.ppf(power)
                
                n = 3 + ((z_alpha + z_beta) / z_effect) ** 2
                
                result = {
                    'effect_size': effect_size,
                    'alpha': alpha,
                    'power': power,
                    'required_sample_size': math.ceil(n),
                    'fisher_z': z_effect,
                    'test_type': 'correlation'
                }
                
            else:
                raise ValueError(f"不支持的检验类型: {test_type}")
            
            logger.info(f"功效分析完成: 需要样本量 = {result['required_sample_size']}")
            return result
            
        except Exception as e:
            logger.error(f"功效分析时出错: {e}")
            raise
    
    def sample_size_calculation(self, effect_size: float,
                              alpha: float = 0.05,
                              power: float = 0.8,
                              test_type: str = 't_test',
                              groups: int = 2) -> Dict[str, float]:
        """
        样本量计算
        
        Args:
            effect_size: 效应量 (Cohen's d)
            alpha: 显著性水平
            power: 统计功效
            test_type: 检验类型
            groups: 组数
            
        Returns:
            样本量计算结果
        """
        try:
            from scipy.stats import norm
            
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            
            if test_type == 't_test':
                if groups == 2:
                    # 两组t检验
                    n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
                    total_n = 2 * n_per_group
                else:
                    # 多组方差分析
                    f_effect = effect_size  # Cohen's f
                    n_per_group = ((z_alpha + z_beta) / f_effect) ** 2
                    total_n = groups * n_per_group
                    
            elif test_type == 'proportion':
                # 比例检验
                p1 = 0.5 + effect_size / 2
                p2 = 0.5 - effect_size / 2
                pooled_p = (p1 + p2) / 2
                n_per_group = 2 * pooled_p * (1 - pooled_p) * ((z_alpha + z_beta) / effect_size) ** 2
                total_n = 2 * n_per_group
                
            else:
                raise ValueError(f"不支持的检验类型: {test_type}")
            
            result = {
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'n_per_group': math.ceil(n_per_group),
                'total_n': math.ceil(total_n),
                'groups': groups,
                'test_type': test_type
            }
            
            logger.info(f"样本量计算完成: 每组需要 {result['n_per_group']} 个样本")
            return result
            
        except Exception as e:
            logger.error(f"样本量计算时出错: {e}")
            raise
    
    def _detect_outliers(self, data: Union[List[float], np.ndarray], 
                        method: str = 'iqr') -> Dict[str, Any]:
        """检测异常值"""
        data = np.array(data)
        
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
            
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = data[np.abs(modified_z_scores) > 3.5]
            
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        return {
            'method': method,
            'outliers': outliers.tolist(),
            'n_outliers': len(outliers),
            'outlier_indices': np.where(np.isin(data, outliers))[0].tolist()
        }
    
    def _fit_multiple_distributions(self, data: Union[List[float], np.ndarray]) -> Dict[str, Dict[str, float]]:
        """拟合多种分布"""
        distributions = ['normal', 't', 'chi_square', 'gamma']
        results = {}
        
        for dist in distributions:
            try:
                fit_result = self.distributions.fit_distribution(data, dist)
                results[dist] = fit_result
            except Exception as e:
                logger.warning(f"拟合{dist}分布时出错: {e}")
                continue
        
        # 找出最佳拟合分布
        if results:
            best_dist = min(results.keys(), key=lambda x: results[x]['aic'])
            results['best_fit'] = best_dist
        
        return results
    
    def __del__(self):
        """析构方法，清理资源"""
        logger.info("J4统计工具已销毁")


# 工具函数和辅助方法
def validate_data(data: Union[List[float], np.ndarray], 
                 required_length: Optional[int] = None) -> np.ndarray:
    """
    验证和清理数据
    
    Args:
        data: 输入数据
        required_length: 期望的数据长度
        
    Returns:
        验证后的numpy数组
        
    Raises:
        ValueError: 当数据不符合要求时
    """
    try:
        data = np.array(data)
        
        # 检查是否有足够的有效数据
        if len(data) == 0:
            raise ValueError("数据不能为空")
        
        # 检查是否包含NaN或无穷大值
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            # 移除无效值
            data = data[np.isfinite(data)]
            logger.warning(f"移除了{len(data) - np.sum(np.isfinite(data))}个无效值")
        
        # 检查长度要求
        if required_length is not None and len(data) != required_length:
            raise ValueError(f"数据长度{len(data)}不等于期望长度{required_length}")
        
        # 检查数据类型
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("数据必须是数值类型")
        
        return data
        
    except Exception as e:
        logger.error(f"数据验证失败: {e}")
        raise


def format_statistical_output(result: StatisticalResult, 
                            decimal_places: int = 4) -> str:
    """
    格式化统计结果输出
    
    Args:
        result: 统计结果对象
        decimal_places: 小数位数
        
    Returns:
        格式化的字符串
    """
    output = f"{result.test_name}:\n"
    output += f"  统计量: {result.statistic:.{decimal_places}f}\n"
    output += f"  p值: {result.p_value:.{decimal_places}f}\n"
    output += f"  解释: {result.interpretation}\n"
    
    if result.confidence_interval:
        ci_lower, ci_upper = result.confidence_interval
        output += f"  95%置信区间: [{ci_lower:.{decimal_places}f}, {ci_upper:.{decimal_places}f}]\n"
    
    if result.degrees_of_freedom:
        output += f"  自由度: {result.degrees_of_freedom}\n"
    
    if result.effect_size:
        output += f"  效应量: {result.effect_size:.{decimal_places}f}\n"
    
    if result.sample_size:
        output += f"  样本量: {result.sample_size}\n"
    
    return output


def create_statistical_report(results: Dict[str, Any], 
                            title: str = "统计分析报告") -> str:
    """
    创建统计报告
    
    Args:
        results: 统计结果字典
        title: 报告标题
        
    Returns:
        格式化的报告字符串
    """
    report = f"{'='*50}\n"
    report += f"{title:^50}\n"
    report += f"{'='*50}\n\n"
    
    for analysis_type, result in results.items():
        report += f"【{analysis_type.upper()}】\n"
        
        if isinstance(result, StatisticalResult):
            report += format_statistical_output(result)
        elif isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    report += f"  {key}: {value:.{4}f}\n"
                else:
                    report += f"  {key}: {value}\n"
        elif isinstance(result, list):
            for i, item in enumerate(result):
                report += f"  [{i}]: {item}\n"
        
        report += "\n"
    
    return report


# 示例使用代码
def example_usage():
    """使用示例"""
    print("J4统计工具使用示例")
    print("="*50)
    
    # 创建统计工具实例
    stats_tool = StatisticalTools()
    
    # 生成示例数据
    np.random.seed(42)
    data1 = np.random.normal(100, 15, 50)  # 第一组数据
    data2 = np.random.normal(110, 15, 50)  # 第二组数据
    
    print("1. 描述性统计")
    desc_stats = stats_tool.descriptive.descriptive_stats(data1)
    print(f"均值: {desc_stats['mean']:.4f}")
    print(f"标准差: {desc_stats['std']:.4f}")
    print(f"偏度: {desc_stats['skewness']:.4f}")
    print(f"峰度: {desc_stats['kurtosis']:.4f}")
    print()
    
    print("2. t检验")
    t_result = stats_tool.inferential.t_test_two_samples(data1, data2)
    print(format_statistical_output(t_result))
    print()
    
    print("3. 相关性分析")
    x = np.random.normal(0, 1, 30)
    y = x + np.random.normal(0, 0.5, 30)
    corr_result = stats_tool.inferential.correlation_analysis(x, y)
    print(format_statistical_output(corr_result))
    print()
    
    print("4. 非参数检验")
    mann_whitney_result = stats_tool.nonparametric.mann_whitney_test(data1, data2)
    print(format_statistical_output(mann_whitney_result))
    print()
    
    print("5. 贝叶斯推断")
    bayesian_result = stats_tool.bayesian.bayesian_normal_mean(data1)
    print(f"后验均值: {bayesian_result['posterior_mean']:.4f}")
    print(f"后验标准差: {bayesian_result['posterior_std']:.4f}")
    print()
    
    print("6. 主成分分析")
    # 创建多变量数据
    multivariate_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
    pca_result = stats_tool.multivariate.principal_component_analysis(multivariate_data, n_components=2)
    print(f"解释方差比例: {pca_result['explained_variance_ratio']}")
    print()
    
    print("7. 综合分析")
    comprehensive_result = stats_tool.comprehensive_analysis(data1)
    print("综合分析完成")
    for key in comprehensive_result.keys():
        print(f"  - {key}")
    print()
    
    print("8. 样本量计算")
    sample_size_result = stats_tool.sample_size_calculation(effect_size=0.5, alpha=0.05, power=0.8)
    print(f"所需样本量: {sample_size_result['total_n']}")
    print(f"每组样本量: {sample_size_result['n_per_group']}")
    print()
    
    print("示例运行完成!")


class TimeSeriesAnalysis:
    """时间序列分析工具类"""
    
    @staticmethod
    def moving_average(data: Union[List[float], np.ndarray], window: int) -> np.ndarray:
        """
        计算移动平均
        
        Args:
            data: 时间序列数据
            window: 移动窗口大小
            
        Returns:
            移动平均序列
        """
        try:
            data = np.array(data)
            if window <= 0 or window > len(data):
                raise ValueError("窗口大小必须大于0且小于等于数据长度")
            
            return np.convolve(data, np.ones(window)/window, mode='valid')
            
        except Exception as e:
            logger.error(f"计算移动平均时出错: {e}")
            raise
    
    @staticmethod
    def exponential_smoothing(data: Union[List[float], np.ndarray], alpha: float) -> np.ndarray:
        """
        指数平滑
        
        Args:
            data: 时间序列数据
            alpha: 平滑参数 (0 < alpha <= 1)
            
        Returns:
            指数平滑序列
        """
        try:
            data = np.array(data)
            if not 0 < alpha <= 1:
                raise ValueError("平滑参数alpha必须在(0,1]范围内")
            
            smoothed = np.zeros_like(data)
            smoothed[0] = data[0]
            
            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            
            return smoothed
            
        except Exception as e:
            logger.error(f"指数平滑时出错: {e}")
            raise
    
    @staticmethod
    def autocorrelation(data: Union[List[float], np.ndarray], max_lags: int = None) -> Dict[str, np.ndarray]:
        """
        计算自相关函数
        
        Args:
            data: 时间序列数据
            max_lags: 最大滞后阶数
            
        Returns:
            包含滞后和自相关系数的字典
        """
        try:
            data = np.array(data)
            n = len(data)
            
            if max_lags is None:
                max_lags = min(n // 4, 40)
            
            # 计算均值
            mean_val = np.mean(data)
            
            # 计算自相关
            autocorr = []
            lags = []
            
            for lag in range(1, max_lags + 1):
                if lag < n:
                    c0 = np.mean((data - mean_val) ** 2)
                    c_lag = np.mean((data[:-lag] - mean_val) * (data[lag:] - mean_val))
                    autocorr.append(c_lag / c0)
                    lags.append(lag)
            
            return {
                'lags': np.array(lags),
                'autocorrelation': np.array(autocorr)
            }
            
        except Exception as e:
            logger.error(f"计算自相关时出错: {e}")
            raise
    
    @staticmethod
    def stationarity_test(data: Union[List[float], np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """
        平稳性检验 (ADF检验)
        
        Args:
            data: 时间序列数据
            alpha: 显著性水平
            
        Returns:
            统计检验结果
        """
        try:
            if not STATSMODELS_AVAILABLE:
                raise ImportError("statsmodels库未安装，无法进行平稳性检验。请安装: pip install statsmodels")
            
            data = np.array(data)
            result = adfuller(data)
            
            adf_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            # 判断是否平稳
            is_stationary = p_value < alpha
            
            result_obj = StatisticalResult(
                statistic=adf_statistic,
                p_value=p_value,
                test_name="ADF平稳性检验",
                interpretation="平稳" if is_stationary else "非平稳"
            )
            
            # 添加临界值信息
            result_obj.critical_values = critical_values
            result_obj.is_stationary = is_stationary
            
            logger.info(f"平稳性检验完成: ADF统计量 = {adf_statistic:.4f}, p = {p_value:.4f}")
            return result_obj
            
        except Exception as e:
            logger.error(f"平稳性检验时出错: {e}")
            raise


class SurvivalAnalysis:
    """生存分析工具类"""
    
    @staticmethod
    def kaplan_meier_estimate(time_data: Union[List[float], np.ndarray],
                            event_data: Union[List[int], np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Kaplan-Meier生存函数估计
        
        Args:
            time_data: 生存时间数据
            event_data: 事件发生指示器 (1=事件发生, 0=删失)
            
        Returns:
            包含生存函数估计的字典
        """
        try:
            time_data = np.array(time_data)
            event_data = np.array(event_data)
            
            if len(time_data) != len(event_data):
                raise ValueError("时间数据和事件数据长度必须相同")
            
            # 排序
            sorted_indices = np.argsort(time_data)
            time_sorted = time_data[sorted_indices]
            event_sorted = event_data[sorted_indices]
            
            # 获取唯一的事件时间点
            unique_times = np.unique(time_sorted[event_sorted == 1])
            
            survival_prob = []
            survival_time = []
            
            n_at_risk = len(time_sorted)
            current_survival = 1.0
            
            for t in unique_times:
                # 在时间t发生事件的人数 (使用容差处理浮点数)
                events_at_t = np.sum((np.abs(time_sorted - t) < 1e-10) & (event_sorted == 1))
                # 在时间t处于风险的人数
                at_risk_at_t = np.sum(time_sorted >= t)
                
                if at_risk_at_t > 0:
                    survival_rate = 1 - (events_at_t / at_risk_at_t)
                    current_survival *= survival_rate
                    
                    survival_time.append(t)
                    survival_prob.append(current_survival)
            
            return {
                'time': np.array(survival_time),
                'survival_probability': np.array(survival_prob),
                'n_at_risk': n_at_risk,
                'n_events': np.sum(event_sorted)
            }
            
        except Exception as e:
            logger.error(f"Kaplan-Meier估计时出错: {e}")
            raise
    
    @staticmethod
    def log_rank_test(time1: Union[List[float], np.ndarray],
                     event1: Union[List[int], np.ndarray],
                     time2: Union[List[float], np.ndarray],
                     event2: Union[List[int], np.ndarray]) -> StatisticalResult:
        """
        Log-rank检验 (比较两组生存曲线)
        
        Args:
            time1: 第一组生存时间
            event1: 第一组事件指示器
            time2: 第二组生存时间
            event2: 第二组事件指示器
            
        Returns:
            统计检验结果
        """
        try:
            time1 = np.array(time1)
            event1 = np.array(event1)
            time2 = np.array(time2)
            event2 = np.array(event2)
            
            # 合并数据
            all_time = np.concatenate([time1, time2])
            all_event = np.concatenate([event1, event2])
            group = np.concatenate([np.zeros(len(time1)), np.ones(len(time2))])
            
            # 排序
            sorted_indices = np.argsort(all_time)
            time_sorted = all_time[sorted_indices]
            event_sorted = all_event[sorted_indices]
            group_sorted = group[sorted_indices]
            
            # 计算log-rank统计量
            observed = 0
            expected = 0
            variance = 0
            
            unique_times = np.unique(time_sorted[event_sorted == 1])
            
            for t in unique_times:
                # 在时间t处于风险的人数
                at_risk = np.sum(time_sorted >= t)
                # 在时间t发生事件的人数 (使用容差处理浮点数)
                events = np.sum((np.abs(time_sorted - t) < 1e-10) & (event_sorted == 1))
                # 在时间t组1发生事件的人数 (使用容差处理浮点数)
                events_group1 = np.sum((np.abs(time_sorted - t) < 1e-10) & (event_sorted == 1) & (group_sorted == 0))
                # 在时间t组1处于风险的人数
                at_risk_group1 = np.sum((time_sorted >= t) & (group_sorted == 0))
                
                if at_risk > 0:
                    expected_events = (at_risk_group1 * events) / at_risk
                    observed += events_group1
                    expected += expected_events
                    
                    if at_risk > 1:
                        variance += (at_risk_group1 * events * (at_risk - events)) / (at_risk**2 * (at_risk - 1))
            
            # 计算统计量
            if variance > 0:
                z_stat = (observed - expected) / math.sqrt(variance)
                chi2_stat = z_stat ** 2
                p_value = 1 - stats.chi2.cdf(chi2_stat, 1)
            else:
                chi2_stat = 0
                p_value = 1
            
            result = StatisticalResult(
                statistic=chi2_stat,
                p_value=p_value,
                degrees_of_freedom=1,
                test_name="Log-rank检验"
            )
            
            logger.info(f"Log-rank检验完成: χ² = {chi2_stat:.4f}, p = {p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Log-rank检验时出错: {e}")
            raise


class QualityControl:
    """质量控制工具类"""
    
    @staticmethod
    def control_charts(data: Union[List[float], np.ndarray],
                      chart_type: str = 'xbar_r',
                      subgroups: Optional[int] = None) -> Dict[str, Any]:
        """
        控制图分析
        
        Args:
            data: 质量控制数据
            chart_type: 控制图类型 ('xbar_r', 'xbar_s', 'individuals')
            subgroups: 子组大小
            
        Returns:
            控制图分析结果
        """
        try:
            data = np.array(data)
            n = len(data)
            
            if chart_type == 'xbar_r':
                if subgroups is None:
                    subgroups = 5  # 默认子组大小
                
                # 重新整理数据为子组
                n_subgroups = n // subgroups
                data_matrix = data[:n_subgroups * subgroups].reshape(n_subgroups, subgroups)
                
                # 计算子组均值和极差
                subgroup_means = np.mean(data_matrix, axis=1)
                subgroup_ranges = np.ptp(data_matrix, axis=1)
                
                # 计算控制限
                overall_mean = np.mean(subgroup_means)
                mean_range = np.mean(subgroup_ranges)
                
                # X-bar图控制限
                A2 = {2: 1.88, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483}
                a2 = A2.get(subgroups, 0.577)
                
                ucl_xbar = overall_mean + a2 * mean_range
                lcl_xbar = overall_mean - a2 * mean_range
                
                # R图控制限
                D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
                D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004}
                
                d3 = D3.get(subgroups, 0)
                d4 = D4.get(subgroups, 2.114)
                
                ucl_r = d4 * mean_range
                lcl_r = d3 * mean_range
                
                result = {
                    'chart_type': 'X-bar and R',
                    'subgroup_size': subgroups,
                    'n_subgroups': n_subgroups,
                    'subgroup_means': subgroup_means,
                    'subgroup_ranges': subgroup_ranges,
                    'overall_mean': overall_mean,
                    'mean_range': mean_range,
                    'xbar_ucl': ucl_xbar,
                    'xbar_lcl': lcl_xbar,
                    'r_ucl': ucl_r,
                    'r_lcl': lcl_r
                }
                
            elif chart_type == 'individuals':
                # 个体控制图
                mean_val = np.mean(data)
                std_val = np.std(data, ddof=1)
                
                ucl = mean_val + 3 * std_val
                lcl = mean_val - 3 * std_val
                
                # 检测异常点
                out_of_control = np.where((data < lcl) | (data > ucl))[0]
                
                result = {
                    'chart_type': 'Individuals',
                    'data': data,
                    'mean': mean_val,
                    'std': std_val,
                    'ucl': ucl,
                    'lcl': lcl,
                    'out_of_control_points': out_of_control.tolist(),
                    'n_out_of_control': len(out_of_control)
                }
            
            else:
                raise ValueError(f"不支持的控制图类型: {chart_type}")
            
            logger.info(f"控制图分析完成: {chart_type}")
            return result
            
        except Exception as e:
            logger.error(f"控制图分析时出错: {e}")
            raise
    
    @staticmethod
    def process_capability(data: Union[List[float], np.ndarray],
                          lsl: float = None,
                          usl: float = None,
                          target: float = None) -> Dict[str, float]:
        """
        过程能力分析
        
        Args:
            data: 过程数据
            lsl: 规格下限
            usl: 规格上限
            target: 目标值
            
        Returns:
            过程能力指标
        """
        try:
            data = np.array(data)
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            
            # 计算过程能力指标
            cp = None
            cpk = None
            pp = None
            ppk = None
            
            if lsl is not None and usl is not None:
                # Cp和Cpk
                cp = (usl - lsl) / (6 * std_val)
                cpu = (usl - mean_val) / (3 * std_val)
                cpl = (mean_val - lsl) / (3 * std_val)
                cpk = min(cpu, cpl)
            
            if target is not None:
                # Pp和Ppk (使用长期标准差)
                pp = (usl - lsl) / (6 * std_val) if lsl is not None and usl is not None else None
                ppu = (usl - mean_val) / (3 * std_val) if usl is not None else None
                ppl = (mean_val - lsl) / (3 * std_val) if lsl is not None else None
                ppk = min(ppu, ppl) if ppu is not None and ppl is not None else None
            
            # 过程性能评估
            capability_assessment = ""
            if cpk is not None:
                if cpk >= 1.33:
                    capability_assessment = "优秀"
                elif cpk >= 1.0:
                    capability_assessment = "良好"
                elif cpk >= 0.67:
                    capability_assessment = "一般"
                else:
                    capability_assessment = "不足"
            
            result = {
                'mean': mean_val,
                'std': std_val,
                'cp': cp,
                'cpk': cpk,
                'pp': pp,
                'ppk': ppk,
                'capability_assessment': capability_assessment
            }
            
            logger.info(f"过程能力分析完成: Cp = {cp:.4f}, Cpk = {cpk:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"过程能力分析时出错: {e}")
            raise


class ExperimentalDesign:
    """实验设计工具类"""
    
    @staticmethod
    def factorial_design(factors: List[str], levels: List[int]) -> Dict[str, Any]:
        """
        创建因子实验设计
        
        Args:
            factors: 因子名称列表
            levels: 各因子的水平数列表
            
        Returns:
            实验设计矩阵
        """
        try:
            if len(factors) != len(levels):
                raise ValueError("因子名称和水平数列表长度必须相同")
            
            # 生成所有因子组合
            design_matrix = []
            
            def generate_combinations(factor_idx, current_combination):
                if factor_idx == len(factors):
                    design_matrix.append(current_combination.copy())
                    return
                
                for level in range(levels[factor_idx]):
                    current_combination.append(level)
                    generate_combinations(factor_idx + 1, current_combination)
                    current_combination.pop()
            
            generate_combinations(0, [])
            
            # 转换为numpy数组
            design_array = np.array(design_matrix)
            
            # 计算总实验次数
            total_experiments = np.prod(levels)
            
            result = {
                'factors': factors,
                'levels': levels,
                'design_matrix': design_array,
                'total_experiments': total_experiments,
                'factor_combinations': design_matrix
            }
            
            logger.info(f"因子实验设计创建完成: {total_experiments}次实验")
            return result
            
        except Exception as e:
            logger.error(f"创建因子实验设计时出错: {e}")
            raise
    
    @staticmethod
    def randomized_complete_block_design(treatments: List[str],
                                       blocks: List[str]) -> Dict[str, np.ndarray]:
        """
        完全随机区组设计
        
        Args:
            treatments: 处理水平列表
            blocks: 区组列表
            
        Returns:
            实验设计布局
        """
        try:
            n_treatments = len(treatments)
            n_blocks = len(blocks)
            
            # 创建设计矩阵
            design_matrix = np.zeros((n_blocks, n_treatments), dtype=int)
            
            # 为每个区组随机分配处理
            np.random.seed(42)
            for block in range(n_blocks):
                treatment_order = np.random.permutation(n_treatments)
                design_matrix[block, :] = treatment_order
            
            result = {
                'design_matrix': design_matrix,
                'treatments': treatments,
                'blocks': blocks,
                'n_treatments': n_treatments,
                'n_blocks': n_blocks,
                'total_experiments': n_treatments * n_blocks
            }
            
            logger.info(f"完全随机区组设计创建完成: {n_treatments * n_blocks}次实验")
            return result
            
        except Exception as e:
            logger.error(f"创建完全随机区组设计时出错: {e}")
            raise
    
    @staticmethod
    def response_surface_design(central_composite: bool = True,
                              factors: List[str] = None,
                              levels: int = 3) -> Dict[str, Any]:
        """
        响应面设计
        
        Args:
            central_composite: 是否使用中心复合设计
            factors: 因子列表
            levels: 水平数
            
        Returns:
            响应面设计矩阵
        """
        try:
            if factors is None:
                factors = ['X1', 'X2']  # 默认2个因子
            
            n_factors = len(factors)
            
            if central_composite:
                # 中心复合设计
                design_points = []
                
                # 立方点 (2^k个点)
                for i in range(2 ** n_factors):
                    point = []
                    for j in range(n_factors):
                        if (i >> j) & 1:
                            point.append(1)  # 高水平
                        else:
                            point.append(-1)  # 低水平
                    design_points.append(point)
                
                # 轴向点 (2*k个点)
                for i in range(n_factors):
                    point = [-1] * n_factors
                    point[i] = -alpha  # 轴向点 (低)
                    design_points.append(point)
                    
                    point = [1] * n_factors
                    point[i] = alpha   # 轴向点 (高)
                    design_points.append(point)
                
                # 中心点 (n_c个点)
                alpha = math.sqrt(n_factors)  # 轴向距离
                n_center = 5  # 中心点重复次数
                for _ in range(n_center):
                    design_points.append([0] * n_factors)
                
                design_array = np.array(design_points)
                
            else:
                # Box-Behnken设计
                design_points = []
                
                def generate_bb_design(factor_idx, current_design):
                    if factor_idx == n_factors:
                        design_points.append(current_design.copy())
                        return
                    
                    # 当前因子取-1和1，其他因子取0
                    for level in [-1, 1]:
                        for other_factor in range(n_factors):
                            if other_factor != factor_idx:
                                current_design[other_factor] = 0
                        current_design[factor_idx] = level
                        generate_bb_design(factor_idx + 1, current_design)
                
                current_design = [0] * n_factors
                generate_bb_design(0, current_design)
                
                # 添加中心点
                for _ in range(3):
                    design_points.append([0] * n_factors)
                
                design_array = np.array(design_points)
            
            result = {
                'design_matrix': design_array,
                'factors': factors,
                'n_experiments': len(design_array),
                'design_type': 'Central Composite' if central_composite else 'Box-Behnken'
            }
            
            logger.info(f"响应面设计创建完成: {len(design_array)}次实验")
            return result
            
        except Exception as e:
            logger.error(f"创建响应面设计时出错: {e}")
            raise


# 扩展主统计工具类
class StatisticalToolsExtended(StatisticalTools):
    """扩展的统计工具类"""
    
    def __init__(self, enable_logging: bool = True, log_level: str = 'INFO'):
        super().__init__(enable_logging, log_level)
        self.timeseries = TimeSeriesAnalysis()
        self.survival = SurvivalAnalysis()
        self.quality_control = QualityControl()
        self.experimental_design = ExperimentalDesign()
        logger.info("扩展统计工具初始化完成")
    
    def comprehensive_quality_analysis(self, data: Union[List[float], np.ndarray],
                                     lsl: float = None,
                                     usl: float = None) -> Dict[str, Any]:
        """
        综合质量分析
        
        Args:
            data: 质量控制数据
            lsl: 规格下限
            usl: 规格上限
            
        Returns:
            综合质量分析结果
        """
        try:
            results = {}
            
            # 控制图分析
            logger.info("执行控制图分析")
            results['control_charts'] = self.quality_control.control_charts(data)
            
            # 过程能力分析
            logger.info("执行过程能力分析")
            results['process_capability'] = self.quality_control.process_capability(data, lsl, usl)
            
            # 描述性统计
            logger.info("执行描述性统计")
            results['descriptive_stats'] = self.descriptive.descriptive_stats(data)
            
            # 异常值检测
            logger.info("检测异常值")
            results['outliers'] = self._detect_outliers(data)
            
            logger.info("综合质量分析完成")
            return results
            
        except Exception as e:
            logger.error(f"综合质量分析时出错: {e}")
            raise
    
    def time_series_forecast(self, data: Union[List[float], np.ndarray],
                           method: str = 'moving_average',
                           **kwargs) -> Dict[str, np.ndarray]:
        """
        时间序列预测
        
        Args:
            data: 时间序列数据
            method: 预测方法 ('moving_average', 'exponential_smoothing')
            **kwargs: 方法特定参数
            
        Returns:
            预测结果
        """
        try:
            data = np.array(data)
            
            if method == 'moving_average':
                window = kwargs.get('window', 5)
                forecast = self.timeseries.moving_average(data, window)
                
            elif method == 'exponential_smoothing':
                alpha = kwargs.get('alpha', 0.3)
                forecast = self.timeseries.exponential_smoothing(data, alpha)
                
            else:
                raise ValueError(f"不支持的预测方法: {method}")
            
            result = {
                'original_data': data,
                'forecast': forecast,
                'method': method,
                'parameters': kwargs
            }
            
            logger.info(f"时间序列预测完成: {method}")
            return result
            
        except Exception as e:
            logger.error(f"时间序列预测时出错: {e}")
            raise
    
    def survival_analysis_complete(self, time_data: Union[List[float], np.ndarray],
                                 event_data: Union[List[int], np.ndarray],
                                 group_data: Optional[Union[List[int], np.ndarray]] = None) -> Dict[str, Any]:
        """
        完整生存分析
        
        Args:
            time_data: 生存时间
            event_data: 事件指示器
            group_data: 分组数据 (可选)
            
        Returns:
            生存分析结果
        """
        try:
            results = {}
            
            # Kaplan-Meier估计
            logger.info("执行Kaplan-Meier生存函数估计")
            results['kaplan_meier'] = self.survival.kaplan_meier_estimate(time_data, event_data)
            
            # 如果有分组数据，进行log-rank检验
            if group_data is not None:
                logger.info("执行log-rank检验")
                group_data = np.array(group_data)
                unique_groups = np.unique(group_data)
                
                if len(unique_groups) == 2:
                    group1_mask = group_data == unique_groups[0]
                    time1 = np.array(time_data)[group1_mask]
                    event1 = np.array(event_data)[group1_mask]
                    time2 = np.array(time_data)[~group1_mask]
                    event2 = np.array(event_data)[~group1_mask]
                    
                    results['log_rank_test'] = self.survival.log_rank_test(time1, event1, time2, event2)
            
            logger.info("生存分析完成")
            return results
            
        except Exception as e:
            logger.error(f"生存分析时出错: {e}")
            raise


# 高级示例和测试函数
def advanced_examples():
    """高级功能示例"""
    print("J4统计工具高级功能示例")
    print("="*60)
    
    # 创建扩展统计工具实例
    stats_tool = StatisticalToolsExtended()
    
    # 1. 时间序列分析示例
    print("1. 时间序列分析")
    np.random.seed(42)
    ts_data = np.cumsum(np.random.randn(100)) + 100
    
    # 移动平均预测
    ma_forecast = stats_tool.time_series_forecast(ts_data, method='moving_average', window=5)
    print(f"移动平均预测完成，预测长度: {len(ma_forecast['forecast'])}")
    
    # 指数平滑
    es_forecast = stats_tool.time_series_forecast(ts_data, method='exponential_smoothing', alpha=0.3)
    print(f"指数平滑预测完成，预测长度: {len(es_forecast['forecast'])}")
    
    # 自相关分析
    autocorr_result = stats_tool.timeseries.autocorrelation(ts_data, max_lags=10)
    print(f"自相关分析完成，最大滞后: {len(autocorr_result['lags'])}")
    print()
    
    # 2. 生存分析示例
    print("2. 生存分析")
    # 生成模拟生存数据
    survival_times = np.random.exponential(50, 100)
    survival_events = np.random.binomial(1, 0.7, 100)
    
    survival_results = stats_tool.survival_analysis_complete(survival_times, survival_events)
    km_result = survival_results['kaplan_meier']
    print(f"Kaplan-Meier估计完成，时间点数: {len(km_result['time'])}")
    print(f"风险集大小: {km_result['n_at_risk']}, 事件数: {km_result['n_events']}")
    print()
    
    # 3. 质量控制示例
    print("3. 质量控制")
    quality_data = np.random.normal(100, 5, 100)
    quality_results = stats_tool.comprehensive_quality_analysis(quality_data, lsl=90, usl=110)
    
    control_chart = quality_results['control_charts']
    print(f"控制图类型: {control_chart['chart_type']}")
    
    capability = quality_results['process_capability']
    print(f"过程能力指数 Cp: {capability['cp']:.4f}")
    print(f"过程能力指数 Cpk: {capability['cpk']:.4f}")
    print(f"能力评估: {capability['capability_assessment']}")
    print()
    
    # 4. 实验设计示例
    print("4. 实验设计")
    # 因子设计
    factors = ['温度', '压力', '浓度']
    levels = [3, 3, 3]
    factorial_design = stats_tool.experimental_design.factorial_design(factors, levels)
    print(f"因子设计完成，总实验次数: {factorial_design['total_experiments']}")
    
    # 响应面设计
    rsm_design = stats_tool.experimental_design.response_surface_design(
        central_composite=True, factors=['温度', '压力'], levels=3
    )
    print(f"响应面设计完成，实验次数: {rsm_design['n_experiments']}")
    print()
    
    # 5. 高级统计分析示例
    print("5. 高级统计分析")
    # 生成多变量数据
    multivariate_data = np.random.multivariate_normal([0, 0, 0], 
                                                     [[1, 0.3, 0.2], 
                                                      [0.3, 1, 0.4], 
                                                      [0.2, 0.4, 1]], 200)
    
    # 主成分分析
    pca_result = stats_tool.multivariate.principal_component_analysis(multivariate_data, n_components=2)
    print(f"主成分分析完成，解释方差比例: {pca_result['cumulative_variance_ratio'][-1]:.4f}")
    
    # 聚类分析
    cluster_result = stats_tool.multivariate.cluster_analysis(multivariate_data, n_clusters=3, method='kmeans')
    print(f"K-means聚类完成，轮廓系数: {cluster_result['silhouette_score']:.4f}")
    print()
    
    print("所有高级功能示例运行完成!")


def performance_benchmark():
    """性能基准测试"""
    print("J4统计工具性能基准测试")
    print("="*40)
    
    import time
    
    stats_tool = StatisticalTools()
    
    # 测试不同数据规模的性能
    sizes = [100, 1000, 5000, 10000]
    
    for size in sizes:
        print(f"\n测试数据规模: {size}")
        
        # 生成测试数据
        np.random.seed(42)
        data = np.random.normal(100, 15, size)
        
        # 描述性统计
        start_time = time.time()
        desc_stats = stats_tool.descriptive.descriptive_stats(data)
        desc_time = time.time() - start_time
        print(f"描述性统计: {desc_time:.4f}秒")
        
        # t检验
        data1 = np.random.normal(100, 15, size//2)
        data2 = np.random.normal(105, 15, size//2)
        
        start_time = time.time()
        t_result = stats_tool.inferential.t_test_two_samples(data1, data2)
        t_time = time.time() - start_time
        print(f"t检验: {t_time:.4f}秒")
        
        # 相关性分析
        x = np.random.normal(0, 1, size)
        y = x + np.random.normal(0, 0.5, size)
        
        start_time = time.time()
        corr_result = stats_tool.inferential.correlation_analysis(x, y)
        corr_time = time.time() - start_time
        print(f"相关性分析: {corr_time:.4f}秒")
    
    print("\n性能基准测试完成!")


def comprehensive_test_suite():
    """综合测试套件"""
    print("J4统计工具综合测试套件")
    print("="*50)
    
    test_results = []
    
    try:
        stats_tool = StatisticalTools()
        
        # 测试1: 描述性统计
        print("测试1: 描述性统计")
        data = np.random.normal(100, 15, 100)
        desc_stats = stats_tool.descriptive.descriptive_stats(data)
        assert 'mean' in desc_stats
        assert 'std' in desc_stats
        test_results.append(("描述性统计", True))
        print("✓ 通过")
        
        # 测试2: t检验
        print("测试2: t检验")
        data1 = np.random.normal(100, 15, 50)
        data2 = np.random.normal(105, 15, 50)
        t_result = stats_tool.inferential.t_test_two_samples(data1, data2)
        assert hasattr(t_result, 'statistic')
        assert hasattr(t_result, 'p_value')
        test_results.append(("t检验", True))
        print("✓ 通过")
        
        # 测试3: 非参数检验
        print("测试3: 非参数检验")
        mw_result = stats_tool.nonparametric.mann_whitney_test(data1, data2)
        assert hasattr(mw_result, 'statistic')
        assert hasattr(mw_result, 'p_value')
        test_results.append(("Mann-Whitney检验", True))
        print("✓ 通过")
        
        # 测试4: 分布拟合
        print("测试4: 分布拟合")
        fit_result = stats_tool.distributions.fit_distribution(data, 'normal')
        assert 'distribution' in fit_result
        assert 'aic' in fit_result
        test_results.append(("分布拟合", True))
        print("✓ 通过")
        
        # 测试5: 贝叶斯推断
        print("测试5: 贝叶斯推断")
        bayes_result = stats_tool.bayesian.bayesian_normal_mean(data)
        assert 'posterior_mean' in bayes_result
        assert 'posterior_std' in bayes_result
        test_results.append(("贝叶斯推断", True))
        print("✓ 通过")
        
        # 测试6: 多变量分析
        print("测试6: 多变量分析")
        mv_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
        pca_result = stats_tool.multivariate.principal_component_analysis(mv_data, n_components=2)
        assert 'components' in pca_result
        assert 'explained_variance_ratio' in pca_result
        test_results.append(("主成分分析", True))
        print("✓ 通过")
        
        # 测试7: 综合分析
        print("测试7: 综合分析")
        comp_result = stats_tool.comprehensive_analysis(data)
        assert 'descriptive_stats' in comp_result
        test_results.append(("综合分析", True))
        print("✓ 通过")
        
        # 测试8: 样本量计算
        print("测试8: 样本量计算")
        sample_result = stats_tool.sample_size_calculation(effect_size=0.5, alpha=0.05, power=0.8)
        assert 'total_n' in sample_result
        assert 'n_per_group' in sample_result
        test_results.append(("样本量计算", True))
        print("✓ 通过")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        test_results.append(("综合测试", False))
    
    # 输出测试结果摘要
    print("\n测试结果摘要:")
    print("-" * 30)
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败")


if __name__ == "__main__":
    print("选择要运行的示例:")
    print("1. 基础功能示例")
    print("2. 高级功能示例")
    print("3. 性能基准测试")
    print("4. 综合测试套件")
    print("5. 全部运行")
    
    choice = input("请选择 (1-5): ").strip()
    
    if choice == "1":
        example_usage()
    elif choice == "2":
        advanced_examples()
    elif choice == "3":
        performance_benchmark()
    elif choice == "4":
        comprehensive_test_suite()
    elif choice == "5":
        print("运行所有示例和测试...")
        print()
        example_usage()
        print("\n" + "="*60 + "\n")
        advanced_examples()
        print("\n" + "="*60 + "\n")
        performance_benchmark()
        print("\n" + "="*60 + "\n")
        comprehensive_test_suite()
    else:
        print("无效选择，运行基础示例...")
        example_usage()