#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C4 关联分析器
实现多种关联分析方法，包括线性、非线性、时间序列等关联分析功能


创建时间: 2025-11-05
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from itertools import combinations
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """
    关联分析器
    
    提供多种关联分析方法：
    - 线性关联分析（Pearson、Spearman、Kendall）
    - 非线性关联分析（互信息、距离相关、Copula相关）
    - 时间序列关联分析（滞后相关、动态相关）
    - 条件关联分析（偏相关、格兰杰因果）
    - 关联模式发现和解释
    """
    
    def __init__(self, alpha: float = 0.05, max_lag: int = 10, 
                 n_permutations: int = 1000, random_state: int = 42):
        """
        初始化关联分析器
        
        参数:
            alpha: 显著性水平
            max_lag: 最大滞后阶数
            n_permutations: 置换检验次数
            random_state: 随机种子
        """
        self.alpha = alpha
        self.max_lag = max_lag
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.results = {}
        
    def pearson_correlation(self, x: np.ndarray, y: np.ndarray, 
                          method: str = 'pearson') -> Dict[str, Any]:
        """
        计算Pearson相关系数
        
        参数:
            x, y: 输入数组
            method: 相关方法 ('pearson', 'spearman', 'kendall')
            
        返回:
            包含相关系数、p值、置信区间的字典
        """
        try:
            if method == 'pearson':
                corr, p_value = pearsonr(x, y)
                # 计算置信区间
                n = len(x)
                z = 0.5 * np.log((1 + corr) / (1 - corr))
                se = 1 / np.sqrt(n - 3)
                z_crit = stats.norm.ppf(1 - self.alpha / 2)
                ci_lower = np.tanh(z - z_crit * se)
                ci_upper = np.tanh(z + z_crit * se)
                
            elif method == 'spearman':
                corr, p_value = spearmanr(x, y)
                # Spearman的置信区间近似计算
                n = len(x)
                se = 1.06 / np.sqrt(n - 1)  # 近似标准误差
                z_crit = stats.norm.ppf(1 - self.alpha / 2)
                ci_lower = corr - z_crit * se
                ci_upper = corr + z_crit * se
                
            elif method == 'kendall':
                corr, p_value = kendalltau(x, y)
                # Kendall的置信区间计算
                n = len(x)
                var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1))
                se = np.sqrt(var_tau)
                z_crit = stats.norm.ppf(1 - self.alpha / 2)
                ci_lower = corr - z_crit * se
                ci_upper = corr + z_crit * se
            else:
                raise ValueError(f"不支持的方法: {method}")
            
            return {
                'correlation': corr,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < self.alpha,
                'method': method,
                'n_samples': len(x)
            }
            
        except Exception as e:
            logger.error(f"计算{method}相关系数时出错: {e}")
            return {
                'correlation': np.nan,
                'p_value': 1.0,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'significant': False,
                'method': method,
                'n_samples': len(x),
                'error': str(e)
            }
    
    def mutual_information(self, x: np.ndarray, y: np.ndarray, 
                         discrete: bool = False) -> Dict[str, Any]:
        """
        计算互信息
        
        参数:
            x, y: 输入数组
            discrete: 是否为离散变量
            
        返回:
            包含互信息和标准化值的字典
        """
        try:
            # 重塑数据以适应sklearn的要求
            x_reshaped = x.reshape(-1, 1)
            
            # 计算互信息
            mi = mutual_info_regression(x_reshaped, y, random_state=self.random_state)[0]
            
            # 计算归一化互信息
            h_x = stats.entropy(np.histogram(x, bins='auto')[0] + 1e-10)
            h_y = stats.entropy(np.histogram(y, bins='auto')[0] + 1e-10)
            normalized_mi = mi / min(h_x, h_y) if min(h_x, h_y) > 0 else 0
            
            # 置换检验计算p值
            mi_permuted = []
            for _ in range(self.n_permutations):
                y_permuted = np.random.permutation(y)
                mi_perm = mutual_info_regression(x_reshaped, y_permuted, 
                                               random_state=self.random_state)[0]
                mi_permuted.append(mi_perm)
            
            p_value = np.mean(np.array(mi_permuted) >= mi)
            
            return {
                'mutual_information': mi,
                'normalized_mi': normalized_mi,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'method': 'mutual_information'
            }
            
        except Exception as e:
            logger.error(f"计算互信息时出错: {e}")
            return {
                'mutual_information': np.nan,
                'normalized_mi': np.nan,
                'p_value': 1.0,
                'significant': False,
                'method': 'mutual_information',
                'error': str(e)
            }
    
    def distance_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        计算距离相关系数
        
        参数:
            x, y: 输入数组
            
        返回:
            包含距离相关系数的字典
        """
        try:
            def distance_matrix(data):
                n = len(data)
                dist_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        dist_matrix[i, j] = np.abs(data[i] - data[j])
                return dist_matrix
            
            def double_center(matrix):
                n = len(matrix)
                row_mean = np.mean(matrix, axis=1, keepdims=True)
                col_mean = np.mean(matrix, axis=0, keepdims=True)
                grand_mean = np.mean(matrix)
                centered = matrix - row_mean - col_mean + grand_mean
                return centered
            
            # 计算距离矩阵
            dx = distance_matrix(x)
            dy = distance_matrix(y)
            
            # 双重中心化
            dx_centered = double_center(dx)
            dy_centered = double_center(dy)
            
            # 计算距离协方差和距离方差
            dcov_xy = np.sqrt(np.mean(dx_centered * dy_centered))
            dvar_x = np.sqrt(np.mean(dx_centered * dx_centered))
            dvar_y = np.sqrt(np.mean(dy_centered * dy_centered))
            
            # 计算距离相关系数
            dcor = dcov_xy / np.sqrt(dvar_x * dvar_y) if dvar_x * dvar_y > 0 else 0
            
            # 置换检验计算p值
            dcor_permuted = []
            for _ in range(self.n_permutations):
                y_permuted = np.random.permutation(y)
                dy_perm = distance_matrix(y_permuted)
                dy_centered_perm = double_center(dy_perm)
                dcov_perm = np.sqrt(np.mean(dx_centered * dy_centered_perm))
                dcor_perm = dcov_perm / np.sqrt(dvar_x * dvar_y) if dvar_x * dvar_y > 0 else 0
                dcor_permuted.append(dcor_perm)
            
            p_value = np.mean(np.array(dcor_permuted) >= dcor)
            
            return {
                'distance_correlation': dcor,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'method': 'distance_correlation'
            }
            
        except Exception as e:
            logger.error(f"计算距离相关系数时出错: {e}")
            return {
                'distance_correlation': np.nan,
                'p_value': 1.0,
                'significant': False,
                'method': 'distance_correlation',
                'error': str(e)
            }
    
    def lag_correlation(self, x: np.ndarray, y: np.ndarray, 
                       max_lag: Optional[int] = None) -> Dict[str, Any]:
        """
        计算滞后相关分析
        
        参数:
            x, y: 时间序列数据
            max_lag: 最大滞后阶数
            
        返回:
            包含滞后相关结果的字典
        """
        if max_lag is None:
            max_lag = self.max_lag
        
        try:
            lags = range(-max_lag, max_lag + 1)
            correlations = []
            p_values = []
            
            for lag in lags:
                if lag < 0:
                    # x滞后于y
                    x_shifted = x[-lag:]
                    y_shifted = y[:len(x_shifted)]
                elif lag > 0:
                    # y滞后于x
                    x_shifted = x[:-lag]
                    y_shifted = y[lag:]
                else:
                    # 同期相关
                    x_shifted = x
                    y_shifted = y
                
                if len(x_shifted) > 3:  # 确保有足够的样本
                    corr, p_val = pearsonr(x_shifted, y_shifted)
                    correlations.append(corr)
                    p_values.append(p_val)
                else:
                    correlations.append(np.nan)
                    p_values.append(1.0)
            
            # 找到最大相关及其滞后
            valid_corrs = [c for c in correlations if not np.isnan(c)]
            if valid_corrs:
                max_corr_idx = np.nanargmax(np.abs(correlations))
                max_lag_optimal = lags[max_corr_idx]
                max_corr = correlations[max_corr_idx]
                max_p_value = p_values[max_corr_idx]
            else:
                max_lag_optimal = 0
                max_corr = np.nan
                max_p_value = 1.0
            
            return {
                'lags': list(lags),
                'correlations': correlations,
                'p_values': p_values,
                'max_correlation': max_corr,
                'optimal_lag': max_lag_optimal,
                'max_p_value': max_p_value,
                'significant': max_p_value < self.alpha if not np.isnan(max_p_value) else False,
                'method': 'lag_correlation'
            }
            
        except Exception as e:
            logger.error(f"计算滞后相关时出错: {e}")
            return {
                'lags': [],
                'correlations': [],
                'p_values': [],
                'max_correlation': np.nan,
                'optimal_lag': 0,
                'max_p_value': 1.0,
                'significant': False,
                'method': 'lag_correlation',
                'error': str(e)
            }
    
    def partial_correlation(self, data: pd.DataFrame, x: str, y: str, 
                          control_vars: List[str]) -> Dict[str, Any]:
        """
        计算偏相关系数
        
        参数:
            data: 包含所有变量的数据框
            x, y: 目标变量名
            control_vars: 控制变量名列表
            
        返回:
            包含偏相关系数的字典
        """
        try:
            # 确保所有变量存在
            all_vars = [x, y] + control_vars
            available_vars = [var for var in all_vars if var in data.columns]
            
            if len(available_vars) < 3:
                return {
                    'partial_correlation': np.nan,
                    'p_value': 1.0,
                    'significant': False,
                    'method': 'partial_correlation',
                    'error': '变量不足'
                }
            
            # 选择数据
            subset_data = data[available_vars].dropna()
            
            if len(subset_data) < 4:
                return {
                    'partial_correlation': np.nan,
                    'p_value': 1.0,
                    'significant': False,
                    'method': 'partial_correlation',
                    'error': '样本不足'
                }
            
            # 标准化数据
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(subset_data)
            scaled_df = pd.DataFrame(scaled_data, columns=available_vars)
            
            # 计算协方差矩阵
            cov_matrix = np.cov(scaled_df.T)
            
            # 获取变量索引
            x_idx = available_vars.index(x)
            y_idx = available_vars.index(y)
            control_indices = [available_vars.index(var) for var in control_vars 
                             if var in available_vars]
            
            # 计算偏相关
            if len(control_indices) == 0:
                # 简单相关
                corr, p_val = pearsonr(scaled_df[x], scaled_df[y])
                partial_corr = corr
            else:
                # 偏相关计算
                cov_xx = cov_matrix[x_idx, x_idx]
                cov_yy = cov_matrix[y_idx, y_idx]
                cov_xy = cov_matrix[x_idx, y_idx]
                
                # 控制变量的协方差子矩阵
                control_cov = cov_matrix[np.ix_(control_indices, control_indices)]
                cov_x_control = cov_matrix[x_idx, control_indices]
                cov_y_control = cov_matrix[y_idx, control_indices]
                
                # 计算偏相关
                try:
                    inv_control_cov = np.linalg.inv(control_cov)
                    partial_corr = (cov_xy - cov_x_control @ inv_control_cov @ cov_y_control) / \
                                 (np.sqrt(cov_xx - cov_x_control @ inv_control_cov @ cov_x_control) *
                                  np.sqrt(cov_yy - cov_y_control @ inv_control_cov @ cov_y_control))
                except np.linalg.LinAlgError:
                    # 如果矩阵奇异，使用伪逆
                    inv_control_cov = np.linalg.pinv(control_cov)
                    partial_corr = (cov_xy - cov_x_control @ inv_control_cov @ cov_y_control) / \
                                 (np.sqrt(cov_xx - cov_x_control @ inv_control_cov @ cov_x_control) *
                                  np.sqrt(cov_yy - cov_y_control @ inv_control_cov @ cov_y_control))
            
            # 计算p值（近似）
            n = len(scaled_df)
            t_stat = partial_corr * np.sqrt((n - len(control_indices) - 2) / (1 - partial_corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - len(control_indices) - 2))
            
            return {
                'partial_correlation': partial_corr,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'method': 'partial_correlation',
                'n_samples': n,
                'control_variables': control_indices
            }
            
        except Exception as e:
            logger.error(f"计算偏相关时出错: {e}")
            return {
                'partial_correlation': np.nan,
                'p_value': 1.0,
                'significant': False,
                'method': 'partial_correlation',
                'error': str(e)
            }
    
    def granger_causality(self, x: np.ndarray, y: np.ndarray, 
                         max_lag: Optional[int] = None) -> Dict[str, Any]:
        """
        格兰杰因果检验
        
        参数:
            x, y: 时间序列数据
            max_lag: 最大滞后阶数
            
        返回:
            包含格兰杰因果检验结果的字典
        """
        if max_lag is None:
            max_lag = self.max_lag
        
        try:
            results = {}
            
            for lag in range(1, max_lag + 1):
                # 准备数据
                if len(x) <= lag + 1:
                    continue
                
                # 构建滞后变量
                y_lagged = []
                x_lagged = []
                y_current = []
                
                for i in range(lag, len(y)):
                    y_current.append(y[i])
                    y_lagged.append(y[i-lag:i])
                    x_lagged.append(x[i-lag:i])
                
                y_current = np.array(y_current)
                y_lagged = np.array(y_lagged)
                x_lagged = np.array(x_lagged)
                
                # 无约束模型（包含x的滞后）
                X_unrestricted = np.column_stack([y_lagged, x_lagged])
                
                # 有约束模型（不包含x的滞后）
                X_restricted = y_lagged
                
                # 拟合模型并计算RSS
                try:
                    # 无约束模型
                    beta_unrestricted = np.linalg.lstsq(X_unrestricted, y_current, rcond=None)[0]
                    y_pred_unrestricted = X_unrestricted @ beta_unrestricted
                    rss_unrestricted = np.sum((y_current - y_pred_unrestricted)**2)
                    
                    # 有约束模型
                    beta_restricted = np.linalg.lstsq(X_restricted, y_current, rcond=None)[0]
                    y_pred_restricted = X_restricted @ beta_restricted
                    rss_restricted = np.sum((y_current - y_pred_restricted)**2)
                    
                    # F检验
                    n = len(y_current)
                    k_unrestricted = X_unrestricted.shape[1]
                    k_restricted = X_restricted.shape[1]
                    
                    f_stat = ((rss_restricted - rss_unrestricted) / (k_unrestricted - k_restricted)) / \
                            (rss_unrestricted / (n - k_unrestricted))
                    
                    p_value = 1 - stats.f.cdf(f_stat, k_unrestricted - k_restricted, n - k_unrestricted)
                    
                    results[lag] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'rss_unrestricted': rss_unrestricted,
                        'rss_restricted': rss_restricted
                    }
                    
                except np.linalg.LinAlgError:
                    results[lag] = {
                        'f_statistic': np.nan,
                        'p_value': 1.0,
                        'significant': False,
                        'error': '线性代数错误'
                    }
            
            # 选择最优滞后
            significant_lags = [lag for lag, result in results.items() 
                              if result.get('significant', False)]
            
            return {
                'granger_results': results,
                'significant_lags': significant_lags,
                'causality_detected': len(significant_lags) > 0,
                'method': 'granger_causality'
            }
            
        except Exception as e:
            logger.error(f"格兰杰因果检验时出错: {e}")
            return {
                'granger_results': {},
                'significant_lags': [],
                'causality_detected': False,
                'method': 'granger_causality',
                'error': str(e)
            }
    
    def nonlinear_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        非线性关联检测
        
        参数:
            x, y: 输入数组
            
        返回:
            包含多种非线性关联度量的字典
        """
        try:
            results = {}
            
            # 1. 互信息
            mi_result = self.mutual_information(x, y)
            results['mutual_information'] = mi_result
            
            # 2. 距离相关
            dcor_result = self.distance_correlation(x, y)
            results['distance_correlation'] = dcor_result
            
            # 3. 基于随机森林的特征重要性
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            X = x.reshape(-1, 1)
            rf.fit(X, y)
            feature_importance = rf.feature_importances_[0]
            
            # 计算R²作为非线性关联度量
            y_pred = rf.predict(X)
            r2 = r2_score(y, y_pred)
            
            # 置换检验计算p值
            r2_permuted = []
            for _ in range(self.n_permutations):
                y_permuted = np.random.permutation(y)
                rf_perm = RandomForestRegressor(n_estimators=100, 
                                              random_state=self.random_state)
                rf_perm.fit(X, y_permuted)
                y_pred_perm = rf_perm.predict(X)
                r2_perm = r2_score(y_permuted, y_pred_perm)
                r2_permuted.append(r2_perm)
            
            p_value = np.mean(np.array(r2_permuted) >= r2)
            
            results['random_forest'] = {
                'feature_importance': feature_importance,
                'r2_score': r2,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'method': 'random_forest'
            }
            
            # 4. Maximal Information Coefficient (MIC) 近似计算
            # 这里使用简化的MIC计算
            def compute_mic_simple(x, y, bins=10):
                """简化的MIC计算"""
                # 创建网格
                x_binned = pd.cut(x, bins=bins, labels=False)
                y_binned = pd.cut(y, bins=bins, labels=False)
                
                # 计算联合分布
                joint_counts = np.histogram2d(x_binned, y_binned, bins=bins)[0]
                joint_probs = joint_counts / np.sum(joint_counts)
                
                # 计算边际分布
                x_marginal = np.sum(joint_probs, axis=1)
                y_marginal = np.sum(joint_probs, axis=0)
                
                # 计算互信息
                mi = 0
                for i in range(bins):
                    for j in range(bins):
                        if joint_probs[i, j] > 0:
                            mi += joint_probs[i, j] * np.log(joint_probs[i, j] / 
                                                           (x_marginal[i] * y_marginal[j] + 1e-10))
                
                # 标准化
                h_x = -np.sum(x_marginal * np.log(x_marginal + 1e-10))
                h_y = -np.sum(y_marginal * np.log(y_marginal + 1e-10))
                mic = mi / min(h_x, h_y) if min(h_x, h_y) > 0 else 0
                
                return mic
            
            mic_value = compute_mic_simple(x, y)
            results['mic_approximation'] = {
                'mic_value': mic_value,
                'method': 'mic_approximation'
            }
            
            # 总结非线性关联强度
            nonlinear_measures = [
                abs(mi_result.get('normalized_mi', 0)),
                abs(dcor_result.get('distance_correlation', 0)),
                abs(r2)
            ]
            
            results['nonlinear_strength'] = {
                'mean_strength': np.mean(nonlinear_measures),
                'max_strength': np.max(nonlinear_measures),
                'measures_count': len(nonlinear_measures)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"非线性关联检测时出错: {e}")
            return {
                'error': str(e),
                'method': 'nonlinear_correlation'
            }
    
    def correlation_matrix_analysis(self, data: pd.DataFrame, 
                                  methods: List[str] = None) -> Dict[str, Any]:
        """
        多变量关联矩阵分析
        
        参数:
            data: 输入数据框
            methods: 要使用的方法列表
            
        返回:
            包含关联矩阵的字典
        """
        if methods is None:
            methods = ['pearson', 'spearman', 'kendall']
        
        try:
            results = {}
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) < 2:
                return {
                    'error': '数据中数值列不足',
                    'method': 'correlation_matrix_analysis'
                }
            
            for method in methods:
                correlation_matrix = np.zeros((len(numeric_columns), len(numeric_columns)))
                p_value_matrix = np.ones((len(numeric_columns), len(numeric_columns)))
                
                for i, col1 in enumerate(numeric_columns):
                    for j, col2 in enumerate(numeric_columns):
                        if i == j:
                            correlation_matrix[i, j] = 1.0
                            p_value_matrix[i, j] = 0.0
                        elif i < j:
                            # 去除缺失值
                            valid_data = data[[col1, col2]].dropna()
                            if len(valid_data) > 3:
                                if method == 'pearson':
                                    corr, p_val = pearsonr(valid_data[col1], valid_data[col2])
                                elif method == 'spearman':
                                    corr, p_val = spearmanr(valid_data[col1], valid_data[col2])
                                elif method == 'kendall':
                                    corr, p_val = kendalltau(valid_data[col1], valid_data[col2])
                                else:
                                    corr, p_val = np.nan, 1.0
                                
                                correlation_matrix[i, j] = corr
                                correlation_matrix[j, i] = corr
                                p_value_matrix[i, j] = p_val
                                p_value_matrix[j, i] = p_val
                
                results[method] = {
                    'correlation_matrix': correlation_matrix,
                    'p_value_matrix': p_value_matrix,
                    'variables': list(numeric_columns)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"关联矩阵分析时出错: {e}")
            return {
                'error': str(e),
                'method': 'correlation_matrix_analysis'
            }
    
    def association_strength_ranking(self, correlations: Dict[str, Dict]) -> pd.DataFrame:
        """
        关联强度评估和排序
        
        参数:
            correlations: 关联分析结果字典
            
        返回:
            排序后的关联强度DataFrame
        """
        try:
            ranking_data = []
            
            for pair, results in correlations.items():
                for method, result in results.items():
                    if isinstance(result, dict) and 'correlation' in result:
                        ranking_data.append({
                            'variable_pair': pair,
                            'method': method,
                            'correlation': result['correlation'],
                            'p_value': result.get('p_value', 1.0),
                            'significant': result.get('significant', False),
                            'abs_correlation': abs(result['correlation']),
                            'strength_category': self._categorize_strength(abs(result['correlation']))
                        })
                    elif isinstance(result, dict) and 'mutual_information' in result:
                        ranking_data.append({
                            'variable_pair': pair,
                            'method': method,
                            'correlation': result['mutual_information'],
                            'p_value': result.get('p_value', 1.0),
                            'significant': result.get('significant', False),
                            'abs_correlation': abs(result['mutual_information']),
                            'strength_category': self._categorize_strength(abs(result['mutual_information']))
                        })
                    elif isinstance(result, dict) and 'distance_correlation' in result:
                        ranking_data.append({
                            'variable_pair': pair,
                            'method': method,
                            'correlation': result['distance_correlation'],
                            'p_value': result.get('p_value', 1.0),
                            'significant': result.get('significant', False),
                            'abs_correlation': abs(result['distance_correlation']),
                            'strength_category': self._categorize_strength(abs(result['distance_correlation']))
                        })
            
            df = pd.DataFrame(ranking_data)
            
            if not df.empty:
                # 按绝对关联强度排序
                df = df.sort_values('abs_correlation', ascending=False)
                df['rank'] = range(1, len(df) + 1)
                
                # 添加统计摘要
                summary = {
                    'total_pairs': len(df),
                    'significant_pairs': len(df[df['significant']]),
                    'mean_correlation': df['correlation'].mean(),
                    'median_correlation': df['correlation'].median(),
                    'strong_correlations': len(df[df['strength_category'] == '强']),
                    'moderate_correlations': len(df[df['strength_category'] == '中']),
                    'weak_correlations': len(df[df['strength_category'] == '弱'])
                }
                
                return df, summary
            else:
                return pd.DataFrame(), {}
                
        except Exception as e:
            logger.error(f"关联强度排序时出错: {e}")
            return pd.DataFrame(), {'error': str(e)}
    
    def _categorize_strength(self, correlation: float) -> str:
        """
        关联强度分类
        
        参数:
            correlation: 相关系数值
            
        返回:
            强度类别字符串
        """
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return '强'
        elif abs_corr >= 0.3:
            return '中'
        else:
            return '弱'
    
    def association_pattern_discovery(self, data: pd.DataFrame, 
                                    min_correlation: float = 0.3) -> Dict[str, Any]:
        """
        关联模式发现和解释
        
        参数:
            data: 输入数据框
            min_correlation: 最小关联阈值
            
        返回:
            包含关联模式的字典
        """
        try:
            patterns = {}
            
            # 1. 强关联对
            corr_results = self.correlation_matrix_analysis(data, ['pearson'])
            if 'pearson' in corr_results:
                corr_matrix = corr_results['pearson']['correlation_matrix']
                variables = corr_results['pearson']['variables']
                
                strong_pairs = []
                for i in range(len(variables)):
                    for j in range(i+1, len(variables)):
                        if abs(corr_matrix[i, j]) >= min_correlation:
                            strong_pairs.append({
                                'var1': variables[i],
                                'var2': variables[j],
                                'correlation': corr_matrix[i, j],
                                'strength': self._categorize_strength(corr_matrix[i, j])
                            })
                
                patterns['strong_pairs'] = strong_pairs
            
            # 2. 变量聚类（基于关联）
            if 'pearson' in corr_results:
                # 使用层次聚类对变量进行分组
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import squareform
                
                # 将相关矩阵转换为距离矩阵
                distance_matrix = 1 - np.abs(corr_matrix)
                
                # 执行层次聚类
                condensed_dist = squareform(distance_matrix)
                linkage_matrix = linkage(condensed_dist, method='average')
                
                # 设定聚类数量
                n_clusters = min(5, len(variables) // 2 + 1)
                clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                # 组织聚类结果
                variable_clusters = {}
                for i, cluster_id in enumerate(clusters):
                    if cluster_id not in variable_clusters:
                        variable_clusters[cluster_id] = []
                    variable_clusters[cluster_id].append(variables[i])
                
                patterns['variable_clusters'] = variable_clusters
            
            # 3. 关联网络分析
            if 'pearson' in corr_results:
                network_stats = self._analyze_association_network(corr_matrix, variables)
                patterns['network_analysis'] = network_stats
            
            # 4. 时间序列模式（如果有时间列）
            time_columns = [col for col in data.columns if 'time' in col.lower() or 
                          'date' in col.lower() or data[col].dtype == 'datetime64[ns]']
            
            if time_columns:
                temporal_patterns = self._discover_temporal_patterns(data, time_columns[0])
                patterns['temporal_patterns'] = temporal_patterns
            
            return patterns
            
        except Exception as e:
            logger.error(f"关联模式发现时出错: {e}")
            return {'error': str(e)}
    
    def _analyze_association_network(self, correlation_matrix: np.ndarray, 
                                   variables: List[str]) -> Dict[str, Any]:
        """
        分析关联网络
        
        参数:
            correlation_matrix: 关联矩阵
            variables: 变量名列表
            
        返回:
            网络分析结果
        """
        try:
            # 计算网络统计量
            n_vars = len(variables)
            
            # 度中心性（每个变量与其他变量的连接数）
            threshold = 0.3
            adjacency_matrix = (np.abs(correlation_matrix) >= threshold).astype(int)
            np.fill_diagonal(adjacency_matrix, 0)  # 移除自环
            
            degree_centrality = np.sum(adjacency_matrix, axis=1)
            
            # 介数中心性（简化计算）
            betweenness_centrality = np.zeros(n_vars)
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    if adjacency_matrix[i, j]:
                        # 简化介数中心性计算
                        paths_through_i = 0
                        for k in range(n_vars):
                            for l in range(k+1, n_vars):
                                if k != i and l != i and adjacency_matrix[k, l]:
                                    # 简化的路径计算
                                    if (adjacency_matrix[k, i] and adjacency_matrix[i, l]) or \
                                       (adjacency_matrix[l, i] and adjacency_matrix[i, k]):
                                        paths_through_i += 1
                        betweenness_centrality[i] += paths_through_i
            
            # 网络密度
            possible_connections = n_vars * (n_vars - 1) / 2
            actual_connections = np.sum(adjacency_matrix) / 2
            network_density = actual_connections / possible_connections if possible_connections > 0 else 0
            
            return {
                'degree_centrality': dict(zip(variables, degree_centrality)),
                'betweenness_centrality': dict(zip(variables, betweenness_centrality)),
                'network_density': network_density,
                'hub_variables': [variables[i] for i, degree in enumerate(degree_centrality) 
                                if degree >= np.percentile(degree_centrality, 80)],
                'isolated_variables': [variables[i] for i, degree in enumerate(degree_centrality) 
                                     if degree == 0]
            }
            
        except Exception as e:
            logger.error(f"网络分析时出错: {e}")
            return {'error': str(e)}
    
    def _discover_temporal_patterns(self, data: pd.DataFrame, time_column: str) -> Dict[str, Any]:
        """
        发现时间模式
        
        参数:
            data: 数据框
            time_column: 时间列名
            
        返回:
            时间模式分析结果
        """
        try:
            # 确保时间列是datetime类型
            data[time_column] = pd.to_datetime(data[time_column])
            
            # 按时间排序
            data_sorted = data.sort_values(time_column)
            
            # 计算时间间隔统计
            time_diffs = data_sorted[time_column].diff().dropna()
            
            patterns = {
                'time_span': (data_sorted[time_column].max() - data_sorted[time_column].min()).days,
                'mean_interval': time_diffs.mean().total_seconds() / 3600,  # 小时
                'interval_std': time_diffs.std().total_seconds() / 3600,   # 小时
                'data_points': len(data_sorted)
            }
            
            # 检查季节性模式（如果有足够数据）
            if len(data_sorted) > 30:
                data_sorted['hour'] = data_sorted[time_column].dt.hour
                data_sorted['day_of_week'] = data_sorted[time_column].dt.dayofweek
                data_sorted['month'] = data_sorted[time_column].dt.month
                
                numeric_cols = data_sorted.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]  # 使用第一个数值列
                    
                    # 小时模式
                    hourly_pattern = data_sorted.groupby('hour')[col].mean().to_dict()
                    
                    # 星期模式
                    weekly_pattern = data_sorted.groupby('day_of_week')[col].mean().to_dict()
                    
                    patterns.update({
                        'hourly_pattern': hourly_pattern,
                        'weekly_pattern': weekly_pattern
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"时间模式发现时出错: {e}")
            return {'error': str(e)}
    
    def visualize_correlation_heatmap(self, data: pd.DataFrame, 
                                    method: str = 'pearson',
                                    figsize: Tuple[int, int] = (10, 8),
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        关联强度热力图可视化
        
        参数:
            data: 输入数据框
            method: 相关方法
            figsize: 图片大小
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        try:
            corr_results = self.correlation_matrix_analysis(data, [method])
            
            if method not in corr_results:
                raise ValueError(f"方法 {method} 的结果不存在")
            
            correlation_matrix = corr_results[method]['correlation_matrix']
            variables = corr_results[method]['variables']
            
            # 创建热力图
            fig, ax = plt.subplots(figsize=figsize)
            
            # 使用seaborn创建热力图
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       xticklabels=variables,
                       yticklabels=variables,
                       cbar_kws={"shrink": .8},
                       ax=ax)
            
            ax.set_title(f'{method.capitalize()} 相关矩阵热力图', fontsize=16, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"热力图已保存到: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"创建热力图时出错: {e}")
            raise
    
    def visualize_lag_correlation(self, lag_results: Dict[str, Any],
                                figsize: Tuple[int, int] = (12, 6),
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        滞后相关可视化
        
        参数:
            lag_results: 滞后相关分析结果
            figsize: 图片大小
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        try:
            lags = lag_results.get('lags', [])
            correlations = lag_results.get('correlations', [])
            p_values = lag_results.get('p_values', [])
            
            if not lags or not correlations:
                raise ValueError("滞后相关结果数据不足")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
            
            # 上图：滞后相关图
            ax1.plot(lags, correlations, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            
            # 标记显著性点
            significant_indices = [i for i, p in enumerate(p_values) if p < self.alpha]
            if significant_indices:
                sig_lags = [lags[i] for i in significant_indices]
                sig_corrs = [correlations[i] for i in significant_indices]
                ax1.scatter(sig_lags, sig_corrs, color='red', s=50, alpha=0.7, 
                           label=f'显著 (p < {self.alpha})', zorder=5)
                ax1.legend()
            
            ax1.set_xlabel('滞后阶数')
            ax1.set_ylabel('相关系数')
            ax1.set_title('滞后相关分析')
            ax1.grid(True, alpha=0.3)
            
            # 下图：p值图
            ax2.plot(lags, p_values, 'g-', linewidth=2, marker='s', markersize=4)
            ax2.axhline(y=self.alpha, color='r', linestyle='--', alpha=0.7, 
                       label=f'显著性水平 ({self.alpha})')
            ax2.set_xlabel('滞后阶数')
            ax2.set_ylabel('p值')
            ax2.set_title('滞后相关的统计显著性')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"滞后相关图已保存到: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"创建滞后相关图时出错: {e}")
            raise
    
    def generate_correlation_report(self, data: pd.DataFrame, 
                                  output_path: str = 'correlation_report.txt') -> str:
        """
        生成关联分析报告
        
        参数:
            data: 输入数据
            output_path: 报告输出路径
            
        返回:
            报告内容字符串
        """
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("关联分析报告")
            report_lines.append("=" * 60)
            report_lines.append(f"分析时间: {pd.Timestamp.now()}")
            report_lines.append(f"数据维度: {data.shape}")
            report_lines.append(f"数值列: {len(data.select_dtypes(include=[np.number]).columns)}")
            report_lines.append("")
            
            # 1. 基本统计
            report_lines.append("1. 数据基本统计")
            report_lines.append("-" * 30)
            desc = data.describe()
            report_lines.append(desc.to_string())
            report_lines.append("")
            
            # 2. 相关矩阵分析
            report_lines.append("2. 相关矩阵分析")
            report_lines.append("-" * 30)
            corr_results = self.correlation_matrix_analysis(data, ['pearson', 'spearman'])
            
            for method in ['pearson', 'spearman']:
                if method in corr_results:
                    report_lines.append(f"\n{method.capitalize()} 相关矩阵:")
                    corr_matrix = corr_results[method]['correlation_matrix']
                    variables = corr_results[method]['variables']
                    
                    # 找出强相关对
                    strong_correlations = []
                    for i in range(len(variables)):
                        for j in range(i+1, len(variables)):
                            corr_val = corr_matrix[i, j]
                            if abs(corr_val) >= 0.5:
                                strong_correlations.append((variables[i], variables[j], corr_val))
                    
                    if strong_correlations:
                        report_lines.append("强相关对 (|r| >= 0.5):")
                        for var1, var2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
                            report_lines.append(f"  {var1} - {var2}: {corr:.4f}")
                    else:
                        report_lines.append("未发现强相关对")
            
            # 3. 关联模式发现
            report_lines.append("\n3. 关联模式发现")
            report_lines.append("-" * 30)
            patterns = self.association_pattern_discovery(data)
            
            if 'strong_pairs' in patterns:
                report_lines.append(f"强关联对数量: {len(patterns['strong_pairs'])}")
                for pair in patterns['strong_pairs'][:10]:  # 只显示前10个
                    report_lines.append(f"  {pair['var1']} - {pair['var2']}: {pair['correlation']:.4f} ({pair['strength']})")
            
            if 'variable_clusters' in patterns:
                report_lines.append("\n变量聚类:")
                for cluster_id, variables in patterns['variable_clusters'].items():
                    report_lines.append(f"  聚类 {cluster_id}: {', '.join(variables)}")
            
            if 'network_analysis' in patterns:
                network = patterns['network_analysis']
                report_lines.append(f"\n网络密度: {network.get('network_density', 0):.4f}")
                if 'hub_variables' in network:
                    report_lines.append(f"中心变量: {', '.join(network['hub_variables'])}")
            
            # 4. 统计摘要
            report_lines.append("\n4. 统计摘要")
            report_lines.append("-" * 30)
            
            # 计算整体统计
            all_correlations = []
            if 'pearson' in corr_results:
                corr_matrix = corr_results['pearson']['correlation_matrix']
                # 提取上三角矩阵的值（排除对角线）
                n = corr_matrix.shape[0]
                upper_tri_indices = np.triu_indices(n, k=1)
                all_correlations.extend(corr_matrix[upper_tri_indices])
            
            if all_correlations:
                report_lines.append(f"平均绝对相关: {np.mean(np.abs(all_correlations)):.4f}")
                report_lines.append(f"最大绝对相关: {np.max(np.abs(all_correlations)):.4f}")
                report_lines.append(f"强相关对数量: {np.sum(np.abs(all_correlations) >= 0.5)}")
                report_lines.append(f"中等相关对数量: {np.sum((np.abs(all_correlations) >= 0.3) & (np.abs(all_correlations) < 0.5))}")
            
            # 5. 建议
            report_lines.append("\n5. 分析建议")
            report_lines.append("-" * 30)
            
            if all_correlations:
                max_corr = np.max(np.abs(all_correlations))
                if max_corr >= 0.8:
                    report_lines.append("- 发现强多重共线性，建议进行特征选择或主成分分析")
                if np.mean(np.abs(all_correlations)) >= 0.3:
                    report_lines.append("- 变量间普遍存在中等程度相关，考虑降维技术")
                if len([c for c in all_correlations if abs(c) >= 0.7]) > 0:
                    report_lines.append("- 存在高度相关变量，建议检查数据质量")
            
            report_lines.append("- 建议结合领域知识解释相关关系")
            report_lines.append("- 对于时间序列数据，考虑进行滞后相关分析")
            report_lines.append("- 对于非线性关系，考虑使用互信息或距离相关等方法")
            
            report_content = "\n".join(report_lines)
            
            # 保存报告
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"关联分析报告已保存到: {output_path}")
            return report_content
            
        except Exception as e:
            logger.error(f"生成关联分析报告时出错: {e}")
            return f"报告生成失败: {str(e)}"
    
    def comprehensive_analysis(self, data: pd.DataFrame, 
                             target_variable: Optional[str] = None,
                             output_dir: str = './correlation_analysis') -> Dict[str, Any]:
        """
        综合关联分析
        
        参数:
            data: 输入数据
            target_variable: 目标变量名（可选）
            output_dir: 输出目录
            
        返回:
            综合分析结果字典
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            results = {}
            
            # 1. 基本信息
            results['basic_info'] = {
                'data_shape': data.shape,
                'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
                'missing_values': data.isnull().sum().to_dict()
            }
            
            # 2. 相关矩阵分析
            logger.info("执行相关矩阵分析...")
            corr_results = self.correlation_matrix_analysis(data, ['pearson', 'spearman', 'kendall'])
            results['correlation_matrices'] = corr_results
            
            # 3. 关联强度排序
            logger.info("计算关联强度排序...")
            correlations_for_ranking = {}
            
            # 转换相关矩阵为排序所需格式
            for method, result in corr_results.items():
                if 'correlation_matrix' in result:
                    matrix = result['correlation_matrix']
                    variables = result['variables']
                    correlations_for_ranking[method] = {}
                    
                    for i, var1 in enumerate(variables):
                        for j, var2 in enumerate(variables):
                            if i < j:  # 只取上三角
                                pair = f"{var1}_vs_{var2}"
                                correlations_for_ranking[method][pair] = {
                                    method: {
                                        'correlation': matrix[i, j],
                                        'p_value': result.get('p_value_matrix', [[1.0]] * len(variables))[i, j],
                                        'significant': result.get('p_value_matrix', [[1.0]] * len(variables))[i, j] < self.alpha
                                    }
                                }
            
            if correlations_for_ranking:
                ranking_df, summary = self.association_strength_ranking(correlations_for_ranking)
                results['strength_ranking'] = {
                    'ranking_dataframe': ranking_df,
                    'summary_statistics': summary
                }
            
            # 4. 关联模式发现
            logger.info("发现关联模式...")
            patterns = self.association_pattern_discovery(data)
            results['association_patterns'] = patterns
            
            # 5. 目标变量分析（如果有）
            if target_variable and target_variable in data.columns:
                logger.info(f"分析目标变量 {target_variable} 的关联...")
                target_results = {}
                
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                target_data = data[target_variable].dropna()
                
                for col in numeric_cols:
                    if col != target_variable:
                        # 获取共同的有效数据点
                        common_data = data[[target_variable, col]].dropna()
                        if len(common_data) > 10:
                            x = common_data[col].values
                            y = common_data[target_variable].values
                            
                            target_results[col] = {
                                'pearson': self.pearson_correlation(x, y, 'pearson'),
                                'spearman': self.pearson_correlation(x, y, 'spearman'),
                                'mutual_info': self.mutual_information(x, y),
                                'distance_corr': self.distance_correlation(x, y),
                                'nonlinear': self.nonlinear_correlation(x, y)
                            }
                
                results['target_analysis'] = target_results
            
            # 6. 生成可视化
            logger.info("生成可视化图表...")
            
            # 相关矩阵热力图
            if 'pearson' in corr_results:
                heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
                self.visualize_correlation_heatmap(data, 'pearson', save_path=heatmap_path)
                results['visualizations'] = {'heatmap': heatmap_path}
            
            # 7. 生成报告
            logger.info("生成分析报告...")
            report_path = os.path.join(output_dir, 'correlation_analysis_report.txt')
            report_content = self.generate_correlation_report(data, report_path)
            results['report_path'] = report_path
            
            # 保存详细结果
            results_path = os.path.join(output_dir, 'detailed_results.pkl')
            import pickle
            with open(results_path, 'wb') as f:
                # 转换不能序列化的对象
                serializable_results = {}
                for key, value in results.items():
                    if key == 'strength_ranking':
                        if 'ranking_dataframe' in value:
                            serializable_results[key] = {
                                'summary_statistics': value['summary_statistics']
                            }
                        else:
                            serializable_results[key] = value
                    else:
                        serializable_results[key] = value
                
                pickle.dump(serializable_results, f)
            results['results_path'] = results_path
            
            logger.info(f"综合关联分析完成，结果保存在: {output_dir}")
            return results
            
        except Exception as e:
            logger.error(f"综合关联分析时出错: {e}")
            return {'error': str(e)}


# 使用示例和测试函数
def demo_correlation_analyzer():
    """
    关联分析器演示函数
    """
    print("C4 关联分析器演示")
    print("=" * 50)
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 创建具有已知关联的数据
    data = pd.DataFrame({
        'x1': np.random.normal(0, 1, n_samples),
        'x2': np.random.normal(0, 1, n_samples),
        'x3': np.random.normal(0, 1, n_samples),
        'x4': np.random.normal(0, 1, n_samples),
        'x5': np.random.normal(0, 1, n_samples)
    })
    
    # 添加一些已知的关联关系
    data['y1'] = 2 * data['x1'] + 0.5 * data['x2'] + np.random.normal(0, 0.5, n_samples)  # 线性相关
    data['y2'] = data['x3']**2 + np.random.normal(0, 0.3, n_samples)  # 非线性相关
    data['y3'] = np.sin(data['x4']) + np.random.normal(0, 0.2, n_samples)  # 非线性相关
    
    # 创建分析器
    analyzer = CorrelationAnalyzer(alpha=0.05, max_lag=5)
    
    # 演示各种分析方法
    print("1. Pearson相关分析")
    x, y = data['x1'].values, data['y1'].values
    pearson_result = analyzer.pearson_correlation(x, y, 'pearson')
    print(f"   结果: {pearson_result}")
    
    print("\n2. 互信息分析")
    mi_result = analyzer.mutual_information(x, y)
    print(f"   结果: {mi_result}")
    
    print("\n3. 距离相关分析")
    dcor_result = analyzer.distance_correlation(x, y)
    print(f"   结果: {dcor_result}")
    
    print("\n4. 滞后相关分析")
    lag_result = analyzer.lag_correlation(data['x1'].values, data['y1'].values)
    print(f"   最优滞后: {lag_result.get('optimal_lag', 'N/A')}")
    print(f"   最大相关: {lag_result.get('max_correlation', 'N/A')}")
    
    print("\n5. 偏相关分析")
    partial_result = analyzer.partial_correlation(data, 'y1', 'x1', ['x2'])
    print(f"   结果: {partial_result}")
    
    print("\n6. 非线性关联分析")
    nonlinear_result = analyzer.nonlinear_correlation(data['x3'].values, data['y2'].values)
    print(f"   结果: {nonlinear_result}")
    
    print("\n7. 综合分析")
    results = analyzer.comprehensive_analysis(data, target_variable='y1', output_dir='./demo_results')
    print(f"   分析完成，结果保存在: ./demo_results")
    
    print("\n演示完成！")


if __name__ == "__main__":
    demo_correlation_analyzer()