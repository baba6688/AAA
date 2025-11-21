#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宏观经济分析器
================

实现宏观经济分析功能，包括：
1. 经济周期识别和分析
2. 宏观经济指标相关性分析
3. 政策影响评估
4. 经济风险预警
5. 跨市场影响分析
6. 经济数据预测模型

作者: AI量化分析系统
日期: 2025-11-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import time
import datetime
import json
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MacroEconomicAnalyzer:
    """
    宏观经济分析器
    
    提供全面的宏观经济分析功能，包括时间序列分析、计量经济学建模、
    风险评估和预测等功能。
    """
    
    def __init__(self, data=None):
        """
        初始化分析器
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            宏观经济数据，包含日期索引和多个经济指标
        """
        self.data = data
        self.processed_data = None
        self.cycle_data = None
        self.correlation_matrix = None
        self.risk_metrics = {}
        self.prediction_models = {}
        self.policy_impact_results = {}
        
    def load_data(self, data):
        """
        加载宏观经济数据
        
        Parameters:
        -----------
        data : pd.DataFrame
            宏观经济数据，索引为日期，列为各种经济指标
        """
        self.data = data.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        print(f"已加载数据，包含 {len(data.columns)} 个指标，{len(data)} 个观测值")
        
    def preprocess_data(self, method='standardize', handle_missing='forward_fill'):
        """
        数据预处理
        
        Parameters:
        -----------
        method : str
            标准化方法：'standardize', 'minmax', 'log', 'diff'
        handle_missing : str
            缺失值处理方法：'forward_fill', 'backward_fill', 'interpolate'
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        processed_data = self.data.copy()
        
        # 处理缺失值
        if handle_missing == 'forward_fill':
            processed_data = processed_data.fillna(method='ffill')
        elif handle_missing == 'backward_fill':
            processed_data = processed_data.fillna(method='bfill')
        elif handle_missing == 'interpolate':
            processed_data = processed_data.interpolate()
            
        # 标准化处理
        if method == 'standardize':
            scaler = StandardScaler()
            processed_data = pd.DataFrame(
                scaler.fit_transform(processed_data),
                columns=processed_data.columns,
                index=processed_data.index
            )
        elif method == 'minmax':
            scaler = MinMaxScaler()
            processed_data = pd.DataFrame(
                scaler.fit_transform(processed_data),
                columns=processed_data.columns,
                index=processed_data.index
            )
        elif method == 'log':
            processed_data = np.log(processed_data + 1)
        elif method == 'diff':
            processed_data = processed_data.diff().dropna()
            
        self.processed_data = processed_data
        print(f"数据预处理完成，使用方法: {method}")
        return processed_data
    
    def identify_business_cycles(self, method='hp_filter', smoothing=1600):
        """
        经济周期识别和分析
        
        Parameters:
        -----------
        method : str
            滤波方法：'hp_filter', 'baxter_king', 'christiano_fitzgerald'
        smoothing : float
            平滑参数，HP滤波的λ参数
            
        Returns:
        --------
        dict : 包含周期成分和趋势成分
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        cycle_results = {}
        
        for column in self.processed_data.columns:
            series = self.processed_data[column].dropna()
            
            if method == 'hp_filter':
                # HP滤波
                cycle, trend = hpfilter(series, lamb=smoothing)
            elif method == 'baxter_king':
                # Baxter-King滤波
                cycle = signal.detrend(series)
                trend = series - cycle
            else:
                # 简单移动平均
                trend = series.rolling(window=12).mean()
                cycle = series - trend
                
            cycle_results[column] = {
                'original': series,
                'trend': trend,
                'cycle': cycle,
                'cycle_amplitude': np.std(cycle),
                'cycle_frequency': len(cycle[cycle > 0]) / len(cycle)
            }
            
        self.cycle_data = cycle_results
        
        # 识别经济周期阶段
        self._identify_cycle_phases()
        
        print(f"经济周期分析完成，使用{method}方法")
        return cycle_results
    
    def _identify_cycle_phases(self):
        """识别经济周期阶段"""
        if self.cycle_data is None:
            return
            
        for column, data in self.cycle_data.items():
            cycle = data['cycle']
            
            # 确保cycle是pandas Series
            if isinstance(cycle, np.ndarray):
                cycle = pd.Series(cycle, index=data['original'].index)
            
            # 计算周期阶段
            phases = []
            for i in range(len(cycle)):
                if cycle.iloc[i] > 0:
                    if i > 0 and cycle.iloc[i-1] <= 0:
                        phases.append('expansion_start')
                    else:
                        phases.append('expansion')
                else:
                    if i > 0 and cycle.iloc[i-1] > 0:
                        phases.append('contraction_start')
                    else:
                        phases.append('contraction')
                        
            self.cycle_data[column]['phases'] = phases
            
    def analyze_correlations(self, method='pearson', lag_analysis=True, max_lags=12):
        """
        宏观经济指标相关性分析
        
        Parameters:
        -----------
        method : str
            相关性方法：'pearson', 'spearman', 'kendall'
        lag_analysis : bool
            是否进行滞后相关性分析
        max_lags : int
            最大滞后阶数
            
        Returns:
        --------
        dict : 相关性分析结果
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        # 基本相关性分析
        correlation_matrix = self.processed_data.corr(method=method)
        self.correlation_matrix = correlation_matrix
        
        results = {
            'correlation_matrix': correlation_matrix,
            'strong_correlations': [],
            'weak_correlations': []
        }
        
        # 识别强相关和弱相关
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    results['strong_correlations'].append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
                elif abs(corr_value) < 0.3:
                    results['weak_correlations'].append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
                    
        # 滞后相关性分析
        if lag_analysis:
            lag_correlations = {}
            for col1 in self.processed_data.columns:
                for col2 in self.processed_data.columns:
                    if col1 != col2:
                        lags = []
                        for lag in range(1, max_lags + 1):
                            corr = self.processed_data[col1].corr(
                                self.processed_data[col2].shift(lag)
                            )
                            lags.append(corr)
                        lag_correlations[f"{col1}_vs_{col2}"] = lags
                        
            results['lag_correlations'] = lag_correlations
            
        print(f"相关性分析完成，发现 {len(results['strong_correlations'])} 个强相关关系")
        return results
    
    def assess_policy_impact(self, policy_date, target_variables, 
                           control_variables=None, window=24):
        """
        政策影响评估
        
        Parameters:
        -----------
        policy_date : str or datetime
            政策实施日期
        target_variables : list
            目标变量列表
        control_variables : list, optional
            控制变量列表
        window : int
            分析窗口期（月）
            
        Returns:
        --------
        dict : 政策影响评估结果
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        policy_date = pd.to_datetime(policy_date)
        
        # 事件研究法
        pre_period = self.processed_data[
            self.processed_data.index < policy_date
        ].tail(window)
        
        post_period = self.processed_data[
            self.processed_data.index >= policy_date
        ].head(window)
        
        impact_results = {}
        
        for var in target_variables:
            if var in self.processed_data.columns:
                # 计算政策前后均值差异
                pre_mean = pre_period[var].mean()
                post_mean = post_period[var].mean()
                change = post_mean - pre_mean
                change_pct = (change / abs(pre_mean)) * 100 if pre_mean != 0 else 0
                
                # t检验
                t_stat, p_value = stats.ttest_ind(
                    pre_period[var].dropna(), 
                    post_period[var].dropna()
                )
                
                # 效应大小 (Cohen's d)
                pooled_std = np.sqrt(
                    (pre_period[var].var() + post_period[var].var()) / 2
                )
                cohens_d = change / pooled_std if pooled_std != 0 else 0
                
                impact_results[var] = {
                    'pre_policy_mean': pre_mean,
                    'post_policy_mean': post_mean,
                    'absolute_change': change,
                    'percentage_change': change_pct,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significance': p_value < 0.05,
                    'effect_size': self._interpret_effect_size(abs(cohens_d))
                }
                
        self.policy_impact_results = impact_results
        print(f"政策影响评估完成，分析了 {len(impact_results)} 个变量")
        return impact_results
    
    def _interpret_effect_size(self, cohens_d):
        """解释效应大小"""
        if cohens_d < 0.2:
            return "小效应"
        elif cohens_d < 0.5:
            return "中等效应"
        elif cohens_d < 0.8:
            return "大效应"
        else:
            return "非常大效应"
    
    def risk_early_warning(self, variables, threshold_percentile=90, 
                          window=12, method='percentile'):
        """
        经济风险预警
        
        Parameters:
        -----------
        variables : list
            要监控的变量列表
        threshold_percentile : float
            阈值百分位数
        window : int
            滚动窗口大小
        method : str
            预警方法：'percentile', 'var', 'extreme_value'
            
        Returns:
        --------
        dict : 风险预警结果
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        risk_results = {}
        
        for var in variables:
            if var in self.processed_data.columns:
                series = self.processed_data[var]
                
                if method == 'percentile':
                    # 基于百分位数的预警
                    threshold = np.percentile(series, threshold_percentile)
                    recent_values = series.tail(window)
                    risk_signals = (recent_values > threshold).sum()
                    
                elif method == 'var':
                    # 基于VaR的预警
                    var_95 = np.percentile(series, 5)
                    recent_values = series.tail(window)
                    risk_signals = (recent_values < var_95).sum()
                    
                else:
                    # 极值预警
                    rolling_mean = series.rolling(window=window).mean()
                    rolling_std = series.rolling(window=window).std()
                    z_scores = (series - rolling_mean) / rolling_std
                    risk_signals = (abs(z_scores) > 2).sum()
                    
                # 计算风险等级
                risk_level = self._calculate_risk_level(risk_signals, window)
                
                risk_results[var] = {
                    'threshold': threshold if method == 'percentile' else None,
                    'risk_signals': risk_signals,
                    'risk_level': risk_level,
                    'current_value': series.iloc[-1],
                    'recent_trend': series.tail(window).mean() - series.head(window).mean(),
                    'volatility': series.rolling(window=window).std().iloc[-1]
                }
                
        self.risk_metrics = risk_results
        print(f"风险预警分析完成，监控 {len(risk_results)} 个变量")
        return risk_results
    
    def _calculate_risk_level(self, signals, window):
        """计算风险等级"""
        signal_ratio = signals / window
        if signal_ratio < 0.1:
            return "低风险"
        elif signal_ratio < 0.3:
            return "中等风险"
        elif signal_ratio < 0.5:
            return "高风险"
        else:
            return "极高风险"
    
    def cross_market_analysis(self, markets_data, method='cointegration'):
        """
        跨市场影响分析
        
        Parameters:
        -----------
        markets_data : dict
            不同市场的数据字典
        method : str
            分析方法：'cointegration', 'granger_causality', 'var'
            
        Returns:
        --------
        dict : 跨市场分析结果
        """
        # 准备数据
        market_data = pd.DataFrame(markets_data)
        
        results = {}
        
        if method == 'cointegration':
            # 协整分析
            results['cointegration_tests'] = {}
            for col1 in market_data.columns:
                for col2 in market_data.columns:
                    if col1 != col2:
                        # Engle-Granger协整检验
                        series1 = market_data[col1].dropna()
                        series2 = market_data[col2].dropna()
                        
                        # 确保数据长度一致
                        min_len = min(len(series1), len(series2))
                        series1 = series1.tail(min_len)
                        series2 = series2.tail(min_len)
                        
                        # 回归分析
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression()
                        reg.fit(series2.values.reshape(-1, 1), series1.values)
                        residuals = series1.values - reg.predict(series2.values.reshape(-1, 1))
                        
                        # ADF检验残差
                        adf_stat, p_value, _, _, critical_values, _ = adfuller(residuals)
                        
                        results['cointegration_tests'][f"{col1}_vs_{col2}"] = {
                            'adf_statistic': adf_stat,
                            'p_value': p_value,
                            'is_cointegrated': p_value < 0.05,
                            'critical_values': critical_values
                        }
                        
        elif method == 'granger_causality':
            # Granger因果检验
            results['granger_causality'] = {}
            for col1 in market_data.columns:
                for col2 in market_data.columns:
                    if col1 != col2:
                        try:
                            # 准备数据
                            data_pair = market_data[[col1, col2]].dropna()
                            
                            # Granger因果检验
                            gc_result = grangercausalitytests(data_pair, maxlag=4, verbose=False)
                            
                            results['granger_causality'][f"{col1}_causes_{col2}"] = {
                                'p_values': {lag: gc_result[lag][0]['ssr_ftest'][1] 
                                           for lag in gc_result.keys()},
                                'significant_lags': [lag for lag in gc_result.keys() 
                                                   if gc_result[lag][0]['ssr_ftest'][1] < 0.05]
                            }
                        except:
                            results['granger_causality'][f"{col1}_causes_{col2}"] = {
                                'error': '检验失败'
                            }
                            
        print(f"跨市场分析完成，使用{method}方法")
        return results
    
    def forecast_economic_data(self, target_variable, forecast_periods=12, 
                             methods=['arima', 'var', 'random_forest']):
        """
        经济数据预测模型
        
        Parameters:
        -----------
        target_variable : str
            目标变量
        forecast_periods : int
            预测期数
        methods : list
            预测方法列表
            
        Returns:
        --------
        dict : 预测结果
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        if target_variable not in self.processed_data.columns:
            raise ValueError(f"变量 {target_variable} 不存在于数据中")
            
        series = self.processed_data[target_variable].dropna()
        forecast_results = {}
        
        for method in methods:
            try:
                if method == 'arima':
                    # ARIMA模型
                    model = ARIMA(series, order=(1, 1, 1))
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=forecast_periods)
                    forecast_results[method] = {
                        'forecast': forecast,
                        'model_summary': str(fitted_model.summary()),
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic
                    }
                    
                elif method == 'var':
                    # VAR模型
                    if len(self.processed_data.columns) > 1:
                        var_data = self.processed_data.dropna()
                        model = VAR(var_data)
                        fitted_model = model.fit(maxlags=4, ic='aic')
                        forecast = fitted_model.forecast(var_data.values, steps=forecast_periods)
                        forecast_results[method] = {
                            'forecast': forecast[:, -1],  # 目标变量的预测值
                            'model_summary': str(fitted_model.summary()),
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic
                        }
                        
                elif method == 'random_forest':
                    # 随机森林模型
                    # 创建滞后特征
                    lags = 3
                    X, y = [], []
                    for i in range(lags, len(series)):
                        X.append(series.iloc[i-lags:i].values)
                        y.append(series.iloc[i])
                        
                    X = np.array(X)
                    y = np.array(y)
                    
                    # 训练模型
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    # 预测
                    forecast = []
                    current_window = series.tail(lags).values
                    for _ in range(forecast_periods):
                        pred = model.predict(current_window.reshape(1, -1))[0]
                        forecast.append(pred)
                        current_window = np.append(current_window[1:], pred)
                        
                    forecast_results[method] = {
                        'forecast': np.array(forecast),
                        'feature_importance': model.feature_importances_,
                        'score': model.score(X, y)
                    }
                    
            except Exception as e:
                forecast_results[method] = {'error': str(e)}
                
        self.prediction_models = forecast_results
        print(f"经济数据预测完成，使用了 {len(methods)} 种方法")
        return forecast_results
    
    def generate_comprehensive_report(self, save_plots=True, output_dir='./'):
        """
        生成综合分析报告
        
        Parameters:
        -----------
        save_plots : bool
            是否保存图表
        output_dir : str
            输出目录
            
        Returns:
        --------
        dict : 综合分析报告
        """
        report = {
            'timestamp': pd.Timestamp.now(),
            'data_summary': self._generate_data_summary(),
            'business_cycles': self._analyze_cycles_summary(),
            'correlations': self._analyze_correlations_summary(),
            'policy_impacts': self._analyze_policy_impacts_summary(),
            'risk_assessment': self._analyze_risk_summary(),
            'cross_market': self._analyze_cross_market_summary(),
            'forecasts': self._analyze_forecasts_summary()
        }
        
        # 生成可视化图表
        if save_plots:
            self._create_visualization_plots(output_dir)
            
        print("综合分析报告生成完成")
        return report
    
    def _generate_data_summary(self):
        """生成数据摘要"""
        if self.data is None:
            return {}
            
        return {
            'variables_count': len(self.data.columns),
            'observations_count': len(self.data),
            'date_range': f"{self.data.index.min()} 到 {self.data.index.max()}",
            'missing_values': self.data.isnull().sum().sum(),
            'data_quality_score': 1 - (self.data.isnull().sum().sum() / 
                                     (len(self.data) * len(self.data.columns)))
        }
    
    def _analyze_cycles_summary(self):
        """生成周期分析摘要"""
        if self.cycle_data is None:
            return {}
            
        summary = {}
        for var, data in self.cycle_data.items():
            summary[var] = {
                'cycle_amplitude': data['cycle_amplitude'],
                'expansion_ratio': data['cycle_frequency'],
                'current_phase': data['phases'][-1] if 'phases' in data else 'unknown'
            }
        return summary
    
    def _analyze_correlations_summary(self):
        """生成相关性分析摘要"""
        if self.correlation_matrix is None:
            return {}
            
        # 找出最强的相关关系
        correlations = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                correlations.append({
                    'variables': f"{self.correlation_matrix.columns[i]} vs {self.correlation_matrix.columns[j]}",
                    'correlation': corr_value
                })
                
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'strongest_positive': correlations[0] if correlations else None,
            'strongest_negative': correlations[-1] if correlations else None,
            'average_correlation': self.correlation_matrix.values[np.triu_indices_from(
                self.correlation_matrix.values, k=1)].mean()
        }
    
    def _analyze_policy_impacts_summary(self):
        """生成政策影响摘要"""
        if not self.policy_impact_results:
            return {}
            
        significant_impacts = {var: result for var, result in self.policy_impact_results.items() 
                             if result['significance']}
        
        return {
            'total_variables_analyzed': len(self.policy_impact_results),
            'significant_impacts': len(significant_impacts),
            'largest_effect': max(significant_impacts.items(), 
                                key=lambda x: abs(x[1]['cohens_d'])) if significant_impacts else None
        }
    
    def _analyze_risk_summary(self):
        """生成风险评估摘要"""
        if not self.risk_metrics:
            return {}
            
        risk_levels = {}
        for var, metrics in self.risk_metrics.items():
            risk_levels[var] = metrics['risk_level']
            
        high_risk_vars = [var for var, level in risk_levels.items() 
                         if level in ['高风险', '极高风险']]
        
        return {
            'total_variables_monitored': len(self.risk_metrics),
            'high_risk_variables': high_risk_vars,
            'average_risk_level': pd.Series(list(risk_levels.values())).mode().iloc[0] if risk_levels else None
        }
    
    def _analyze_cross_market_summary(self):
        """生成跨市场分析摘要"""
        # 这里需要根据实际的跨市场分析结果来生成摘要
        return {
            'analysis_completed': True,
            'note': '跨市场分析摘要需要具体的分析结果'
        }
    
    def _analyze_forecasts_summary(self):
        """生成预测分析摘要"""
        if not self.prediction_models:
            return {}
            
        summary = {}
        for method, result in self.prediction_models.items():
            if 'error' not in result:
                summary[method] = {
                    'model_quality': result.get('aic', 'N/A'),
                    'forecast_available': len(result.get('forecast', [])) > 0
                }
                
        return summary
    
    def _create_visualization_plots(self, output_dir):
        """创建可视化图表"""
        plt.style.use('seaborn-v0_8')
        
        # 1. 经济周期图
        if self.cycle_data:
            self._plot_business_cycles(output_dir)
            
        # 2. 相关性热力图
        if self.correlation_matrix is not None:
            self._plot_correlation_heatmap(output_dir)
            
        # 3. 风险预警图
        if self.risk_metrics:
            self._plot_risk_dashboard(output_dir)
            
        # 4. 预测结果图
        if self.prediction_models:
            self._plot_forecasts(output_dir)
            
    def _plot_business_cycles(self, output_dir):
        """绘制经济周期图"""
        n_vars = min(len(self.cycle_data), 4)  # 最多显示4个变量
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3*n_vars))
        if n_vars == 1:
            axes = [axes]
            
        for i, (var, data) in enumerate(list(self.cycle_data.items())[:n_vars]):
            axes[i].plot(data['original'].index, data['original'], label='原始数据', alpha=0.7)
            axes[i].plot(data['trend'].index, data['trend'], label='趋势', linewidth=2)
            axes[i].plot(data['cycle'].index, data['cycle'], label='周期', linewidth=1)
            axes[i].set_title(f'{var} - 经济周期分析')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f'{output_dir}/business_cycles.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_correlation_heatmap(self, output_dir):
        """绘制相关性热力图"""
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        sns.heatmap(self.correlation_matrix, mask=mask, annot=True, 
                   cmap='RdBu_r', center=0, square=True, fmt='.2f')
        plt.title('宏观经济指标相关性分析')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_risk_dashboard(self, output_dir):
        """绘制风险预警仪表板"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 风险等级分布
        risk_levels = [metrics['risk_level'] for metrics in self.risk_metrics.values()]
        risk_counts = pd.Series(risk_levels).value_counts()
        
        axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('风险等级分布')
        
        # 风险信号数量
        var_names = list(self.risk_metrics.keys())
        signal_counts = [self.risk_metrics[var]['risk_signals'] for var in var_names]
        
        axes[0, 1].bar(range(len(var_names)), signal_counts)
        axes[0, 1].set_xticks(range(len(var_names)))
        axes[0, 1].set_xticklabels(var_names, rotation=45)
        axes[0, 1].set_title('风险信号数量')
        
        # 波动性分析
        volatilities = [self.risk_metrics[var]['volatility'] for var in var_names]
        axes[1, 0].bar(range(len(var_names)), volatilities)
        axes[1, 0].set_xticks(range(len(var_names)))
        axes[1, 0].set_xticklabels(var_names, rotation=45)
        axes[1, 0].set_title('波动性分析')
        
        # 最近趋势
        trends = [self.risk_metrics[var]['recent_trend'] for var in var_names]
        colors = ['red' if t < 0 else 'green' for t in trends]
        axes[1, 1].bar(range(len(var_names)), trends, color=colors)
        axes[1, 1].set_xticks(range(len(var_names)))
        axes[1, 1].set_xticklabels(var_names, rotation=45)
        axes[1, 1].set_title('最近趋势')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/risk_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_forecasts(self, output_dir):
        """绘制预测结果图"""
        if not self.prediction_models:
            return
            
        n_methods = len([m for m in self.prediction_models.keys() if 'error' not in self.prediction_models[m]])
        if n_methods == 0:
            return
            
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4*n_methods))
        if n_methods == 1:
            axes = [axes]
            
        method_idx = 0
        for method, result in self.prediction_models.items():
            if 'error' not in result and 'forecast' in result:
                forecast = result['forecast']
                axes[method_idx].plot(range(len(forecast)), forecast, marker='o')
                axes[method_idx].set_title(f'{method.upper()} 预测结果')
                axes[method_idx].grid(True, alpha=0.3)
                method_idx += 1
                
        plt.tight_layout()
        plt.savefig(f'{output_dir}/forecasts.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_sample_data():
    """创建示例宏观经济数据"""
    dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='M')
    
    np.random.seed(42)
    n_periods = len(dates)
    
    # 生成相关的宏观经济指标
    base_trend = np.linspace(100, 150, n_periods)
    cycle = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 48)  # 4年周期
    noise = np.random.normal(0, 5, n_periods)
    
    data = {
        'GDP': base_trend + cycle + noise,
        'CPI': 100 + 0.1 * base_trend + 0.5 * cycle + np.random.normal(0, 2, n_periods),
        'Unemployment_Rate': 5 + 0.01 * cycle + np.random.normal(0, 0.5, n_periods),
        'Interest_Rate': 3 + 0.02 * cycle + np.random.normal(0, 0.3, n_periods),
        'Exchange_Rate': 6.5 + 0.01 * base_trend + np.random.normal(0, 0.2, n_periods),
        'Stock_Index': 3000 + 0.5 * base_trend + 20 * cycle + np.random.normal(0, 100, n_periods),
        'Trade_Balance': 100 + 0.1 * cycle + np.random.normal(0, 20, n_periods),
        'Money_Supply': 200 + 0.3 * base_trend + np.random.normal(0, 10, n_periods)
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


def main():
    """主函数 - 演示宏观经济分析器的使用"""
    print("=== 宏观经济分析器演示 ===\n")
    
    # 1. 创建示例数据
    print("1. 创建示例宏观经济数据...")
    sample_data = create_sample_data()
    print(f"数据包含 {len(sample_data.columns)} 个指标，{len(sample_data)} 个观测值")
    print(f"指标包括: {', '.join(sample_data.columns)}")
    print()
    
    # 2. 初始化分析器
    print("2. 初始化宏观经济分析器...")
    analyzer = MacroEconomicAnalyzer(sample_data)
    
    # 3. 数据预处理
    print("3. 进行数据预处理...")
    analyzer.preprocess_data(method='standardize')
    
    # 4. 经济周期分析
    print("4. 进行经济周期识别和分析...")
    cycle_results = analyzer.identify_business_cycles(method='hp_filter')
    print(f"识别了 {len(cycle_results)} 个变量的经济周期")
    print()
    
    # 5. 相关性分析
    print("5. 进行宏观经济指标相关性分析...")
    corr_results = analyzer.analyze_correlations(method='pearson')
    print(f"发现 {len(corr_results['strong_correlations'])} 个强相关关系")
    print()
    
    # 6. 政策影响评估
    print("6. 进行政策影响评估...")
    policy_date = '2020-01-01'  # 假设政策实施日期
    target_vars = ['GDP', 'CPI', 'Unemployment_Rate']
    policy_results = analyzer.assess_policy_impact(policy_date, target_vars)
    print(f"评估了 {len(policy_results)} 个变量的政策影响")
    print()
    
    # 7. 风险预警
    print("7. 进行经济风险预警...")
    risk_results = analyzer.risk_early_warning(target_vars, threshold_percentile=90)
    print(f"监控了 {len(risk_results)} 个变量的风险状况")
    print()
    
    # 8. 跨市场分析
    print("8. 进行跨市场影响分析...")
    markets_data = {
        'Stock_Market': sample_data['Stock_Index'],
        'Bond_Yield': sample_data['Interest_Rate'],
        'FX_Rate': sample_data['Exchange_Rate']
    }
    cross_market_results = analyzer.cross_market_analysis(markets_data, method='cointegration')
    print("跨市场分析完成")
    print()
    
    # 9. 经济数据预测
    print("9. 进行经济数据预测...")
    forecast_results = analyzer.forecast_economic_data('GDP', forecast_periods=12, 
                                                     methods=['arima', 'random_forest'])
    print(f"使用了 {len(forecast_results)} 种预测方法")
    print()
    
    # 10. 生成综合报告
    print("10. 生成综合分析报告...")
    report = analyzer.generate_comprehensive_report(save_plots=True, output_dir='./')
    
    # 打印报告摘要
    print("\n=== 分析报告摘要 ===")
    print(f"数据质量评分: {report['data_summary']['data_quality_score']:.2%}")
    print(f"平均相关性: {report['correlations']['average_correlation']:.3f}")
    print(f"政策影响显著变量: {report['policy_impacts']['significant_impacts']} 个")
    print(f"高风险变量: {len(report['risk_assessment']['high_risk_variables'])} 个")
    
    print("\n=== 宏观经济分析器演示完成 ===")


if __name__ == "__main__":
    main()