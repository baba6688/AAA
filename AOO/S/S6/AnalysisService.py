"""
S6分析服务核心模块
提供数据分析、统计计算、报告生成等功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import psutil
import time

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AnalysisService:
    """S6分析服务核心类"""
    
    def __init__(self):
        """初始化分析服务"""
        self.data = None
        self.analysis_results = {}
        self.report_data = {}
        
    def load_data(self, data: Union[pd.DataFrame, str, dict], **kwargs) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            data: 数据源，可以是DataFrame、文件路径或字典
            **kwargs: 传递给pandas读取函数的参数
            
        Returns:
            加载的DataFrame
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str):
            # 根据文件扩展名自动识别格式
            if data.endswith('.csv'):
                self.data = pd.read_csv(data, **kwargs)
            elif data.endswith('.xlsx') or data.endswith('.xls'):
                self.data = pd.read_excel(data, **kwargs)
            elif data.endswith('.json'):
                self.data = pd.read_json(data, **kwargs)
            else:
                raise ValueError("不支持的文件格式")
        elif isinstance(data, dict):
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("不支持的数据类型")
            
        print(f"数据加载成功，形状: {self.data.shape}")
        return self.data
    
    def data_cleaning(self, drop_duplicates: bool = True, handle_missing: str = 'drop', 
                     missing_threshold: float = 0.5) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            drop_duplicates: 是否删除重复行
            handle_missing: 处理缺失值的方法 ('drop', 'fill_mean', 'fill_median', 'fill_mode')
            missing_threshold: 缺失值比例阈值，超过该比例的列将被删除
            
        Returns:
            清洗后的DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        cleaned_data = self.data.copy()
        original_shape = cleaned_data.shape
        
        # 删除重复行
        if drop_duplicates:
            cleaned_data = cleaned_data.drop_duplicates()
            print(f"删除了 {original_shape[0] - cleaned_data.shape[0]} 个重复行")
        
        # 删除缺失值过多的列
        missing_ratio = cleaned_data.isnull().sum() / len(cleaned_data)
        cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index
        if len(cols_to_drop) > 0:
            cleaned_data = cleaned_data.drop(columns=cols_to_drop)
            print(f"删除了缺失值过多的列: {list(cols_to_drop)}")
        
        # 处理剩余缺失值
        if handle_missing == 'drop':
            cleaned_data = cleaned_data.dropna()
        elif handle_missing == 'fill_mean':
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(
                cleaned_data[numeric_cols].mean())
        elif handle_missing == 'fill_median':
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(
                cleaned_data[numeric_cols].median())
        elif handle_missing == 'fill_mode':
            for col in cleaned_data.columns:
                mode_value = cleaned_data[col].mode()
                if len(mode_value) > 0:
                    cleaned_data[col] = cleaned_data[col].fillna(mode_value[0])
        
        self.data = cleaned_data
        print(f"数据清洗完成，最终形状: {cleaned_data.shape}")
        return self.data
    
    def descriptive_statistics(self) -> Dict[str, Any]:
        """
        描述性统计
        
        Returns:
            包含统计信息的字典
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        stats_dict = {}
        
        # 数值型变量统计
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_dict['numeric'] = {
                'count': len(numeric_cols),
                'columns': list(numeric_cols),
                'mean': self.data[numeric_cols].mean().to_dict(),
                'std': self.data[numeric_cols].std().to_dict(),
                'min': self.data[numeric_cols].min().to_dict(),
                'max': self.data[numeric_cols].max().to_dict(),
                'median': self.data[numeric_cols].median().to_dict(),
                'q25': self.data[numeric_cols].quantile(0.25).to_dict(),
                'q75': self.data[numeric_cols].quantile(0.75).to_dict()
            }
        
        # 分类型变量统计
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            stats_dict['categorical'] = {
                'count': len(categorical_cols),
                'columns': list(categorical_cols),
                'value_counts': {col: self.data[col].value_counts().to_dict() 
                               for col in categorical_cols}
            }
        
        # 数据基本信息
        stats_dict['basic_info'] = {
            'shape': self.data.shape,
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }
        
        self.analysis_results['descriptive_stats'] = stats_dict
        return stats_dict
    
    def correlation_analysis(self, method: str = 'pearson') -> pd.DataFrame:
        """
        相关性分析
        
        Args:
            method: 相关系数计算方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
            相关系数矩阵
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise ValueError("需要至少两个数值型变量进行相关性分析")
            
        correlation_matrix = self.data[numeric_cols].corr(method=method)
        self.analysis_results['correlation'] = correlation_matrix
        
        return correlation_matrix
    
    def outlier_detection(self, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """
        异常值检测
        
        Args:
            method: 检测方法 ('iqr', 'zscore', 'isolation_forest')
            threshold: 阈值参数
            
        Returns:
            异常值检测结果
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = self.data[(self.data[col] < lower_bound) | 
                                   (self.data[col] > upper_bound)]
                outlier_results[col] = {
                    'count': len(outliers),
                    'indices': outliers.index.tolist(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        elif method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                outlier_indices = self.data[col].dropna().index[z_scores > threshold]
                outlier_results[col] = {
                    'count': len(outlier_indices),
                    'indices': outlier_indices.tolist(),
                    'threshold': threshold
                }
        
        elif method == 'isolation_forest':
            if len(numeric_cols) > 0:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(self.data[numeric_cols])
                outlier_indices = self.data.index[outlier_labels == -1]
                outlier_results['isolation_forest'] = {
                    'count': len(outlier_indices),
                    'indices': outlier_indices.tolist()
                }
        
        self.analysis_results['outliers'] = outlier_results
        return outlier_results
    
    def trend_analysis(self, time_col: str, value_col: str, 
                      periods: int = 12) -> Dict[str, Any]:
        """
        时间序列趋势分析
        
        Args:
            time_col: 时间列名
            value_col: 数值列名
            periods: 预测期数
            
        Returns:
            趋势分析结果
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        if time_col not in self.data.columns or value_col not in self.data.columns:
            raise ValueError("指定的时间列或数值列不存在")
            
        # 确保时间列是datetime类型
        time_series = self.data.copy()
        time_series[time_col] = pd.to_datetime(time_series[time_col])
        time_series = time_series.sort_values(time_col)
        
        # 计算移动平均
        time_series['ma_3'] = time_series[value_col].rolling(window=3).mean()
        time_series['ma_6'] = time_series[value_col].rolling(window=6).mean()
        time_series['ma_12'] = time_series[value_col].rolling(window=12).mean()
        
        # 线性趋势
        time_numeric = (time_series[time_col] - time_series[time_col].min()).dt.days
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            time_numeric, time_series[value_col])
        
        # 简单预测（线性回归）
        future_time = time_numeric.max() + np.arange(1, periods + 1)
        future_predictions = slope * future_time + intercept
        
        trend_results = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend_direction': '上升' if slope > 0 else '下降',
            'moving_averages': {
                'ma_3': time_series['ma_3'].dropna().tolist(),
                'ma_6': time_series['ma_6'].dropna().tolist(),
                'ma_12': time_series['ma_12'].dropna().tolist()
            },
            'predictions': future_predictions.tolist()
        }
        
        self.analysis_results['trend'] = trend_results
        return trend_results
    
    def clustering_analysis(self, n_clusters: int = 3, 
                          features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        聚类分析
        
        Args:
            n_clusters: 聚类数量
            features: 特征列名列表，默认为所有数值列
            
        Returns:
            聚类分析结果
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if features is None:
            features = numeric_cols.tolist()
        else:
            features = [col for col in features if col in numeric_cols]
            
        if len(features) == 0:
            raise ValueError("没有可用的数值特征进行聚类分析")
            
        # 数据标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[features])
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # 计算聚类中心
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # 计算聚类评估指标
        silhouette_score = 0  # 简化实现
        
        clustering_results = {
            'n_clusters': n_clusters,
            'features_used': features,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': cluster_centers.tolist(),
            'silhouette_score': silhouette_score,
            'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
        }
        
        self.analysis_results['clustering'] = clustering_results
        return clustering_results
    
    def hypothesis_testing(self, col1: str, col2: str = None, 
                          test_type: str = 'ttest') -> Dict[str, Any]:
        """
        假设检验
        
        Args:
            col1: 第一个变量列名
            col2: 第二个变量列名（可选）
            test_type: 检验类型 ('ttest', 'chisquare', 'anova')
            
        Returns:
            假设检验结果
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        results = {}
        
        if test_type == 'ttest':
            if col2 is None:
                # 单样本t检验
                data1 = self.data[col1].dropna()
                t_stat, p_value = stats.ttest_1samp(data1, 0)
                results = {
                    'test_type': '单样本t检验',
                    'variable': col1,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            else:
                # 双样本t检验
                data1 = self.data[col1].dropna()
                data2 = self.data[col2].dropna()
                t_stat, p_value = stats.ttest_ind(data1, data2)
                results = {
                    'test_type': '双样本t检验',
                    'variable1': col1,
                    'variable2': col2,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        elif test_type == 'chisquare':
            # 卡方检验
            if col2 is None:
                # 拟合优度检验
                observed = self.data[col1].value_counts()
                expected = np.full(len(observed), observed.sum() / len(observed))
                chi2_stat, p_value = stats.chisquare(observed, expected)
            else:
                # 独立性检验
                contingency_table = pd.crosstab(self.data[col1], self.data[col2])
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            results = {
                'test_type': '卡方检验',
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        elif test_type == 'anova':
            # 方差分析
            groups = []
            for group_name in self.data[col2].unique():
                group_data = self.data[self.data[col2] == group_name][col1].dropna()
                groups.append(group_data)
            
            f_stat, p_value = stats.f_oneway(*groups)
            results = {
                'test_type': '方差分析',
                'dependent_var': col1,
                'independent_var': col2,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        self.analysis_results['hypothesis_test'] = results
        return results
    
    def generate_visualizations(self, output_dir: str = 'visualizations') -> List[str]:
        """
        生成可视化图表
        
        Args:
            output_dir: 输出目录
            
        Returns:
            生成的图表文件路径列表
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []
        
        # 1. 数据分布图
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('数据分布分析', fontsize=16)
            
            # 直方图
            self.data[numeric_cols[:4]].hist(bins=20, ax=axes[0, 0])
            axes[0, 0].set_title('数据分布直方图')
            
            # 箱线图
            if len(numeric_cols) >= 4:
                self.data[numeric_cols[:4]].boxplot(ax=axes[0, 1])
                axes[0, 1].set_title('箱线图')
            
            # 相关性热力图
            if len(numeric_cols) > 1:
                corr_matrix = self.data[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
                axes[1, 0].set_title('相关性热力图')
            
            # 散点图（如果有至少两个数值列）
            if len(numeric_cols) >= 2:
                axes[1, 1].scatter(self.data[numeric_cols[0]], self.data[numeric_cols[1]], alpha=0.6)
                axes[1, 1].set_xlabel(numeric_cols[0])
                axes[1, 1].set_ylabel(numeric_cols[1])
                axes[1, 1].set_title('散点图')
            
            plt.tight_layout()
            dist_plot_path = os.path.join(output_dir, 'data_distribution.png')
            plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(dist_plot_path)
        
        # 2. 趋势分析图
        if 'trend' in self.analysis_results:
            trend_data = self.analysis_results['trend']
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 这里需要原始时间序列数据来绘制图表
            # 简化实现
            ax.plot(trend_data.get('predictions', []), label='预测值')
            ax.set_title('趋势分析图')
            ax.legend()
            
            trend_plot_path = os.path.join(output_dir, 'trend_analysis.png')
            plt.savefig(trend_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(trend_plot_path)
        
        # 3. 聚类分析图
        if 'clustering' in self.analysis_results:
            cluster_data = self.analysis_results['clustering']
            if len(cluster_data.get('features_used', [])) >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 简化实现：使用前两个特征进行可视化
                features = cluster_data['features_used'][:2]
                labels = cluster_data['cluster_labels']
                
                scatter = ax.scatter(self.data[features[0]], self.data[features[1]], 
                                   c=labels, cmap='viridis', alpha=0.6)
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_title('聚类分析结果')
                plt.colorbar(scatter)
                
                cluster_plot_path = os.path.join(output_dir, 'clustering_analysis.png')
                plt.savefig(cluster_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(cluster_plot_path)
        
        print(f"生成了 {len(generated_files)} 个可视化图表")
        return generated_files
    
    def generate_report(self, output_file: str = 'analysis_report.md') -> str:
        """
        生成分析报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            报告内容
        """
        report_content = f"""# S6分析服务报告

生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 数据概览

"""
        
        if 'basic_info' in self.analysis_results.get('descriptive_stats', {}):
            basic_info = self.analysis_results['descriptive_stats']['basic_info']
            report_content += f"""
- 数据形状: {basic_info['shape']}
- 内存使用: {basic_info['memory_usage']} 字节
- 缺失值: {sum(basic_info['missing_values'].values())} 个
"""
        
        # 描述性统计
        if 'descriptive_stats' in self.analysis_results:
            stats = self.analysis_results['descriptive_stats']
            report_content += """
## 2. 描述性统计

### 数值型变量统计
"""
            if 'numeric' in stats:
                report_content += f"""
- 数值型变量数量: {stats['numeric']['count']}
- 主要统计量:
  - 均值: {stats['numeric']['mean']}
  - 标准差: {stats['numeric']['std']}
  - 最小值: {stats['numeric']['min']}
  - 最大值: {stats['numeric']['max']}
"""
        
        # 相关性分析
        if 'correlation' in self.analysis_results:
            report_content += """
## 3. 相关性分析

相关性矩阵已生成，详见可视化图表。
"""
        
        # 异常值检测
        if 'outliers' in self.analysis_results:
            outliers = self.analysis_results['outliers']
            report_content += """
## 4. 异常值检测

"""
            for col, result in outliers.items():
                report_content += f"""
- {col}: 检测到 {result['count']} 个异常值
"""
        
        # 趋势分析
        if 'trend' in self.analysis_results:
            trend = self.analysis_results['trend']
            report_content += f"""
## 5. 趋势分析

- 趋势方向: {trend['trend_direction']}
- R²: {trend['r_squared']:.4f}
- P值: {trend['p_value']:.4f}
"""
        
        # 聚类分析
        if 'clustering' in self.analysis_results:
            cluster = self.analysis_results['clustering']
            report_content += f"""
## 6. 聚类分析

- 聚类数量: {cluster['n_clusters']}
- 特征数量: {len(cluster['features_used'])}
- 聚类大小: {cluster['cluster_sizes']}
"""
        
        # 假设检验
        if 'hypothesis_test' in self.analysis_results:
            test = self.analysis_results['hypothesis_test']
            report_content += f"""
## 7. 假设检验

- 检验类型: {test['test_type']}
- P值: {test['p_value']:.4f}
- 是否显著: {'是' if test['significant'] else '否'}
"""
        
        report_content += """
## 8. 建议和结论

基于以上分析结果，建议：

1. 数据质量：检查并处理异常值和缺失值
2. 特征工程：根据相关性分析选择重要特征
3. 进一步分析：根据业务需求进行深度挖掘
4. 模型建立：基于分析结果选择合适的机器学习模型

---
*S6分析服务生成*
"""
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.report_data['content'] = report_content
        print(f"分析报告已生成: {output_file}")
        return report_content
    
    def performance_analysis(self) -> Dict[str, Any]:
        """
        系统性能分析
        
        Returns:
            性能分析结果
        """
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        
        # 数据处理性能
        start_time = time.time()
        if self.data is not None:
            # 模拟一些数据处理操作
            _ = self.data.describe()
            _ = self.data.corr()
        processing_time = time.time() - start_time
        
        performance_results = {
            'cpu_usage': cpu_percent,
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'data_processing_time': processing_time,
            'data_shape': self.data.shape if self.data is not None else None
        }
        
        self.analysis_results['performance'] = performance_results
        return performance_results
    
    def export_results(self, output_dir: str = 'analysis_results') -> Dict[str, str]:
        """
        导出分析结果
        
        Args:
            output_dir: 输出目录
            
        Returns:
            导出的文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        exported_files = {}
        
        # 保存分析结果为JSON
        results_file = os.path.join(output_dir, 'analysis_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
        exported_files['results'] = results_file
        
        # 生成并保存报告
        report_file = os.path.join(output_dir, 'analysis_report.md')
        self.generate_report(report_file)
        exported_files['report'] = report_file
        
        # 生成可视化图表
        viz_dir = os.path.join(output_dir, 'visualizations')
        viz_files = self.generate_visualizations(viz_dir)
        exported_files['visualizations'] = viz_files
        
        # 保存清洗后的数据
        if self.data is not None:
            cleaned_data_file = os.path.join(output_dir, 'cleaned_data.csv')
            self.data.to_csv(cleaned_data_file, index=False, encoding='utf-8')
            exported_files['cleaned_data'] = cleaned_data_file
        
        print(f"分析结果已导出到: {output_dir}")
        return exported_files
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        获取分析摘要
        
        Returns:
            分析摘要信息
        """
        summary = {
            'data_info': {
                'shape': self.data.shape if self.data is not None else None,
                'columns': self.data.columns.tolist() if self.data is not None else [],
                'data_types': self.data.dtypes.to_dict() if self.data is not None else {}
            },
            'analysis_performed': list(self.analysis_results.keys()),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return summary


# 示例使用函数
def create_sample_data():
    """创建示例数据用于测试"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'sales': np.random.normal(1000, 200, n_samples) + np.arange(n_samples) * 2,
        'temperature': np.random.normal(25, 10, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'price': np.random.uniform(10, 100, n_samples),
        'quantity': np.random.poisson(5, n_samples)
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # 示例使用
    print("S6分析服务示例")
    
    # 创建示例数据
    sample_data = create_sample_data()
    
    # 初始化分析服务
    analyzer = AnalysisService()
    
    # 加载数据
    analyzer.load_data(sample_data)
    
    # 数据清洗
    analyzer.data_cleaning()
    
    # 描述性统计
    stats = analyzer.descriptive_statistics()
    print("描述性统计完成")
    
    # 相关性分析
    corr = analyzer.correlation_analysis()
    print("相关性分析完成")
    
    # 异常值检测
    outliers = analyzer.outlier_detection()
    print("异常值检测完成")
    
    # 趋势分析
    trend = analyzer.trend_analysis('date', 'sales')
    print("趋势分析完成")
    
    # 聚类分析
    cluster = analyzer.clustering_analysis(n_clusters=3)
    print("聚类分析完成")
    
    # 假设检验
    test = analyzer.hypothesis_testing('sales', 'category', 'anova')
    print("假设检验完成")
    
    # 生成可视化
    viz_files = analyzer.generate_visualizations()
    print(f"生成了 {len(viz_files)} 个图表")
    
    # 生成报告
    report = analyzer.generate_report('sample_report.md')
    print("报告生成完成")
    
    # 性能分析
    perf = analyzer.performance_analysis()
    print("性能分析完成")
    
    # 导出结果
    exported = analyzer.export_results()
    print("结果导出完成")
    
    print("S6分析服务示例执行完成！")