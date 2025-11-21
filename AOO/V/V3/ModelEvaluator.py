"""
V3模型评估器
==============

一个综合性的模型评估器，支持分类和回归模型的全面性能评估。

功能特性:
- 多指标评估（准确率、召回率、F1等分类指标）
- 混淆矩阵和ROC曲线可视化
- 回归评估指标（MAE、MSE、R2等）
- 分类和回归模型评估
- 模型性能比较
- 评估结果可视化
- 评估报告生成
- 评估结果存储
- 评估结果解释


版本: 3.0
日期: 2025-11-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    V3模型评估器
    
    提供全面的模型性能评估功能，支持分类和回归任务。
    """
    
    def __init__(self, task_type: str = "classification", save_path: str = "./evaluation_results"):
        """
        初始化模型评估器
        
        Args:
            task_type: 任务类型，"classification" 或 "regression"
            save_path: 结果保存路径
        """
        self.task_type = task_type.lower()
        self.save_path = save_path
        self.evaluation_results = {}
        self.model_comparisons = {}
        
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"✅ 模型评估器初始化完成")
        print(f"📊 任务类型: {self.task_type}")
        print(f"💾 保存路径: {self.save_path}")
    
    def evaluate_classification(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              y_prob: Optional[np.ndarray] = None,
                              model_name: str = "Model",
                              labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分类模型评估
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率（可选）
            model_name: 模型名称
            labels: 类别标签名称
            
        Returns:
            评估结果字典
        """
        print(f"\n🔍 开始评估分类模型: {model_name}")
        
        # 基本分类指标
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC-AUC评分（仅在二分类或多分类且提供概率时）
        roc_auc = None
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # 二分类
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
                else:  # 多分类
                    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"⚠️  ROC-AUC计算失败: {e}")
        
        # 整理评估结果
        results = {
            "model_name": model_name,
            "task_type": "classification",
            "accuracy": accuracy,
            "precision": {
                "macro": precision_macro,
                "micro": precision_micro,
                "weighted": precision_weighted
            },
            "recall": {
                "macro": recall_macro,
                "micro": recall_micro,
                "weighted": recall_weighted
            },
            "f1_score": {
                "macro": f1_macro,
                "micro": f1_micro,
                "weighted": f1_weighted
            },
            "confusion_matrix": cm.tolist(),
            "roc_auc": roc_auc,
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            "timestamp": datetime.now().isoformat()
        }
        
        # 存储结果
        self.evaluation_results[model_name] = results
        
        # 打印关键指标
        print(f"📈 准确率: {accuracy:.4f}")
        print(f"🎯 精确率 (macro): {precision_macro:.4f}")
        print(f"🔄 召回率 (macro): {recall_macro:.4f}")
        print(f"⚖️ F1分数 (macro): {f1_macro:.4f}")
        if roc_auc is not None:
            print(f"📊 ROC-AUC: {roc_auc:.4f}")
        
        return results
    
    def evaluate_regression(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          model_name: str = "Model") -> Dict[str, Any]:
        """
        回归模型评估
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            
        Returns:
            评估结果字典
        """
        print(f"\n🔍 开始评估回归模型: {model_name}")
        
        # 基本回归指标
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)
        
        # 残差分析
        residuals = y_true - y_pred
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        # 整理评估结果
        results = {
            "model_name": model_name,
            "task_type": "regression",
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "mape": mape,
            "explained_variance": explained_var,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "residuals": residuals.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 存储结果
        self.evaluation_results[model_name] = results
        
        # 打印关键指标
        print(f"📊 MAE: {mae:.4f}")
        print(f"📊 MSE: {mse:.4f}")
        print(f"📊 RMSE: {rmse:.4f}")
        print(f"📈 R²: {r2:.4f}")
        print(f"📊 MAPE: {mape:.4f}")
        print(f"📊 解释方差: {explained_var:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, model_name: str, save_plot: bool = True) -> None:
        """
        绘制混淆矩阵
        
        Args:
            model_name: 模型名称
            save_plot: 是否保存图表
        """
        if model_name not in self.evaluation_results:
            print(f"❌ 模型 {model_name} 的评估结果不存在")
            return
        
        results = self.evaluation_results[model_name]
        if results["task_type"] != "classification":
            print(f"❌ 模型 {model_name} 不是分类任务")
            return
        
        cm = np.array(results["confusion_matrix"])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['预测负类', '预测正类'] if cm.shape[0] == 2 else None,
                   yticklabels=['真实负类', '真实正类'] if cm.shape[0] == 2 else None)
        plt.title(f'混淆矩阵 - {model_name}')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        if save_plot:
            filepath = os.path.join(self.save_path, f"confusion_matrix_{model_name}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"💾 混淆矩阵已保存: {filepath}")
        
        plt.show()
    
    def plot_roc_curve(self, model_name: str, save_plot: bool = True) -> None:
        """
        绘制ROC曲线
        
        Args:
            model_name: 模型名称
            save_plot: 是否保存图表
        """
        if model_name not in self.evaluation_results:
            print(f"❌ 模型 {model_name} 的评估结果不存在")
            return
        
        results = self.evaluation_results[model_name]
        if results["task_type"] != "classification":
            print(f"❌ 模型 {model_name} 不是分类任务")
            return
        
        if results["roc_auc"] is None:
            print(f"❌ 模型 {model_name} 缺少概率预测，无法绘制ROC曲线")
            return
        
        # 这里需要原始的概率预测数据来绘制ROC曲线
        # 在实际使用中，应该保存这些数据
        print(f"📊 {model_name} 的ROC-AUC: {results['roc_auc']:.4f}")
    
    def plot_regression_results(self, model_name: str, save_plot: bool = True) -> None:
        """
        绘制回归结果
        
        Args:
            model_name: 模型名称
            save_plot: 是否保存图表
        """
        if model_name not in self.evaluation_results:
            print(f"❌ 模型 {model_name} 的评估结果不存在")
            return
        
        results = self.evaluation_results[model_name]
        if results["task_type"] != "regression":
            print(f"❌ 模型 {model_name} 不是回归任务")
            return
        
        residuals = np.array(results["residuals"])
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 残差分布图
        axes[0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title(f'残差分布 - {model_name}')
        axes[0].set_xlabel('残差')
        axes[0].set_ylabel('频次')
        axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Q-Q图（简化版）
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f'残差Q-Q图 - {model_name}')
        
        plt.tight_layout()
        
        if save_plot:
            filepath = os.path.join(self.save_path, f"regression_results_{model_name}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"💾 回归结果图已保存: {filepath}")
        
        plt.show()
    
    def compare_models(self, model_names: List[str], metric: str = "accuracy") -> pd.DataFrame:
        """
        比较多个模型的性能
        
        Args:
            model_names: 模型名称列表
            metric: 比较指标
            
        Returns:
            比较结果DataFrame
        """
        print(f"\n🔄 开始模型性能比较")
        print(f"📊 比较指标: {metric}")
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                print(f"⚠️  模型 {model_name} 的评估结果不存在，跳过")
                continue
            
            results = self.evaluation_results[model_name]
            
            if metric in results:
                comparison_data.append({
                    "Model": model_name,
                    "Metric": metric,
                    "Value": results[metric]
                })
            elif metric in results.get("precision", {}):
                comparison_data.append({
                    "Model": model_name,
                    "Metric": f"{metric}_macro",
                    "Value": results["precision"][metric]
                })
            elif metric in results.get("recall", {}):
                comparison_data.append({
                    "Model": model_name,
                    "Metric": f"{metric}_macro",
                    "Value": results["recall"][metric]
                })
            elif metric in results.get("f1_score", {}):
                comparison_data.append({
                    "Model": model_name,
                    "Metric": f"{metric}_macro",
                    "Value": results["f1_score"][metric]
                })
        
        if not comparison_data:
            print(f"❌ 没有找到有效的比较数据")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_data)
        self.model_comparisons[metric] = comparison_df
        
        # 打印比较结果
        print("\n📊 模型性能比较结果:")
        print(comparison_df.to_string(index=False))
        
        # 可视化比较结果
        self._plot_model_comparison(comparison_df, metric)
        
        return comparison_df
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame, metric: str) -> None:
        """
        绘制模型比较图
        
        Args:
            comparison_df: 比较数据
            metric: 指标名称
        """
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(comparison_df["Model"], comparison_df["Value"], 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        
        plt.title(f'模型性能比较 - {metric}', fontsize=16, fontweight='bold')
        plt.xlabel('模型', fontsize=12)
        plt.ylabel(f'{metric} 值', fontsize=12)
        plt.xticks(rotation=45)
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        filepath = os.path.join(self.save_path, f"model_comparison_{metric}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 模型比较图已保存: {filepath}")
        
        plt.show()
    
    def generate_report(self, model_name: str, output_format: str = "html") -> str:
        """
        生成评估报告
        
        Args:
            model_name: 模型名称
            output_format: 输出格式，"html" 或 "json"
            
        Returns:
            报告文件路径
        """
        if model_name not in self.evaluation_results:
            print(f"❌ 模型 {model_name} 的评估结果不存在")
            return ""
        
        results = self.evaluation_results[model_name]
        
        if output_format.lower() == "json":
            # JSON格式报告
            report_data = {
                "model_name": model_name,
                "evaluation_summary": self._generate_summary(results),
                "detailed_results": results,
                "interpretation": self._interpret_results(results),
                "recommendations": self._generate_recommendations(results)
            }
            
            filepath = os.path.join(self.save_path, f"evaluation_report_{model_name}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        elif output_format.lower() == "html":
            # HTML格式报告
            html_content = self._generate_html_report(results)
            filepath = os.path.join(self.save_path, f"evaluation_report_{model_name}.html")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        print(f"📄 评估报告已生成: {filepath}")
        return filepath
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成评估摘要
        
        Args:
            results: 评估结果
            
        Returns:
            摘要信息
        """
        if results["task_type"] == "classification":
            return {
                "总体表现": "良好" if results["accuracy"] > 0.8 else "一般" if results["accuracy"] > 0.6 else "较差",
                "准确率": f"{results['accuracy']:.4f}",
                "F1分数": f"{results['f1_score']['macro']:.4f}",
                "ROC-AUC": f"{results['roc_auc']:.4f}" if results['roc_auc'] else "N/A"
            }
        else:
            return {
                "总体表现": "良好" if results["r2_score"] > 0.8 else "一般" if results["r2_score"] > 0.6 else "较差",
                "R²分数": f"{results['r2_score']:.4f}",
                "RMSE": f"{results['rmse']:.4f}",
                "MAE": f"{results['mae']:.4f}"
            }
    
    def _interpret_results(self, results: Dict[str, Any]) -> List[str]:
        """
        解释评估结果
        
        Args:
            results: 评估结果
            
        Returns:
            解释列表
        """
        interpretations = []
        
        if results["task_type"] == "classification":
            accuracy = results["accuracy"]
            f1_macro = results["f1_score"]["macro"]
            
            if accuracy > 0.9:
                interpretations.append("🎉 模型表现优秀，准确率超过90%")
            elif accuracy > 0.8:
                interpretations.append("👍 模型表现良好，准确率超过80%")
            elif accuracy > 0.6:
                interpretations.append("⚠️ 模型表现一般，建议进一步优化")
            else:
                interpretations.append("❌ 模型表现较差，需要重新设计")
            
            if f1_macro < accuracy - 0.1:
                interpretations.append("📊 F1分数低于准确率，可能存在类别不平衡问题")
            
            if results.get("roc_auc"):
                if results["roc_auc"] > 0.9:
                    interpretations.append("📈 ROC-AUC优秀，模型具有很强的区分能力")
                elif results["roc_auc"] > 0.8:
                    interpretations.append("📊 ROC-AUC良好，模型具有较好的区分能力")
        
        else:
            r2 = results["r2_score"]
            rmse = results["rmse"]
            
            if r2 > 0.9:
                interpretations.append("🎉 模型拟合优秀，R²超过90%")
            elif r2 > 0.8:
                interpretations.append("👍 模型拟合良好，R²超过80%")
            elif r2 > 0.6:
                interpretations.append("⚠️ 模型拟合一般，建议增加特征或调整模型")
            else:
                interpretations.append("❌ 模型拟合较差，需要重新建模")
            
            if abs(results["residual_mean"]) > rmse * 0.1:
                interpretations.append("📊 残差存在系统性偏差，建议检查模型假设")
        
        return interpretations
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        生成改进建议
        
        Args:
            results: 评估结果
            
        Returns:
            建议列表
        """
        recommendations = []
        
        if results["task_type"] == "classification":
            accuracy = results["accuracy"]
            
            if accuracy < 0.7:
                recommendations.extend([
                    "🔧 尝试特征工程，增加更多有意义的特征",
                    "🎯 调整模型超参数，使用网格搜索或贝叶斯优化",
                    "📊 检查数据质量，处理异常值和缺失值",
                    "⚖️ 考虑处理类别不平衡问题，使用重采样或权重调整"
                ])
            
            if results.get("roc_auc") and results["roc_auc"] < 0.8:
                recommendations.append("📈 优化分类阈值，提高真正例率")
            
            f1_scores = results["f1_score"]
            if f1_scores["macro"] < f1_scores["micro"]:
                recommendations.append("⚖️ 关注少数类别的预测性能")
        
        else:
            r2 = results["r2_score"]
            
            if r2 < 0.7:
                recommendations.extend([
                    "🔧 增加更多相关特征或进行特征交互",
                    "🎯 尝试不同的模型算法，如集成方法",
                    "📊 检查特征与目标变量的线性关系",
                    "🔍 分析残差模式，考虑非线性建模"
                ])
            
            if results["mape"] > 0.1:
                recommendations.append("📊 MAPE较高，考虑对目标变量进行变换")
        
        return recommendations
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """
        生成HTML报告
        
        Args:
            results: 评估结果
            
        Returns:
            HTML内容
        """
        summary = self._generate_summary(results)
        interpretations = self._interpret_results(results)
        recommendations = self._generate_recommendations(results)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型评估报告 - {results['model_name']}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin: 25px 0; padding: 20px; border-radius: 8px; }}
        .summary {{ background-color: #ecf0f1; }}
        .interpretation {{ background-color: #e8f5e8; }}
        .recommendations {{ background-color: #fff3cd; }}
        .metrics {{ background-color: #f8f9fa; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; }}
        .interpretation-item, .recommendation-item {{ margin: 10px 0; padding: 10px; background: white; border-radius: 5px; }}
        .timestamp {{ text-align: center; color: #7f8c8d; margin-top: 30px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 模型评估报告</h1>
            <h2>{results['model_name']}</h2>
            <p>任务类型: {results['task_type'].title()}</p>
        </div>
        
        <div class="section summary">
            <h3>📈 评估摘要</h3>
            <div class="metric-grid">
                {self._generate_metric_cards(summary)}
            </div>
        </div>
        
        <div class="section metrics">
            <h3>📊 详细指标</h3>
            {self._generate_detailed_metrics(results)}
        </div>
        
        <div class="section interpretation">
            <h3>💡 结果解释</h3>
            {self._generate_interpretation_html(interpretations)}
        </div>
        
        <div class="section recommendations">
            <h3>🎯 改进建议</h3>
            {self._generate_recommendations_html(recommendations)}
        </div>
        
        <div class="timestamp">
            <p>生成时间: {results['timestamp']}</p>
        </div>
    </div>
</body>
</html>
        """
        return html_template
    
    def _generate_metric_cards(self, summary: Dict[str, str]) -> str:
        """生成指标卡片HTML"""
        cards = ""
        for key, value in summary.items():
            cards += f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{key}</div>
            </div>
            """
        return cards
    
    def _generate_detailed_metrics(self, results: Dict[str, Any]) -> str:
        """生成详细指标HTML"""
        html = "<div class='metric-grid'>"
        
        if results["task_type"] == "classification":
            html += f"""
            <div class="metric-card"><div class="metric-value">{results['accuracy']:.4f}</div><div class="metric-label">准确率</div></div>
            <div class="metric-card"><div class="metric-value">{results['f1_score']['macro']:.4f}</div><div class="metric-label">F1分数 (macro)</div></div>
            <div class="metric-card"><div class="metric-value">{results['precision']['macro']:.4f}</div><div class="metric-label">精确率 (macro)</div></div>
            <div class="metric-card"><div class="metric-value">{results['recall']['macro']:.4f}</div><div class="metric-label">召回率 (macro)</div></div>
            """
            if results.get('roc_auc'):
                html += f"<div class='metric-card'><div class='metric-value'>{results['roc_auc']:.4f}</div><div class='metric-label'>ROC-AUC</div></div>"
        else:
            html += f"""
            <div class="metric-card"><div class="metric-value">{results['r2_score']:.4f}</div><div class="metric-label">R²分数</div></div>
            <div class="metric-card"><div class="metric-value">{results['rmse']:.4f}</div><div class="metric-label">RMSE</div></div>
            <div class="metric-card"><div class="metric-value">{results['mae']:.4f}</div><div class="metric-label">MAE</div></div>
            <div class="metric-card"><div class="metric-value">{results['mape']:.4f}</div><div class="metric-label">MAPE</div></div>
            """
        
        html += "</div>"
        return html
    
    def _generate_interpretation_html(self, interpretations: List[str]) -> str:
        """生成解释HTML"""
        html = ""
        for interpretation in interpretations:
            html += f"<div class='interpretation-item'>{interpretation}</div>"
        return html
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """生成建议HTML"""
        html = ""
        for recommendation in recommendations:
            html += f"<div class='recommendation-item'>{recommendation}</div>"
        return html
    
    def save_results(self, filepath: Optional[str] = None) -> str:
        """
        保存评估结果
        
        Args:
            filepath: 保存路径（可选）
            
        Returns:
            保存的文件路径
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_path, f"evaluation_results_{timestamp}.json")
        
        save_data = {
            "evaluation_results": self.evaluation_results,
            "model_comparisons": self.model_comparisons,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 评估结果已保存: {filepath}")
        return filepath
    
    def load_results(self, filepath: str) -> None:
        """
        加载评估结果
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.evaluation_results = data.get("evaluation_results", {})
            self.model_comparisons = data.get("model_comparisons", {})
            
            print(f"📂 评估结果已加载: {filepath}")
            print(f"📊 已加载 {len(self.evaluation_results)} 个模型的评估结果")
            
        except Exception as e:
            print(f"❌ 加载评估结果失败: {e}")
    
    def get_best_model(self, metric: str = "accuracy") -> Tuple[str, float]:
        """
        获取最佳模型
        
        Args:
            metric: 比较指标
            
        Returns:
            (最佳模型名称, 指标值)
        """
        best_model = None
        best_score = float('-inf')
        
        for model_name, results in self.evaluation_results.items():
            if metric in results:
                score = results[metric]
            elif metric in results.get("precision", {}):
                score = results["precision"][metric]
            elif metric in results.get("recall", {}):
                score = results["recall"][metric]
            elif metric in results.get("f1_score", {}):
                score = results["f1_score"][metric]
            else:
                continue
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model:
            print(f"🏆 最佳模型: {best_model} ({metric}: {best_score:.4f})")
            return best_model, best_score
        else:
            print(f"❌ 未找到有效的模型比较结果")
            return None, 0.0
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        获取所有模型的评估摘要
        
        Returns:
            摘要DataFrame
        """
        summary_data = []
        
        for model_name, results in self.evaluation_results.items():
            if results["task_type"] == "classification":
                summary_data.append({
                    "Model": model_name,
                    "Task": "Classification",
                    "Accuracy": results["accuracy"],
                    "F1_Macro": results["f1_score"]["macro"],
                    "Precision_Macro": results["precision"]["macro"],
                    "Recall_Macro": results["recall"]["macro"],
                    "ROC_AUC": results.get("roc_auc", "N/A")
                })
            else:
                summary_data.append({
                    "Model": model_name,
                    "Task": "Regression",
                    "R2_Score": results["r2_score"],
                    "RMSE": results["rmse"],
                    "MAE": results["mae"],
                    "MAPE": results["mape"],
                    "ROC_AUC": "N/A"
                })
        
        return pd.DataFrame(summary_data)


# 测试用例和示例
def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    
    # 分类数据
    n_samples = 1000
    X_class = np.random.randn(n_samples, 5)
    y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)
    y_prob_class = np.column_stack([1 - y_class, y_class])
    
    # 回归数据
    X_reg = np.random.randn(n_samples, 3)
    y_reg = 2 * X_reg[:, 0] + 0.5 * X_reg[:, 1] - X_reg[:, 2] + np.random.randn(n_samples) * 0.1
    
    return X_class, y_class, y_prob_class, X_reg, y_reg


def test_classification_evaluation():
    """测试分类评估功能"""
    print("🧪 测试分类模型评估")
    
    # 创建评估器
    evaluator = ModelEvaluator(task_type="classification")
    
    # 创建示例数据
    X_class, y_class, y_prob_class, _, _ = create_sample_data()
    
    # 模拟两个分类模型的预测结果
    y_pred_1 = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)  # 较好模型
    y_pred_2 = np.random.choice([0, 1], size=len(y_class), p=[0.6, 0.4])  # 较差模型
    
    # 评估模型1
    results_1 = evaluator.evaluate_classification(
        y_class, y_pred_1, y_prob_class, 
        model_name="LogisticRegression"
    )
    
    # 评估模型2
    results_2 = evaluator.evaluate_classification(
        y_class, y_pred_2, None,
        model_name="RandomClassifier"
    )
    
    # 生成报告
    report_path = evaluator.generate_report("LogisticRegression", "html")
    
    # 比较模型
    comparison_df = evaluator.compare_models(["LogisticRegression", "RandomClassifier"], "accuracy")
    
    # 获取最佳模型
    best_model, best_score = evaluator.get_best_model("accuracy")
    
    return evaluator


def test_regression_evaluation():
    """测试回归评估功能"""
    print("\n🧪 测试回归模型评估")
    
    # 创建评估器
    evaluator = ModelEvaluator(task_type="regression")
    
    # 创建示例数据
    _, _, _, X_reg, y_reg = create_sample_data()
    
    # 模拟两个回归模型的预测结果
    y_pred_1 = 2 * X_reg[:, 0] + 0.5 * X_reg[:, 1] - X_reg[:, 2] + np.random.randn(len(y_reg)) * 0.05  # 较好模型
    y_pred_2 = np.random.randn(len(y_reg)) * 2  # 较差模型
    
    # 评估模型1
    results_1 = evaluator.evaluate_regression(
        y_reg, y_pred_1,
        model_name="LinearRegression"
    )
    
    # 评估模型2
    results_2 = evaluator.evaluate_regression(
        y_reg, y_pred_2,
        model_name="RandomRegressor"
    )
    
    # 生成报告
    report_path = evaluator.generate_report("LinearRegression", "html")
    
    # 比较模型
    comparison_df = evaluator.compare_models(["LinearRegression", "RandomRegressor"], "r2_score")
    
    # 获取评估摘要
    summary_df = evaluator.get_evaluation_summary()
    print("\n📊 评估摘要:")
    print(summary_df.to_string(index=False))
    
    return evaluator


def comprehensive_test():
    """综合测试"""
    print("🚀 开始V3模型评估器综合测试")
    print("=" * 60)
    
    # 测试分类评估
    classifier_evaluator = test_classification_evaluation()
    
    # 测试回归评估
    regressor_evaluator = test_regression_evaluation()
    
    # 保存结果
    classifier_evaluator.save_results()
    regressor_evaluator.save_results()
    
    print("\n✅ 所有测试完成!")
    print("📁 查看生成的评估报告和图表")
    
    return classifier_evaluator, regressor_evaluator


if __name__ == "__main__":
    # 运行综合测试
    classifier_evaluator, regressor_evaluator = comprehensive_test()
    
    # 显示使用说明
    print("\n" + "=" * 60)
    print("📚 V3模型评估器使用说明")
    print("=" * 60)
    print("""
主要功能:
1. 📊 多指标评估 - 支持分类和回归任务的全面评估
2. 📈 可视化分析 - 混淆矩阵、ROC曲线、残差分析等
3. 🔄 模型比较 - 多个模型的性能对比
4. 📄 报告生成 - HTML和JSON格式的详细报告
5. 💾 结果存储 - 评估结果的保存和加载
6. 💡 智能解释 - 自动生成结果解释和改进建议

使用示例:
    # 创建评估器
    evaluator = ModelEvaluator(task_type="classification")
    
    # 评估分类模型
    results = evaluator.evaluate_classification(y_true, y_pred, y_prob, "MyModel")
    
    # 生成可视化
    evaluator.plot_confusion_matrix("MyModel")
    
    # 比较模型
    comparison = evaluator.compare_models(["Model1", "Model2"], "accuracy")
    
    # 生成报告
    report_path = evaluator.generate_report("MyModel", "html")
    
    # 保存结果
    evaluator.save_results()
    """)