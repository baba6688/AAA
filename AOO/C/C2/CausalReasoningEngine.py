"""
C2因果推理引擎
实现因果关系识别、建模、推理和评估功能


日期: 2025-11-05
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from enum import Enum
import json
import logging

# 因果推理库
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import chisq, fisherz, kci
    from causallearn.utils.GraphUtils import draw_nx_graph
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    warnings.warn("causal-learn library not available. Some features will be limited.")

try:
    import pgmpy
    from pgmpy.models import BayesianModel
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    warnings.warn("pgmpy library not available. Some features will be limited.")

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    warnings.warn("graphviz library not available. Visualization will be limited.")


class CausalAlgorithm(Enum):
    """支持的因果推理算法"""
    PC = "pc"
    FCI = "fci"
    GES = "ges"
    NOTEARS = "notears"
    LINGAM = "lingam"


@dataclass
class CausalEffect:
    """因果效应数据结构"""
    variable: str
    treatment: str
    outcome: str
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    significance_level: float = 0.05
    method: str = ""


@dataclass
class CausalRelation:
    """因果关系数据结构"""
    cause: str
    effect: str
    strength: float
    confidence: float
    type: str  # "direct", "indirect", "confounding"
    evidence: List[str]


class CausalReasoningEngine:
    """C2因果推理引擎主类"""
    
    def __init__(self, alpha: float = 0.05, method: str = "fisherz"):
        """
        初始化因果推理引擎
        
        参数:
            alpha: 显著性水平
            method: 独立性测试方法 ("fisherz", "chisq", "kci")
        """
        self.alpha = alpha
        self.method = method
        self.data = None
        self.causal_graph = None
        self.adjacency_matrix = None
        self.algorithm_results = {}
        self.causal_effects = {}
        self.logger = self._setup_logger()
        
        # 验证依赖库
        self._check_dependencies()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('CausalReasoningEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _check_dependencies(self):
        """检查依赖库可用性"""
        missing_libs = []
        
        if not CAUSAL_LEARN_AVAILABLE:
            missing_libs.append("causal-learn")
        
        if not PGMPY_AVAILABLE:
            missing_libs.append("pgmpy")
        
        if missing_libs:
            self.logger.warning(f"缺少依赖库: {', '.join(missing_libs)}")
            self.logger.warning("某些功能可能无法使用")
    
    def load_data(self, data: Union[pd.DataFrame, np.ndarray, str]) -> 'CausalReasoningEngine':
        """
        加载数据
        
        参数:
            data: 数据（DataFrame, ndarray或文件路径）
        
        返回:
            self: 支持链式调用
        """
        if isinstance(data, str):
            # 从文件加载
            if data.endswith('.csv'):
                self.data = pd.read_csv(data)
            elif data.endswith('.xlsx'):
                self.data = pd.read_excel(data)
            else:
                raise ValueError("不支持的文件格式")
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("不支持的数据类型")
        
        # 数据预处理
        self.data = self._preprocess_data(self.data)
        
        self.logger.info(f"成功加载数据，形状: {self.data.shape}")
        return self
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 移除缺失值过多的列
        threshold = 0.5  # 缺失值比例阈值
        data = data.dropna(axis=1, thresh=len(data) * (1 - threshold))
        
        # 移除缺失值过多的行
        data = data.dropna(axis=0, thresh=len(data.columns) * (1 - threshold))
        
        # 处理缺失值
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                data[col].fillna(data[col].median(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        return data
    
    def discover_causal_structure(self, 
                                 algorithm: CausalAlgorithm = CausalAlgorithm.PC,
                                 **kwargs) -> 'CausalReasoningEngine':
        """
        发现因果结构
        
        参数:
            algorithm: 因果发现算法
            **kwargs: 算法特定参数
        
        返回:
            self: 支持链式调用
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        self.logger.info(f"开始使用 {algorithm.value} 算法发现因果结构")
        
        if algorithm == CausalAlgorithm.PC:
            self._discover_with_pc(**kwargs)
        elif algorithm == CausalAlgorithm.FCI:
            self._discover_with_fci(**kwargs)
        else:
            raise ValueError(f"不支持的算法: {algorithm.value}")
        
        self.logger.info("因果结构发现完成")
        return self
    
    def _discover_with_pc(self, **kwargs):
        """使用PC算法发现因果结构"""
        if not CAUSAL_LEARN_AVAILABLE:
            self.logger.warning("causal-learn库不可用，使用简化方法")
            self._discover_with_correlation(**kwargs)
            return
        
        # 数据转换为numpy数组
        data_array = self.data.values
        
        # PC算法参数
        alpha = kwargs.get('alpha', self.alpha)
        indep_test = kwargs.get('indep_test', fisherz if self.method == 'fisherz' else chisq)
        
        # 执行PC算法
        cg = pc(data_array, alpha=alpha, indep_test=indep_test)
        
        # 保存结果
        self.algorithm_results['pc'] = cg
        self.causal_graph = cg.G
        self.adjacency_matrix = cg.G.graph
        
        self.logger.info("PC算法执行完成")
    
    def _discover_with_fci(self, **kwargs):
        """使用FCI算法发现因果结构"""
        if not CAUSAL_LEARN_AVAILABLE:
            self.logger.warning("causal-learn库不可用，使用简化方法")
            self._discover_with_correlation(**kwargs)
            return
        
        # 数据转换为numpy数组
        data_array = self.data.values
        
        # FCI算法参数
        alpha = kwargs.get('alpha', self.alpha)
        indep_test = kwargs.get('indep_test', fisherz if self.method == 'fisherz' else chisq)
        
        # 执行FCI算法
        cg = fci(data_array, alpha=alpha, indep_test=indep_test)
        
        # 保存结果
        self.algorithm_results['fci'] = cg
        self.causal_graph = cg.G
        self.adjacency_matrix = cg.G.graph
        
        self.logger.info("FCI算法执行完成")
    
    def _discover_with_correlation(self, **kwargs):
        """使用相关性分析进行简化的因果发现"""
        self.logger.info("使用相关性分析进行简化的因果发现")
        
        variables = self.data.columns.tolist()
        n_vars = len(variables)
        
        # 计算相关性矩阵
        corr_matrix = self.data.corr().values
        
        # 设置阈值
        alpha = kwargs.get('alpha', self.alpha)
        threshold = 0.1  # 相关性阈值
        
        # 构建邻接矩阵（简化版本）
        adjacency = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and abs(corr_matrix[i, j]) > threshold:
                    # 简化的因果方向判断：假设方差大的变量是原因
                    var_i = self.data[variables[i]].var()
                    var_j = self.data[variables[j]].var()
                    if var_i > var_j:
                        adjacency[i, j] = 1  # i -> j
                    else:
                        adjacency[j, i] = 1  # j -> i
        
        # 保存结果
        self.algorithm_results['correlation'] = {
            'correlation_matrix': corr_matrix,
            'threshold': threshold
        }
        self.adjacency_matrix = adjacency
        self.causal_graph = adjacency
        
        self.logger.info(f"简化因果发现完成，发现 {np.sum(adjacency)} 个因果关系")
    
    def estimate_causal_effects(self, 
                               treatment: str, 
                               outcome: str,
                               confounders: Optional[List[str]] = None,
                               method: str = "linear") -> CausalEffect:
        """
        估计因果效应
        
        参数:
            treatment: 处理变量
            outcome: 结果变量
            confounders: 混杂变量
            method: 估计方法 ("linear", "propensity_score", "instrumental_variable")
        
        返回:
            CausalEffect: 因果效应估计结果
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        if treatment not in self.data.columns or outcome not in self.data.columns:
            raise ValueError("处理变量或结果变量不存在于数据中")
        
        self.logger.info(f"估计 {treatment} 对 {outcome} 的因果效应")
        
        if method == "linear":
            effect = self._estimate_linear_effect(treatment, outcome, confounders)
        elif method == "propensity_score":
            effect = self._estimate_propensity_score_effect(treatment, outcome, confounders)
        else:
            raise ValueError(f"不支持的估计方法: {method}")
        
        self.causal_effects[f"{treatment}->{outcome}"] = effect
        return effect
    
    def _estimate_linear_effect(self, treatment: str, outcome: str, confounders: Optional[List[str]]) -> CausalEffect:
        """线性回归估计因果效应"""
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        
        # 构建回归模型
        features = [treatment]
        if confounders:
            features.extend(confounders)
        
        X = self.data[features]
        y = self.data[outcome]
        
        # 拟合模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 计算效应大小
        effect_size = model.coef_[0]
        
        # 计算置信区间和p值
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        
        # 简化计算（实际应该使用更精确的方法）
        se = np.sqrt(mse / len(X))
        t_stat = effect_size / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(X) - len(features) - 1))
        
        ci_lower = effect_size - 1.96 * se
        ci_upper = effect_size + 1.96 * se
        
        return CausalEffect(
            variable=treatment,
            treatment=treatment,
            outcome=outcome,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method="linear_regression"
        )
    
    def _estimate_propensity_score_effect(self, treatment: str, outcome: str, confounders: Optional[List[str]]) -> CausalEffect:
        """倾向评分估计因果效应"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
        
        if not confounders:
            # 如果没有指定混杂变量，使用所有其他变量
            confounders = [col for col in self.data.columns if col not in [treatment, outcome]]
        
        # 计算倾向评分
        X_prop = self.data[confounders]
        y_treatment = self.data[treatment]
        
        prop_model = LogisticRegression()
        prop_model.fit(X_prop, y_treatment)
        propensity_scores = prop_model.predict_proba(X_prop)[:, 1]
        
        # 匹配样本
        treated_idx = self.data[treatment] == 1
        control_idx = self.data[treatment] == 0
        
        treated_data = self.data[treated_idx]
        control_data = self.data[control_idx]
        
        # 简化的匹配算法
        if len(treated_data) > 0 and len(control_data) > 0:
            treated_outcomes = treated_data[outcome].mean()
            control_outcomes = control_data[outcome].mean()
            effect_size = treated_outcomes - control_outcomes
            
            # 估计标准误
            pooled_se = np.sqrt(
                treated_data[outcome].var() / len(treated_data) +
                control_data[outcome].var() / len(control_data)
            )
            
            # 计算置信区间和p值
            from scipy import stats
            t_stat = effect_size / pooled_se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(self.data) - 2))
            
            ci_lower = effect_size - 1.96 * pooled_se
            ci_upper = effect_size + 1.96 * pooled_se
        else:
            effect_size = 0
            ci_lower = ci_upper = 0
            p_value = 1.0
        
        return CausalEffect(
            variable=treatment,
            treatment=treatment,
            outcome=outcome,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method="propensity_score_matching"
        )
    
    def counterfactual_reasoning(self, 
                                treatment: str, 
                                outcome: str,
                                intervention_value: Any,
                                factual_value: Any = None) -> Dict[str, float]:
        """
        反事实推理
        
        参数:
            treatment: 处理变量
            outcome: 结果变量
            intervention_value: 干预值
            factual_value: 事实值（如果为None则使用观测值）
        
        返回:
            Dict: 反事实结果
        """
        if self.causal_graph is None:
            raise ValueError("请先发现因果结构")
        
        self.logger.info(f"对 {treatment} 进行反事实推理")
        
        # 构建结构方程模型
        sem_model = self._build_structural_equation_model()
        
        # 计算反事实结果
        counterfactual_result = self._compute_counterfactual(
            sem_model, treatment, outcome, intervention_value, factual_value
        )
        
        return counterfactual_result
    
    def _build_structural_equation_model(self) -> Dict[str, Any]:
        """构建结构方程模型"""
        sem_model = {}
        
        # 基于因果图构建SEM
        if self.adjacency_matrix is not None:
            variables = self.data.columns.tolist()
            
            for i, outcome_var in enumerate(variables):
                # 找到父节点（直接原因）
                parents = []
                for j, cause_var in enumerate(variables):
                    if self.adjacency_matrix[j, i] != 0:  # 存在因果关系
                        parents.append(cause_var)
                
                # 构建方程
                if parents:
                    # 多元回归
                    X = self.data[parents]
                    y = self.data[outcome_var]
                    
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    sem_model[outcome_var] = {
                        'type': 'linear',
                        'parents': parents,
                        'coefficients': model.coef_,
                        'intercept': model.intercept_,
                        'residual_var': np.var(y - model.predict(X))
                    }
                else:
                    # 仅截距模型
                    sem_model[outcome_var] = {
                        'type': 'constant',
                        'mean': self.data[outcome_var].mean(),
                        'var': self.data[outcome_var].var()
                    }
        
        return sem_model
    
    def _compute_counterfactual(self, 
                               sem_model: Dict[str, Any], 
                               treatment: str, 
                               outcome: str,
                               intervention_value: Any,
                               factual_value: Any) -> Dict[str, float]:
        """计算反事实结果"""
        # 简化实现：假设线性关系
        if treatment in sem_model and outcome in sem_model:
            treatment_model = sem_model[treatment]
            outcome_model = sem_model[outcome]
            
            # 如果outcome依赖于treatment
            if treatment in outcome_model.get('parents', []):
                treatment_idx = outcome_model['parents'].index(treatment)
                coefficient = outcome_model['coefficients'][treatment_idx]
                
                # 计算反事实结果
                if factual_value is not None:
                    factual_outcome = outcome_model['intercept'] + coefficient * factual_value
                    counterfactual_outcome = outcome_model['intercept'] + coefficient * intervention_value
                else:
                    # 使用观测数据的平均值
                    factual_outcome = self.data[outcome].mean()
                    counterfactual_outcome = outcome_model['intercept'] + coefficient * intervention_value
                
                return {
                    'factual_outcome': factual_outcome,
                    'counterfactual_outcome': counterfactual_outcome,
                    'causal_effect': counterfactual_outcome - factual_outcome
                }
        
        return {'factual_outcome': 0, 'counterfactual_outcome': 0, 'causal_effect': 0}
    
    def validate_causal_relations(self, 
                                 relations: List[CausalRelation],
                                 validation_method: str = "bootstrap") -> Dict[str, Dict[str, float]]:
        """
        验证因果关系
        
        参数:
            relations: 要验证的因果关系
            validation_method: 验证方法 ("bootstrap", "cross_validation", "holdout")
        
        返回:
            Dict: 验证结果
        """
        self.logger.info(f"使用 {validation_method} 方法验证因果关系")
        
        validation_results = {}
        
        for relation in relations:
            if validation_method == "bootstrap":
                result = self._bootstrap_validation(relation)
            elif validation_method == "cross_validation":
                result = self._cross_validation(relation)
            else:
                result = self._holdout_validation(relation)
            
            validation_results[f"{relation.cause}->{relation.effect}"] = result
        
        return validation_results
    
    def _bootstrap_validation(self, relation: CausalRelation) -> Dict[str, float]:
        """Bootstrap验证"""
        n_bootstrap = 1000
        effects = []
        
        for _ in range(n_bootstrap):
            # 重采样
            bootstrap_indices = np.random.choice(len(self.data), len(self.data), replace=True)
            bootstrap_data = self.data.iloc[bootstrap_indices]
            
            # 计算效应
            effect = self._compute_simple_effect(bootstrap_data, relation.cause, relation.effect)
            effects.append(effect)
        
        # 计算统计量
        effects = np.array(effects)
        mean_effect = np.mean(effects)
        std_effect = np.std(effects)
        ci_lower = np.percentile(effects, 2.5)
        ci_upper = np.percentile(effects, 97.5)
        
        return {
            'mean_effect': mean_effect,
            'std_effect': std_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': not (ci_lower <= 0 <= ci_upper)
        }
    
    def _cross_validation(self, relation: CausalRelation) -> Dict[str, float]:
        """交叉验证"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        effects = []
        
        for train_idx, test_idx in kf.split(self.data):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # 在训练集上拟合模型
            effect = self._compute_simple_effect(train_data, relation.cause, relation.effect)
            effects.append(effect)
        
        mean_effect = np.mean(effects)
        std_effect = np.std(effects)
        
        return {
            'mean_effect': mean_effect,
            'std_effect': std_effect,
            'cv_score': std_effect / abs(mean_effect) if mean_effect != 0 else float('inf')
        }
    
    def _holdout_validation(self, relation: CausalRelation) -> Dict[str, float]:
        """留出验证"""
        from sklearn.model_selection import train_test_split
        
        train_data, test_data = train_test_split(self.data, test_size=0.3, random_state=42)
        
        train_effect = self._compute_simple_effect(train_data, relation.cause, relation.effect)
        test_effect = self._compute_simple_effect(test_data, relation.cause, relation.effect)
        
        return {
            'train_effect': train_effect,
            'test_effect': test_effect,
            'difference': abs(train_effect - test_effect)
        }
    
    def _compute_simple_effect(self, data: pd.DataFrame, cause: str, effect: str) -> float:
        """计算简单因果效应"""
        if cause in data.columns and effect in data.columns:
            correlation = data[cause].corr(data[effect])
            return correlation
        return 0.0
    
    def explain_causal_results(self) -> Dict[str, Any]:
        """
        解释因果推理结果
        
        返回:
            Dict: 解释结果
        """
        explanation = {
            'summary': self._generate_summary(),
            'causal_effects': self._explain_causal_effects(),
            'structural_insights': self._explain_structure(),
            'recommendations': self._generate_recommendations()
        }
        
        return explanation
    
    def _generate_summary(self) -> str:
        """生成总结"""
        if self.adjacency_matrix is not None:
            n_relations = np.sum(self.adjacency_matrix != 0)
            return f"发现 {n_relations} 个因果关系，涉及 {self.data.shape[1]} 个变量"
        return "尚未发现因果结构"
    
    def _explain_causal_effects(self) -> Dict[str, Any]:
        """解释因果效应"""
        explanations = {}
        
        for key, effect in self.causal_effects.items():
            explanations[key] = {
                'effect_size': effect.effect_size,
                'significance': 'significant' if effect.p_value < self.alpha else 'not_significant',
                'confidence_interval': effect.confidence_interval,
                'interpretation': self._interpret_effect_size(effect.effect_size)
            }
        
        return explanations
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """解释效应大小"""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.1:
            return "效应很小"
        elif abs_effect < 0.3:
            return "效应较小"
        elif abs_effect < 0.5:
            return "效应中等"
        else:
            return "效应较大"
    
    def _explain_structure(self) -> Dict[str, Any]:
        """解释结构"""
        if self.adjacency_matrix is None:
            return {'message': '尚未发现因果结构'}
        
        variables = self.data.columns.tolist()
        insights = {
            'total_variables': len(variables),
            'causal_relations': [],
            'key_variables': []
        }
        
        # 分析因果关系
        for i, effect_var in enumerate(variables):
            causes = []
            for j, cause_var in enumerate(variables):
                if self.adjacency_matrix[j, i] != 0:
                    causes.append(cause_var)
            
            if causes:
                insights['causal_relations'].append({
                    'effect': effect_var,
                    'causes': causes
                })
        
        # 识别关键变量（入度和出度较高的变量）
        in_degrees = np.sum(self.adjacency_matrix != 0, axis=0)
        out_degrees = np.sum(self.adjacency_matrix != 0, axis=1)
        
        for i, var in enumerate(variables):
            if in_degrees[i] + out_degrees[i] > np.mean(in_degrees + out_degrees):
                insights['key_variables'].append(var)
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if len(self.causal_effects) == 0:
            recommendations.append("建议先进行因果发现分析")
        else:
            significant_effects = [k for k, v in self.causal_effects.items() 
                                 if v.p_value < self.alpha]
            
            if significant_effects:
                recommendations.append(f"发现 {len(significant_effects)} 个显著因果效应")
            else:
                recommendations.append("未发现显著的因果效应，可能需要更多数据")
        
        if self.data.shape[0] < 100:
            recommendations.append("数据量较小，建议收集更多数据以提高因果发现的可靠性")
        
        return recommendations
    
    def visualize_causal_graph(self, 
                              layout: str = "spring",
                              node_size: int = 1000,
                              font_size: int = 12,
                              save_path: Optional[str] = None) -> None:
        """
        可视化因果图
        
        参数:
            layout: 布局算法 ("spring", "circular", "hierarchical")
            node_size: 节点大小
            font_size: 字体大小
            save_path: 保存路径
        """
        if self.adjacency_matrix is None:
            raise ValueError("请先发现因果结构")
        
        # 创建NetworkX图
        G = nx.DiGraph()
        variables = self.data.columns.tolist()
        G.add_nodes_from(variables)
        
        # 添加边
        for i, effect_var in enumerate(variables):
            for j, cause_var in enumerate(variables):
                if self.adjacency_matrix[j, i] != 0:
                    G.add_edge(cause_var, effect_var)
        
        # 设置布局
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "hierarchical":
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                pos = nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # 绘制图形
        plt.figure(figsize=(12, 8))
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, 
                              node_size=node_size,
                              node_color='lightblue',
                              alpha=0.7)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              alpha=0.6)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos,
                               font_size=font_size,
                               font_weight='bold')
        
        plt.title("因果关系图", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"因果图已保存到: {save_path}")
        
        plt.show()
    
    def visualize_causal_effects(self, save_path: Optional[str] = None) -> None:
        """可视化因果效应"""
        if not self.causal_effects:
            print("没有因果效应数据可可视化")
            return
        
        # 准备数据
        effects = list(self.causal_effects.keys())
        effect_sizes = [self.causal_effects[key].effect_size for key in effects]
        p_values = [self.causal_effects[key].p_value for key in effects]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 效应大小图
        colors = ['red' if p < self.alpha else 'blue' for p in p_values]
        ax1.barh(effects, effect_sizes, color=colors, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('因果效应大小')
        ax1.set_title('因果效应估计')
        ax1.grid(True, alpha=0.3)
        
        # p值图
        ax2.barh(effects, [-np.log10(p) for p in p_values], alpha=0.7)
        ax2.axvline(x=-np.log10(self.alpha), color='red', linestyle='--', 
                   label=f'显著性水平 (α={self.alpha})')
        ax2.set_xlabel('-log10(p值)')
        ax2.set_title('统计显著性')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"因果效应图已保存到: {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str) -> None:
        """
        保存结果到文件
        
        参数:
            filepath: 保存路径
        """
        results = {
            'data_shape': self.data.shape if self.data is not None else None,
            'algorithm_results': {},
            'causal_effects': {},
            'explanation': self.explain_causal_results() if self.data is not None else {}
        }
        
        # 保存算法结果
        for algo, result in self.algorithm_results.items():
            if hasattr(result, 'G') and hasattr(result.G, 'graph'):
                results['algorithm_results'][algo] = {
                    'adjacency_matrix': result.G.graph.tolist()
                }
        
        # 保存因果效应
        for key, effect in self.causal_effects.items():
            results['causal_effects'][key] = {
                'effect_size': effect.effect_size,
                'confidence_interval': effect.confidence_interval,
                'p_value': effect.p_value,
                'method': effect.method
            }
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存到: {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """
        从文件加载结果
        
        参数:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 恢复因果效应
        self.causal_effects = {}
        for key, effect_data in results.get('causal_effects', {}).items():
            self.causal_effects[key] = CausalEffect(
                variable=key,
                treatment=effect_data['treatment'],
                outcome=effect_data['outcome'],
                effect_size=effect_data['effect_size'],
                confidence_interval=tuple(effect_data['confidence_interval']),
                p_value=effect_data['p_value'],
                method=effect_data['method']
            )
        
        self.logger.info(f"结果已从 {filepath} 加载")


# 示例使用代码
def example_usage():
    """示例使用代码"""
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成数据：X -> Y -> Z, X -> W
    X = np.random.normal(0, 1, n_samples)
    Y = 0.5 * X + np.random.normal(0, 0.5, n_samples)
    Z = 0.3 * Y + np.random.normal(0, 0.5, n_samples)
    W = 0.4 * X + np.random.normal(0, 0.5, n_samples)
    
    data = pd.DataFrame({
        'X': X,
        'Y': Y,
        'Z': Z,
        'W': W
    })
    
    # 创建因果推理引擎
    engine = CausalReasoningEngine(alpha=0.05)
    
    # 加载数据
    engine.load_data(data)
    
    # 发现因果结构
    engine.discover_causal_structure(CausalAlgorithm.PC)
    
    # 估计因果效应
    effect_xy = engine.estimate_causal_effects('X', 'Y')
    effect_yz = engine.estimate_causal_effects('Y', 'Z')
    
    # 反事实推理
    counterfactual = engine.counterfactual_reasoning('X', 'Y', 2.0)
    
    # 验证因果关系
    relations = [CausalRelation('X', 'Y', 0.5, 0.9, 'direct', ['correlation', 'temporal_precedence'])]
    validation_results = engine.validate_causal_relations(relations)
    
    # 解释结果
    explanation = engine.explain_causal_results()
    
    # 可视化
    engine.visualize_causal_graph()
    engine.visualize_causal_effects()
    
    # 保存结果
    engine.save_results('causal_results.json')
    
    return engine


if __name__ == "__main__":
    # 运行示例
    engine = example_usage()
    print("因果推理引擎示例运行完成")