"""
因子分析器 (Factor Analyzer)

提供完整的因子有效性分析功能，包括：
- IC分析（信息系数）
- 因子收益分析
- 因子稳定性分析
- 因子相关性分析
- 因子衰减分析
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date, timedelta
import warnings
from scipy import stats
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class ICAnalysis:
    """IC分析结果"""
    ic_mean: float          # IC均值
    ic_std: float           # IC标准差
    ic_ir: float            # IC信息比率 (IC均值/IC标准差)
    ic_series: pd.Series    # IC时间序列
    ic_positive_rate: float # IC为正的比例
    ic_abs_mean: float      # IC绝对值均值
    
    def __post_init__(self):
        """计算衍生指标"""
        if self.ic_series is not None and len(self.ic_series) > 0:
            self.ic_positive_rate = (self.ic_series > 0).mean()
            self.ic_abs_mean = self.ic_series.abs().mean()


@dataclass
class FactorPerformance:
    """因子表现分析结果"""
    factor_name: str
    ic_analysis: ICAnalysis
    turnover: float                    # 换手率
    factor_autocorr: float            # 因子自相关性
    long_short_return: float          # 多空收益
    long_short_sharpe: float          # 多空夏普比
    quantile_returns: pd.DataFrame    # 分位数收益
    factor_stats: Dict[str, float]    # 因子统计特征


class FactorAnalyzer:
    """
    因子分析器
    
    提供全面的因子有效性分析功能
    """
    
    def __init__(self, min_periods: int = 20):
        """
        Args:
            min_periods: 计算IC等指标的最小期数
        """
        self.min_periods = min_periods
        self.results_cache = {}
        
    def calculate_ic(self, factor_data: pd.Series, returns: pd.Series, 
                    method: str = 'pearson', forward_periods: int = 1) -> ICAnalysis:
        """
        计算信息系数(IC)
        
        Args:
            factor_data: 因子值序列
            returns: 收益率序列
            method: 相关系数方法 ('pearson', 'spearman', 'kendall')
            forward_periods: 前向收益期数
            
        Returns:
            IC分析结果
        """
        # 对齐数据
        factor_data = factor_data.dropna()
        returns = returns.dropna()
        
        # 计算前向收益
        if forward_periods > 1:
            forward_returns = returns.rolling(forward_periods).apply(
                lambda x: (1 + x).prod() - 1
            ).shift(-forward_periods + 1)
        else:
            forward_returns = returns.shift(-forward_periods + 1)
        
        # 对齐时间
        common_index = factor_data.index.intersection(forward_returns.index)
        if len(common_index) < self.min_periods:
            warnings.warn(f"Insufficient data points: {len(common_index)} < {self.min_periods}")
            return ICAnalysis(0, 0, 0, pd.Series(), 0, 0)
        
        factor_aligned = factor_data.loc[common_index]
        returns_aligned = forward_returns.loc[common_index]
        
        # 计算IC时间序列（滚动窗口）
        ic_series = pd.Series(index=common_index, dtype=float)
        
        for i in range(self.min_periods - 1, len(common_index)):
            end_idx = common_index[i]
            start_idx = common_index[max(0, i - self.min_periods + 1)]
            
            factor_window = factor_aligned.loc[start_idx:end_idx]
            returns_window = returns_aligned.loc[start_idx:end_idx]
            
            # 去除缺失值
            valid_data = pd.DataFrame({
                'factor': factor_window,
                'returns': returns_window
            }).dropna()
            
            if len(valid_data) >= 3:  # 至少需要3个观测点
                if method == 'pearson':
                    ic_value = valid_data['factor'].corr(valid_data['returns'])
                elif method == 'spearman':
                    ic_value = valid_data['factor'].corr(valid_data['returns'], method='spearman')
                elif method == 'kendall':
                    ic_value = valid_data['factor'].corr(valid_data['returns'], method='kendall')
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                ic_series.loc[end_idx] = ic_value
        
        # 去除NaN值
        ic_series = ic_series.dropna()
        
        if len(ic_series) == 0:
            return ICAnalysis(0, 0, 0, pd.Series(), 0, 0)
        
        # 计算IC统计量
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        return ICAnalysis(
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_series=ic_series,
            ic_positive_rate=0,  # 将在__post_init__中计算
            ic_abs_mean=0       # 将在__post_init__中计算
        )
    
    def calculate_turnover(self, factor_data: pd.Series, quantiles: int = 5) -> float:
        """
        计算因子换手率
        
        Args:
            factor_data: 因子值序列
            quantiles: 分位数数量
            
        Returns:
            换手率
        """
        # 计算分位数标签
        factor_quantiles = pd.cut(factor_data.dropna(), quantiles, labels=False)
        
        # 计算相邻期间的变化
        changes = factor_quantiles.diff().abs()
        turnover = changes.mean() / (quantiles - 1)  # 标准化到[0,1]
        
        return turnover if not np.isnan(turnover) else 0
    
    def calculate_factor_autocorr(self, factor_data: pd.Series, lag: int = 1) -> float:
        """
        计算因子自相关性
        
        Args:
            factor_data: 因子值序列
            lag: 滞后期数
            
        Returns:
            自相关系数
        """
        try:
            return factor_data.autocorr(lag=lag)
        except:
            return 0
    
    def analyze_quantile_returns(self, factor_data: pd.Series, returns: pd.Series,
                                quantiles: int = 5, forward_periods: int = 1) -> pd.DataFrame:
        """
        分析分位数收益
        
        Args:
            factor_data: 因子值序列
            returns: 收益率序列
            quantiles: 分位数数量
            forward_periods: 前向收益期数
            
        Returns:
            分位数收益分析结果
        """
        # 对齐数据
        common_index = factor_data.index.intersection(returns.index)
        factor_aligned = factor_data.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # 计算前向收益
        if forward_periods > 1:
            forward_returns = returns_aligned.rolling(forward_periods).apply(
                lambda x: (1 + x).prod() - 1
            ).shift(-forward_periods + 1)
        else:
            forward_returns = returns_aligned.shift(-forward_periods + 1)
        
        # 创建分析DataFrame
        analysis_data = pd.DataFrame({
            'factor': factor_aligned,
            'returns': forward_returns
        }).dropna()
        
        if len(analysis_data) < quantiles:
            return pd.DataFrame()
        
        # 计算分位数
        analysis_data['quantile'] = pd.qcut(
            analysis_data['factor'], 
            quantiles, 
            labels=[f'Q{i+1}' for i in range(quantiles)]
        )
        
        # 分组统计
        quantile_stats = analysis_data.groupby('quantile')['returns'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        # 计算累计收益
        quantile_stats['cumulative_return'] = analysis_data.groupby('quantile')['returns'].apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # 计算夏普比率
        quantile_stats['sharpe_ratio'] = quantile_stats['mean'] / quantile_stats['std']
        
        return quantile_stats
    
    def calculate_long_short_performance(self, factor_data: pd.Series, returns: pd.Series,
                                       top_pct: float = 0.2, bottom_pct: float = 0.2,
                                       forward_periods: int = 1) -> Tuple[float, float]:
        """
        计算多空组合表现
        
        Args:
            factor_data: 因子值序列
            returns: 收益率序列
            top_pct: 做多比例（因子值最高的部分）
            bottom_pct: 做空比例（因子值最低的部分）
            forward_periods: 前向收益期数
            
        Returns:
            (多空收益, 多空夏普比)
        """
        # 对齐数据
        common_index = factor_data.index.intersection(returns.index)
        factor_aligned = factor_data.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # 计算前向收益
        if forward_periods > 1:
            forward_returns = returns_aligned.rolling(forward_periods).apply(
                lambda x: (1 + x).prod() - 1
            ).shift(-forward_periods + 1)
        else:
            forward_returns = returns_aligned.shift(-forward_periods + 1)
        
        # 创建分析数据
        analysis_data = pd.DataFrame({
            'factor': factor_aligned,
            'returns': forward_returns
        }).dropna()
        
        if len(analysis_data) == 0:
            return 0, 0
        
        # 计算分位数阈值
        top_threshold = analysis_data['factor'].quantile(1 - top_pct)
        bottom_threshold = analysis_data['factor'].quantile(bottom_pct)
        
        # 选择多空组合
        long_returns = analysis_data[analysis_data['factor'] >= top_threshold]['returns']
        short_returns = analysis_data[analysis_data['factor'] <= bottom_threshold]['returns']
        
        if len(long_returns) == 0 or len(short_returns) == 0:
            return 0, 0
        
        # 计算多空收益
        long_mean = long_returns.mean()
        short_mean = short_returns.mean()
        long_short_return = long_mean - short_mean
        
        # 计算多空夏普比
        long_short_returns = long_returns.append(short_returns * -1)  # 做空收益取负
        long_short_sharpe = long_short_returns.mean() / long_short_returns.std() if long_short_returns.std() > 0 else 0
        
        return long_short_return, long_short_sharpe
    
    def comprehensive_factor_analysis(self, factor_data: pd.Series, returns: pd.Series,
                                    factor_name: str = "factor", **kwargs) -> FactorPerformance:
        """
        综合因子分析
        
        Args:
            factor_data: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            **kwargs: 其他参数
            
        Returns:
            因子表现分析结果
        """
        # IC分析
        ic_analysis = self.calculate_ic(factor_data, returns, **kwargs)
        
        # 换手率分析
        turnover = self.calculate_turnover(factor_data)
        
        # 自相关性分析
        factor_autocorr = self.calculate_factor_autocorr(factor_data)
        
        # 多空收益分析
        long_short_return, long_short_sharpe = self.calculate_long_short_performance(
            factor_data, returns
        )
        
        # 分位数收益分析
        quantile_returns = self.analyze_quantile_returns(factor_data, returns)
        
        # 因子统计特征
        factor_clean = factor_data.dropna()
        factor_stats = {
            'mean': factor_clean.mean(),
            'std': factor_clean.std(),
            'skew': factor_clean.skew(),
            'kurt': factor_clean.kurtosis(),
            'min': factor_clean.min(),
            'max': factor_clean.max(),
            'count': len(factor_clean)
        }
        
        return FactorPerformance(
            factor_name=factor_name,
            ic_analysis=ic_analysis,
            turnover=turnover,
            factor_autocorr=factor_autocorr,
            long_short_return=long_short_return,
            long_short_sharpe=long_short_sharpe,
            quantile_returns=quantile_returns,
            factor_stats=factor_stats
        )
    
    def batch_factor_analysis(self, factor_data_dict: Dict[str, pd.Series], 
                            returns: pd.Series, **kwargs) -> Dict[str, FactorPerformance]:
        """
        批量因子分析
        
        Args:
            factor_data_dict: 因子数据字典，key为因子名，value为因子序列
            returns: 收益率序列
            **kwargs: 其他参数
            
        Returns:
            因子分析结果字典
        """
        results = {}
        
        for factor_name, factor_data in factor_data_dict.items():
            try:
                result = self.comprehensive_factor_analysis(
                    factor_data, returns, factor_name, **kwargs
                )
                results[factor_name] = result
            except Exception as e:
                warnings.warn(f"Failed to analyze factor '{factor_name}': {e}")
        
        return results
    
    def factor_correlation_analysis(self, factor_data_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        因子相关性分析
        
        Args:
            factor_data_dict: 因子数据字典
            
        Returns:
            因子相关性矩阵
        """
        factor_df = pd.DataFrame(factor_data_dict)
        correlation_matrix = factor_df.corr()
        return correlation_matrix
    
    def create_factor_ranking(self, analysis_results: Dict[str, FactorPerformance],
                            ranking_criteria: str = 'ic_ir') -> pd.DataFrame:
        """
        创建因子排名
        
        Args:
            analysis_results: 因子分析结果字典
            ranking_criteria: 排名标准 ('ic_ir', 'ic_mean', 'long_short_return', 'long_short_sharpe')
            
        Returns:
            因子排名DataFrame
        """
        ranking_data = []
        
        for factor_name, result in analysis_results.items():
            row = {
                'factor_name': factor_name,
                'ic_mean': result.ic_analysis.ic_mean,
                'ic_std': result.ic_analysis.ic_std,
                'ic_ir': result.ic_analysis.ic_ir,
                'ic_positive_rate': result.ic_analysis.ic_positive_rate,
                'turnover': result.turnover,
                'factor_autocorr': result.factor_autocorr,
                'long_short_return': result.long_short_return,
                'long_short_sharpe': result.long_short_sharpe
            }
            ranking_data.append(row)
        
        ranking_df = pd.DataFrame(ranking_data)
        
        if ranking_criteria in ranking_df.columns:
            ranking_df = ranking_df.sort_values(ranking_criteria, ascending=False)
        
        ranking_df.reset_index(drop=True, inplace=True)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df
    
    def plot_ic_analysis(self, ic_analysis: ICAnalysis, factor_name: str = "Factor",
                        save_path: Optional[str] = None):
        """
        绘制IC分析图表
        
        Args:
            ic_analysis: IC分析结果
            factor_name: 因子名称
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. IC时间序列
        axes[0, 0].plot(ic_analysis.ic_series.index, ic_analysis.ic_series.values)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f'{factor_name} - IC Time Series')
        axes[0, 0].set_ylabel('IC')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. IC分布直方图
        axes[0, 1].hist(ic_analysis.ic_series.dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=ic_analysis.ic_mean, color='red', linestyle='--', 
                          label=f'Mean: {ic_analysis.ic_mean:.3f}')
        axes[0, 1].set_title(f'{factor_name} - IC Distribution')
        axes[0, 1].set_xlabel('IC')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 滚动IC均值
        rolling_ic_mean = ic_analysis.ic_series.rolling(20).mean()
        axes[1, 0].plot(rolling_ic_mean.index, rolling_ic_mean.values)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title(f'{factor_name} - Rolling IC Mean (20 periods)')
        axes[1, 0].set_ylabel('Rolling IC Mean')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. IC统计摘要
        stats_text = f"""
        IC Mean: {ic_analysis.ic_mean:.4f}
        IC Std: {ic_analysis.ic_std:.4f}
        IC IR: {ic_analysis.ic_ir:.4f}
        IC Positive Rate: {ic_analysis.ic_positive_rate:.2%}
        IC Abs Mean: {ic_analysis.ic_abs_mean:.4f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title(f'{factor_name} - IC Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_factor_performance_summary(self, analysis_results: Dict[str, FactorPerformance],
                                      save_path: Optional[str] = None):
        """
        绘制因子表现摘要图
        
        Args:
            analysis_results: 因子分析结果字典
            save_path: 保存路径
        """
        # 准备数据
        factor_names = list(analysis_results.keys())
        ic_means = [result.ic_analysis.ic_mean for result in analysis_results.values()]
        ic_irs = [result.ic_analysis.ic_ir for result in analysis_results.values()]
        long_short_returns = [result.long_short_return for result in analysis_results.values()]
        long_short_sharpes = [result.long_short_sharpe for result in analysis_results.values()]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. IC Mean排名
        sorted_data = sorted(zip(factor_names, ic_means), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_data)
        axes[0, 0].barh(range(len(names)), values)
        axes[0, 0].set_yticks(range(len(names)))
        axes[0, 0].set_yticklabels(names)
        axes[0, 0].set_title('IC Mean Ranking')
        axes[0, 0].set_xlabel('IC Mean')
        
        # 2. IC IR排名
        sorted_data = sorted(zip(factor_names, ic_irs), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_data)
        axes[0, 1].barh(range(len(names)), values)
        axes[0, 1].set_yticks(range(len(names)))
        axes[0, 1].set_yticklabels(names)
        axes[0, 1].set_title('IC Information Ratio Ranking')
        axes[0, 1].set_xlabel('IC IR')
        
        # 3. 多空收益排名
        sorted_data = sorted(zip(factor_names, long_short_returns), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_data)
        axes[1, 0].barh(range(len(names)), values)
        axes[1, 0].set_yticks(range(len(names)))
        axes[1, 0].set_yticklabels(names)
        axes[1, 0].set_title('Long-Short Return Ranking')
        axes[1, 0].set_xlabel('Long-Short Return')
        
        # 4. 多空夏普比排名
        sorted_data = sorted(zip(factor_names, long_short_sharpes), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_data)
        axes[1, 1].barh(range(len(names)), values)
        axes[1, 1].set_yticks(range(len(names)))
        axes[1, 1].set_yticklabels(names)
        axes[1, 1].set_title('Long-Short Sharpe Ranking')
        axes[1, 1].set_xlabel('Long-Short Sharpe')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def create_factor_analyzer(min_periods: int = 20) -> FactorAnalyzer:
    """
    创建因子分析器的便捷函数
    
    Args:
        min_periods: 最小计算期数
        
    Returns:
        FactorAnalyzer实例
    """
    return FactorAnalyzer(min_periods)