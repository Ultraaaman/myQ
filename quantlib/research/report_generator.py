"""
研究报告生成器 (Report Generator)

提供完整的研究报告生成功能，包括：
- 因子研究报告
- 回测结果报告
- HTML/PDF报告生成
- 图表可视化
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
import warnings
from pathlib import Path
import json
import base64
from io import BytesIO

# 图表库
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
sns.set_palette("husl")

# HTML模板
from string import Template

from .factor_analyzer import FactorPerformance, ICAnalysis
from .research_framework import BacktestResult


class ReportGenerator:
    """
    研究报告生成器
    
    生成专业的因子研究和回测分析报告
    """
    
    def __init__(self, output_path: str = "reports"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置图表样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        
    def generate_factor_analysis_report(self, analysis_results: Dict[str, FactorPerformance],
                                      title: str = "因子分析报告",
                                      format: str = "html") -> str:
        """
        生成因子分析报告
        
        Args:
            analysis_results: 因子分析结果字典
            title: 报告标题
            format: 输出格式 ('html', 'markdown')
            
        Returns:
            报告文件路径
        """
        if format == "html":
            return self._generate_html_factor_report(analysis_results, title)
        elif format == "markdown":
            return self._generate_markdown_factor_report(analysis_results, title)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_backtest_report(self, backtest_results: Dict[str, BacktestResult],
                               title: str = "因子回测报告",
                               format: str = "html") -> str:
        """
        生成回测报告
        
        Args:
            backtest_results: 回测结果字典
            title: 报告标题
            format: 输出格式
            
        Returns:
            报告文件路径
        """
        if format == "html":
            return self._generate_html_backtest_report(backtest_results, title)
        elif format == "markdown":
            return self._generate_markdown_backtest_report(backtest_results, title)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, FactorPerformance],
                                    backtest_results: Dict[str, BacktestResult],
                                    factor_ranking: pd.DataFrame,
                                    correlation_matrix: pd.DataFrame,
                                    summary: Dict[str, Any],
                                    title: str = "综合因子研究报告",
                                    format: str = "html") -> str:
        """
        生成综合研究报告
        
        Args:
            analysis_results: 因子分析结果
            backtest_results: 回测结果
            factor_ranking: 因子排名
            correlation_matrix: 相关性矩阵
            summary: 研究摘要
            title: 报告标题
            format: 输出格式
            
        Returns:
            报告文件路径
        """
        if format == "html":
            return self._generate_html_comprehensive_report(
                analysis_results, backtest_results, factor_ranking, 
                correlation_matrix, summary, title
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_factor_report(self, analysis_results: Dict[str, FactorPerformance],
                                   title: str) -> str:
        """生成HTML因子分析报告"""
        # 生成图表
        charts = {}
        
        # 1. 因子IC分布图
        charts['ic_distribution'] = self._create_ic_distribution_chart(analysis_results)
        
        # 2. 因子表现对比图
        charts['performance_comparison'] = self._create_performance_comparison_chart(analysis_results)
        
        # 3. 因子统计表格
        summary_table = self._create_factor_summary_table(analysis_results)
        
        # HTML模板
        html_template = Template("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$title</title>
    <style>
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }
        .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border: 1px solid #ddd; }
        th { background-color: #f5f5f5; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .metric { display: inline-block; margin: 10px 20px; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>$title</h1>
        <p>生成时间: $timestamp</p>
        <p>分析因子数量: $factor_count</p>
    </div>
    
    <div class="section">
        <h2>📊 关键指标概览</h2>
        <div class="metric">
            <div class="metric-value">$avg_ic_mean</div>
            <div class="metric-label">平均IC</div>
        </div>
        <div class="metric">
            <div class="metric-value">$avg_ic_ir</div>
            <div class="metric-label">平均IC_IR</div>
        </div>
        <div class="metric">
            <div class="metric-value">$best_factor</div>
            <div class="metric-label">最佳因子</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 因子IC分布</h2>
        <div class="chart">
            <img src="data:image/png;base64,$ic_distribution_chart" alt="IC分布图">
        </div>
    </div>
    
    <div class="section">
        <h2>🏆 因子表现对比</h2>
        <div class="chart">
            <img src="data:image/png;base64,$performance_chart" alt="表现对比图">
        </div>
    </div>
    
    <div class="section">
        <h2>📋 详细统计表</h2>
        $summary_table
    </div>
    
    <div class="section">
        <h2>💡 分析结论</h2>
        <ul>
            <li><strong>最佳IC因子:</strong> $best_ic_factor (IC_IR: $best_ic_ir)</li>
            <li><strong>最佳多空因子:</strong> $best_ls_factor (收益: $best_ls_return)</li>
            <li><strong>平均换手率:</strong> $avg_turnover</li>
            <li><strong>建议:</strong> $recommendations</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>⚠️ 风险提示</h2>
        <p>本报告基于历史数据分析，不构成投资建议。因子表现可能存在时效性，请结合最新市场情况使用。</p>
    </div>
</body>
</html>
        """)
        
        # 计算摘要统计
        ic_means = [r.ic_analysis.ic_mean for r in analysis_results.values()]
        ic_irs = [r.ic_analysis.ic_ir for r in analysis_results.values()]
        turnovers = [r.turnover for r in analysis_results.values()]
        
        best_ic_factor = max(analysis_results.items(), key=lambda x: x[1].ic_analysis.ic_ir)
        best_ls_factor = max(analysis_results.items(), key=lambda x: x[1].long_short_return)
        
        # 生成建议
        recommendations = self._generate_factor_recommendations(analysis_results)
        
        # 填充模板
        html_content = html_template.substitute(
            title=title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            factor_count=len(analysis_results),
            avg_ic_mean=f"{np.mean(ic_means):.4f}",
            avg_ic_ir=f"{np.mean(ic_irs):.4f}",
            best_factor=best_ic_factor[0],
            ic_distribution_chart=charts['ic_distribution'],
            performance_chart=charts['performance_comparison'],
            summary_table=summary_table,
            best_ic_factor=best_ic_factor[0],
            best_ic_ir=f"{best_ic_factor[1].ic_analysis.ic_ir:.4f}",
            best_ls_factor=best_ls_factor[0],
            best_ls_return=f"{best_ls_factor[1].long_short_return:.2%}",
            avg_turnover=f"{np.mean(turnovers):.2%}",
            recommendations=recommendations
        )
        
        # 保存文件
        filename = f"factor_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def _create_ic_distribution_chart(self, analysis_results: Dict[str, FactorPerformance]) -> str:
        """创建IC分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # IC均值分布
        ic_means = [r.ic_analysis.ic_mean for r in analysis_results.values()]
        factor_names = list(analysis_results.keys())
        
        axes[0].bar(range(len(factor_names)), ic_means)
        axes[0].set_xticks(range(len(factor_names)))
        axes[0].set_xticklabels(factor_names, rotation=45, ha='right')
        axes[0].set_title('因子IC均值分布')
        axes[0].set_ylabel('IC均值')
        axes[0].grid(True, alpha=0.3)
        
        # IC信息比率分布
        ic_irs = [r.ic_analysis.ic_ir for r in analysis_results.values()]
        
        axes[1].bar(range(len(factor_names)), ic_irs, color='orange')
        axes[1].set_xticks(range(len(factor_names)))
        axes[1].set_xticklabels(factor_names, rotation=45, ha='right')
        axes[1].set_title('因子IC信息比率分布')
        axes[1].set_ylabel('IC信息比率')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 转换为base64
        return self._fig_to_base64(fig)
    
    def _create_performance_comparison_chart(self, analysis_results: Dict[str, FactorPerformance]) -> str:
        """创建表现对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        factor_names = list(analysis_results.keys())
        
        # 1. IC vs 换手率散点图
        ic_means = [r.ic_analysis.ic_mean for r in analysis_results.values()]
        turnovers = [r.turnover for r in analysis_results.values()]
        
        axes[0, 0].scatter(turnovers, ic_means, alpha=0.7, s=100)
        for i, name in enumerate(factor_names):
            axes[0, 0].annotate(name, (turnovers[i], ic_means[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 0].set_xlabel('换手率')
        axes[0, 0].set_ylabel('IC均值')
        axes[0, 0].set_title('IC vs 换手率')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 多空收益排名
        ls_returns = [r.long_short_return for r in analysis_results.values()]
        sorted_data = sorted(zip(factor_names, ls_returns), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_data)
        
        axes[0, 1].barh(range(len(names)), values)
        axes[0, 1].set_yticks(range(len(names)))
        axes[0, 1].set_yticklabels(names)
        axes[0, 1].set_title('多空收益排名')
        axes[0, 1].set_xlabel('多空收益')
        
        # 3. IC稳定性（IC_IR）
        ic_irs = [r.ic_analysis.ic_ir for r in analysis_results.values()]
        
        axes[1, 0].bar(range(len(factor_names)), ic_irs, color='green', alpha=0.7)
        axes[1, 0].set_xticks(range(len(factor_names)))
        axes[1, 0].set_xticklabels(factor_names, rotation=45, ha='right')
        axes[1, 0].set_title('IC信息比率（稳定性指标）')
        axes[1, 0].set_ylabel('IC_IR')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 因子自相关性
        autocorrs = [r.factor_autocorr for r in analysis_results.values()]
        
        axes[1, 1].bar(range(len(factor_names)), autocorrs, color='red', alpha=0.7)
        axes[1, 1].set_xticks(range(len(factor_names)))
        axes[1, 1].set_xticklabels(factor_names, rotation=45, ha='right')
        axes[1, 1].set_title('因子自相关性')
        axes[1, 1].set_ylabel('自相关系数')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_factor_summary_table(self, analysis_results: Dict[str, FactorPerformance]) -> str:
        """创建因子摘要表格"""
        table_data = []
        
        for name, result in analysis_results.items():
            table_data.append({
                '因子名称': name,
                'IC均值': f"{result.ic_analysis.ic_mean:.4f}",
                'IC标准差': f"{result.ic_analysis.ic_std:.4f}",
                'IC信息比率': f"{result.ic_analysis.ic_ir:.4f}",
                'IC胜率': f"{result.ic_analysis.ic_positive_rate:.2%}",
                '换手率': f"{result.turnover:.2%}",
                '自相关性': f"{result.factor_autocorr:.4f}",
                '多空收益': f"{result.long_short_return:.2%}",
                '多空夏普': f"{result.long_short_sharpe:.4f}"
            })
        
        df = pd.DataFrame(table_data)
        return df.to_html(index=False, table_id='factor_summary', classes='table table-striped')
    
    def _generate_factor_recommendations(self, analysis_results: Dict[str, FactorPerformance]) -> str:
        """生成因子建议"""
        recommendations = []
        
        # 找出最佳因子
        best_ic_factor = max(analysis_results.items(), key=lambda x: x[1].ic_analysis.ic_ir)
        if best_ic_factor[1].ic_analysis.ic_ir > 0.5:
            recommendations.append(f"推荐使用{best_ic_factor[0]}因子，IC信息比率较高")
        
        # 检查换手率
        high_turnover_factors = [name for name, result in analysis_results.items() if result.turnover > 0.5]
        if high_turnover_factors:
            recommendations.append(f"注意{', '.join(high_turnover_factors)}等因子换手率较高，交易成本需考虑")
        
        # 检查相关性
        low_autocorr_factors = [name for name, result in analysis_results.items() if abs(result.factor_autocorr) < 0.1]
        if low_autocorr_factors:
            recommendations.append(f"{', '.join(low_autocorr_factors)}等因子稳定性较好")
        
        return "; ".join(recommendations) if recommendations else "需要进一步分析具体使用建议"
    
    def _fig_to_base64(self, fig) -> str:
        """将matplotlib图形转换为base64字符串"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close(fig)
        return image_base64
    
    def _generate_html_backtest_report(self, backtest_results: Dict[str, BacktestResult],
                                     title: str) -> str:
        """生成HTML回测报告"""
        # 简化版本的回测报告
        html_template = Template("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>$title</title>
    <style>
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }
        .header { background: #007bff; color: white; padding: 20px; border-radius: 10px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #f5f5f5; }
    </style>
</head>
<body>
    <div class="header">
        <h1>$title</h1>
        <p>生成时间: $timestamp</p>
    </div>
    
    <div class="section">
        <h2>回测结果摘要</h2>
        $results_table
    </div>
</body>
</html>
        """)
        
        # 创建结果表格
        table_data = []
        for name, result in backtest_results.items():
            table_data.append({
                '因子名称': name,
                '总收益率': f"{result.total_return:.2%}",
                '年化收益率': f"{result.annual_return:.2%}",
                '年化波动率': f"{result.volatility:.2%}",
                '夏普比率': f"{result.sharpe_ratio:.2f}",
                '最大回撤': f"{result.max_drawdown:.2%}",
                '胜率': f"{result.win_rate:.2%}"
            })
        
        results_df = pd.DataFrame(table_data)
        results_table = results_df.to_html(index=False, classes='table table-striped')
        
        html_content = html_template.substitute(
            title=title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results_table=results_table
        )
        
        filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def _generate_html_comprehensive_report(self, analysis_results: Dict[str, FactorPerformance],
                                          backtest_results: Dict[str, BacktestResult],
                                          factor_ranking: pd.DataFrame,
                                          correlation_matrix: pd.DataFrame,
                                          summary: Dict[str, Any],
                                          title: str) -> str:
        """生成HTML综合报告"""
        # 这里可以实现一个更复杂的综合报告模板
        # 暂时返回简化版本
        return self._generate_html_factor_report(analysis_results, title)
    
    def _generate_markdown_factor_report(self, analysis_results: Dict[str, FactorPerformance],
                                       title: str) -> str:
        """生成Markdown因子报告"""
        content = [f"# {title}", ""]
        content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"分析因子数量: {len(analysis_results)}")
        content.append("")
        
        content.append("## 因子分析结果")
        content.append("")
        
        for name, result in analysis_results.items():
            content.append(f"### {name}")
            content.append(f"- IC均值: {result.ic_analysis.ic_mean:.4f}")
            content.append(f"- IC信息比率: {result.ic_analysis.ic_ir:.4f}")
            content.append(f"- 换手率: {result.turnover:.2%}")
            content.append(f"- 多空收益: {result.long_short_return:.2%}")
            content.append("")
        
        markdown_content = "\n".join(content)
        
        filename = f"factor_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return str(filepath)
    
    def _generate_markdown_backtest_report(self, backtest_results: Dict[str, BacktestResult],
                                         title: str) -> str:
        """生成Markdown回测报告"""
        content = [f"# {title}", ""]
        content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        content.append("## 回测结果")
        content.append("")
        
        for name, result in backtest_results.items():
            content.append(f"### {name}")
            content.append(f"- 总收益率: {result.total_return:.2%}")
            content.append(f"- 年化收益率: {result.annual_return:.2%}")
            content.append(f"- 夏普比率: {result.sharpe_ratio:.2f}")
            content.append(f"- 最大回撤: {result.max_drawdown:.2%}")
            content.append("")
        
        markdown_content = "\n".join(content)
        
        filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return str(filepath)


def create_research_report(analysis_results: Optional[Dict[str, FactorPerformance]] = None,
                         backtest_results: Optional[Dict[str, BacktestResult]] = None,
                         title: str = "研究报告",
                         format: str = "html",
                         output_path: str = "reports") -> str:
    """
    创建研究报告的便捷函数
    
    Args:
        analysis_results: 因子分析结果
        backtest_results: 回测结果
        title: 报告标题
        format: 输出格式
        output_path: 输出路径
        
    Returns:
        报告文件路径
    """
    generator = ReportGenerator(output_path)
    
    if analysis_results and backtest_results:
        # 综合报告
        return generator.generate_comprehensive_report(
            analysis_results, backtest_results, 
            pd.DataFrame(), pd.DataFrame(), {}, title, format
        )
    elif analysis_results:
        # 因子分析报告
        return generator.generate_factor_analysis_report(analysis_results, title, format)
    elif backtest_results:
        # 回测报告
        return generator.generate_backtest_report(backtest_results, title, format)
    else:
        raise ValueError("Must provide either analysis_results or backtest_results")