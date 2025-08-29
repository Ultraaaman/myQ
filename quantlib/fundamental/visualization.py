"""
可视化模块 - 负责生成各种财务分析图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class FinancialChartGenerator:
    """财务图表生成器"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_comprehensive_analysis(self, ratios, peer_comparison=None, financial_data=None):
        """绘制综合分析图表"""
        if not ratios and not peer_comparison and not financial_data:
            print("请先进行财务分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.symbol} 基本面分析图表', fontsize=16)
        
        # 1. 财务比率雷达图
        if ratios:
            self.plot_ratios_radar(axes[0, 0], ratios)
        
        # 2. 同行对比
        if peer_comparison is not None and not peer_comparison.empty:
            self.plot_peer_comparison(axes[0, 1], peer_comparison)
        
        # 3. 财务趋势
        self.plot_financial_trends(axes[1, 0], financial_data)
        
        # 4. 估值分析
        if ratios:
            self.plot_valuation_analysis(axes[1, 1], ratios)
        
        plt.tight_layout()
        plt.show()
    
    def plot_ratios_radar(self, ax, ratios):
        """绘制财务比率雷达图"""
        if not ratios:
            ax.text(0.5, 0.5, '无财务比率数据', ha='center', va='center')
            ax.set_title('财务比率分析')
            return
        
        # 选择关键比率进行展示
        key_ratios = ['ROE', 'ROA', '净利率', '流动比率']
        available_ratios = {k: v for k, v in ratios.items() if k in key_ratios}
        
        if not available_ratios:
            ax.text(0.5, 0.5, '无关键比率数据', ha='center', va='center')
            ax.set_title('财务比率分析')
            return
        
        # 标准化数据（转换为0-10分）
        normalized_values = []
        labels = []
        
        for ratio_name, value in available_ratios.items():
            labels.append(ratio_name)
            if ratio_name == 'ROE':
                normalized_values.append(min(value / 2, 10))  # ROE 20%为满分
            elif ratio_name == 'ROA':
                normalized_values.append(min(value, 10))  # ROA 10%为满分
            elif ratio_name == '净利率':
                normalized_values.append(min(value, 10))  # 净利率 10%为满分
            elif ratio_name == '流动比率':
                normalized_values.append(min(value * 3, 10))  # 流动比率 3.33为满分
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        normalized_values += normalized_values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax.plot(angles, normalized_values, 'o-', linewidth=2, label=self.symbol)
        ax.fill(angles, normalized_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 10)
        ax.set_title('财务比率分析')
        ax.grid(True)
    
    def plot_peer_comparison(self, ax, peer_comparison):
        """绘制同行对比图"""
        if peer_comparison.empty:
            ax.text(0.5, 0.5, '无同行对比数据', ha='center', va='center')
            ax.set_title('同行对比')
            return
        
        # PE比较
        if 'PE' in peer_comparison.columns:
            pe_data = peer_comparison[peer_comparison['PE'] > 0]['PE']  # 过滤掉无效PE值
            symbols = peer_comparison[peer_comparison['PE'] > 0]['Symbol']
            
            if not pe_data.empty:
                bars = ax.bar(symbols, pe_data)
                
                # 高亮目标股票
                for i, symbol in enumerate(symbols):
                    if symbol == self.symbol:
                        bars[i].set_color('red')
                        bars[i].set_alpha(0.8)
                
                ax.set_title('PE比较')
                ax.set_ylabel('市盈率')
                plt.setp(ax.get_xticklabels(), rotation=45)
            else:
                ax.text(0.5, 0.5, '无有效PE数据', ha='center', va='center')
                ax.set_title('同行对比')
        else:
            ax.text(0.5, 0.5, '无PE数据', ha='center', va='center')
            ax.set_title('同行对比')
    
    def plot_financial_trends(self, ax, financial_data):
        """绘制财务趋势图"""
        try:
            if not financial_data:
                ax.text(0.5, 0.5, '无财务趋势数据', ha='center', va='center')
                ax.set_title('财务趋势')
                return
            
            if 'income_statement' in financial_data:
                # 美股财务趋势
                self._plot_us_trends(ax, financial_data)
            elif 'indicators' in financial_data:
                # 中国股票财务趋势
                self._plot_cn_trends(ax, financial_data)
            else:
                ax.text(0.5, 0.5, '无财务趋势数据', ha='center', va='center')
                ax.set_title('财务趋势')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'趋势图绘制失败:\n{str(e)}', ha='center', va='center')
            ax.set_title('财务趋势')
    
    def _plot_us_trends(self, ax, financial_data):
        """绘制美股财务趋势"""
        financials = financial_data['income_statement']
        
        if 'Total Revenue' in financials.index:
            revenue_data = financials.loc['Total Revenue']
            dates = [col.strftime('%Y') for col in revenue_data.index]
            values = revenue_data.values / 1e9  # 转换为十亿
            
            ax.plot(dates, values, marker='o', label='营业收入(十亿)')
        
        if 'Net Income' in financials.index:
            income_data = financials.loc['Net Income']
            dates = [col.strftime('%Y') for col in income_data.index]
            values = income_data.values / 1e9  # 转换为十亿
            
            ax.plot(dates, values, marker='s', label='净利润(十亿)')
        
        ax.set_title('财务趋势')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cn_trends(self, ax, financial_data):
        """绘制中国股票财务趋势"""
        indicators = financial_data['indicators']
        
        if len(indicators) > 1:
            # 寻找营业收入列
            revenue_col = None
            for col in indicators.columns:
                if '营业收入' in col and '营业收入-营业收入' == col:
                    revenue_col = col
                    break
            
            if revenue_col:
                revenue_data = indicators[revenue_col].dropna()
                dates = indicators['日期'].iloc[:len(revenue_data)]
                
                ax.plot(range(len(revenue_data)), revenue_data, marker='o', label='营业收入')
                ax.set_xticks(range(len(dates)))
                ax.set_xticklabels([str(d)[:7] for d in dates], rotation=45)  # 显示年-月
                ax.set_title('营业收入趋势')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无营业收入趋势数据', ha='center', va='center')
                ax.set_title('财务趋势')
        else:
            ax.text(0.5, 0.5, '数据不足，无法显示趋势', ha='center', va='center')
            ax.set_title('财务趋势')
    
    def plot_valuation_analysis(self, ax, ratios):
        """绘制估值分析图"""
        if not ratios:
            ax.text(0.5, 0.5, '无估值数据', ha='center', va='center')
            ax.set_title('估值分析')
            return
        
        # 估值指标对比
        valuation_metrics = {}
        if 'PE' in ratios:
            valuation_metrics['PE'] = ratios['PE']
        if 'PB' in ratios:
            valuation_metrics['PB'] = ratios['PB']
        
        if valuation_metrics:
            metrics = list(valuation_metrics.keys())
            values = list(valuation_metrics.values())
            
            bars = ax.bar(metrics, values, color=['skyblue', 'lightgreen'])
            
            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v + max(values) * 0.01, f'{v:.1f}', ha='center', va='bottom')
            
            ax.set_title('估值指标')
            ax.set_ylabel('倍数')
            
            # 添加合理估值区间的参考线
            if 'PE' in valuation_metrics:
                pe_idx = metrics.index('PE')
                ax.axhline(y=15, color='red', linestyle='--', alpha=0.5)
                ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, '无估值指标数据', ha='center', va='center')
            ax.set_title('估值分析')
    
    def plot_detailed_ratios_summary(self, ratios):
        """绘制详细财务比率分类图表"""
        if not ratios:
            print("没有可显示的财务比率数据")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.symbol} 详细财务指标分析', fontsize=16)
        
        # 1. 盈利能力指标
        self._plot_profitability_ratios(axes[0, 0], ratios)
        
        # 2. 成长性指标  
        self._plot_growth_ratios(axes[0, 1], ratios)
        
        # 3. 偿债能力指标
        self._plot_solvency_ratios(axes[0, 2], ratios)
        
        # 4. 估值指标
        self._plot_valuation_ratios(axes[1, 0], ratios)
        
        # 5. 现金流质量指标
        self._plot_cashflow_ratios(axes[1, 1], ratios)
        
        # 6. 市场表现指标
        self._plot_market_ratios(axes[1, 2], ratios)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_profitability_ratios(self, ax, ratios):
        """绘制盈利能力指标"""
        profitability_ratios = {}
        for key in ['ROE', 'ROA', '净利率', '毛利率']:
            if key in ratios:
                profitability_ratios[key] = ratios[key]
        
        if profitability_ratios:
            keys = list(profitability_ratios.keys())
            values = list(profitability_ratios.values())
            
            bars = ax.bar(keys, values, color='lightblue')
            ax.set_title('盈利能力指标 (%)')
            ax.set_ylabel('百分比')
            
            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v + max(values) * 0.01, f'{v:.1f}%', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, '无盈利能力数据', ha='center', va='center')
            ax.set_title('盈利能力指标')
    
    def _plot_growth_ratios(self, ax, ratios):
        """绘制成长性指标"""
        growth_ratios = {}
        for key in ['Revenue Growth', 'Net Income Growth', 'EPS Growth']:
            if key in ratios:
                growth_ratios[key] = ratios[key]
        
        if growth_ratios:
            keys = list(growth_ratios.keys())
            values = list(growth_ratios.values())
            
            # 使用不同颜色表示正负增长
            colors = ['green' if v > 0 else 'red' for v in values]
            bars = ax.bar(keys, values, color=colors, alpha=0.7)
            ax.set_title('成长性指标 (%)')
            ax.set_ylabel('增长率')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(values):
                y_pos = v + (max(values) - min(values)) * 0.01 if v >= 0 else v - (max(values) - min(values)) * 0.01
                ax.text(i, y_pos, f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top')
            
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, '无成长性数据', ha='center', va='center')
            ax.set_title('成长性指标')
    
    def _plot_solvency_ratios(self, ax, ratios):
        """绘制偿债能力指标"""
        solvency_ratios = {}
        for key in ['资产负债率', '流动比率', '速动比率']:
            if key in ratios:
                solvency_ratios[key] = ratios[key]
        
        if solvency_ratios:
            keys = list(solvency_ratios.keys())
            values = list(solvency_ratios.values())
            
            bars = ax.bar(keys, values, color='lightcoral')
            ax.set_title('偿债能力指标')
            ax.set_ylabel('比率')
            
            # 添加数值标签
            for i, v in enumerate(values):
                suffix = '%' if '率' in keys[i] and keys[i] not in ['流动比率', '速动比率'] else ''
                ax.text(i, v + max(values) * 0.01, f'{v:.2f}{suffix}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, '无偿债能力数据', ha='center', va='center')
            ax.set_title('偿债能力指标')
    
    def _plot_valuation_ratios(self, ax, ratios):
        """绘制估值指标"""
        valuation_ratios = {}
        for key in ['PE', 'PB']:
            if key in ratios:
                valuation_ratios[key] = ratios[key]
        
        if valuation_ratios:
            keys = list(valuation_ratios.keys())
            values = list(valuation_ratios.values())
            
            bars = ax.bar(keys, values, color='lightgreen')
            ax.set_title('估值指标')
            ax.set_ylabel('倍数')
            
            # 添加合理区间参考线
            if 'PE' in valuation_ratios:
                ax.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='PE合理线')
                ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='PE警戒线')
            
            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v + max(values) * 0.01, f'{v:.1f}', ha='center', va='bottom')
            
            if 'PE' in valuation_ratios:
                ax.legend()
        else:
            ax.text(0.5, 0.5, '无估值数据', ha='center', va='center')
            ax.set_title('估值指标')
    
    def _plot_cashflow_ratios(self, ax, ratios):
        """绘制现金流质量指标"""
        cashflow_ratios = {}
        for key in ['Operating CF to Net Income', 'Free Cash Flow Yield']:
            if key in ratios:
                cashflow_ratios[key] = ratios[key]
        
        if cashflow_ratios:
            keys = [k.replace('Operating CF to Net Income', 'CF/NI') 
                   .replace('Free Cash Flow Yield', 'FCF Yield') for k in cashflow_ratios.keys()]
            values = list(cashflow_ratios.values())
            
            bars = ax.bar(keys, values, color='gold')
            ax.set_title('现金流质量指标')
            ax.set_ylabel('百分比')
            
            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v + max(values) * 0.01, f'{v:.1f}%', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, '无现金流数据', ha='center', va='center')
            ax.set_title('现金流质量指标')
    
    def _plot_market_ratios(self, ax, ratios):
        """绘制市场表现指标"""
        market_ratios = {}
        for key in ['Beta', 'Volatility', '1M Price Change', '3M Price Change']:
            if key in ratios:
                market_ratios[key] = ratios[key]
        
        if market_ratios:
            keys = [k.replace('1M Price Change', '1M Change')
                   .replace('3M Price Change', '3M Change') for k in market_ratios.keys()]
            values = list(market_ratios.values())
            
            # 根据指标类型使用不同颜色
            colors = []
            for i, key in enumerate(market_ratios.keys()):
                if 'Change' in key:
                    colors.append('green' if values[i] > 0 else 'red')
                else:
                    colors.append('purple')
            
            bars = ax.bar(keys, values, color=colors, alpha=0.7)
            ax.set_title('市场表现指标')
            ax.set_ylabel('数值')
            
            if any('Change' in k for k in market_ratios.keys()):
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(values):
                suffix = '%' if 'Change' in list(market_ratios.keys())[i] or 'Volatility' in list(market_ratios.keys())[i] else ''
                ax.text(i, v + (max(values) - min(values)) * 0.01, f'{v:.1f}{suffix}', 
                       ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, '无市场表现数据', ha='center', va='center')
            ax.set_title('市场表现指标')