"""
可视化模块 - 负责生成各种财务分析图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


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
    
    def plot_hs300_benchmark(self, period='1y', comparison_symbol=None):
        """绘制沪深300大盘走势图，可选择与个股对比"""
        try:
            # 获取沪深300数据
            hs300 = yf.Ticker("000300.SS")  # 沪深300指数代码
            
            # 设置时间范围
            if period == '1d':
                hist_data = hs300.history(period='1d', interval='1h')
                title_period = '日内'
            elif period == '1w':
                hist_data = hs300.history(period='7d')
                title_period = '一周'
            elif period == '1m':
                hist_data = hs300.history(period='1mo')
                title_period = '一个月'
            elif period == '3m':
                hist_data = hs300.history(period='3mo')
                title_period = '三个月'
            elif period == '6m':
                hist_data = hs300.history(period='6mo')
                title_period = '六个月'
            elif period == '1y':
                hist_data = hs300.history(period='1y')
                title_period = '一年'
            elif period == '2y':
                hist_data = hs300.history(period='2y')
                title_period = '两年'
            else:
                hist_data = hs300.history(period='1y')
                title_period = '一年'
            
            if hist_data.empty:
                print("无法获取沪深300数据，请检查网络连接")
                return
            
            # 创建图表
            if comparison_symbol:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
                fig.suptitle(f'沪深300大盘走势与{comparison_symbol}对比 ({title_period})', fontsize=16)
                
                # 上图：沪深300走势
                self._plot_single_index(ax1, hist_data, '沪深300指数', 'blue')
                
                # 下图：个股对比
                try:
                    # 尝试获取个股数据
                    stock_symbol = comparison_symbol
                    if not stock_symbol.endswith(('.SS', '.SZ')):
                        # 如果没有交易所后缀，尝试添加
                        if stock_symbol.startswith('0') or stock_symbol.startswith('3'):
                            stock_symbol += '.SZ'
                        else:
                            stock_symbol += '.SS'
                    
                    stock = yf.Ticker(stock_symbol)
                    stock_data = stock.history(period=period)
                    
                    if not stock_data.empty:
                        self._plot_comparison_chart(ax2, hist_data, stock_data, comparison_symbol)
                    else:
                        ax2.text(0.5, 0.5, f'无法获取{comparison_symbol}数据', ha='center', va='center')
                        ax2.set_title(f'{comparison_symbol} vs 沪深300')
                        
                except Exception as e:
                    ax2.text(0.5, 0.5, f'获取{comparison_symbol}数据失败:\n{str(e)}', ha='center', va='center')
                    ax2.set_title(f'{comparison_symbol} vs 沪深300')
            else:
                fig, ax = plt.subplots(1, 1, figsize=(15, 8))
                fig.suptitle(f'沪深300大盘走势 ({title_period})', fontsize=16)
                self._plot_single_index(ax, hist_data, '沪深300指数', 'blue')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"绘制沪深300走势图失败: {str(e)}")
    
    def _plot_single_index(self, ax, data, title, color):
        """绘制单个指数走势"""
        # 绘制收盘价线图
        ax.plot(data.index, data['Close'], color=color, linewidth=2, label=title)
        ax.fill_between(data.index, data['Close'], alpha=0.3, color=color)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_ylabel('指数点位')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加最新点位和涨跌幅信息
        if len(data) > 1:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[0]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            # 在图上添加当前信息
            info_text = f"当前: {current_price:.2f}\n涨跌: {change:+.2f} ({change_pct:+.2f}%)"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                   verticalalignment='top', fontsize=10)
        
        # 格式化x轴日期
        if len(data) > 30:
            # 如果数据点太多，只显示部分日期标签
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_comparison_chart(self, ax, hs300_data, stock_data, stock_symbol):
        """绘制沪深300与个股的对比图（归一化）"""
        # 确保两个数据集有相同的时间范围
        common_dates = hs300_data.index.intersection(stock_data.index)
        if len(common_dates) == 0:
            ax.text(0.5, 0.5, '数据时间范围不匹配', ha='center', va='center')
            ax.set_title(f'{stock_symbol} vs 沪深300')
            return
        
        hs300_aligned = hs300_data.loc[common_dates]['Close']
        stock_aligned = stock_data.loc[common_dates]['Close']
        
        # 归一化到起始点为100
        hs300_normalized = (hs300_aligned / hs300_aligned.iloc[0]) * 100
        stock_normalized = (stock_aligned / stock_aligned.iloc[0]) * 100
        
        # 绘制对比图
        ax.plot(common_dates, hs300_normalized, color='blue', linewidth=2, label='沪深300')
        ax.plot(common_dates, stock_normalized, color='red', linewidth=2, label=stock_symbol)
        
        ax.set_title(f'{stock_symbol} vs 沪深300 (归一化对比)')
        ax.set_ylabel('相对表现 (起始=100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        # 添加最终表现对比
        if len(hs300_normalized) > 0 and len(stock_normalized) > 0:
            hs300_final = hs300_normalized.iloc[-1]
            stock_final = stock_normalized.iloc[-1]
            outperformance = stock_final - hs300_final
            
            info_text = f"沪深300: {hs300_final:.1f}\n{stock_symbol}: {stock_final:.1f}\n相对表现: {outperformance:+.1f}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                   verticalalignment='top', fontsize=10)
        
        ax.tick_params(axis='x', rotation=45)
    
    def plot_market_overview(self, period='1d'):
        """绘制市场概览图，包括沪深300、上证指数、深证成指"""
        try:
            # 主要指数代码
            indices = {
                '沪深300': '000300.SS',
                '上证指数': '000001.SS', 
                '深证成指': '399001.SZ',
                '创业板指': '399006.SZ'
            }
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'A股市场概览 ({period})', fontsize=16)
            
            colors = ['blue', 'green', 'red', 'purple']
            
            for i, (name, symbol) in enumerate(indices.items()):
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    
                    if not data.empty:
                        row = i // 2
                        col = i % 2
                        ax = axes[row, col]
                        self._plot_single_index(ax, data, name, colors[i])
                    else:
                        row = i // 2
                        col = i % 2
                        ax = axes[row, col]
                        ax.text(0.5, 0.5, f'无{name}数据', ha='center', va='center')
                        ax.set_title(name)
                        
                except Exception as e:
                    row = i // 2
                    col = i % 2
                    ax = axes[row, col]
                    ax.text(0.5, 0.5, f'{name}数据获取失败', ha='center', va='center')
                    ax.set_title(name)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"绘制市场概览失败: {str(e)}")