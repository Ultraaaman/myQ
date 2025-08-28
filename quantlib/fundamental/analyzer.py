import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# 尝试导入更多数据源
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

class FundamentalAnalyzer:
    """全面基本面分析工具"""
    
    def __init__(self, symbol, market='US'):
        """
        初始化分析器
        Args:
            symbol: 股票代码
            market: 市场类型 ('US', 'CN', 'HK')
        """
        self.symbol = symbol.upper()
        self.market = market
        self.ticker = None
        self.financial_data = {}
        self.ratios = {}
        self.peer_comparison = {}
        
    def load_company_data(self):
        """加载公司基本数据"""
        try:
            if self.market == 'US':
                self.ticker = yf.Ticker(self.symbol)
                self.company_info = self.ticker.info
                print(f"✓ 成功加载 {self.symbol} 的公司信息")
                return True
            elif self.market == 'CN' and AKSHARE_AVAILABLE:
                # 中国股票使用akshare
                self.company_info = self._get_cn_company_info()
                print(f"✓ 成功加载 {self.symbol} 的公司信息")
                return True
            else:
                print("不支持的市场或缺少相应的数据源")
                return False
        except Exception as e:
            print(f"加载公司数据失败: {e}")
            return False
    
    def _get_cn_company_info(self):
        """获取中国股票基本信息"""
        try:
            # 获取股票基本信息
            stock_info = ak.stock_individual_info_em(symbol=self.symbol)
            info_dict = {}
            for _, row in stock_info.iterrows():
                info_dict[row['item']] = row['value']
            return info_dict
        except Exception as e:
            print(f"获取中国股票信息失败: {e}")
            return {}
    
    def analyze_financial_statements(self, start_year="2020"):
        """分析财务报表"""
        print(f"\n{'='*60}")
        print("财务报表分析")
        print('='*60)
        
        if self.market == 'US' and self.ticker:
            return self._analyze_us_financials()
        elif self.market == 'CN':
            return self._analyze_cn_financials(start_year)
    
    def _analyze_us_financials(self):
        """分析美股财务数据"""
        try:
            # 获取财务数据
            financials = self.ticker.financials
            balance_sheet = self.ticker.balance_sheet  
            cash_flow = self.ticker.cashflow
            
            # 存储财务数据
            self.financial_data = {
                'income_statement': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
            
            print("财务报表数据获取成功:")
            print(f"  损益表: {financials.shape[1]} 个报告期")
            print(f"  资产负债表: {balance_sheet.shape[1]} 个报告期")  
            print(f"  现金流量表: {cash_flow.shape[1]} 个报告期")
            
            # 分析最近一期财务数据
            latest_period = financials.columns[0]
            print(f"\n最新财务数据 ({latest_period.strftime('%Y-%m-%d')}):")
            
            # 损益表关键项目
            if 'Total Revenue' in financials.index:
                revenue = financials.loc['Total Revenue', latest_period]
                print(f"  营业收入: ${revenue:,.0f}")
            
            if 'Net Income' in financials.index:
                net_income = financials.loc['Net Income', latest_period]
                print(f"  净利润: ${net_income:,.0f}")
            
            # 资产负债表关键项目
            if 'Total Assets' in balance_sheet.index:
                total_assets = balance_sheet.loc['Total Assets', latest_period]
                print(f"  总资产: ${total_assets:,.0f}")
                
            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Debt', latest_period]
                print(f"  总负债: ${total_debt:,.0f}")
            
            # 现金流关键项目
            if 'Operating Cash Flow' in cash_flow.index:
                operating_cf = cash_flow.loc['Operating Cash Flow', latest_period]
                print(f"  经营现金流: ${operating_cf:,.0f}")
            
            return True
            
        except Exception as e:
            print(f"分析美股财务数据失败: {e}")
            return False
    
    def _analyze_cn_financials(self, start_year="2020"):
        """分析中国股票财务数据"""
        if not AKSHARE_AVAILABLE:
            print("需要安装akshare: pip install akshare")
            return False
            
        try:
            # 获取财务指标
            financial_indicators = ak.stock_financial_analysis_indicator(symbol=self.symbol, start_year=start_year)
            
            if not financial_indicators.empty:
                # 按日期排序，确保最新数据在最后
                financial_indicators['日期'] = pd.to_datetime(financial_indicators['日期'])
                financial_indicators = financial_indicators.sort_values('日期', ascending=True)
                
                print("财务指标数据获取成功:")
                print(f"  数据期间: {len(financial_indicators)} 个报告期")
                print(f"  数据范围: {financial_indicators['日期'].iloc[0].strftime('%Y-%m-%d')} 至 {financial_indicators['日期'].iloc[-1].strftime('%Y-%m-%d')}")
                
                # 最新财务指标（按日期排序后的最后一行）
                latest = financial_indicators.iloc[-1]
                print(f"\n最新财务指标 ({latest.get('日期', 'N/A')}):")
                
                key_metrics = [
                    ('摊薄每股收益', '摊薄每股收益(元)'),
                    ('加权每股收益', '加权每股收益(元)'),
                    ('每股净资产', '每股净资产_调整后(元)'),
                    ('每股经营性现金流', '每股经营性现金流(元)'),
                    ('总资产净利润率', '总资产净利润率(%)'),
                    ('销售净利率', '销售净利率(%)'),
                    ('净资产收益率', '净资产收益率(%)'),
                    ('加权净资产收益率', '加权净资产收益率(%)'),
                    ('总资产', '总资产(元)'),
                    ('资产负债率', '资产负债率(%)'),
                    ('流动比率', '流动比率'),
                    ('速动比率', '速动比率')
                ]
                
                for name, key in key_metrics:
                    if key in latest.index:
                        value = latest[key]
                        if pd.notna(value):
                            if key.endswith('(%)'):
                                print(f"  {name}: {value:.2f}%")
                            elif key.endswith('(元)') and '每股' not in key:
                                print(f"  {name}: {value:,.0f} 元")
                            elif key.endswith('(元)') and '每股' in key:
                                print(f"  {name}: {value:.2f} 元")
                            else:
                                print(f"  {name}: {value:.2f}")
                
                self.financial_data['indicators'] = financial_indicators
                return True
            
        except Exception as e:
            print(f"分析中国股票财务数据失败: {e}")
            return False
    
    def calculate_financial_ratios(self, start_year="2020"):
        """计算财务比率"""
        print(f"\n{'='*60}")
        print("财务比率分析")
        print('='*60)
        
        if self.market == 'US':
            return self._calculate_us_ratios()
        elif self.market == 'CN':
            return self._calculate_cn_ratios(start_year)
    
    def _calculate_us_ratios(self):
        """计算美股财务比率"""
        try:
            if not self.financial_data:
                print("请先分析财务报表")
                return False
            
            financials = self.financial_data['income_statement']
            balance_sheet = self.financial_data['balance_sheet']
            
            # 获取股票价格信息
            hist_data = self.ticker.history(period='1d')
            current_price = hist_data['Close'][-1] if not hist_data.empty else None
            shares_outstanding = self.company_info.get('sharesOutstanding', None)
            
            ratios = {}
            latest_period = financials.columns[0]
            
            # 盈利能力比率
            if 'Total Revenue' in financials.index and 'Net Income' in financials.index:
                revenue = financials.loc['Total Revenue', latest_period]
                net_income = financials.loc['Net Income', latest_period]
                
                if revenue != 0:
                    ratios['净利率'] = (net_income / revenue) * 100
            
            # ROE和ROA
            if 'Net Income' in financials.index:
                net_income = financials.loc['Net Income', latest_period]
                
                if 'Total Stockholder Equity' in balance_sheet.index:
                    equity = balance_sheet.loc['Total Stockholder Equity', latest_period]
                    if equity != 0:
                        ratios['ROE'] = (net_income / equity) * 100
                
                if 'Total Assets' in balance_sheet.index:
                    assets = balance_sheet.loc['Total Assets', latest_period]
                    if assets != 0:
                        ratios['ROA'] = (net_income / assets) * 100
            
            # 成长性指标
            if len(financials.columns) >= 2:
                current_period = financials.columns[0]
                previous_period = financials.columns[1]
                
                # 营收增长率
                if 'Total Revenue' in financials.index:
                    current_revenue = financials.loc['Total Revenue', current_period]
                    previous_revenue = financials.loc['Total Revenue', previous_period]
                    if previous_revenue != 0 and pd.notna(previous_revenue):
                        ratios['Revenue Growth'] = ((current_revenue - previous_revenue) / previous_revenue) * 100
                
                # 净利润增长率
                if 'Net Income' in financials.index:
                    current_income = financials.loc['Net Income', current_period]
                    previous_income = financials.loc['Net Income', previous_period]
                    if previous_income != 0 and pd.notna(previous_income) and previous_income > 0:
                        ratios['Net Income Growth'] = ((current_income - previous_income) / previous_income) * 100
                
                # EPS增长率
                if shares_outstanding:
                    current_eps = financials.loc['Net Income', current_period] / shares_outstanding
                    previous_eps = financials.loc['Net Income', previous_period] / shares_outstanding
                    if previous_eps != 0 and pd.notna(previous_eps) and previous_eps > 0:
                        ratios['EPS Growth'] = ((current_eps - previous_eps) / previous_eps) * 100

            # 估值比率
            if current_price and shares_outstanding:
                market_cap = current_price * shares_outstanding
                
                if 'Net Income' in financials.index:
                    net_income = financials.loc['Net Income', latest_period]
                    if net_income > 0:
                        ratios['PE'] = market_cap / net_income
                
                if 'Total Stockholder Equity' in balance_sheet.index:
                    equity = balance_sheet.loc['Total Stockholder Equity', latest_period]
                    if equity > 0:
                        ratios['PB'] = market_cap / equity
                
                # 股息相关指标
                dividends_info = self.company_info.get('dividendYield', 0)
                if dividends_info:
                    ratios['Dividend Yield'] = dividends_info * 100
                
                payout_ratio = self.company_info.get('payoutRatio', 0)
                if payout_ratio:
                    ratios['Payout Ratio'] = payout_ratio * 100
            
            # 资产质量指标
            if 'Total Assets' in balance_sheet.index:
                total_assets = balance_sheet.loc['Total Assets', latest_period]
                
                # 商誉占比
                if 'Goodwill' in balance_sheet.index:
                    goodwill = balance_sheet.loc['Goodwill', latest_period]
                    if pd.notna(goodwill) and total_assets != 0:
                        ratios['Goodwill Ratio'] = (goodwill / total_assets) * 100
                
                # 无形资产比例
                if 'Intangible Assets' in balance_sheet.index:
                    intangible = balance_sheet.loc['Intangible Assets', latest_period]
                    if pd.notna(intangible) and total_assets != 0:
                        ratios['Intangible Assets Ratio'] = (intangible / total_assets) * 100
                
                # 有形资产净值比率
                tangible_assets = total_assets
                if 'Goodwill' in balance_sheet.index:
                    tangible_assets -= balance_sheet.loc['Goodwill', latest_period] or 0
                if 'Intangible Assets' in balance_sheet.index:
                    tangible_assets -= balance_sheet.loc['Intangible Assets', latest_period] or 0
                
                if 'Total Stockholder Equity' in balance_sheet.index:
                    equity = balance_sheet.loc['Total Stockholder Equity', latest_period]
                    if equity != 0:
                        ratios['Tangible Book Value Ratio'] = (tangible_assets - (balance_sheet.loc['Total Liabilities Net Minority Interest', latest_period] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else 0)) / equity

            # 现金流质量指标
            cash_flow = self.financial_data.get('cash_flow')
            if cash_flow is not None and 'Operating Cash Flow' in cash_flow.index:
                operating_cf = cash_flow.loc['Operating Cash Flow', latest_period]
                
                # 经营现金流/净利润比率
                if 'Net Income' in financials.index:
                    net_income = financials.loc['Net Income', latest_period]
                    if net_income != 0 and pd.notna(net_income) and net_income > 0:
                        ratios['Operating CF to Net Income'] = (operating_cf / net_income) * 100
                
                # 自由现金流
                capex = 0
                if 'Capital Expenditures' in cash_flow.index:
                    capex = abs(cash_flow.loc['Capital Expenditures', latest_period] or 0)
                
                free_cash_flow = operating_cf - capex
                ratios['Free Cash Flow'] = free_cash_flow
                
                # 自由现金流收益率
                if shares_outstanding and current_price:
                    market_cap = current_price * shares_outstanding
                    if market_cap != 0:
                        ratios['Free Cash Flow Yield'] = (free_cash_flow / market_cap) * 100

            # 偿债能力比率
            if 'Current Assets' in balance_sheet.index and 'Current Liab' in balance_sheet.index:
                current_assets = balance_sheet.loc['Current Assets', latest_period]
                current_liab = balance_sheet.loc['Current Liab', latest_period]
                if current_liab != 0:
                    ratios['流动比率'] = current_assets / current_liab
            
            if 'Total Debt' in balance_sheet.index and 'Total Assets' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Debt', latest_period]
                total_assets = balance_sheet.loc['Total Assets', latest_period]
                if total_assets != 0:
                    ratios['资产负债率'] = (total_debt / total_assets) * 100
            
            # 市场表现指标
            market_ratios = self._calculate_market_performance_ratios()
            ratios.update(market_ratios)
            
            self.ratios = ratios
            
            print("关键财务比率:")
            for ratio_name, ratio_value in ratios.items():
                if '率' in ratio_name or 'ROE' in ratio_name or 'ROA' in ratio_name or 'Growth' in ratio_name or 'Yield' in ratio_name or 'Ratio' in ratio_name and ratio_name not in ['流动比率']:
                    print(f"  {ratio_name}: {ratio_value:.2f}%")
                else:
                    print(f"  {ratio_name}: {ratio_value:.2f}")
            
            return True
            
        except Exception as e:
            print(f"计算美股财务比率失败: {e}")
            return False
    
    def _calculate_market_performance_ratios(self):
        """计算市场表现指标"""
        market_ratios = {}
        
        try:
            if self.market == 'US' and self.ticker:
                # 获取历史价格数据计算波动率和beta
                hist_data = self.ticker.history(period="2y")  # 2年数据
                
                if not hist_data.empty and len(hist_data) > 252:  # 至少一年数据
                    # 计算日收益率
                    returns = hist_data['Close'].pct_change().dropna()
                    
                    # 计算年化波动率
                    volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率(%)
                    market_ratios['Volatility'] = volatility
                    
                    # 计算beta（相对于SPY的beta）
                    try:
                        spy = yf.Ticker("SPY")
                        spy_hist = spy.history(period="2y")
                        
                        if not spy_hist.empty:
                            spy_returns = spy_hist['Close'].pct_change().dropna()
                            
                            # 对齐时间序列
                            common_dates = returns.index.intersection(spy_returns.index)
                            if len(common_dates) > 100:
                                stock_aligned = returns.loc[common_dates]
                                market_aligned = spy_returns.loc[common_dates]
                                
                                # 计算beta
                                covariance = np.cov(stock_aligned, market_aligned)[0, 1]
                                market_variance = np.var(market_aligned)
                                if market_variance != 0:
                                    beta = covariance / market_variance
                                    market_ratios['Beta'] = beta
                    except:
                        pass  # 如果无法获取SPY数据，跳过beta计算
                    
                    # 计算夏普比率（假设无风险利率为3%）
                    risk_free_rate = 0.03
                    excess_returns = returns.mean() * 252 - risk_free_rate  # 年化超额收益
                    if volatility > 0:
                        sharpe_ratio = excess_returns / (volatility / 100)
                        market_ratios['Sharpe Ratio'] = sharpe_ratio
                    
                    # 价格趋势指标
                    if len(hist_data) >= 50:
                        # 相对强弱指标 (简化版)
                        price_change_1m = (hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-22]) / hist_data['Close'].iloc[-22] * 100 if len(hist_data) >= 22 else 0
                        price_change_3m = (hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-66]) / hist_data['Close'].iloc[-66] * 100 if len(hist_data) >= 66 else 0
                        
                        market_ratios['1M Price Change'] = price_change_1m
                        market_ratios['3M Price Change'] = price_change_3m
            
            elif self.market == 'CN' and AKSHARE_AVAILABLE:
                # A股市场表现指标
                try:
                    # 获取历史价格数据
                    stock_hist = ak.stock_zh_a_hist(symbol=self.symbol, period="daily", adjust="qfq")
                    
                    if not stock_hist.empty and len(stock_hist) > 252:
                        # 计算日收益率
                        stock_hist['收盘'] = pd.to_numeric(stock_hist['收盘'], errors='coerce')
                        returns = stock_hist['收盘'].pct_change().dropna()
                        
                        # 年化波动率
                        volatility = returns.std() * np.sqrt(252) * 100
                        market_ratios['Volatility'] = volatility
                        
                        # 获取沪深300作为市场基准计算beta
                        try:
                            market_hist = ak.stock_zh_index_daily(symbol="sh000300")  # 沪深300
                            if not market_hist.empty:
                                market_hist['close'] = pd.to_numeric(market_hist['close'], errors='coerce')
                                market_returns = market_hist['close'].pct_change().dropna()
                                
                                # 对齐时间序列（简化处理）
                                min_len = min(len(returns), len(market_returns))
                                if min_len > 100:
                                    stock_ret = returns.iloc[-min_len:]
                                    market_ret = market_returns.iloc[-min_len:]
                                    
                                    covariance = np.cov(stock_ret, market_ret)[0, 1]
                                    market_variance = np.var(market_ret)
                                    if market_variance != 0:
                                        beta = covariance / market_variance
                                        market_ratios['Beta'] = beta
                        except:
                            pass
                        
                        # 价格变化趋势
                        if len(stock_hist) >= 22:
                            current_price = stock_hist['收盘'].iloc[-1]
                            price_1m = stock_hist['收盘'].iloc[-22] if len(stock_hist) >= 22 else current_price
                            price_3m = stock_hist['收盘'].iloc[-66] if len(stock_hist) >= 66 else current_price
                            
                            if price_1m != 0:
                                market_ratios['1M Price Change'] = (current_price - price_1m) / price_1m * 100
                            if price_3m != 0:
                                market_ratios['3M Price Change'] = (current_price - price_3m) / price_3m * 100
                
                except Exception as e:
                    print(f"计算A股市场指标失败: {e}")
            
        except Exception as e:
            print(f"计算市场表现指标失败: {e}")
        
        return market_ratios
    
    def print_detailed_ratios_summary(self):
        """打印详细的分类指标汇总"""
        if not self.ratios:
            print("没有可显示的财务比率数据")
            return
        
        print(f"\n{'='*80}")
        print("详细财务指标分类汇总")
        print('='*80)
        
        # 盈利能力指标
        profitability_ratios = {}
        for key in ['ROE', 'ROA', '净利率', '毛利率', '营业利润率', '总资产利润率']:
            if key in self.ratios:
                profitability_ratios[key] = self.ratios[key]
        
        if profitability_ratios:
            print("\n📈 盈利能力指标:")
            for name, value in profitability_ratios.items():
                print(f"  {name:12}: {value:8.2f}%")
        
        # 成长性指标
        growth_ratios = {}
        for key in ['Revenue Growth', 'Net Income Growth', 'EPS Growth', 'Total Assets Growth']:
            if key in self.ratios:
                growth_ratios[key] = self.ratios[key]
        
        if growth_ratios:
            print("\n🚀 成长性指标:")
            for name, value in growth_ratios.items():
                print(f"  {name:18}: {value:8.2f}%")
        
        # 现金流质量指标
        cashflow_ratios = {}
        for key in ['Operating CF to Net Income', 'Free Cash Flow', 'Free Cash Flow Yield']:
            if key in self.ratios:
                cashflow_ratios[key] = self.ratios[key]
        
        if cashflow_ratios:
            print("\n💰 现金流质量指标:")
            for name, value in cashflow_ratios.items():
                if 'Yield' in name or 'Net Income' in name:
                    print(f"  {name:25}: {value:8.2f}%")
                else:
                    print(f"  {name:25}: {value:,.0f}")
        
        # 偿债能力指标
        solvency_ratios = {}
        for key in ['资产负债率', '流动比率', '速动比率', '股东权益比率']:
            if key in self.ratios:
                solvency_ratios[key] = self.ratios[key]
        
        if solvency_ratios:
            print("\n🛡️ 偿债能力指标:")
            for name, value in solvency_ratios.items():
                if '比率' in name and name not in ['流动比率', '速动比率']:
                    print(f"  {name:12}: {value:8.2f}%")
                else:
                    print(f"  {name:12}: {value:8.2f}")
        
        # 营运能力指标
        efficiency_ratios = {}
        for key in ['存货周转率', '应收账款周转率', '总资产周转率']:
            if key in self.ratios:
                efficiency_ratios[key] = self.ratios[key]
        
        if efficiency_ratios:
            print("\n⚡ 营运能力指标:")
            for name, value in efficiency_ratios.items():
                print(f"  {name:15}: {value:8.2f}次")
        
        # 估值指标
        valuation_ratios = {}
        for key in ['PE', 'PB']:
            if key in self.ratios:
                valuation_ratios[key] = self.ratios[key]
        
        if valuation_ratios:
            print("\n💎 估值指标:")
            for name, value in valuation_ratios.items():
                print(f"  {name:12}: {value:8.2f}倍")
        
        # 股息指标
        dividend_ratios = {}
        for key in ['Dividend Yield', 'Payout Ratio']:
            if key in self.ratios:
                dividend_ratios[key] = self.ratios[key]
        
        if dividend_ratios:
            print("\n💵 股息指标:")
            for name, value in dividend_ratios.items():
                print(f"  {name:15}: {value:8.2f}%")
        
        # 资产质量指标
        quality_ratios = {}
        for key in ['Goodwill Ratio', 'Intangible Assets Ratio', 'Tangible Book Value Ratio']:
            if key in self.ratios:
                quality_ratios[key] = self.ratios[key]
        
        if quality_ratios:
            print("\n🏗️ 资产质量指标:")
            for name, value in quality_ratios.items():
                if 'Ratio' in name and name != 'Tangible Book Value Ratio':
                    print(f"  {name:25}: {value:8.2f}%")
                else:
                    print(f"  {name:25}: {value:8.2f}")
        
        # 市场表现指标
        market_ratios = {}
        for key in ['Beta', 'Volatility', 'Sharpe Ratio', '1M Price Change', '3M Price Change']:
            if key in self.ratios:
                market_ratios[key] = self.ratios[key]
        
        if market_ratios:
            print("\n📊 市场表现指标:")
            for name, value in market_ratios.items():
                if 'Change' in name or 'Volatility' in name:
                    print(f"  {name:18}: {value:8.2f}%")
                else:
                    print(f"  {name:18}: {value:8.2f}")
        
        print(f"\n{'='*80}")
    
    def _calculate_cn_ratios(self, start_year="2020"):
        """计算中国股票财务比率"""
        try:
            if 'indicators' not in self.financial_data:
                print("请先分析财务报表")
                return False
            
            indicators = self.financial_data['indicators']
            # 使用排序后的最新数据
            latest = indicators.iloc[-1]
            
            # 从财务指标中提取关键比率
            ratios = {}
            
            ratio_mappings = {
                'ROE': '净资产收益率(%)',
                'ROA': '总资产净利润率(%)',
                '净利率': '销售净利率(%)',
                '毛利率': '销售毛利率(%)',
                '营业利润率': '营业利润率(%)',
                '总资产利润率': '总资产利润率(%)',
                '资产负债率': '资产负债率(%)',
                '流动比率': '流动比率',
                '速动比率': '速动比率',
                '存货周转率': '存货周转率(次)',
                '应收账款周转率': '应收账款周转率(次)',
                '总资产周转率': '总资产周转率(次)',
                '股东权益比率': '股东权益比率(%)'
            }
            
            # 计算成长性指标
            growth_mappings = {
                'Revenue Growth': '主营业务收入增长率(%)',
                'Net Income Growth': '净利润增长率(%)',
                'Total Assets Growth': '总资产增长率(%)'
            }
            
            for ratio_name, column_name in ratio_mappings.items():
                if column_name in latest.index:
                    value = latest[column_name]
                    if pd.notna(value):
                        ratios[ratio_name] = float(value)
            
            # 添加成长性指标
            for ratio_name, column_name in growth_mappings.items():
                if column_name in latest.index:
                    value = latest[column_name]
                    if pd.notna(value):
                        ratios[ratio_name] = float(value)
            
            # 计算现金流质量指标
            if len(indicators) >= 2:
                # 使用最近两期数据计算现金流质量
                prev_latest = indicators.iloc[-2]
                
                # 经营现金流/净利润比率
                if '每股经营性现金流(元)' in latest.index and '摊薄每股收益(元)' in latest.index:
                    ocf_per_share = latest['每股经营性现金流(元)']
                    eps = latest['摊薄每股收益(元)']
                    if pd.notna(ocf_per_share) and pd.notna(eps) and eps != 0:
                        ratios['Operating CF to Net Income'] = (ocf_per_share / eps) * 100
            
            # 获取估值数据 - 使用多种方法
            try:
                # 方法1: 从个股信息获取
                stock_info = ak.stock_individual_info_em(symbol=self.symbol)
                pe_found = False
                pb_found = False
                
                print(f"调试：获取到 {len(stock_info)} 个信息字段")
                
                for _, row in stock_info.iterrows():
                    item_name = row['item']
                    value = row['value']
                    
                    # 更全面的PE搜索关键字
                    pe_keywords = ['市盈率', 'PE', 'P/E', '盈率', '动态市盈率', '静态市盈率', 'TTM市盈率']
                    if not pe_found and any(keyword in item_name for keyword in pe_keywords):
                        try:
                            # 处理可能的字符串格式
                            if isinstance(value, str):
                                value = value.replace(',', '').replace('倍', '')
                            pe_val = float(value)
                            if pe_val > 0 and pe_val < 1000:  # 合理的PE范围
                                ratios['PE'] = pe_val
                                pe_found = True
                                print(f"找到PE: {item_name} = {pe_val}")
                        except (ValueError, TypeError):
                            pass
                    
                    # 更全面的PB搜索关键字
                    pb_keywords = ['市净率', 'PB', 'P/B', '净率']
                    if not pb_found and any(keyword in item_name for keyword in pb_keywords):
                        try:
                            # 处理可能的字符串格式
                            if isinstance(value, str):
                                value = value.replace(',', '').replace('倍', '')
                            pb_val = float(value)
                            if pb_val > 0 and pb_val < 100:  # 合理的PB范围
                                ratios['PB'] = pb_val
                                pb_found = True
                                print(f"找到PB: {item_name} = {pb_val}")
                        except (ValueError, TypeError):
                            pass
                
                
                
                # 方法3: 手动计算PE/PB（如果有必要数据）
                if not pe_found or not pb_found:
                    try:
                        # 尝试从财务指标获取EPS和股价来计算PE
                        if 'indicators' in self.financial_data and not pe_found:
                            indicators = self.financial_data['indicators']
                            latest_financial = indicators.iloc[-1]
                            
                            # 寻找每股收益
                            eps_candidates = ['摊薄每股收益(元)', '基本每股收益(元)', '每股收益(元)']
                            for eps_col in eps_candidates:
                                if eps_col in latest_financial.index and pd.notna(latest_financial[eps_col]):
                                    eps = float(latest_financial[eps_col])
                                    if eps > 0:
                                        # 获取当前股价
                                        try:
                                            stock_zh_a_hist = ak.stock_zh_a_hist(symbol=self.symbol, period="daily", adjust="")
                                            if not stock_zh_a_hist.empty:
                                                current_price = stock_zh_a_hist.iloc[-1]['收盘']
                                                pe_calculated = current_price / eps
                                                if pe_calculated > 0 and pe_calculated < 1000:
                                                    ratios['PE'] = pe_calculated
                                                    pe_found = True
                                                    print(f"手动计算PE: {current_price} / {eps} = {pe_calculated:.2f}")
                                                break
                                        except:
                                            pass
                        
                        # 尝试从财务指标获取每股净资产来计算PB
                        if 'indicators' in self.financial_data and not pb_found:
                            indicators = self.financial_data['indicators']
                            latest_financial = indicators.iloc[-1]
                            
                            # 寻找每股净资产
                            bps_candidates = ['每股净资产_调整后(元)', '每股净资产(元)', '每股账面价值(元)']
                            for bps_col in bps_candidates:
                                if bps_col in latest_financial.index and pd.notna(latest_financial[bps_col]):
                                    bps = float(latest_financial[bps_col])
                                    if bps > 0:
                                        # 获取当前股价
                                        try:
                                            stock_zh_a_hist = ak.stock_zh_a_hist(symbol=self.symbol, period="daily", adjust="")
                                            if not stock_zh_a_hist.empty:
                                                current_price = stock_zh_a_hist.iloc[-1]['收盘']
                                                pb_calculated = current_price / bps
                                                if pb_calculated > 0 and pb_calculated < 100:
                                                    ratios['PB'] = pb_calculated
                                                    pb_found = True
                                                    print(f"手动计算PB: {current_price} / {bps} = {pb_calculated:.2f}")
                                                break
                                        except:
                                            pass
                    except Exception as e3:
                        print(f"手动计算PE/PB失败: {e3}")
                
                if not pe_found:
                    print("未能获取PE数据")
                if not pb_found:
                    print("未能获取PB数据")
                        
            except Exception as e:
                print(f"获取估值数据时出错: {e}")
            
            # 添加市场表现指标（中国股票）
            market_ratios = self._calculate_market_performance_ratios()
            ratios.update(market_ratios)
            
            self.ratios = ratios
            
            print("关键财务比率:")
            for ratio_name, ratio_value in ratios.items():
                if ('率' in ratio_name or 'ROE' in ratio_name or 'ROA' in ratio_name or 'Growth' in ratio_name or 'Yield' in ratio_name or 'Volatility' in ratio_name or 'Change' in ratio_name) and ratio_name not in ['PE', 'PB', '流动比率', '速动比率', 'Beta', 'Sharpe Ratio']:
                    print(f"  {ratio_name}: {ratio_value:.2f}%")
                else:
                    print(f"  {ratio_name}: {ratio_value:.2f}")
            
            return True
            
        except Exception as e:
            print(f"计算中国股票财务比率失败: {e}")
            return False
    
    def _get_cn_stock_ratios(self, symbol, start_year="2020"):
        """获取中国股票的财务比率"""
        try:
            # 获取财务指标
            financial_indicators = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
            
            if financial_indicators.empty:
                return {}
                
            # 按日期排序，确保最新数据在最后
            financial_indicators['日期'] = pd.to_datetime(financial_indicators['日期'])
            financial_indicators = financial_indicators.sort_values('日期', ascending=True)
            
            # 使用最新数据
            latest = financial_indicators.iloc[-1]
            
            # 财务比率映射
            ratio_mappings = {
                'ROE': '净资产收益率(%)',
                'ROA': '总资产净利润率(%)',
                '净利率': '销售净利率(%)',
                '毛利率': '销售毛利率(%)',
                '营业利润率': '营业利润率(%)',
                '总资产利润率': '总资产利润率(%)',
                '资产负债率': '资产负债率(%)',
                '流动比率': '流动比率',
                '速动比率': '速动比率',
                '股东权益比率': '股东权益比率(%)'
            }
            
            ratios = {}
            for ratio_name, column_name in ratio_mappings.items():
                if column_name in latest.index:
                    value = latest[column_name]
                    if pd.notna(value):
                        ratios[ratio_name] = float(value)
            
            # 计算营收增长率
            if len(financial_indicators) >= 2 and '主营业务收入增长率(%)' in latest.index:
                revenue_growth = latest['主营业务收入增长率(%)']
                if pd.notna(revenue_growth):
                    ratios['Revenue Growth'] = float(revenue_growth)
            
            # 计算负债权益比 (Debt/Equity)
            if '资产负债率' in ratios and '股东权益比率' in ratios:
                debt_ratio = ratios['资产负债率'] / 100  # 转换为小数
                equity_ratio = ratios['股东权益比率'] / 100
                if equity_ratio > 0:
                    ratios['Debt/Equity'] = debt_ratio / equity_ratio
            elif '资产负债率' in ratios:
                # 使用资产负债率估算
                debt_ratio = ratios['资产负债率'] / 100
                if debt_ratio < 1:
                    ratios['Debt/Equity'] = debt_ratio / (1 - debt_ratio)
            
            # 添加PE和PB计算 - 使用与calculate_financial_ratios相同的逻辑
            try:
                # 从个股信息获取PE和PB
                stock_info = ak.stock_individual_info_em(symbol=symbol)
                pe_found = False
                pb_found = False
                
                for _, row in stock_info.iterrows():
                    item_name = row['item']
                    value = row['value']
                    
                    # PE搜索
                    pe_keywords = ['市盈率', 'PE', 'P/E', '盈率', '动态市盈率', '静态市盈率', 'TTM市盈率']
                    if not pe_found and any(keyword in item_name for keyword in pe_keywords):
                        try:
                            if isinstance(value, str):
                                value = value.replace(',', '').replace('倍', '')
                            pe_val = float(value)
                            if pe_val > 0 and pe_val < 1000:
                                ratios['PE'] = pe_val
                                pe_found = True
                        except (ValueError, TypeError):
                            pass
                    
                    # PB搜索
                    pb_keywords = ['市净率', 'PB', 'P/B', '净率']
                    if not pb_found and any(keyword in item_name for keyword in pb_keywords):
                        try:
                            if isinstance(value, str):
                                value = value.replace(',', '').replace('倍', '')
                            pb_val = float(value)
                            if pb_val > 0 and pb_val < 100:
                                ratios['PB'] = pb_val
                                pb_found = True
                        except (ValueError, TypeError):
                            pass
                
                # 手动计算PE/PB（如果没有找到）
                if not pe_found or not pb_found:
                    try:
                        # 寻找每股收益来计算PE
                        if not pe_found:
                            eps_candidates = ['摊薄每股收益(元)', '基本每股收益(元)', '每股收益(元)']
                            for eps_col in eps_candidates:
                                if eps_col in latest.index and pd.notna(latest[eps_col]):
                                    eps = float(latest[eps_col])
                                    if eps > 0:
                                        try:
                                            stock_zh_a_hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
                                            if not stock_zh_a_hist.empty:
                                                current_price = stock_zh_a_hist.iloc[-1]['收盘']
                                                pe_calculated = current_price / eps
                                                if pe_calculated > 0 and pe_calculated < 1000:
                                                    ratios['PE'] = pe_calculated
                                                    pe_found = True
                                                break
                                        except:
                                            pass
                        
                        # 寻找每股净资产来计算PB
                        if not pb_found:
                            bps_candidates = ['每股净资产_调整后(元)', '每股净资产(元)', '每股账面价值(元)']
                            for bps_col in bps_candidates:
                                if bps_col in latest.index and pd.notna(latest[bps_col]):
                                    bps = float(latest[bps_col])
                                    if bps > 0:
                                        try:
                                            stock_zh_a_hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
                                            if not stock_zh_a_hist.empty:
                                                current_price = stock_zh_a_hist.iloc[-1]['收盘']
                                                pb_calculated = current_price / bps
                                                if pb_calculated > 0 and pb_calculated < 100:
                                                    ratios['PB'] = pb_calculated
                                                    pb_found = True
                                                break
                                        except:
                                            pass
                    except Exception:
                        pass
                        
            except Exception:
                pass
                    
            return ratios
            
        except Exception as e:
            print(f"获取 {symbol} 财务比率失败: {e}")
            return {}
    
    def peer_comparison_analysis(self, peer_symbols, start_year="2020"):
        """同行对比分析"""
        print(f"\n{'='*60}")
        print("同行对比分析")
        print('='*60)
        
        comparison_data = []
        all_symbols = [self.symbol] + peer_symbols
        
        for symbol in all_symbols:
            try:
                if self.market == 'US':
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    data = {
                        'Symbol': symbol,
                        'Company': info.get('shortName', symbol),
                        'Market Cap': info.get('marketCap', 0) / 1e9,  # 转换为十亿
                        'PE': info.get('forwardPE', 0),
                        'PB': info.get('priceToBook', 0),
                        'ROE': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                        'Debt/Equity': info.get('debtToEquity', 0),
                        'Revenue Growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
                    }
                    
                elif self.market == 'CN' and AKSHARE_AVAILABLE:
                    # 获取中国股票基本信息
                    stock_info = ak.stock_individual_info_em(symbol=symbol)
                    info_dict = {}
                    for _, row in stock_info.iterrows():
                        info_dict[row['item']] = row['value']
                    
                    # 获取财务比率数据（现在包含PE和PB）
                    ratios = self._get_cn_stock_ratios(symbol, start_year)
                    
                    # 安全获取数值函数
                    def safe_float(value, default=0):
                        try:
                            if value is None or value == '' or value == '-':
                                return default
                            return float(str(value).replace(',', ''))
                        except (ValueError, TypeError):
                            return default
                    
                    data = {
                        'Symbol': symbol,
                        'Company': info_dict.get('股票简称', symbol),
                        'Market Cap': safe_float(info_dict.get('总市值', 0)) / 1e8,  # 转换为亿
                        'PE': ratios.get('PE', 0),  # 现在从ratios获取，而不是info_dict
                        'PB': ratios.get('PB', 0),  # 现在从ratios获取，而不是info_dict
                        'ROE': ratios.get('ROE', 0),
                        'ROA': ratios.get('ROA', 0),
                        'Net Margin': ratios.get('净利率', 0),
                        'Asset Turnover': ratios.get('总资产周转率', 0),
                        'Debt/Equity': ratios.get('Debt/Equity', 0),
                        'Current Ratio': ratios.get('流动比率', 0),
                        'Revenue Growth': ratios.get('Revenue Growth', 0)
                    }
                
                comparison_data.append(data)
                print(f"✓ 获取 {symbol} 数据")
                
            except Exception as e:
                print(f"✗ 获取 {symbol} 数据失败: {e}")
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            self.peer_comparison = comparison_df
            
            print(f"\n同行对比表:")
            print(comparison_df.round(2).to_string(index=False))
            
            # 计算行业平均值
            if len(comparison_df) > 1:
                print(f"\n行业平均值:")
                if self.market == 'CN':
                    numeric_columns = ['Market Cap', 'PE', 'PB', 'ROE', 'ROA', 'Net Margin', 'Debt/Equity', 'Current Ratio', 'Revenue Growth']
                else:
                    numeric_columns = ['Market Cap', 'PE', 'PB', 'ROE', 'Debt/Equity', 'Revenue Growth']
                    
                for col in numeric_columns:
                    if col in comparison_df.columns:
                        # 过滤掉0值来计算更准确的平均值
                        non_zero_values = comparison_df[comparison_df[col] != 0][col]
                        if len(non_zero_values) > 0:
                            avg_value = non_zero_values.mean()
                            target_value = comparison_df[comparison_df['Symbol'] == self.symbol][col].iloc[0]
                            
                            # 格式化显示
                            if col in ['PE', 'PB', 'Market Cap']:
                                print(f"  {col}: 行业均值 {avg_value:.2f}, {self.symbol} {target_value:.2f}")
                            else:
                                print(f"  {col}: 行业均值 {avg_value:.2f}%, {self.symbol} {target_value:.2f}%")
            
            return comparison_df
        
        return None
    
    def dcf_valuation(self, growth_years=5, terminal_growth=2.5, discount_rate=10):
        """DCF估值模型"""
        print(f"\n{'='*60}")
        print("DCF估值分析")
        print('='*60)
        
        try:
            if self.market == 'US' and self.ticker:
                # 获取现金流数据
                cash_flow = self.ticker.cashflow
                
                if 'Operating Cash Flow' not in cash_flow.index:
                    print("无法获取经营现金流数据")
                    return None
                
                # 最近一年的自由现金流
                latest_period = cash_flow.columns[0]
                operating_cf = cash_flow.loc['Operating Cash Flow', latest_period]
                
                # 估算资本支出（如果有的话）
                capex = 0
                if 'Capital Expenditures' in cash_flow.index:
                    capex = abs(cash_flow.loc['Capital Expenditures', latest_period])
                
                free_cash_flow = operating_cf - capex
                
                print(f"基础数据:")
                print(f"  经营现金流: ${operating_cf:,.0f}")
                print(f"  资本支出: ${capex:,.0f}")
                print(f"  自由现金流: ${free_cash_flow:,.0f}")
                
                # 计算历史现金流增长率
                if len(cash_flow.columns) >= 3:
                    cf_values = []
                    for col in cash_flow.columns[:3]:
                        if 'Operating Cash Flow' in cash_flow.index:
                            cf = cash_flow.loc['Operating Cash Flow', col]
                            if pd.notna(cf) and cf > 0:
                                cf_values.append(cf)
                    
                    if len(cf_values) >= 2:
                        growth_rates = []
                        for i in range(1, len(cf_values)):
                            growth_rate = (cf_values[i-1] / cf_values[i] - 1) * 100
                            growth_rates.append(growth_rate)
                        
                        historical_growth = np.mean(growth_rates)
                        print(f"  历史现金流增长率: {historical_growth:.1f}%")
                
                # DCF计算
                print(f"\nDCF假设:")
                print(f"  预测期: {growth_years} 年")
                print(f"  年增长率: 假设10% (可调整)")
                print(f"  永续增长率: {terminal_growth}%")
                print(f"  折现率: {discount_rate}%")
                
                # 预测未来现金流
                future_cf = []
                growth_rate = 0.10  # 假设10%增长率
                
                for year in range(1, growth_years + 1):
                    projected_cf = free_cash_flow * ((1 + growth_rate) ** year)
                    future_cf.append(projected_cf)
                
                # 计算预测期现值
                pv_future_cf = []
                for i, cf in enumerate(future_cf):
                    pv = cf / ((1 + discount_rate/100) ** (i + 1))
                    pv_future_cf.append(pv)
                
                # 计算终值
                terminal_cf = future_cf[-1] * (1 + terminal_growth/100)
                terminal_value = terminal_cf / (discount_rate/100 - terminal_growth/100)
                pv_terminal_value = terminal_value / ((1 + discount_rate/100) ** growth_years)
                
                # 企业价值
                enterprise_value = sum(pv_future_cf) + pv_terminal_value
                
                print(f"\nDCF计算结果:")
                print(f"  预测期现金流现值: ${sum(pv_future_cf):,.0f}")
                print(f"  终值现值: ${pv_terminal_value:,.0f}")
                print(f"  企业价值: ${enterprise_value:,.0f}")
                
                # 计算每股价值（需要股本数据）
                shares_outstanding = self.company_info.get('sharesOutstanding')
                if shares_outstanding:
                    # 减去净债务得到股权价值
                    balance_sheet = self.ticker.balance_sheet
                    net_debt = 0
                    if 'Total Debt' in balance_sheet.index and 'Cash' in balance_sheet.index:
                        total_debt = balance_sheet.loc['Total Debt', balance_sheet.columns[0]]
                        cash = balance_sheet.loc['Cash', balance_sheet.columns[0]]
                        net_debt = total_debt - cash
                    
                    equity_value = enterprise_value - net_debt
                    value_per_share = equity_value / shares_outstanding
                    
                    # 获取当前股价
                    current_price = self.ticker.history(period='1d')['Close'][-1]
                    
                    print(f"  净债务: ${net_debt:,.0f}")
                    print(f"  股权价值: ${equity_value:,.0f}")
                    print(f"  每股内在价值: ${value_per_share:.2f}")
                    print(f"  当前股价: ${current_price:.2f}")
                    
                    upside_potential = (value_per_share / current_price - 1) * 100
                    print(f"  上涨空间: {upside_potential:+.1f}%")
                
                return {
                    'enterprise_value': enterprise_value,
                    'value_per_share': value_per_share if shares_outstanding else None,
                    'current_price': current_price if shares_outstanding else None,
                    'upside_potential': upside_potential if shares_outstanding else None
                }
                
        except Exception as e:
            print(f"DCF估值计算失败: {e}")
            return None
    
    def generate_investment_summary(self):
        """生成投资分析摘要"""
        print(f"\n{'='*60}")
        print(f"{self.symbol} 投资分析摘要")
        print('='*60)
        
        if hasattr(self, 'company_info') and self.company_info:
            if self.market == 'US':
                company_name = self.company_info.get('shortName', self.symbol)
                sector = self.company_info.get('sector', 'N/A')
                industry = self.company_info.get('industry', 'N/A')
                
                print(f"公司: {company_name}")
                print(f"行业: {sector} - {industry}")
                
                # 获取当前股价
                hist = self.ticker.history(period='1d')
                if not hist.empty:
                    current_price = hist['Close'][-1]
                    print(f"当前股价: ${current_price:.2f}")
                
            elif self.market == 'CN':
                company_name = self.company_info.get('股票简称', self.symbol)
                print(f"公司: {company_name}")
        
        # 财务健康度评分
        health_score = self._calculate_financial_health_score()
        print(f"\n财务健康度评分: {health_score}/100")
        
        # 投资建议
        recommendation = self._generate_recommendation()
        print(f"投资建议: {recommendation}")
        
        # 风险提示
        risks = self._identify_risks()
        if risks:
            print(f"\n风险提示:")
            for risk in risks:
                print(f"  - {risk}")
    
    def _calculate_financial_health_score(self):
        """计算财务健康度评分（更新版，包含新增指标）"""
        score = 0
        max_score = 100
        
        if not self.ratios:
            return score
        
        # 盈利能力评分 (25分)
        if 'ROE' in self.ratios:
            roe = self.ratios['ROE']
            if roe > 15:
                score += 10
            elif roe > 10:
                score += 8
            elif roe > 5:
                score += 5
        
        if '净利率' in self.ratios:
            net_margin = self.ratios['净利率']
            if net_margin > 10:
                score += 10
            elif net_margin > 5:
                score += 8
            elif net_margin > 0:
                score += 5
        
        # 新增：现金流质量评分 (5分)
        if 'Operating CF to Net Income' in self.ratios:
            cf_quality = self.ratios['Operating CF to Net Income']
            if cf_quality > 120:
                score += 5
            elif cf_quality > 90:
                score += 3
            elif cf_quality > 60:
                score += 1
        
        # 成长性评分 (15分)
        if 'Revenue Growth' in self.ratios:
            revenue_growth = self.ratios['Revenue Growth']
            if revenue_growth > 15:
                score += 8
            elif revenue_growth > 5:
                score += 5
            elif revenue_growth > 0:
                score += 3
        
        if 'Net Income Growth' in self.ratios:
            income_growth = self.ratios['Net Income Growth']
            if income_growth > 20:
                score += 7
            elif income_growth > 10:
                score += 5
            elif income_growth > 0:
                score += 3
        
        # 偿债能力评分 (20分)
        if '资产负债率' in self.ratios:
            debt_ratio = self.ratios['资产负债率']
            if debt_ratio < 30:
                score += 10
            elif debt_ratio < 50:
                score += 8
            elif debt_ratio < 70:
                score += 5
        
        if '流动比率' in self.ratios:
            current_ratio = self.ratios['流动比率']
            if current_ratio > 2:
                score += 10
            elif current_ratio > 1.5:
                score += 8
            elif current_ratio > 1:
                score += 5
        
        # 估值合理性评分 (15分)
        if 'PE' in self.ratios:
            pe = self.ratios['PE']
            if 0 < pe < 15:
                score += 8
            elif pe < 25:
                score += 6
            elif pe < 35:
                score += 3
        
        if 'PB' in self.ratios:
            pb = self.ratios['PB']
            if 0 < pb < 1.5:
                score += 7
            elif pb < 3:
                score += 5
            elif pb < 5:
                score += 3
        
        # 营运能力评分 (10分)
        if 'ROA' in self.ratios:
            roa = self.ratios['ROA']
            if roa > 8:
                score += 5
            elif roa > 5:
                score += 3
            elif roa > 2:
                score += 1
        
        if '存货周转率' in self.ratios:
            inventory_turnover = self.ratios['存货周转率']
            if inventory_turnover > 6:
                score += 5
            elif inventory_turnover > 4:
                score += 3
            elif inventory_turnover > 2:
                score += 1
        
        # 资产质量评分 (10分)
        asset_quality_bonus = 0
        
        # 商誉占比低更好
        if 'Goodwill Ratio' in self.ratios:
            goodwill_ratio = self.ratios['Goodwill Ratio']
            if goodwill_ratio < 5:
                asset_quality_bonus += 3
            elif goodwill_ratio < 15:
                asset_quality_bonus += 2
            elif goodwill_ratio < 30:
                asset_quality_bonus += 1
        else:
            asset_quality_bonus += 3  # 没有商誉也是好事
        
        # 无形资产比例适中
        if 'Intangible Assets Ratio' in self.ratios:
            intangible_ratio = self.ratios['Intangible Assets Ratio']
            if intangible_ratio < 20:
                asset_quality_bonus += 2
            elif intangible_ratio < 40:
                asset_quality_bonus += 1
        else:
            asset_quality_bonus += 2
        
        score += min(asset_quality_bonus, 10)
        
        # 市场风险评分 (5分) - 低风险获得更高分数
        if 'Volatility' in self.ratios:
            volatility = self.ratios['Volatility']
            if volatility < 20:
                score += 5
            elif volatility < 30:
                score += 3
            elif volatility < 50:
                score += 1
        
        return min(score, max_score)
    
    def _generate_recommendation(self):
        """生成投资建议"""
        if not self.ratios:
            return "数据不足，无法给出建议"
        
        score = self._calculate_financial_health_score()
        
        if score >= 80:
            return "强烈推荐 - 财务状况优秀"
        elif score >= 60:
            return "推荐 - 财务状况良好"
        elif score >= 40:
            return "中性 - 财务状况一般"
        elif score >= 20:
            return "谨慎 - 财务状况较差"
        else:
            return "不推荐 - 财务状况堪忧"
    
    def _identify_risks(self):
        """识别投资风险（更新版，包含新增指标风险评估）"""
        risks = []
        
        if not self.ratios:
            return ["数据不足，无法评估风险"]
        
        # 盈利能力风险
        if 'ROE' in self.ratios and self.ratios['ROE'] < 5:
            risks.append("盈利能力较弱，ROE低于5%")
        
        if '净利率' in self.ratios and self.ratios['净利率'] < 0:
            risks.append("公司处于亏损状态")
        
        # 成长性风险
        if 'Revenue Growth' in self.ratios and self.ratios['Revenue Growth'] < -5:
            risks.append("营收增长率为负，业务可能萎缩")
        
        if 'Net Income Growth' in self.ratios and self.ratios['Net Income Growth'] < -10:
            risks.append("净利润大幅下滑，盈利恶化")
        
        # 现金流质量风险
        if 'Operating CF to Net Income' in self.ratios and self.ratios['Operating CF to Net Income'] < 50:
            risks.append("经营现金流质量较差，可能存在利润操纵")
        
        if 'Free Cash Flow' in self.ratios and self.ratios['Free Cash Flow'] < 0:
            risks.append("自由现金流为负，资本支出压力大")
        
        # 资产质量风险
        if 'Goodwill Ratio' in self.ratios and self.ratios['Goodwill Ratio'] > 30:
            risks.append("商誉占比过高，存在减值风险")
        
        if 'Intangible Assets Ratio' in self.ratios and self.ratios['Intangible Assets Ratio'] > 50:
            risks.append("无形资产占比过高，资产质量存疑")
        
        # 偿债能力风险
        if '资产负债率' in self.ratios and self.ratios['资产负债率'] > 70:
            risks.append("资产负债率过高，偿债压力较大")
        
        if '流动比率' in self.ratios and self.ratios['流动比率'] < 1:
            risks.append("流动比率小于1，短期偿债能力不足")
        
        # 估值风险
        if 'PE' in self.ratios and self.ratios['PE'] > 50:
            risks.append("市盈率过高，可能存在估值泡沫")
        
        if 'PB' in self.ratios and self.ratios['PB'] > 10:
            risks.append("市净率过高，估值偏贵")
        
        # 市场风险
        if 'Volatility' in self.ratios and self.ratios['Volatility'] > 50:
            risks.append("股价波动率过高，投资风险较大")
        
        if 'Beta' in self.ratios and self.ratios['Beta'] > 2:
            risks.append("Beta系数过高，系统性风险敏感度大")
        
        # 股息风险（对于美股）
        if 'Payout Ratio' in self.ratios and self.ratios['Payout Ratio'] > 100:
            risks.append("派息比率超过100%，股息可持续性存疑")
        
        return risks
    
    def plot_financial_analysis(self):
        """绘制财务分析图表"""
        if not self.financial_data and not self.ratios:
            print("请先进行财务分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.symbol} 基本面分析图表', fontsize=16)
        
        # 1. 财务比率雷达图
        if self.ratios:
            ax1 = axes[0, 0]
            self._plot_ratios_radar(ax1)
        
        # 2. 同行对比
        if hasattr(self, 'peer_comparison') and not self.peer_comparison.empty:
            ax2 = axes[0, 1]
            self._plot_peer_comparison(ax2)
        
        # 3. 财务趋势（如果有历史数据）
        ax3 = axes[1, 0]
        self._plot_financial_trends(ax3)
        
        # 4. 估值分析
        ax4 = axes[1, 1]
        self._plot_valuation_analysis(ax4)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_ratios_radar(self, ax):
        """绘制财务比率雷达图"""
        if not self.ratios:
            ax.text(0.5, 0.5, '无财务比率数据', ha='center', va='center')
            return
        
        # 选择关键比率进行展示
        key_ratios = ['ROE', 'ROA', '净利率', '流动比率']
        available_ratios = {k: v for k, v in self.ratios.items() if k in key_ratios}
        
        if not available_ratios:
            ax.text(0.5, 0.5, '无关键比率数据', ha='center', va='center')
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
    
    def _plot_peer_comparison(self, ax):
        """绘制同行对比图"""
        if not hasattr(self, 'peer_comparison') or self.peer_comparison.empty:
            ax.text(0.5, 0.5, '无同行对比数据', ha='center', va='center')
            ax.set_title('同行对比')
            return
        
        # PE比较
        df = self.peer_comparison
        if 'PE' in df.columns:
            pe_data = df[df['PE'] > 0]['PE']  # 过滤掉无效PE值
            symbols = df[df['PE'] > 0]['Symbol']
            
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
            ax.text(0.5, 0.5, '无PE数据', ha='center', va='center')
            ax.set_title('同行对比')
    
    def _plot_financial_trends(self, ax):
        """绘制财务趋势图"""
        try:
            if self.market == 'US' and 'income_statement' in self.financial_data:
                financials = self.financial_data['income_statement']
                
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
                
            elif self.market == 'CN' and 'indicators' in self.financial_data:
                # 中国股票的趋势分析
                indicators = self.financial_data['indicators']
                
                if len(indicators) > 1:
                    # 营业收入趋势
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
                else:
                    ax.text(0.5, 0.5, '数据不足，无法显示趋势', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, '无财务趋势数据', ha='center', va='center')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'趋势图绘制失败:\n{str(e)}', ha='center', va='center')
        
        ax.set_title('财务趋势')
    
    def _plot_valuation_analysis(self, ax):
        """绘制估值分析图"""
        if not self.ratios:
            ax.text(0.5, 0.5, '无估值数据', ha='center', va='center')
            ax.set_title('估值分析')
            return
        
        # 估值指标对比
        valuation_metrics = {}
        if 'PE' in self.ratios:
            valuation_metrics['PE'] = self.ratios['PE']
        if 'PB' in self.ratios:
            valuation_metrics['PB'] = self.ratios['PB']
        
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

# 完整的使用示例和演示
def main():
    """主函数：演示基本面分析的完整流程"""
    
    print("="*80)
    print("基本面分析工具演示")
    print("="*80)
    
    # 分析美股示例
    print("\n1. 美股分析示例 - Apple Inc. (AAPL)")
    print("-" * 50)
    
    analyzer_us = FundamentalAnalyzer('AAPL', market='US')
    
    if analyzer_us.load_company_data():
        # 分析财务报表
        analyzer_us.analyze_financial_statements()
        
        # 计算财务比率
        analyzer_us.calculate_financial_ratios()
        
        # 同行对比（科技股）
        tech_peers = ['MSFT', 'GOOGL', 'AMZN', 'TSLA']
        analyzer_us.peer_comparison_analysis(tech_peers)
        
        # DCF估值
        analyzer_us.dcf_valuation(growth_years=5, terminal_growth=3, discount_rate=10)
        
        # 生成投资摘要
        analyzer_us.generate_investment_summary()
        
        # 绘制分析图表
        try:
            analyzer_us.plot_financial_analysis()
        except Exception as e:
            print(f"图表绘制失败: {e}")
    
    # 分析中国股票示例（如果有akshare）
    if AKSHARE_AVAILABLE:
        print(f"\n\n2. A股分析示例 - 平安银行 (000001)")
        print("-" * 50)
        
        analyzer_cn = FundamentalAnalyzer('000001', market='CN')
        
        if analyzer_cn.load_company_data():
            # 分析财务报表
            analyzer_cn.analyze_financial_statements()
            
            # 计算财务比率  
            analyzer_cn.calculate_financial_ratios()
            
            # 同行对比（银行股）
            bank_peers = ['000002', '600036', '601988']
            analyzer_cn.peer_comparison_analysis(bank_peers, start_year='2020')
            
            # 生成投资摘要
            analyzer_cn.generate_investment_summary()
    
    print(f"\n\n基本面分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()