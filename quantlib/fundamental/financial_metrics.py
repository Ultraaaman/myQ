"""
财务指标计算模块 - 负责各种财务比率和指标的计算
"""
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


class FinancialMetricsCalculator:
    """财务指标计算器"""
    
    def __init__(self, symbol, market='US'):
        self.symbol = symbol
        self.market = market
    
    def calculate_us_ratios(self, financial_data, company_info, ticker):
        """计算美股财务比率"""
        ratios = {}
        
        if not financial_data:
            return ratios
        
        financials = financial_data['income_statement']
        balance_sheet = financial_data['balance_sheet']
        cash_flow = financial_data['cash_flow']
        
        # 获取股票价格信息
        hist_data = ticker.history(period='1d')
        current_price = hist_data['Close'][-1] if not hist_data.empty else None
        shares_outstanding = company_info.get('sharesOutstanding', None)
        
        latest_period = financials.columns[0]
        
        # 盈利能力比率
        ratios.update(self._calculate_profitability_ratios(financials, balance_sheet, latest_period))
        
        # 成长性指标
        ratios.update(self._calculate_growth_ratios(financials, shares_outstanding))
        
        # 估值比率
        ratios.update(self._calculate_valuation_ratios(
            financials, balance_sheet, current_price, shares_outstanding, company_info, latest_period
        ))
        
        # 资产质量指标
        ratios.update(self._calculate_asset_quality_ratios(balance_sheet, latest_period))
        
        # 现金流质量指标
        ratios.update(self._calculate_cashflow_ratios(cash_flow, financials, 
                                                    current_price, shares_outstanding, latest_period))
        
        # 偿债能力比率
        ratios.update(self._calculate_solvency_ratios(balance_sheet, latest_period))
        
        return ratios
    
    def calculate_cn_ratios(self, financial_data, symbol, start_year="2020"):
        """计算中国股票财务比率"""
        ratios = {}
        
        if 'indicators' not in financial_data:
            return ratios
        
        indicators = financial_data['indicators']
        latest = indicators.iloc[-1]  # 使用排序后的最新数据
        
        # 基础财务比率映射
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
        
        # 成长性指标映射
        growth_mappings = {
            'Revenue Growth': '主营业务收入增长率(%)',
            'Net Income Growth': '净利润增长率(%)',
            'Total Assets Growth': '总资产增长率(%)'
        }
        
        # 提取基础财务比率
        for ratio_name, column_name in ratio_mappings.items():
            if column_name in latest.index:
                value = latest[column_name]
                if pd.notna(value):
                    ratios[ratio_name] = float(value)
        
        # 提取成长性指标
        for ratio_name, column_name in growth_mappings.items():
            if column_name in latest.index:
                value = latest[column_name]
                if pd.notna(value):
                    ratios[ratio_name] = float(value)
        
        # 计算现金流质量指标
        if len(indicators) >= 2:
            ratios.update(self._calculate_cn_cashflow_ratios(latest))
        
        # 获取估值数据
        ratios.update(self._calculate_cn_valuation_ratios(symbol, latest))
        
        return ratios
    
    def _calculate_profitability_ratios(self, financials, balance_sheet, latest_period):
        """计算盈利能力比率"""
        ratios = {}
        
        # 净利率
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
        
        return ratios
    
    def _calculate_growth_ratios(self, financials, shares_outstanding):
        """计算成长性指标"""
        ratios = {}
        
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
            if shares_outstanding and 'Net Income' in financials.index:
                current_eps = financials.loc['Net Income', current_period] / shares_outstanding
                previous_eps = financials.loc['Net Income', previous_period] / shares_outstanding
                if previous_eps != 0 and pd.notna(previous_eps) and previous_eps > 0:
                    ratios['EPS Growth'] = ((current_eps - previous_eps) / previous_eps) * 100
        
        return ratios
    
    def _calculate_valuation_ratios(self, financials, balance_sheet, current_price, shares_outstanding, 
                                  company_info, latest_period):
        """计算估值比率"""
        ratios = {}
        
        if current_price and shares_outstanding:
            market_cap = current_price * shares_outstanding
            
            # PE比率
            if 'Net Income' in financials.index:
                net_income = financials.loc['Net Income', latest_period]
                if net_income > 0:
                    ratios['PE'] = market_cap / net_income
            
            # PB比率
            if 'Total Stockholder Equity' in balance_sheet.index:
                equity = balance_sheet.loc['Total Stockholder Equity', latest_period]
                if equity > 0:
                    ratios['PB'] = market_cap / equity
            
            # 股息相关指标
            dividends_info = company_info.get('dividendYield', 0)
            if dividends_info:
                ratios['Dividend Yield'] = dividends_info * 100
            
            payout_ratio = company_info.get('payoutRatio', 0)
            if payout_ratio:
                ratios['Payout Ratio'] = payout_ratio * 100
        
        return ratios
    
    def _calculate_asset_quality_ratios(self, balance_sheet, latest_period):
        """计算资产质量指标"""
        ratios = {}
        
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
        
        return ratios
    
    def _calculate_cashflow_ratios(self, cash_flow, financials, current_price, 
                                 shares_outstanding, latest_period):
        """计算现金流质量指标"""
        ratios = {}
        
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
        
        return ratios
    
    def _calculate_solvency_ratios(self, balance_sheet, latest_period):
        """计算偿债能力比率"""
        ratios = {}
        
        # 流动比率
        if 'Current Assets' in balance_sheet.index and 'Current Liab' in balance_sheet.index:
            current_assets = balance_sheet.loc['Current Assets', latest_period]
            current_liab = balance_sheet.loc['Current Liab', latest_period]
            if current_liab != 0:
                ratios['流动比率'] = current_assets / current_liab
        
        # 资产负债率
        if 'Total Debt' in balance_sheet.index and 'Total Assets' in balance_sheet.index:
            total_debt = balance_sheet.loc['Total Debt', latest_period]
            total_assets = balance_sheet.loc['Total Assets', latest_period]
            if total_assets != 0:
                ratios['资产负债率'] = (total_debt / total_assets) * 100
        
        return ratios
    
    def _calculate_cn_cashflow_ratios(self, latest):
        """计算中国股票现金流质量指标"""
        ratios = {}
        
        # 经营现金流/净利润比率
        if '每股经营性现金流(元)' in latest.index and '摊薄每股收益(元)' in latest.index:
            ocf_per_share = latest['每股经营性现金流(元)']
            eps = latest['摊薄每股收益(元)']
            if pd.notna(ocf_per_share) and pd.notna(eps) and eps != 0:
                ratios['Operating CF to Net Income'] = (ocf_per_share / eps) * 100
        
        return ratios
    
    def _calculate_cn_valuation_ratios(self, symbol, latest):
        """计算中国股票估值比率"""
        ratios = {}
        
        if not AKSHARE_AVAILABLE:
            return ratios
        
        try:
            # 从个股信息获取PE和PB
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            pe_found = False
            pb_found = False
            
            print(f"调试：获取到 {len(stock_info)} 个信息字段")
            
            # 在一次循环中同时搜索PE和PB
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
                            print(f"找到PE: {item_name} = {pe_val}")
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
                            print(f"找到PB: {item_name} = {pb_val}")
                    except (ValueError, TypeError):
                        pass
                
                # 如果都找到了就可以退出循环
                if pe_found and pb_found:
                    break
            
            # 如果没有找到PE/PB，尝试手动计算
            if not pe_found or not pb_found:
                try:
                    # 获取财务指标数据用于手动计算
                    financial_indicators = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2020")
                    if not financial_indicators.empty:
                        financial_indicators['日期'] = pd.to_datetime(financial_indicators['日期'])
                        financial_indicators = financial_indicators.sort_values('日期', ascending=True)
                        latest_financial = financial_indicators.iloc[-1]
                        
                        # 手动计算PE
                        if not pe_found:
                            eps_candidates = ['摊薄每股收益(元)', '基本每股收益(元)', '每股收益(元)']
                            for eps_col in eps_candidates:
                                if eps_col in latest_financial.index and pd.notna(latest_financial[eps_col]):
                                    eps = float(latest_financial[eps_col])
                                    if eps > 0:
                                        try:
                                            stock_zh_a_hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
                                            if not stock_zh_a_hist.empty:
                                                current_price = float(stock_zh_a_hist.iloc[-1]['收盘'])
                                                pe_calculated = current_price / eps
                                                if pe_calculated > 0 and pe_calculated < 1000:
                                                    ratios['PE'] = pe_calculated
                                                    pe_found = True
                                                    print(f"手动计算PE: {current_price} / {eps} = {pe_calculated:.2f}")
                                                    break
                                        except Exception as e_pe:
                                            print(f"PE计算中获取股价失败: {e_pe}")
                                            pass
                        
                        # 手动计算PB
                        if not pb_found:
                            bps_candidates = ['每股净资产_调整后(元)', '每股净资产(元)', '每股账面价值(元)']
                            for bps_col in bps_candidates:
                                if bps_col in latest_financial.index and pd.notna(latest_financial[bps_col]):
                                    bps = float(latest_financial[bps_col])
                                    if bps > 0:
                                        try:
                                            stock_zh_a_hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
                                            if not stock_zh_a_hist.empty:
                                                current_price = float(stock_zh_a_hist.iloc[-1]['收盘'])
                                                pb_calculated = current_price / bps
                                                if pb_calculated > 0 and pb_calculated < 100:
                                                    ratios['PB'] = pb_calculated
                                                    pb_found = True
                                                    print(f"手动计算PB: {current_price} / {bps} = {pb_calculated:.2f}")
                                                    break
                                        except Exception as e_pb:
                                            print(f"PB计算中获取股价失败: {e_pb}")
                                            pass
                                            
                except Exception as e_manual:
                    print(f"手动计算PE/PB失败: {e_manual}")
                    
            if not pe_found:
                print("未能获取PE数据")
            if not pb_found:
                print("未能获取PB数据")
                        
        except Exception as e:
            print(f"获取估值数据失败: {e}")
        
        return ratios
    
    def calculate_market_performance_ratios(self, symbol, market='US'):
        """计算市场表现指标"""
        market_ratios = {}
        
        try:
            if market == 'US':
                market_ratios.update(self._calculate_us_market_ratios(symbol))
            elif market == 'CN' and AKSHARE_AVAILABLE:
                market_ratios.update(self._calculate_cn_market_ratios(symbol))
        except Exception as e:
            print(f"计算市场表现指标失败: {e}")
        
        return market_ratios
    
    def _calculate_us_market_ratios(self, symbol):
        """计算美股市场表现指标"""
        ratios = {}
        
        try:
            if not YFINANCE_AVAILABLE:
                print(f"⚠ 跳过 {symbol}，需要安装 yfinance")
                return {}
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period="2y")
            
            if not hist_data.empty and len(hist_data) > 252:
                returns = hist_data['Close'].pct_change().dropna()
                
                # 年化波动率
                volatility = returns.std() * np.sqrt(252) * 100
                ratios['Volatility'] = volatility
                
                # Beta计算（相对于SPY）
                try:
                    spy = yf.Ticker("SPY")
                    spy_hist = spy.history(period="2y")
                    
                    if not spy_hist.empty:
                        spy_returns = spy_hist['Close'].pct_change().dropna()
                        common_dates = returns.index.intersection(spy_returns.index)
                        
                        if len(common_dates) > 100:
                            stock_aligned = returns.loc[common_dates]
                            market_aligned = spy_returns.loc[common_dates]
                            
                            covariance = np.cov(stock_aligned, market_aligned)[0, 1]
                            market_variance = np.var(market_aligned)
                            if market_variance != 0:
                                ratios['Beta'] = covariance / market_variance
                except:
                    pass
                
                # 夏普比率
                risk_free_rate = 0.03
                excess_returns = returns.mean() * 252 - risk_free_rate
                if volatility > 0:
                    ratios['Sharpe Ratio'] = excess_returns / (volatility / 100)
                
                # 价格趋势指标
                if len(hist_data) >= 66:
                    current_price = hist_data['Close'].iloc[-1]
                    price_1m = hist_data['Close'].iloc[-22] if len(hist_data) >= 22 else current_price
                    price_3m = hist_data['Close'].iloc[-66]
                    
                    ratios['1M Price Change'] = (current_price - price_1m) / price_1m * 100
                    ratios['3M Price Change'] = (current_price - price_3m) / price_3m * 100
        
        except Exception as e:
            print(f"计算美股市场指标失败: {e}")
        
        return ratios
    
    def _calculate_cn_market_ratios(self, symbol):
        """计算中国股票市场表现指标"""
        ratios = {}
        
        try:
            stock_hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
            
            if not stock_hist.empty and len(stock_hist) > 252:
                stock_hist['收盘'] = pd.to_numeric(stock_hist['收盘'], errors='coerce')
                returns = stock_hist['收盘'].pct_change().dropna()
                
                # 年化波动率
                volatility = returns.std() * np.sqrt(252) * 100
                ratios['Volatility'] = volatility
                
                # Beta计算（相对于沪深300）
                try:
                    market_hist = ak.stock_zh_index_daily(symbol="sh000300")
                    if not market_hist.empty:
                        market_hist['close'] = pd.to_numeric(market_hist['close'], errors='coerce')
                        market_returns = market_hist['close'].pct_change().dropna()
                        
                        min_len = min(len(returns), len(market_returns))
                        if min_len > 100:
                            stock_ret = returns.iloc[-min_len:]
                            market_ret = market_returns.iloc[-min_len:]
                            
                            covariance = np.cov(stock_ret, market_ret)[0, 1]
                            market_variance = np.var(market_ret)
                            if market_variance != 0:
                                ratios['Beta'] = covariance / market_variance
                except:
                    pass
                
                # 价格变化趋势
                if len(stock_hist) >= 66:
                    current_price = stock_hist['收盘'].iloc[-1]
                    price_1m = stock_hist['收盘'].iloc[-22] if len(stock_hist) >= 22 else current_price
                    price_3m = stock_hist['收盘'].iloc[-66]
                    
                    if price_1m != 0:
                        ratios['1M Price Change'] = (current_price - price_1m) / price_1m * 100
                    if price_3m != 0:
                        ratios['3M Price Change'] = (current_price - price_3m) / price_3m * 100
        
        except Exception as e:
            print(f"计算中国股票市场指标失败: {e}")
        
        return ratios