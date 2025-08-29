"""
估值模型模块 - 负责各种估值方法的实现
"""
import pandas as pd
import numpy as np


class DCFValuationModel:
    """DCF现金流折现估值模型"""
    
    def __init__(self, symbol, market='US'):
        self.symbol = symbol
        self.market = market
    
    def calculate_dcf_valuation(self, financial_data, company_info, ticker=None, 
                              growth_years=5, terminal_growth=2.5, discount_rate=10):
        """DCF估值计算"""
        print(f"\n{'='*60}")
        print("DCF估值分析")
        print('='*60)
        
        try:
            if self.market == 'US' and ticker:
                return self._calculate_us_dcf(financial_data, company_info, ticker, 
                                           growth_years, terminal_growth, discount_rate)
            elif self.market == 'CN':
                return self._calculate_cn_dcf(financial_data, company_info,
                                           growth_years, terminal_growth, discount_rate)
            else:
                print("不支持的市场类型或缺少必要参数")
                return None
                
        except Exception as e:
            print(f"DCF估值计算失败: {e}")
            return None
    
    def _calculate_us_dcf(self, financial_data, company_info, ticker, 
                         growth_years, terminal_growth, discount_rate):
        """计算美股DCF估值"""
        cash_flow = financial_data.get('cash_flow')
        
        if not cash_flow or 'Operating Cash Flow' not in cash_flow.index:
            print("无法获取经营现金流数据")
            return None
        
        # 最近一年的自由现金流
        latest_period = cash_flow.columns[0]
        operating_cf = cash_flow.loc['Operating Cash Flow', latest_period]
        
        # 估算资本支出
        capex = 0
        if 'Capital Expenditures' in cash_flow.index:
            capex = abs(cash_flow.loc['Capital Expenditures', latest_period])
        
        free_cash_flow = operating_cf - capex
        
        print(f"基础数据:")
        print(f"  经营现金流: ${operating_cf:,.0f}")
        print(f"  资本支出: ${capex:,.0f}")
        print(f"  自由现金流: ${free_cash_flow:,.0f}")
        
        # 计算历史现金流增长率
        historical_growth = self._calculate_historical_cf_growth(cash_flow)
        if historical_growth is not None:
            print(f"  历史现金流增长率: {historical_growth:.1f}%")
        
        # DCF计算
        print(f"\nDCF假设:")
        print(f"  预测期: {growth_years} 年")
        print(f"  年增长率: 假设10% (可调整)")
        print(f"  永续增长率: {terminal_growth}%")
        print(f"  折现率: {discount_rate}%")
        
        # 预测未来现金流
        growth_rate = 0.10  # 假设10%增长率
        future_cf = []
        
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
        
        # 计算每股价值
        shares_outstanding = company_info.get('sharesOutstanding')
        if shares_outstanding:
            # 减去净债务得到股权价值
            net_debt = self._calculate_net_debt(financial_data.get('balance_sheet'))
            equity_value = enterprise_value - net_debt
            value_per_share = equity_value / shares_outstanding
            
            # 获取当前股价
            current_price = ticker.history(period='1d')['Close'][-1]
            
            print(f"  净债务: ${net_debt:,.0f}")
            print(f"  股权价值: ${equity_value:,.0f}")
            print(f"  每股内在价值: ${value_per_share:.2f}")
            print(f"  当前股价: ${current_price:.2f}")
            
            upside_potential = (value_per_share / current_price - 1) * 100
            print(f"  上涨空间: {upside_potential:+.1f}%")
            
            return {
                'enterprise_value': enterprise_value,
                'value_per_share': value_per_share,
                'current_price': current_price,
                'upside_potential': upside_potential
            }
        
        return {
            'enterprise_value': enterprise_value,
            'value_per_share': None,
            'current_price': None,
            'upside_potential': None
        }
    
    def _calculate_cn_dcf(self, financial_data, company_info, 
                         growth_years, terminal_growth, discount_rate):
        """计算中国股票DCF估值"""
        # 中国股票DCF估值的简化实现
        # 由于akshare数据结构不同，这里提供基础框架
        print("中国股票DCF估值功能开发中...")
        return None
    
    def _calculate_historical_cf_growth(self, cash_flow):
        """计算历史现金流增长率"""
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
                
                return np.mean(growth_rates)
        return None
    
    def _calculate_net_debt(self, balance_sheet):
        """计算净债务"""
        if not balance_sheet:
            return 0
        
        net_debt = 0
        latest_period = balance_sheet.columns[0]
        
        if 'Total Debt' in balance_sheet.index and 'Cash' in balance_sheet.index:
            total_debt = balance_sheet.loc['Total Debt', latest_period] or 0
            cash = balance_sheet.loc['Cash', latest_period] or 0
            net_debt = total_debt - cash
        
        return net_debt


class ComparativeValuationModel:
    """相对估值模型"""
    
    def __init__(self, symbol, market='US'):
        self.symbol = symbol
        self.market = market
    
    def calculate_relative_valuation(self, target_ratios, peer_comparison):
        """相对估值分析"""
        print(f"\n{'='*60}")
        print("相对估值分析")
        print('='*60)
        
        if not peer_comparison or peer_comparison.empty:
            print("缺少同行对比数据")
            return None
        
        try:
            results = {}
            
            # PE相对估值
            pe_valuation = self._calculate_pe_relative_valuation(target_ratios, peer_comparison)
            if pe_valuation:
                results['PE_Valuation'] = pe_valuation
            
            # PB相对估值
            pb_valuation = self._calculate_pb_relative_valuation(target_ratios, peer_comparison)
            if pb_valuation:
                results['PB_Valuation'] = pb_valuation
            
            # EV/EBITDA相对估值（如果有数据）
            # 这里可以扩展更多相对估值指标
            
            self._print_relative_valuation_results(results)
            
            return results
            
        except Exception as e:
            print(f"相对估值计算失败: {e}")
            return None
    
    def _calculate_pe_relative_valuation(self, target_ratios, peer_comparison):
        """PE相对估值"""
        if 'PE' not in peer_comparison.columns or 'PE' not in target_ratios:
            return None
        
        # 计算同行平均PE
        peer_data = peer_comparison[peer_comparison['Symbol'] != self.symbol]
        valid_pe = peer_data[peer_data['PE'] > 0]['PE']
        
        if len(valid_pe) == 0:
            return None
        
        peer_avg_pe = valid_pe.mean()
        target_pe = target_ratios['PE']
        
        # 计算PE相对溢价/折价
        pe_premium = (target_pe / peer_avg_pe - 1) * 100
        
        return {
            'peer_avg_pe': peer_avg_pe,
            'target_pe': target_pe,
            'pe_premium': pe_premium
        }
    
    def _calculate_pb_relative_valuation(self, target_ratios, peer_comparison):
        """PB相对估值"""
        if 'PB' not in peer_comparison.columns or 'PB' not in target_ratios:
            return None
        
        # 计算同行平均PB
        peer_data = peer_comparison[peer_comparison['Symbol'] != self.symbol]
        valid_pb = peer_data[peer_data['PB'] > 0]['PB']
        
        if len(valid_pb) == 0:
            return None
        
        peer_avg_pb = valid_pb.mean()
        target_pb = target_ratios['PB']
        
        # 计算PB相对溢价/折价
        pb_premium = (target_pb / peer_avg_pb - 1) * 100
        
        return {
            'peer_avg_pb': peer_avg_pb,
            'target_pb': target_pb,
            'pb_premium': pb_premium
        }
    
    def _print_relative_valuation_results(self, results):
        """打印相对估值结果"""
        print("相对估值分析结果:")
        
        if 'PE_Valuation' in results:
            pe_val = results['PE_Valuation']
            print(f"\nPE相对估值:")
            print(f"  同行平均PE: {pe_val['peer_avg_pe']:.1f}")
            print(f"  目标公司PE: {pe_val['target_pe']:.1f}")
            print(f"  PE溢价/折价: {pe_val['pe_premium']:+.1f}%")
            
            if pe_val['pe_premium'] > 20:
                print("  评价: PE显著高于同行，可能存在高估")
            elif pe_val['pe_premium'] < -20:
                print("  评价: PE显著低于同行，可能存在低估")
            else:
                print("  评价: PE与同行水平接近")
        
        if 'PB_Valuation' in results:
            pb_val = results['PB_Valuation']
            print(f"\nPB相对估值:")
            print(f"  同行平均PB: {pb_val['peer_avg_pb']:.1f}")
            print(f"  目标公司PB: {pb_val['target_pb']:.1f}")
            print(f"  PB溢价/折价: {pb_val['pb_premium']:+.1f}%")
            
            if pb_val['pb_premium'] > 20:
                print("  评价: PB显著高于同行，可能存在高估")
            elif pb_val['pb_premium'] < -20:
                print("  评价: PB显著低于同行，可能存在低估")
            else:
                print("  评价: PB与同行水平接近")


class DividendDiscountModel:
    """股息折现模型（Gordon增长模型）"""
    
    def __init__(self, symbol, market='US'):
        self.symbol = symbol
        self.market = market
    
    def calculate_ddm_valuation(self, ratios, financial_data, required_return=0.10):
        """股息折现模型估值"""
        print(f"\n{'='*60}")
        print("股息折现模型估值")
        print('='*60)
        
        try:
            # 检查是否有股息数据
            if 'Dividend Yield' not in ratios or ratios['Dividend Yield'] == 0:
                print("该股票不支付股息，不适用股息折现模型")
                return None
            
            dividend_yield = ratios['Dividend Yield'] / 100  # 转换为小数
            
            # 估算股息增长率（简化处理，使用收益增长率作为代理）
            dividend_growth_rate = 0.03  # 默认3%
            if 'Net Income Growth' in ratios:
                # 使用净利润增长率的一半作为股息增长率估算
                dividend_growth_rate = max(0.01, ratios['Net Income Growth'] / 200)
            
            print(f"股息折现模型参数:")
            print(f"  当前股息率: {dividend_yield*100:.2f}%")
            print(f"  预期股息增长率: {dividend_growth_rate*100:.2f}%")
            print(f"  要求回报率: {required_return*100:.1f}%")
            
            # Gordon增长模型: V = D1 / (r - g)
            if required_return <= dividend_growth_rate:
                print("要求回报率必须大于增长率，模型不适用")
                return None
            
            # 假设当前股价来计算当前股息
            if 'current_price' in financial_data:
                current_price = financial_data['current_price']
                current_dividend = current_price * dividend_yield
                next_dividend = current_dividend * (1 + dividend_growth_rate)
                
                # DDM估值
                ddm_value = next_dividend / (required_return - dividend_growth_rate)
                
                upside_potential = (ddm_value / current_price - 1) * 100
                
                print(f"\n股息折现模型结果:")
                print(f"  当前股价: ${current_price:.2f}")
                print(f"  预期下年股息: ${next_dividend:.2f}")
                print(f"  DDM内在价值: ${ddm_value:.2f}")
                print(f"  上涨空间: {upside_potential:+.1f}%")
                
                return {
                    'ddm_value': ddm_value,
                    'current_price': current_price,
                    'upside_potential': upside_potential
                }
            else:
                print("缺少当前股价数据")
                return None
                
        except Exception as e:
            print(f"DDM估值计算失败: {e}")
            return None


class ValuationSummary:
    """估值汇总分析"""
    
    def __init__(self, symbol):
        self.symbol = symbol
    
    def generate_valuation_summary(self, dcf_result=None, relative_result=None, ddm_result=None):
        """生成估值汇总"""
        print(f"\n{'='*60}")
        print(f"{self.symbol} 估值汇总")
        print('='*60)
        
        valuations = []
        
        if dcf_result and dcf_result.get('value_per_share'):
            valuations.append({
                'method': 'DCF估值',
                'value': dcf_result['value_per_share'],
                'upside': dcf_result.get('upside_potential', 0)
            })
        
        if ddm_result:
            valuations.append({
                'method': '股息折现',
                'value': ddm_result['ddm_value'],
                'upside': ddm_result['upside_potential']
            })
        
        if relative_result:
            print("相对估值分析:")
            if 'PE_Valuation' in relative_result:
                pe_premium = relative_result['PE_Valuation']['pe_premium']
                print(f"  PE相对同行: {pe_premium:+.1f}%")
            
            if 'PB_Valuation' in relative_result:
                pb_premium = relative_result['PB_Valuation']['pb_premium']
                print(f"  PB相对同行: {pb_premium:+.1f}%")
        
        if valuations:
            print(f"\n绝对估值汇总:")
            total_upside = 0
            valid_methods = 0
            
            for val in valuations:
                print(f"  {val['method']}: ${val['value']:.2f} (上涨空间: {val['upside']:+.1f}%)")
                if not np.isnan(val['upside']) and not np.isinf(val['upside']):
                    total_upside += val['upside']
                    valid_methods += 1
            
            if valid_methods > 0:
                avg_upside = total_upside / valid_methods
                print(f"\n平均上涨空间: {avg_upside:+.1f}%")
                
                if avg_upside > 20:
                    conclusion = "显著低估"
                elif avg_upside > 10:
                    conclusion = "适度低估"
                elif avg_upside > -10:
                    conclusion = "合理估值"
                elif avg_upside > -20:
                    conclusion = "适度高估"
                else:
                    conclusion = "显著高估"
                
                print(f"估值结论: {conclusion}")
            
        return {
            'dcf_result': dcf_result,
            'relative_result': relative_result,
            'ddm_result': ddm_result,
            'valuations': valuations
        }