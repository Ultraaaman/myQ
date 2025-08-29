"""
分析引擎模块 - 负责分析逻辑和评分系统
"""
import pandas as pd
import numpy as np


class FinancialHealthAnalyzer:
    """财务健康度分析器"""
    
    def __init__(self):
        pass
    
    def calculate_financial_health_score(self, ratios):
        """计算财务健康度评分"""
        score = 0
        max_score = 100
        
        if not ratios:
            return score
        
        # 盈利能力评分 (25分)
        score += self._score_profitability(ratios)
        
        # 成长性评分 (15分)
        score += self._score_growth(ratios)
        
        # 偿债能力评分 (20分)
        score += self._score_solvency(ratios)
        
        # 估值合理性评分 (15分)
        score += self._score_valuation(ratios)
        
        # 营运能力评分 (10分)
        score += self._score_efficiency(ratios)
        
        # 资产质量评分 (10分)
        score += self._score_asset_quality(ratios)
        
        # 市场风险评分 (5分)
        score += self._score_market_risk(ratios)
        
        return min(score, max_score)
    
    def _score_profitability(self, ratios):
        """盈利能力评分"""
        score = 0
        
        if 'ROE' in ratios:
            roe = ratios['ROE']
            if roe > 15:
                score += 10
            elif roe > 10:
                score += 8
            elif roe > 5:
                score += 5
        
        if '净利率' in ratios:
            net_margin = ratios['净利率']
            if net_margin > 10:
                score += 10
            elif net_margin > 5:
                score += 8
            elif net_margin > 0:
                score += 5
        
        # 现金流质量评分
        if 'Operating CF to Net Income' in ratios:
            cf_quality = ratios['Operating CF to Net Income']
            if cf_quality > 120:
                score += 5
            elif cf_quality > 90:
                score += 3
            elif cf_quality > 60:
                score += 1
        
        return score
    
    def _score_growth(self, ratios):
        """成长性评分"""
        score = 0
        
        if 'Revenue Growth' in ratios:
            revenue_growth = ratios['Revenue Growth']
            if revenue_growth > 15:
                score += 8
            elif revenue_growth > 5:
                score += 5
            elif revenue_growth > 0:
                score += 3
        
        if 'Net Income Growth' in ratios:
            income_growth = ratios['Net Income Growth']
            if income_growth > 20:
                score += 7
            elif income_growth > 10:
                score += 5
            elif income_growth > 0:
                score += 3
        
        return score
    
    def _score_solvency(self, ratios):
        """偿债能力评分"""
        score = 0
        
        if '资产负债率' in ratios:
            debt_ratio = ratios['资产负债率']
            if debt_ratio < 30:
                score += 10
            elif debt_ratio < 50:
                score += 8
            elif debt_ratio < 70:
                score += 5
        
        if '流动比率' in ratios:
            current_ratio = ratios['流动比率']
            if current_ratio > 2:
                score += 10
            elif current_ratio > 1.5:
                score += 8
            elif current_ratio > 1:
                score += 5
        
        return score
    
    def _score_valuation(self, ratios):
        """估值合理性评分"""
        score = 0
        
        if 'PE' in ratios:
            pe = ratios['PE']
            if 0 < pe < 15:
                score += 8
            elif pe < 25:
                score += 6
            elif pe < 35:
                score += 3
        
        if 'PB' in ratios:
            pb = ratios['PB']
            if 0 < pb < 1.5:
                score += 7
            elif pb < 3:
                score += 5
            elif pb < 5:
                score += 3
        
        return score
    
    def _score_efficiency(self, ratios):
        """营运能力评分"""
        score = 0
        
        if 'ROA' in ratios:
            roa = ratios['ROA']
            if roa > 8:
                score += 5
            elif roa > 5:
                score += 3
            elif roa > 2:
                score += 1
        
        if '存货周转率' in ratios:
            inventory_turnover = ratios['存货周转率']
            if inventory_turnover > 6:
                score += 5
            elif inventory_turnover > 4:
                score += 3
            elif inventory_turnover > 2:
                score += 1
        
        return score
    
    def _score_asset_quality(self, ratios):
        """资产质量评分"""
        score = 0
        
        # 商誉占比低更好
        if 'Goodwill Ratio' in ratios:
            goodwill_ratio = ratios['Goodwill Ratio']
            if goodwill_ratio < 5:
                score += 5
            elif goodwill_ratio < 15:
                score += 3
            elif goodwill_ratio < 30:
                score += 1
        else:
            score += 5  # 没有商誉也是好事
        
        # 无形资产比例适中
        if 'Intangible Assets Ratio' in ratios:
            intangible_ratio = ratios['Intangible Assets Ratio']
            if intangible_ratio < 20:
                score += 5
            elif intangible_ratio < 40:
                score += 3
        else:
            score += 5
        
        return score
    
    def _score_market_risk(self, ratios):
        """市场风险评分"""
        score = 0
        
        if 'Volatility' in ratios:
            volatility = ratios['Volatility']
            if volatility < 20:
                score += 5
            elif volatility < 30:
                score += 3
            elif volatility < 50:
                score += 1
        
        return score
    
    def generate_recommendation(self, ratios):
        """生成投资建议"""
        if not ratios:
            return "数据不足，无法给出建议"
        
        score = self.calculate_financial_health_score(ratios)
        
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
    
    def identify_risks(self, ratios):
        """识别投资风险"""
        risks = []
        
        if not ratios:
            return ["数据不足，无法评估风险"]
        
        # 盈利能力风险
        if 'ROE' in ratios and ratios['ROE'] < 5:
            risks.append("盈利能力较弱，ROE低于5%")
        
        if '净利率' in ratios and ratios['净利率'] < 0:
            risks.append("公司处于亏损状态")
        
        # 成长性风险
        if 'Revenue Growth' in ratios and ratios['Revenue Growth'] < -5:
            risks.append("营收增长率为负，业务可能萎缩")
        
        if 'Net Income Growth' in ratios and ratios['Net Income Growth'] < -10:
            risks.append("净利润大幅下滑，盈利恶化")
        
        # 现金流质量风险
        if 'Operating CF to Net Income' in ratios and ratios['Operating CF to Net Income'] < 50:
            risks.append("经营现金流质量较差，可能存在利润操纵")
        
        if 'Free Cash Flow' in ratios and ratios['Free Cash Flow'] < 0:
            risks.append("自由现金流为负，资本支出压力大")
        
        # 资产质量风险
        if 'Goodwill Ratio' in ratios and ratios['Goodwill Ratio'] > 30:
            risks.append("商誉占比过高，存在减值风险")
        
        if 'Intangible Assets Ratio' in ratios and ratios['Intangible Assets Ratio'] > 50:
            risks.append("无形资产占比过高，资产质量存疑")
        
        # 偿债能力风险
        if '资产负债率' in ratios and ratios['资产负债率'] > 70:
            risks.append("资产负债率过高，偿债压力较大")
        
        if '流动比率' in ratios and ratios['流动比率'] < 1:
            risks.append("流动比率小于1，短期偿债能力不足")
        
        # 估值风险
        if 'PE' in ratios and ratios['PE'] > 50:
            risks.append("市盈率过高，可能存在估值泡沫")
        
        if 'PB' in ratios and ratios['PB'] > 10:
            risks.append("市净率过高，估值偏贵")
        
        # 市场风险
        if 'Volatility' in ratios and ratios['Volatility'] > 50:
            risks.append("股价波动率过高，投资风险较大")
        
        if 'Beta' in ratios and ratios['Beta'] > 2:
            risks.append("Beta系数过高，系统性风险敏感度大")
        
        # 股息风险
        if 'Payout Ratio' in ratios and ratios['Payout Ratio'] > 100:
            risks.append("派息比率超过100%，股息可持续性存疑")
        
        return risks


class PeerComparator:
    """同行对比分析器"""
    
    def __init__(self, target_symbol, market='US'):
        self.target_symbol = target_symbol
        self.market = market
    
    def compare_with_peers(self, peer_symbols, data_source_factory, metrics_calculator, start_year="2020"):
        """同行对比分析"""
        print(f"\n{'='*60}")
        print("同行对比分析")
        print('='*60)
        
        comparison_data = []
        all_symbols = [self.target_symbol] + peer_symbols
        
        for symbol in all_symbols:
            try:
                # 创建数据源
                data_source = data_source_factory.create_data_source(symbol, self.market)
                
                if data_source.load_company_data():
                    if self.market == 'US':
                        data = self._get_us_comparison_data(data_source, symbol)
                    elif self.market == 'CN':
                        data = self._get_cn_comparison_data(data_source, symbol, metrics_calculator, start_year)
                    
                    if data:
                        comparison_data.append(data)
                        print(f"✓ 获取 {symbol} 数据")
                
            except Exception as e:
                print(f"✗ 获取 {symbol} 数据失败: {e}")
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            print(f"\n同行对比表:")
            print(comparison_df.round(2).to_string(index=False))
            
            # 计算行业平均值
            self._print_industry_averages(comparison_df)
            
            return comparison_df
        
        return None
    
    def _get_us_comparison_data(self, data_source, symbol):
        """获取美股对比数据"""
        info = data_source.company_info
        
        return {
            'Symbol': symbol,
            'Company': info.get('shortName', symbol),
            'Market Cap': info.get('marketCap', 0) / 1e9,  # 转换为十亿
            'PE': info.get('forwardPE', 0),
            'PB': info.get('priceToBook', 0),
            'ROE': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'Debt/Equity': info.get('debtToEquity', 0),
            'Revenue Growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
        }
    
    def _get_cn_comparison_data(self, data_source, symbol, metrics_calculator, start_year):
        """获取中国股票对比数据"""
        info_dict = data_source.company_info
        
        # 获取财务指标数据
        financial_data = data_source.get_financial_statements(start_year)
        ratios = {}
        if financial_data:
            ratios = metrics_calculator.calculate_cn_ratios(financial_data, symbol, start_year)
        
        # 安全获取数值函数
        def safe_float(value, default=0):
            try:
                if value is None or value == '' or value == '-':
                    return default
                return float(str(value).replace(',', ''))
            except (ValueError, TypeError):
                return default
        
        return {
            'Symbol': symbol,
            'Company': info_dict.get('股票简称', symbol),
            'Market Cap': safe_float(info_dict.get('总市值', 0)) / 1e8,  # 转换为亿
            'PE': ratios.get('PE', 0),
            'PB': ratios.get('PB', 0),
            'ROE': ratios.get('ROE', 0),
            'ROA': ratios.get('ROA', 0),
            'Net Margin': ratios.get('净利率', 0),
            'Asset Turnover': ratios.get('总资产周转率', 0),
            'Debt/Equity': ratios.get('Debt/Equity', 0),
            'Current Ratio': ratios.get('流动比率', 0),
            'Revenue Growth': ratios.get('Revenue Growth', 0)
        }
    
    def _print_industry_averages(self, comparison_df):
        """打印行业平均值"""
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
                        target_value = comparison_df[comparison_df['Symbol'] == self.target_symbol][col].iloc[0]
                        
                        # 格式化显示
                        if col in ['PE', 'PB', 'Market Cap']:
                            print(f"  {col}: 行业均值 {avg_value:.2f}, {self.target_symbol} {target_value:.2f}")
                        else:
                            print(f"  {col}: 行业均值 {avg_value:.2f}%, {self.target_symbol} {target_value:.2f}%")