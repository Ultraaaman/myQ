"""
重构后的基本面分析器 - 使用模块化架构
"""
from .data_sources import DataSourceFactory
from .financial_metrics import FinancialMetricsCalculator
from .analysis_engine import FinancialHealthAnalyzer, PeerComparator
from .visualization import FinancialChartGenerator
from .valuation import DCFValuationModel, ComparativeValuationModel, DividendDiscountModel, ValuationSummary


class FundamentalAnalyzer:
    """重构后的基本面分析工具"""
    
    def __init__(self, symbol, market='US'):
        """
        初始化分析器
        Args:
            symbol: 股票代码
            market: 市场类型 ('US', 'CN', 'HK')
        """
        self.symbol = symbol.upper()
        self.market = market
        
        # 初始化各个组件
        self.data_source_factory = DataSourceFactory()
        self.data_source = self.data_source_factory.create_data_source(symbol, market)
        self.metrics_calculator = FinancialMetricsCalculator(symbol, market)
        self.health_analyzer = FinancialHealthAnalyzer()
        self.chart_generator = FinancialChartGenerator(symbol)
        self.dcf_model = DCFValuationModel(symbol, market)
        self.relative_model = ComparativeValuationModel(symbol, market)
        self.ddm_model = DividendDiscountModel(symbol, market)
        self.valuation_summary = ValuationSummary(symbol)
        
        # 数据存储
        self.company_info = {}
        self.financial_data = {}
        self.ratios = {}
        self.peer_comparison = None
    
    def load_company_data(self):
        """加载公司基本数据"""
        success = self.data_source.load_company_data()
        if success:
            self.company_info = self.data_source.company_info
        return success
    
    def analyze_financial_statements(self, start_year="2020"):
        """分析财务报表"""
        print(f"\n{'='*60}")
        print("财务报表分析")
        print('='*60)
        
        self.financial_data = self.data_source.get_financial_statements(start_year)
        
        if not self.financial_data:
            print("无法获取财务数据")
            return False
        
        self._print_financial_statements_summary()
        return True
    
    def _print_financial_statements_summary(self):
        """打印财务报表摘要"""
        if self.market == 'US':
            self._print_us_financial_summary()
        elif self.market == 'CN':
            self._print_cn_financial_summary()
    
    def _print_us_financial_summary(self):
        """打印美股财务摘要"""
        if not self.financial_data:
            return
        
        financials = self.financial_data.get('income_statement')
        balance_sheet = self.financial_data.get('balance_sheet')
        cash_flow = self.financial_data.get('cash_flow')
        
        print("财务报表数据获取成功:")
        if financials is not None:
            print(f"  损益表: {financials.shape[1]} 个报告期")
        if balance_sheet is not None:
            print(f"  资产负债表: {balance_sheet.shape[1]} 个报告期")
        if cash_flow is not None:
            print(f"  现金流量表: {cash_flow.shape[1]} 个报告期")
        
        # 打印最新财务数据
        if financials is not None and not financials.empty:
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
            if balance_sheet is not None and 'Total Assets' in balance_sheet.index:
                total_assets = balance_sheet.loc['Total Assets', latest_period]
                print(f"  总资产: ${total_assets:,.0f}")
            
            if balance_sheet is not None and 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Debt', latest_period]
                print(f"  总负债: ${total_debt:,.0f}")
            
            # 现金流关键项目
            if cash_flow is not None and 'Operating Cash Flow' in cash_flow.index:
                operating_cf = cash_flow.loc['Operating Cash Flow', latest_period]
                print(f"  经营现金流: ${operating_cf:,.0f}")
    
    def _print_cn_financial_summary(self):
        """打印中国股票财务摘要"""
        indicators = self.financial_data.get('indicators')
        
        if indicators is not None and not indicators.empty:
            print("财务指标数据获取成功:")
            print(f"  数据期间: {len(indicators)} 个报告期")
            print(f"  数据范围: {indicators['日期'].iloc[0].strftime('%Y-%m-%d')} 至 {indicators['日期'].iloc[-1].strftime('%Y-%m-%d')}")
            
            # 最新财务指标
            latest = indicators.iloc[-1]
            print(f"\n最新财务指标 ({latest.get('日期', 'N/A')}):")
            
            key_metrics = [
                ('摊薄每股收益', '摊薄每股收益(元)'),
                ('每股净资产', '每股净资产_调整后(元)'),
                ('净资产收益率', '净资产收益率(%)'),
                ('总资产净利润率', '总资产净利润率(%)'),
                ('销售净利率', '销售净利率(%)'),
                ('资产负债率', '资产负债率(%)'),
                ('流动比率', '流动比率')
            ]
            
            for name, key in key_metrics:
                if key in latest.index:
                    value = latest[key]
                    if pd.notna(value):
                        if key.endswith('(%)'):
                            print(f"  {name}: {value:.2f}%")
                        elif key.endswith('(元)'):
                            print(f"  {name}: {value:.2f} 元")
                        else:
                            print(f"  {name}: {value:.2f}")
    
    def calculate_financial_ratios(self, start_year="2020"):
        """计算财务比率"""
        print(f"\n{'='*60}")
        print("财务比率分析")
        print('='*60)
        
        if not self.financial_data:
            print("请先分析财务报表")
            return False
        
        if self.market == 'US':
            self.ratios = self.metrics_calculator.calculate_us_ratios(
                self.financial_data, self.company_info, self.data_source.ticker
            )
        elif self.market == 'CN':
            self.ratios = self.metrics_calculator.calculate_cn_ratios(
                self.financial_data, self.symbol, start_year
            )
        
        # 添加市场表现指标
        market_ratios = self.metrics_calculator.calculate_market_performance_ratios(
            self.symbol, self.market
        )
        self.ratios.update(market_ratios)
        
        # 打印财务比率
        if self.ratios:
            print("关键财务比率:")
            for ratio_name, ratio_value in self.ratios.items():
                if ('率' in ratio_name or 'ROE' in ratio_name or 'ROA' in ratio_name or 
                    'Growth' in ratio_name or 'Yield' in ratio_name or 'Volatility' in ratio_name or 
                    'Change' in ratio_name) and ratio_name not in ['PE', 'PB', '流动比率', '速动比率', 
                                                                 'Beta', 'Sharpe Ratio']:
                    print(f"  {ratio_name}: {ratio_value:.2f}%")
                else:
                    print(f"  {ratio_name}: {ratio_value:.2f}")
            return True
        
        return False
    
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
    
    def peer_comparison_analysis(self, peer_symbols, start_year="2020"):
        """同行对比分析"""
        peer_comparator = PeerComparator(self.symbol, self.market)
        self.peer_comparison = peer_comparator.compare_with_peers(
            peer_symbols, self.data_source_factory, self.metrics_calculator, start_year
        )
        return self.peer_comparison
    
    def dcf_valuation(self, growth_years=5, terminal_growth=2.5, discount_rate=10):
        """DCF估值模型"""
        return self.dcf_model.calculate_dcf_valuation(
            self.financial_data, self.company_info, 
            getattr(self.data_source, 'ticker', None),
            growth_years, terminal_growth, discount_rate
        )
    
    def relative_valuation(self):
        """相对估值分析"""
        if not self.peer_comparison or self.peer_comparison.empty:
            print("需要先进行同行对比分析")
            return None
        
        return self.relative_model.calculate_relative_valuation(
            self.ratios, self.peer_comparison
        )
    
    def dividend_valuation(self, required_return=0.10):
        """股息折现模型估值"""
        # 准备包含当前价格的财务数据
        financial_data_with_price = self.financial_data.copy()
        
        # 尝试获取当前价格
        if hasattr(self.data_source, 'ticker') and self.data_source.ticker:
            hist_data = self.data_source.ticker.history(period='1d')
            if not hist_data.empty:
                financial_data_with_price['current_price'] = hist_data['Close'][-1]
        
        return self.ddm_model.calculate_ddm_valuation(
            self.ratios, financial_data_with_price, required_return
        )
    
    def comprehensive_valuation(self, growth_years=5, terminal_growth=2.5, 
                              discount_rate=10, required_return=0.10):
        """综合估值分析"""
        dcf_result = self.dcf_valuation(growth_years, terminal_growth, discount_rate)
        relative_result = self.relative_valuation()
        ddm_result = self.dividend_valuation(required_return)
        
        return self.valuation_summary.generate_valuation_summary(
            dcf_result, relative_result, ddm_result
        )
    
    def generate_investment_summary(self):
        """生成投资分析摘要"""
        print(f"\n{'='*60}")
        print(f"{self.symbol} 投资分析摘要")
        print('='*60)
        
        if self.company_info:
            if self.market == 'US':
                company_name = self.company_info.get('shortName', self.symbol)
                sector = self.company_info.get('sector', 'N/A')
                industry = self.company_info.get('industry', 'N/A')
                
                print(f"公司: {company_name}")
                print(f"行业: {sector} - {industry}")
                
                # 获取当前股价
                if hasattr(self.data_source, 'ticker'):
                    hist = self.data_source.ticker.history(period='1d')
                    if not hist.empty:
                        current_price = hist['Close'][-1]
                        print(f"当前股价: ${current_price:.2f}")
                
            elif self.market == 'CN':
                company_name = self.company_info.get('股票简称', self.symbol)
                print(f"公司: {company_name}")
        
        # 财务健康度评分
        health_score = self.health_analyzer.calculate_financial_health_score(self.ratios)
        print(f"\n财务健康度评分: {health_score}/100")
        
        # 投资建议
        recommendation = self.health_analyzer.generate_recommendation(self.ratios)
        print(f"投资建议: {recommendation}")
        
        # 风险提示
        risks = self.health_analyzer.identify_risks(self.ratios)
        if risks:
            print(f"\n风险提示:")
            for risk in risks:
                print(f"  - {risk}")
    
    def plot_financial_analysis(self):
        """绘制财务分析图表"""
        self.chart_generator.plot_comprehensive_analysis(
            self.ratios, self.peer_comparison, self.financial_data
        )
    
    def plot_detailed_ratios_chart(self):
        """绘制详细财务比率图表"""
        self.chart_generator.plot_detailed_ratios_summary(self.ratios)


# 导入pandas用于某些函数
import pandas as pd