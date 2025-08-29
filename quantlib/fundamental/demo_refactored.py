"""
重构后代码的演示文件
"""
from analyzer_refactored import FundamentalAnalyzer


def demo_us_stock():
    """演示美股分析"""
    print("="*80)
    print("重构后基本面分析工具演示 - 美股")
    print("="*80)
    
    print("\n📊 分析Apple Inc. (AAPL)")
    print("-" * 50)
    
    # 创建分析器
    analyzer = FundamentalAnalyzer('AAPL', market='US')
    
    # 加载公司数据
    if analyzer.load_company_data():
        # 分析财务报表
        analyzer.analyze_financial_statements()
        
        # 计算财务比率
        analyzer.calculate_financial_ratios()
        
        # 打印详细比率摘要
        analyzer.print_detailed_ratios_summary()
        
        # 同行对比（科技股）
        tech_peers = ['MSFT', 'GOOGL', 'AMZN']
        analyzer.peer_comparison_analysis(tech_peers)
        
        # DCF估值
        analyzer.dcf_valuation(growth_years=5, terminal_growth=3, discount_rate=10)
        
        # 综合估值分析
        analyzer.comprehensive_valuation()
        
        # 生成投资摘要
        analyzer.generate_investment_summary()
        
        # 绘制分析图表
        try:
            analyzer.plot_financial_analysis()
            analyzer.plot_detailed_ratios_chart()
        except Exception as e:
            print(f"图表绘制遇到问题: {e}")


def demo_cn_stock():
    """演示中国股票分析"""
    try:
        import akshare as ak
    except ImportError:
        print("需要安装akshare库: pip install akshare")
        return
    
    print(f"\n\n📊 分析平安银行 (000001)")
    print("-" * 50)
    
    # 创建分析器
    analyzer = FundamentalAnalyzer('000001', market='CN')
    
    # 加载公司数据
    if analyzer.load_company_data():
        # 分析财务报表
        analyzer.analyze_financial_statements()
        
        # 计算财务比率  
        analyzer.calculate_financial_ratios()
        
        # 打印详细比率摘要
        analyzer.print_detailed_ratios_summary()
        
        # 同行对比（银行股）
        bank_peers = ['000002', '600036']
        analyzer.peer_comparison_analysis(bank_peers, start_year='2020')
        
        # 相对估值分析
        analyzer.relative_valuation()
        
        # 生成投资摘要
        analyzer.generate_investment_summary()


def demo_modular_usage():
    """演示模块化使用"""
    print(f"\n\n📚 模块化使用演示")
    print("-" * 50)
    
    from data_sources import DataSourceFactory
    from financial_metrics import FinancialMetricsCalculator
    from analysis_engine import FinancialHealthAnalyzer
    
    # 使用工厂模式创建数据源
    data_source = DataSourceFactory.create_data_source('AAPL', 'US')
    
    if data_source.load_company_data():
        print("✓ 使用DataSourceFactory成功加载数据")
        
        # 使用财务指标计算器
        financial_data = data_source.get_financial_statements()
        if financial_data:
            metrics_calc = FinancialMetricsCalculator('AAPL', 'US')
            ratios = metrics_calc.calculate_us_ratios(
                financial_data, data_source.company_info, data_source.ticker
            )
            print(f"✓ 计算出 {len(ratios)} 个财务指标")
            
            # 使用健康度分析器
            health_analyzer = FinancialHealthAnalyzer()
            score = health_analyzer.calculate_financial_health_score(ratios)
            recommendation = health_analyzer.generate_recommendation(ratios)
            
            print(f"✓ 财务健康度评分: {score}/100")
            print(f"✓ 投资建议: {recommendation}")


def main():
    """主函数"""
    print("🚀 重构后基本面分析工具完整演示")
    print("="*80)
    print("新的模块化架构具有以下优势：")
    print("✓ 代码结构清晰，易于维护")
    print("✓ 模块职责单一，易于扩展")
    print("✓ 支持单独使用各个模块")
    print("✓ 提高了代码的可测试性")
    print("✓ 便于添加新的数据源和分析方法")
    
    # 美股分析演示
    demo_us_stock()
    
    # 中国股票分析演示
    demo_cn_stock()
    
    # 模块化使用演示
    demo_modular_usage()
    
    print(f"\n\n🎉 演示完成！")
    print("="*80)
    print("现在您可以：")
    print("1. 使用 FundamentalAnalyzer 进行完整分析")
    print("2. 单独使用各个模块进行特定功能开发")
    print("3. 轻松扩展新的数据源或分析方法")
    print("4. 对代码进行单元测试")


if __name__ == "__main__":
    main()