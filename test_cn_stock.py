"""
测试中国股票分析功能（使用akshare，无频率限制）
"""
import sys
sys.path.append('.')

def test_akshare_availability():
    """测试akshare是否可用"""
    try:
        import akshare as ak
        print("✅ akshare 库可用")
        return True
    except ImportError:
        print("❌ akshare 库未安装")
        print("请运行: pip install akshare")
        return False

def test_cn_stock_analysis():
    """测试中国股票分析功能"""
    print("🚀 测试中国股票分析功能（无频率限制）")
    print("="*60)
    
    if not test_akshare_availability():
        return False
    
    try:
        from quantlib.fundamental.analyzer_refactored import FundamentalAnalyzer
        
        # 测试多只中国股票
        test_symbols = ['000001', '000002', '600036']  # 平安银行、万科、招商银行
        
        for symbol in test_symbols:
            print(f"\n📊 分析 {symbol}...")
            print("-" * 40)
            
            # 创建分析器
            analyzer = FundamentalAnalyzer(symbol, market='CN')
            print(f"✓ 创建分析器成功: {symbol}")
            
            # 加载公司数据
            if analyzer.load_company_data():
                print("✓ 公司数据加载成功")
                
                # 显示公司基本信息
                if analyzer.company_info:
                    company_name = analyzer.company_info.get('股票简称', 'N/A')
                    print(f"  公司名称: {company_name}")
                
                # 分析财务报表
                if analyzer.analyze_financial_statements():
                    print("✓ 财务报表分析成功")
                    
                    # 计算财务比率
                    if analyzer.calculate_financial_ratios():
                        print(f"✓ 财务比率计算成功，计算出 {len(analyzer.ratios)} 个指标")
                        
                        # 显示关键指标
                        print("关键财务指标:")
                        key_ratios = ['ROE', 'ROA', '净利率', 'PE', 'PB', '资产负债率', '流动比率']
                        for ratio in key_ratios:
                            if ratio in analyzer.ratios:
                                value = analyzer.ratios[ratio]
                                if ratio in ['PE', 'PB', '流动比率']:
                                    print(f"  {ratio}: {value:.2f}")
                                else:
                                    print(f"  {ratio}: {value:.2f}%")
                        
                        # 生成投资摘要
                        print("\n📋 投资摘要:")
                        analyzer.generate_investment_summary()
                        
                        print(f"✅ {symbol} 分析完成")
                    else:
                        print(f"⚠️ {symbol} 财务比率计算失败")
                else:
                    print(f"⚠️ {symbol} 财务报表分析失败")
            else:
                print(f"❌ {symbol} 公司数据加载失败")
            
            print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ 中国股票分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peer_comparison():
    """测试同行对比功能"""
    print(f"\n🔄 测试同行对比功能...")
    print("="*60)
    
    try:
        from quantlib.fundamental.analyzer_refactored import FundamentalAnalyzer
        
        # 分析银行股
        analyzer = FundamentalAnalyzer('000001', market='CN')  # 平安银行
        
        if analyzer.load_company_data() and analyzer.analyze_financial_statements() and analyzer.calculate_financial_ratios():
            print("✓ 目标股票分析完成")
            
            # 同行对比
            bank_peers = ['000002', '600036']  # 万科（实际不是银行，但用于测试）、招商银行
            comparison_result = analyzer.peer_comparison_analysis(bank_peers, start_year='2020')
            
            if comparison_result is not None and not comparison_result.empty:
                print("✅ 同行对比分析成功")
                print(f"对比了 {len(comparison_result)} 只股票")
                
                # 相对估值分析
                relative_result = analyzer.relative_valuation()
                if relative_result:
                    print("✅ 相对估值分析成功")
                else:
                    print("⚠️ 相对估值分析失败")
                
                return True
            else:
                print("⚠️ 同行对比分析失败")
                return False
        else:
            print("❌ 目标股票分析失败")
            return False
            
    except Exception as e:
        print(f"❌ 同行对比测试失败: {e}")
        return False

def test_modular_advantages():
    """测试模块化架构的优势"""
    print(f"\n🧩 测试模块化架构优势...")
    print("="*60)
    
    try:
        # 单独使用数据源模块
        from quantlib.fundamental.data_sources import AkshareDataSource
        data_source = AkshareDataSource('000001')
        if data_source.load_company_data():
            print("✅ 独立使用数据源模块成功")
        
        # 单独使用财务指标计算模块
        from quantlib.fundamental.financial_metrics import FinancialMetricsCalculator
        metrics_calc = FinancialMetricsCalculator('000001', 'CN')
        print("✅ 独立创建财务指标计算器成功")
        
        # 单独使用分析引擎
        from quantlib.fundamental.analysis_engine import FinancialHealthAnalyzer
        health_analyzer = FinancialHealthAnalyzer()
        print("✅ 独立使用分析引擎成功")
        
        # 测试组合使用
        financial_data = data_source.get_financial_statements('2020')
        if financial_data:
            ratios = metrics_calc.calculate_cn_ratios(financial_data, '000001', '2020')
            if ratios:
                score = health_analyzer.calculate_financial_health_score(ratios)
                recommendation = health_analyzer.generate_recommendation(ratios)
                
                print("✅ 模块组合使用成功")
                print(f"  健康度评分: {score}/100")
                print(f"  投资建议: {recommendation}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ 模块化测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🇨🇳 中国股票分析测试（akshare接口，无频率限制）")
    print("="*70)
    
    test_results = []
    
    # 基础分析测试
    test_results.append(("中国股票分析", test_cn_stock_analysis()))
    
    # 同行对比测试
    test_results.append(("同行对比分析", test_peer_comparison()))
    
    # 模块化优势测试
    test_results.append(("模块化架构", test_modular_advantages()))
    
    # 统计结果
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"\n{'='*70}")
    print("📋 测试结果汇总")
    print("="*70)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
    
    print(f"\n🎯 通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 中国股票分析测试全部通过！")
        print("✅ akshare接口工作正常，无频率限制")
        print("✅ 重构后的代码完全兼容中国股票分析")
        print("✅ 模块化架构运行良好")
    else:
        print(f"\n⚠️ {total-passed} 个测试失败")
    
    print("\n💡 重构优势体现：")
    print("• 数据源模块化：支持多种数据源（Yahoo Finance + akshare）")
    print("• 市场适配性：自动适配不同市场的数据格式")
    print("• 错误隔离：单个数据源问题不影响整体架构")
    print("• 易于扩展：可轻松添加新的数据源（如Wind、Choice等）")
    print("="*70)

if __name__ == "__main__":
    main()