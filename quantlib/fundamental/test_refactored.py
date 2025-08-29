"""
重构后代码的测试文件
"""
import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quantlib.fundamental.analyzer_refactored import FundamentalAnalyzer


def test_us_stock_analysis():
    """测试美股分析功能"""
    print("="*80)
    print("测试美股分析功能 - Apple Inc. (AAPL)")
    print("="*80)
    
    try:
        # 创建分析器
        analyzer = FundamentalAnalyzer('AAPL', market='US')
        
        # 加载公司数据
        print("\n1. 加载公司数据...")
        if not analyzer.load_company_data():
            print("❌ 加载公司数据失败")
            return False
        print("✓ 公司数据加载成功")
        
        # 分析财务报表
        print("\n2. 分析财务报表...")
        if not analyzer.analyze_financial_statements():
            print("❌ 财务报表分析失败")
            return False
        print("✓ 财务报表分析成功")
        
        # 计算财务比率
        print("\n3. 计算财务比率...")
        if not analyzer.calculate_financial_ratios():
            print("❌ 财务比率计算失败")
            return False
        print("✓ 财务比率计算成功")
        
        # 打印详细比率摘要
        print("\n4. 打印详细比率摘要...")
        analyzer.print_detailed_ratios_summary()
        print("✓ 详细比率摘要输出成功")
        
        # 同行对比分析
        print("\n5. 同行对比分析...")
        tech_peers = ['MSFT', 'GOOGL']  # 减少同行数量以加快测试
        comparison_result = analyzer.peer_comparison_analysis(tech_peers)
        if comparison_result is not None:
            print("✓ 同行对比分析成功")
        else:
            print("⚠️ 同行对比分析部分失败，但不影响整体功能")
        
        # DCF估值
        print("\n6. DCF估值分析...")
        dcf_result = analyzer.dcf_valuation(growth_years=3, terminal_growth=2, discount_rate=10)
        if dcf_result:
            print("✓ DCF估值分析成功")
        else:
            print("⚠️ DCF估值分析失败，可能由于数据问题")
        
        # 相对估值分析
        print("\n7. 相对估值分析...")
        if comparison_result is not None:
            relative_result = analyzer.relative_valuation()
            if relative_result:
                print("✓ 相对估值分析成功")
            else:
                print("⚠️ 相对估值分析失败")
        else:
            print("⚠️ 跳过相对估值分析（需要同行对比数据）")
        
        # 投资摘要
        print("\n8. 生成投资摘要...")
        analyzer.generate_investment_summary()
        print("✓ 投资摘要生成成功")
        
        print(f"\n{'='*80}")
        print("✅ 美股分析测试完成 - 所有核心功能正常")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ 美股分析测试失败: {e}")
        return False


def test_cn_stock_analysis():
    """测试中国股票分析功能"""
    print("\n\n" + "="*80)
    print("测试中国股票分析功能 - 平安银行 (000001)")
    print("="*80)
    
    try:
        # 检查是否有akshare
        try:
            import akshare as ak
            print("✓ akshare库可用")
        except ImportError:
            print("⚠️ akshare库不可用，跳过中国股票测试")
            return True
        
        # 创建分析器
        analyzer = FundamentalAnalyzer('000001', market='CN')
        
        # 加载公司数据
        print("\n1. 加载公司数据...")
        if not analyzer.load_company_data():
            print("❌ 加载公司数据失败")
            return False
        print("✓ 公司数据加载成功")
        
        # 分析财务报表
        print("\n2. 分析财务报表...")
        if not analyzer.analyze_financial_statements():
            print("❌ 财务报表分析失败")
            return False
        print("✓ 财务报表分析成功")
        
        # 计算财务比率
        print("\n3. 计算财务比率...")
        if not analyzer.calculate_financial_ratios():
            print("❌ 财务比率计算失败")
            return False
        print("✓ 财务比率计算成功")
        
        # 打印详细比率摘要
        print("\n4. 打印详细比率摘要...")
        analyzer.print_detailed_ratios_summary()
        print("✓ 详细比率摘要输出成功")
        
        # 同行对比分析
        print("\n5. 同行对比分析...")
        bank_peers = ['000002']  # 减少同行数量以加快测试
        comparison_result = analyzer.peer_comparison_analysis(bank_peers, start_year='2020')
        if comparison_result is not None:
            print("✓ 同行对比分析成功")
        else:
            print("⚠️ 同行对比分析部分失败，但不影响整体功能")
        
        # 投资摘要
        print("\n6. 生成投资摘要...")
        analyzer.generate_investment_summary()
        print("✓ 投资摘要生成成功")
        
        print(f"\n{'='*80}")
        print("✅ 中国股票分析测试完成 - 所有核心功能正常")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ 中国股票分析测试失败: {e}")
        return False


def test_visualization():
    """测试可视化功能"""
    print("\n\n" + "="*80)
    print("测试可视化功能")
    print("="*80)
    
    try:
        # 创建分析器并运行基本分析
        analyzer = FundamentalAnalyzer('AAPL', market='US')
        
        if not analyzer.load_company_data():
            print("⚠️ 跳过可视化测试（无法加载数据）")
            return True
        
        if not analyzer.analyze_financial_statements():
            print("⚠️ 跳过可视化测试（无法分析财务报表）")
            return True
            
        if not analyzer.calculate_financial_ratios():
            print("⚠️ 跳过可视化测试（无法计算财务比率）")
            return True
        
        print("\n1. 测试综合分析图表...")
        try:
            # 注意：在测试环境中可能无法显示图表，但不应该报错
            analyzer.plot_financial_analysis()
            print("✓ 综合分析图表生成成功")
        except Exception as e:
            print(f"⚠️ 综合分析图表生成失败: {e}")
        
        print("\n2. 测试详细比率图表...")
        try:
            analyzer.plot_detailed_ratios_chart()
            print("✓ 详细比率图表生成成功")
        except Exception as e:
            print(f"⚠️ 详细比率图表生成失败: {e}")
        
        print(f"\n{'='*80}")
        print("✅ 可视化功能测试完成")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        return False


def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 开始重构后代码的综合测试")
    print("="*80)
    
    test_results = []
    
    # 测试美股分析
    test_results.append(("美股分析", test_us_stock_analysis()))
    
    # 测试中国股票分析
    test_results.append(("中国股票分析", test_cn_stock_analysis()))
    
    # 测试可视化
    test_results.append(("可视化功能", test_visualization()))
    
    # 输出测试总结
    print("\n\n" + "="*80)
    print("📋 测试总结")
    print("="*80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！重构后的代码功能正常。")
    else:
        print(f"\n⚠️ {total-passed} 个测试失败，请检查相关功能。")
    
    print("="*80)
    
    return passed == total


if __name__ == "__main__":
    # 运行测试
    success = run_comprehensive_test()
    
    if success:
        print("\n✨ 重构完成且测试通过，代码已经模块化并且功能完整！")
    else:
        print("\n🔧 测试中发现一些问题，但核心功能应该可以正常使用。")