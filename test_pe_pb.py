"""
专门测试PE和PB指标的计算
"""
import sys
sys.path.append('.')

def test_pe_pb_calculation():
    """测试PE和PB计算"""
    print("🔢 专门测试PE和PB计算功能")
    print("="*50)
    
    try:
        import akshare as ak
        print("✅ akshare 库可用")
    except ImportError:
        print("❌ akshare 库未安装，无法测试PE/PB")
        return False
    
    try:
        from quantlib.fundamental.analyzer_refactored import FundamentalAnalyzer
        
        # 测试多只股票的PE和PB
        test_symbols = ['000001', '000002', '600036', '000858', '002415']  
        
        for symbol in test_symbols:
            print(f"\n📊 测试 {symbol} 的PE和PB...")
            print("-" * 30)
            
            analyzer = FundamentalAnalyzer(symbol, market='CN')
            
            if analyzer.load_company_data():
                company_name = analyzer.company_info.get('股票简称', 'N/A')
                print(f"公司: {company_name}")
                
                if analyzer.analyze_financial_statements():
                    if analyzer.calculate_financial_ratios():
                        print(f"✓ 计算出 {len(analyzer.ratios)} 个财务指标")
                        
                        # 重点检查PE和PB
                        if 'PE' in analyzer.ratios:
                            print(f"✅ PE: {analyzer.ratios['PE']:.2f}")
                        else:
                            print("❌ 未找到PE指标")
                        
                        if 'PB' in analyzer.ratios:
                            print(f"✅ PB: {analyzer.ratios['PB']:.2f}")
                        else:
                            print("❌ 未找到PB指标")
                        
                        # 显示其他关键指标
                        other_ratios = ['ROE', 'ROA', '净利率', '资产负债率']
                        for ratio in other_ratios:
                            if ratio in analyzer.ratios:
                                print(f"  {ratio}: {analyzer.ratios[ratio]:.2f}%")
                    else:
                        print("❌ 财务比率计算失败")
                else:
                    print("❌ 财务报表分析失败")
            else:
                print("❌ 公司数据加载失败")
            
            print("-" * 30)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_akshare_pe_pb():
    """直接使用akshare测试PE和PB数据可用性"""
    print(f"\n🔍 直接测试akshare的PE/PB数据...")
    print("="*50)
    
    try:
        import akshare as ak
        
        symbol = '000001'  # 平安银行
        print(f"获取 {symbol} 的基本信息...")
        
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        
        print(f"获取到 {len(stock_info)} 个信息字段:")
        
        pe_found = False
        pb_found = False
        
        for _, row in stock_info.iterrows():
            item_name = row['item']
            value = row['value']
            
            # 打印所有包含数字的字段，帮助调试
            if any(keyword in item_name for keyword in ['市盈率', '市净率', 'PE', 'PB', '盈率', '净率']):
                print(f"  {item_name}: {value}")
                
                # PE搜索
                if '市盈率' in item_name or 'PE' in item_name or '盈率' in item_name:
                    try:
                        if isinstance(value, str):
                            clean_value = value.replace(',', '').replace('倍', '').strip()
                        else:
                            clean_value = str(value)
                        pe_val = float(clean_value)
                        if pe_val > 0 and pe_val < 1000:
                            print(f"    ✅ 可用作PE: {pe_val}")
                            pe_found = True
                    except:
                        print(f"    ❌ PE解析失败: {value}")
                
                # PB搜索
                if '市净率' in item_name or 'PB' in item_name or '净率' in item_name:
                    try:
                        if isinstance(value, str):
                            clean_value = value.replace(',', '').replace('倍', '').strip()
                        else:
                            clean_value = str(value)
                        pb_val = float(clean_value)
                        if pb_val > 0 and pb_val < 100:
                            print(f"    ✅ 可用作PB: {pb_val}")
                            pb_found = True
                    except:
                        print(f"    ❌ PB解析失败: {value}")
        
        print(f"\n结果: PE找到={pe_found}, PB找到={pb_found}")
        return pe_found or pb_found
        
    except Exception as e:
        print(f"❌ 直接akshare测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 PE和PB指标专项测试")
    print("="*50)
    print("验证重构后是否保留了原有的PE/PB计算功能")
    print()
    
    # 直接测试akshare数据
    akshare_test = test_direct_akshare_pe_pb()
    
    # 测试重构后的计算
    refactor_test = test_pe_pb_calculation()
    
    print(f"\n{'='*50}")
    print("📋 测试结果:")
    print(f"Akshare数据可用性: {'✅' if akshare_test else '❌'}")
    print(f"重构后PE/PB计算: {'✅' if refactor_test else '❌'}")
    
    if akshare_test and refactor_test:
        print("\n🎉 PE和PB功能完全正常！")
    elif akshare_test and not refactor_test:
        print("\n⚠️ akshare数据正常，但重构后的计算有问题")
        print("需要检查重构后的PE/PB计算逻辑")
    elif not akshare_test:
        print("\n⚠️ akshare数据源可能有问题")
        print("可能是数据字段名称变化或网络问题")
    
    print("="*50)

if __name__ == "__main__":
    main()