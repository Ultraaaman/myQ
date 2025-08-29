"""
手动调试akshare API问题
"""
import sys
sys.path.append('.')

def debug_financial_analysis_indicator():
    """详细调试财务分析指标API"""
    print("🔍 详细调试 stock_financial_analysis_indicator 接口")
    print("="*60)
    
    try:
        import akshare as ak
        print(f"✅ akshare版本: {ak.__version__}")
        
        # 测试参数
        test_params = [
            {"symbol": "000001", "start_year": "2020"},
            {"symbol": "000001", "start_year": "2021"},
            {"symbol": "600036", "start_year": "2020"},
            {"symbol": "000002", "start_year": "2020"},
        ]
        
        for i, params in enumerate(test_params, 1):
            print(f"\n📊 测试 {i}: {params}")
            print("-" * 40)
            
            try:
                print("调用前状态检查...")
                print(f"参数类型: symbol={type(params['symbol'])}, start_year={type(params['start_year'])}")
                
                print("正在调用 ak.stock_financial_analysis_indicator...")
                result = ak.stock_financial_analysis_indicator(
                    symbol=params["symbol"], 
                    start_year=params["start_year"]
                )
                
                print(f"返回结果类型: {type(result)}")
                
                if result is None:
                    print("❌ 返回 None")
                elif hasattr(result, 'empty'):
                    if result.empty:
                        print("⚠️ 返回空DataFrame")
                    else:
                        print(f"✅ 返回有效DataFrame")
                        print(f"   形状: {result.shape}")
                        print(f"   列名: {list(result.columns)[:10]}...")  # 前10个列名
                        print(f"   前几行预览:")
                        try:
                            print(result.head(2))
                        except:
                            print("   无法显示预览")
                else:
                    print(f"⚠️ 返回非DataFrame类型: {type(result)}")
                    print(f"   内容: {str(result)[:200]}...")
                    
            except Exception as e:
                print(f"❌ 调用失败: {e}")
                print(f"   错误类型: {type(e)}")
                
                # 尝试获取更详细的错误信息
                import traceback
                error_trace = traceback.format_exc()
                print(f"   详细错误追踪:")
                print(f"   {error_trace}")
                
                # 分析错误原因
                error_str = str(e)
                if "'NoneType' object has no attribute 'find'" in error_str:
                    print("🔍 分析: 这个错误通常表示:")
                    print("   1. 网络请求返回了None而不是预期的HTML/XML")
                    print("   2. akshare内部解析逻辑遇到了空响应")
                    print("   3. 目标网站返回了异常页面")
                    
        return True
        
    except ImportError:
        print("❌ akshare未安装")
        return False
    except Exception as e:
        print(f"❌ 整体测试失败: {e}")
        return False

def debug_individual_info_api():
    """调试股票个股信息API"""
    print(f"\n🏢 调试 stock_individual_info_em 接口")
    print("="*50)
    
    try:
        import akshare as ak
        
        test_symbols = ["000001", "600036", "000002"]
        
        for symbol in test_symbols:
            print(f"\n📈 测试股票: {symbol}")
            print("-" * 25)
            
            try:
                result = ak.stock_individual_info_em(symbol=symbol)
                
                if result is None:
                    print("❌ 返回 None")
                elif hasattr(result, 'empty'):
                    if result.empty:
                        print("⚠️ 返回空DataFrame")
                    else:
                        print(f"✅ 返回有效数据: {len(result)} 行")
                        # 显示部分数据
                        for i, (_, row) in enumerate(result.head(5).iterrows()):
                            print(f"   {row['item']}: {row['value']}")
                        if len(result) > 5:
                            print(f"   ... 还有 {len(result) - 5} 行数据")
                else:
                    print(f"⚠️ 返回类型: {type(result)}")
                    
            except Exception as e:
                print(f"❌ 失败: {e}")
                
    except Exception as e:
        print(f"❌ 个股信息测试失败: {e}")

def debug_stock_history_api():
    """调试股票历史数据API"""
    print(f"\n📈 调试 stock_zh_a_hist 接口")
    print("="*50)
    
    try:
        import akshare as ak
        
        test_cases = [
            {"symbol": "000001", "period": "daily", "adjust": ""},
            {"symbol": "000001", "period": "daily", "adjust": "qfq"},
            {"symbol": "600036", "period": "daily", "adjust": ""},
        ]
        
        for case in test_cases:
            print(f"\n📊 测试: {case}")
            print("-" * 30)
            
            try:
                result = ak.stock_zh_a_hist(**case)
                
                if result is None:
                    print("❌ 返回 None")
                elif hasattr(result, 'empty'):
                    if result.empty:
                        print("⚠️ 返回空DataFrame")
                    else:
                        print(f"✅ 返回有效数据: {result.shape}")
                        print(f"   列名: {list(result.columns)}")
                        print(f"   最新价格: {result.iloc[-1]['收盘']}")
                else:
                    print(f"⚠️ 返回类型: {type(result)}")
                    
            except Exception as e:
                print(f"❌ 失败: {e}")
                
    except Exception as e:
        print(f"❌ 历史数据测试失败: {e}")

def check_network_and_environment():
    """检查网络和环境"""
    print(f"\n🌐 检查网络和环境")
    print("="*40)
    
    try:
        import requests
        import time
        
        # 测试网络连通性
        print("📡 测试网络连接...")
        test_urls = [
            "https://www.baidu.com",
            "https://push2.eastmoney.com",  # akshare常用的数据源之一
            "https://datacenter-web.eastmoney.com"
        ]
        
        for url in test_urls:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                end_time = time.time()
                
                if response.status_code == 200:
                    print(f"✅ {url} - {response.status_code} ({end_time-start_time:.2f}s)")
                else:
                    print(f"⚠️ {url} - {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {url} - {e}")
        
        # 检查Python环境
        print(f"\n🐍 Python环境信息:")
        print(f"   Python版本: {sys.version}")
        
        # 检查相关包版本
        packages = ['akshare', 'pandas', 'requests', 'lxml']
        print(f"\n📦 相关包版本:")
        for pkg in packages:
            try:
                module = __import__(pkg)
                version = getattr(module, '__version__', '未知')
                print(f"   {pkg}: {version}")
            except ImportError:
                print(f"   {pkg}: 未安装")
                
    except Exception as e:
        print(f"❌ 环境检查失败: {e}")

def suggest_fixes():
    """建议修复方法"""
    print(f"\n💡 问题修复建议")
    print("="*40)
    
    print("🔧 针对 'NoneType' object has no attribute 'find' 错误:")
    print("1. 网络问题 - 检查网络连接和防火墙设置")
    print("2. akshare版本 - 尝试更新: pip install akshare --upgrade")
    print("3. 数据源问题 - 东方财富等网站可能临时不可用")
    print("4. 请求频率 - 添加延迟: time.sleep(1)")
    print("5. 参数问题 - 确认股票代码和年份格式正确")
    
    print(f"\n🛠️ 具体操作步骤:")
    print("# 1. 更新akshare")
    print("pip install akshare --upgrade")
    print()
    print("# 2. 手动测试单个API")
    print("import akshare as ak")
    print("result = ak.stock_financial_analysis_indicator(symbol='000001', start_year='2020')")
    print("print(type(result), result)")
    print()
    print("# 3. 检查网络")
    print("import requests")
    print("r = requests.get('https://push2.eastmoney.com')")
    print("print(r.status_code)")
    
    print(f"\n🚀 重构架构的优势:")
    print("✅ 即使这个API有问题，其他功能仍然正常")
    print("✅ 可以轻松切换到其他数据源")
    print("✅ 支持缓存机制减少API依赖")
    print("✅ 模拟数据可以验证业务逻辑")

def main():
    """主调试函数"""
    print("🔍 akshare API详细调试分析")
    print("="*60)
    print("专门针对财务分析指标接口的错误进行深度调试")
    print()
    
    # 1. 检查环境
    check_network_and_environment()
    
    # 2. 调试财务分析指标API（问题API）
    debug_financial_analysis_indicator()
    
    # 3. 调试其他API作为对比
    debug_individual_info_api()
    debug_stock_history_api()
    
    # 4. 提供修复建议
    suggest_fixes()
    
    print(f"\n{'='*60}")
    print("🎯 调试完成")
    print("根据上面的详细信息，您可以:")
    print("1. 确定具体是哪个步骤失败")
    print("2. 尝试相应的修复方法")
    print("3. 如果问题持续，可以暂时使用模拟数据")
    print("="*60)

if __name__ == "__main__":
    main()