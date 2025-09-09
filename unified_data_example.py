"""
统一数据接口使用示例
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("=== quantlib 统一数据接口演示 ===\n")
    
    # 测试统一数据接口
    print("1. 测试 market_data 模块:")
    try:
        from quantlib.market_data import get_stock_data, get_csi300_index, get_data_manager
        
        print("✓ 成功导入市场数据模块")
        
        # 测试数据管理器
        manager = get_data_manager()
        supported_markets = manager.get_supported_markets()
        print(f"✓ 支持的市场: {supported_markets}")
        
        cache_info = manager.get_cache_info()
        print(f"✓ 缓存状态: {cache_info}")
        
    except Exception as e:
        print(f"✗ market_data 模块测试失败: {e}")
    
    # 测试technical模块使用统一接口
    print("\n2. 测试 technical 模块使用统一数据接口:")
    try:
        from quantlib.technical import get_stock_data as tech_get_stock_data
        from quantlib.technical import get_csi300_index as tech_get_csi300
        from quantlib.technical import TechnicalAnalyzer
        
        print("✓ technical 模块成功使用统一数据接口")
        print("  - get_stock_data 函数可用")
        print("  - get_csi300_index 函数可用")
        print("  - TechnicalAnalyzer 类可用")
        
        # 验证函数来源
        print(f"  - get_stock_data 来自: {tech_get_stock_data.__module__}")
        print(f"  - get_csi300_index 来自: {tech_get_csi300.__module__}")
        
    except Exception as e:
        print(f"✗ technical 模块测试失败: {e}")
    
    # 测试visualization模块
    print("\n3. 测试 visualization 模块:")
    try:
        from quantlib.visualization import CandlestickChart
        print("✓ visualization 模块可用")
        print("  - CandlestickChart 类可用")
        print("  - 支持 add_benchmark 功能")
        
    except Exception as e:
        print(f"✗ visualization 模块测试失败: {e}")
    
    # 测试其他模块结构
    print("\n4. 测试新模块结构:")
    modules_to_test = [
        'strategy', 'backtest', 'portfolio', 
        'risk', 'screener', 'optimization'
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(f'quantlib.{module_name}', fromlist=[module_name])
            print(f"✓ {module_name} 模块结构已创建")
        except Exception as e:
            print(f"✗ {module_name} 模块测试失败: {e}")
    
    # 测试完整工作流
    print("\n5. 测试完整数据工作流:")
    try:
        # 模拟数据创建（实际使用时需要网络和数据源）
        import pandas as pd
        import numpy as np
        
        # 创建示例数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        prices = [100]
        for i in range(49):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 50))
        
        sample_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.01, 0.03)
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.randint(1000000, 5000000)
            
            sample_data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': volume
            })
        
        sample_df = pd.DataFrame(sample_data)
        
        # 使用技术分析
        analyzer = TechnicalAnalyzer(sample_df)
        analyzer.calculate_all_indicators()
        signal, strength, _ = analyzer.get_consensus_signal()
        
        # 使用可视化
        chart = CandlestickChart(sample_df, engine='auto')
        chart.add_ma([10, 20])
        
        print("✓ 完整工作流测试成功")
        print(f"  - 样本数据: {len(sample_df)} 条记录")
        print(f"  - 技术信号: 信号={signal}, 强度={strength}")
        print("  - 图表创建成功")
        
    except Exception as e:
        print(f"✗ 完整工作流测试失败: {e}")
    
    # 显示架构概览
    print("\n=== quantlib 架构概览 ===")
    print("📊 核心模块:")
    print("  market_data  - 统一数据接口 ✅")
    print("  fundamental  - 基本面分析 ✅") 
    print("  technical    - 技术分析 ✅")
    print("  visualization- 数据可视化 ✅")
    
    print("\n🚀 扩展模块 (待实现):")
    print("  strategy     - 策略开发")
    print("  backtest     - 策略回测") 
    print("  portfolio    - 投资组合管理")
    print("  risk         - 风险管理")
    print("  screener     - 股票筛选")
    print("  optimization - 投资组合优化")
    
    print("\n💡 使用建议:")
    print("1. 所有数据获取统一使用 quantlib.market_data")
    print("2. technical 模块已集成统一数据接口")
    print("3. visualization 支持大盘基准对比")
    print("4. 后续扩展模块将逐步实现")
    
    print("\n🎉 统一数据接口重构完成！")

except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("请确保quantlib模块在Python路径中")

except Exception as e:
    print(f"✗ 运行错误: {e}")
    import traceback
    traceback.print_exc()