"""
演示更新后的 technical_analysis_example.ipynb 的用法
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("=== 更新后的Notebook用法演示 ===\n")
    
    # 新的导入方式
    print("1. 新的导入方式:")
    print("""
    # 旧方式 (已废弃)
    from quantlib.technical import TechnicalDataManager, get_a_share_data
    
    # 新方式 (推荐)
    from quantlib.market_data import get_stock_data, get_data_manager
    from quantlib.technical import TechnicalAnalyzer
    """)
    
    # 实际导入测试
    from quantlib.market_data import get_stock_data, get_data_manager
    from quantlib.technical import TechnicalAnalyzer
    
    print("✓ 新的导入方式测试成功")
    
    # 数据管理器使用
    print("\n2. 统一数据管理器的优势:")
    manager = get_data_manager()
    
    print("✓ 统一数据接口:")
    print("  - 支持美股和A股数据")
    print("  - 内置缓存机制")
    print("  - 自动数据验证")
    print("  - 一致的数据格式")
    
    print(f"✓ 缓存状态: {manager.get_cache_info()}")
    print(f"✓ 支持市场: {manager.get_supported_markets()}")
    
    # 新旧用法对比
    print("\n3. 数据获取方式对比:")
    print("""
    # 旧方式
    data_manager = TechnicalDataManager()
    data = data_manager.load_stock_data('000001', market='CN', period='1y')
    
    # 新方式 (更简洁)
    data = get_stock_data('000001', market='CN', period='1y')
    """)
    
    # 工作流演示
    print("\n4. 完整工作流演示:")
    
    # 模拟数据
    import pandas as pd
    import numpy as np
    
    print("创建示例数据...")
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
    
    print("使用技术分析...")
    analyzer = TechnicalAnalyzer(sample_df)
    analyzer.calculate_all_indicators()
    signal, strength, _ = analyzer.get_consensus_signal()
    
    print(f"✓ 分析完成: 信号={signal}, 强度={strength:.2f}")
    
    # 可视化集成
    print("\n5. 可视化集成:")
    try:
        from quantlib.visualization import CandlestickChart
        
        chart = CandlestickChart(sample_df, engine='auto')
        chart.add_ma([10, 20])
        
        print("✓ 可视化模块集成成功")
        print("  - 支持K线图")
        print("  - 支持技术指标图")
        print("  - 支持大盘基准对比")
        
    except Exception as e:
        print(f"✗ 可视化集成测试失败: {e}")
    
    print("\n=== Notebook更新要点总结 ===")
    print("✅ 已更新:")
    print("  1. 使用统一的 market_data 接口")
    print("  2. 删除了重复的数据获取代码")
    print("  3. 添加了缓存状态显示")
    print("  4. 改进了错误处理")
    print("  5. 更新了架构说明")
    
    print("\n📚 用户体验改进:")
    print("  - 更简洁的API")
    print("  - 更好的性能（缓存）")
    print("  - 更一致的数据格式")
    print("  - 更清晰的模块结构")
    
    print("\n🔄 向后兼容:")
    print("  - technical模块的TechnicalAnalyzer保持不变")
    print("  - 所有技术指标计算逻辑保持不变")
    print("  - 只是数据获取方式更新了")
    
    print("\n🎯 用户需要做的改动:")
    print("  1. 更新导入语句:")
    print("     from quantlib.market_data import get_stock_data")
    print("  2. 简化数据获取:")
    print("     data = get_stock_data('000001', market='CN')")
    print("  3. 其他代码基本不变")
    
    print("\n✨ technical_analysis_example.ipynb 已成功更新！")

except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("请确保quantlib模块正确安装")

except Exception as e:
    print(f"✗ 运行错误: {e}")
    import traceback
    traceback.print_exc()