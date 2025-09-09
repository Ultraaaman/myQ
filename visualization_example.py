"""
可视化模块使用示例
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from quantlib.visualization import CandlestickChart, TechnicalChart, VolumeChart
    from quantlib.technical import get_stock_data, TechnicalAnalyzer
    import pandas as pd
    import numpy as np
    
    print("✓ 成功导入可视化模块")
    
    # 创建示例数据
    def create_sample_data():
        """创建示例OHLCV数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 模拟股价走势
        price_base = 100
        prices = [price_base]
        
        for i in range(99):
            change = np.random.normal(0, 0.02)  # 2%标准差
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # 价格不能为负
        
        # 生成OHLCV数据
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            daily_volatility = np.random.uniform(0.01, 0.05)
            high = price * (1 + daily_volatility)
            low = price * (1 - daily_volatility)
            
            # 确保开盘价在合理范围内
            if i == 0:
                open_price = price
            else:
                open_price = prices[i-1] * np.random.uniform(0.98, 1.02)
            
            # 确保价格逻辑正确
            open_price = max(min(open_price, high), low)
            close_price = max(min(price, high), low)
            
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    # 创建示例数据
    sample_data = create_sample_data()
    print(f"✓ 创建示例数据: {len(sample_data)} 条记录")
    
    print("\n=== 测试K线图功能 ===")
    try:
        # 测试K线图
        candlestick_chart = CandlestickChart(sample_data, engine='auto')
        candlestick_chart.add_ma([5, 20])
        candlestick_chart.add_volume()
        candlestick_chart.set_title("示例K线图")
        print("✓ K线图创建成功")
        
        # 注意: 实际显示需要安装相应的绘图库
        # candlestick_chart.plot().show()
        
    except Exception as e:
        print(f"✗ K线图测试失败: {e}")
    
    print("\n=== 测试技术指标图功能 ===")
    try:
        # 测试技术指标图
        technical_chart = TechnicalChart(sample_data, engine='auto')
        technical_chart.add_rsi()
        technical_chart.add_macd()
        technical_chart.add_bollinger_bands()
        print("✓ 技术指标图创建成功")
        
    except Exception as e:
        print(f"✗ 技术指标图测试失败: {e}")
    
    print("\n=== 测试成交量图功能 ===")
    try:
        # 测试成交量图
        volume_chart = VolumeChart(sample_data, engine='auto')
        volume_chart.add_volume_ma([5, 20])
        volume_stats = volume_chart.get_volume_statistics()
        print(f"✓ 成交量图创建成功")
        print(f"  平均成交量: {volume_stats['average_volume']:,.0f}")
        print(f"  最大成交量: {volume_stats['max_volume']:,.0f}")
        
    except Exception as e:
        print(f"✗ 成交量图测试失败: {e}")
    
    print("\n=== 测试工具函数 ===")
    try:
        from quantlib.visualization.utils import ChartUtils
        
        # 测试价格变化计算
        price_stats = ChartUtils.calculate_price_change(sample_data)
        print(f"✓ 价格统计计算成功")
        print(f"  当前价格: {price_stats['current_price']:.2f}")
        print(f"  涨跌幅: {price_stats['change_percent']:.2f}%")
        
        # 测试周期检测
        period = ChartUtils.detect_chart_periods(sample_data)
        print(f"  检测到数据周期: {period}")
        
        # 测试支撑阻力位识别
        levels = ChartUtils.identify_support_resistance(sample_data)
        print(f"  支撑位数量: {len(levels['support_levels'])}")
        print(f"  阻力位数量: {len(levels['resistance_levels'])}")
        
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
    
    print("\n=== 测试主题功能 ===")
    try:
        from quantlib.visualization import get_theme, list_themes
        
        themes = list_themes()
        print(f"✓ 主题功能测试成功")
        print(f"  可用主题: {themes}")
        
        # 测试不同主题
        for theme_name in themes:
            theme = get_theme(theme_name)
            print(f"  {theme_name} 主题: {theme.colors['up']} (上涨色)")
            
    except Exception as e:
        print(f"✗ 主题功能测试失败: {e}")
    
    print("\n🎉 可视化模块基本功能测试完成！")
    print("\n📝 使用说明:")
    print("1. 安装绘图库: pip install matplotlib plotly mplfinance")
    print("2. 使用 chart.plot().show() 显示图表")
    print("3. 使用 chart.save('filename.png') 保存图表")
    
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("请确保quantlib模块在Python路径中")

except Exception as e:
    print(f"✗ 运行错误: {e}")
    import traceback
    traceback.print_exc()