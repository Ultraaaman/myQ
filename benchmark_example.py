"""
简单的大盘基准对比功能演示
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from quantlib.technical import get_stock_data, get_csi300_index
    from quantlib.visualization import CandlestickChart
    import pandas as pd
    import numpy as np
    
    print("=== 大盘基准对比功能演示 ===\n")
    
    def create_sample_stock_data():
        """创建示例个股数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 模拟个股走势（比大盘波动更大）
        base_price = 50
        prices = [base_price]
        
        for i in range(99):
            change = np.random.normal(0, 0.03)  # 3%标准差
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            daily_volatility = np.random.uniform(0.01, 0.06)
            high = price * (1 + daily_volatility)
            low = price * (1 - daily_volatility)
            
            if i == 0:
                open_price = price
            else:
                open_price = prices[i-1] * np.random.uniform(0.97, 1.03)
            
            open_price = max(min(open_price, high), low)
            close_price = max(min(price, high), low)
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def create_sample_benchmark_data():
        """创建示例大盘数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(123)
        
        # 模拟大盘走势（相对稳定）
        base_price = 3000  # 模拟指数点位
        prices = [base_price]
        
        for i in range(99):
            change = np.random.normal(0, 0.015)  # 1.5%标准差
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))
        
        data = []
        for date, price in zip(dates, prices):
            data.append({
                'date': date,
                'close': round(price, 2)
            })
        
        return pd.DataFrame(data)
    
    # 1. 创建示例数据
    print("1. 创建示例数据...")
    stock_data = create_sample_stock_data()
    benchmark_data = create_sample_benchmark_data()
    
    print(f"✓ 个股数据: {len(stock_data)} 条记录")
    print(f"  价格范围: {stock_data['close'].min():.2f} - {stock_data['close'].max():.2f}")
    print(f"✓ 大盘数据: {len(benchmark_data)} 条记录") 
    print(f"  指数范围: {benchmark_data['close'].min():.2f} - {benchmark_data['close'].max():.2f}")
    
    # 2. 基本使用 - 添加大盘基准线
    print("\n2. 基本大盘对比功能...")
    try:
        # 创建个股图表
        chart = CandlestickChart(stock_data, engine='auto')
        chart.add_ma([20, 60])  # 添加均线
        chart.add_volume()      # 添加成交量
        
        # 添加大盘基准对比
        chart.add_benchmark(benchmark_data, name="沪深300", color="gray")
        
        print("✓ 大盘基准对比图表创建成功")
        print("  - 个股K线图")
        print("  - 20日和60日移动平均线")
        print("  - 成交量")
        print("  - 沪深300基准线（标准化显示）")
        
    except Exception as e:
        print(f"✗ 图表创建失败: {e}")
    
    # 3. 实际使用示例代码
    print("\n3. 实际使用代码示例:")
    print("""
# 获取个股数据
stock_data = get_stock_data('000001', market='CN', period='1y')  # 平安银行

# 获取大盘数据
benchmark_data = get_csi300_index(period='1y')  # 沪深300

# 创建对比图表
chart = CandlestickChart(stock_data, engine='plotly')
chart.add_ma([20, 60])                          # 添加均线
chart.add_benchmark(benchmark_data, 
                   name="沪深300", 
                   color="gray")                # 添加大盘基准
chart.add_volume()                              # 添加成交量
chart.set_title("平安银行 vs 沪深300")
chart.plot().show()                             # 显示图表
""")
    
    # 4. 功能说明
    print("\n4. 功能特点:")
    print("✓ 自动标准化: 将大盘指数标准化到与个股相同起点")
    print("✓ 时间对齐: 自动匹配个股和大盘的时间范围") 
    print("✓ 简单易用: 只需一行代码 add_benchmark() 即可")
    print("✓ 视觉区分: 大盘线用虚线显示，颜色可自定义")
    print("✓ 支持多基准: 可以添加多条基准线对比")
    
    # 5. 数据要求
    print("\n5. 数据格式要求:")
    print("个股数据需要包含: date, open, high, low, close, volume")
    print("基准数据只需要包含: date, close")
    print("\n示例数据格式:")
    print("个股数据:")
    print(stock_data[['date', 'open', 'high', 'low', 'close', 'volume']].head(3).to_string(index=False))
    print("\n基准数据:")
    print(benchmark_data[['date', 'close']].head(3).to_string(index=False))
    
    print("\n=== 简化功能演示完成 ===")
    print("这个功能专门用于个股分析时的大盘对比，简单实用！")

except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("请确保quantlib模块正确安装")

except Exception as e:
    print(f"✗ 运行错误: {e}")
    import traceback
    traceback.print_exc()