#!/usr/bin/env python3
"""
测试NaN值覆盖逻辑
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlib.data_collector.storage import DataStorage

def test_nan_overwrite():
    """测试NaN值覆盖功能"""
    print("=== 测试NaN值覆盖逻辑 ===")
    
    # 创建测试存储管理器
    test_storage = DataStorage("data/test_nan_overwrite", "parquet")
    symbol = "TEST001"
    
    # 创建第一天的数据（含NaN）
    dates = [datetime(2024, 3, 1, 9, 30) + timedelta(minutes=i) for i in range(5)]
    
    print("\n1. 第一天数据（部分NaN）:")
    day1_data = pd.DataFrame({
        'date': dates,
        'open': [100.0, np.nan, 102.0, np.nan, 104.0],
        'high': [101.0, np.nan, 103.0, np.nan, 105.0],
        'low': [99.0, np.nan, 101.0, np.nan, 103.0],
        'close': [100.5, np.nan, 102.5, np.nan, 104.5],
        'volume': [1000, 0, 1200, 0, 1400]
    })
    
    print(day1_data.to_string(index=False))
    
    # 保存第一天数据
    success1 = test_storage.save_data(symbol, day1_data, append_mode=True)
    print(f"\n第一天保存结果: {'成功' if success1 else '失败'}")
    
    # 创建第二天的数据（修正NaN）
    print("\n2. 第二天数据（修正NaN值）:")
    day2_data = pd.DataFrame({
        'date': dates,  # 相同的时间点
        'open': [100.0, 101.5, 102.0, 103.5, 104.0],  # 填充了NaN
        'high': [101.0, 102.0, 103.0, 104.0, 105.0],  # 填充了NaN
        'low': [99.0, 100.5, 101.0, 102.5, 103.0],    # 填充了NaN
        'close': [100.5, 101.8, 102.5, 103.8, 104.5], # 填充了NaN
        'volume': [1000, 1100, 1200, 1300, 1400]       # 填充了0
    })
    
    print(day2_data.to_string(index=False))
    
    # 保存第二天数据（应该覆盖）
    success2 = test_storage.save_data(symbol, day2_data, append_mode=True)
    print(f"\n第二天保存结果: {'成功' if success2 else '失败'}")
    
    # 读取最终结果
    print("\n3. 最终存储的数据:")
    final_data = test_storage.load_symbol_data(symbol)
    if final_data is not None:
        print(final_data.to_string(index=False))
        
        # 检查是否还有NaN
        nan_count = final_data.isnull().sum().sum()
        print(f"\n最终数据中的NaN数量: {nan_count}")
        
        if nan_count == 0:
            print("✅ 测试通过：NaN值已被正确覆盖！")
        else:
            print("❌ 测试失败：仍有NaN值存在")
            print("NaN分布:")
            print(final_data.isnull().sum())
    else:
        print("❌ 无法读取数据")
    
    # 清理测试数据
    test_dir = Path("data/test_nan_overwrite")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print(f"\n测试数据已清理: {test_dir}")

def test_partial_update():
    """测试部分时间点更新"""
    print("\n=== 测试部分时间点更新 ===")
    
    test_storage = DataStorage("data/test_partial_update", "parquet")
    symbol = "TEST002"
    
    # 第一次：保存一整天的数据（部分有NaN）
    base_time = datetime(2024, 3, 1, 9, 30)
    dates1 = [base_time + timedelta(minutes=i) for i in range(10)]
    
    print("\n1. 初始数据（第1-10分钟，部分NaN）:")
    initial_data = pd.DataFrame({
        'date': dates1,
        'open': [100 + i if i % 3 != 1 else np.nan for i in range(10)],
        'high': [101 + i if i % 3 != 1 else np.nan for i in range(10)],
        'low': [99 + i if i % 3 != 1 else np.nan for i in range(10)],
        'close': [100.5 + i if i % 3 != 1 else np.nan for i in range(10)],
        'volume': [1000 + i*100 if i % 3 != 1 else 0 for i in range(10)]
    })
    
    print(initial_data.to_string(index=False))
    test_storage.save_data(symbol, initial_data, append_mode=False)
    
    # 第二次：只更新部分时间点（修正NaN）
    dates2 = [base_time + timedelta(minutes=i) for i in [1, 4, 7]]  # 只更新几个时间点
    
    print(f"\n2. 部分更新数据（第2, 5, 8分钟）:")
    update_data = pd.DataFrame({
        'date': dates2,
        'open': [101.5, 104.5, 107.5],     # 修正的值
        'high': [102.0, 105.0, 108.0],     # 修正的值
        'low': [100.5, 103.5, 106.5],      # 修正的值
        'close': [101.8, 104.8, 107.8],    # 修正的值
        'volume': [1150, 1450, 1750]       # 修正的值
    })
    
    print(update_data.to_string(index=False))
    test_storage.save_data(symbol, update_data, append_mode=True)
    
    # 检查最终结果
    print("\n3. 最终合并结果:")
    final_data = test_storage.load_symbol_data(symbol)
    if final_data is not None:
        print(final_data.to_string(index=False))
        
        # 检查特定时间点是否被正确更新
        updated_points = final_data[final_data['date'].isin(dates2)]
        print(f"\n被更新的时间点:")
        print(updated_points[['date', 'close']].to_string(index=False))
        
        # 检查NaN情况
        nan_count = final_data.isnull().sum().sum()
        print(f"\n剩余NaN数量: {nan_count}")
        
        if nan_count < initial_data.isnull().sum().sum():
            print("✅ 部分更新成功：NaN数量减少了！")
        else:
            print("❌ 部分更新可能有问题")
    
    # 清理
    test_dir = Path("data/test_partial_update")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print(f"\n测试数据已清理: {test_dir}")

def simulate_real_scenario():
    """模拟真实场景：数据源逐步完善"""
    print("\n=== 模拟真实数据更新场景 ===")
    
    test_storage = DataStorage("data/test_real_scenario", "parquet")
    symbol = "000001"
    
    base_time = datetime(2024, 3, 1, 9, 30)
    
    scenarios = [
        {
            'name': '第1次收集（数据源延迟）',
            'data_quality': 0.7,  # 30%的数据为NaN
            'description': '部分数据源还未更新'
        },
        {
            'name': '第2次收集（数据逐步完善）', 
            'data_quality': 0.9,  # 10%的数据为NaN
            'description': '大部分数据源已更新'
        },
        {
            'name': '第3次收集（数据完整）',
            'data_quality': 1.0,  # 0%的数据为NaN
            'description': '所有数据源已更新'
        }
    ]
    
    # 生成30分钟的数据
    dates = [base_time + timedelta(minutes=i) for i in range(30)]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n{i+1}. {scenario['name']} ({scenario['description']})")
        
        # 生成数据，根据质量随机设置NaN
        np.random.seed(42 + i)  # 确保可重现
        quality = scenario['data_quality']
        
        # 创建基础数据
        base_price = 100
        data = pd.DataFrame({
            'date': dates,
            'open': [base_price + j + np.random.normal(0, 0.5) for j in range(30)],
            'high': [base_price + j + 1 + np.random.normal(0, 0.5) for j in range(30)],
            'low': [base_price + j - 1 + np.random.normal(0, 0.5) for j in range(30)],
            'close': [base_price + j + 0.5 + np.random.normal(0, 0.5) for j in range(30)],
            'volume': [1000 + j*50 + np.random.randint(-200, 200) for j in range(30)]
        })
        
        # 根据数据质量随机设置NaN
        mask = np.random.random(30) > quality
        for col in ['open', 'high', 'low', 'close']:
            data.loc[mask, col] = np.nan
        data.loc[mask, 'volume'] = 0
        
        # 保存数据
        success = test_storage.save_data(symbol, data, append_mode=True)
        
        # 统计NaN
        nan_count = data.isnull().sum().sum()
        total_values = len(data) * 5  # 5个数值列
        nan_rate = nan_count / total_values * 100
        
        print(f"  保存结果: {'成功' if success else '失败'}")
        print(f"  NaN比例: {nan_rate:.1f}% ({nan_count}/{total_values})")
        
        # 检查当前存储的数据
        current_data = test_storage.load_symbol_data(symbol)
        if current_data is not None:
            stored_nan = current_data.isnull().sum().sum()
            stored_total = len(current_data) * 5
            stored_rate = stored_nan / stored_total * 100
            print(f"  存储中NaN比例: {stored_rate:.1f}% ({stored_nan}/{stored_total})")
    
    # 最终检查
    print(f"\n=== 最终结果 ===")
    final_data = test_storage.load_symbol_data(symbol)
    if final_data is not None:
        final_nan = final_data.isnull().sum().sum()
        final_total = len(final_data) * 5
        final_rate = final_nan / final_total * 100
        
        print(f"最终数据质量:")
        print(f"  总记录数: {len(final_data)}")
        print(f"  NaN比例: {final_rate:.1f}% ({final_nan}/{final_total})")
        print(f"  完整数据比例: {100-final_rate:.1f}%")
        
        if final_rate < 5:  # 少于5%的NaN认为是成功的
            print("✅ 模拟成功：数据质量逐步提升，NaN被正确覆盖！")
        else:
            print("⚠️  仍有较多NaN，可能需要更多轮收集")
    
    # 清理
    test_dir = Path("data/test_real_scenario")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print(f"\n测试数据已清理: {test_dir}")

def main():
    print("NaN值覆盖逻辑测试")
    print("=" * 50)
    
    # 运行各种测试
    test_nan_overwrite()
    test_partial_update()
    simulate_real_scenario()
    
    print("\n" + "=" * 50)
    print("测试完成！")

if __name__ == '__main__':
    main()