"""
数据重采样功能使用示例
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlib.data_collector.resample import DataResampler, resample_data, resample_ohlcv, batch_resample
from quantlib.data_collector.storage import DataStorage

# 设置日志
logging.basicConfig(level=logging.INFO)


def create_sample_data():
    """创建示例OHLCV数据"""
    print("=== 创建示例数据 ===")
    
    # 生成1小时的分钟级数据
    start_time = datetime(2024, 3, 1, 9, 0)  # 9:00开始
    dates = [start_time + timedelta(minutes=i) for i in range(60)]
    
    # 模拟股价走势
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    for _ in range(59):
        change = np.random.normal(0, 0.5)  # 随机价格变动
        new_price = max(prices[-1] + change, 1.0)  # 价格不能为负
        prices.append(new_price)
    
    # 生成OHLCV数据
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # 模拟开高低价
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) + np.random.uniform(0, 1)
        low = min(open_price, close) - np.random.uniform(0, 1)
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"生成示例数据: {len(df)} 条1分钟K线记录")
    print(f"时间范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    
    return df


def example_basic_resample():
    """基础重采样示例"""
    print("\n=== 基础重采样示例 ===")
    
    # 创建示例数据
    data = create_sample_data()
    
    # 初始化重采样器
    resampler = DataResampler()
    
    # 重采样到5分钟
    print("\n1. 重采样到5分钟K线:")
    data_5min = resampler.resample_ohlcv(data, '5min')
    print(f"原始数据: {len(data)} 条")
    print(f"5分钟数据: {len(data_5min)} 条")
    print("前3条5分钟数据:")
    print(data_5min.head(3).to_string())
    
    # 重采样到15分钟
    print("\n2. 重采样到15分钟K线:")
    data_15min = resampler.resample_ohlcv(data, '15min')
    print(f"15分钟数据: {len(data_15min)} 条")
    print("15分钟数据:")
    print(data_15min.to_string())


def example_custom_aggregation():
    """自定义聚合规则示例"""
    print("\n=== 自定义聚合规则示例 ===")
    
    # 创建包含更多列的数据
    data = create_sample_data()
    data['amount'] = data['close'] * data['volume']  # 成交额
    data['trades'] = np.random.randint(10, 100, len(data))  # 成交笔数
    
    resampler = DataResampler()
    
    # 自定义聚合规则
    custom_agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'amount': 'sum',
        'trades': 'sum'
    }
    
    print("重采样到10分钟，使用自定义聚合:")
    data_10min = resampler.resample(
        data=data,
        target_interval='10min',
        custom_agg=custom_agg
    )
    
    print(f"原始数据: {len(data)} 条")
    print(f"10分钟数据: {len(data_10min)} 条")
    print("10分钟数据:")
    print(data_10min.to_string())


def example_batch_resample():
    """批量重采样示例"""
    print("\n=== 批量重采样示例 ===")
    
    data = create_sample_data()
    resampler = DataResampler()
    
    # 批量重采样到多个时间间隔
    intervals = ['5min', '10min', '15min', '30min']
    
    print(f"批量重采样到: {intervals}")
    results = resampler.batch_resample(data, intervals)
    
    print("\n批量重采样结果:")
    for interval, resampled_data in results.items():
        print(f"  {interval}: {len(resampled_data)} 条记录")
        
    # 显示30分钟数据详情
    if '30min' in results:
        print(f"\n30分钟K线数据:")
        print(results['30min'].to_string())


def example_upsample():
    """上采样示例"""
    print("\n=== 上采样示例 ===")
    
    # 创建10分钟数据
    data_10min = create_sample_data()
    data_10min = data_10min[::10]  # 每10条取1条，模拟10分钟数据
    data_10min.reset_index(drop=True, inplace=True)
    
    print(f"10分钟数据: {len(data_10min)} 条")
    print("原始10分钟数据:")
    print(data_10min.to_string())
    
    resampler = DataResampler()
    
    # 上采样到5分钟（向前填充）
    print(f"\n上采样到5分钟:")
    data_5min = resampler.upsample(data_10min, '5min', method='ffill')
    print(f"上采样后: {len(data_5min)} 条记录")
    print("上采样结果（前10条）:")
    print(data_5min.head(10).to_string())


def example_storage_integration():
    """与存储系统集成示例"""
    print("\n=== 存储系统集成示例 ===")
    
    # 创建存储管理器
    storage = DataStorage("data/test_resample", "parquet")
    
    # 创建并保存原始1分钟数据
    data_1min = create_sample_data()
    symbol = "TEST001"
    
    print(f"保存原始1分钟数据: {symbol}")
    success = storage.save_data(symbol, data_1min, append_mode=False)
    print(f"保存结果: {'成功' if success else '失败'}")
    
    if success:
        # 使用存储系统的重采样功能
        print(f"\n通过存储系统重采样:")
        
        # 重采样到5分钟
        data_5min = storage.load_and_resample(symbol, '5min')
        if data_5min is not None:
            print(f"5分钟数据: {len(data_5min)} 条")
        
        # 批量重采样
        intervals = ['5min', '15min', '30min']
        batch_results = storage.batch_resample_symbol(
            symbol, intervals, save_results=True
        )
        
        print(f"\n批量重采样结果:")
        for interval, data in batch_results.items():
            print(f"  {interval}: {len(data)} 条记录")
        
        # 获取重采样摘要
        summary = storage.get_resample_summary(symbol, intervals)
        print(f"\n重采样摘要:")
        print(f"  原始记录数: {summary['original_records']}")
        for interval, info in summary['intervals'].items():
            if 'records' in info:
                ratio = info['compression_ratio']
                print(f"  {interval}: {info['records']} 条 (压缩比: {ratio:.3f})")


def example_supported_intervals():
    """支持的时间间隔示例"""
    print("\n=== 支持的时间间隔 ===")
    
    resampler = DataResampler()
    
    # 获取所有支持的间隔
    intervals = resampler.get_supported_intervals()
    print(f"支持 {len(intervals)} 种时间间隔:")
    
    # 分类显示
    categories = {
        '分钟级': [i for i in intervals if 'min' in i or 'm' in i and 'month' not in i],
        '小时级': [i for i in intervals if 'h' in i or 'hour' in i],
        '日级': [i for i in intervals if 'd' in i or 'day' in i],
        '周级': [i for i in intervals if 'w' in i or 'week' in i],
        '月级': [i for i in intervals if 'M' in i or 'month' in i or '月' in i],
        '年级': [i for i in intervals if 'Y' in i or 'year' in i or '年' in i]
    }
    
    for category, cat_intervals in categories.items():
        if cat_intervals:
            print(f"\n{category}: {', '.join(cat_intervals)}")
    
    # 显示几个间隔的详细信息
    test_intervals = ['1min', '5min', '1h', '1d', '1w']
    print(f"\n详细信息:")
    for interval in test_intervals:
        info = resampler.get_interval_info(interval)
        if 'error' not in info:
            print(f"  {interval}: {info['description']}, {info['seconds']}秒")


def example_convenience_functions():
    """便捷函数使用示例"""
    print("\n=== 便捷函数示例 ===")
    
    data = create_sample_data()
    
    # 使用便捷函数
    print("1. 使用 resample_data 便捷函数:")
    data_5min = resample_data(data, '5min')
    print(f"重采样结果: {len(data_5min)} 条记录")
    
    print("\n2. 使用 resample_ohlcv 便捷函数:")
    data_15min = resample_ohlcv(data, '15min')
    print(f"OHLCV重采样结果: {len(data_15min)} 条记录")
    
    print("\n3. 使用 batch_resample 便捷函数:")
    batch_results = batch_resample(data, ['5min', '15min'])
    for interval, result_data in batch_results.items():
        print(f"  {interval}: {len(result_data)} 条记录")


def main():
    """主函数"""
    print("数据重采样功能使用示例")
    print("=" * 60)
    
    try:
        # 运行各种示例
        example_basic_resample()
        example_custom_aggregation()
        example_batch_resample()
        example_upsample()
        example_storage_integration()
        example_supported_intervals()
        example_convenience_functions()
        
        print("\n" + "=" * 60)
        print("所有示例执行完成!")
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()