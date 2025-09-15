"""
数据收集服务使用示例
"""
import sys
from pathlib import Path
import pandas as pd
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlib.data_collector.service import DataCollectionService
from quantlib.data_collector.storage import DataStorage
from quantlib.data_collector.scheduler import DataScheduler

# 设置日志
logging.basicConfig(level=logging.INFO)


def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 1. 创建数据收集服务
    service = DataCollectionService("config/data_collection.json")
    
    # 2. 执行一次数据收集
    print("执行一次数据收集...")
    results = service.collect_data_once()
    
    print(f"收集结果: 成功 {results['success_count']}, 失败 {results['failed_count']}")
    print(f"总记录数: {results['total_records']}")
    
    # 3. 查看数据摘要
    summary = service.get_data_summary()
    print(f"\n数据摘要:")
    print(f"总股票数: {summary['total_symbols']}")
    print(f"总记录数: {summary['total_records']}")


def example_storage_operations():
    """存储操作示例"""
    print("\n=== 存储操作示例 ===")
    
    # 1. 创建存储管理器
    storage = DataStorage("data/minute_data", "parquet")
    
    # 2. 获取可用股票列表
    symbols = storage.get_available_symbols()
    print(f"可用股票: {symbols}")
    
    if symbols:
        symbol = symbols[0]
        
        # 3. 获取股票的月份列表
        months = storage.get_symbol_months(symbol)
        print(f"\n{symbol} 可用月份: {months}")
        
        # 4. 加载特定月份的数据
        if months:
            year, month = months[-1]  # 最新月份
            data = storage.load_data(symbol, year, month)
            if data is not None:
                print(f"\n{symbol} {year}-{month:02d} 数据:")
                print(f"  记录数: {len(data)}")
                print(f"  列名: {list(data.columns)}")
                print(f"  时间范围: {data['date'].min()} ~ {data['date'].max()}")
        
        # 5. 加载整个股票的所有数据
        all_data = storage.load_symbol_data(symbol)
        if all_data is not None:
            print(f"\n{symbol} 全部数据:")
            print(f"  总记录数: {len(all_data)}")
            print(f"  时间范围: {all_data['date'].min()} ~ {all_data['date'].max()}")
        
        # 6. 获取数据统计信息
        info = storage.get_data_info(symbol)
        print(f"\n{symbol} 统计信息:")
        print(f"  月份数: {info['months']}")
        print(f"  记录数: {info['records']}")
        if info['date_range']:
            print(f"  日期范围: {info['date_range'][0]} ~ {info['date_range'][1]}")


def example_scheduler_usage():
    """调度器使用示例"""
    print("\n=== 调度器使用示例 ===")
    
    # 1. 创建服务和调度器
    service = DataCollectionService("config/data_collection.json")
    scheduler = DataScheduler(service)
    
    # 2. 执行一次收集
    print("手动执行一次数据收集...")
    results = scheduler.start_once()
    print(f"收集完成: {results['success_count']} 成功, {results['failed_count']} 失败")
    
    # 3. 查看调度器状态
    status = scheduler.get_status()
    print(f"\n调度器状态:")
    print(f"  运行中: {status['is_running']}")
    print(f"  计划任务数: {status['scheduled_jobs']}")
    print(f"  下次运行: {status['next_run_time']}")


def example_data_analysis():
    """数据分析示例"""
    print("\n=== 数据分析示例 ===")
    
    storage = DataStorage("data/minute_data", "parquet")
    symbols = storage.get_available_symbols()
    
    if not symbols:
        print("没有可用数据，请先执行数据收集")
        return
    
    symbol = symbols[0]
    data = storage.load_symbol_data(symbol)
    
    if data is None or data.empty:
        print(f"没有 {symbol} 的数据")
        return
    
    print(f"{symbol} 数据分析:")
    
    # 基本统计
    print(f"  数据点数: {len(data)}")
    print(f"  时间跨度: {(data['date'].max() - data['date'].min()).days} 天")
    
    # 价格统计
    if 'close' in data.columns:
        print(f"  收盘价范围: {data['close'].min():.2f} ~ {data['close'].max():.2f}")
        print(f"  平均收盘价: {data['close'].mean():.2f}")
    
    # 成交量统计
    if 'volume' in data.columns:
        print(f"  成交量范围: {data['volume'].min():,} ~ {data['volume'].max():,}")
        print(f"  平均成交量: {data['volume'].mean():,.0f}")
    
    # 按日期分组统计
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        daily_data = data.groupby(data['date'].dt.date).agg({
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        print(f"  交易日数: {len(daily_data)}")
        if len(daily_data) > 1:
            daily_returns = daily_data['close'].pct_change().dropna()
            print(f"  日收益率标准差: {daily_returns.std():.4f}")


def example_config_management():
    """配置管理示例"""
    print("\n=== 配置管理示例 ===")
    
    service = DataCollectionService("config/data_collection.json")
    
    # 查看当前配置
    config = service.config
    print("当前配置:")
    print(f"  监控股票: {config['symbols']}")
    print(f"  时间间隔: {config['intervals']}")
    print(f"  收集频率: {config['collection_frequency_hours']} 小时")
    
    # 添加新股票
    print("\n添加新股票...")
    success = service.add_symbol("600000")
    print(f"添加结果: {'成功' if success else '失败'}")
    
    # 更新配置
    print("\n更新配置...")
    success = service.update_config(collection_frequency_hours=72)  # 改为3天一次
    print(f"更新结果: {'成功' if success else '失败'}")
    
    # 查看更新后的配置
    updated_config = service.config
    print(f"新的收集频率: {updated_config['collection_frequency_hours']} 小时")


def main():
    """主函数"""
    print("数据收集服务使用示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        example_basic_usage()
        example_storage_operations()
        example_scheduler_usage()
        example_data_analysis()
        example_config_management()
        
        print("\n=" * 50)
        print("所有示例执行完成!")
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()