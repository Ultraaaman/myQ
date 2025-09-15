#!/usr/bin/env python3
"""
数据分析工具 - 基于1分钟数据进行各种时间间隔的分析
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, timedelta

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlib.data_collector.storage import DataStorage
from quantlib.data_collector.resample import get_resampler
import matplotlib.pyplot as plt

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_data_with_interval(symbol: str, interval: str = "1min", 
                          start_date: str = None, end_date: str = None,
                          data_path: str = "data/minute_data") -> pd.DataFrame:
    """
    获取指定时间间隔的数据（基于1分钟数据重采样）
    
    Args:
        symbol: 股票代码
        interval: 时间间隔 (1min, 5min, 15min, 30min, 1h, 4h, 1d等)
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)  
        data_path: 数据路径
        
    Returns:
        指定间隔的OHLCV数据
    """
    storage = DataStorage(data_path, "parquet")
    
    # 转换日期
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else None
    
    if interval == "1min":
        # 直接加载1分钟数据
        return storage.load_symbol_data(symbol, start_dt, end_dt)
    else:
        # 重采样到目标间隔
        return storage.load_and_resample(symbol, interval, start_dt, end_dt)

def cmd_show_data(args):
    """显示数据概览"""
    print(f"=== {args.symbol} 数据概览 ({args.interval}) ===")
    
    data = get_data_with_interval(
        args.symbol, args.interval, 
        args.start_date, args.end_date, args.data_path
    )
    
    if data is None or data.empty:
        print("没有找到数据")
        return
    
    print(f"\n基本信息:")
    print(f"  时间间隔: {args.interval}")
    print(f"  数据点数: {len(data):,}")
    print(f"  时间范围: {data['date'].min()} ~ {data['date'].max()}")
    print(f"  交易天数: {data['date'].dt.date.nunique()}")
    
    print(f"\n价格统计:")
    print(f"  最高价: {data['high'].max():.2f}")
    print(f"  最低价: {data['low'].min():.2f}")
    print(f"  平均收盘价: {data['close'].mean():.2f}")
    print(f"  价格波动率: {data['close'].pct_change().std() * 100:.2f}%")
    
    print(f"\n成交量统计:")
    print(f"  总成交量: {data['volume'].sum():,}")
    print(f"  平均成交量: {data['volume'].mean():,.0f}")
    print(f"  最大成交量: {data['volume'].max():,}")
    
    # 显示最近几条数据
    print(f"\n最近{min(5, len(data))}条数据:")
    print(data.tail().to_string(index=False))

def cmd_compare_intervals(args):
    """比较不同时间间隔的数据"""
    print(f"=== {args.symbol} 不同时间间隔对比 ===")
    
    intervals = args.intervals.split(',')
    storage = DataStorage(args.data_path, "parquet")
    
    # 获取日期范围
    start_dt = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else None
    end_dt = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else None
    
    print(f"对比时间间隔: {intervals}")
    print(f"日期范围: {start_dt or '全部'} ~ {end_dt or '全部'}")
    
    results = {}
    
    # 获取各间隔数据
    for interval in intervals:
        try:
            if interval.strip() == "1min":
                data = storage.load_symbol_data(args.symbol, start_dt, end_dt)
            else:
                data = storage.load_and_resample(args.symbol, interval.strip(), start_dt, end_dt)
            
            if data is not None and not data.empty:
                results[interval.strip()] = {
                    'count': len(data),
                    'date_range': (data['date'].min(), data['date'].max()),
                    'price_range': (data['close'].min(), data['close'].max()),
                    'avg_volume': data['volume'].mean(),
                    'volatility': data['close'].pct_change().std() * 100
                }
            else:
                results[interval.strip()] = None
                
        except Exception as e:
            print(f"获取 {interval} 数据失败: {e}")
            results[interval.strip()] = None
    
    # 显示对比结果
    print(f"\n{'间隔':<8} {'数据点数':<12} {'价格范围':<20} {'平均成交量':<15} {'波动率%':<10}")
    print("-" * 80)
    
    for interval, info in results.items():
        if info:
            price_range = f"{info['price_range'][0]:.2f}-{info['price_range'][1]:.2f}"
            print(f"{interval:<8} {info['count']:<12,} {price_range:<20} {info['avg_volume']:<15,.0f} {info['volatility']:<10.2f}")
        else:
            print(f"{interval:<8} {'无数据':<12}")

def cmd_price_analysis(args):
    """价格走势分析"""
    print(f"=== {args.symbol} 价格走势分析 ({args.interval}) ===")
    
    data = get_data_with_interval(
        args.symbol, args.interval,
        args.start_date, args.end_date, args.data_path
    )
    
    if data is None or data.empty:
        print("没有找到数据")
        return
    
    # 计算技术指标
    data = data.copy()
    data['returns'] = data['close'].pct_change()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['volatility'] = data['returns'].rolling(20).std() * 100
    
    print(f"\n价格走势分析:")
    total_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
    print(f"  总收益率: {total_return:+.2f}%")
    print(f"  最大回撤: {calculate_max_drawdown(data['close']):.2f}%")
    print(f"  平均日波动率: {data['volatility'].mean():.2f}%")
    
    # 涨跌统计
    up_days = (data['returns'] > 0).sum()
    down_days = (data['returns'] < 0).sum()
    flat_days = (data['returns'] == 0).sum()
    
    print(f"\n涨跌统计:")
    print(f"  上涨周期: {up_days} ({up_days/len(data)*100:.1f}%)")
    print(f"  下跌周期: {down_days} ({down_days/len(data)*100:.1f}%)")
    print(f"  平盘周期: {flat_days} ({flat_days/len(data)*100:.1f}%)")
    
    # 价格分布
    print(f"\n价格分布:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        value = np.percentile(data['close'], p)
        print(f"  {p}%分位数: {value:.2f}")

def cmd_volume_analysis(args):
    """成交量分析"""
    print(f"=== {args.symbol} 成交量分析 ({args.interval}) ===")
    
    data = get_data_with_interval(
        args.symbol, args.interval,
        args.start_date, args.end_date, args.data_path
    )
    
    if data is None or data.empty:
        print("没有找到数据")
        return
    
    data = data.copy()
    data['volume_ma'] = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['price_volume'] = data['close'] * data['volume']  # 成交额
    
    print(f"\n成交量分析:")
    print(f"  平均成交量: {data['volume'].mean():,.0f}")
    print(f"  成交量标准差: {data['volume'].std():,.0f}")
    print(f"  最大成交量: {data['volume'].max():,}")
    print(f"  最小成交量: {data['volume'].min():,}")
    
    # 成交量与价格关系
    price_up = data['close'] > data['open']
    vol_up = data.loc[price_up, 'volume'].mean()
    vol_down = data.loc[~price_up, 'volume'].mean()
    
    print(f"\n量价关系:")
    print(f"  上涨时平均成交量: {vol_up:,.0f}")
    print(f"  下跌时平均成交量: {vol_down:,.0f}")
    print(f"  量价比: {vol_up/vol_down:.2f}")
    
    # 异常成交量
    high_volume = data['volume'] > data['volume'].quantile(0.9)
    print(f"\n异常成交量 (top 10%):")
    print(f"  异常成交量次数: {high_volume.sum()}")
    if high_volume.any():
        avg_return_high_vol = data.loc[high_volume, 'close'].pct_change().mean() * 100
        print(f"  异常成交量时平均收益: {avg_return_high_vol:+.2f}%")

def cmd_trading_session_analysis(args):
    """交易时段分析"""
    print(f"=== {args.symbol} 交易时段分析 ===")
    
    # 只对1分钟或5分钟数据有意义
    if args.interval not in ['1min', '5min']:
        print("交易时段分析需要使用1min或5min数据")
        return
    
    data = get_data_with_interval(
        args.symbol, args.interval,
        args.start_date, args.end_date, args.data_path
    )
    
    if data is None or data.empty:
        print("没有找到数据")
        return
    
    data = data.copy()
    data['hour'] = pd.to_datetime(data['date']).dt.hour
    data['minute'] = pd.to_datetime(data['date']).dt.minute
    data['returns'] = data['close'].pct_change()
    
    # 按小时分析
    hourly_stats = data.groupby('hour').agg({
        'volume': 'mean',
        'returns': ['mean', 'std'],
        'close': 'count'
    }).round(4)
    
    print(f"\n按小时统计 (仅显示有数据的时段):")
    print(f"{'时间':<6} {'平均成交量':<15} {'平均收益%':<12} {'收益波动%':<12} {'数据点数':<10}")
    print("-" * 70)
    
    for hour in sorted(hourly_stats.index):
        if hourly_stats.loc[hour, ('close', 'count')] > 0:
            avg_vol = hourly_stats.loc[hour, ('volume', 'mean')]
            avg_ret = hourly_stats.loc[hour, ('returns', 'mean')] * 100
            ret_std = hourly_stats.loc[hour, ('returns', 'std')] * 100
            count = hourly_stats.loc[hour, ('close', 'count')]
            
            print(f"{hour:02d}:00 {avg_vol:<15,.0f} {avg_ret:<+12.3f} {ret_std:<12.3f} {count:<10.0f}")
    
    # 开盘和收盘效应
    if args.interval == '1min':
        opening_data = data[data['hour'].isin([9]) & data['minute'].isin([30, 31, 32])]
        closing_data = data[data['hour'].isin([14, 15]) & data['minute'].isin([57, 58, 59])]
        
        if not opening_data.empty and not closing_data.empty:
            print(f"\n开盘收盘效应:")
            print(f"  开盘3分钟平均收益: {opening_data['returns'].mean()*100:+.3f}%")
            print(f"  收盘3分钟平均收益: {closing_data['returns'].mean()*100:+.3f}%")
            print(f"  开盘3分钟平均成交量: {opening_data['volume'].mean():,.0f}")
            print(f"  收盘3分钟平均成交量: {closing_data['volume'].mean():,.0f}")

def calculate_max_drawdown(prices):
    """计算最大回撤"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak * 100
    return drawdown.min()

def cmd_export_data(args):
    """导出数据到CSV"""
    print(f"=== 导出 {args.symbol} 数据 ({args.interval}) ===")
    
    data = get_data_with_interval(
        args.symbol, args.interval,
        args.start_date, args.end_date, args.data_path
    )
    
    if data is None or data.empty:
        print("没有找到数据")
        return
    
    # 生成输出文件名
    if args.output:
        output_file = args.output
    else:
        date_suffix = ""
        if args.start_date and args.end_date:
            date_suffix = f"_{args.start_date}_to_{args.end_date}"
        elif args.start_date:
            date_suffix = f"_from_{args.start_date}"
        elif args.end_date:
            date_suffix = f"_to_{args.end_date}"
        
        output_file = f"{args.symbol}_{args.interval}{date_suffix}.csv"
    
    # 导出数据
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(output_file, index=False)
    
    print(f"数据已导出到: {output_file}")
    print(f"记录数: {len(data):,}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='数据分析工具 - 基于1分钟数据的多时间间隔分析')
    parser.add_argument('--data-path', default='data/minute_data', help='数据路径')
    
    subparsers = parser.add_subparsers(dest='command', help='分析命令')
    
    # 通用参数
    def add_common_args(p):
        p.add_argument('symbol', help='股票代码')
        p.add_argument('--interval', default='1min', 
                      help='时间间隔 (1min, 5min, 15min, 30min, 1h, 4h, 1d等)')
        p.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
        p.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    
    # show 命令
    parser_show = subparsers.add_parser('show', help='显示数据概览')
    add_common_args(parser_show)
    
    # compare 命令
    parser_compare = subparsers.add_parser('compare', help='比较不同时间间隔')
    parser_compare.add_argument('symbol', help='股票代码')
    parser_compare.add_argument('--intervals', default='1min,5min,15min,1h,1d',
                               help='时间间隔列表，逗号分隔')
    parser_compare.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    parser_compare.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    
    # price 命令
    parser_price = subparsers.add_parser('price', help='价格走势分析')
    add_common_args(parser_price)
    
    # volume 命令
    parser_volume = subparsers.add_parser('volume', help='成交量分析')
    add_common_args(parser_volume)
    
    # session 命令
    parser_session = subparsers.add_parser('session', help='交易时段分析')
    parser_session.add_argument('symbol', help='股票代码')
    parser_session.add_argument('--interval', default='1min', choices=['1min', '5min'],
                                help='时间间隔 (仅支持1min和5min)')
    parser_session.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    parser_session.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    
    # export 命令
    parser_export = subparsers.add_parser('export', help='导出数据到CSV')
    add_common_args(parser_export)
    parser_export.add_argument('--output', help='输出文件名')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging()
    
    # 执行命令
    if args.command == 'show':
        cmd_show_data(args)
    elif args.command == 'compare':
        cmd_compare_intervals(args)
    elif args.command == 'price':
        cmd_price_analysis(args)
    elif args.command == 'volume':
        cmd_volume_analysis(args)
    elif args.command == 'session':
        cmd_trading_session_analysis(args)
    elif args.command == 'export':
        cmd_export_data(args)

if __name__ == '__main__':
    main()