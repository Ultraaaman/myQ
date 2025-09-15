#!/usr/bin/env python3
"""
数据收集守护进程
使用方式:
    python scripts/data_collection_daemon.py --help
    python scripts/data_collection_daemon.py once
    python scripts/data_collection_daemon.py start --schedule weekly --time "02:00" --day sunday
    python scripts/data_collection_daemon.py continuous --interval 168
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlib.data_collector.service import DataCollectionService
from quantlib.data_collector.scheduler import DataScheduler, create_and_start_service, run_once


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'data_collection.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def cmd_once(args):
    """执行一次数据收集"""
    print("=== 执行一次数据收集 ===")
    setup_logging(args.log_level)
    
    results = run_once(args.config)
    
    print(f"\n收集结果:")
    print(f"  开始时间: {results['start_time']}")
    print(f"  结束时间: {results['end_time']}")
    print(f"  耗时: {results['duration']:.1f} 秒")
    print(f"  成功: {results['success_count']}")
    print(f"  失败: {results['failed_count']}")
    print(f"  总记录数: {results['total_records']}")
    
    if results['errors']:
        print(f"\n错误:")
        for error in results['errors']:
            print(f"  - {error}")


def cmd_start(args):
    """启动定时调度服务"""
    print(f"=== 启动定时调度服务 ===")
    print(f"调度类型: {args.schedule}")
    print(f"执行时间: {args.time}")
    if args.schedule == 'weekly':
        print(f"星期几: {args.day}")
    
    setup_logging(args.log_level)
    
    scheduler = create_and_start_service(
        config_path=args.config,
        schedule_type=args.schedule,
        schedule_time=args.time,
        weekday=args.day
    )
    
    try:
        print("\n服务已启动，按 Ctrl+C 停止...")
        scheduler.wait_for_completion()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        scheduler.stop()
        print("服务已停止")


def cmd_continuous(args):
    """启动连续运行模式"""
    print(f"=== 启动连续运行模式 ===")
    print(f"收集间隔: {args.interval} 小时")
    
    setup_logging(args.log_level)
    
    scheduler = create_and_start_service(
        config_path=args.config,
        schedule_type="continuous",
        schedule_time=str(args.interval)
    )
    
    try:
        print("\n服务已启动，按 Ctrl+C 停止...")
        scheduler.wait_for_completion()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        scheduler.stop()
        print("服务已停止")


def cmd_status(args):
    """查看服务状态和数据摘要"""
    print("=== 数据收集服务状态 ===")
    setup_logging("ERROR")  # 只显示错误日志
    
    service = DataCollectionService(args.config)
    
    # 显示配置信息
    config = service.config
    print(f"\n配置信息:")
    print(f"  监控股票: {', '.join(config.get('symbols', []))}")
    print(f"  时间间隔: {', '.join(config.get('intervals', []))}")
    print(f"  收集频率: {config.get('collection_frequency_hours', 168)} 小时")
    print(f"  存储路径: {config.get('storage_path', 'data/minute_data')}")
    print(f"  文件格式: {config.get('file_format', 'parquet')}")
    print(f"  保留月数: {config.get('keep_months', 12)}")
    
    # 显示数据摘要
    summary = service.get_data_summary()
    print(f"\n数据摘要:")
    print(f"  总股票数: {summary.get('total_symbols', 0)}")
    print(f"  总记录数: {summary.get('total_records', 0)}")
    print(f"  总月份数: {summary.get('total_months', 0)}")
    
    # 显示各股票详情
    if 'symbols' in summary:
        print(f"\n各股票详情:")
        for symbol, info in summary['symbols'].items():
            date_range = info.get('date_range')
            if date_range:
                date_str = f"{date_range[0].strftime('%Y-%m-%d')} ~ {date_range[1].strftime('%Y-%m-%d')}"
            else:
                date_str = "无数据"
            print(f"  {symbol}: {info['records']} 条记录, {info['months']} 个月, {date_str}")


def cmd_add_symbol(args):
    """添加新股票到监控列表"""
    print(f"=== 添加股票 {args.symbol} ===")
    setup_logging("ERROR")
    
    service = DataCollectionService(args.config)
    success = service.add_symbol(args.symbol)
    
    if success:
        print(f"✓ 成功添加股票: {args.symbol}")
    else:
        print(f"× 添加股票失败: {args.symbol}")


def cmd_remove_symbol(args):
    """从监控列表移除股票"""
    print(f"=== 移除股票 {args.symbol} ===")
    setup_logging("ERROR")
    
    service = DataCollectionService(args.config)
    success = service.remove_symbol(args.symbol)
    
    if success:
        print(f"✓ 成功移除股票: {args.symbol}")
    else:
        print(f"× 移除股票失败: {args.symbol}")


def cmd_resample(args):
    """数据重采样命令"""
    print(f"=== 重采样 {args.symbol} 到 {args.interval} ===")
    setup_logging(args.log_level)
    
    from quantlib.data_collector.storage import DataStorage
    
    storage = DataStorage(args.data_path, "parquet")
    
    if args.batch:
        # 批量重采样
        intervals = args.interval.split(',')
        print(f"批量重采样到: {intervals}")
        
        results = storage.batch_resample_symbol(
            args.symbol, 
            intervals,
            save_results=args.save
        )
        
        print(f"\n重采样结果:")
        for interval, data in results.items():
            print(f"  {interval}: {len(data)} 条记录")
    else:
        # 单个间隔重采样
        data = storage.load_and_resample(args.symbol, args.interval)
        if data is not None:
            print(f"重采样完成: {len(data)} 条记录")
            
            if args.save:
                # 保存重采样结果
                interval_storage = DataStorage(
                    f"{args.data_path}_resampled_{args.interval}",
                    "parquet"
                )
                success = interval_storage.save_data(args.symbol, data)
                print(f"保存结果: {'成功' if success else '失败'}")
        else:
            print(f"重采样失败: 没有找到 {args.symbol} 的数据")


def cmd_resample_summary(args):
    """重采样摘要命令"""
    print(f"=== {args.symbol} 重采样摘要 ===")
    setup_logging("ERROR")
    
    from quantlib.data_collector.storage import DataStorage
    
    storage = DataStorage(args.data_path, "parquet")
    intervals = args.intervals.split(',')
    
    summary = storage.get_resample_summary(args.symbol, intervals)
    
    if 'error' in summary:
        print(f"错误: {summary['error']}")
        return
    
    print(f"原始数据: {summary['original_records']} 条记录")
    if summary['original_date_range']:
        start, end = summary['original_date_range']
        print(f"时间范围: {start} ~ {end}")
    
    print(f"\n重采样结果:")
    for interval, info in summary['intervals'].items():
        if 'error' in info:
            print(f"  {interval}: 错误 - {info['error']}")
        else:
            ratio = info['compression_ratio']
            print(f"  {interval}: {info['records']} 条记录 (压缩比: {ratio:.3f})")


def main():
    parser = argparse.ArgumentParser(description='数据收集守护进程')
    parser.add_argument('--config', '-c', default='config/data_collection.json',
                       help='配置文件路径 (默认: config/data_collection.json)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # once 命令
    parser_once = subparsers.add_parser('once', help='执行一次数据收集')
    parser_once.set_defaults(func=cmd_once)
    
    # start 命令
    parser_start = subparsers.add_parser('start', help='启动定时调度服务')
    parser_start.add_argument('--schedule', choices=['daily', 'weekly', 'hourly'],
                             default='weekly', help='调度类型 (默认: weekly)')
    parser_start.add_argument('--time', default='02:00',
                             help='执行时间 (daily/weekly: HH:MM, hourly: 小时间隔) (默认: 02:00)')
    parser_start.add_argument('--day', default='sunday',
                             choices=['monday', 'tuesday', 'wednesday', 'thursday',
                                    'friday', 'saturday', 'sunday'],
                             help='星期几 (仅weekly模式) (默认: sunday)')
    parser_start.set_defaults(func=cmd_start)
    
    # continuous 命令
    parser_continuous = subparsers.add_parser('continuous', help='启动连续运行模式')
    parser_continuous.add_argument('--interval', type=int, default=168,
                                  help='收集间隔 (小时) (默认: 168 = 1周)')
    parser_continuous.set_defaults(func=cmd_continuous)
    
    # status 命令
    parser_status = subparsers.add_parser('status', help='查看服务状态和数据摘要')
    parser_status.set_defaults(func=cmd_status)
    
    # add-symbol 命令
    parser_add = subparsers.add_parser('add-symbol', help='添加新股票到监控列表')
    parser_add.add_argument('symbol', help='股票代码')
    parser_add.set_defaults(func=cmd_add_symbol)
    
    # remove-symbol 命令
    parser_remove = subparsers.add_parser('remove-symbol', help='从监控列表移除股票')
    parser_remove.add_argument('symbol', help='股票代码')
    parser_remove.set_defaults(func=cmd_remove_symbol)
    
    # resample 命令
    parser_resample = subparsers.add_parser('resample', help='重采样数据到不同时间间隔')
    parser_resample.add_argument('symbol', help='股票代码')
    parser_resample.add_argument('interval', help='目标时间间隔 (如: 5min, 1h, 1d) 或多个间隔用逗号分隔')
    parser_resample.add_argument('--data-path', default='data/minute_data', 
                               help='数据存储路径 (默认: data/minute_data)')
    parser_resample.add_argument('--batch', action='store_true',
                               help='批量重采样模式（interval参数用逗号分隔多个间隔）')
    parser_resample.add_argument('--save', action='store_true',
                               help='保存重采样结果到文件')
    parser_resample.set_defaults(func=cmd_resample)
    
    # resample-summary 命令
    parser_summary = subparsers.add_parser('resample-summary', help='查看重采样摘要')
    parser_summary.add_argument('symbol', help='股票代码')
    parser_summary.add_argument('--intervals', default='5min,15min,1h,1d',
                              help='时间间隔列表，用逗号分隔 (默认: 5min,15min,1h,1d)')
    parser_summary.add_argument('--data-path', default='data/minute_data',
                              help='数据存储路径 (默认: data/minute_data)')
    parser_summary.set_defaults(func=cmd_resample_summary)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 确保配置目录存在
    config_path = Path(args.config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 执行命令
    args.func(args)


if __name__ == '__main__':
    main()