#!/usr/bin/env python3
"""
数据格式转换工具 - 在parquet和csv之间转换
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlib.data_collector.storage import DataStorage

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def convert_storage_format(source_path: str, target_path: str, 
                          source_format: str, target_format: str):
    """转换整个存储目录的格式"""
    print(f"=== 转换存储格式 {source_format} -> {target_format} ===")
    print(f"源路径: {source_path}")
    print(f"目标路径: {target_path}")
    
    # 创建存储管理器
    source_storage = DataStorage(source_path, source_format)
    target_storage = DataStorage(target_path, target_format)
    
    # 获取所有股票
    symbols = source_storage.get_available_symbols()
    if not symbols:
        print("源路径中没有找到数据")
        return
    
    print(f"找到 {len(symbols)} 个股票: {symbols}")
    
    total_converted = 0
    total_records = 0
    
    for symbol in symbols:
        print(f"\n转换股票: {symbol}")
        
        # 获取该股票的所有月份
        months = source_storage.get_symbol_months(symbol)
        print(f"  找到 {len(months)} 个月份的数据")
        
        for year, month in months:
            try:
                # 读取源数据
                data = source_storage.load_data(symbol, year, month)
                if data is not None and not data.empty:
                    # 保存为目标格式
                    success = target_storage.save_data(symbol, data, append_mode=False)
                    if success:
                        print(f"  ✓ {year}-{month:02d}: {len(data)} 条记录")
                        total_converted += 1
                        total_records += len(data)
                    else:
                        print(f"  × {year}-{month:02d}: 转换失败")
                        
            except Exception as e:
                print(f"  × {year}-{month:02d}: 错误 - {e}")
    
    print(f"\n=== 转换完成 ===")
    print(f"成功转换: {total_converted} 个文件")
    print(f"总记录数: {total_records:,} 条")

def convert_single_file(input_file: str, output_file: str):
    """转换单个文件格式"""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    print(f"=== 转换单个文件 ===")
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    
    # 判断输入格式
    input_format = input_path.suffix.lower()
    output_format = output_path.suffix.lower()
    
    try:
        # 读取数据
        if input_format == '.parquet':
            data = pd.read_parquet(input_file)
        elif input_format == '.csv':
            data = pd.read_csv(input_file, parse_dates=['date'] if 'date' in pd.read_csv(input_file, nrows=0).columns else False)
        else:
            raise ValueError(f"不支持的输入格式: {input_format}")
        
        print(f"读取数据: {len(data)} 条记录, {len(data.columns)} 列")
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        if output_format == '.parquet':
            data.to_parquet(output_file, index=False)
        elif output_format == '.csv':
            data.to_csv(output_file, index=False)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
        
        # 比较文件大小
        input_size = input_path.stat().st_size
        output_size = output_path.stat().st_size
        
        print(f"转换完成!")
        print(f"输入文件大小: {input_size / 1024 / 1024:.2f} MB")
        print(f"输出文件大小: {output_size / 1024 / 1024:.2f} MB")
        print(f"大小变化: {((output_size - input_size) / input_size * 100):+.1f}%")
        
    except Exception as e:
        print(f"转换失败: {e}")

def compare_formats(data_path: str):
    """比较不同格式的性能"""
    print("=== 格式性能比较 ===")
    
    storage = DataStorage(data_path, "parquet")
    symbols = storage.get_available_symbols()
    
    if not symbols:
        print("没有找到测试数据")
        return
    
    # 选择第一个有数据的股票
    symbol = symbols[0]
    months = storage.get_symbol_months(symbol)
    if not months:
        print(f"股票 {symbol} 没有数据")
        return
    
    year, month = months[0]
    
    # 加载测试数据
    data = storage.load_data(symbol, year, month)
    if data is None:
        print("无法加载测试数据")
        return
    
    print(f"测试数据: {symbol} {year}-{month:02d}, {len(data)} 条记录")
    
    # 创建临时文件进行测试
    temp_dir = Path("temp_format_test")
    temp_dir.mkdir(exist_ok=True)
    
    import time
    
    try:
        # 测试parquet
        parquet_file = temp_dir / "test.parquet"
        
        start_time = time.time()
        data.to_parquet(parquet_file, index=False)
        parquet_write_time = time.time() - start_time
        
        start_time = time.time()
        data_p = pd.read_parquet(parquet_file)
        parquet_read_time = time.time() - start_time
        
        parquet_size = parquet_file.stat().st_size
        
        # 测试csv
        csv_file = temp_dir / "test.csv"
        
        start_time = time.time()
        data.to_csv(csv_file, index=False)
        csv_write_time = time.time() - start_time
        
        start_time = time.time()
        data_c = pd.read_csv(csv_file, parse_dates=['date'] if 'date' in data.columns else False)
        csv_read_time = time.time() - start_time
        
        csv_size = csv_file.stat().st_size
        
        # 显示结果
        print(f"\n性能对比结果:")
        print(f"{'格式':<8} {'文件大小':<12} {'写入时间':<12} {'读取时间':<12}")
        print(f"{'-'*50}")
        print(f"{'Parquet':<8} {parquet_size/1024/1024:>8.2f} MB {parquet_write_time:>8.3f}s {parquet_read_time:>8.3f}s")
        print(f"{'CSV':<8} {csv_size/1024/1024:>8.2f} MB {csv_write_time:>8.3f}s {csv_read_time:>8.3f}s")
        
        print(f"\nParquet优势:")
        print(f"  文件大小减少: {(1 - parquet_size/csv_size)*100:.1f}%")
        print(f"  写入速度提升: {csv_write_time/parquet_write_time:.1f}倍")
        print(f"  读取速度提升: {csv_read_time/parquet_read_time:.1f}倍")
        
    finally:
        # 清理临时文件
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def analyze_storage(data_path: str, format_type: str):
    """分析存储使用情况"""
    print(f"=== 存储分析 ({format_type}) ===")
    
    storage = DataStorage(data_path, format_type)
    symbols = storage.get_available_symbols()
    
    if not symbols:
        print("没有找到数据")
        return
    
    total_files = 0
    total_size = 0
    total_records = 0
    
    print(f"{'股票':<8} {'文件数':<8} {'总大小':<12} {'记录数':<12} {'平均大小':<12}")
    print("-" * 70)
    
    for symbol in symbols:
        symbol_path = Path(data_path) / symbol
        if not symbol_path.exists():
            continue
            
        files = list(symbol_path.glob(f"*.{format_type}"))
        if not files:
            continue
            
        symbol_size = sum(f.stat().st_size for f in files)
        symbol_info = storage.get_data_info(symbol)
        symbol_records = symbol_info['records']
        
        avg_size = symbol_size / len(files) if files else 0
        
        print(f"{symbol:<8} {len(files):<8} {symbol_size/1024/1024:>8.2f} MB {symbol_records:<12,} {avg_size/1024/1024:>8.2f} MB")
        
        total_files += len(files)
        total_size += symbol_size
        total_records += symbol_records
    
    print("-" * 70)
    print(f"{'总计':<8} {total_files:<8} {total_size/1024/1024:>8.2f} MB {total_records:<12,} {total_size/total_files/1024/1024:>8.2f} MB")
    
    # 分析每条记录的平均存储大小
    if total_records > 0:
        bytes_per_record = total_size / total_records
        print(f"\n存储效率:")
        print(f"  每条记录平均大小: {bytes_per_record:.2f} 字节")
        print(f"  每万条记录大小: {bytes_per_record * 10000 / 1024 / 1024:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='数据格式转换工具')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # convert-storage 命令
    parser_storage = subparsers.add_parser('convert-storage', help='转换整个存储目录格式')
    parser_storage.add_argument('source_path', help='源数据路径')
    parser_storage.add_argument('target_path', help='目标数据路径')
    parser_storage.add_argument('--source-format', choices=['parquet', 'csv'], 
                               default='parquet', help='源格式')
    parser_storage.add_argument('--target-format', choices=['parquet', 'csv'], 
                               default='csv', help='目标格式')
    
    # convert-file 命令
    parser_file = subparsers.add_parser('convert-file', help='转换单个文件')
    parser_file.add_argument('input_file', help='输入文件路径')
    parser_file.add_argument('output_file', help='输出文件路径')
    
    # compare 命令
    parser_compare = subparsers.add_parser('compare', help='比较格式性能')
    parser_compare.add_argument('data_path', help='测试数据路径')
    
    # analyze 命令
    parser_analyze = subparsers.add_parser('analyze', help='分析存储使用情况')
    parser_analyze.add_argument('data_path', help='数据路径')
    parser_analyze.add_argument('--format', choices=['parquet', 'csv'], 
                               default='parquet', help='数据格式')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging()
    
    # 执行命令
    if args.command == 'convert-storage':
        convert_storage_format(args.source_path, args.target_path, 
                             args.source_format, args.target_format)
    elif args.command == 'convert-file':
        convert_single_file(args.input_file, args.output_file)
    elif args.command == 'compare':
        compare_formats(args.data_path)
    elif args.command == 'analyze':
        analyze_storage(args.data_path, args.format)

if __name__ == '__main__':
    main()