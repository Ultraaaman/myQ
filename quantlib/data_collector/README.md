# 数据收集服务

定期获取A股分钟级数据并按symbol和月份存储的服务。

## 功能特性

- **定期数据收集**: 支持定时或连续运行模式，自动获取分钟级数据
- **智能存储**: 按股票代码和月份组织数据，支持增量更新和去重
- **多种调度**: 支持每日、每周、每小时或自定义间隔的数据收集
- **数据管理**: 自动清理旧数据，可配置保留月数
- **错误重试**: 支持多次重试和错误处理
- **状态监控**: 提供详细的运行状态和数据统计

## 架构组件

### 1. DataStorage (storage.py)
负责数据存储管理，按以下结构组织文件：
```
data/minute_data/
├── 000001/
│   ├── 2024-01.parquet
│   ├── 2024-02.parquet
│   └── ...
├── 600519/
│   ├── 2024-01.parquet
│   └── ...
└── ...
```

### 2. DataCollectionService (service.py)
主服务类，负责：
- 配置管理
- 数据收集协调
- 错误处理和重试
- 数据统计

### 3. DataScheduler (scheduler.py)
调度器，支持：
- 定时执行
- 连续运行
- 信号处理
- 状态监控

## 快速开始

### 1. 安装依赖

确保安装以下依赖包：
```bash
pip install pandas pyarrow akshare schedule
```

### 2. 配置文件

创建 `config/data_collection.json`:
```json
{
  "symbols": ["000001", "000002", "600519"],
  "intervals": ["1min"],
  "storage_path": "data/minute_data",
  "file_format": "parquet",
  "collection_frequency_hours": 168,
  "market": "CN",
  "max_retries": 3,
  "retry_delay_seconds": 60,
  "cleanup_enabled": true,
  "keep_months": 12,
  "log_level": "INFO"
}
```

### 3. 运行方式

#### 方式一：使用命令行工具
```bash
# 执行一次数据收集
python data_collection_daemon.py once

# 启动定时服务（每周日凌晨2点）
python data_collection_daemon.py start --schedule weekly --time "02:00" --day sunday

# 启动连续运行模式（每168小时=1周）
python data_collection_daemon.py continuous --interval 168

# 查看状态
python data_collection_daemon.py status

# 添加新股票
python data_collection_daemon.py add-symbol 600000

# 移除股票
python data_collection_daemon.py remove-symbol 600000
```

#### 方式二：编程方式
```python
from quantlib.data_collector.service import DataCollectionService
from quantlib.data_collector.scheduler import DataScheduler

# 创建服务
service = DataCollectionService("config/data_collection.json")

# 执行一次收集
results = service.collect_data_once()

# 或创建调度器
scheduler = DataScheduler(service)
scheduler.start_scheduled("weekly", "02:00", "sunday")
```

## 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| symbols | 监控的股票代码列表 | ["000001", "000002", ...] |
| intervals | 数据时间间隔 | ["1min", "5min"] |
| storage_path | 数据存储路径 | "data/minute_data" |
| file_format | 文件格式 | "parquet" |
| collection_frequency_hours | 收集频率（小时） | 168 |
| market | 市场类型 | "CN" |
| max_retries | 最大重试次数 | 3 |
| retry_delay_seconds | 重试延迟（秒） | 60 |
| cleanup_enabled | 是否启用清理 | true |
| keep_months | 保留月份数 | 12 |
| log_level | 日志级别 | "INFO" |

## 数据格式

存储的分钟级数据包含以下列：
- `date`: 时间戳
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

## API 使用

### 数据收集
```python
# 创建服务
service = DataCollectionService()

# 收集数据
results = service.collect_data_once()
print(f"成功: {results['success_count']}, 失败: {results['failed_count']}")

# 添加监控股票
service.add_symbol("600000")

# 更新配置
service.update_config(collection_frequency_hours=72)
```

### 数据读取
```python
from quantlib.data_collector.storage import DataStorage

# 创建存储管理器
storage = DataStorage("data/minute_data")

# 获取可用股票
symbols = storage.get_available_symbols()

# 读取特定月份数据
data = storage.load_data("000001", 2024, 3)

# 读取股票所有数据
all_data = storage.load_symbol_data("000001")

# 读取日期范围数据
from datetime import date
range_data = storage.load_symbol_data(
    "000001", 
    start_date=date(2024, 1, 1),
    end_date=date(2024, 3, 31)
)
```

### 调度管理
```python
from quantlib.data_collector.scheduler import DataScheduler

# 创建调度器
scheduler = DataScheduler(service)

# 定时调度
scheduler.start_scheduled("daily", "02:00")

# 连续运行
scheduler.start_continuous(168)  # 168小时间隔

# 停止服务
scheduler.stop()

# 查看状态
status = scheduler.get_status()
```

## 注意事项

1. **数据限制**: akshare的分钟级数据只能获取近5个交易日，因此需要定期运行以避免数据丢失
2. **网络依赖**: 需要稳定的网络连接来获取数据
3. **存储空间**: 分钟级数据量较大，注意磁盘空间
4. **运行权限**: 确保有写入存储路径的权限
5. **时区问题**: 数据时间基于交易所时区

## 监控和维护

### 日志文件
- 主日志: `logs/data_collection.log`
- 收集日志: `logs/data_collection/collection_YYYY-MM-DD.json`

### 状态检查
```bash
# 查看服务状态
python data_collection_daemon.py status

# 查看日志
tail -f logs/data_collection.log
```

### 数据清理
服务会自动清理超过 `keep_months` 的旧数据，也可手动清理：
```python
storage.clean_old_data("000001", keep_months=6)
```

## 故障排除

### 常见问题

1. **无法获取数据**
   - 检查网络连接
   - 确认股票代码正确
   - 查看akshare是否正常

2. **存储失败**
   - 检查磁盘空间
   - 确认目录权限
   - 查看错误日志

3. **调度不工作**
   - 检查系统时间
   - 确认调度配置
   - 查看进程状态

### 错误代码
- 获取数据失败: 检查网络和股票代码
- 存储失败: 检查磁盘空间和权限
- 调度失败: 检查时间配置和系统资源

## 数据重采样功能

新增强大的数据重采样功能，支持将分钟级数据转换到不同时间间隔：

### 支持的时间间隔
- **分钟级**: 1min, 2min, 3min, 5min, 10min, 15min, 30min, 60min
- **小时级**: 1h, 2h, 4h, 6h, 8h, 12h  
- **日级**: 1d, 2d, 3d
- **周级**: 1w, 2w
- **月级**: 1M, 3M, 6M
- **年级**: 1Y

### 重采样使用方法

**命令行方式:**
```bash
# 重采样单个股票到5分钟
python data_collection_daemon.py resample 000001 5min --save

# 批量重采样到多个间隔
python data_collection_daemon.py resample 000001 "5min,15min,1h" --batch --save

# 查看重采样摘要
python data_collection_daemon.py resample-summary 000001 --intervals "5min,15min,1h,1d"
```

**编程方式:**
```python
from quantlib.data_collector.storage import DataStorage
from quantlib.data_collector.resample import resample_ohlcv

# 通过存储系统重采样
storage = DataStorage("data/minute_data")
data_5min = storage.load_and_resample("000001", "5min")

# 批量重采样
intervals = ["5min", "15min", "1h"]
results = storage.batch_resample_symbol("000001", intervals, save_results=True)

# 直接使用重采样器
from quantlib.data_collector.resample import DataResampler
resampler = DataResampler()
data_15min = resampler.resample_ohlcv(raw_data, "15min")
```

### 重采样特性
- **OHLCV聚合**: 开盘价(first)、最高价(max)、最低价(min)、收盘价(last)、成交量(sum)
- **自定义聚合**: 支持自定义聚合规则
- **批量处理**: 一次重采样到多个时间间隔
- **上采样**: 支持低频到高频的插值
- **数据完整性**: 自动处理不完整周期

### 应用场景
- 技术分析: 生成不同周期K线图
- 策略回测: 多时间框架分析
- 数据压缩: 减少存储空间
- 性能优化: 减少计算量

## 扩展功能

可以根据需要扩展以下功能：
- 支持更多数据源
- 添加数据验证
- 实现分布式收集
- 添加Web界面
- 集成监控告警

## 示例代码

完整示例请参考 `examples/data_collection_example.py`。