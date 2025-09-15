# 数据收集服务使用指南

## 📁 文件结构

```
myQ/
├── scripts/
│   └── data_collection_daemon.py    # 数据收集守护进程（主要入口）
├── config/
│   └── data_collection.json         # 配置文件（自动创建）
├── quantlib/data_collector/         # 核心服务模块
│   ├── service.py                   # 数据收集服务
│   ├── storage.py                   # 数据存储管理
│   ├── scheduler.py                 # 任务调度器
│   ├── resample.py                  # 数据重采样工具
│   └── README.md                    # 详细技术文档
├── examples/
│   ├── data_collection_example.py   # 编程使用示例
│   └── resample_example.py         # 重采样功能示例
└── data/minute_data/               # 数据存储目录（自动创建）
    ├── 000001/
    │   ├── 2024-01.parquet
    │   └── 2024-02.parquet
    └── 600519/
        └── 2024-01.parquet
```

## 🚀 快速开始

### 1. 配置文件

**配置文件是自动读取的**，默认路径为 `config/data_collection.json`。

- **首次运行时**：会自动创建默认配置文件
- **自定义配置**：可以通过 `--config` 参数指定其他配置文件
- **配置内容**：包括监控股票、时间间隔、存储路径等

默认配置内容：
```json
{
  "symbols": ["000001", "000002", "000858", "600519", "600036", "000300"],
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

### 2. 基础使用

```bash
# 进入项目目录
cd D:\projects\q\myQ

# 执行一次数据收集（测试用）
python scripts/data_collection_daemon.py once

# 查看服务状态和已收集的数据
python scripts/data_collection_daemon.py status

# 启动定时服务（推荐：每周日凌晨2点执行）
python scripts/data_collection_daemon.py start --schedule weekly --time "02:00" --day sunday
```

### 3. 股票管理

```bash
# 添加新股票到监控列表
python scripts/data_collection_daemon.py add-symbol 600000

# 移除股票
python scripts/data_collection_daemon.py remove-symbol 600000

# 查看当前监控的股票和数据情况
python scripts/data_collection_daemon.py status
```

### 4. 数据重采样

```bash
# 将1分钟数据重采样为5分钟数据
python scripts/data_collection_daemon.py resample 000001 5min --save

# 批量重采样到多个时间间隔
python scripts/data_collection_daemon.py resample 000001 "5min,15min,1h,1d" --batch --save

# 查看重采样摘要（压缩比等统计信息）
python scripts/data_collection_daemon.py resample-summary 000001
```

## ⚙️ 运行模式

### 1. 一次性执行
```bash
python scripts/data_collection_daemon.py once
```
适用于：测试、手动补充数据

### 2. 定时调度（推荐）
```bash
# 每周执行
python scripts/data_collection_daemon.py start --schedule weekly --time "02:00" --day sunday

# 每天执行
python scripts/data_collection_daemon.py start --schedule daily --time "03:00"

# 每小时执行
python scripts/data_collection_daemon.py start --schedule hourly --time "1"
```

### 3. 连续运行
```bash
# 每168小时（1周）执行一次
python scripts/data_collection_daemon.py continuous --interval 168
```

## 📊 配置说明

### 配置文件位置
- **默认位置**：`config/data_collection.json`（自动读取）
- **自定义位置**：使用 `--config` 参数指定

### 重要配置项

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| `symbols` | 监控的股票代码列表 | 根据需要添加 |
| `intervals` | 数据时间间隔 | `["1min"]` (推荐只保存1分钟) |
| `collection_frequency_hours` | 收集频率（小时） | `168`（1周）|
| `storage_path` | 数据存储路径 | `"data/minute_data"` |
| `keep_months` | 保留数据月数 | `12` |
| `max_retries` | 失败重试次数 | `3` |

### 自动功能
1. **配置创建**：首次运行自动创建默认配置
2. **目录创建**：自动创建数据存储目录
3. **日志记录**：自动创建logs目录和日志文件
4. **数据清理**：自动清理超过保留期限的旧数据

## 💾 数据存储

### 存储结构
```
data/minute_data/
├── 000001/              # 股票代码目录
│   ├── 2024-01.parquet   # 按月分文件
│   ├── 2024-02.parquet
│   └── ...
└── 600519/
    ├── 2024-01.parquet
    └── ...
```

### 数据格式
每个文件包含该股票该月的分钟级OHLCV数据：
- `date`: 时间戳
- `open`: 开盘价
- `high`: 最高价  
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

### 重采样数据
重采样后的数据可选择保存到独立目录：
```
data/minute_data_resampled_5min/
data/minute_data_resampled_1h/
```

## 🔧 高级用法

### 编程接口
```python
from quantlib.data_collector.service import DataCollectionService
from quantlib.data_collector.storage import DataStorage

# 创建服务
service = DataCollectionService("config/data_collection.json")

# 执行收集
results = service.collect_data_once()

# 数据操作
storage = DataStorage("data/minute_data")
data = storage.load_symbol_data("000001")  # 加载所有数据
data_5min = storage.load_and_resample("000001", "5min")  # 重采样
```

### 自定义配置文件
```bash
# 使用自定义配置
python scripts/data_collection_daemon.py once --config my_config.json
```

### 日志和监控
- **主日志**：`logs/data_collection.log`
- **收集记录**：`logs/data_collection/collection_YYYY-MM-DD.json`
- **实时监控**：`tail -f logs/data_collection.log`

## 📝 常见问题

### Q1: 配置文件在哪里？
A: 默认在 `config/data_collection.json`，首次运行会自动创建。可用 `--config` 指定其他位置。

### Q2: 数据收集频率如何设置？
A: 由于akshare分钟级数据限制为近5个交易日，建议每周收集一次（168小时）以避免数据丢失。

### Q3: 如何添加新股票？
A: 使用命令 `python scripts/data_collection_daemon.py add-symbol 股票代码`

### Q4: 数据存储在哪里？
A: 默认存储在 `data/minute_data/`，按股票代码和月份组织文件。

### Q5: 如何生成日线数据？
A: 使用重采样功能：`python scripts/data_collection_daemon.py resample 000001 1d --save`

### Q6: 服务如何后台运行？
A: Linux/Mac使用nohup：`nohup python scripts/data_collection_daemon.py start &`
   Windows可使用任务计划程序或服务方式运行。

## 🛠️ 系统要求

- Python 3.7+
- 依赖包：pandas, pyarrow, akshare, schedule
- 网络连接（获取数据）
- 足够的磁盘空间（分钟级数据量较大）

## 💡 **数据存储策略说明**

### 为什么只保存1分钟数据？

**遵循数据存储最佳实践**：
- ✅ **信息无损**: 1分钟是最细粒度，包含所有信息
- ✅ **灵活转换**: 可以转换为任意粗粒度（5min, 15min, 1h, 1d等）
- ✅ **不可逆性**: 粗粒度数据无法还原成细粒度
- ✅ **存储优化**: 避免重复存储，节省空间

### 使用方式
```bash
# 获取1分钟原始数据
python scripts/data_analyzer.py show 000001 --interval 1min

# 实时转换为5分钟数据
python scripts/data_analyzer.py show 000001 --interval 5min

# 实时转换为日线数据  
python scripts/data_analyzer.py show 000001 --interval 1d

# 比较不同时间间隔
python scripts/data_analyzer.py compare 000001 --intervals "1min,5min,15min,1h,1d"
```

### 新增数据分析工具

创建了 `scripts/data_analyzer.py` 专门用于基于1分钟数据的分析：

**主要功能：**
- 📊 数据概览和统计
- 📈 价格走势分析  
- 📉 成交量分析
- ⏰ 交易时段分析
- 📋 多时间间隔对比
- 💾 数据导出

**使用示例：**
```bash
# 显示数据概览
python scripts/data_analyzer.py show 000001

# 价格走势分析（自动转换为日线）
python scripts/data_analyzer.py price 000001 --interval 1d

# 成交量分析
python scripts/data_analyzer.py volume 000001 --interval 5min

# 交易时段分析（基于1分钟数据）
python scripts/data_analyzer.py session 000001

# 导出为CSV格式
python scripts/data_analyzer.py export 000001 --interval 1h --output hourly_data.csv
```

## 📈 最佳实践

1. **存储策略**: 只保存1分钟数据，按需重采样
2. **定期运行**：建议每周自动运行一次
3. **监控日志**：定期查看日志文件确保正常运行
4. **数据备份**：重要数据建议定期备份
5. **磁盘空间**：监控磁盘使用，必要时调整保留月数
6. **网络稳定**：确保运行环境网络稳定
7. **按需分析**: 使用data_analyzer.py进行各种时间间隔的分析

## 🔗 相关文档

- 详细技术文档：`quantlib/data_collector/README.md`  
- 编程示例：`examples/data_collection_example.py`
- 重采样示例：`examples/resample_example.py`