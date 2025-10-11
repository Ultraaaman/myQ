# 新闻数据处理流水线

这是一个完整的新闻数据采集和分析流水线，包含两个核心脚本。

## 📁 目录结构

```
news_pipeline/
├── README.md                      # 本文档
├── period_news_fetcher.py         # 步骤1：新闻抓取脚本
└── batch_sentiment_analyzer.py    # 步骤2：情感分析脚本
```

## 🔄 工作流程

```
步骤1: 抓取新闻               步骤2: 情感分析
┌─────────────────┐           ┌─────────────────┐
│ period_news_    │           │ batch_sentiment_│
│ fetcher.py      │  ───────> │ analyzer.py     │
│                 │  CSV文件   │                 │
└─────────────────┘           └─────────────────┘
       ↓                             ↓
   原始新闻CSV                   分析结果CSV
```

## 📋 使用步骤

### 第一步：抓取新闻数据

```bash
# 进入目录
cd D:/projects/q/myQ/news_pipeline

# 抓取指定时间段的新闻
python period_news_fetcher.py --start_date 2025-09-01 --end_date 2025-09-30
```

**输出文件**：
- 位置：`D:/projects/q/myQ/output/period_news/`
- 文件名：`news_20250901_20250930.csv`

### 第二步：情感分析

```bash
# 对抓取的新闻进行情感分析
python batch_sentiment_analyzer.py --input news_20250901_20250930.csv
```

**输出文件**：
- 位置：`D:/projects/q/myQ/output/analyzed_news/`
- 文件名：`news_20250901_20250930_analyzed.csv`

## 📊 脚本详细说明

### 1. period_news_fetcher.py - 新闻抓取脚本

**功能**：
- 从Tushare抓取指定时间段的新闻
- 自动匹配股票池中的股票
- 自动过滤周末，只处理交易日
- 增量写入，数据安全

**参数**：
```bash
--start_date    开始日期 (YYYY-MM-DD)  [必需]
--end_date      结束日期 (YYYY-MM-DD)  [必需]
```

**示例**：
```bash
# 抓取2025年9月的新闻
python period_news_fetcher.py --start_date 2025-09-01 --end_date 2025-09-30

# 抓取单日新闻
python period_news_fetcher.py --start_date 2025-09-28 --end_date 2025-09-28
```

**输出CSV字段**：
- `datetime`: 新闻发布时间
- `title`: 新闻标题
- `content`: 新闻内容
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `industry`: 所属行业
- `matched_keyword`: 匹配的关键词
- `fetch_date`: 抓取日期

### 2. batch_sentiment_analyzer.py - 情感分析脚本

**功能**：
- 读取抓取的新闻CSV文件
- 使用大模型进行批量情感分析
- 智能重复检测，避免重复分析
- 增量写入，保证数据安全

**参数**：
```bash
--input         输入的新闻CSV文件名  [必需]
--batch_size    批次大小，默认4条
```

**示例**：
```bash
# 基本用法
python batch_sentiment_analyzer.py --input news_20250901_20250930.csv

# 指定批次大小（每次分析6条新闻）
python batch_sentiment_analyzer.py --input news_20250901_20250930.csv --batch_size 6
```

**分析维度**：
- `sentiment`: 情感标签（强烈正面/正面/中性偏正/中性/中性偏负/负面/强烈负面）
- `direct_impact_score`: 直接影响评分（-5到+5）
- `direct_impact_desc`: 直接影响描述
- `indirect_impact_score`: 间接影响评分（-5到+5）
- `indirect_impact_desc`: 间接影响描述
- `certainty`: 确定性（0-1）
- `time_to_effect`: 影响时间窗口
- `overall_score`: 综合评分（-10到+10）
- `risk_factors`: 主要风险因素
- `action_suggestion`: 建议操作
- `analysis_time`: 分析时间

## 🔧 配置要求

### API密钥配置

需要在 `D:/projects/q/myQ/config/api_config.py` 中配置：

```python
# Tushare API Token (用于新闻抓取)
TUSHARE_TOKEN = "your_tushare_token_here"

# OpenRouter API Key (用于情感分析)
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
```

### 股票池配置

确保 `D:/projects/q/myQ/config/stock_pool.json` 文件存在，格式如下：

```json
{
  "stocks": [
    {
      "stock_code": "600000.SH",
      "stock_name": "浦发银行",
      "industry": "银行"
    }
  ]
}
```

## 💡 使用技巧

### 1. 增量更新

如果需要更新某个时间段的数据：

```bash
# 第一次抓取
python period_news_fetcher.py --start_date 2025-09-01 --end_date 2025-09-30

# 如果发现有新新闻，再次运行（文件会被覆盖）
# 建议：先备份原文件或使用不同的日期范围
```

### 2. 分批分析（节省费用）

情感分析脚本支持多次运行，会自动跳过已分析的新闻：

```bash
# 第一次分析（分析所有新闻）
python batch_sentiment_analyzer.py --input news_20250901_20250930.csv

# 如果中途中断，再次运行会从上次停止的地方继续
python batch_sentiment_analyzer.py --input news_20250901_20250930.csv

# 💰 已分析过的新闻会被自动跳过，节省API费用
```

### 3. 数据安全

两个脚本都采用**增量写入**策略：
- 每处理一天/一批数据，立即写入磁盘
- 即使脚本中途中断，已处理的数据不会丢失
- 可以随时查看部分结果

### 4. 批次大小调优

情感分析的 `--batch_size` 参数影响：
- **较小值(2-4)**：更稳定，但耗时更长
- **较大值(6-8)**：更快，但可能遇到API限制
- **建议值**：4（默认值，平衡速度和稳定性）

## 📈 典型使用场景

### 场景1：月度新闻分析

```bash
# 1. 抓取整月新闻
python period_news_fetcher.py --start_date 2025-09-01 --end_date 2025-09-30

# 2. 进行情感分析
python batch_sentiment_analyzer.py --input news_20250901_20250930.csv

# 3. 结果可用于后续策略研究
```

### 场景2：历史数据回溯

```bash
# 批量抓取历史数据（按月抓取）
python period_news_fetcher.py --start_date 2025-01-01 --end_date 2025-01-31
python period_news_fetcher.py --start_date 2025-02-01 --end_date 2025-02-28
# ... 依此类推

# 然后分别分析
python batch_sentiment_analyzer.py --input news_20250101_20250131.csv
python batch_sentiment_analyzer.py --input news_20250201_20250228.csv
```

### 场景3：单日快速分析

```bash
# 分析今天的新闻
TODAY=$(date +%Y-%m-%d)
python period_news_fetcher.py --start_date $TODAY --end_date $TODAY
python batch_sentiment_analyzer.py --input news_${TODAY//-/}_${TODAY//-/}.csv
```

## ⚠️ 注意事项

1. **API限制**：
   - Tushare可能有调用频率限制
   - OpenRouter免费额度有限，注意成本控制
   - 脚本已内置延迟和重试机制

2. **数据时效性**：
   - 新闻数据可能有延迟
   - 周末和节假日通常无交易日新闻

3. **文件覆盖**：
   - 抓取脚本会覆盖同名文件
   - 分析脚本采用追加模式，不会覆盖

4. **中断恢复**：
   - 两个脚本都支持中断后继续
   - 重新运行即可从断点继续

## 📝 更新日志

- **2025-01-XX**: 初始版本
  - 创建新闻抓取脚本
  - 创建情感分析脚本
  - 采用增量写入策略
  - 支持重复检测

## 🤝 维护说明

这两个脚本设计为长期使用，建议：
1. 定期备份输出数据
2. 监控API使用量和费用
3. 根据实际情况调整批次大小
4. 定期更新股票池配置

---

**目录路径**: `D:/projects/q/myQ/news_pipeline/`
**输出路径**: `D:/projects/q/myQ/output/`
