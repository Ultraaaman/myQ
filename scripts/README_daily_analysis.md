# 每日新闻因子分析系统

## 功能介绍

这个系统可以自动分析每日新闻，找出与股票池中股票相关的新闻，并使用大模型对新闻进行情感分析和评分，最后生成因子强度报告。

## 核心功能

1. **全量新闻获取**：通过Tushare API获取每日财经新闻
2. **智能匹配**：将新闻与候选股票池进行关键词匹配
3. **AI评分**：使用大模型分析新闻对股票的影响，给出综合评分
4. **因子报告**：生成强因子股票列表和详细分析报告
5. **追加模式**：支持同日多次运行，新数据会覆盖旧数据，不同日期间不会覆盖

## 文件结构

```
D:/projects/q/myQ/
├── config/
│   ├── stock_pool.json          # 候选股票池
│   └── api_config.py            # API配置文件
├── scripts/
│   ├── daily_news_analyzer.py   # 主分析脚本
│   ├── run_daily_analysis.py    # 启动脚本
│   └── README_daily_analysis.md # 说明文档
└── output/
    └── daily_analysis/
        ├── news_analysis_20241224.csv      # 新闻分析结果
        ├── stock_scores_20241224.csv       # 股票评分结果
        └── factor_report_20241224.md       # 因子强度报告
```

## 配置步骤

### 1. 配置API密钥

编辑 `config/api_config.py`：

```python
# Tushare API配置（在 https://tushare.pro/ 获取）
TUSHARE_TOKEN = "your_actual_tushare_token"

# OpenRouter API配置（在 https://openrouter.ai/ 获取）
OPENROUTER_API_KEY = "your_actual_openrouter_api_key"
```

### 2. 安装依赖

```bash
pip install tushare pandas requests
```

## 使用方法

### 方法1: 分析今天的新闻

```bash
cd D:/projects/q/myQ/scripts
python run_daily_analysis.py
```

### 方法2: 分析指定日期的新闻

```bash
python run_daily_analysis.py --date 2024-12-24
```

### 方法3: 直接运行主脚本

```bash
python daily_news_analyzer.py
```

## 输出结果

### 1. 新闻分析结果 (news_analysis_YYYYMMDD.csv)

包含以下字段：
- `news_id`: 新闻唯一标识
- `original_title`: 原始新闻标题
- `original_content`: 原始新闻内容
- `stock_name`: 相关股票名称
- `stock_code`: 股票代码
- `sentiment`: 情感倾向
- `overall_score`: 综合评分(-10到+10)
- `certainty`: 确定性(0-1)
- `action_suggestion`: 建议操作

### 2. 股票评分结果 (stock_scores_YYYYMMDD.csv)

包含以下字段：
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `factor_strength`: 因子强度
- `avg_score`: 平均评分
- `max_score`: 最高评分
- `news_count`: 相关新闻数量

### 3. 因子强度报告 (factor_report_YYYYMMDD.md)

Markdown格式的详细报告，包括：
- 强因子股票排行榜
- 每只股票的详细新闻分析
- 操作建议

## 评分机制

### 因子强度计算公式

```
因子强度 = 平均评分 × 0.4 + 最高评分 × 0.3 + 确定性 × 10 × 0.3
```

### 强因子股票筛选

- 因子强度 ≥ 3.0 的股票被认为是强因子股票
- 报告会重点关注这些股票

## 数据管理

### 追加模式机制

- **同日覆盖**：多次运行同一天的分析时，新结果会覆盖旧结果
- **日期隔离**：不同日期的数据互不影响
- **历史保留**：历史数据会被完整保留

### 示例场景

```bash
# 第一次运行（上午10点）
python run_daily_analysis.py --date 2024-12-24

# 第二次运行（下午3点，有新的新闻）
python run_daily_analysis.py --date 2024-12-24
# 结果：2024-12-24的数据被完全更新

# 运行前一天的分析
python run_daily_analysis.py --date 2024-12-23
# 结果：2024-12-23的数据独立保存，不影响12-24的数据
```

## 股票池配置

股票池定义在 `config/stock_pool.json` 中，包含：
- 股票名称和代码
- 行业分类
- 主营业务
- 市值信息

可以根据需要修改这个文件来调整分析的股票范围。

## 注意事项

1. **API限制**：注意Tushare和OpenRouter的API调用频率限制
2. **网络稳定**：确保网络连接稳定，大模型API调用可能需要一些时间
3. **数据量**：新闻数量较多时，分析过程可能需要较长时间
4. **成本控制**：OpenRouter API按使用量收费，注意控制成本

## 常见问题

### Q: 程序运行很慢？
A: 这是正常的，因为需要调用大模型API对每条新闻进行分析。可以通过减少新闻内容长度或调整API延迟来优化。

### Q: 没有找到相关股票新闻？
A: 检查股票池配置和关键词匹配逻辑，可能需要添加更多行业关键词。

### Q: API调用失败？
A: 检查API密钥配置，确认网络连接，查看API服务状态。

## 扩展功能

可以考虑添加的功能：
- 微信/邮件通知强因子股票
- 历史因子强度趋势分析
- 与股价表现的关联分析
- 自动生成交易信号