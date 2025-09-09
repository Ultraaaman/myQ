# 可视化模块 (Visualization Module)

quantlib可视化模块提供了全面的金融数据可视化功能，支持K线图、技术指标图表、成交量分析图、市场概览图表等多种图表类型。

## 📁 模块结构

```
quantlib/visualization/
├── __init__.py          # 模块初始化
├── base.py             # 基础图表类
├── candlestick.py      # K线图实现
├── technical.py        # 技术指标图表
├── volume.py           # 成交量图表
├── market.py           # 市场概览图表
├── utils.py            # 图表工具函数
├── themes.py           # 图表主题和样式
└── README.md           # 文档说明
```

## 🎨 支持的绘图引擎

- **matplotlib**: 静态图表，适合报告和打印
- **plotly**: 交互式图表，适合网页展示和探索性分析
- **mplfinance**: 专门的金融图表库，K线图效果最佳

## 💿 安装依赖

```bash
# 基础绘图支持
pip install matplotlib plotly

# 专业金融图表支持
pip install mplfinance

# 图像导出支持（可选）
pip install kaleido
```

## 🚀 快速开始

### 基本K线图

```python
from quantlib.visualization import CandlestickChart
from quantlib.technical import get_stock_data

# 获取数据
data = get_stock_data('AAPL', period='6mo')

# 创建K线图
chart = CandlestickChart(data, engine='plotly')
chart.add_ma([5, 20, 50])           # 添加移动平均线
chart.add_volume()                   # 添加成交量
chart.set_title("苹果公司股价走势")
chart.plot().show()                  # 显示图表
```

### 技术指标分析图

```python
from quantlib.visualization import TechnicalChart
from quantlib.technical import TechnicalAnalyzer

# 获取数据并计算指标
data = get_stock_data('000001', market='CN', period='1y')  # 平安银行
analyzer = TechnicalAnalyzer(data)
analyzer.calculate_all_indicators()

# 创建技术指标图
tech_chart = TechnicalChart(data, engine='plotly')
tech_chart.add_rsi()                 # 添加RSI指标
tech_chart.add_macd()                # 添加MACD指标
tech_chart.add_bollinger_bands()     # 添加布林带
tech_chart.plot().show()
```

### 成交量分析图

```python
from quantlib.visualization import VolumeChart

# 创建成交量分析图
volume_chart = VolumeChart(data, engine='plotly')
volume_chart.add_volume_ma([5, 20])  # 添加成交量移动平均
volume_chart.enable_volume_profile() # 启用成交量分布
volume_chart.plot().show()
```

### 市场概览图

```python
from quantlib.visualization import MarketChart
from quantlib.technical import get_multiple_stocks_data

# 获取多只股票数据
stocks_data = get_multiple_stocks_data(
    ['AAPL', 'GOOGL', 'MSFT', 'TSLA'], 
    market='US', 
    period='1y'
)

# 创建市场概览图
market_chart = MarketChart(stocks_data, engine='plotly')
market_chart.plot_market_overview().show()
```

### 大盘基准对比

在分析个股时，添加大盘指数作为基准对比：

```python
from quantlib.visualization import CandlestickChart
from quantlib.technical import get_stock_data, get_csi300_index

# 获取个股和大盘数据
stock_data = get_stock_data('000001', market='CN', period='1y')  # 平安银行
benchmark_data = get_csi300_index(period='1y')  # 沪深300指数

# 创建带大盘对比的K线图
chart = CandlestickChart(stock_data, engine='plotly')
chart.add_ma([20, 60])                          # 添加均线
chart.add_benchmark(benchmark_data, 
                   name="沪深300", 
                   color="gray")                # 添加大盘基准线
chart.add_volume()                              # 添加成交量
chart.set_title("平安银行 vs 沪深300对比")
chart.plot().show()
```

## 📊 K线图功能 (CandlestickChart)

### 基本功能

```python
from quantlib.visualization import CandlestickChart

chart = CandlestickChart(data)

# 设置基本属性
chart.set_title("股票K线图")
chart.set_size(width=14, height=10)

# 添加技术指标
chart.add_ma([5, 10, 20, 50])       # 移动平均线
chart.add_volume()                   # 成交量

# 设置日期范围
chart.set_date_range('2023-01-01', '2023-12-31')

# 绘制并显示
chart.plot().show()
```

### 高级功能

```python
# 添加支撑阻力位
chart.add_support_resistance([150.0, 180.0, 200.0], 
                            colors=['green', 'red', 'blue'])

# 添加趋势线
chart.add_trend_line('2023-01-01', 150.0, 
                    '2023-06-01', 180.0, 
                    color='blue', style='--')

# 将技术指标数据添加到图表
from quantlib.technical import TechnicalAnalyzer
analyzer = TechnicalAnalyzer(data)
analyzer.calculate_all_indicators()

chart.add_indicator({
    'RSI': analyzer.indicators['rsi'].results['RSI'],
    'MACD': analyzer.indicators['macd'].results['MACD']
})
```

### 保存图表

```python
# 保存为图片
chart.save('stock_chart.png', dpi=300)

# 保存为HTML（plotly）
chart.save('stock_chart.html')
```

### 大盘基准对比

```python
# 添加大盘基准线进行对比
from quantlib.technical import get_csi300_index

benchmark_data = get_csi300_index(period='1y')
chart.add_benchmark(benchmark_data, name="沪深300", color="gray")

# 可以添加多个基准
other_benchmark = get_stock_data('000016', market='CN')  # 上证50ETF
chart.add_benchmark(other_benchmark, name="上证50", color="blue")
```

**基准对比特点**：
- 自动标准化显示，便于对比相对表现
- 虚线样式，与个股K线区分
- 自动时间对齐，只显示交集部分

## 📈 技术指标图表 (TechnicalChart)

### 支持的指标

```python
from quantlib.visualization import TechnicalChart

tech_chart = TechnicalChart(data, engine='plotly')

# RSI相对强弱指标
tech_chart.add_rsi(period=14, subplot=True)

# MACD指标
tech_chart.add_macd(fast=12, slow=26, signal=9, subplot=True)

# 布林带（主图显示）
tech_chart.add_bollinger_bands(period=20, std_dev=2.0)

# KDJ随机指标
tech_chart.add_kdj(k_period=9, d_period=3, subplot=True)

# 成交量指标
tech_chart.add_volume_indicators(subplot=True)

tech_chart.plot().show()
```

### 指标参数说明

| 指标 | 参数 | 说明 |
|------|------|------|
| RSI | period | 计算周期，默认14 |
| MACD | fast, slow, signal | 快线、慢线、信号线周期 |
| 布林带 | period, std_dev | 周期和标准差倍数 |
| KDJ | k_period, d_period | K值和D值计算周期 |

## 📊 成交量图表 (VolumeChart)

### 基本用法

```python
from quantlib.visualization import VolumeChart

volume_chart = VolumeChart(data, engine='plotly')

# 添加成交量移动平均线
volume_chart.add_volume_ma([5, 10, 20])

# 启用成交量分布图
volume_chart.enable_volume_profile(True)

# 添加成交量震荡器
volume_chart.add_volume_oscillator(period=14)

# 添加价量趋势指标
volume_chart.add_price_volume_trend()

volume_chart.plot().show()
```

### 成交量统计

```python
# 获取成交量统计信息
stats = volume_chart.get_volume_statistics()
print(f"平均成交量: {stats['average_volume']:,.0f}")
print(f"最大成交量: {stats['max_volume']:,.0f}")
print(f"成交量标准差: {stats['volume_std']:,.0f}")
print(f"大成交量日数: {stats['high_volume_days']}")
```

## 🏪 市场概览图表 (MarketChart)

### 相关性分析

```python
from quantlib.visualization import MarketChart

# 多股票数据字典
stocks_data = {
    'AAPL': apple_data,
    'GOOGL': google_data,
    'MSFT': microsoft_data
}

market_chart = MarketChart(stocks_data, engine='plotly')

# 绘制股票相关性矩阵
market_chart.plot_correlation_matrix().show()
```

### 收益率比较

```python
# 不同时间周期的收益率比较
market_chart.plot_returns_comparison(period='1M').show()  # 月收益率
market_chart.plot_returns_comparison(period='1Y').show()  # 年收益率
```

### 波动率分析

```python
# 波动率比较
market_chart.plot_volatility_comparison(window=20).show()  # 20日波动率
```

### 综合市场概览

```python
# 一个图表显示所有信息
market_chart.plot_market_overview().show()
```

### 市场摘要统计

```python
# 获取市场摘要
summary = market_chart.get_market_summary()
print(f"最佳表现: {summary['best_performer']}")
print(f"最差表现: {summary['worst_performer']}")
print(f"最高波动: {summary['most_volatile']}")
print(f"最低波动: {summary['least_volatile']}")
```


## 🎨 主题和样式

### 内置主题

```python
from quantlib.visualization import get_theme, list_themes

# 查看所有可用主题
themes = list_themes()
print("可用主题:", themes)

# 使用不同主题
light_theme = get_theme('light')
dark_theme = get_theme('dark')
minimal_theme = get_theme('minimal')
colorblind_theme = get_theme('colorblind')

# 应用主题到图表
chart = CandlestickChart(data, engine='matplotlib')
chart.theme = dark_theme
```

### 自定义主题

```python
from quantlib.visualization import ChartTheme

class MyCustomTheme(ChartTheme):
    def _default_colors(self):
        return {
            'up': '#ff0000',      # 自定义上涨色
            'down': '#00ff00',    # 自定义下跌色
            'background': '#f0f0f0',
            # ... 其他颜色配置
        }

# 使用自定义主题
custom_theme = MyCustomTheme()
chart.theme = custom_theme
```

## 🛠️ 工具函数

### ChartUtils 工具类

```python
from quantlib.visualization.utils import ChartUtils

# 自动检测数据周期
period = ChartUtils.detect_chart_periods(data)
print(f"数据周期: {period}")

# 计算价格变化统计
price_stats = ChartUtils.calculate_price_change(data)
print(f"当前价格: {price_stats['current_price']}")
print(f"涨跌幅: {price_stats['change_percent']:.2f}%")

# 识别支撑阻力位
levels = ChartUtils.identify_support_resistance(data, window=20)
print(f"支撑位: {levels['support_levels']}")
print(f"阻力位: {levels['resistance_levels']}")

# 计算成交量分布
volume_profile = ChartUtils.calculate_volume_profile(data, bins=50)

# 格式化显示
price_formatted = ChartUtils.format_price(1234567.89)  # "1.23M"
volume_formatted = ChartUtils.format_volume(1000000)   # "1.00M"
```

### 数据重采样

```python
# 将日线数据重采样为周线
weekly_data = ChartUtils.resample_data(data, frequency='W')

# 重采样为月线
monthly_data = ChartUtils.resample_data(data, frequency='M')
```

## 🎯 最佳实践

### 1. 选择合适的绘图引擎

```python
# 静态报告使用matplotlib
chart = CandlestickChart(data, engine='matplotlib')

# 交互式分析使用plotly
chart = CandlestickChart(data, engine='plotly')

# 专业金融分析使用mplfinance
chart = CandlestickChart(data, engine='mplfinance')
```

### 2. 合理设置图表尺寸

```python
# 根据数据量自动调整尺寸
optimal_size = ChartUtils.get_optimal_chart_size(len(data))
chart.set_size(*optimal_size)

# 手动设置
chart.set_size(width=16, height=10)  # 大图表
chart.set_size(width=10, height=6)   # 小图表
```

### 3. 组合使用多种图表

```python
# 创建综合分析图
candlestick_chart = CandlestickChart(data)
candlestick_chart.add_ma([20, 50]).add_volume()

technical_chart = TechnicalChart(data)
technical_chart.add_rsi().add_macd()

volume_chart = VolumeChart(data)
volume_chart.add_volume_ma([20]).enable_volume_profile()

# 分别显示或保存
candlestick_chart.plot().show()
technical_chart.plot().show()
volume_chart.plot().show()
```

### 4. 批量处理多只股票

```python
# 批量创建图表
symbols = ['AAPL', 'GOOGL', 'MSFT']
for symbol in symbols:
    data = get_stock_data(symbol, period='1y')
    chart = CandlestickChart(data, engine='plotly')
    chart.add_ma([20, 50]).add_volume()
    chart.set_title(f"{symbol} 股价分析")
    chart.save(f"{symbol}_chart.html")
```

## ⚠️ 注意事项

### 1. 数据格式要求

- 必需列: `['date', 'open', 'high', 'low', 'close']`
- 可选列: `['volume']`
- 日期列必须是datetime格式
- 价格列必须是数值格式

### 2. 性能考虑

- 大数据量时推荐使用matplotlib
- Plotly在数据点超过10000时可能较慢
- 可以使用数据重采样减少数据点

### 3. 内存使用

- 大图表会占用较多内存
- 及时释放不需要的图表对象
- 批量处理时考虑分批进行

### 4. 导出格式

```python
# 不同引擎支持的导出格式
# matplotlib: png, jpg, pdf, svg
chart.save('chart.png', dpi=300)

# plotly: html, png, jpg, pdf, svg
chart.save('chart.html')
chart.save('chart.png')
```

---

**🎉 现在你已经掌握了quantlib可视化模块的完整用法！**

可视化是数据分析的重要环节，合理运用这些图表工具可以帮助你更直观地理解市场数据和技术指标。建议结合具体的分析需求选择合适的图表类型和绘图引擎。