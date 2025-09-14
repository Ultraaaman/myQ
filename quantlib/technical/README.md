# 技术指标分析模块 (Technical Analysis Module)

quantlib技术指标模块提供了全面的技术分析工具，包括趋势指标、震荡指标、成交量指标等，适用于股票、期货、外汇等金融市场的技术分析。模块内置多种数据源支持，可直接获取美股（Yahoo Finance）和A股（Akshare）的历史价格数据，实现一站式技术分析解决方案。

## 📁 模块结构

```
quantlib/technical/
├── __init__.py          # 模块初始化
├── base.py             # 基础类定义
├── trend.py            # 趋势指标
├── oscillator.py       # 震荡指标
├── volume.py           # 成交量指标
├── analyzer.py         # 综合分析器
├── data_sources.py     # 数据源管理（支持美股、A股数据获取）
└── README.md           # 文档说明
```

## 🚀 快速开始

### 数据获取与分析

```python
from quantlib.technical import (
    TechnicalAnalyzer, 
    get_stock_data, 
    get_a_share_data,
    TechnicalDataManager
)

# 方法一：直接获取美股数据进行分析
us_data = get_stock_data('AAPL', market='US', period='1y')
analyzer = TechnicalAnalyzer(us_data)
signal, strength, _ = analyzer.get_consensus_signal()
print(f"AAPL 综合信号: {signal}, 强度: {strength}")

# 方法二：获取A股数据进行分析
a_share_data = get_a_share_data('000001', period='1y')  # 平安银行
analyzer = TechnicalAnalyzer(a_share_data)
signal, strength, _ = analyzer.get_consensus_signal()
print(f"平安银行 综合信号: {signal}, 强度: {strength}")

# 方法三：使用数据管理器批量获取
manager = TechnicalDataManager()
stocks_data = manager.load_multiple_stocks(['AAPL', 'GOOGL'], market='US')
for symbol, data in stocks_data.items():
    analyzer = TechnicalAnalyzer(data)
    signal, strength, _ = analyzer.get_consensus_signal()
    print(f"{symbol}: 信号={signal}, 强度={strength}")
```

### 传统用法（自备数据）

```python
import pandas as pd
from quantlib.technical import TechnicalAnalyzer

# 准备OHLCV数据
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]  # 可选
})

# 创建技术分析器
analyzer = TechnicalAnalyzer(data)

# 计算所有指标
analyzer.calculate_all_indicators()

# 生成交易信号
analyzer.generate_all_signals()

# 获取综合信号
signal, strength, analysis = analyzer.get_consensus_signal()
print(f"综合信号: {signal}, 强度: {strength}")

# 生成分析报告
report = analyzer.generate_analysis_report()
print(report)

# 绘制技术分析图表
analyzer.plot_analysis()
```

## 📡 数据源管理 (Data Sources)

技术分析模块内置了多种数据源支持，可以轻松获取美股、A股的历史价格数据。

### 支持的数据源

- **Yahoo Finance**: 美股数据（需要安装 `yfinance`）
- **Akshare**: A股数据（需要安装 `akshare`）

### 安装依赖

```bash
# 美股数据支持
pip install yfinance

# A股数据支持  
pip install akshare
```

### 基本使用

#### 1. 便捷函数

```python
from quantlib.technical import get_stock_data, get_a_share_data, get_multiple_stocks_data

# 获取美股数据
apple_data = get_stock_data('AAPL', market='US', period='1y')

# 获取A股数据
ping_an_data = get_a_share_data('000001', period='6mo')  # 平安银行

# 批量获取美股数据
us_stocks = get_multiple_stocks_data(['AAPL', 'GOOGL', 'TSLA'], market='US')

# 批量获取A股数据
a_stocks = get_multiple_stocks_data(['000001', '600519', '000858'], market='CN')

# 获取沪深300指数数据
csi300_data = get_csi300_index(period='1y')  # 1年沪深300数据
```

#### 2. 数据管理器

```python
from quantlib.technical import TechnicalDataManager

# 创建数据管理器
manager = TechnicalDataManager()

# 加载单只股票（支持缓存）
data = manager.load_stock_data('AAPL', market='US', period='1y', use_cache=True)

# 批量加载多只股票
stocks_data = manager.load_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'], market='US')

# 清空缓存
manager.clear_cache()

# 获取A股热门股票列表
popular_stocks = manager.get_a_share_popular_stocks()
print("A股热门股票:", popular_stocks)
```

#### 3. 直接使用数据源类

```python
from quantlib.technical.data_sources import YahooFinanceDataSource, AkshareDataSource

# Yahoo Finance数据源
us_source = YahooFinanceDataSource('AAPL')
apple_data = us_source.get_historical_data(period='1y', interval='1d')

# Akshare数据源  
cn_source = AkshareDataSource('000001')
ping_an_data = cn_source.get_historical_data(period='1y', interval='daily')
```

### 数据格式

所有数据源返回的数据都会标准化为统一格式：

```python
# 标准化后的数据列
columns = ['date', 'open', 'high', 'low', 'close', 'volume']

# 示例数据
print(data.head())
#         date   open   high    low  close    volume
# 0 2023-01-03  125.0  126.5  124.2  125.8  50000000
# 1 2023-01-04  125.8  127.2  125.0  126.1  45000000
```

### 参数说明

#### period（时间周期）
- `'1y'`: 1年（默认）
- `'6mo'`: 6个月
- `'3mo'`: 3个月
- `'1mo'`: 1个月
- `'max'`: 最大可获取范围

#### interval（数据间隔）
- **美股**: `'1d'`（日线）, `'1h'`（小时线）, `'1m'`（分钟线）
- **A股**: `'daily'`（日线）

### A股股票代码格式

A股股票需要使用6位数字代码：

```python
# 正确的A股代码格式
codes = [
    '000001',  # 平安银行（深圳主板）
    '000002',  # 万科A
    '600519',  # 贵州茅台（上海主板）
    '002415',  # 海康威视（深圳中小板）
    '300750'   # 宁德时代（深圳创业板）
]

# 获取数据
for code in codes:
    data = get_a_share_data(code)
    if data is not None:
        print(f"{code}: 获取了 {len(data)} 条记录")
```

### 沪深300指数数据

提供了获取沪深300指数数据的功能，主要用于大盘基准对比：

```python
from quantlib.technical import get_csi300_index

# 获取沪深300数据
csi300_data = get_csi300_index(period='1y')     # 1年（默认）

# 用于大盘对比
from quantlib.visualization import CandlestickChart
stock_data = get_stock_data('000001', market='CN', period='1y')

chart = CandlestickChart(stock_data)
chart.add_benchmark(csi300_data, name="沪深300", color="gray")
chart.plot().show()
```

**支持的时间周期**: `'1mo'`, `'3mo'`, `'6mo'`, `'1y'`, `'5y'`

### 错误处理

```python
# 数据获取失败时的处理
data = get_stock_data('INVALID_SYMBOL')
if data is None:
    print("数据获取失败，请检查股票代码")
else:
    print(f"成功获取 {len(data)} 条记录")

# 沪深300数据获取错误处理
try:
    csi300_data = get_csi300_index(period='1y')
    if csi300_data is not None:
        print(f"沪深300数据获取成功: {len(csi300_data)} 条记录")
    else:
        print("沪深300数据获取失败")
except Exception as e:
    print(f"沪深300数据获取异常: {e}")
```

### 数据质量检查

```python
# 检查数据完整性
def check_data_quality(data):
    if data is None or data.empty:
        return False
    
    # 检查必要列
    required_columns = ['date', 'open', 'high', 'low', 'close']
    if not all(col in data.columns for col in required_columns):
        return False
    
    # 检查数据空值
    if data[required_columns].isnull().any().any():
        print("警告: 数据包含空值")
    
    # 检查价格逻辑性
    invalid_prices = (data['high'] < data['low']) | (data['high'] < data['close']) | (data['low'] > data['close'])
    if invalid_prices.any():
        print(f"警告: 发现 {invalid_prices.sum()} 条价格异常记录")
    
    return True

# 使用示例
data = get_stock_data('AAPL')
if check_data_quality(data):
    analyzer = TechnicalAnalyzer(data)
```

## 🔄 向后兼容函数 (Backward Compatibility Functions)

为了方便使用，technical模块提供了一系列向后兼容的便捷函数，可以直接计算单个指标：

### 快速使用示例

```python
from quantlib.technical import (
    calculate_ma, calculate_rsi, calculate_bollinger_bands,
    calculate_macd, calculate_stochastic
)
import pandas as pd

# 准备数据
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# 1. 移动平均线
sma_20 = calculate_ma(data, period=20, ma_type='sma')
ema_20 = calculate_ma(data, period=20, ma_type='ema')

# 2. RSI指标
rsi_14 = calculate_rsi(data, period=14)

# 3. 布林带
upper, middle, lower = calculate_bollinger_bands(data, period=20, std_dev=2.0)

# 4. MACD指标
macd_line, signal_line, histogram = calculate_macd(data, fast=12, slow=26, signal=9)

# 5. 随机指标
k_values, d_values = calculate_stochastic(data, k_period=14, d_period=3)
```

### 函数详细说明

#### 1. calculate_ma() - 移动平均线

```python
calculate_ma(data, period=20, ma_type='sma', price_column='close')
```

**参数**:
- `data`: DataFrame，包含OHLCV数据
- `period`: int，计算周期，默认20
- `ma_type`: str，均线类型 ('sma'|'ema')，默认'sma'
- `price_column`: str，价格列名，默认'close'

**返回**: pandas.Series，移动平均线值

**示例**:
```python
# 20日简单移动平均线
sma_20 = calculate_ma(data, period=20, ma_type='sma')

# 10日指数移动平均线
ema_10 = calculate_ma(data, period=10, ma_type='ema')

# 使用high价格计算均线
high_ma = calculate_ma(data, period=20, price_column='high')
```

#### 2. calculate_rsi() - 相对强弱指标

```python
calculate_rsi(data, period=14, price_column='close')
```

**参数**:
- `data`: DataFrame，包含OHLCV数据
- `period`: int，计算周期，默认14
- `price_column`: str，价格列名，默认'close'

**返回**: pandas.Series，RSI值(0-100)

**示例**:
```python
# 标准14日RSI
rsi = calculate_rsi(data, period=14)

# 短期9日RSI
rsi_short = calculate_rsi(data, period=9)

# 检查超买超卖
overbought = rsi > 70
oversold = rsi < 30
```

#### 3. calculate_bollinger_bands() - 布林带

```python
calculate_bollinger_bands(data, period=20, std_dev=2.0, price_column='close')
```

**参数**:
- `data`: DataFrame，包含OHLCV数据
- `period`: int，移动平均周期，默认20
- `std_dev`: float，标准差倍数，默认2.0
- `price_column`: str，价格列名，默认'close'

**返回**: tuple，(上轨, 中轨, 下轨)

**示例**:
```python
# 标准布林带
upper, middle, lower = calculate_bollinger_bands(data, period=20, std_dev=2.0)

# 紧窄布林带
upper_tight, middle_tight, lower_tight = calculate_bollinger_bands(data, period=20, std_dev=1.5)

# 检查突破信号
price = data['close']
breakout_up = price > upper
breakout_down = price < lower
```

#### 4. calculate_macd() - MACD指标

```python
calculate_macd(data, fast=12, slow=26, signal=9, price_column='close')
```

**参数**:
- `data`: DataFrame，包含OHLCV数据
- `fast`: int，快线周期，默认12
- `slow`: int，慢线周期，默认26
- `signal`: int，信号线周期，默认9
- `price_column`: str，价格列名，默认'close'

**返回**: tuple，(MACD线, 信号线, 柱状图)

**示例**:
```python
# 标准MACD
macd, signal, histogram = calculate_macd(data, fast=12, slow=26, signal=9)

# 快速MACD
macd_fast, signal_fast, hist_fast = calculate_macd(data, fast=5, slow=10, signal=5)

# 交易信号
golden_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))
death_cross = (macd < signal) & (macd.shift(1) >= signal.shift(1))
```

#### 5. calculate_stochastic() - 随机指标

```python
calculate_stochastic(data, k_period=14, d_period=3, smooth_k=3)
```

**参数**:
- `data`: DataFrame，包含OHLCV数据
- `k_period`: int，K值计算周期，默认14
- `d_period`: int，D值平滑周期，默认3
- `smooth_k`: int，K值平滑周期，默认3

**返回**: tuple，(K值, D值)

**示例**:
```python
# 标准随机指标
k, d = calculate_stochastic(data, k_period=14, d_period=3)

# 快速随机指标
k_fast, d_fast = calculate_stochastic(data, k_period=9, d_period=3)

# 交易信号
golden_cross = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 20)
death_cross = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 80)
```

### 组合使用示例

```python
from quantlib.technical import *
import pandas as pd
import numpy as np

# 获取数据
data = get_stock_data('AAPL', market='US', period='1y')

# 计算多个指标
sma_20 = calculate_ma(data, period=20, ma_type='sma')
ema_12 = calculate_ma(data, period=12, ma_type='ema')
rsi = calculate_rsi(data, period=14)
upper, middle, lower = calculate_bollinger_bands(data, period=20)
macd, signal, histogram = calculate_macd(data)

# 综合分析
price = data['close']

# 趋势分析
uptrend = price > sma_20
ema_signal = price > ema_12

# 超买超卖分析
overbought = rsi > 70
oversold = rsi < 30

# 布林带分析
bb_squeeze = (upper - lower) / middle < 0.1  # 布林带收缩
bb_breakout_up = price > upper
bb_breakout_down = price < lower

# MACD分析
macd_bullish = macd > signal
macd_cross_up = (macd > signal) & (macd.shift(1) <= signal.shift(1))

# 综合信号
buy_signal = (
    uptrend &
    oversold &
    (bb_breakout_down | macd_cross_up) &
    ema_signal
)

sell_signal = (
    ~uptrend &
    overbought &
    bb_breakout_up &
    ~macd_bullish
)

print(f"最新买入信号: {buy_signal.iloc[-1]}")
print(f"最新卖出信号: {sell_signal.iloc[-1]}")
```

## 📊 趋势指标 (Trend Indicators)

### 1. 移动平均线 (Moving Averages)

**作用**: 平滑价格波动，识别趋势方向

**类型**:
- **SMA (Simple Moving Average)**: 简单移动平均线
- **EMA (Exponential Moving Average)**: 指数移动平均线

**用法**:

**方法一：使用便捷函数（推荐）**
```python
from quantlib.technical import calculate_ma

# 简单移动平均线
sma_20 = calculate_ma(data, period=20, ma_type='sma')
print(f"最新20日SMA: {sma_20.iloc[-1]:.2f}")

# 指数移动平均线
ema_12 = calculate_ma(data, period=12, ma_type='ema')
print(f"最新12日EMA: {ema_12.iloc[-1]:.2f}")

# 金叉死叉信号
golden_cross = (sma_20 > ema_12) & (sma_20.shift(1) <= ema_12.shift(1))
death_cross = (sma_20 < ema_12) & (sma_20.shift(1) >= ema_12.shift(1))
```

**方法二：使用类方式**
```python
from quantlib.technical.trend import TrendIndicators

trend = TrendIndicators(data)
ma = trend.moving_averages(periods=[5, 10, 20, 50, 200])

# 获取结果
results = ma.results
print(f"20日SMA: {results['SMA_20'].iloc[-1]:.2f}")
print(f"20日EMA: {results['EMA_20'].iloc[-1]:.2f}")

# 获取交易信号
signals = ma.get_signals()
print(f"当前信号: {signals['signal'].iloc[-1]}")

# 多周期分析
short_ma = results['SMA_5']
long_ma = results['SMA_20']
trend_direction = "上升" if short_ma.iloc[-1] > long_ma.iloc[-1] else "下降"
print(f"短期趋势: {trend_direction}")
```

**交易信号**:
- 金叉: 短期均线上穿长期均线 → 买入信号
- 死叉: 短期均线下穿长期均线 → 卖出信号

### 2. MACD指标 (Moving Average Convergence Divergence)

**作用**: 判断趋势变化和买卖时机

**组成**:
- **MACD线**: 快线EMA - 慢线EMA
- **Signal线**: MACD线的EMA
- **Histogram**: MACD线 - Signal线

**用法**:

**方法一：使用便捷函数（推荐）**
```python
from quantlib.technical import calculate_macd

# 标准MACD设置
macd_line, signal_line, histogram = calculate_macd(data, fast=12, slow=26, signal=9)

print(f"MACD: {macd_line.iloc[-1]:.4f}")
print(f"Signal: {signal_line.iloc[-1]:.4f}")
print(f"Histogram: {histogram.iloc[-1]:.4f}")

# 交易信号判断
current_macd = macd_line.iloc[-1]
current_signal = signal_line.iloc[-1]
prev_macd = macd_line.iloc[-2]
prev_signal = signal_line.iloc[-2]

if current_macd > current_signal and prev_macd <= prev_signal:
    print("MACD金叉，买入信号")
elif current_macd < current_signal and prev_macd >= prev_signal:
    print("MACD死叉，卖出信号")

# MACD零轴穿越
zero_cross_up = (macd_line > 0) & (macd_line.shift(1) <= 0)
zero_cross_down = (macd_line < 0) & (macd_line.shift(1) >= 0)
```

**方法二：使用类方式**
```python
from quantlib.technical.trend import TrendIndicators

trend = TrendIndicators(data)
macd = trend.macd(fast=12, slow=26, signal=9)
results = macd.results

# 获取各组件
macd_line = results['MACD']
signal_line = results['Signal']
histogram = results['Histogram']

# 获取交易信号
signals = macd.get_signals()
print(f"MACD信号: {signals['macd_signal'].iloc[-1]}")
print(f"Histogram信号: {signals['histogram_signal'].iloc[-1]}")
```

**交易信号**:
- MACD上穿Signal线 → 买入信号
- MACD下穿Signal线 → 卖出信号
- Histogram > 0 → 多头市场
- Histogram < 0 → 空头市场

### 3. 布林带 (Bollinger Bands)

**作用**: 判断价格超买超卖状态

**组成**:
- **上轨**: MA + (标准差 × 倍数)
- **中轨**: 移动平均线
- **下轨**: MA - (标准差 × 倍数)

**用法**:

**方法一：使用便捷函数（推荐）**
```python
from quantlib.technical import calculate_bollinger_bands

# 标准布林带
upper, middle, lower = calculate_bollinger_bands(data, period=20, std_dev=2.0)
price = data['close']

print(f"上轨: {upper.iloc[-1]:.2f}")
print(f"中轨: {middle.iloc[-1]:.2f}")
print(f"下轨: {lower.iloc[-1]:.2f}")
print(f"当前价格: {price.iloc[-1]:.2f}")

# 布林带位置分析
bb_position = (price - lower) / (upper - lower) * 100
print(f"布林带位置: {bb_position.iloc[-1]:.1f}%")

# 交易信号
oversold_signal = price <= lower  # 触及下轨，超卖
overbought_signal = price >= upper  # 触及上轨，超买

# 布林带收缩/扩张
bandwidth = (upper - lower) / middle * 100
squeeze = bandwidth < bandwidth.rolling(20).mean()  # 收缩
expansion = bandwidth > bandwidth.rolling(20).mean()  # 扩张

print(f"当前带宽: {bandwidth.iloc[-1]:.2f}%")
print(f"是否收缩: {squeeze.iloc[-1]}")
```

**方法二：使用类方式**
```python
from quantlib.technical.trend import TrendIndicators

trend = TrendIndicators(data)
bb = trend.bollinger_bands(period=20, std_dev=2.0)
results = bb.results

# 获取各组件
upper = results['Upper_Band']
middle = results['Middle_Band']
lower = results['Lower_Band']
bandwidth = results['Bandwidth']
bb_position = results['BB_Position']

# 获取信号
signals = bb.get_signals()
```

**交易信号**:
- 价格触及下轨 → 超卖，考虑买入
- 价格触及上轨 → 超买，考虑卖出
- 布林带收缩 → 市场即将突破

### 4. ADX平均趋向指标 (Average Directional Index)

**作用**: 判断趋势强度，不判断方向

**组成**:
- **+DI**: 上升趋向指标
- **-DI**: 下降趋向指标  
- **ADX**: 趋势强度指标

**用法**:
```python
adx = trend.adx(period=14)
results = adx.results
```

**判断标准**:
- ADX > 25: 强趋势
- ADX < 20: 弱趋势或震荡市场
- +DI > -DI: 上升趋势
- +DI < -DI: 下降趋势

### 5. 抛物线SAR (Parabolic SAR)

**作用**: 追踪止损点，判断趋势反转

**特点**:
- 价格在SAR之上 → 上升趋势
- 价格在SAR之下 → 下降趋势
- SAR点位可作为止损位

**用法**:
```python
sar = trend.parabolic_sar()
results = sar.results
```

## 📈 震荡指标 (Oscillator Indicators)

### 1. RSI相对强弱指标 (Relative Strength Index)

**作用**: 判断超买超卖状态

**计算**: RSI = 100 - (100 / (1 + RS))  
其中 RS = 平均涨幅 / 平均跌幅

**用法**:

**方法一：使用便捷函数（推荐）**
```python
from quantlib.technical import calculate_rsi

# 标准14日RSI
rsi = calculate_rsi(data, period=14)
current_rsi = rsi.iloc[-1]

print(f"当前RSI: {current_rsi:.2f}")

# RSI区间判断
if current_rsi >= 80:
    status = "极度超买"
elif current_rsi >= 70:
    status = "超买"
elif current_rsi <= 20:
    status = "极度超卖"
elif current_rsi <= 30:
    status = "超卖"
else:
    status = "正常区间"

print(f"RSI状态: {status}")

# RSI背离分析（简化版）
price = data['close']
price_peaks = price.rolling(5).max() == price  # 价格高点
rsi_peaks = rsi.rolling(5).max() == rsi  # RSI高点

# 顶背离：价格创新高但RSI没有创新高
bearish_divergence = price_peaks & (rsi < rsi.shift(10))
# 底背离：价格创新低但RSI没有创新低
bullish_divergence = (price.rolling(5).min() == price) & (rsi > rsi.shift(10))

# RSI金叉死叉（与50线）
rsi_bullish = rsi > 50
rsi_cross_up = (rsi > 50) & (rsi.shift(1) <= 50)
rsi_cross_down = (rsi < 50) & (rsi.shift(1) >= 50)
```

**方法二：使用类方式**
```python
from quantlib.technical.oscillator import OscillatorIndicators

osc = OscillatorIndicators(data)
rsi_indicator = osc.rsi(period=14)
results = rsi_indicator.results

# 获取RSI值
rsi = results['RSI']
avg_gain = results['Avg_Gain']
avg_loss = results['Avg_Loss']

# 获取交易信号
signals = rsi_indicator.get_signals()
print(f"RSI信号: {signals['signal'].iloc[-1]}")

# 检查背离
if signals['bearish_divergence'].iloc[-1]:
    print("检测到顶背离，注意风险")
if signals['bullish_divergence'].iloc[-1]:
    print("检测到底背离，关注机会")
```

**判断标准**:
- RSI > 70: 超买区域，考虑卖出
- RSI < 30: 超卖区域，考虑买入
- RSI > 80: 极度超买
- RSI < 20: 极度超卖

### 2. KDJ随机指标

**作用**: 判断超买超卖，寻找买卖点

**组成**:
- **K值**: 随机值的平滑
- **D值**: K值的平滑
- **J值**: 3K - 2D

**用法**:

**方法一：使用便捷函数（推荐）**
```python
from quantlib.technical import calculate_stochastic

# 标准KDJ参数
k, d = calculate_stochastic(data, k_period=14, d_period=3, smooth_k=3)

# 计算J值
j = 3 * k - 2 * d

current_k = k.iloc[-1]
current_d = d.iloc[-1]
current_j = j.iloc[-1]

print(f"K值: {current_k:.2f}")
print(f"D值: {current_d:.2f}")
print(f"J值: {current_j:.2f}")

# KDJ区间判断
if current_k > 80 and current_d > 80:
    status = "超买区域"
elif current_k < 20 and current_d < 20:
    status = "超卖区域"
else:
    status = "正常区域"

print(f"KDJ状态: {status}")

# 金叉死叉信号
golden_cross = (k > d) & (k.shift(1) <= d.shift(1))
death_cross = (k < d) & (k.shift(1) >= d.shift(1))

# 超买超卖区域的金叉死叉（更可靠）
oversold_golden = golden_cross & (k < 20) & (d < 20)  # 超卖区金叉
overbought_death = death_cross & (k > 80) & (d > 80)  # 超买区死叉

print(f"最新金叉: {golden_cross.iloc[-1]}")
print(f"最新死叉: {death_cross.iloc[-1]}")
print(f"超卖区金叉: {oversold_golden.iloc[-1]}")
print(f"超买区死叉: {overbought_death.iloc[-1]}")

# KDJ钝化判断
k_stagnant = (k > 80).rolling(5).sum() >= 4  # K值在高位停留
d_stagnant = (d < 20).rolling(5).sum() >= 4  # D值在低位停留
```

**方法二：使用类方式**
```python
from quantlib.technical.oscillator import OscillatorIndicators

osc = OscillatorIndicators(data)
kdj = osc.kdj(k_period=9, d_period=3)
results = kdj.results

# 获取KDJ值
k = results['K']
d = results['D']
j = results['J']
rsv = results['RSV']  # 随机值

# 获取交易信号
signals = kdj.get_signals()
print(f"KD信号: {signals['kd_signal'].iloc[-1]}")
print(f"是否超买: {signals['overbought'].iloc[-1]}")
print(f"是否超卖: {signals['oversold'].iloc[-1]}")
print(f"综合信号: {signals['signal'].iloc[-1]}")

# 金叉死叉点
if signals['golden_cross'].iloc[-1]:
    print("K线上穿D线，金叉信号")
if signals['death_cross'].iloc[-1]:
    print("K线下穿D线，死叉信号")
```

**交易信号**:
- K > D: 多头信号
- K < D: 空头信号
- K、D在20以下金叉: 买入信号
- K、D在80以上死叉: 卖出信号

### 3. 威廉指标 (Williams %R)

**作用**: 判断超买超卖状态

**计算**: %R = (最高价 - 收盘价) / (最高价 - 最低价) × -100

**用法**:
```python
williams = osc.williams(period=14)
```

**判断标准**:
- %R < -80: 超卖，考虑买入
- %R > -20: 超买，考虑卖出

### 4. CCI顺势指标 (Commodity Channel Index)

**作用**: 判断价格偏离程度

**用法**:
```python
cci = osc.cci(period=20)
```

**判断标准**:
- CCI > +100: 超买
- CCI < -100: 超卖
- CCI > +200: 强烈超买
- CCI < -200: 强烈超卖

### 5. 随机震荡指标 (Stochastic Oscillator)

**作用**: 比较收盘价在一定时期内的相对位置

**用法**:
```python
stoch = osc.stochastic(k_period=14, d_period=3)
```

### 6. ROC变动率指标 (Rate of Change)

**作用**: 衡量价格变化速度

**计算**: ROC = (当前价格 / n期前价格 - 1) × 100

**用法**:
```python
roc = osc.roc(period=12)
```

## 📊 成交量指标 (Volume Indicators)

### 1. OBV能量潮 (On-Balance Volume)

**作用**: 通过成交量变化预测价格走势

**计算逻辑**:
- 收盘价上涨: OBV = 前OBV + 当日成交量
- 收盘价下跌: OBV = 前OBV - 当日成交量
- 收盘价持平: OBV = 前OBV

**用法**:

**方法一：通过TechnicalAnalyzer（推荐）**
```python
from quantlib.technical import TechnicalAnalyzer

analyzer = TechnicalAnalyzer(data)
analyzer.calculate_all_indicators()

# 获取OBV结果
obv_results = analyzer.indicators['obv'].results
obv_values = obv_results['OBV']
obv_ma = obv_results['OBV_MA']

print(f"当前OBV: {obv_values.iloc[-1]:.0f}")
print(f"OBV均线: {obv_ma.iloc[-1]:.0f}")

# 获取交易信号
obv_signals = analyzer.indicators['obv'].get_signals()
current_signal = obv_signals['signal'].iloc[-1]
divergence = obv_signals['divergence'].iloc[-1]
```

**方法二：直接使用VolumeIndicators类**
```python
from quantlib.technical.volume import VolumeIndicators

vol = VolumeIndicators(data)
obv_indicator = vol.obv()

# 获取计算结果
results = obv_indicator.results
obv_values = results['OBV']
obv_ma = results['OBV_MA']

# 获取交易信号
signals = obv_indicator.get_signals()
```

**分析要点**:
- OBV上升，价格上升: 确认上升趋势
- OBV下降，价格下降: 确认下降趋势
- 价量背离: 预警信号

### 2. VPT量价趋势 (Volume Price Trend)

**作用**: 结合价格和成交量分析趋势

**用法**:
```python
# 方法一：通过TechnicalAnalyzer
analyzer = TechnicalAnalyzer(data)
analyzer.calculate_all_indicators()
vpt_results = analyzer.indicators['vpt'].results
vpt_values = vpt_results['VPT']
vpt_ma = vpt_results['VPT_MA']

# 方法二：直接使用
from quantlib.technical.volume import VolumeIndicators
vol = VolumeIndicators(data)
vpt_indicator = vol.vpt()
results = vpt_indicator.results
```

### 3. VWAP成交量加权平均价格 (Volume Weighted Average Price)

**作用**: 反映真实的平均成交价格

**用法**:
```python
# 方法一：通过TechnicalAnalyzer
analyzer = TechnicalAnalyzer(data)
analyzer.calculate_all_indicators()
vwap_results = analyzer.indicators['vwap'].results
vwap_values = vwap_results['VWAP']
vwap_upper = vwap_results['VWAP_Upper']
vwap_lower = vwap_results['VWAP_Lower']

print(f"当前VWAP: {vwap_values.iloc[-1]:.2f}")
print(f"VWAP上轨: {vwap_upper.iloc[-1]:.2f}")
print(f"VWAP下轨: {vwap_lower.iloc[-1]:.2f}")

# 方法二：直接使用
from quantlib.technical.volume import VolumeIndicators
vol = VolumeIndicators(data)
vwap_20 = vol.vwap(period=20)  # 20日VWAP
vwap_cumulative = vol.vwap()   # 累积VWAP

# 获取结果
vwap_20_results = vwap_20.results
vwap_cum_results = vwap_cumulative.results
```

**应用**:
- 价格在VWAP之上: 多头市场
- 价格在VWAP之下: 空头市场
- 机构交易员常用的基准价格

### 4. CMF蔡金资金流量 (Chaikin Money Flow)

**作用**: 衡量资金流入流出情况

**用法**:
```python
# 方法一：通过TechnicalAnalyzer
analyzer = TechnicalAnalyzer(data)
analyzer.calculate_all_indicators()
cmf_results = analyzer.indicators['cmf'].results
cmf_values = cmf_results['CMF']
mf_volume = cmf_results['MF_Volume']

current_cmf = cmf_values.iloc[-1]
print(f"当前CMF: {current_cmf:.3f}")

if current_cmf > 0.1:
    print("资金流入信号")
elif current_cmf < -0.1:
    print("资金流出信号")
else:
    print("资金流动平衡")

# 方法二：直接使用
from quantlib.technical.volume import VolumeIndicators
vol = VolumeIndicators(data)
cmf_indicator = vol.chaikin_money_flow(period=20)
results = cmf_indicator.results
```

**判断标准**:
- CMF > 0.1: 资金流入
- CMF < -0.1: 资金流出
- CMF > 0.25: 强烈买入
- CMF < -0.25: 强烈卖出

### 5. A/D累积/派发线 (Accumulation/Distribution Line)

**作用**: 通过成交量和价位关系判断买卖压力

**用法**:
```python
# 方法一：通过TechnicalAnalyzer
analyzer = TechnicalAnalyzer(data)
analyzer.calculate_all_indicators()
ad_results = analyzer.indicators['ad'].results
ad_line = ad_results['AD_Line']
ad_ma = ad_results['AD_MA']

print(f"当前A/D线: {ad_line.iloc[-1]:.0f}")
print(f"A/D均线: {ad_ma.iloc[-1]:.0f}")

# 判断趋势
if ad_line.iloc[-1] > ad_ma.iloc[-1]:
    print("累积趋势，看涨")
else:
    print("派发趋势，看跌")

# 方法二：直接使用
from quantlib.technical.volume import VolumeIndicators
vol = VolumeIndicators(data)
ad_indicator = vol.accumulation_distribution()
results = ad_indicator.results

# 获取交易信号
signals = ad_indicator.get_signals()
```

## ⚠️ 重要使用说明

### 指标类方法的正确用法

**技术指标集合类（TrendIndicators, OscillatorIndicators, VolumeIndicators）的方法返回的是指标类实例，不是直接的计算结果。**

**❌ 错误用法：**
```python
# 这样会得到一个类实例，而不是计算结果
analyzer = TechnicalAnalyzer(data)
vpt = analyzer.volume.vpt()  # 返回VPT类实例
print(vpt)  # 打印的是类实例，不是数值
```

**✅ 正确用法：**
```python
# 方法一：通过results属性获取计算结果
analyzer = TechnicalAnalyzer(data)
vpt_indicator = analyzer.volume.vpt()  # 返回VPT类实例
vpt_values = vpt_indicator.results['VPT']  # 获取VPT数值
print(f"当前VPT: {vpt_values.iloc[-1]}")

# 方法二：使用TechnicalAnalyzer的统一接口（推荐）
analyzer = TechnicalAnalyzer(data)
analyzer.calculate_all_indicators()
vpt_values = analyzer.indicators['vpt'].results['VPT']
print(f"当前VPT: {vpt_values.iloc[-1]}")

# 方法三：使用便捷函数（适合单个指标计算）
from quantlib.technical import calculate_rsi, calculate_ma
rsi = calculate_rsi(data, period=14)  # 直接返回Series
sma = calculate_ma(data, period=20)   # 直接返回Series
```

### 常见结果字典键名

不同指标的结果字典包含不同的键名，以下是常见的：

**趋势指标结果键名：**
- MovingAverages: `'SMA_5'`, `'SMA_20'`, `'EMA_5'`, `'EMA_20'` 等
- MACD: `'MACD'`, `'Signal'`, `'Histogram'`
- BollingerBands: `'Upper_Band'`, `'Middle_Band'`, `'Lower_Band'`, `'Bandwidth'`, `'BB_Position'`
- ADX: `'Plus_DI'`, `'Minus_DI'`, `'ADX'`
- ParabolicSAR: `'SAR'`, `'Trend'`, `'AF'`, `'EP'`

**震荡指标结果键名：**
- RSI: `'RSI'`, `'Avg_Gain'`, `'Avg_Loss'`
- KDJ: `'K'`, `'D'`, `'J'`, `'RSV'`
- Williams: `'Williams_R'`
- CCI: `'CCI'`, `'Typical_Price'`
- Stochastic: `'K'`, `'D'`
- ROC: `'ROC'`, `'ROC_MA'`

**成交量指标结果键名：**
- OBV: `'OBV'`, `'OBV_MA'`
- VPT: `'VPT'`, `'VPT_MA'`, `'Price_Change_Rate'`
- VWAP: `'VWAP'`, `'VWAP_Upper'`, `'VWAP_Lower'`, `'VWAP_Std'`
- ChaikinMoneyFlow: `'CMF'`, `'MF_Multiplier'`, `'MF_Volume'`
- AccumulationDistribution: `'AD_Line'`, `'AD_MA'`, `'MF_Volume'`

### 获取交易信号

所有指标类实例都有`get_signals()`方法：

```python
# 获取任何指标的交易信号
indicator = analyzer.trend.macd()
signals = indicator.get_signals()

# 查看信号结构
print(signals.columns)  # 查看可用的信号列
print(signals.tail())   # 查看最近的信号

# 获取当前信号
current_signal = signals['signal'].iloc[-1]
if current_signal == 1:
    print("买入信号")
elif current_signal == -1:
    print("卖出信号")
else:
    print("中性信号")
```

## 🎯 综合分析器 (Technical Analyzer)

### 主要功能

1. **一键计算所有指标**
2. **生成综合交易信号**
3. **支撑阻力位识别**
4. **技术分析报告生成**
5. **可视化图表展示**

### 使用示例

```python
# 创建分析器
analyzer = TechnicalAnalyzer(data)

# 获取综合信号
signal, strength, details = analyzer.get_consensus_signal()

# 信号解读
if signal == 2:
    print("强烈看涨信号")
elif signal == 1:
    print("看涨信号")
elif signal == 0:
    print("中性信号")
elif signal == -1:
    print("看跌信号")
elif signal == -2:
    print("强烈看跌信号")

print(f"信号强度: {strength:.2f}")

# 识别支撑阻力位
levels = analyzer.identify_support_resistance()
print(f"支撑位: {levels['support_levels']}")
print(f"阻力位: {levels['resistance_levels']}")
```

## 📈 信号解读指南

### 信号强度分级

| 信号值 | 含义 | 建议操作 |
|--------|------|----------|
| +2 | 强烈看涨 | 重仓买入 |
| +1 | 看涨 | 适量买入 |
| 0 | 中性 | 持有观望 |
| -1 | 看跌 | 适量卖出 |
| -2 | 强烈看跌 | 重仓卖出 |

### 信号组合分析

**多头信号组合**:
- MA金叉 + RSI < 70 + MACD上穿 + 成交量放大
- 布林带下轨支撑 + KDJ低位金叉 + OBV上升

**空头信号组合**:
- MA死叉 + RSI > 70 + MACD下穿 + 成交量萎缩  
- 布林带上轨阻力 + KDJ高位死叉 + OBV下降

## ⚠️ 使用注意事项

### 1. 指标局限性
- **滞后性**: 大部分指标基于历史数据，存在滞后
- **假信号**: 震荡市场中容易产生频繁的假突破信号
- **参数敏感**: 不同参数设置会显著影响信号质量

### 2. 市场环境适应性
- **趋势市场**: 趋势指标效果较好（MA、MACD、ADX）
- **震荡市场**: 震荡指标效果较好（RSI、KDJ、布林带）
- **成交量确认**: 成交量指标用于确认价格信号的可靠性

### 3. 风险管理
- 不要依赖单一指标做决策
- 结合多个指标进行综合分析
- 设置合理的止损止盈位
- 控制仓位大小

### 4. 参数优化建议
- **短线交易**: 使用较短周期参数（5, 10, 20）
- **中线投资**: 使用中等周期参数（20, 50, 100）
- **长线投资**: 使用较长周期参数（100, 200, 250）

## 🔧 自定义和扩展

### 添加新指标

```python
from quantlib.technical.base import TechnicalIndicator

class MyCustomIndicator(TechnicalIndicator):
    def calculate(self, **kwargs):
        # 实现你的指标计算逻辑
        pass
    
    def _generate_signals(self):
        # 实现信号生成逻辑
        pass
```

### 参数调优

```python
# 针对不同市场调整参数
analyzer = TechnicalAnalyzer(data)

# 股票市场参数
stock_rsi = analyzer.oscillator.rsi(period=14)

# 外汇市场参数（波动性更大）
forex_rsi = analyzer.oscillator.rsi(period=21)

# 加密货币参数（高波动性）
crypto_rsi = analyzer.oscillator.rsi(period=10)
```

## 📚 进阶用法

### 多时间框架分析

```python
# 日线数据
daily_analyzer = TechnicalAnalyzer(daily_data)
daily_signal, _, _ = daily_analyzer.get_consensus_signal()

# 小时线数据
hourly_analyzer = TechnicalAnalyzer(hourly_data)
hourly_signal, _, _ = hourly_analyzer.get_consensus_signal()

# 综合判断
if daily_signal > 0 and hourly_signal > 0:
    print("多时间框架确认看涨信号")
```

### 动态阈值

```python
# 根据市场波动性调整RSI阈值
volatility = data['close'].rolling(20).std()
high_vol_threshold = volatility > volatility.quantile(0.8)

rsi = analyzer.indicators['rsi'].results['RSI']

# 高波动性时放宽阈值
buy_threshold = np.where(high_vol_threshold, 25, 30)
sell_threshold = np.where(high_vol_threshold, 75, 70)
```

## 📊 性能优化

### 批量处理
```python
# 对多个股票进行技术分析
stocks = ['AAPL', 'GOOGL', 'MSFT']
results = {}

for stock in stocks:
    data = get_stock_data(stock)  # 你的数据获取函数
    analyzer = TechnicalAnalyzer(data)
    signal, strength, _ = analyzer.get_consensus_signal()
    results[stock] = {'signal': signal, 'strength': strength}
```

### 增量更新
```python
# 只计算新增数据的指标值
def update_indicators(analyzer, new_data):
    # 追加新数据
    analyzer.data = pd.concat([analyzer.data, new_data])
    
    # 重新计算最近的指标值
    analyzer.calculate_all_indicators()
```

---

## 🚀 完整实战示例

下面是一个完整的技术分析实战示例，展示如何综合运用多个指标进行股票分析：

```python
from quantlib.technical import *
import pandas as pd
import numpy as np

def comprehensive_technical_analysis(symbol, market='US', period='1y'):
    """
    完整的技术分析示例

    Args:
        symbol: 股票代码
        market: 市场类型 ('US' | 'CN')
        period: 时间周期

    Returns:
        dict: 分析结果
    """

    # 1. 获取数据
    print(f"🔍 正在分析 {symbol}...")
    data = get_stock_data(symbol, market=market, period=period)

    if data is None or len(data) < 50:
        return {"error": "数据获取失败或数据不足"}

    # 2. 计算基础指标
    print("📊 计算技术指标...")

    # 移动平均线
    sma_5 = calculate_ma(data, period=5, ma_type='sma')
    sma_20 = calculate_ma(data, period=20, ma_type='sma')
    sma_50 = calculate_ma(data, period=50, ma_type='sma')
    ema_12 = calculate_ma(data, period=12, ma_type='ema')

    # RSI
    rsi = calculate_rsi(data, period=14)

    # MACD
    macd_line, signal_line, histogram = calculate_macd(data)

    # 布林带
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data, period=20)

    # KDJ
    k, d = calculate_stochastic(data, k_period=14, d_period=3)
    j = 3 * k - 2 * d

    # 3. 当前价格和指标值
    current_price = data['close'].iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_k = k.iloc[-1]
    current_d = d.iloc[-1]

    print(f"📈 当前价格: {current_price:.2f}")

    # 4. 趋势分析
    trend_signals = []

    # 均线趋势
    if current_price > sma_5.iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1]:
        trend_signals.append("多头排列")
        trend_score = 2
    elif current_price > sma_20.iloc[-1]:
        trend_signals.append("短期上升")
        trend_score = 1
    elif current_price < sma_5.iloc[-1] < sma_20.iloc[-1] < sma_50.iloc[-1]:
        trend_signals.append("空头排列")
        trend_score = -2
    else:
        trend_signals.append("震荡整理")
        trend_score = 0

    # 均线金叉死叉
    sma_golden = (sma_5.iloc[-1] > sma_20.iloc[-1]) and (sma_5.iloc[-2] <= sma_20.iloc[-2])
    sma_death = (sma_5.iloc[-1] < sma_20.iloc[-1]) and (sma_5.iloc[-2] >= sma_20.iloc[-2])

    if sma_golden:
        trend_signals.append("均线金叉")
    elif sma_death:
        trend_signals.append("均线死叉")

    # 5. 超买超卖分析
    momentum_signals = []
    momentum_score = 0

    # RSI分析
    if current_rsi >= 80:
        momentum_signals.append("RSI极度超买")
        momentum_score -= 2
    elif current_rsi >= 70:
        momentum_signals.append("RSI超买")
        momentum_score -= 1
    elif current_rsi <= 20:
        momentum_signals.append("RSI极度超卖")
        momentum_score += 2
    elif current_rsi <= 30:
        momentum_signals.append("RSI超卖")
        momentum_score += 1

    # KDJ分析
    kdj_golden = (current_k > current_d) and (k.iloc[-2] <= d.iloc[-2])
    kdj_death = (current_k < current_d) and (k.iloc[-2] >= d.iloc[-2])

    if kdj_golden and current_k < 20:
        momentum_signals.append("KDJ超卖区金叉")
        momentum_score += 1
    elif kdj_death and current_k > 80:
        momentum_signals.append("KDJ超买区死叉")
        momentum_score -= 1
    elif kdj_golden:
        momentum_signals.append("KDJ金叉")
    elif kdj_death:
        momentum_signals.append("KDJ死叉")

    # 6. MACD分析
    macd_signals = []
    macd_score = 0

    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]

    # MACD金叉死叉
    macd_golden = (current_macd > current_signal) and (macd_line.iloc[-2] <= signal_line.iloc[-2])
    macd_death = (current_macd < current_signal) and (macd_line.iloc[-2] >= signal_line.iloc[-2])

    if macd_golden:
        macd_signals.append("MACD金叉")
        macd_score += 1
    elif macd_death:
        macd_signals.append("MACD死叉")
        macd_score -= 1

    # MACD零轴
    if current_macd > 0:
        macd_signals.append("MACD多头区域")
        macd_score += 0.5
    else:
        macd_signals.append("MACD空头区域")
        macd_score -= 0.5

    # 7. 布林带分析
    bb_signals = []
    bb_score = 0

    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    if current_price >= bb_upper.iloc[-1]:
        bb_signals.append("突破布林上轨")
        bb_score -= 1
    elif current_price <= bb_lower.iloc[-1]:
        bb_signals.append("跌破布林下轨")
        bb_score += 1

    bb_signals.append(f"布林带位置: {bb_position*100:.1f}%")

    # 8. 综合评分
    total_score = trend_score + momentum_score + macd_score + bb_score

    if total_score >= 3:
        overall_signal = "强烈看涨"
        recommendation = "建议重仓买入"
    elif total_score >= 1:
        overall_signal = "看涨"
        recommendation = "建议适量买入"
    elif total_score <= -3:
        overall_signal = "强烈看跌"
        recommendation = "建议重仓卖出"
    elif total_score <= -1:
        overall_signal = "看跌"
        recommendation = "建议适量卖出"
    else:
        overall_signal = "中性"
        recommendation = "建议持有观望"

    # 9. 支撑阻力位（简化版）
    recent_high = data['high'].rolling(20).max().iloc[-1]
    recent_low = data['low'].rolling(20).min().iloc[-1]

    # 10. 整理结果
    result = {
        'symbol': symbol,
        'current_price': current_price,
        'analysis_date': data.index[-1].strftime('%Y-%m-%d'),
        'overall_signal': overall_signal,
        'total_score': total_score,
        'recommendation': recommendation,

        'trend': {
            'signals': trend_signals,
            'score': trend_score,
            'sma_5': sma_5.iloc[-1],
            'sma_20': sma_20.iloc[-1],
            'sma_50': sma_50.iloc[-1],
        },

        'momentum': {
            'signals': momentum_signals,
            'score': momentum_score,
            'rsi': current_rsi,
            'k': current_k,
            'd': current_d,
            'j': j.iloc[-1],
        },

        'macd': {
            'signals': macd_signals,
            'score': macd_score,
            'macd': current_macd,
            'signal': current_signal,
            'histogram': histogram.iloc[-1],
        },

        'bollinger': {
            'signals': bb_signals,
            'score': bb_score,
            'upper': bb_upper.iloc[-1],
            'middle': bb_middle.iloc[-1],
            'lower': bb_lower.iloc[-1],
            'position': bb_position,
        },

        'support_resistance': {
            'resistance': recent_high,
            'support': recent_low,
            'current_position': (current_price - recent_low) / (recent_high - recent_low),
        }
    }

    return result

def print_analysis_report(result):
    """打印分析报告"""
    if 'error' in result:
        print(f"❌ {result['error']}")
        return

    print(f"\n{'='*50}")
    print(f"📊 {result['symbol']} 技术分析报告")
    print(f"📅 分析日期: {result['analysis_date']}")
    print(f"💰 当前价格: {result['current_price']:.2f}")
    print(f"{'='*50}")

    print(f"\n🎯 综合评价")
    print(f"信号: {result['overall_signal']}")
    print(f"评分: {result['total_score']}")
    print(f"建议: {result['recommendation']}")

    print(f"\n📈 趋势分析 (评分: {result['trend']['score']})")
    for signal in result['trend']['signals']:
        print(f"  • {signal}")
    print(f"  SMA5: {result['trend']['sma_5']:.2f}")
    print(f"  SMA20: {result['trend']['sma_20']:.2f}")
    print(f"  SMA50: {result['trend']['sma_50']:.2f}")

    print(f"\n📊 动量分析 (评分: {result['momentum']['score']})")
    for signal in result['momentum']['signals']:
        print(f"  • {signal}")
    print(f"  RSI: {result['momentum']['rsi']:.2f}")
    print(f"  KDJ: K={result['momentum']['k']:.2f}, D={result['momentum']['d']:.2f}, J={result['momentum']['j']:.2f}")

    print(f"\n🔄 MACD分析 (评分: {result['macd']['score']})")
    for signal in result['macd']['signals']:
        print(f"  • {signal}")
    print(f"  MACD: {result['macd']['macd']:.4f}")
    print(f"  Signal: {result['macd']['signal']:.4f}")

    print(f"\n🎈 布林带分析 (评分: {result['bollinger']['score']})")
    for signal in result['bollinger']['signals']:
        print(f"  • {signal}")
    print(f"  上轨: {result['bollinger']['upper']:.2f}")
    print(f"  中轨: {result['bollinger']['middle']:.2f}")
    print(f"  下轨: {result['bollinger']['lower']:.2f}")

    print(f"\n📊 支撑阻力")
    print(f"  阻力位: {result['support_resistance']['resistance']:.2f}")
    print(f"  支撑位: {result['support_resistance']['support']:.2f}")
    print(f"  位置: {result['support_resistance']['current_position']*100:.1f}%")

# 使用示例
if __name__ == "__main__":
    # 分析苹果公司股票
    result = comprehensive_technical_analysis('AAPL', market='US', period='6mo')
    print_analysis_report(result)

    print("\n" + "="*80 + "\n")

    # 分析A股股票（平安银行）
    result_cn = comprehensive_technical_analysis('000001', market='CN', period='6mo')
    print_analysis_report(result_cn)
```

### 运行结果示例

```
==================================================
📊 AAPL 技术分析报告
📅 分析日期: 2024-01-15
💰 当前价格: 185.92
==================================================

🎯 综合评价
信号: 看涨
评分: 2
建议: 建议适量买入

📈 趋势分析 (评分: 1)
  • 短期上升
  • 均线金叉
  SMA5: 186.45
  SMA20: 182.30
  SMA50: 175.20

📊 动量分析 (评分: 0)
  • RSI正常区域
  • KDJ金叉
  RSI: 58.32
  KDJ: K=65.20, D=62.15, J=71.30

🔄 MACD分析 (评分: 1)
  • MACD金叉
  • MACD多头区域
  MACD: 0.0156
  Signal: 0.0098

🎈 布林带分析 (评分: 0)
  • 布林带位置: 72.5%
  上轨: 188.50
  中轨: 182.30
  下轨: 176.10

📊 支撑阻力
  阻力位: 195.50
  支撑位: 170.25
  位置: 62.0%
```

**🎉 现在你已经掌握了quantlib技术指标模块的完整用法！**

技术分析是量化投资的重要工具，合理运用这些指标可以帮助你更好地把握市场机会。记住，技术指标只是辅助工具，最重要的是结合基本面分析和风险管理，建立完整的投资体系。

### 📝 关键要点总结

1. **多指标确认**: 不要依赖单一指标，要综合多个指标的信号
2. **趋势为王**: 趋势指标确定大方向，震荡指标寻找买卖点
3. **成交量确认**: 价格突破需要成交量的配合
4. **风险控制**: 设置止损止盈，控制仓位大小
5. **参数调优**: 根据不同市场和品种调整指标参数
6. **市场适应**: 了解不同市场环境下指标的有效性