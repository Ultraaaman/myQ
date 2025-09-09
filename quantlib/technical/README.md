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

## 📊 趋势指标 (Trend Indicators)

### 1. 移动平均线 (Moving Averages)

**作用**: 平滑价格波动，识别趋势方向

**类型**:
- **SMA (Simple Moving Average)**: 简单移动平均线
- **EMA (Exponential Moving Average)**: 指数移动平均线

**用法**:
```python
from quantlib.technical.trend import TrendIndicators

trend = TrendIndicators(data)
ma = trend.moving_averages(periods=[5, 10, 20, 50, 200])

# 获取结果
results = ma.results
print(f"20日SMA: {results['SMA_20'].iloc[-1]}")

# 获取交易信号
signals = ma.get_signals()
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
```python
macd = trend.macd(fast=12, slow=26, signal=9)
results = macd.results
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
```python
bb = trend.bollinger_bands(period=20, std_dev=2.0)
results = bb.results
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
```python
from quantlib.technical.oscillator import OscillatorIndicators

osc = OscillatorIndicators(data)
rsi = osc.rsi(period=14)
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
```python
kdj = osc.kdj(k_period=9, d_period=3)
results = kdj.results
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
```python
from quantlib.technical.volume import VolumeIndicators

vol = VolumeIndicators(data)
obv = vol.obv()
```

**分析要点**:
- OBV上升，价格上升: 确认上升趋势
- OBV下降，价格下降: 确认下降趋势
- 价量背离: 预警信号

### 2. VPT量价趋势 (Volume Price Trend)

**作用**: 结合价格和成交量分析趋势

**用法**:
```python
vpt = vol.vpt()
```

### 3. VWAP成交量加权平均价格 (Volume Weighted Average Price)

**作用**: 反映真实的平均成交价格

**用法**:
```python
vwap = vol.vwap(period=20)  # 20日VWAP
vwap_cumulative = vol.vwap()  # 累积VWAP
```

**应用**:
- 价格在VWAP之上: 多头市场
- 价格在VWAP之下: 空头市场
- 机构交易员常用的基准价格

### 4. CMF蔡金资金流量 (Chaikin Money Flow)

**作用**: 衡量资金流入流出情况

**用法**:
```python
cmf = vol.chaikin_money_flow(period=20)
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
ad = vol.accumulation_distribution()
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

**🎉 现在你已经掌握了quantlib技术指标模块的完整用法！**

技术分析是量化投资的重要工具，合理运用这些指标可以帮助你更好地把握市场机会。记住，技术指标只是辅助工具，最重要的是结合基本面分析和风险管理，建立完整的投资体系。