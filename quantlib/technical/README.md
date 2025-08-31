# 技术指标分析模块 (Technical Analysis Module)

quantlib技术指标模块提供了全面的技术分析工具，包括趋势指标、震荡指标、成交量指标等，适用于股票、期货、外汇等金融市场的技术分析。

## 📁 模块结构

```
quantlib/technical/
├── __init__.py          # 模块初始化
├── base.py             # 基础类定义
├── trend.py            # 趋势指标
├── oscillator.py       # 震荡指标
├── volume.py           # 成交量指标
├── analyzer.py         # 综合分析器
└── README.md           # 文档说明
```

## 🚀 快速开始

### 基本用法

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