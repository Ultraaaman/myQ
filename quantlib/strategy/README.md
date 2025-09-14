# 策略模块 (Strategy Module)

量化交易策略开发和管理模块，提供完整的策略框架和多种预建策略。

## 🚀 快速开始

### 基本使用

```python
from quantlib.strategy import BaseStrategy, create_ma_cross_strategy
from quantlib.market_data import get_stock_data

# 获取股票数据
data = get_stock_data('000001', market='CN', period='1y')

# 创建均线交叉策略
strategy = create_ma_cross_strategy(['000001'], short_window=20, long_window=60)
strategy.set_data({'000001': data})
strategy.initialize()

# 生成交易信号
from datetime import datetime
current_data = {'000001': data.iloc[-1]}
signals = strategy.generate_signals(datetime.now(), current_data)

for signal in signals:
    print(f"信号: {signal.signal_type} {signal.symbol} at {signal.timestamp}")
```

## 📋 核心组件

### 1. BaseStrategy - 策略基类

所有策略的基础类，定义了策略接口和通用功能。

```python
from quantlib.strategy.base import BaseStrategy, SignalType, TradingSignal

class MyStrategy(BaseStrategy):
    def __init__(self, symbols, **kwargs):
        super().__init__(symbols, **kwargs)
        # 初始化策略参数

    def initialize(self):
        # 计算技术指标
        for symbol in self.symbols:
            df = self.data[symbol]
            # 添加指标
            self.add_indicator(symbol, 'sma_20', df['close'].rolling(20).mean())

    def generate_signals(self, current_time, current_data):
        signals = []
        # 生成交易信号逻辑
        return signals
```

### 2. 信号类型

```python
from quantlib.strategy.base import SignalType

# 可用的信号类型
SignalType.BUY      # 买入信号
SignalType.SELL     # 卖出信号
SignalType.HOLD     # 持有信号
```

### 3. 交易信号

```python
from quantlib.strategy.base import TradingSignal

signal = TradingSignal(
    symbol='000001',
    signal_type=SignalType.BUY,
    timestamp=datetime.now(),
    confidence=0.8,  # 信号置信度 (0-1)
    metadata={'reason': 'MA crossover'}
)
```

## 🛠️ 预建策略

### 1. 均线交叉策略 (MA Cross Strategy)

基于短期和长期移动平均线交叉的策略。

```python
from quantlib.strategy.examples import MovingAverageCrossStrategy

# 方法1: 直接创建
strategy = MovingAverageCrossStrategy(['000001'], short_window=20, long_window=60)

# 方法2: 使用便捷函数
from quantlib.strategy import create_ma_cross_strategy
strategy = create_ma_cross_strategy(['000001'], 20, 60)

# 设置数据并初始化
strategy.set_data({'000001': data})
strategy.initialize()

print(f"策略名称: {strategy.name}")
print(f"策略描述: {strategy.description}")
```

**策略逻辑:**
- 买入信号: 短期均线上穿长期均线
- 卖出信号: 短期均线下穿长期均线

### 2. RSI策略

基于相对强弱指标的超买超卖策略。

```python
from quantlib.strategy.examples import RSIStrategy

# 创建RSI策略
strategy = RSIStrategy(['000001'], period=14, oversold=30, overbought=70)

# 使用便捷函数
from quantlib.strategy import create_rsi_strategy
strategy = create_rsi_strategy(['000001'], period=14, oversold=30, overbought=70)
```

**策略逻辑:**
- 买入信号: RSI < 30 (超卖)
- 卖出信号: RSI > 70 (超买)

### 3. 布林带策略

基于布林带的均值回归策略。

```python
from quantlib.strategy.examples import BollingerBandsStrategy

strategy = BollingerBandsStrategy(['000001'], period=20, std_dev=2.0)
strategy.set_data({'000001': data})
strategy.initialize()
```

**策略逻辑:**
- 买入信号: 价格跌破下轨
- 卖出信号: 价格涨破上轨

### 4. MACD策略

基于MACD指标的趋势跟随策略。

```python
from quantlib.strategy.examples import MACDStrategy

strategy = MACDStrategy(['000001'], fast=12, slow=26, signal=9)
```

**策略逻辑:**
- 买入信号: MACD线上穿信号线
- 卖出信号: MACD线下穿信号线

### 5. 动量策略

基于价格动量的策略。

```python
from quantlib.strategy.examples import MomentumStrategy

strategy = MomentumStrategy(['000001'], lookback_period=20, threshold=0.02)
```

**策略逻辑:**
- 买入信号: 价格动量 > threshold
- 卖出信号: 价格动量 < -threshold

### 6. 均值回归策略

基于统计学均值回归的策略。

```python
from quantlib.strategy.examples import MeanReversionStrategy

strategy = MeanReversionStrategy(['000001'], window=20, threshold=2.0)
```

**策略逻辑:**
- 买入信号: 价格偏离均值 < -threshold倍标准差
- 卖出信号: 价格偏离均值 > threshold倍标准差

### 7. 多因子策略

综合多个技术指标的策略。

```python
from quantlib.strategy.examples import MultiFactorStrategy

strategy = MultiFactorStrategy(['000001'],
    ma_short=10, ma_long=30,
    rsi_period=14, rsi_oversold=30, rsi_overbought=70
)
```

**策略逻辑:**
- 综合考虑均线交叉和RSI指标
- 多个信号确认时才产生交易信号

## 📊 策略管理

### 设置策略数据

```python
# 单个股票
strategy.set_data({'000001': data})

# 多个股票
data_dict = {
    '000001': get_stock_data('000001', market='CN'),
    '000002': get_stock_data('000002', market='CN')
}
strategy.set_data(data_dict)
```

### 初始化策略

```python
# 初始化策略（计算技术指标）
strategy.initialize()

# 检查初始化状态
print(f"策略是否已初始化: {strategy.is_initialized}")

# 查看指标
for symbol in strategy.symbols:
    indicators = strategy.get_indicators(symbol)
    print(f"{symbol} 的指标: {list(indicators.keys())}")
```

### 生成交易信号

```python
from datetime import datetime

# 获取当前数据
current_data = {}
for symbol in strategy.symbols:
    current_data[symbol] = data[symbol].iloc[-1]  # 最新一条数据

# 生成信号
signals = strategy.generate_signals(datetime.now(), current_data)

# 处理信号
for signal in signals:
    print(f"""
    交易信号:
    - 股票代码: {signal.symbol}
    - 信号类型: {signal.signal_type}
    - 时间戳: {signal.timestamp}
    - 置信度: {signal.confidence:.2f}
    - 附加信息: {signal.metadata}
    """)
```

## 🔧 高级功能

### 1. 自定义策略开发

```python
from quantlib.strategy.base import BaseStrategy, SignalType, TradingSignal
from quantlib.technical import calculate_ma, calculate_rsi

class CustomStrategy(BaseStrategy):
    def __init__(self, symbols, ma_period=20, rsi_period=14):
        super().__init__(symbols)
        self.ma_period = ma_period
        self.rsi_period = rsi_period
        self.name = "Custom Strategy"
        self.description = "自定义策略示例"

    def initialize(self):
        """初始化策略，计算技术指标"""
        for symbol in self.symbols:
            df = self.data[symbol]

            # 计算移动平均线
            ma = calculate_ma(df['close'], period=self.ma_period)
            self.add_indicator(symbol, 'ma', ma)

            # 计算RSI
            rsi = calculate_rsi(df['close'], period=self.rsi_period)
            self.add_indicator(symbol, 'rsi', rsi)

        self.is_initialized = True

    def generate_signals(self, current_time, current_data):
        """生成交易信号"""
        signals = []

        for symbol in self.symbols:
            if symbol not in current_data:
                continue

            indicators = self.indicators.get(symbol, {})
            if not indicators:
                continue

            current_price = current_data[symbol]['close']
            ma_value = indicators['ma'].iloc[-1]
            rsi_value = indicators['rsi'].iloc[-1]

            # 买入条件: 价格突破均线且RSI不超买
            if current_price > ma_value and rsi_value < 70:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    timestamp=current_time,
                    confidence=0.7,
                    metadata={
                        'ma': ma_value,
                        'rsi': rsi_value,
                        'price': current_price
                    }
                )
                signals.append(signal)

            # 卖出条件: 价格跌破均线或RSI超买
            elif current_price < ma_value or rsi_value > 70:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    timestamp=current_time,
                    confidence=0.6,
                    metadata={
                        'ma': ma_value,
                        'rsi': rsi_value,
                        'price': current_price
                    }
                )
                signals.append(signal)

        return signals

# 使用自定义策略
strategy = CustomStrategy(['000001'], ma_period=20, rsi_period=14)
strategy.set_data({'000001': data})
strategy.initialize()
```

### 2. 策略参数优化

```python
# 策略参数网格搜索示例
def optimize_strategy_parameters():
    best_params = None
    best_return = -float('inf')

    # 参数组合
    ma_periods = [10, 20, 30]
    rsi_periods = [10, 14, 21]

    for ma_period in ma_periods:
        for rsi_period in rsi_periods:
            # 创建策略实例
            strategy = CustomStrategy(['000001'],
                                    ma_period=ma_period,
                                    rsi_period=rsi_period)

            # 回测策略（这里需要配合backtest模块）
            # returns = backtest_strategy(strategy, data)
            #
            # if returns > best_return:
            #     best_return = returns
            #     best_params = {'ma_period': ma_period, 'rsi_period': rsi_period}

    return best_params
```

### 3. 多时间框架策略

```python
class MultiTimeframeStrategy(BaseStrategy):
    def __init__(self, symbols):
        super().__init__(symbols)
        self.name = "Multi Timeframe Strategy"

    def initialize(self):
        for symbol in self.symbols:
            daily_data = self.data[symbol]  # 日线数据

            # 计算不同时间框架的指标
            # 短期指标 (5日)
            short_ma = calculate_ma(daily_data['close'], period=5)
            self.add_indicator(symbol, 'short_ma', short_ma)

            # 中期指标 (20日)
            medium_ma = calculate_ma(daily_data['close'], period=20)
            self.add_indicator(symbol, 'medium_ma', medium_ma)

            # 长期指标 (60日)
            long_ma = calculate_ma(daily_data['close'], period=60)
            self.add_indicator(symbol, 'long_ma', long_ma)

        self.is_initialized = True

    def generate_signals(self, current_time, current_data):
        signals = []

        for symbol in self.symbols:
            indicators = self.indicators.get(symbol, {})
            if not indicators:
                continue

            short_ma = indicators['short_ma'].iloc[-1]
            medium_ma = indicators['medium_ma'].iloc[-1]
            long_ma = indicators['long_ma'].iloc[-1]

            # 多层确认信号
            if short_ma > medium_ma > long_ma:  # 上升趋势
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    timestamp=current_time,
                    confidence=0.8,
                    metadata={'trend': 'uptrend'}
                )
                signals.append(signal)
            elif short_ma < medium_ma < long_ma:  # 下降趋势
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    timestamp=current_time,
                    confidence=0.8,
                    metadata={'trend': 'downtrend'}
                )
                signals.append(signal)

        return signals
```

## 🔍 调试和监控

### 策略状态检查

```python
# 检查策略状态
print(f"策略名称: {strategy.name}")
print(f"策略描述: {strategy.description}")
print(f"交易品种: {strategy.symbols}")
print(f"是否初始化: {strategy.is_initialized}")

# 检查数据状态
if strategy.data:
    for symbol, df in strategy.data.items():
        print(f"{symbol}: {len(df)} 条数据记录")

# 检查指标状态
for symbol in strategy.symbols:
    indicators = strategy.get_indicators(symbol)
    print(f"{symbol} 指标数量: {len(indicators)}")
    for name, indicator in indicators.items():
        print(f"  {name}: 最新值 = {indicator.iloc[-1]:.4f}")
```

### 信号历史记录

```python
# 记录历史信号
signal_history = []

def record_signals(signals):
    signal_history.extend(signals)
    return signals

# 使用装饰器记录信号
original_generate_signals = strategy.generate_signals

def logged_generate_signals(current_time, current_data):
    signals = original_generate_signals(current_time, current_data)
    record_signals(signals)
    return signals

strategy.generate_signals = logged_generate_signals

# 查看信号统计
def analyze_signals():
    if not signal_history:
        print("暂无交易信号记录")
        return

    buy_signals = [s for s in signal_history if s.signal_type == SignalType.BUY]
    sell_signals = [s for s in signal_history if s.signal_type == SignalType.SELL]

    print(f"总信号数: {len(signal_history)}")
    print(f"买入信号: {len(buy_signals)}")
    print(f"卖出信号: {len(sell_signals)}")

    # 按股票统计
    from collections import Counter
    symbol_counts = Counter([s.symbol for s in signal_history])
    print("各股票信号数量:", dict(symbol_counts))
```

## ⚠️ 注意事项

### 1. 数据质量
- 确保输入数据包含必要的OHLC列
- 数据应按时间顺序排列
- 处理缺失值和异常值

### 2. 策略开发
- 避免未来函数（使用未来数据）
- 考虑交易成本和滑点
- 进行充分的历史回测

### 3. 风险管理
- 设置止损和止盈条件
- 控制单笔交易仓位
- 考虑相关性风险

### 4. 性能优化
- 避免在generate_signals中进行重复计算
- 使用向量化操作
- 合理使用缓存机制

## 📖 API 参考

### BaseStrategy 方法

| 方法 | 说明 | 参数 |
|------|------|------|
| `__init__(symbols, **kwargs)` | 初始化策略 | symbols: 交易品种列表 |
| `set_data(data)` | 设置历史数据 | data: 股票数据字典 |
| `initialize()` | 初始化策略指标 | 无 |
| `generate_signals(time, data)` | 生成交易信号 | time: 当前时间, data: 当前数据 |
| `add_indicator(symbol, name, values)` | 添加技术指标 | symbol: 股票代码, name: 指标名, values: 指标值 |
| `get_indicators(symbol)` | 获取指标数据 | symbol: 股票代码 |

### 便捷创建函数

| 函数 | 说明 | 参数 |
|------|------|------|
| `create_ma_cross_strategy()` | 创建均线交叉策略 | symbols, short_window, long_window |
| `create_rsi_strategy()` | 创建RSI策略 | symbols, period, oversold, overbought |

## 🤝 贡献

欢迎提交问题和改进建议！请遵循项目的代码风格和测试要求。

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。