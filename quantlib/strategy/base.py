"""
策略基类模块 (Base Strategy Module)

定义量化交易策略的基础框架和接口
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, date
from enum import Enum

class SignalType(Enum):
    """交易信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"  # 平仓

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"      # 市价单
    LIMIT = "limit"        # 限价单
    STOP = "stop"          # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单

class Position:
    """持仓信息"""
    def __init__(self, symbol: str, quantity: float, entry_price: float,
                 entry_time: datetime, order_type: OrderType = OrderType.MARKET):
        self.symbol = symbol
        self.quantity = quantity  # 正数为多头，负数为空头
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.order_type = order_type
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

    @property
    def market_value(self) -> float:
        """当前市值"""
        return self.quantity * self.current_price

    @property
    def is_long(self) -> bool:
        """是否为多头持仓"""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """是否为空头持仓"""
        return self.quantity < 0

    def update_price(self, price: float):
        """更新当前价格"""
        self.current_price = price
        if self.is_long:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * abs(self.quantity)

class TradingSignal:
    """交易信号"""
    def __init__(self, symbol: str, signal_type: SignalType,
                 timestamp: datetime, price: Optional[float] = None,
                 quantity: Optional[float] = None, order_type: OrderType = OrderType.MARKET,
                 stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                 confidence: float = 1.0, metadata: Optional[Dict] = None):
        self.symbol = symbol
        self.signal_type = signal_type
        self.timestamp = timestamp
        self.price = price
        self.quantity = quantity
        self.order_type = order_type
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence
        self.metadata = metadata or {}

class BaseStrategy(ABC):
    """
    策略基类

    所有交易策略都应该继承此类并实现必要的抽象方法
    """

    def __init__(self, name: str, symbols: List[str], initial_capital: float = 100000.0):
        self.name = name
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # 持仓管理
        self.positions: Dict[str, Position] = {}
        self.historical_positions: List[Position] = []

        # 数据存储
        self.data: Dict[str, pd.DataFrame] = {}
        self.indicators: Dict[str, Dict[str, pd.Series]] = {}

        # 交易记录
        self.signals: List[TradingSignal] = []
        self.trades: List[Dict] = []

        # 策略参数
        self.parameters: Dict[str, Any] = {}

        # 风控参数
        self.max_position_size: float = 0.1  # 单仓位最大占比
        self.stop_loss_pct: Optional[float] = None  # 止损百分比
        self.take_profit_pct: Optional[float] = None  # 止盈百分比

        # 状态标识
        self.is_initialized = False

    def set_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        """设置策略数据"""
        if isinstance(data, pd.DataFrame):
            if len(self.symbols) == 1:
                self.data[self.symbols[0]] = data.copy()
            else:
                raise ValueError("Multiple symbols but single DataFrame provided")
        elif isinstance(data, dict):
            self.data = {k: v.copy() for k, v in data.items()}
        else:
            raise ValueError("Data must be DataFrame or dict of DataFrames")

        # 确保数据包含必要的列
        for symbol, df in self.data.items():
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns for {symbol}: {missing_cols}")

    def set_parameters(self, **params):
        """设置策略参数"""
        self.parameters.update(params)
        return self

    def add_indicator(self, symbol: str, name: str, values: Union[pd.Series, np.ndarray]):
        """添加技术指标"""
        if symbol not in self.indicators:
            self.indicators[symbol] = {}

        if isinstance(values, np.ndarray):
            values = pd.Series(values)

        self.indicators[symbol][name] = values

    @abstractmethod
    def initialize(self):
        """
        策略初始化方法

        在回测开始前调用，用于：
        - 计算技术指标
        - 设置策略参数
        - 进行必要的准备工作
        """
        pass

    @abstractmethod
    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """
        生成交易信号

        Args:
            current_time: 当前时间
            current_data: 当前时刻各股票的数据

        Returns:
            交易信号列表
        """
        pass

    def on_bar(self, current_time: datetime, current_data: Dict[str, pd.Series]):
        """
        每个时间步调用的方法

        Args:
            current_time: 当前时间
            current_data: 当前时刻各股票的数据
        """
        # 更新持仓价格
        for symbol, position in self.positions.items():
            if symbol in current_data:
                position.update_price(current_data[symbol]['close'])

        # 生成交易信号
        signals = self.generate_signals(current_time, current_data)

        # 执行交易信号
        for signal in signals:
            self.execute_signal(signal, current_data)

    def execute_signal(self, signal: TradingSignal, current_data: Dict[str, pd.Series]):
        """执行交易信号"""
        symbol = signal.symbol

        if symbol not in current_data:
            return

        current_price = current_data[symbol]['close']
        execution_price = signal.price or current_price

        if signal.signal_type == SignalType.BUY:
            self._execute_buy(signal, execution_price)
        elif signal.signal_type == SignalType.SELL:
            self._execute_sell(signal, execution_price)
        elif signal.signal_type == SignalType.CLOSE:
            self._execute_close(signal, execution_price)

        # 记录信号
        self.signals.append(signal)

    def _execute_buy(self, signal: TradingSignal, price: float):
        """执行买入信号"""
        symbol = signal.symbol

        # 计算买入数量
        if signal.quantity:
            quantity = signal.quantity
        else:
            # 根据最大仓位比例计算
            max_value = self.current_capital * self.max_position_size
            quantity = int(max_value / price)

        if quantity <= 0:
            return

        # 检查资金是否充足
        required_capital = quantity * price
        if required_capital > self.current_capital:
            return

        # 创建或更新持仓
        if symbol in self.positions:
            # 加仓
            old_position = self.positions[symbol]
            new_quantity = old_position.quantity + quantity
            new_avg_price = (old_position.quantity * old_position.entry_price +
                           quantity * price) / new_quantity
            old_position.quantity = new_quantity
            old_position.entry_price = new_avg_price
        else:
            # 新开仓
            position = Position(symbol, quantity, price, signal.timestamp)
            self.positions[symbol] = position

        # 更新资金
        self.current_capital -= required_capital

        # 记录交易
        self.trades.append({
            'timestamp': signal.timestamp,
            'symbol': symbol,
            'action': 'buy',
            'quantity': quantity,
            'price': price,
            'value': required_capital,
            'remaining_capital': self.current_capital
        })

    def _execute_sell(self, signal: TradingSignal, price: float):
        """执行卖出信号"""
        symbol = signal.symbol

        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # 计算卖出数量
        if signal.quantity:
            sell_quantity = min(signal.quantity, position.quantity)
        else:
            sell_quantity = position.quantity

        if sell_quantity <= 0:
            return

        # 计算收益
        sell_value = sell_quantity * price
        cost_basis = sell_quantity * position.entry_price
        realized_pnl = sell_value - cost_basis

        # 更新持仓
        position.quantity -= sell_quantity
        position.realized_pnl += realized_pnl

        # 更新资金
        self.current_capital += sell_value

        # 如果完全平仓，移除持仓
        if position.quantity == 0:
            self.historical_positions.append(position)
            del self.positions[symbol]

        # 记录交易
        self.trades.append({
            'timestamp': signal.timestamp,
            'symbol': symbol,
            'action': 'sell',
            'quantity': sell_quantity,
            'price': price,
            'value': sell_value,
            'pnl': realized_pnl,
            'remaining_capital': self.current_capital
        })

    def _execute_close(self, signal: TradingSignal, price: float):
        """执行平仓信号"""
        signal.quantity = None  # 全部平仓
        self._execute_sell(signal, price)

    def get_portfolio_value(self) -> float:
        """获取组合总价值"""
        total_value = self.current_capital
        for position in self.positions.values():
            total_value += position.market_value
        return total_value

    def get_positions_summary(self) -> pd.DataFrame:
        """获取持仓摘要"""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for symbol, position in self.positions.items():
            data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
            })

        return pd.DataFrame(data)

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取策略表现指标"""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)

        # 计算基本指标
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['action'] == 'buy'])
        sell_trades = len(trades_df[trades_df['action'] == 'sell'])

        # 计算收益率
        current_value = self.get_portfolio_value()
        total_return = (current_value - self.initial_capital) / self.initial_capital

        # 计算已实现收益
        realized_trades = trades_df[trades_df['action'] == 'sell']
        total_realized_pnl = realized_trades['pnl'].sum() if 'pnl' in realized_trades.columns else 0

        # 胜率统计
        winning_trades = realized_trades[realized_trades.get('pnl', 0) > 0] if 'pnl' in realized_trades.columns else pd.DataFrame()
        win_rate = len(winning_trades) / len(realized_trades) if len(realized_trades) > 0 else 0

        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'portfolio_value': current_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'realized_pnl': total_realized_pnl,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'positions_count': len(self.positions)
        }

    def get_trades_history(self) -> pd.DataFrame:
        """获取交易历史"""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def reset(self):
        """重置策略状态"""
        self.current_capital = self.initial_capital
        self.positions.clear()
        self.historical_positions.clear()
        self.signals.clear()
        self.trades.clear()
        self.is_initialized = False

class SimpleMovingAverageStrategy(BaseStrategy):
    """
    简单移动平均策略示例

    当短期均线上穿长期均线时买入，下穿时卖出
    """

    def __init__(self, symbols: List[str], short_window: int = 20, long_window: int = 60, **kwargs):
        super().__init__("Simple Moving Average Strategy", symbols, **kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def initialize(self):
        """计算移动平均线指标"""
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            df = self.data[symbol]

            # 计算短期和长期移动平均线
            short_ma = df['close'].rolling(window=self.short_window).mean()
            long_ma = df['close'].rolling(window=self.long_window).mean()

            # 添加指标
            self.add_indicator(symbol, 'short_ma', short_ma)
            self.add_indicator(symbol, 'long_ma', long_ma)

            # 计算交叉信号
            ma_diff = short_ma - long_ma
            ma_diff_prev = ma_diff.shift(1)

            # 金叉和死叉
            golden_cross = (ma_diff > 0) & (ma_diff_prev <= 0)
            death_cross = (ma_diff < 0) & (ma_diff_prev >= 0)

            self.add_indicator(symbol, 'golden_cross', golden_cross)
            self.add_indicator(symbol, 'death_cross', death_cross)

        self.is_initialized = True

    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []

        if not self.is_initialized:
            return signals

        for symbol in self.symbols:
            if symbol not in current_data or symbol not in self.indicators:
                continue

            # 获取当前时间对应的指标值
            try:
                # 将current_time转换为字符串进行匹配
                time_str = current_time.strftime('%Y-%m-%d')

                indicators = self.indicators[symbol]

                # 查找最接近current_time的索引
                data_df = self.data[symbol]
                if 'date' in data_df.columns:
                    # 如果有date列，转换为字符串进行匹配
                    date_strs = data_df['date'].dt.strftime('%Y-%m-%d')
                    matching_indices = date_strs[date_strs == time_str].index
                else:
                    # 如果索引是时间，直接匹配
                    matching_indices = data_df.index[data_df.index.strftime('%Y-%m-%d') == time_str]

                if len(matching_indices) == 0:
                    continue

                idx = matching_indices[0]

                # 检查金叉信号（买入）
                if 'golden_cross' in indicators and idx in indicators['golden_cross'].index:
                    if indicators['golden_cross'].loc[idx]:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            timestamp=current_time,
                            confidence=0.8
                        )
                        signals.append(signal)

                # 检查死叉信号（卖出）
                if 'death_cross' in indicators and idx in indicators['death_cross'].index:
                    if indicators['death_cross'].loc[idx] and symbol in self.positions:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            timestamp=current_time,
                            confidence=0.8
                        )
                        signals.append(signal)

            except (KeyError, IndexError, ValueError):
                # 处理索引错误，继续处理下一个股票
                continue

        return signals