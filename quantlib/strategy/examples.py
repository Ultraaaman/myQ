"""
策略示例模块 (Strategy Examples Module)

包含多种常用的量化交易策略实现，包括：
- 移动平均策略
- 均值回归策略
- 动量策略
- 布林带策略
- RSI策略
- 多因子策略
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
import warnings

from .base import BaseStrategy, TradingSignal, SignalType
from ..technical import (
    calculate_ma, calculate_rsi, calculate_bollinger_bands,
    calculate_macd, calculate_stochastic
)

class MovingAverageCrossStrategy(BaseStrategy):
    """
    移动平均线交叉策略

    当短期均线上穿长期均线时买入，下穿时卖出
    """

    def __init__(self, symbols: List[str], short_window: int = 20, long_window: int = 60, **kwargs):
        super().__init__("Moving Average Cross Strategy", symbols, **kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def initialize(self):
        """初始化策略，计算技术指标"""
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            df = self.data[symbol]

            # 计算移动平均线
            short_ma = calculate_ma(df['close'], period=self.short_window)
            long_ma = calculate_ma(df['close'], period=self.long_window)

            self.add_indicator(symbol, 'short_ma', short_ma)
            self.add_indicator(symbol, 'long_ma', long_ma)

            # 计算交叉信号
            ma_diff = short_ma - long_ma
            ma_diff_prev = ma_diff.shift(1)

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

            try:
                # 查找匹配的时间索引
                data_df = self.data[symbol]
                time_str = current_time.strftime('%Y-%m-%d')

                if 'date' in data_df.columns:
                    date_strs = data_df['date'].dt.strftime('%Y-%m-%d')
                    matching_indices = date_strs[date_strs == time_str].index
                else:
                    matching_indices = data_df.index[data_df.index.strftime('%Y-%m-%d') == time_str]

                if len(matching_indices) == 0:
                    continue

                idx = matching_indices[0]
                indicators = self.indicators[symbol]

                # 金叉买入信号
                if 'golden_cross' in indicators and idx in indicators['golden_cross'].index:
                    if indicators['golden_cross'].loc[idx]:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            timestamp=current_time,
                            confidence=0.8,
                            metadata={'strategy': 'ma_cross', 'signal': 'golden_cross'}
                        )
                        signals.append(signal)

                # 死叉卖出信号
                if 'death_cross' in indicators and idx in indicators['death_cross'].index:
                    if indicators['death_cross'].loc[idx] and symbol in self.positions:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            timestamp=current_time,
                            confidence=0.8,
                            metadata={'strategy': 'ma_cross', 'signal': 'death_cross'}
                        )
                        signals.append(signal)

            except (KeyError, IndexError, ValueError):
                continue

        return signals

class RSIStrategy(BaseStrategy):
    """
    RSI相对强弱指标策略

    RSI低于超卖线时买入，高于超买线时卖出
    """

    def __init__(self, symbols: List[str], rsi_period: int = 14,
                 oversold_threshold: float = 30, overbought_threshold: float = 70, **kwargs):
        super().__init__("RSI Strategy", symbols, **kwargs)
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def initialize(self):
        """初始化策略"""
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            df = self.data[symbol]

            # 计算RSI
            rsi = calculate_rsi(df['close'], period=self.rsi_period)
            self.add_indicator(symbol, 'rsi', rsi)

            # 生成买卖信号
            buy_signal = (rsi < self.oversold_threshold) & (rsi.shift(1) >= self.oversold_threshold)
            sell_signal = (rsi > self.overbought_threshold) & (rsi.shift(1) <= self.overbought_threshold)

            self.add_indicator(symbol, 'rsi_buy', buy_signal)
            self.add_indicator(symbol, 'rsi_sell', sell_signal)

        self.is_initialized = True

    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []

        if not self.is_initialized:
            return signals

        for symbol in self.symbols:
            if symbol not in current_data or symbol not in self.indicators:
                continue

            try:
                data_df = self.data[symbol]
                time_str = current_time.strftime('%Y-%m-%d')

                if 'date' in data_df.columns:
                    date_strs = data_df['date'].dt.strftime('%Y-%m-%d')
                    matching_indices = date_strs[date_strs == time_str].index
                else:
                    matching_indices = data_df.index[data_df.index.strftime('%Y-%m-%d') == time_str]

                if len(matching_indices) == 0:
                    continue

                idx = matching_indices[0]
                indicators = self.indicators[symbol]

                # RSI买入信号
                if 'rsi_buy' in indicators and idx in indicators['rsi_buy'].index:
                    if indicators['rsi_buy'].loc[idx]:
                        current_rsi = indicators['rsi'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            timestamp=current_time,
                            confidence=0.7,
                            metadata={'strategy': 'rsi', 'rsi_value': current_rsi}
                        )
                        signals.append(signal)

                # RSI卖出信号
                if 'rsi_sell' in indicators and idx in indicators['rsi_sell'].index:
                    if indicators['rsi_sell'].loc[idx] and symbol in self.positions:
                        current_rsi = indicators['rsi'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            timestamp=current_time,
                            confidence=0.7,
                            metadata={'strategy': 'rsi', 'rsi_value': current_rsi}
                        )
                        signals.append(signal)

            except (KeyError, IndexError, ValueError):
                continue

        return signals

class BollingerBandsStrategy(BaseStrategy):
    """
    布林带策略

    价格跌破下轨时买入，突破上轨时卖出
    """

    def __init__(self, symbols: List[str], period: int = 20, std_dev: float = 2.0, **kwargs):
        super().__init__("Bollinger Bands Strategy", symbols, **kwargs)
        self.period = period
        self.std_dev = std_dev

    def initialize(self):
        """初始化策略"""
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            df = self.data[symbol]

            # 计算布林带
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
                df['close'], period=self.period, std_dev=self.std_dev
            )

            self.add_indicator(symbol, 'bb_upper', bb_upper)
            self.add_indicator(symbol, 'bb_middle', bb_middle)
            self.add_indicator(symbol, 'bb_lower', bb_lower)

            # 生成交易信号
            price = df['close']

            # 确保所有Series使用相同的索引
            common_index = price.index.intersection(bb_lower.index).intersection(bb_upper.index)
            price_aligned = price.loc[common_index]
            bb_upper_aligned = bb_upper.loc[common_index]
            bb_lower_aligned = bb_lower.loc[common_index]

            # 重置索引以确保比较操作正常工作
            price_aligned = price_aligned.reset_index(drop=True)
            bb_upper_aligned = bb_upper_aligned.reset_index(drop=True)
            bb_lower_aligned = bb_lower_aligned.reset_index(drop=True)

            buy_signal = (price_aligned < bb_lower_aligned) & (price_aligned.shift(1) >= bb_lower_aligned.shift(1))
            sell_signal = (price_aligned > bb_upper_aligned) & (price_aligned.shift(1) <= bb_upper_aligned.shift(1))

            # 恢复原始索引
            buy_signal.index = common_index
            sell_signal.index = common_index

            self.add_indicator(symbol, 'bb_buy', buy_signal)
            self.add_indicator(symbol, 'bb_sell', sell_signal)

        self.is_initialized = True

    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []

        if not self.is_initialized:
            return signals

        for symbol in self.symbols:
            if symbol not in current_data or symbol not in self.indicators:
                continue

            try:
                data_df = self.data[symbol]
                time_str = current_time.strftime('%Y-%m-%d')

                if 'date' in data_df.columns:
                    date_strs = data_df['date'].dt.strftime('%Y-%m-%d')
                    matching_indices = date_strs[date_strs == time_str].index
                else:
                    matching_indices = data_df.index[data_df.index.strftime('%Y-%m-%d') == time_str]

                if len(matching_indices) == 0:
                    continue

                idx = matching_indices[0]
                indicators = self.indicators[symbol]

                current_price = current_data[symbol]['close']

                # 布林带买入信号
                if 'bb_buy' in indicators and idx in indicators['bb_buy'].index:
                    if indicators['bb_buy'].loc[idx]:
                        bb_lower = indicators['bb_lower'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            timestamp=current_time,
                            confidence=0.75,
                            metadata={
                                'strategy': 'bollinger_bands',
                                'bb_lower': bb_lower,
                                'price': current_price
                            }
                        )
                        signals.append(signal)

                # 布林带卖出信号
                if 'bb_sell' in indicators and idx in indicators['bb_sell'].index:
                    if indicators['bb_sell'].loc[idx] and symbol in self.positions:
                        bb_upper = indicators['bb_upper'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            timestamp=current_time,
                            confidence=0.75,
                            metadata={
                                'strategy': 'bollinger_bands',
                                'bb_upper': bb_upper,
                                'price': current_price
                            }
                        )
                        signals.append(signal)

            except (KeyError, IndexError, ValueError):
                continue

        return signals

class MACDStrategy(BaseStrategy):
    """
    MACD策略

    MACD线上穿信号线时买入，下穿时卖出
    """

    def __init__(self, symbols: List[str], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, **kwargs):
        super().__init__("MACD Strategy", symbols, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def initialize(self):
        """初始化策略"""
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            df = self.data[symbol]

            # 计算MACD
            macd_line, signal_line, histogram = calculate_macd(
                df['close'], fast=self.fast_period, slow=self.slow_period, signal=self.signal_period
            )

            self.add_indicator(symbol, 'macd', macd_line)
            self.add_indicator(symbol, 'signal', signal_line)
            self.add_indicator(symbol, 'histogram', histogram)

            # 生成交易信号
            buy_signal = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
            sell_signal = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

            self.add_indicator(symbol, 'macd_buy', buy_signal)
            self.add_indicator(symbol, 'macd_sell', sell_signal)

        self.is_initialized = True

    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []

        if not self.is_initialized:
            return signals

        for symbol in self.symbols:
            if symbol not in current_data or symbol not in self.indicators:
                continue

            try:
                data_df = self.data[symbol]
                time_str = current_time.strftime('%Y-%m-%d')

                if 'date' in data_df.columns:
                    date_strs = data_df['date'].dt.strftime('%Y-%m-%d')
                    matching_indices = date_strs[date_strs == time_str].index
                else:
                    matching_indices = data_df.index[data_df.index.strftime('%Y-%m-%d') == time_str]

                if len(matching_indices) == 0:
                    continue

                idx = matching_indices[0]
                indicators = self.indicators[symbol]

                # MACD买入信号
                if 'macd_buy' in indicators and idx in indicators['macd_buy'].index:
                    if indicators['macd_buy'].loc[idx]:
                        macd_value = indicators['macd'].loc[idx]
                        signal_value = indicators['signal'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            timestamp=current_time,
                            confidence=0.75,
                            metadata={
                                'strategy': 'macd',
                                'macd_value': macd_value,
                                'signal_value': signal_value
                            }
                        )
                        signals.append(signal)

                # MACD卖出信号
                if 'macd_sell' in indicators and idx in indicators['macd_sell'].index:
                    if indicators['macd_sell'].loc[idx] and symbol in self.positions:
                        macd_value = indicators['macd'].loc[idx]
                        signal_value = indicators['signal'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            timestamp=current_time,
                            confidence=0.75,
                            metadata={
                                'strategy': 'macd',
                                'macd_value': macd_value,
                                'signal_value': signal_value
                            }
                        )
                        signals.append(signal)

            except (KeyError, IndexError, ValueError):
                continue

        return signals

class MomentumStrategy(BaseStrategy):
    """
    动量策略

    基于价格动量进行交易，价格上涨动量强时买入，下跌动量强时卖出
    """

    def __init__(self, symbols: List[str], lookback_period: int = 10, momentum_threshold: float = 0.02, **kwargs):
        super().__init__("Momentum Strategy", symbols, **kwargs)
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold

    def initialize(self):
        """初始化策略"""
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            df = self.data[symbol]

            # 计算价格动量
            price_momentum = df['close'].pct_change(self.lookback_period)
            self.add_indicator(symbol, 'momentum', price_momentum)

            # 计算成交量加权动量
            volume_momentum = (df['close'] * df['volume']).rolling(self.lookback_period).sum() / df['volume'].rolling(self.lookback_period).sum()
            volume_momentum_pct = volume_momentum.pct_change(self.lookback_period)
            self.add_indicator(symbol, 'volume_momentum', volume_momentum_pct)

            # 生成交易信号
            buy_signal = price_momentum > self.momentum_threshold
            sell_signal = price_momentum < -self.momentum_threshold

            self.add_indicator(symbol, 'momentum_buy', buy_signal)
            self.add_indicator(symbol, 'momentum_sell', sell_signal)

        self.is_initialized = True

    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []

        if not self.is_initialized:
            return signals

        for symbol in self.symbols:
            if symbol not in current_data or symbol not in self.indicators:
                continue

            try:
                data_df = self.data[symbol]
                time_str = current_time.strftime('%Y-%m-%d')

                if 'date' in data_df.columns:
                    date_strs = data_df['date'].dt.strftime('%Y-%m-%d')
                    matching_indices = date_strs[date_strs == time_str].index
                else:
                    matching_indices = data_df.index[data_df.index.strftime('%Y-%m-%d') == time_str]

                if len(matching_indices) == 0:
                    continue

                idx = matching_indices[0]
                indicators = self.indicators[symbol]

                # 动量买入信号
                if 'momentum_buy' in indicators and idx in indicators['momentum_buy'].index:
                    if indicators['momentum_buy'].loc[idx] and symbol not in self.positions:
                        momentum_value = indicators['momentum'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            timestamp=current_time,
                            confidence=min(0.9, abs(momentum_value) * 10),  # 动量越强信心越高
                            metadata={
                                'strategy': 'momentum',
                                'momentum_value': momentum_value
                            }
                        )
                        signals.append(signal)

                # 动量卖出信号
                if 'momentum_sell' in indicators and idx in indicators['momentum_sell'].index:
                    if indicators['momentum_sell'].loc[idx] and symbol in self.positions:
                        momentum_value = indicators['momentum'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            timestamp=current_time,
                            confidence=min(0.9, abs(momentum_value) * 10),
                            metadata={
                                'strategy': 'momentum',
                                'momentum_value': momentum_value
                            }
                        )
                        signals.append(signal)

            except (KeyError, IndexError, ValueError):
                continue

        return signals

class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略

    当价格偏离移动平均线过远时，预期价格会回归平均值
    """

    def __init__(self, symbols: List[str], window: int = 20, deviation_threshold: float = 0.05, **kwargs):
        super().__init__("Mean Reversion Strategy", symbols, **kwargs)
        self.window = window
        self.deviation_threshold = deviation_threshold

    def initialize(self):
        """初始化策略"""
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            df = self.data[symbol]

            # 计算移动平均线和标准差
            rolling_mean = df['close'].rolling(self.window).mean()
            rolling_std = df['close'].rolling(self.window).std()

            self.add_indicator(symbol, 'rolling_mean', rolling_mean)
            self.add_indicator(symbol, 'rolling_std', rolling_std)

            # 计算Z-score
            z_score = (df['close'] - rolling_mean) / rolling_std
            self.add_indicator(symbol, 'z_score', z_score)

            # 生成交易信号 (价格偏离均值过远时反向操作)
            buy_signal = z_score < -self.deviation_threshold  # 价格过低，买入
            sell_signal = z_score > self.deviation_threshold   # 价格过高，卖出

            self.add_indicator(symbol, 'mean_reversion_buy', buy_signal)
            self.add_indicator(symbol, 'mean_reversion_sell', sell_signal)

        self.is_initialized = True

    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []

        if not self.is_initialized:
            return signals

        for symbol in self.symbols:
            if symbol not in current_data or symbol not in self.indicators:
                continue

            try:
                data_df = self.data[symbol]
                time_str = current_time.strftime('%Y-%m-%d')

                if 'date' in data_df.columns:
                    date_strs = data_df['date'].dt.strftime('%Y-%m-%d')
                    matching_indices = date_strs[date_strs == time_str].index
                else:
                    matching_indices = data_df.index[data_df.index.strftime('%Y-%m-%d') == time_str]

                if len(matching_indices) == 0:
                    continue

                idx = matching_indices[0]
                indicators = self.indicators[symbol]

                # 均值回归买入信号
                if 'mean_reversion_buy' in indicators and idx in indicators['mean_reversion_buy'].index:
                    if indicators['mean_reversion_buy'].loc[idx]:
                        z_score = indicators['z_score'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            timestamp=current_time,
                            confidence=min(0.9, abs(z_score) * 0.3),  # Z-score越高信心越高
                            metadata={
                                'strategy': 'mean_reversion',
                                'z_score': z_score
                            }
                        )
                        signals.append(signal)

                # 均值回归卖出信号
                if 'mean_reversion_sell' in indicators and idx in indicators['mean_reversion_sell'].index:
                    if indicators['mean_reversion_sell'].loc[idx] and symbol in self.positions:
                        z_score = indicators['z_score'].loc[idx]
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            timestamp=current_time,
                            confidence=min(0.9, abs(z_score) * 0.3),
                            metadata={
                                'strategy': 'mean_reversion',
                                'z_score': z_score
                            }
                        )
                        signals.append(signal)

            except (KeyError, IndexError, ValueError):
                continue

        return signals

class MultiFactorStrategy(BaseStrategy):
    """
    多因子策略

    综合多个技术指标进行决策
    """

    def __init__(self, symbols: List[str], ma_short: int = 20, ma_long: int = 60,
                 rsi_period: int = 14, rsi_oversold: float = 30, rsi_overbought: float = 70, **kwargs):
        super().__init__("Multi-Factor Strategy", symbols, **kwargs)
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def initialize(self):
        """初始化策略"""
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            df = self.data[symbol]

            # 移动平均线
            short_ma = calculate_ma(df['close'], period=self.ma_short)
            long_ma = calculate_ma(df['close'], period=self.ma_long)
            self.add_indicator(symbol, 'short_ma', short_ma)
            self.add_indicator(symbol, 'long_ma', long_ma)

            # RSI
            rsi = calculate_rsi(df['close'], period=self.rsi_period)
            self.add_indicator(symbol, 'rsi', rsi)

            # MACD
            macd_line, signal_line, histogram = calculate_macd(df['close'])
            self.add_indicator(symbol, 'macd', macd_line)
            self.add_indicator(symbol, 'macd_signal', signal_line)

            # 综合评分 (多因子模型)
            ma_score = np.where(short_ma > long_ma, 1, -1)  # 均线趋势评分
            rsi_score = np.where(rsi < self.rsi_oversold, 1,  # RSI评分
                               np.where(rsi > self.rsi_overbought, -1, 0))
            macd_score = np.where(macd_line > signal_line, 1, -1)  # MACD评分

            # 综合评分
            composite_score = ma_score + rsi_score + macd_score
            self.add_indicator(symbol, 'composite_score', pd.Series(composite_score, index=df.index))

        self.is_initialized = True

    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []

        if not self.is_initialized:
            return signals

        for symbol in self.symbols:
            if symbol not in current_data or symbol not in self.indicators:
                continue

            try:
                data_df = self.data[symbol]
                time_str = current_time.strftime('%Y-%m-%d')

                if 'date' in data_df.columns:
                    date_strs = data_df['date'].dt.strftime('%Y-%m-%d')
                    matching_indices = date_strs[date_strs == time_str].index
                else:
                    matching_indices = data_df.index[data_df.index.strftime('%Y-%m-%d') == time_str]

                if len(matching_indices) == 0:
                    continue

                idx = matching_indices[0]
                indicators = self.indicators[symbol]

                if 'composite_score' in indicators and idx in indicators['composite_score'].index:
                    score = indicators['composite_score'].loc[idx]

                    # 强买入信号 (评分 >= 2)
                    if score >= 2 and symbol not in self.positions:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            timestamp=current_time,
                            confidence=min(0.95, 0.5 + score * 0.15),
                            metadata={
                                'strategy': 'multi_factor',
                                'composite_score': score,
                                'ma_trend': 'up' if indicators['short_ma'].loc[idx] > indicators['long_ma'].loc[idx] else 'down',
                                'rsi_value': indicators['rsi'].loc[idx]
                            }
                        )
                        signals.append(signal)

                    # 强卖出信号 (评分 <= -2)
                    elif score <= -2 and symbol in self.positions:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            timestamp=current_time,
                            confidence=min(0.95, 0.5 + abs(score) * 0.15),
                            metadata={
                                'strategy': 'multi_factor',
                                'composite_score': score,
                                'ma_trend': 'up' if indicators['short_ma'].loc[idx] > indicators['long_ma'].loc[idx] else 'down',
                                'rsi_value': indicators['rsi'].loc[idx]
                            }
                        )
                        signals.append(signal)

            except (KeyError, IndexError, ValueError):
                continue

        return signals

# 便捷函数用于创建策略
def create_ma_cross_strategy(symbols: List[str], short_window: int = 20, long_window: int = 60, **kwargs):
    """创建移动平均线交叉策略"""
    return MovingAverageCrossStrategy(symbols, short_window, long_window, **kwargs)

def create_rsi_strategy(symbols: List[str], rsi_period: int = 14, oversold: float = 30, overbought: float = 70, **kwargs):
    """创建RSI策略"""
    return RSIStrategy(symbols, rsi_period, oversold, overbought, **kwargs)

def create_bollinger_bands_strategy(symbols: List[str], period: int = 20, std_dev: float = 2.0, **kwargs):
    """创建布林带策略"""
    return BollingerBandsStrategy(symbols, period, std_dev, **kwargs)

def create_macd_strategy(symbols: List[str], fast: int = 12, slow: int = 26, signal: int = 9, **kwargs):
    """创建MACD策略"""
    return MACDStrategy(symbols, fast, slow, signal, **kwargs)

def create_momentum_strategy(symbols: List[str], lookback: int = 10, threshold: float = 0.02, **kwargs):
    """创建动量策略"""
    return MomentumStrategy(symbols, lookback, threshold, **kwargs)

def create_mean_reversion_strategy(symbols: List[str], window: int = 20, deviation: float = 0.05, **kwargs):
    """创建均值回归策略"""
    return MeanReversionStrategy(symbols, window, deviation, **kwargs)

def create_multi_factor_strategy(symbols: List[str], **kwargs):
    """创建多因子策略"""
    return MultiFactorStrategy(symbols, **kwargs)