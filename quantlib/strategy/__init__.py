"""
策略模块 (Strategy Module)

本模块提供量化交易策略的基础框架，包括：
- 策略基类定义
- 常用策略实现
- 信号生成系统
- 策略组合管理
"""

from .base import (
    BaseStrategy,
    SimpleMovingAverageStrategy,
    Position,
    TradingSignal,
    SignalType,
    OrderType
)

from .examples import (
    MovingAverageCrossStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    MACDStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    MultiFactorStrategy,
    create_ma_cross_strategy,
    create_rsi_strategy,
    create_bollinger_bands_strategy,
    create_macd_strategy,
    create_momentum_strategy,
    create_mean_reversion_strategy,
    create_multi_factor_strategy
)

__all__ = [
    # 基础类
    'BaseStrategy',
    'SimpleMovingAverageStrategy',
    'Position',
    'TradingSignal',
    'SignalType',
    'OrderType',

    # 策略示例
    'MovingAverageCrossStrategy',
    'RSIStrategy',
    'BollingerBandsStrategy',
    'MACDStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'MultiFactorStrategy',

    # 便捷函数
    'create_ma_cross_strategy',
    'create_rsi_strategy',
    'create_bollinger_bands_strategy',
    'create_macd_strategy',
    'create_momentum_strategy',
    'create_mean_reversion_strategy',
    'create_multi_factor_strategy'
]

__version__ = '1.0.0'