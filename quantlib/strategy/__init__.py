"""
策略模块 (Strategy Module)

本模块提供量化交易策略的基础框架，包括：
- 策略基类定义
- 技术分析策略
- 因子投资策略  
- 基本面策略
- 信号生成系统
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

from .factor_strategies import (
    FactorType,
    SingleFactorStrategy,
    MultiFactorStrategy as FactorMultiFactorStrategy,
    create_factor_strategy,
    create_multi_factor_strategy as create_factor_multi_strategy
)

__all__ = [
    # 基础类
    'BaseStrategy',
    'SimpleMovingAverageStrategy',
    'Position',
    'TradingSignal',
    'SignalType',
    'OrderType',

    # 技术分析策略
    'MovingAverageCrossStrategy',
    'RSIStrategy',
    'BollingerBandsStrategy',
    'MACDStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'MultiFactorStrategy',

    # 因子投资策略
    'FactorType',
    'SingleFactorStrategy',
    'FactorMultiFactorStrategy',

    # 便捷函数 - 技术分析
    'create_ma_cross_strategy',
    'create_rsi_strategy',
    'create_bollinger_bands_strategy',
    'create_macd_strategy',
    'create_momentum_strategy',
    'create_mean_reversion_strategy',
    'create_multi_factor_strategy',
    
    # 便捷函数 - 因子策略
    'create_factor_strategy',
    'create_factor_multi_strategy'
]

__version__ = '1.0.0'