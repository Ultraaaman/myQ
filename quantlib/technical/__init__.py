"""
技术指标分析模块 (Technical Analysis Module)

本模块提供全面的技术指标分析功能，包括：
- 趋势指标：MA、EMA、MACD、布林带等
- 震荡指标：RSI、KDJ、威廉指标等  
- 成交量指标：OBV、成交量价格趋势等
- 支撑阻力：关键价位分析
"""

from .base import TechnicalIndicator
from .trend import TrendIndicators
from .oscillator import OscillatorIndicators
from .volume import VolumeIndicators
from .analyzer import TechnicalAnalyzer

__all__ = [
    'TechnicalIndicator',
    'TrendIndicators', 
    'OscillatorIndicators',
    'VolumeIndicators',
    'TechnicalAnalyzer'
]

__version__ = '1.0.0'