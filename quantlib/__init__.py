"""
quantlib - 全面的量化金融分析库

提供完整的量化投资工具链，包括：
- market_data: 统一的市场数据接口
- fundamental: 基本面分析
- technical: 技术分析
- visualization: 数据可视化
- strategy: 策略开发（待实现）
- backtest: 策略回测（待实现） 
- portfolio: 投资组合管理（待实现）
- risk: 风险管理（待实现）
- screener: 股票筛选（待实现）
- optimization: 投资组合优化（待实现）
"""

from . import fundamental
from . import technical  
from . import market_data
from . import visualization
from . import strategy
from . import backtest
from . import portfolio
from . import risk
from . import screener
from . import optimization

__version__ = '1.0.0'
__all__ = [
    'market_data',    # 核心数据模块
    'fundamental', 
    'technical',
    'visualization',
    'strategy',       # 策略模块（待实现）
    'backtest',       # 回测模块（待实现）
    'portfolio',      # 投资组合模块（待实现）
    'risk',          # 风险管理模块（待实现）
    'screener',      # 股票筛选模块（待实现）
    'optimization'   # 投资组合优化模块（待实现）
]