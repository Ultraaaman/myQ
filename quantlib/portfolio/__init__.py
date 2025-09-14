"""
投资组合管理模块 (Portfolio Module)

本模块提供完整的投资组合管理功能，包括：
- 投资组合管理器
- 风险控制
- 资产配置
- 绩效评估
- 再平衡策略
"""

from .manager import (
    PortfolioManager,
    PortfolioPosition,
    RiskModel,
    create_portfolio_manager
)

__all__ = [
    'PortfolioManager',
    'PortfolioPosition',
    'RiskModel',
    'create_portfolio_manager'
]

__version__ = '1.0.0'