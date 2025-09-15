"""
投资组合管理模块 (Portfolio Module)

本模块提供完整的投资组合管理功能，包括：
- 投资组合管理器 (支持Live和Backtest模式)
- 因子投资策略
- 统一策略执行框架
- 风险控制
- 资产配置
- 绩效评估
- 再平衡策略
- 与Strategy和Backtest模块的无缝集成
"""

from .manager import (
    PortfolioManager,
    PortfolioPosition,
    RiskModel,
    FactorPortfolioManager,
    create_portfolio_manager,
    create_factor_portfolio_manager
)

from .strategy_executor import (
    StrategyExecutor,
    ExecutionMode,
    StrategyType,
    create_strategy_executor,
    create_factor_executor
)

__all__ = [
    # 基础组合管理
    'PortfolioManager',
    'PortfolioPosition', 
    'RiskModel',
    'create_portfolio_manager',
    
    # 因子投资组合管理
    'FactorPortfolioManager',
    'create_factor_portfolio_manager',
    
    # 策略执行框架
    'StrategyExecutor',
    'ExecutionMode',
    'StrategyType', 
    'create_strategy_executor',
    'create_factor_executor'
]

__version__ = '1.0.0'