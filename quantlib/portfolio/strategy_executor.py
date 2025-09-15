"""
统一策略执行框架 (Strategy Execution Framework)

提供Portfolio模块与Strategy模块、Backtest模块的统一集成接口
支持多种策略类型的统一管理和执行
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date
from enum import Enum
import warnings

from ..strategy.base import BaseStrategy, SignalType, TradingSignal
from ..backtest.backtrader_engine import BacktraderEngine, run_strategy_backtest
from .manager import PortfolioManager
from .manager import FactorPortfolioManager


class ExecutionMode(Enum):
    """执行模式"""
    LIVE = "live"                # 实时交易模式
    BACKTEST = "backtest"        # 历史回测模式
    SIMULATION = "simulation"    # 模拟交易模式
    PAPER = "paper"             # 纸上交易模式


class StrategyType(Enum):
    """策略类型"""
    TECHNICAL = "technical"      # 技术分析策略
    FUNDAMENTAL = "fundamental"  # 基本面策略
    FACTOR = "factor"           # 因子策略
    QUANTITATIVE = "quantitative"  # 量化策略
    MULTI_STRATEGY = "multi_strategy"  # 多策略组合


class StrategyExecutor:
    """
    统一策略执行器
    
    整合Portfolio、Strategy、Backtest模块，提供统一的策略执行接口
    """
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.LIVE, 
                 initial_capital: float = 1000000.0):
        self.mode = mode
        self.initial_capital = initial_capital
        
        # 创建组合管理器（根据模式选择类型）
        if mode == ExecutionMode.BACKTEST:
            self.portfolio = PortfolioManager(initial_capital, "Strategy Portfolio", "backtest")
        else:
            self.portfolio = PortfolioManager(initial_capital, "Strategy Portfolio", "live")
        
        # 策略管理
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_types: Dict[str, StrategyType] = {}
        
        # 回测引擎（仅在回测模式下使用）
        self.backtest_engine = None
        if mode == ExecutionMode.BACKTEST:
            try:
                self.backtest_engine = BacktraderEngine(initial_capital)
            except:
                warnings.warn("Backtest engine initialization failed")
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
    def add_strategy(self, name: str, strategy: BaseStrategy, weight: float = 1.0, 
                    strategy_type: StrategyType = StrategyType.QUANTITATIVE) -> bool:
        """
        添加策略到执行器
        
        Args:
            name: 策略名称
            strategy: 策略实例
            weight: 策略权重
            strategy_type: 策略类型
            
        Returns:
            是否成功添加
        """
        if name in self.strategies:
            warnings.warn(f"Strategy '{name}' already exists, replacing...")
        
        self.strategies[name] = strategy
        self.strategy_weights[name] = weight
        self.strategy_types[name] = strategy_type
        
        # 同时添加到portfolio管理器
        self.portfolio.add_strategy(name, strategy, weight)
        
        return True
    
    def remove_strategy(self, name: str) -> bool:
        """移除策略"""
        if name not in self.strategies:
            return False
        
        del self.strategies[name]
        del self.strategy_weights[name]
        del self.strategy_types[name]
        
        self.portfolio.remove_strategy(name)
        
        return True
    
    def set_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        """设置所有策略的数据"""
        for strategy in self.strategies.values():
            if hasattr(strategy, 'set_data'):
                strategy.set_data(data)
    
    def initialize_strategies(self):
        """初始化所有策略"""
        for name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'initialize') and not strategy.is_initialized:
                    strategy.initialize()
            except Exception as e:
                warnings.warn(f"Failed to initialize strategy '{name}': {e}")
    
    def execute_single_step(self, current_time: datetime, current_data: Dict[str, pd.Series],
                          prices: Dict[str, float] = None) -> Dict[str, Any]:
        """
        执行单个时间步的策略
        
        Args:
            current_time: 当前时间
            current_data: 当前市场数据
            prices: 当前价格（用于交易执行）
            
        Returns:
            执行结果
        """
        if prices is None:
            prices = {symbol: data['close'] for symbol, data in current_data.items()}
        
        # 更新组合价格
        self.portfolio.update_prices(prices)
        
        all_signals = []
        strategy_signals = {}
        
        # 收集所有策略信号
        for name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'generate_signals'):
                    signals = strategy.generate_signals(current_time, current_data)
                    strategy_signals[name] = signals
                    
                    # 按权重调整信号
                    weight = self.strategy_weights[name]
                    for signal in signals:
                        signal.confidence *= weight
                    
                    all_signals.extend(signals)
                    
            except Exception as e:
                warnings.warn(f"Error generating signals for strategy '{name}': {e}")
        
        # 信号聚合和去重
        aggregated_signals = self._aggregate_signals(all_signals)
        
        # 执行交易
        executed_trades = []
        for signal in aggregated_signals:
            if self._should_execute_signal(signal, prices):
                trade_result = self._execute_signal(signal, prices)
                if trade_result:
                    executed_trades.append(trade_result)
        
        # 记录执行结果
        execution_result = {
            'timestamp': current_time,
            'total_signals': len(all_signals),
            'aggregated_signals': len(aggregated_signals),
            'executed_trades': len(executed_trades),
            'portfolio_value': self.portfolio.get_total_value(),
            'cash': self.portfolio.current_cash,
            'positions_count': len(self.portfolio.positions),
            'strategy_signals': {name: len(signals) for name, signals in strategy_signals.items()}
        }
        
        self.execution_history.append(execution_result)
        
        return execution_result
    
    def _aggregate_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """聚合多个策略的信号"""
        # 按股票分组
        signals_by_symbol = {}
        for signal in signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)
        
        aggregated = []
        
        for symbol, symbol_signals in signals_by_symbol.items():
            # 按信号类型分组
            buy_signals = [s for s in symbol_signals if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in symbol_signals if s.signal_type == SignalType.SELL]
            
            # 聚合买入信号
            if buy_signals:
                total_confidence = sum(s.confidence for s in buy_signals)
                avg_confidence = total_confidence / len(buy_signals)
                
                if total_confidence > 0.5:  # 阈值可配置
                    aggregated_signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        timestamp=buy_signals[0].timestamp,
                        confidence=min(avg_confidence, 1.0),
                        metadata={
                            'source_strategies': [s.metadata.get('strategy_name', 'unknown') for s in buy_signals],
                            'signal_count': len(buy_signals),
                            'total_confidence': total_confidence
                        }
                    )
                    aggregated.append(aggregated_signal)
            
            # 聚合卖出信号
            if sell_signals:
                total_confidence = sum(s.confidence for s in sell_signals)
                avg_confidence = total_confidence / len(sell_signals)
                
                if total_confidence > 0.5:  # 阈值可配置
                    aggregated_signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        timestamp=sell_signals[0].timestamp,
                        confidence=min(avg_confidence, 1.0),
                        metadata={
                            'source_strategies': [s.metadata.get('strategy_name', 'unknown') for s in sell_signals],
                            'signal_count': len(sell_signals),
                            'total_confidence': total_confidence
                        }
                    )
                    aggregated.append(aggregated_signal)
        
        return aggregated
    
    def _should_execute_signal(self, signal: TradingSignal, prices: Dict[str, float]) -> bool:
        """判断是否应该执行信号"""
        # 基本检查
        if signal.symbol not in prices:
            return False
        
        # 置信度检查
        if signal.confidence < 0.6:  # 可配置阈值
            return False
        
        # 风险控制检查
        current_value = self.portfolio.get_total_value()
        max_trade_value = current_value * 0.1  # 单笔交易不超过10%
        
        if signal.signal_type == SignalType.BUY:
            # 检查是否有足够资金
            if self.portfolio.current_cash < max_trade_value:
                return False
        elif signal.signal_type == SignalType.SELL:
            # 检查是否有持仓
            if signal.symbol not in self.portfolio.positions:
                return False
        
        return True
    
    def _execute_signal(self, signal: TradingSignal, prices: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """执行交易信号"""
        price = prices.get(signal.symbol)
        if not price:
            return None
        
        try:
            if signal.signal_type == SignalType.BUY:
                # 计算买入数量（基于可用资金的一定比例）
                available_cash = self.portfolio.current_cash
                max_investment = available_cash * 0.1  # 每次最多投入10%的资金
                quantity = int(max_investment / price)
                
                if quantity > 0:
                    success = self.portfolio.buy(signal.symbol, quantity, price=price)
                    if success:
                        return {
                            'action': 'buy',
                            'symbol': signal.symbol,
                            'quantity': quantity,
                            'price': price,
                            'confidence': signal.confidence
                        }
            
            elif signal.signal_type == SignalType.SELL:
                if signal.symbol in self.portfolio.positions:
                    position = self.portfolio.positions[signal.symbol]
                    # 卖出一定比例的持仓
                    sell_ratio = min(signal.confidence, 0.5)  # 最多卖出50%
                    quantity = int(position.quantity * sell_ratio)
                    
                    if quantity > 0:
                        success = self.portfolio.sell(signal.symbol, quantity, price=price)
                        if success:
                            return {
                                'action': 'sell',
                                'symbol': signal.symbol,
                                'quantity': quantity,
                                'price': price,
                                'confidence': signal.confidence
                            }
        
        except Exception as e:
            warnings.warn(f"Failed to execute signal for {signal.symbol}: {e}")
        
        return None
    
    def run_backtest(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            data: 历史数据
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果
        """
        if self.mode != ExecutionMode.BACKTEST:
            raise ValueError("Executor must be in BACKTEST mode to run backtests")
        
        # 设置数据
        self.set_data(data)
        self.initialize_strategies()
        
        # 使用Portfolio的回测功能
        portfolio_result = self.portfolio.run_backtest(data, start_date, end_date)
        
        # 如果有backtest引擎，也运行详细回测
        detailed_results = {}
        if self.backtest_engine:
            for name, strategy in self.strategies.items():
                try:
                    strategy_result = self.backtest_engine.run_backtest(strategy, data)
                    detailed_results[name] = strategy_result
                except Exception as e:
                    warnings.warn(f"Detailed backtest failed for strategy '{name}': {e}")
        
        # 整合结果
        final_result = {
            'portfolio_result': portfolio_result,
            'detailed_results': detailed_results,
            'execution_history': self.execution_history,
            'final_portfolio_value': self.portfolio.get_total_value(),
            'total_return': (self.portfolio.get_total_value() - self.initial_capital) / self.initial_capital,
            'strategies_count': len(self.strategies),
            'total_trades': len(self.portfolio.trades)
        }
        
        return final_result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取执行器的性能摘要"""
        portfolio_metrics = self.portfolio.get_performance_metrics()
        
        summary = {
            'portfolio_metrics': portfolio_metrics,
            'strategies_count': len(self.strategies),
            'execution_mode': self.mode.value,
            'total_execution_steps': len(self.execution_history),
            'strategy_weights': self.strategy_weights.copy(),
            'strategy_types': {name: stype.value for name, stype in self.strategy_types.items()}
        }
        
        # 添加策略级别的统计
        if self.execution_history:
            recent_executions = self.execution_history[-10:]  # 最近10次执行
            avg_signals = np.mean([e['total_signals'] for e in recent_executions])
            avg_executed = np.mean([e['executed_trades'] for e in recent_executions])
            
            summary['recent_performance'] = {
                'avg_signals_per_step': avg_signals,
                'avg_trades_per_step': avg_executed,
                'signal_execution_rate': avg_executed / avg_signals if avg_signals > 0 else 0
            }
        
        return summary
    
    def print_summary(self):
        """打印执行器摘要"""
        summary = self.get_performance_summary()
        
        print(f"📊 策略执行器摘要")
        print(f"{'='*50}")
        print(f"执行模式: {summary['execution_mode']}")
        print(f"策略数量: {summary['strategies_count']}")
        print(f"执行步骤: {summary['total_execution_steps']}")
        print(f"组合价值: ${summary['portfolio_metrics'].get('current_value', 0):,.2f}")
        print(f"总收益率: {summary['portfolio_metrics'].get('total_return_pct', 0):.2f}%")
        
        print(f"\n📈 策略配置:")
        for name, weight in summary['strategy_weights'].items():
            strategy_type = summary['strategy_types'][name]
            print(f"  {name}: {weight:.1%} ({strategy_type})")
        
        if 'recent_performance' in summary:
            perf = summary['recent_performance']
            print(f"\n⚡ 最近执行表现:")
            print(f"  平均信号数: {perf['avg_signals_per_step']:.1f}")
            print(f"  平均执行数: {perf['avg_trades_per_step']:.1f}")
            print(f"  信号执行率: {perf['signal_execution_rate']:.1%}")


def create_strategy_executor(mode: str = "live", initial_capital: float = 1000000.0) -> StrategyExecutor:
    """
    创建策略执行器的便捷函数
    
    Args:
        mode: 执行模式 ("live", "backtest", "simulation", "paper")
        initial_capital: 初始资金
        
    Returns:
        StrategyExecutor实例
    """
    execution_mode = ExecutionMode(mode.lower())
    return StrategyExecutor(execution_mode, initial_capital)


def create_factor_executor(initial_capital: float = 1000000.0, mode: str = "live") -> StrategyExecutor:
    """
    创建专门用于因子策略的执行器
    
    Args:
        initial_capital: 初始资金
        mode: 执行模式
        
    Returns:
        配置好的StrategyExecutor实例
    """
    executor = create_strategy_executor(mode, initial_capital)
    
    # 替换为因子投资组合管理器
    if mode == "backtest":
        executor.portfolio = FactorPortfolioManager(initial_capital, "Factor Portfolio", "backtest")
    else:
        executor.portfolio = FactorPortfolioManager(initial_capital, "Factor Portfolio", "live")
    
    return executor