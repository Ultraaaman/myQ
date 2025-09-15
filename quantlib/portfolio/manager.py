"""
投资组合管理模块 (Portfolio Manager)

提供投资组合管理功能，包括：
- 仓位管理
- 风险控制
- 资金分配
- 绩效评估
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum
import warnings

class RiskModel(Enum):
    """风险模型类型"""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MEAN_REVERSION = "mean_reversion"

class PortfolioPosition:
    """投资组合持仓"""

    def __init__(self, symbol: str, quantity: float, avg_cost: float,
                 current_price: float = None, weight: float = 0.0):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.current_price = current_price or avg_cost
        self.weight = weight
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

    @property
    def market_value(self) -> float:
        """当前市值"""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """成本基础"""
        return self.quantity * self.avg_cost

    @property
    def unrealized_return(self) -> float:
        """未实现收益率"""
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost

    def update_price(self, price: float):
        """更新当前价格"""
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_cost) * self.quantity

class PortfolioManager:
    """
    投资组合管理器

    负责管理投资组合的所有方面，包括仓位、风险、绩效等
    支持Live交易和Backtest模式的统一接口
    """

    def __init__(self, initial_capital: float = 1000000.0, name: str = "Portfolio", mode: str = "live"):
        self.name = name
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions: Dict[str, PortfolioPosition] = {}
        self.historical_positions: List[PortfolioPosition] = []
        
        # 添加模式支持
        self.mode = mode  # "live" or "backtest"
        self.backtest_engine = None

        # 交易记录
        self.trades: List[Dict[str, Any]] = []
        self.daily_values: List[Dict[str, Any]] = []

        # 风险控制参数
        self.max_position_weight = 0.1  # 单仓位最大权重
        self.max_sector_weight = 0.3    # 单行业最大权重
        self.max_leverage = 1.0         # 最大杠杆
        self.stop_loss_pct = 0.05       # 止损比例
        
        # 策略集成支持
        self.strategies: Dict[str, Any] = {}
        self.strategy_weights: Dict[str, float] = {}

        # 再平衡参数
        self.rebalance_threshold = 0.05  # 再平衡阈值
        self.target_weights: Dict[str, float] = {}
        
        # 回测模式专用属性
        if mode == "backtest":
            self._setup_backtest_mode()
            
    def _setup_backtest_mode(self):
        """设置回测模式专用配置"""
        # 导入回测相关模块
        try:
            from ..backtest.backtrader_engine import BacktraderEngine
            self.backtest_engine = BacktraderEngine(self.initial_capital)
        except ImportError:
            warnings.warn("Backtest engine not available. Running in simulation mode.")
            self.backtest_engine = None
            
    def add_strategy(self, name: str, strategy, weight: float = 1.0):
        """添加策略到投资组合"""
        self.strategies[name] = strategy
        self.strategy_weights[name] = weight
        
    def remove_strategy(self, name: str):
        """移除策略"""
        if name in self.strategies:
            del self.strategies[name]
        if name in self.strategy_weights:
            del self.strategy_weights[name]
            
    def run_backtest(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """运行投资组合回测"""
        if self.mode != "backtest":
            raise ValueError("Portfolio must be in backtest mode to run backtests")
            
        if not self.strategies:
            raise ValueError("No strategies added to portfolio")
            
        results = {}
        
        # 对每个策略运行回测
        for strategy_name, strategy in self.strategies.items():
            weight = self.strategy_weights[strategy_name]
            
            if self.backtest_engine:
                # 使用backtrader引擎
                strategy_result = self.backtest_engine.run_backtest(strategy, data)
            else:
                # 简化模拟回测
                strategy_result = self._simulate_strategy_backtest(strategy, data)
            
            # 按权重调整结果
            strategy_result['weight'] = weight
            strategy_result['weighted_return'] = strategy_result.get('total_return', 0) * weight
            
            results[strategy_name] = strategy_result
        
        # 计算组合总体表现
        total_weighted_return = sum(r['weighted_return'] for r in results.values())
        
        portfolio_result = {
            'strategies': results,
            'portfolio_return': total_weighted_return,
            'initial_value': self.initial_capital,
            'final_value': self.initial_capital * (1 + total_weighted_return),
            'strategy_count': len(self.strategies)
        }
        
        return portfolio_result
        
    def _simulate_strategy_backtest(self, strategy, data) -> Dict[str, Any]:
        """简化的策略回测模拟"""
        # 这是一个简化版本，当backtrader不可用时使用
        return {
            'total_return': 0.05,  # 模拟5%收益
            'total_return_pct': 5.0,
            'trades': [],
            'note': 'Simulated backtest result'
        }

    def add_cash(self, amount: float):
        """添加现金"""
        self.current_cash += amount

    def get_total_value(self) -> float:
        """获取投资组合总价值"""
        return self.current_cash + sum(pos.market_value for pos in self.positions.values())

    def get_positions_value(self) -> float:
        """获取持仓总价值"""
        return sum(pos.market_value for pos in self.positions.values())

    def get_cash_weight(self) -> float:
        """获取现金权重"""
        total_value = self.get_total_value()
        return self.current_cash / total_value if total_value > 0 else 1.0

    def update_prices(self, prices: Dict[str, float]):
        """更新所有持仓的当前价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
                
        # 在回测模式下记录每日价值
        if self.mode == "backtest":
            self._record_daily_value(prices)

    def calculate_weights(self) -> Dict[str, float]:
        """计算当前权重"""
        total_value = self.get_total_value()
        if total_value <= 0:
            return {}

        weights = {}
        for symbol, position in self.positions.items():
            weights[symbol] = position.market_value / total_value
            position.weight = weights[symbol]

        return weights

    def buy(self, symbol: str, quantity: float = None, amount: float = None,
            price: float = None) -> bool:
        """
        买入股票

        Args:
            symbol: 股票代码
            quantity: 买入数量
            amount: 买入金额
            price: 买入价格

        Returns:
            是否成功买入
        """
        if price is None:
            raise ValueError("Must specify price for buy order")

        # 计算买入数量
        if quantity is None and amount is not None:
            quantity = amount / price
        elif quantity is None:
            raise ValueError("Must specify either quantity or amount")

        cost = quantity * price

        # 检查资金是否充足
        if cost > self.current_cash:
            warnings.warn(f"Insufficient cash: need {cost:.2f}, have {self.current_cash:.2f}")
            return False

        # 风险检查
        total_value_after = self.get_total_value() + cost - self.current_cash
        new_weight = (cost) / total_value_after

        if symbol in self.positions:
            current_value = self.positions[symbol].market_value
            new_weight = (current_value + cost) / total_value_after

        if new_weight > self.max_position_weight:
            warnings.warn(f"Position weight {new_weight:.1%} exceeds maximum {self.max_position_weight:.1%}")
            return False

        # 执行买入
        if symbol in self.positions:
            # 加仓
            pos = self.positions[symbol]
            new_quantity = pos.quantity + quantity
            new_avg_cost = (pos.quantity * pos.avg_cost + quantity * price) / new_quantity
            pos.quantity = new_quantity
            pos.avg_cost = new_avg_cost
        else:
            # 新建仓位
            self.positions[symbol] = PortfolioPosition(symbol, quantity, price, price)

        self.current_cash -= cost

        # 记录交易
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'buy',
            'quantity': quantity,
            'price': price,
            'amount': cost,
            'cash_after': self.current_cash
        }
        self.trades.append(trade_record)
        
        # 在回测模式下同时记录到策略系统
        if self.mode == "backtest" and self.backtest_engine:
            self._log_trade_to_backtest(trade_record)

        return True

    def sell(self, symbol: str, quantity: float = None, weight: float = None,
             price: float = None) -> bool:
        """
        卖出股票

        Args:
            symbol: 股票代码
            quantity: 卖出数量
            weight: 卖出权重（相对于当前持仓）
            price: 卖出价格

        Returns:
            是否成功卖出
        """
        if symbol not in self.positions:
            warnings.warn(f"No position in {symbol} to sell")
            return False

        if price is None:
            raise ValueError("Must specify price for sell order")

        position = self.positions[symbol]

        # 计算卖出数量
        if quantity is None and weight is not None:
            quantity = position.quantity * weight
        elif quantity is None:
            quantity = position.quantity  # 全部卖出

        if quantity > position.quantity:
            quantity = position.quantity

        if quantity <= 0:
            return False

        # 执行卖出
        revenue = quantity * price
        cost_basis = quantity * position.avg_cost
        realized_pnl = revenue - cost_basis

        position.quantity -= quantity
        position.realized_pnl += realized_pnl
        self.current_cash += revenue

        # 如果全部卖出，移除持仓
        if position.quantity == 0:
            self.historical_positions.append(position)
            del self.positions[symbol]

        # 记录交易
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'sell',
            'quantity': quantity,
            'price': price,
            'amount': revenue,
            'realized_pnl': realized_pnl,
            'cash_after': self.current_cash
        }
        self.trades.append(trade_record)
        
        # 在回测模式下同时记录到策略系统
        if self.mode == "backtest" and self.backtest_engine:
            self._log_trade_to_backtest(trade_record)

        return True

    def rebalance_to_target(self, target_weights: Dict[str, float],
                           prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        重新平衡到目标权重

        Args:
            target_weights: 目标权重字典
            prices: 当前价格字典

        Returns:
            需要执行的交易列表
        """
        self.target_weights = target_weights.copy()
        current_weights = self.calculate_weights()
        total_value = self.get_total_value()

        trades = []

        # 计算需要调整的仓位
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                continue

            current_weight = current_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) > self.rebalance_threshold:
                target_value = target_weight * total_value
                current_value = current_weight * total_value
                value_diff = target_value - current_value

                if value_diff > 0:  # 需要买入
                    quantity = value_diff / prices[symbol]
                    if self.buy(symbol, quantity, price=prices[symbol]):
                        trades.append({
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': quantity,
                            'target_weight': target_weight,
                            'current_weight': current_weight
                        })
                else:  # 需要卖出
                    if symbol in self.positions:
                        sell_value = abs(value_diff)
                        quantity = sell_value / prices[symbol]
                        if self.sell(symbol, quantity, price=prices[symbol]):
                            trades.append({
                                'symbol': symbol,
                                'action': 'sell',
                                'quantity': quantity,
                                'target_weight': target_weight,
                                'current_weight': current_weight
                            })

        return trades

    def _record_daily_value(self, prices: Dict[str, float]):
        """记录每日组合价值（回测模式专用）"""
        total_value = self.get_total_value()
        positions_value = self.get_positions_value()
        
        daily_record = {
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.current_cash,
            'positions_value': positions_value,
            'return_since_inception': (total_value - self.initial_capital) / self.initial_capital
        }
        self.daily_values.append(daily_record)
        
    def _log_trade_to_backtest(self, trade_record: Dict[str, Any]):
        """将交易记录到回测引擎（如果可用）"""
        if self.backtest_engine and hasattr(self.backtest_engine, 'log_trade'):
            self.backtest_engine.log_trade(trade_record)

    def calculate_equal_weights(self, symbols: List[str]) -> Dict[str, float]:
        """计算等权重配置"""
        if not symbols:
            return {}
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}

    def calculate_market_cap_weights(self, market_caps: Dict[str, float]) -> Dict[str, float]:
        """计算市值加权配置"""
        total_cap = sum(market_caps.values())
        if total_cap <= 0:
            return self.calculate_equal_weights(list(market_caps.keys()))
        return {symbol: cap / total_cap for symbol, cap in market_caps.items()}

    def calculate_risk_parity_weights(self, returns_data: pd.DataFrame,
                                    lookback_days: int = 252) -> Dict[str, float]:
        """
        计算风险平价权重

        基于历史波动率的倒数来分配权重，使每个资产对组合风险的贡献相等
        """
        if returns_data.empty:
            return {}

        # 计算最近lookback_days的数据
        recent_data = returns_data.tail(lookback_days)

        # 计算各资产的波动率
        volatilities = recent_data.std()

        # 风险平价：权重与波动率的倒数成正比
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()

        return weights.to_dict()

    def calculate_minimum_variance_weights(self, returns_data: pd.DataFrame,
                                         lookback_days: int = 252) -> Dict[str, float]:
        """
        计算最小方差权重

        找到使组合总风险最小的权重配置
        """
        if returns_data.empty:
            return {}

        try:
            from scipy.optimize import minimize

            recent_data = returns_data.tail(lookback_days)
            cov_matrix = recent_data.cov()

            n = len(cov_matrix)
            if n == 0:
                return {}

            # 目标函数：最小化组合方差
            def portfolio_variance(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))

            # 约束条件：权重和为1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

            # 边界条件：权重在0和1之间
            bounds = tuple((0, 1) for _ in range(n))

            # 初始猜测：等权重
            x0 = np.array([1/n] * n)

            # 优化
            result = minimize(portfolio_variance, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)

            if result.success:
                symbols = cov_matrix.columns.tolist()
                return dict(zip(symbols, result.x))
            else:
                return self.calculate_equal_weights(cov_matrix.columns.tolist())

        except ImportError:
            warnings.warn("scipy not available, using equal weights")
            return self.calculate_equal_weights(returns_data.columns.tolist())
        except Exception as e:
            warnings.warn(f"Minimum variance optimization failed: {e}")
            return self.calculate_equal_weights(returns_data.columns.tolist())

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取投资组合绩效指标"""
        if not self.daily_values:
            return {}

        df = pd.DataFrame(self.daily_values)
        if 'total_value' not in df.columns:
            return {}

        values = df['total_value']
        returns = values.pct_change().dropna()

        if len(returns) == 0:
            return {}

        # 基本指标
        total_return = (values.iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 胜率
        win_rate = (returns > 0).sum() / len(returns)

        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annual_return': annual_return,
            'annual_return_pct': annual_return * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'current_value': values.iloc[-1],
            'total_trades': len(self.trades)
        }

    def get_positions_summary(self) -> pd.DataFrame:
        """获取持仓摘要"""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for symbol, position in self.positions.items():
            data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'avg_cost': position.avg_cost,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'cost_basis': position.cost_basis,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_return': position.unrealized_return,
                'weight': position.weight
            })

        return pd.DataFrame(data)

    def get_trades_history(self) -> pd.DataFrame:
        """获取交易历史"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def record_daily_value(self, date: datetime = None):
        """记录日度价值"""
        if date is None:
            date = datetime.now()

        total_value = self.get_total_value()
        positions_value = self.get_positions_value()

        self.daily_values.append({
            'date': date,
            'total_value': total_value,
            'cash': self.current_cash,
            'positions_value': positions_value,
            'positions_count': len(self.positions)
        })

    def apply_risk_limits(self, prices: Dict[str, float]) -> List[str]:
        """
        应用风险限制

        Returns:
            被强制平仓的股票列表
        """
        forced_sales = []

        # 更新价格
        self.update_prices(prices)

        # 检查止损
        for symbol, position in list(self.positions.items()):
            if position.unrealized_return < -self.stop_loss_pct:
                if self.sell(symbol, price=position.current_price):
                    forced_sales.append(symbol)

        # 检查仓位集中度
        weights = self.calculate_weights()
        for symbol, weight in weights.items():
            if weight > self.max_position_weight:
                excess_weight = weight - self.max_position_weight
                if symbol in self.positions:
                    sell_quantity = self.positions[symbol].quantity * (excess_weight / weight)
                    if self.sell(symbol, sell_quantity, price=prices.get(symbol)):
                        if symbol not in forced_sales:
                            forced_sales.append(symbol)

        return forced_sales

    def reset(self):
        """重置投资组合"""
        self.current_cash = self.initial_capital
        self.positions.clear()
        self.historical_positions.clear()
        self.trades.clear()
        self.daily_values.clear()

    def __str__(self) -> str:
        """投资组合信息字符串"""
        total_value = self.get_total_value()
        positions_count = len(self.positions)
        cash_weight = self.get_cash_weight()

        return (f"Portfolio '{self.name}': "
                f"Value=${total_value:,.2f}, "
                f"Positions={positions_count}, "
                f"Cash={cash_weight:.1%}")

class FactorPortfolioManager(PortfolioManager):
    """
    因子投资组合管理器
    
    专门用于因子策略的组合管理，扩展了基础PortfolioManager功能
    """
    
    def __init__(self, initial_capital: float = 1000000.0, name: str = "Factor Portfolio", mode: str = "live"):
        super().__init__(initial_capital, name, mode)
        
        # 因子相关属性
        self.factor_exposures: Dict[str, float] = {}
        self.factor_returns: Dict[str, List[float]] = {}
        self.risk_model = None
        
    def set_factor_exposures(self, exposures: Dict[str, float]):
        """设置组合的因子暴露"""
        self.factor_exposures = exposures
        
    def get_factor_attribution(self) -> Dict[str, float]:
        """计算因子归因分析"""
        if not self.factor_exposures:
            return {}
        
        performance_metrics = self.get_performance_metrics()
        portfolio_return = performance_metrics.get('total_return', 0)
        
        attribution = {}
        total_exposure = sum(abs(exp) for exp in self.factor_exposures.values())
        
        for factor_name, exposure in self.factor_exposures.items():
            if total_exposure > 0:
                # 简化的归因计算（实际中需要更复杂的模型）
                attribution[factor_name] = (exposure / total_exposure) * portfolio_return
            else:
                attribution[factor_name] = 0
        
        return attribution


def create_portfolio_manager(initial_capital: float = 1000000.0,
                           name: str = "Portfolio") -> PortfolioManager:
    """
    创建投资组合管理器的便捷函数

    Args:
        initial_capital: 初始资金
        name: 投资组合名称

    Returns:
        PortfolioManager实例
    """
    return PortfolioManager(initial_capital=initial_capital, name=name)


def create_factor_portfolio_manager(initial_capital: float = 1000000.0, 
                                  name: str = "Factor Portfolio",
                                  mode: str = "live") -> FactorPortfolioManager:
    """
    创建因子投资组合管理器的便捷函数
    
    Args:
        initial_capital: 初始资金
        name: 组合名称
        mode: 运行模式 ("live" 或 "backtest")
    
    Returns:
        FactorPortfolioManager实例
    """
    return FactorPortfolioManager(initial_capital, name, mode)