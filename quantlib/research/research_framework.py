"""
研究框架 (Research Framework)

提供完整的量化研究框架，整合因子库、分析器和回测功能
支持批量因子研究、策略开发和研究报告生成
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date, timedelta
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
import json

from .factor_library import FactorLibrary, BaseFactor, FactorCategory
from .factor_analyzer import FactorAnalyzer, FactorPerformance, ICAnalysis


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000
    commission: float = 0.001
    long_pct: float = 0.2      # 做多比例
    short_pct: float = 0.2     # 做空比例
    rebalance_freq: str = 'M'  # 调仓频率 ('D', 'W', 'M', 'Q')
    min_stocks: int = 5        # 最小股票数量


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_loss_ratio: float
    portfolio_values: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    performance_metrics: Dict[str, float]


class FactorBacktester:
    """
    因子回测器
    
    基于因子构建投资组合并进行回测
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results_cache = {}
        
    def run_factor_backtest(self, factor_data: pd.DataFrame, price_data: pd.DataFrame,
                          factor_name: str) -> BacktestResult:
        """
        运行单因子回测
        
        Args:
            factor_data: 因子数据，index为时间，columns为股票代码
            price_data: 价格数据，index为时间，columns为股票代码
            factor_name: 因子名称
            
        Returns:
            回测结果
        """
        # 对齐数据
        common_dates = factor_data.index.intersection(price_data.index)
        common_stocks = factor_data.columns.intersection(price_data.columns)
        
        if len(common_dates) < 20 or len(common_stocks) < self.config.min_stocks:
            warnings.warn(f"Insufficient data for backtesting factor '{factor_name}'")
            return self._create_empty_result()
        
        factor_aligned = factor_data.loc[common_dates, common_stocks]
        price_aligned = price_data.loc[common_dates, common_stocks]
        
        # 计算收益率
        returns = price_aligned.pct_change().fillna(0)
        
        # 生成调仓日期
        rebalance_dates = self._generate_rebalance_dates(common_dates)
        
        # 初始化
        portfolio_values = []
        positions_history = []
        trades_history = []
        current_capital = self.config.initial_capital
        current_positions = {}
        
        for i, rebalance_date in enumerate(rebalance_dates):
            if rebalance_date not in factor_aligned.index:
                continue
            
            # 获取因子值
            factor_values = factor_aligned.loc[rebalance_date].dropna()
            
            if len(factor_values) < self.config.min_stocks:
                continue
            
            # 选股
            selected_stocks = self._select_stocks(factor_values)
            
            # 如果不是第一次调仓，先平仓
            if i > 0:
                sell_trades = self._close_positions(
                    current_positions, rebalance_date, price_aligned
                )
                trades_history.extend(sell_trades)
                current_capital += sum(trade['amount'] for trade in sell_trades)
                current_positions = {}
            
            # 开新仓
            if selected_stocks:
                new_positions, buy_trades = self._open_positions(
                    selected_stocks, current_capital, rebalance_date, price_aligned
                )
                current_positions = new_positions
                trades_history.extend(buy_trades)
                current_capital -= sum(trade['amount'] for trade in buy_trades)
            
            # 记录持仓
            positions_history.append({
                'date': rebalance_date,
                'positions': current_positions.copy(),
                'cash': current_capital
            })
        
        # 计算组合价值序列
        portfolio_value_series = self._calculate_portfolio_values(
            positions_history, price_aligned, returns
        )
        
        # 计算绩效指标
        performance_metrics = self._calculate_performance_metrics(portfolio_value_series)
        
        return BacktestResult(
            total_return=performance_metrics['total_return'],
            annual_return=performance_metrics['annual_return'],
            volatility=performance_metrics['volatility'],
            sharpe_ratio=performance_metrics['sharpe_ratio'],
            max_drawdown=performance_metrics['max_drawdown'],
            calmar_ratio=performance_metrics['calmar_ratio'],
            win_rate=performance_metrics.get('win_rate', 0),
            profit_loss_ratio=performance_metrics.get('profit_loss_ratio', 0),
            portfolio_values=portfolio_value_series,
            positions=pd.DataFrame(positions_history),
            trades=pd.DataFrame(trades_history),
            performance_metrics=performance_metrics
        )
    
    def _generate_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[datetime]:
        """生成调仓日期"""
        if self.config.rebalance_freq == 'D':
            return dates.tolist()
        elif self.config.rebalance_freq == 'W':
            return [date for date in dates if date.weekday() == 0]  # 每周一
        elif self.config.rebalance_freq == 'M':
            return [date for date in dates if date.day <= 7 and date.weekday() < 5]  # 每月第一个工作日
        elif self.config.rebalance_freq == 'Q':
            return [date for date in dates if date.month % 3 == 1 and date.day <= 7]  # 每季度第一个月
        else:
            return dates.tolist()
    
    def _select_stocks(self, factor_values: pd.Series) -> Dict[str, str]:
        """
        基于因子选股
        
        Returns:
            {'long': [...], 'short': [...]}
        """
        # 排序
        factor_sorted = factor_values.sort_values(ascending=False)
        n_stocks = len(factor_sorted)
        
        # 计算选股数量
        n_long = max(1, int(n_stocks * self.config.long_pct))
        n_short = max(1, int(n_stocks * self.config.short_pct))
        
        # 选择股票
        long_stocks = factor_sorted.head(n_long).index.tolist()
        short_stocks = factor_sorted.tail(n_short).index.tolist()
        
        return {
            'long': long_stocks,
            'short': short_stocks
        }
    
    def _open_positions(self, selected_stocks: Dict[str, List[str]], available_capital: float,
                       date: datetime, price_data: pd.DataFrame) -> Tuple[Dict, List]:
        """开仓"""
        positions = {}
        trades = []
        
        if date not in price_data.index:
            return positions, trades
        
        current_prices = price_data.loc[date]
        
        # 计算仓位资金
        long_stocks = selected_stocks.get('long', [])
        short_stocks = selected_stocks.get('short', [])
        total_stocks = len(long_stocks) + len(short_stocks)
        
        if total_stocks == 0:
            return positions, trades
        
        position_value = available_capital / total_stocks
        
        # 开多头仓位
        for stock in long_stocks:
            if stock in current_prices.index and current_prices[stock] > 0:
                price = current_prices[stock]
                quantity = int(position_value / price)
                if quantity > 0:
                    positions[stock] = {
                        'quantity': quantity,
                        'price': price,
                        'side': 'long',
                        'value': quantity * price
                    }
                    trades.append({
                        'date': date,
                        'stock': stock,
                        'action': 'buy',
                        'quantity': quantity,
                        'price': price,
                        'amount': quantity * price,
                        'commission': quantity * price * self.config.commission
                    })
        
        # 开空头仓位（在实际应用中需要考虑融券机制）
        for stock in short_stocks:
            if stock in current_prices.index and current_prices[stock] > 0:
                price = current_prices[stock]
                quantity = int(position_value / price)
                if quantity > 0:
                    positions[stock] = {
                        'quantity': -quantity,  # 负数表示空头
                        'price': price,
                        'side': 'short',
                        'value': -quantity * price
                    }
                    trades.append({
                        'date': date,
                        'stock': stock,
                        'action': 'sell_short',
                        'quantity': quantity,
                        'price': price,
                        'amount': quantity * price,
                        'commission': quantity * price * self.config.commission
                    })
        
        return positions, trades
    
    def _close_positions(self, positions: Dict, date: datetime, 
                        price_data: pd.DataFrame) -> List[Dict]:
        """平仓"""
        trades = []
        
        if date not in price_data.index:
            return trades
        
        current_prices = price_data.loc[date]
        
        for stock, position in positions.items():
            if stock in current_prices.index and current_prices[stock] > 0:
                price = current_prices[stock]
                quantity = abs(position['quantity'])
                
                if position['side'] == 'long':
                    action = 'sell'
                    amount = quantity * price
                else:  # short
                    action = 'cover'
                    amount = quantity * price
                
                trades.append({
                    'date': date,
                    'stock': stock,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'amount': amount,
                    'commission': amount * self.config.commission,
                    'pnl': self._calculate_position_pnl(position, price)
                })
        
        return trades
    
    def _calculate_position_pnl(self, position: Dict, current_price: float) -> float:
        """计算持仓盈亏"""
        entry_price = position['price']
        quantity = position['quantity']
        
        if quantity > 0:  # 多头
            return (current_price - entry_price) * quantity
        else:  # 空头
            return (entry_price - current_price) * abs(quantity)
    
    def _calculate_portfolio_values(self, positions_history: List[Dict],
                                  price_data: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        """计算组合价值序列"""
        values = []
        dates = []
        
        for pos_record in positions_history:
            date = pos_record['date']
            positions = pos_record['positions']
            cash = pos_record['cash']
            
            # 计算持仓市值
            position_value = 0
            if date in price_data.index:
                current_prices = price_data.loc[date]
                for stock, position in positions.items():
                    if stock in current_prices.index:
                        quantity = position['quantity']
                        price = current_prices[stock]
                        position_value += quantity * price
            
            total_value = cash + position_value
            values.append(total_value)
            dates.append(date)
        
        return pd.Series(values, index=dates)
    
    def _calculate_performance_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """计算绩效指标"""
        if len(portfolio_values) < 2:
            return {}
        
        # 计算收益率
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # 基础指标
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # 年化收益率（假设252个交易日）
        n_periods = len(returns)
        if n_periods > 0:
            annual_return = (1 + total_return) ** (252 / n_periods) - 1
        else:
            annual_return = 0
        
        # 波动率
        volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 盈亏比
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            profit_loss_ratio = positive_returns.mean() / abs(negative_returns.mean())
        else:
            profit_loss_ratio = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio
        }
    
    def _create_empty_result(self) -> BacktestResult:
        """创建空的回测结果"""
        return BacktestResult(
            total_return=0,
            annual_return=0,
            volatility=0,
            sharpe_ratio=0,
            max_drawdown=0,
            calmar_ratio=0,
            win_rate=0,
            profit_loss_ratio=0,
            portfolio_values=pd.Series(),
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            performance_metrics={}
        )


class ResearchFramework:
    """
    研究框架
    
    整合因子库、分析器和回测器，提供完整的量化研究功能
    """
    
    def __init__(self, factor_library: Optional[FactorLibrary] = None,
                 factor_analyzer: Optional[FactorAnalyzer] = None,
                 storage_path: str = "data/research"):
        """
        Args:
            factor_library: 因子库实例
            factor_analyzer: 因子分析器实例
            storage_path: 研究结果存储路径
        """
        self.factor_library = factor_library or FactorLibrary()
        self.factor_analyzer = factor_analyzer or FactorAnalyzer()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 研究结果缓存
        self.research_results = {}
        self.backtest_results = {}
        
    def conduct_factor_research(self, data: pd.DataFrame, returns: pd.Series,
                              factor_names: Optional[List[str]] = None,
                              save_results: bool = True) -> Dict[str, FactorPerformance]:
        """
        进行因子研究
        
        Args:
            data: 股票数据，包含OHLCV和基本面数据
            returns: 收益率数据
            factor_names: 要研究的因子名称列表，None表示研究所有因子
            save_results: 是否保存结果
            
        Returns:
            因子研究结果字典
        """
        # 确定要研究的因子
        if factor_names is None:
            factor_names = list(self.factor_library.factors.keys())
        
        # 计算因子值
        print(f"计算 {len(factor_names)} 个因子的值...")
        factor_data = self.factor_library.calculate_factors(factor_names, data)
        
        # 批量分析
        print("进行因子有效性分析...")
        research_results = {}
        
        for factor_name in factor_names:
            if factor_name in factor_data.columns:
                factor_series = factor_data[factor_name]
                try:
                    result = self.factor_analyzer.comprehensive_factor_analysis(
                        factor_series, returns, factor_name
                    )
                    research_results[factor_name] = result
                    print(f"✅ {factor_name}: IC={result.ic_analysis.ic_mean:.3f}, IR={result.ic_analysis.ic_ir:.3f}")
                except Exception as e:
                    warnings.warn(f"Failed to analyze factor '{factor_name}': {e}")
        
        # 保存结果
        if save_results:
            self._save_research_results(research_results, "factor_research_results")
        
        self.research_results.update(research_results)
        return research_results
    
    def conduct_factor_backtest(self, factor_data: pd.DataFrame, price_data: pd.DataFrame,
                              config: BacktestConfig, factor_names: Optional[List[str]] = None,
                              save_results: bool = True) -> Dict[str, BacktestResult]:
        """
        进行因子回测研究
        
        Args:
            factor_data: 因子数据
            price_data: 价格数据
            config: 回测配置
            factor_names: 要回测的因子名称
            save_results: 是否保存结果
            
        Returns:
            回测结果字典
        """
        if factor_names is None:
            factor_names = factor_data.columns.tolist()
        
        backtest_results = {}
        backtester = FactorBacktester(config)
        
        print(f"开始回测 {len(factor_names)} 个因子...")
        
        for factor_name in factor_names:
            if factor_name in factor_data.columns:
                print(f"回测因子: {factor_name}")
                
                # 提取单因子数据
                single_factor_data = factor_data[[factor_name]].dropna()
                
                try:
                    result = backtester.run_factor_backtest(
                        single_factor_data, price_data, factor_name
                    )
                    backtest_results[factor_name] = result
                    print(f"✅ {factor_name}: 年化收益={result.annual_return:.2%}, 夏普比={result.sharpe_ratio:.2f}")
                except Exception as e:
                    warnings.warn(f"Failed to backtest factor '{factor_name}': {e}")
        
        # 保存结果
        if save_results:
            self._save_backtest_results(backtest_results, "factor_backtest_results")
        
        self.backtest_results.update(backtest_results)
        return backtest_results
    
    def comprehensive_factor_study(self, data: pd.DataFrame, price_data: pd.DataFrame,
                                 returns: pd.Series, config: BacktestConfig,
                                 factor_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        综合因子研究（包含分析和回测）
        
        Args:
            data: 股票数据
            price_data: 价格数据
            returns: 收益率数据
            config: 回测配置
            factor_names: 因子名称列表
            
        Returns:
            综合研究结果
        """
        print("🔬 开始综合因子研究...")
        
        # 1. 因子有效性分析
        print("\n1️⃣ 因子有效性分析")
        analysis_results = self.conduct_factor_research(data, returns, factor_names)
        
        # 2. 计算因子值用于回测
        print("\n2️⃣ 计算因子值")
        if factor_names is None:
            factor_names = list(analysis_results.keys())
        
        factor_data = self.factor_library.calculate_factors(factor_names, data)
        
        # 3. 因子回测
        print("\n3️⃣ 因子策略回测")
        backtest_results = self.conduct_factor_backtest(factor_data, price_data, config, factor_names)
        
        # 4. 综合排名
        print("\n4️⃣ 生成综合排名")
        ranking = self.factor_analyzer.create_factor_ranking(analysis_results, 'ic_ir')
        
        # 5. 相关性分析
        print("\n5️⃣ 因子相关性分析")
        correlation_matrix = self.factor_analyzer.factor_correlation_analysis(
            {name: factor_data[name] for name in factor_names if name in factor_data.columns}
        )
        
        # 整合结果
        comprehensive_results = {
            'analysis_results': analysis_results,
            'backtest_results': backtest_results,
            'factor_ranking': ranking,
            'factor_correlation': correlation_matrix,
            'factor_data': factor_data,
            'config': asdict(config),
            'summary': self._generate_study_summary(analysis_results, backtest_results)
        }
        
        # 保存综合结果
        self._save_comprehensive_results(comprehensive_results, "comprehensive_factor_study")
        
        print("\n🎉 综合因子研究完成!")
        return comprehensive_results
    
    def add_custom_factor(self, name: str, calc_func, description: str = "",
                         category: FactorCategory = FactorCategory.CUSTOM) -> bool:
        """添加自定义因子到库中"""
        return self.factor_library.create_custom_factor(name, calc_func, description, category)
    
    def get_factor_library_summary(self) -> Dict[str, Any]:
        """获取因子库摘要"""
        return self.factor_library.get_summary()
    
    def _generate_study_summary(self, analysis_results: Dict[str, FactorPerformance],
                              backtest_results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """生成研究摘要"""
        if not analysis_results:
            return {}
        
        # 分析结果统计
        ic_means = [result.ic_analysis.ic_mean for result in analysis_results.values()]
        ic_irs = [result.ic_analysis.ic_ir for result in analysis_results.values()]
        
        # 回测结果统计
        annual_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for factor_name in analysis_results.keys():
            if factor_name in backtest_results:
                bt_result = backtest_results[factor_name]
                annual_returns.append(bt_result.annual_return)
                sharpe_ratios.append(bt_result.sharpe_ratio)
                max_drawdowns.append(bt_result.max_drawdown)
        
        # 找出最佳因子
        best_ic_factor = max(analysis_results.items(), key=lambda x: x[1].ic_analysis.ic_ir)
        
        best_backtest_factor = None
        if backtest_results:
            best_backtest_factor = max(backtest_results.items(), key=lambda x: x[1].sharpe_ratio)
        
        return {
            'total_factors': len(analysis_results),
            'avg_ic_mean': np.mean(ic_means) if ic_means else 0,
            'avg_ic_ir': np.mean(ic_irs) if ic_irs else 0,
            'avg_annual_return': np.mean(annual_returns) if annual_returns else 0,
            'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'avg_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
            'best_ic_factor': {
                'name': best_ic_factor[0],
                'ic_ir': best_ic_factor[1].ic_analysis.ic_ir
            },
            'best_backtest_factor': {
                'name': best_backtest_factor[0] if best_backtest_factor else None,
                'sharpe_ratio': best_backtest_factor[1].sharpe_ratio if best_backtest_factor else 0
            } if best_backtest_factor else None
        }
    
    def _save_research_results(self, results: Dict[str, FactorPerformance], filename: str):
        """保存研究结果"""
        # 转换为可序列化的格式
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {
                'factor_name': result.factor_name,
                'ic_mean': result.ic_analysis.ic_mean,
                'ic_std': result.ic_analysis.ic_std,
                'ic_ir': result.ic_analysis.ic_ir,
                'ic_positive_rate': result.ic_analysis.ic_positive_rate,
                'turnover': result.turnover,
                'factor_autocorr': result.factor_autocorr,
                'long_short_return': result.long_short_return,
                'long_short_sharpe': result.long_short_sharpe,
                'factor_stats': result.factor_stats
            }
        
        # 保存JSON文件
        filepath = self.storage_path / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_backtest_results(self, results: Dict[str, BacktestResult], filename: str):
        """保存回测结果"""
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'calmar_ratio': result.calmar_ratio,
                'win_rate': result.win_rate,
                'profit_loss_ratio': result.profit_loss_ratio,
                'performance_metrics': result.performance_metrics
            }
        
        filepath = self.storage_path / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_comprehensive_results(self, results: Dict[str, Any], filename: str):
        """保存综合研究结果"""
        # 简化结果用于保存
        simplified_results = {
            'summary': results['summary'],
            'config': results['config'],
            'factor_ranking': results['factor_ranking'].to_dict() if hasattr(results['factor_ranking'], 'to_dict') else results['factor_ranking'],
            'correlation_matrix': results['factor_correlation'].to_dict() if hasattr(results['factor_correlation'], 'to_dict') else results['factor_correlation']
        }
        
        filepath = self.storage_path / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False, default=str)


def create_research_framework(factor_library: Optional[FactorLibrary] = None,
                            factor_analyzer: Optional[FactorAnalyzer] = None,
                            storage_path: str = "data/research") -> ResearchFramework:
    """
    创建研究框架的便捷函数
    
    Args:
        factor_library: 因子库实例
        factor_analyzer: 因子分析器实例
        storage_path: 存储路径
        
    Returns:
        ResearchFramework实例
    """
    return ResearchFramework(factor_library, factor_analyzer, storage_path)