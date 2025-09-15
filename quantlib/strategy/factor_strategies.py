"""
因子投资策略模块 (Factor Investment Strategies)

基于因子模型的投资策略实现，包括：
- 单因子策略
- 多因子模型
- 因子组合优化
- 因子回测和分析

与Portfolio模块集成，提供完整的因子投资解决方案
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date
import warnings

from .base import BaseStrategy, SignalType, TradingSignal


class FactorType:
    """因子类型定义"""
    # 基本面因子
    VALUE = "value"              # 价值因子 (P/E, P/B等)
    GROWTH = "growth"            # 成长因子 (ROE, 营收增长等)
    PROFITABILITY = "profitability"  # 盈利因子 (毛利率, ROA等)
    QUALITY = "quality"          # 质量因子 (负债率, 现金流等)
    
    # 技术面因子
    MOMENTUM = "momentum"        # 动量因子 (价格趋势等)
    REVERSAL = "reversal"        # 反转因子 (超买超卖等)
    VOLATILITY = "volatility"    # 波动率因子
    
    # 宏观因子
    SIZE = "size"                # 规模因子 (市值大小)
    SECTOR = "sector"            # 行业因子
    BETA = "beta"                # 市场敏感性


class SingleFactorStrategy(BaseStrategy):
    """
    单因子策略
    
    基于单一因子构建投资组合
    """
    
    def __init__(self, symbols: List[str], factor_type: str, factor_data: pd.DataFrame,
                 long_pct: float = 0.3, short_pct: float = 0.3, **kwargs):
        """
        Args:
            symbols: 股票代码列表
            factor_type: 因子类型
            factor_data: 因子数据，index为日期，columns为股票代码
            long_pct: 做多股票比例（基于因子排序）
            short_pct: 做空股票比例（基于因子排序）
        """
        super().__init__(f"Single Factor Strategy ({factor_type})", symbols, **kwargs)
        self.factor_type = factor_type
        self.factor_data = factor_data
        self.long_pct = long_pct
        self.short_pct = short_pct
        
    def initialize(self):
        """初始化因子策略"""
        # 计算因子排名
        factor_ranks = self.factor_data.rank(axis=1, method='dense', ascending=False)
        self.add_indicator("all", f"{self.factor_type}_rank", factor_ranks)
        
        # 计算分位数阈值
        self.long_threshold = self.long_pct
        self.short_threshold = 1 - self.short_pct
        
        self.is_initialized = True
    
    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """基于因子排名生成交易信号"""
        signals = []
        
        if not self.is_initialized:
            return signals
        
        # 获取当前因子数据
        time_key = current_time.strftime('%Y-%m-%d')
        
        if 'all' not in self.indicators:
            return signals
            
        factor_ranks = self.indicators['all'][f"{self.factor_type}_rank"]
        
        try:
            # 查找最接近的日期
            available_dates = factor_ranks.index.strftime('%Y-%m-%d')
            matching_dates = available_dates[available_dates == time_key]
            
            if len(matching_dates) == 0:
                return signals
                
            current_ranks = factor_ranks.loc[matching_dates[0]]
            
            # 计算分位数
            valid_ranks = current_ranks.dropna()
            if len(valid_ranks) == 0:
                return signals
                
            n_stocks = len(valid_ranks)
            long_cutoff = int(n_stocks * self.long_pct)
            short_cutoff = int(n_stocks * (1 - self.short_pct))
            
            # 生成做多信号（排名前long_pct的股票）
            long_symbols = valid_ranks.nsmallest(long_cutoff).index.tolist()
            for symbol in long_symbols:
                if symbol in current_data:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        timestamp=current_time,
                        confidence=0.7,
                        metadata={
                            'factor_type': self.factor_type,
                            'rank': current_ranks[symbol],
                            'percentile': current_ranks[symbol] / n_stocks
                        }
                    ))
            
            # 生成做空信号（排名后short_pct的股票）
            short_symbols = valid_ranks.nlargest(n_stocks - short_cutoff).index.tolist()
            for symbol in short_symbols:
                if symbol in current_data and symbol in self.positions:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        timestamp=current_time,
                        confidence=0.7,
                        metadata={
                            'factor_type': self.factor_type,
                            'rank': current_ranks[symbol],
                            'percentile': current_ranks[symbol] / n_stocks
                        }
                    ))
                    
        except Exception as e:
            warnings.warn(f"Error generating factor signals: {e}")
            
        return signals


class MultiFactorStrategy(BaseStrategy):
    """
    多因子策略
    
    基于多个因子的综合评分构建投资组合
    """
    
    def __init__(self, symbols: List[str], factor_data: Dict[str, pd.DataFrame], 
                 factor_weights: Dict[str, float] = None, **kwargs):
        """
        Args:
            symbols: 股票代码列表
            factor_data: 因子数据字典，key为因子名，value为因子数据
            factor_weights: 因子权重字典，key为因子名，value为权重
        """
        super().__init__("Multi Factor Strategy", symbols, **kwargs)
        self.factor_data = factor_data
        self.factor_weights = factor_weights or {}
        
        # 如果没有指定权重，使用等权重
        if not self.factor_weights:
            n_factors = len(factor_data)
            self.factor_weights = {factor: 1/n_factors for factor in factor_data.keys()}
            
    def initialize(self):
        """初始化多因子模型"""
        # 计算各因子的标准化得分
        factor_scores = {}
        
        for factor_name, factor_df in self.factor_data.items():
            # 标准化因子值
            standardized = factor_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            factor_scores[factor_name] = standardized
            self.add_indicator("all", f"{factor_name}_score", standardized)
        
        # 计算综合因子得分
        composite_score = pd.DataFrame(index=factor_scores[list(factor_scores.keys())[0]].index,
                                     columns=factor_scores[list(factor_scores.keys())[0]].columns)
        
        for factor_name, score_df in factor_scores.items():
            weight = self.factor_weights[factor_name]
            if composite_score.isna().all().all():
                composite_score = score_df * weight
            else:
                composite_score = composite_score.add(score_df * weight, fill_value=0)
        
        self.add_indicator("all", "composite_score", composite_score)
        self.is_initialized = True
    
    def generate_signals(self, current_time: datetime, current_data: Dict[str, pd.Series]) -> List[TradingSignal]:
        """基于综合因子得分生成交易信号"""
        signals = []
        
        if not self.is_initialized or 'all' not in self.indicators:
            return signals
            
        time_key = current_time.strftime('%Y-%m-%d')
        composite_score = self.indicators['all']['composite_score']
        
        try:
            # 查找最接近的日期
            available_dates = composite_score.index.strftime('%Y-%m-%d')
            matching_dates = available_dates[available_dates == time_key]
            
            if len(matching_dates) == 0:
                return signals
                
            current_scores = composite_score.loc[matching_dates[0]]
            valid_scores = current_scores.dropna()
            
            if len(valid_scores) == 0:
                return signals
            
            # 基于得分分位数生成信号
            top_quartile = valid_scores.quantile(0.75)
            bottom_quartile = valid_scores.quantile(0.25)
            
            for symbol, score in valid_scores.items():
                if symbol not in current_data:
                    continue
                    
                if score >= top_quartile:
                    # 高分股票买入
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        timestamp=current_time,
                        confidence=0.8,
                        metadata={
                            'composite_score': score,
                            'quartile': 'top',
                            'factors': {f: self.factor_weights[f] for f in self.factor_weights}
                        }
                    ))
                elif score <= bottom_quartile and symbol in self.positions:
                    # 低分股票卖出
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        timestamp=current_time,
                        confidence=0.8,
                        metadata={
                            'composite_score': score,
                            'quartile': 'bottom',
                            'factors': {f: self.factor_weights[f] for f in self.factor_weights}
                        }
                    ))
                    
        except Exception as e:
            warnings.warn(f"Error generating multi-factor signals: {e}")
            
        return signals


# FactorPortfolioManager 移到了 portfolio.manager 模块中
# 这里只保留策略算法，组合管理交给Portfolio模块


def create_factor_strategy(factor_type: str, symbols: List[str], factor_data: pd.DataFrame, 
                         **kwargs) -> SingleFactorStrategy:
    """
    创建单因子策略的便捷函数
    
    Args:
        factor_type: 因子类型
        symbols: 股票代码列表
        factor_data: 因子数据
        **kwargs: 其他策略参数
    
    Returns:
        SingleFactorStrategy实例
    """
    return SingleFactorStrategy(symbols, factor_type, factor_data, **kwargs)


def create_multi_factor_strategy(symbols: List[str], factor_data: Dict[str, pd.DataFrame],
                               factor_weights: Dict[str, float] = None, **kwargs) -> MultiFactorStrategy:
    """
    创建多因子策略的便捷函数
    
    Args:
        symbols: 股票代码列表
        factor_data: 因子数据字典
        factor_weights: 因子权重字典
        **kwargs: 其他策略参数
    
    Returns:
        MultiFactorStrategy实例
    """
    return MultiFactorStrategy(symbols, factor_data, factor_weights, **kwargs)


# create_factor_portfolio_manager 移到了 portfolio 模块