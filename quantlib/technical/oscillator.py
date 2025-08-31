"""
震荡指标模块 - 包含各种震荡类技术指标
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base import TechnicalIndicator, PriceBasedIndicator


class RSI(PriceBasedIndicator):
    """相对强弱指标 (Relative Strength Index)"""
    
    def calculate(self, period: int = 14) -> Dict[str, pd.Series]:
        """
        计算RSI指标
        
        Args:
            period: 计算周期
        """
        # 计算价格变化
        delta = self.price_series.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均涨幅和跌幅
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        self.results = {
            'RSI': rsi,
            'Avg_Gain': avg_gain,
            'Avg_Loss': avg_loss
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成RSI交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.price_series
        
        rsi = self.results['RSI']
        
        # 超买超卖信号
        signals['signal'] = 0
        signals.loc[rsi <= 30, 'signal'] = 1  # 超卖买入
        signals.loc[rsi >= 70, 'signal'] = -1  # 超买卖出
        
        # 极端区域
        signals.loc[rsi <= 20, 'signal'] = 2  # 极度超卖
        signals.loc[rsi >= 80, 'signal'] = -2  # 极度超买
        
        # RSI背离信号（简化版）
        price_high = self.price_series.rolling(20).max()
        price_low = self.price_series.rolling(20).min()
        rsi_high = rsi.rolling(20).max()
        rsi_low = rsi.rolling(20).min()
        
        # 顶背离：价格新高但RSI没有新高
        signals['bearish_divergence'] = (self.price_series == price_high) & (rsi < rsi_high.shift(1))
        # 底背离：价格新低但RSI没有新低  
        signals['bullish_divergence'] = (self.price_series == price_low) & (rsi > rsi_low.shift(1))
        
        return signals


class KDJ(TechnicalIndicator):
    """KDJ随机指标"""
    
    def calculate(self, k_period: int = 9, d_period: int = 3, j_period: int = 3) -> Dict[str, pd.Series]:
        """
        计算KDJ指标
        
        Args:
            k_period: K值计算周期
            d_period: D值平滑周期
            j_period: J值计算周期
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # 计算最高价和最低价
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # 计算RSV（随机值）
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        
        # 计算K值（对RSV进行平滑）
        k_values = rsv.ewm(com=2).mean()  # 等同于3日移动平均的权重
        
        # 计算D值（对K值进行平滑）
        d_values = k_values.ewm(com=2).mean()
        
        # 计算J值
        j_values = 3 * k_values - 2 * d_values
        
        self.results = {
            'K': k_values,
            'D': d_values,
            'J': j_values,
            'RSV': rsv
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成KDJ交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        k = self.results['K']
        d = self.results['D']
        j = self.results['J']
        
        # KD金叉死叉
        signals['kd_signal'] = 0
        signals.loc[k > d, 'kd_signal'] = 1
        signals.loc[k < d, 'kd_signal'] = -1
        
        # 超买超卖区域
        signals['overbought'] = (k > 80) & (d > 80)
        signals['oversold'] = (k < 20) & (d < 20)
        
        # 综合信号
        signals['signal'] = 0
        signals.loc[signals['oversold'] & (signals['kd_signal'] == 1), 'signal'] = 1  # 超卖区金叉买入
        signals.loc[signals['overbought'] & (signals['kd_signal'] == -1), 'signal'] = -1  # 超买区死叉卖出
        
        # KD线交叉点
        kd_cross = signals['kd_signal'].diff()
        signals['golden_cross'] = kd_cross == 2  # K线上穿D线
        signals['death_cross'] = kd_cross == -2  # K线下穿D线
        
        return signals


class Williams(TechnicalIndicator):
    """威廉指标 (Williams %R)"""
    
    def calculate(self, period: int = 14) -> Dict[str, pd.Series]:
        """
        计算威廉指标
        
        Args:
            period: 计算周期
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # 计算周期内最高价和最低价
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        # 计算威廉指标
        williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
        
        self.results = {
            'Williams_R': williams_r,
            'Highest_High': highest_high,
            'Lowest_Low': lowest_low
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成威廉指标交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        wr = self.results['Williams_R']
        
        # 超买超卖信号
        signals['signal'] = 0
        signals.loc[wr <= -80, 'signal'] = 1  # 超卖买入
        signals.loc[wr >= -20, 'signal'] = -1  # 超买卖出
        
        # 极端信号
        signals.loc[wr <= -90, 'signal'] = 2  # 极度超卖
        signals.loc[wr >= -10, 'signal'] = -2  # 极度超买
        
        return signals


class CCI(TechnicalIndicator):
    """顺势指标 (Commodity Channel Index)"""
    
    def calculate(self, period: int = 20, constant: float = 0.015) -> Dict[str, pd.Series]:
        """
        计算CCI指标
        
        Args:
            period: 计算周期
            constant: 常数因子
        """
        # 计算典型价格
        typical_price = self._typical_price()
        
        # 计算移动平均
        sma = typical_price.rolling(window=period).mean()
        
        # 计算平均绝对偏差
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        # 计算CCI
        cci = (typical_price - sma) / (constant * mad)
        
        self.results = {
            'CCI': cci,
            'Typical_Price': typical_price,
            'SMA': sma,
            'MAD': mad
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成CCI交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        cci = self.results['CCI']
        
        # 超买超卖信号
        signals['signal'] = 0
        signals.loc[cci <= -100, 'signal'] = 1  # 超卖买入
        signals.loc[cci >= 100, 'signal'] = -1  # 超买卖出
        
        # 强势信号
        signals.loc[cci <= -200, 'signal'] = 2  # 强烈超卖
        signals.loc[cci >= 200, 'signal'] = -2  # 强烈超买
        
        # CCI穿越零轴
        signals['zero_cross'] = 0
        signals.loc[cci > 0, 'zero_cross'] = 1
        signals.loc[cci < 0, 'zero_cross'] = -1
        
        return signals


class Stochastic(TechnicalIndicator):
    """随机震荡指标 (Stochastic Oscillator)"""
    
    def calculate(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Dict[str, pd.Series]:
        """
        计算随机震荡指标
        
        Args:
            k_period: %K计算周期
            d_period: %D平滑周期
            smooth_k: %K平滑周期
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # 计算最高价和最低价
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # 计算快速%K
        fast_k = (close - lowest_low) / (highest_high - lowest_low) * 100
        
        # 计算慢速%K（对快速%K进行平滑）
        slow_k = fast_k.rolling(window=smooth_k).mean()
        
        # 计算%D（对慢速%K进行平滑）
        slow_d = slow_k.rolling(window=d_period).mean()
        
        self.results = {
            'Fast_K': fast_k,
            'Slow_K': slow_k,
            'Slow_D': slow_d
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成随机震荡指标交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        slow_k = self.results['Slow_K']
        slow_d = self.results['Slow_D']
        
        # %K和%D交叉信号
        signals['stoch_signal'] = 0
        signals.loc[slow_k > slow_d, 'stoch_signal'] = 1
        signals.loc[slow_k < slow_d, 'stoch_signal'] = -1
        
        # 超买超卖区域
        signals['overbought'] = (slow_k > 80) & (slow_d > 80)
        signals['oversold'] = (slow_k < 20) & (slow_d < 20)
        
        # 综合信号
        signals['signal'] = 0
        signals.loc[signals['oversold'] & (slow_k > slow_d), 'signal'] = 1
        signals.loc[signals['overbought'] & (slow_k < slow_d), 'signal'] = -1
        
        return signals


class ROC(PriceBasedIndicator):
    """变动率指标 (Rate of Change)"""
    
    def calculate(self, period: int = 12) -> Dict[str, pd.Series]:
        """
        计算ROC指标
        
        Args:
            period: 计算周期
        """
        # 计算变动率
        roc = (self.price_series / self.price_series.shift(period) - 1) * 100
        
        # 计算ROC的移动平均
        roc_ma = roc.rolling(window=10).mean()
        
        self.results = {
            'ROC': roc,
            'ROC_MA': roc_ma
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成ROC交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.price_series
        
        roc = self.results['ROC']
        
        # ROC穿越零轴
        signals['signal'] = 0
        signals.loc[roc > 0, 'signal'] = 1  # 正值买入
        signals.loc[roc < 0, 'signal'] = -1  # 负值卖出
        
        # 极端值信号
        roc_std = roc.rolling(50).std()
        signals.loc[roc > 2 * roc_std, 'signal'] = -1  # 极度超买
        signals.loc[roc < -2 * roc_std, 'signal'] = 1  # 极度超卖
        
        return signals


class OscillatorIndicators:
    """震荡指标集合类"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def rsi(self, price_type: str = 'close', period: int = 14) -> RSI:
        """相对强弱指标"""
        rsi = RSI(self.data, price_type)
        rsi.calculate(period)
        return rsi
    
    def kdj(self, k_period: int = 9, d_period: int = 3, j_period: int = 3) -> KDJ:
        """KDJ随机指标"""
        kdj = KDJ(self.data)
        kdj.calculate(k_period, d_period, j_period)
        return kdj
    
    def williams(self, period: int = 14) -> Williams:
        """威廉指标"""
        williams = Williams(self.data)
        williams.calculate(period)
        return williams
    
    def cci(self, period: int = 20, constant: float = 0.015) -> CCI:
        """顺势指标"""
        cci = CCI(self.data)
        cci.calculate(period, constant)
        return cci
    
    def stochastic(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Stochastic:
        """随机震荡指标"""
        stoch = Stochastic(self.data)
        stoch.calculate(k_period, d_period, smooth_k)
        return stoch
    
    def roc(self, price_type: str = 'close', period: int = 12) -> ROC:
        """变动率指标"""
        roc = ROC(self.data, price_type)
        roc.calculate(period)
        return roc