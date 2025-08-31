"""
趋势指标模块 - 包含各种趋势分析技术指标
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .base import TechnicalIndicator, PriceBasedIndicator


class MovingAverages(PriceBasedIndicator):
    """移动平均线指标"""
    
    def calculate(self, periods: list = [5, 10, 20, 50, 200]) -> Dict[str, pd.Series]:
        """
        计算移动平均线
        
        Args:
            periods: 移动平均线周期列表
        """
        self.results = {}
        
        for period in periods:
            # 简单移动平均线
            self.results[f'SMA_{period}'] = self._sma(self.price_series, period)
            # 指数移动平均线
            self.results[f'EMA_{period}'] = self._ema(self.price_series, period)
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成移动平均线交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.price_series
        
        # 获取短期和长期均线
        short_ma = None
        long_ma = None
        
        for key in self.results.keys():
            if 'SMA_5' in key or 'EMA_5' in key:
                short_ma = self.results[key]
            elif 'SMA_20' in key or 'EMA_20' in key:
                long_ma = self.results[key]
        
        if short_ma is not None and long_ma is not None:
            # 金叉死叉信号
            signals['signal'] = 0
            signals.loc[short_ma > long_ma, 'signal'] = 1  # 金叉买入
            signals.loc[short_ma < long_ma, 'signal'] = -1  # 死叉卖出
            
            # 信号变化点
            signals['position_change'] = signals['signal'].diff()
            signals['buy_signal'] = signals['position_change'] == 2  # -1 to 1
            signals['sell_signal'] = signals['position_change'] == -2  # 1 to -1
        
        return signals


class MACD(PriceBasedIndicator):
    """MACD指标 (Moving Average Convergence Divergence)"""
    
    def calculate(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        计算MACD指标
        
        Args:
            fast_period: 快线周期
            slow_period: 慢线周期  
            signal_period: 信号线周期
        """
        # 快线和慢线
        fast_ema = self._ema(self.price_series, fast_period)
        slow_ema = self._ema(self.price_series, slow_period)
        
        # MACD线
        macd_line = fast_ema - slow_ema
        
        # 信号线
        signal_line = self._ema(macd_line, signal_period)
        
        # MACD柱
        histogram = macd_line - signal_line
        
        self.results = {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成MACD交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.price_series
        
        macd = self.results['MACD']
        signal = self.results['Signal']
        histogram = self.results['Histogram']
        
        # MACD线穿越信号线
        signals['macd_signal'] = 0
        signals.loc[macd > signal, 'macd_signal'] = 1
        signals.loc[macd < signal, 'macd_signal'] = -1
        
        # MACD柱状图穿越零轴
        signals['histogram_signal'] = 0
        signals.loc[histogram > 0, 'histogram_signal'] = 1
        signals.loc[histogram < 0, 'histogram_signal'] = -1
        
        # 综合信号
        signals['signal'] = (signals['macd_signal'] + signals['histogram_signal']) / 2
        
        # 买卖点
        macd_change = signals['macd_signal'].diff()
        signals['buy_signal'] = macd_change == 2  # -1 to 1
        signals['sell_signal'] = macd_change == -2  # 1 to -1
        
        return signals


class BollingerBands(PriceBasedIndicator):
    """布林带指标"""
    
    def calculate(self, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        计算布林带
        
        Args:
            period: 移动平均线周期
            std_dev: 标准差倍数
        """
        # 中轨（移动平均线）
        middle_band = self._sma(self.price_series, period)
        
        # 标准差
        std = self.price_series.rolling(window=period).std()
        
        # 上轨和下轨
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # 带宽和位置
        bandwidth = (upper_band - lower_band) / middle_band * 100
        bb_position = (self.price_series - lower_band) / (upper_band - lower_band) * 100
        
        self.results = {
            'Upper_Band': upper_band,
            'Middle_Band': middle_band,
            'Lower_Band': lower_band,
            'Bandwidth': bandwidth,
            'BB_Position': bb_position
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成布林带交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.price_series
        
        upper = self.results['Upper_Band']
        lower = self.results['Lower_Band']
        position = self.results['BB_Position']
        
        # 超买超卖信号
        signals['signal'] = 0
        signals.loc[self.price_series <= lower, 'signal'] = 1  # 超卖买入
        signals.loc[self.price_series >= upper, 'signal'] = -1  # 超买卖出
        
        # 基于布林带位置的信号
        signals['bb_signal'] = 0
        signals.loc[position <= 20, 'bb_signal'] = 1  # 低位买入
        signals.loc[position >= 80, 'bb_signal'] = -1  # 高位卖出
        
        # 布林带收缩/扩张
        bandwidth = self.results['Bandwidth']
        signals['squeeze'] = bandwidth < bandwidth.rolling(20).mean()
        
        return signals


class ADX(TechnicalIndicator):
    """ADX平均趋向指标 (Average Directional Index)"""
    
    def calculate(self, period: int = 14) -> Dict[str, pd.Series]:
        """
        计算ADX指标
        
        Args:
            period: 计算周期
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # 计算+DM和-DM
        plus_dm = pd.Series(np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0))
        minus_dm = pd.Series(np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0))
        
        # 真实波动范围
        tr = self._true_range()
        
        # 平滑处理
        plus_dm_smooth = plus_dm.ewm(alpha=1/period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period).mean()
        tr_smooth = tr.ewm(alpha=1/period).mean()
        
        # 计算+DI和-DI
        plus_di = (plus_dm_smooth / tr_smooth) * 100
        minus_di = (minus_dm_smooth / tr_smooth) * 100
        
        # 计算DX
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        
        # 计算ADX
        adx = dx.ewm(alpha=1/period).mean()
        
        self.results = {
            'Plus_DI': plus_di,
            'Minus_DI': minus_di,
            'ADX': adx
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成ADX交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        plus_di = self.results['Plus_DI']
        minus_di = self.results['Minus_DI']
        adx = self.results['ADX']
        
        # 趋势强度信号
        signals['trend_strength'] = 0
        signals.loc[adx > 25, 'trend_strength'] = 1  # 强趋势
        signals.loc[adx < 20, 'trend_strength'] = -1  # 弱趋势
        
        # 方向信号
        signals['direction'] = 0
        signals.loc[plus_di > minus_di, 'direction'] = 1  # 上升趋势
        signals.loc[plus_di < minus_di, 'direction'] = -1  # 下降趋势
        
        # 综合信号（只在强趋势时交易）
        signals['signal'] = 0
        signals.loc[(signals['trend_strength'] == 1) & (signals['direction'] == 1), 'signal'] = 1
        signals.loc[(signals['trend_strength'] == 1) & (signals['direction'] == -1), 'signal'] = -1
        
        return signals


class ParabolicSAR(TechnicalIndicator):
    """抛物线SAR指标"""
    
    def calculate(self, af_initial: float = 0.02, af_increment: float = 0.02, af_maximum: float = 0.2) -> Dict[str, pd.Series]:
        """
        计算抛物线SAR
        
        Args:
            af_initial: 初始加速因子
            af_increment: 加速因子增量
            af_maximum: 最大加速因子
        """
        high = self.data['high']
        low = self.data['low']
        
        sar = pd.Series(index=self.data.index, dtype=float)
        trend = pd.Series(index=self.data.index, dtype=int)
        af = pd.Series(index=self.data.index, dtype=float)
        ep = pd.Series(index=self.data.index, dtype=float)
        
        # 初始化
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # 1为上升趋势，-1为下降趋势
        af.iloc[0] = af_initial
        ep.iloc[0] = high.iloc[0]
        
        for i in range(1, len(self.data)):
            # 计算SAR
            sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
            
            # 检查趋势反转
            if trend.iloc[i-1] == 1:  # 上升趋势
                if low.iloc[i] <= sar.iloc[i]:
                    # 趋势反转为下降
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]  # SAR设为前期极值
                    af.iloc[i] = af_initial
                    ep.iloc[i] = low.iloc[i]
                else:
                    # 继续上升趋势
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_increment, af_maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
                    
                    # SAR不能高于前两期的最低价
                    sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
            
            else:  # 下降趋势
                if high.iloc[i] >= sar.iloc[i]:
                    # 趋势反转为上升
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]  # SAR设为前期极值
                    af.iloc[i] = af_initial
                    ep.iloc[i] = high.iloc[i]
                else:
                    # 继续下降趋势
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_increment, af_maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
                    
                    # SAR不能低于前两期的最高价
                    sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
        
        self.results = {
            'SAR': sar,
            'Trend': trend,
            'AF': af,
            'EP': ep
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成SAR交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        sar = self.results['SAR']
        trend = self.results['Trend']
        
        signals['signal'] = trend
        
        # 趋势反转点
        trend_change = trend.diff()
        signals['buy_signal'] = trend_change == 2  # -1 to 1
        signals['sell_signal'] = trend_change == -2  # 1 to -1
        
        # 止损位
        signals['stop_loss'] = sar
        
        return signals


class TrendIndicators:
    """趋势指标集合类"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def moving_averages(self, price_type: str = 'close', periods: list = [5, 10, 20, 50, 200]) -> MovingAverages:
        """移动平均线"""
        ma = MovingAverages(self.data, price_type)
        ma.calculate(periods)
        return ma
    
    def macd(self, price_type: str = 'close', fast: int = 12, slow: int = 26, signal: int = 9) -> MACD:
        """MACD指标"""
        macd = MACD(self.data, price_type)
        macd.calculate(fast, slow, signal)
        return macd
    
    def bollinger_bands(self, price_type: str = 'close', period: int = 20, std_dev: float = 2.0) -> BollingerBands:
        """布林带"""
        bb = BollingerBands(self.data, price_type)
        bb.calculate(period, std_dev)
        return bb
    
    def adx(self, period: int = 14) -> ADX:
        """ADX平均趋向指标"""
        adx = ADX(self.data)
        adx.calculate(period)
        return adx
    
    def parabolic_sar(self, af_initial: float = 0.02, af_increment: float = 0.02, af_maximum: float = 0.2) -> ParabolicSAR:
        """抛物线SAR"""
        sar = ParabolicSAR(self.data)
        sar.calculate(af_initial, af_increment, af_maximum)
        return sar