"""
成交量指标模块 - 包含各种基于成交量的技术指标
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base import TechnicalIndicator, VolumeBasedIndicator


class OBV(VolumeBasedIndicator):
    """能量潮指标 (On-Balance Volume)"""
    
    def calculate(self) -> Dict[str, pd.Series]:
        """计算OBV指标"""
        close = self.data['close']
        volume = self.volume_series
        
        # 计算价格变化方向
        price_change = close.diff()
        
        # 计算OBV
        obv = pd.Series(index=self.data.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(self.data)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # OBV移动平均线
        obv_ma = obv.rolling(window=20).mean()
        
        self.results = {
            'OBV': obv,
            'OBV_MA': obv_ma
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成OBV交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        obv = self.results['OBV']
        obv_ma = self.results['OBV_MA']
        
        # OBV与移动平均线交叉
        signals['signal'] = 0
        signals.loc[obv > obv_ma, 'signal'] = 1  # OBV上穿均线
        signals.loc[obv < obv_ma, 'signal'] = -1  # OBV下穿均线
        
        # 价量背离
        price_trend = self.data['close'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        obv_trend = obv.rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        signals['divergence'] = 0
        signals.loc[(price_trend == 1) & (obv_trend == -1), 'divergence'] = -1  # 顶背离
        signals.loc[(price_trend == -1) & (obv_trend == 1), 'divergence'] = 1   # 底背离
        
        return signals


class VPT(VolumeBasedIndicator):
    """量价趋势指标 (Volume Price Trend)"""
    
    def calculate(self) -> Dict[str, pd.Series]:
        """计算VPT指标"""
        close = self.data['close']
        volume = self.volume_series
        
        # 计算价格变化率
        price_change_rate = close.pct_change()
        
        # 计算VPT
        vpt = (price_change_rate * volume).cumsum()
        
        # VPT移动平均线
        vpt_ma = vpt.rolling(window=20).mean()
        
        self.results = {
            'VPT': vpt,
            'VPT_MA': vpt_ma,
            'Price_Change_Rate': price_change_rate
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成VPT交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        vpt = self.results['VPT']
        vpt_ma = self.results['VPT_MA']
        
        # VPT与移动平均线交叉
        signals['signal'] = 0
        signals.loc[vpt > vpt_ma, 'signal'] = 1
        signals.loc[vpt < vpt_ma, 'signal'] = -1
        
        return signals


class VolumeSMA(VolumeBasedIndicator):
    """成交量移动平均线"""
    
    def calculate(self, periods: list = [5, 10, 20, 50]) -> Dict[str, pd.Series]:
        """
        计算成交量移动平均线
        
        Args:
            periods: 移动平均线周期
        """
        self.results = {}
        
        for period in periods:
            self.results[f'Volume_SMA_{period}'] = self.volume_series.rolling(window=period).mean()
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成成交量信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        signals['volume'] = self.volume_series
        
        # 成交量突破信号
        vol_sma_20 = self.results.get('Volume_SMA_20')
        if vol_sma_20 is not None:
            signals['volume_signal'] = 0
            signals.loc[self.volume_series > vol_sma_20 * 1.5, 'volume_signal'] = 1  # 放量
            signals.loc[self.volume_series < vol_sma_20 * 0.5, 'volume_signal'] = -1  # 缩量
        
        return signals


class VWAP(VolumeBasedIndicator):
    """成交量加权平均价格 (Volume Weighted Average Price)"""
    
    def calculate(self, period: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        计算VWAP
        
        Args:
            period: 计算周期，如果为None则计算累积VWAP
        """
        typical_price = self._typical_price()
        volume = self.volume_series
        
        if period is None:
            # 累积VWAP
            cumulative_pv = (typical_price * volume).cumsum()
            cumulative_volume = volume.cumsum()
            vwap = cumulative_pv / cumulative_volume
        else:
            # 滚动VWAP
            pv = typical_price * volume
            vwap = pv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # 计算标准差带
        if period:
            price_vol_var = ((typical_price - vwap) ** 2 * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
            vwap_std = np.sqrt(price_vol_var)
        else:
            price_vol_var = ((typical_price - vwap) ** 2 * volume).expanding().sum() / volume.expanding().sum()
            vwap_std = np.sqrt(price_vol_var)
        
        upper_band = vwap + vwap_std
        lower_band = vwap - vwap_std
        
        self.results = {
            'VWAP': vwap,
            'VWAP_Upper': upper_band,
            'VWAP_Lower': lower_band,
            'VWAP_Std': vwap_std
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成VWAP交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        vwap = self.results['VWAP']
        upper_band = self.results['VWAP_Upper']
        lower_band = self.results['VWAP_Lower']
        
        # VWAP信号
        signals['signal'] = 0
        signals.loc[self.data['close'] > vwap, 'signal'] = 1  # 价格在VWAP之上
        signals.loc[self.data['close'] < vwap, 'signal'] = -1  # 价格在VWAP之下
        
        # 带状信号
        signals.loc[self.data['close'] > upper_band, 'signal'] = -1  # 价格过高，卖出
        signals.loc[self.data['close'] < lower_band, 'signal'] = 1   # 价格过低，买入
        
        return signals


class ChaikinMoneyFlow(VolumeBasedIndicator):
    """蔡金资金流量指标 (Chaikin Money Flow)"""
    
    def calculate(self, period: int = 20) -> Dict[str, pd.Series]:
        """
        计算CMF指标
        
        Args:
            period: 计算周期
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        volume = self.volume_series
        
        # 计算资金流量乘数
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = mf_multiplier.fillna(0)  # 处理high=low的情况
        
        # 计算资金流量量
        mf_volume = mf_multiplier * volume
        
        # 计算CMF
        cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        self.results = {
            'CMF': cmf,
            'MF_Multiplier': mf_multiplier,
            'MF_Volume': mf_volume
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成CMF交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        cmf = self.results['CMF']
        
        # CMF信号
        signals['signal'] = 0
        signals.loc[cmf > 0.1, 'signal'] = 1   # 资金流入
        signals.loc[cmf < -0.1, 'signal'] = -1  # 资金流出
        
        # 极端信号
        signals.loc[cmf > 0.25, 'signal'] = 2   # 强烈买入
        signals.loc[cmf < -0.25, 'signal'] = -2  # 强烈卖出
        
        return signals


class AccumulationDistribution(VolumeBasedIndicator):
    """累积/派发线 (Accumulation/Distribution Line)"""
    
    def calculate(self) -> Dict[str, pd.Series]:
        """计算A/D线"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        volume = self.volume_series
        
        # 计算资金流量乘数
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = mf_multiplier.fillna(0)
        
        # 计算资金流量量
        mf_volume = mf_multiplier * volume
        
        # 计算累积/派发线
        ad_line = mf_volume.cumsum()
        
        # A/D线的移动平均
        ad_ma = ad_line.rolling(window=20).mean()
        
        self.results = {
            'AD_Line': ad_line,
            'AD_MA': ad_ma,
            'MF_Volume': mf_volume
        }
        
        return self.results
    
    def _generate_signals(self) -> pd.DataFrame:
        """生成A/D线交易信号"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['close']
        
        ad_line = self.results['AD_Line']
        ad_ma = self.results['AD_MA']
        
        # A/D线与移动平均线交叉
        signals['signal'] = 0
        signals.loc[ad_line > ad_ma, 'signal'] = 1
        signals.loc[ad_line < ad_ma, 'signal'] = -1
        
        # 价格与A/D线背离
        price_trend = self.data['close'].rolling(20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        ad_trend = ad_line.rolling(20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        signals['divergence'] = 0
        signals.loc[(price_trend == 1) & (ad_trend == -1), 'divergence'] = -1  # 顶背离
        signals.loc[(price_trend == -1) & (ad_trend == 1), 'divergence'] = 1   # 底背离
        
        return signals


class VolumeIndicators:
    """成交量指标集合类"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        if 'volume' not in data.columns:
            raise ValueError("成交量指标需要volume数据")
    
    def obv(self) -> OBV:
        """能量潮指标"""
        obv = OBV(self.data)
        obv.calculate()
        return obv
    
    def vpt(self) -> VPT:
        """量价趋势指标"""
        vpt = VPT(self.data)
        vpt.calculate()
        return vpt
    
    def volume_sma(self, periods: list = [5, 10, 20, 50]) -> VolumeSMA:
        """成交量移动平均线"""
        vol_sma = VolumeSMA(self.data)
        vol_sma.calculate(periods)
        return vol_sma
    
    def vwap(self, period: Optional[int] = None) -> VWAP:
        """成交量加权平均价格"""
        vwap = VWAP(self.data)
        vwap.calculate(period)
        return vwap
    
    def chaikin_money_flow(self, period: int = 20) -> ChaikinMoneyFlow:
        """蔡金资金流量指标"""
        cmf = ChaikinMoneyFlow(self.data)
        cmf.calculate(period)
        return cmf
    
    def accumulation_distribution(self) -> AccumulationDistribution:
        """累积/派发线"""
        ad = AccumulationDistribution(self.data)
        ad.calculate()
        return ad