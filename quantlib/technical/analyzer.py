"""
技术指标综合分析器
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from .trend import TrendIndicators
from .oscillator import OscillatorIndicators
from .volume import VolumeIndicators


class TechnicalAnalyzer:
    """技术指标综合分析器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化技术分析器
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        self.data = data.copy()
        self._validate_data()
        
        # 初始化各类指标分析器
        self.trend = TrendIndicators(self.data)
        self.oscillator = OscillatorIndicators(self.data)
        if 'volume' in self.data.columns:
            self.volume = VolumeIndicators(self.data)
        else:
            self.volume = None
        
        self.indicators = {}
        self.signals = {}
        self.analysis_results = {}
    
    def _validate_data(self):
        """验证数据格式"""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        if self.data.empty:
            raise ValueError("数据不能为空")
    
    def calculate_all_indicators(self):
        """计算所有技术指标"""
        print("计算技术指标...")
        
        # 趋势指标
        self.indicators['ma'] = self.trend.moving_averages()
        self.indicators['macd'] = self.trend.macd()
        self.indicators['bb'] = self.trend.bollinger_bands()
        self.indicators['adx'] = self.trend.adx()
        self.indicators['sar'] = self.trend.parabolic_sar()
        
        # 震荡指标
        self.indicators['rsi'] = self.oscillator.rsi()
        self.indicators['kdj'] = self.oscillator.kdj()
        self.indicators['williams'] = self.oscillator.williams()
        self.indicators['cci'] = self.oscillator.cci()
        self.indicators['stoch'] = self.oscillator.stochastic()
        self.indicators['roc'] = self.oscillator.roc()
        
        # 成交量指标
        if self.volume:
            self.indicators['obv'] = self.volume.obv()
            self.indicators['vpt'] = self.volume.vpt()
            self.indicators['vwap'] = self.volume.vwap()
            self.indicators['cmf'] = self.volume.chaikin_money_flow()
            self.indicators['ad'] = self.volume.accumulation_distribution()
        
        print("技术指标计算完成")
    
    def generate_all_signals(self):
        """生成所有交易信号"""
        if not self.indicators:
            self.calculate_all_indicators()
        
        print("生成交易信号...")
        
        for name, indicator in self.indicators.items():
            try:
                self.signals[name] = indicator.get_signals()
            except Exception as e:
                print(f"生成{name}信号时出错: {e}")
        
        print("交易信号生成完成")
    
    def get_current_signals(self) -> Dict[str, Dict]:
        """获取当前最新信号"""
        if not self.signals:
            self.generate_all_signals()
        
        current_signals = {}
        
        for name, signal_df in self.signals.items():
            if not signal_df.empty:
                latest = signal_df.iloc[-1]
                current_signals[name] = latest.to_dict()
        
        return current_signals
    
    def get_consensus_signal(self) -> Tuple[int, float, Dict]:
        """
        获取综合信号
        
        Returns:
            综合信号(-2到2), 信号强度(0到1), 详细分析
        """
        current_signals = self.get_current_signals()
        
        signals_summary = {
            'trend_signals': [],
            'oscillator_signals': [],
            'volume_signals': []
        }
        
        # 分类统计信号
        trend_indicators = ['ma', 'macd', 'bb', 'adx', 'sar']
        oscillator_indicators = ['rsi', 'kdj', 'williams', 'cci', 'stoch', 'roc']
        volume_indicators = ['obv', 'vpt', 'cmf', 'ad']
        
        for name, signals in current_signals.items():
            signal_value = signals.get('signal', 0)
            
            if name in trend_indicators:
                signals_summary['trend_signals'].append(signal_value)
            elif name in oscillator_indicators:
                signals_summary['oscillator_signals'].append(signal_value)
            elif name in volume_indicators:
                signals_summary['volume_signals'].append(signal_value)
        
        # 计算各类别的平均信号
        trend_avg = np.mean(signals_summary['trend_signals']) if signals_summary['trend_signals'] else 0
        oscillator_avg = np.mean(signals_summary['oscillator_signals']) if signals_summary['oscillator_signals'] else 0
        volume_avg = np.mean(signals_summary['volume_signals']) if signals_summary['volume_signals'] else 0
        
        # 加权综合信号（趋势权重更高）
        consensus_signal = 0.5 * trend_avg + 0.3 * oscillator_avg + 0.2 * volume_avg
        
        # 计算信号强度
        all_signals = []
        all_signals.extend(signals_summary['trend_signals'])
        all_signals.extend(signals_summary['oscillator_signals'])
        all_signals.extend(signals_summary['volume_signals'])
        
        if all_signals:
            signal_consistency = 1 - (np.std(all_signals) / 2)  # 信号一致性
            signal_strength = abs(consensus_signal) * signal_consistency
        else:
            signal_strength = 0
        
        # 详细分析
        analysis = {
            'trend_signal': trend_avg,
            'oscillator_signal': oscillator_avg,
            'volume_signal': volume_avg,
            'total_indicators': len(all_signals),
            'bullish_count': sum(1 for s in all_signals if s > 0),
            'bearish_count': sum(1 for s in all_signals if s < 0),
            'neutral_count': sum(1 for s in all_signals if s == 0),
            'signal_consistency': signal_consistency if all_signals else 0
        }
        
        return int(np.clip(consensus_signal, -2, 2)), min(signal_strength, 1), analysis
    
    def identify_support_resistance(self, window: int = 20, min_touches: int = 2) -> Dict[str, List[float]]:
        """
        识别支撑阻力位
        
        Args:
            window: 查找窗口
            min_touches: 最小触及次数
        """
        high = self.data['high']
        low = self.data['low']
        
        # 寻找局部高低点
        local_maxima = []
        local_minima = []
        
        for i in range(window, len(self.data) - window):
            # 局部高点
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                local_maxima.append(high.iloc[i])
            
            # 局部低点
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                local_minima.append(low.iloc[i])
        
        # 识别支撑阻力位（价格聚集区域）
        def find_levels(prices, tolerance=0.02):
            levels = []
            for price in prices:
                # 统计在容差范围内的价格数量
                touches = sum(1 for p in prices if abs(p - price) / price <= tolerance)
                if touches >= min_touches:
                    levels.append(price)
            
            # 去重相近的价位
            unique_levels = []
            for level in sorted(set(levels)):
                if not unique_levels or abs(level - unique_levels[-1]) / level > tolerance:
                    unique_levels.append(level)
            
            return unique_levels
        
        resistance_levels = find_levels(local_maxima)
        support_levels = find_levels(local_minima)
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    def generate_analysis_report(self) -> str:
        """生成技术分析报告"""
        if not self.indicators:
            self.calculate_all_indicators()
        
        consensus_signal, signal_strength, analysis = self.get_consensus_signal()
        support_resistance = self.identify_support_resistance()
        current_price = self.data['close'].iloc[-1]
        
        # 信号解读
        signal_interpretation = {
            2: "强烈看涨",
            1: "看涨",
            0: "中性",
            -1: "看跌",
            -2: "强烈看跌"
        }
        
        report = f"""
技术分析报告
{'='*50}

当前价格: {current_price:.2f}

综合信号: {signal_interpretation[consensus_signal]} (信号强度: {signal_strength:.2f})

信号分解:
• 趋势信号: {analysis['trend_signal']:.2f}
• 震荡信号: {analysis['oscillator_signal']:.2f}
• 成交量信号: {analysis['volume_signal']:.2f}

信号统计:
• 看涨指标: {analysis['bullish_count']} 个
• 看跌指标: {analysis['bearish_count']} 个
• 中性指标: {analysis['neutral_count']} 个
• 信号一致性: {analysis['signal_consistency']:.2f}

支撑阻力位:
• 支撑位: {', '.join([f'{level:.2f}' for level in support_resistance['support_levels'][-3:]])}
• 阻力位: {', '.join([f'{level:.2f}' for level in support_resistance['resistance_levels'][-3:]])}

关键指标当前状态:
"""
        
        # 添加关键指标状态
        current_signals = self.get_current_signals()
        key_indicators = ['rsi', 'macd', 'kdj', 'bb']
        
        for indicator in key_indicators:
            if indicator in current_signals:
                signals = current_signals[indicator]
                signal_value = signals.get('signal', 0)
                
                if indicator == 'rsi':
                    rsi_value = self.indicators['rsi'].results['RSI'].iloc[-1]
                    report += f"• RSI({rsi_value:.1f}): {signal_interpretation.get(signal_value, '中性')}\n"
                elif indicator == 'macd':
                    macd_value = self.indicators['macd'].results['MACD'].iloc[-1]
                    signal_line = self.indicators['macd'].results['Signal'].iloc[-1]
                    report += f"• MACD({macd_value:.4f}): {signal_interpretation.get(signal_value, '中性')}\n"
                elif indicator == 'kdj':
                    k_value = self.indicators['kdj'].results['K'].iloc[-1]
                    d_value = self.indicators['kdj'].results['D'].iloc[-1]
                    report += f"• KDJ(K:{k_value:.1f}, D:{d_value:.1f}): {signal_interpretation.get(signal_value, '中性')}\n"
                elif indicator == 'bb':
                    bb_pos = self.indicators['bb'].results['BB_Position'].iloc[-1]
                    report += f"• 布林带位置({bb_pos:.1f}%): {signal_interpretation.get(signal_value, '中性')}\n"
        
        return report
    
    def plot_analysis(self, figsize: Tuple[int, int] = (15, 12)):
        """绘制技术分析图表"""
        if not self.indicators:
            self.calculate_all_indicators()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(4, 1, figsize=figsize, height_ratios=[3, 1, 1, 1])
        
        # 1. 价格和趋势指标
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['close'], label='Close Price', linewidth=1.5)
        
        # 移动平均线
        if 'ma' in self.indicators:
            ma_results = self.indicators['ma'].results
            for key, ma_line in ma_results.items():
                if 'SMA' in key:
                    period = key.split('_')[1]
                    ax1.plot(self.data.index, ma_line, label=f'MA{period}', alpha=0.7)
        
        # 布林带
        if 'bb' in self.indicators:
            bb_results = self.indicators['bb'].results
            ax1.plot(self.data.index, bb_results['Upper_Band'], 'r--', alpha=0.6, label='BB Upper')
            ax1.plot(self.data.index, bb_results['Lower_Band'], 'g--', alpha=0.6, label='BB Lower')
            ax1.fill_between(self.data.index, bb_results['Upper_Band'], bb_results['Lower_Band'], alpha=0.1)
        
        ax1.set_title('Price Trend & Technical Indicators')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. MACD
        ax2 = axes[1]
        if 'macd' in self.indicators:
            macd_results = self.indicators['macd'].results
            ax2.plot(self.data.index, macd_results['MACD'], label='MACD', color='blue')
            ax2.plot(self.data.index, macd_results['Signal'], label='Signal', color='red')
            ax2.bar(self.data.index, macd_results['Histogram'], label='Histogram', alpha=0.6)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_title('MACD')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. RSI和KDJ
        ax3 = axes[2]
        if 'rsi' in self.indicators:
            rsi_result = self.indicators['rsi'].results['RSI']
            ax3.plot(self.data.index, rsi_result, label='RSI', color='purple')
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax3.set_ylim(0, 100)
        
        if 'kdj' in self.indicators:
            kdj_results = self.indicators['kdj'].results
            ax3_twin = ax3.twinx()
            ax3_twin.plot(self.data.index, kdj_results['K'], label='K', color='orange', alpha=0.7)
            ax3_twin.plot(self.data.index, kdj_results['D'], label='D', color='cyan', alpha=0.7)
            ax3_twin.set_ylim(0, 100)
        
        ax3.set_title('RSI & KDJ')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Volume indicators
        ax4 = axes[3]
        if 'volume' in self.data.columns:
            ax4.bar(self.data.index, self.data['volume'], alpha=0.6, label='Volume')
            
            if 'obv' in self.indicators:
                ax4_twin = ax4.twinx()
                obv_result = self.indicators['obv'].results['OBV']
                ax4_twin.plot(self.data.index, obv_result, label='OBV', color='red')
                ax4_twin.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        
        ax4.set_title('Volume & OBV')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def backtest_signals(self, initial_capital: float = 100000, 
                        transaction_cost: float = 0.001) -> Dict:
        """
        信号回测
        
        Args:
            initial_capital: 初始资金
            transaction_cost: 交易成本率
        """
        if not self.signals:
            self.generate_all_signals()
        
        consensus_signal, _, _ = self.get_consensus_signal()
        
        # 简化的回测逻辑
        results = {
            'total_return': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        # 这里应该实现完整的回测逻辑
        # 由于篇幅限制，这里只提供框架
        
        return results