"""
订单簿数据分析模块 - 提供市场微观结构分析功能
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class OrderBookAnalyzer:
    """订单簿数据分析器"""

    def __init__(self, order_book_data: Dict[str, Any]):
        """
        初始化订单簿分析器

        Args:
            order_book_data: 订单簿数据，包含bids, asks, spread等信息
        """
        self.order_book = order_book_data
        self.bids = order_book_data.get('bids', [])
        self.asks = order_book_data.get('asks', [])
        self.symbol = order_book_data.get('symbol', 'Unknown')

    def get_market_depth_metrics(self) -> Dict[str, Any]:
        """计算市场深度指标"""
        metrics = {
            'symbol': self.symbol,
            'timestamp': self.order_book.get('timestamp'),
            'total_bid_volume': 0,
            'total_ask_volume': 0,
            'total_volume': 0,
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
            'imbalance_ratio': 0,  # 买卖盘量失衡比率
            'volume_weighted_bid': 0,
            'volume_weighted_ask': 0,
            'depth_5_bid': 0,  # 买5档总量
            'depth_5_ask': 0,  # 卖5档总量
        }

        if self.bids:
            metrics['total_bid_volume'] = sum(bid['volume'] for bid in self.bids)
            # 计算成交量加权买价
            if metrics['total_bid_volume'] > 0:
                metrics['volume_weighted_bid'] = sum(
                    bid['price'] * bid['volume'] for bid in self.bids
                ) / metrics['total_bid_volume']

        if self.asks:
            metrics['total_ask_volume'] = sum(ask['volume'] for ask in self.asks)
            # 计算成交量加权卖价
            if metrics['total_ask_volume'] > 0:
                metrics['volume_weighted_ask'] = sum(
                    ask['price'] * ask['volume'] for ask in self.asks
                ) / metrics['total_ask_volume']

        metrics['total_volume'] = metrics['total_bid_volume'] + metrics['total_ask_volume']

        # 计算买卖盘失衡比率
        if metrics['total_volume'] > 0:
            metrics['imbalance_ratio'] = (
                metrics['total_bid_volume'] - metrics['total_ask_volume']
            ) / metrics['total_volume']

        # 计算5档深度
        metrics['depth_5_bid'] = sum(bid['volume'] for bid in self.bids[:5])
        metrics['depth_5_ask'] = sum(ask['volume'] for ask in self.asks[:5])

        return metrics

    def calculate_spread_metrics(self) -> Dict[str, float]:
        """计算买卖价差相关指标"""
        if not self.bids or not self.asks:
            return {}

        best_bid = max(self.bids, key=lambda x: x['price'])['price']
        best_ask = min(self.asks, key=lambda x: x['price'])['price']

        spread_absolute = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_relative = spread_absolute / mid_price if mid_price > 0 else 0

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread_absolute': spread_absolute,
            'spread_relative': spread_relative * 100,  # 百分比
            'spread_bps': spread_relative * 10000,  # 基点
            'mid_price': mid_price
        }

    def analyze_order_distribution(self) -> Dict[str, Any]:
        """分析订单分布特征"""
        analysis = {
            'bid_distribution': [],
            'ask_distribution': [],
            'price_levels': {
                'bid_price_range': 0,
                'ask_price_range': 0,
                'bid_volume_concentration': {},  # 各档位成交量占比
                'ask_volume_concentration': {}
            }
        }

        # 买盘分析
        if self.bids:
            bid_prices = [bid['price'] for bid in self.bids]
            bid_volumes = [bid['volume'] for bid in self.bids]
            total_bid_volume = sum(bid_volumes)

            analysis['bid_distribution'] = [
                {
                    'level': bid['level'],
                    'price': bid['price'],
                    'volume': bid['volume'],
                    'volume_pct': bid['volume'] / total_bid_volume * 100 if total_bid_volume > 0 else 0
                }
                for bid in self.bids
            ]

            analysis['price_levels']['bid_price_range'] = max(bid_prices) - min(bid_prices)

            # 计算各档位集中度
            for i, bid in enumerate(self.bids[:5], 1):
                pct = bid['volume'] / total_bid_volume * 100 if total_bid_volume > 0 else 0
                analysis['price_levels']['bid_volume_concentration'][f'bid_{i}'] = pct

        # 卖盘分析
        if self.asks:
            ask_prices = [ask['price'] for ask in self.asks]
            ask_volumes = [ask['volume'] for ask in self.asks]
            total_ask_volume = sum(ask_volumes)

            analysis['ask_distribution'] = [
                {
                    'level': ask['level'],
                    'price': ask['price'],
                    'volume': ask['volume'],
                    'volume_pct': ask['volume'] / total_ask_volume * 100 if total_ask_volume > 0 else 0
                }
                for ask in self.asks
            ]

            analysis['price_levels']['ask_price_range'] = max(ask_prices) - min(ask_prices)

            # 计算各档位集中度
            for i, ask in enumerate(self.asks[:5], 1):
                pct = ask['volume'] / total_ask_volume * 100 if total_ask_volume > 0 else 0
                analysis['price_levels']['ask_volume_concentration'][f'ask_{i}'] = pct

        return analysis

    def calculate_liquidity_metrics(self) -> Dict[str, float]:
        """计算流动性指标"""
        metrics = {}

        if not self.bids or not self.asks:
            return metrics

        # 获取价差信息
        spread_info = self.calculate_spread_metrics()
        depth_info = self.get_market_depth_metrics()

        # 流动性比率 (买卖盘总量 / 价差)
        if spread_info.get('spread_absolute', 0) > 0:
            metrics['liquidity_ratio'] = depth_info['total_volume'] / spread_info['spread_absolute']
        else:
            metrics['liquidity_ratio'] = float('inf')

        # 市场冲击成本估算 (简化版)
        metrics['market_impact_1pct'] = self._estimate_market_impact(0.01)  # 1%成交量的冲击
        metrics['market_impact_5pct'] = self._estimate_market_impact(0.05)  # 5%成交量的冲击

        # 有效价差 (考虑市场深度)
        metrics['effective_spread_bps'] = spread_info.get('spread_bps', 0)

        # 价格改善机会
        metrics['price_improvement_bid'] = self._calculate_price_improvement('bid')
        metrics['price_improvement_ask'] = self._calculate_price_improvement('ask')

        return metrics

    def _estimate_market_impact(self, volume_fraction: float) -> float:
        """估算市场冲击成本

        Args:
            volume_fraction: 相对于总挂单量的成交比例

        Returns:
            估算的价格冲击 (基点)
        """
        depth_info = self.get_market_depth_metrics()
        total_volume = depth_info['total_volume']

        if total_volume == 0:
            return float('inf')

        target_volume = total_volume * volume_fraction
        cumulative_volume = 0
        weighted_price = 0

        # 模拟买入冲击 (消耗卖盘)
        for ask in sorted(self.asks, key=lambda x: x['price']):
            if cumulative_volume >= target_volume:
                break
            take_volume = min(ask['volume'], target_volume - cumulative_volume)
            weighted_price += ask['price'] * take_volume
            cumulative_volume += take_volume

        if cumulative_volume > 0:
            avg_execution_price = weighted_price / cumulative_volume
            best_ask = min(self.asks, key=lambda x: x['price'])['price']
            impact_bps = (avg_execution_price - best_ask) / best_ask * 10000
            return impact_bps

        return 0

    def _calculate_price_improvement(self, side: str) -> float:
        """计算价格改善空间 (基点)"""
        if side == 'bid' and len(self.bids) > 1:
            best_bid = max(self.bids, key=lambda x: x['price'])['price']
            second_bid = sorted(self.bids, key=lambda x: x['price'], reverse=True)[1]['price']
            return (best_bid - second_bid) / best_bid * 10000

        elif side == 'ask' and len(self.asks) > 1:
            best_ask = min(self.asks, key=lambda x: x['price'])['price']
            second_ask = sorted(self.asks, key=lambda x: x['price'])[1]['price']
            return (second_ask - best_ask) / best_ask * 10000

        return 0

    def generate_analysis_report(self) -> str:
        """生成订单簿分析报告"""
        depth_metrics = self.get_market_depth_metrics()
        spread_metrics = self.calculate_spread_metrics()
        liquidity_metrics = self.calculate_liquidity_metrics()
        distribution_analysis = self.analyze_order_distribution()

        report = f"""
订单簿分析报告 - {self.symbol}
{'='*50}

时间: {depth_metrics.get('timestamp', 'N/A')}

价差分析:
• 最佳买价: {spread_metrics.get('best_bid', 0):.4f}
• 最佳卖价: {spread_metrics.get('best_ask', 0):.4f}
• 价差 (绝对): {spread_metrics.get('spread_absolute', 0):.4f}
• 价差 (相对): {spread_metrics.get('spread_relative', 0):.2f}%
• 价差 (基点): {spread_metrics.get('spread_bps', 0):.1f}
• 中间价: {spread_metrics.get('mid_price', 0):.4f}

市场深度:
• 买盘档位: {depth_metrics['bid_levels']}
• 卖盘档位: {depth_metrics['ask_levels']}
• 买盘总量: {depth_metrics['total_bid_volume']:,}
• 卖盘总量: {depth_metrics['total_ask_volume']:,}
• 总挂单量: {depth_metrics['total_volume']:,}
• 买卖失衡: {depth_metrics['imbalance_ratio']:.2%}

流动性指标:
• 流动性比率: {liquidity_metrics.get('liquidity_ratio', 0):.2f}
• 市场冲击(1%): {liquidity_metrics.get('market_impact_1pct', 0):.1f}bp
• 市场冲击(5%): {liquidity_metrics.get('market_impact_5pct', 0):.1f}bp
• 有效价差: {liquidity_metrics.get('effective_spread_bps', 0):.1f}bp

买卖盘集中度 (前5档):
"""

        # 添加买卖盘集中度信息
        bid_concentration = distribution_analysis['price_levels']['bid_volume_concentration']
        ask_concentration = distribution_analysis['price_levels']['ask_volume_concentration']

        report += "买盘集中度:\n"
        for level, pct in bid_concentration.items():
            report += f"  {level}: {pct:.1f}%\n"

        report += "卖盘集中度:\n"
        for level, pct in ask_concentration.items():
            report += f"  {level}: {pct:.1f}%\n"

        return report


class TickDataAnalyzer:
    """逐笔交易数据分析器"""

    def __init__(self, tick_data: pd.DataFrame):
        """
        初始化逐笔数据分析器

        Args:
            tick_data: 逐笔交易数据DataFrame
        """
        self.tick_data = tick_data.copy() if not tick_data.empty else pd.DataFrame()

    def calculate_vwap(self, time_window: str = None) -> float:
        """计算成交量加权平均价 (VWAP)

        Args:
            time_window: 时间窗口，如'30min', '1H'等，None表示全天

        Returns:
            VWAP价格
        """
        if self.tick_data.empty:
            return 0

        # 检查必要的列
        required_columns = ['price', 'volume']
        missing_columns = [col for col in required_columns if col not in self.tick_data.columns]
        if missing_columns:
            print(f"警告: 缺少列 {missing_columns}，无法计算VWAP")
            return 0

        data = self.tick_data.copy()

        # 过滤掉无效数据
        data = data[(data['price'] > 0) & (data['volume'] > 0)]
        if data.empty:
            return 0

        if time_window and 'time' in data.columns:
            # 按时间窗口分组计算
            data.set_index('time', inplace=True)
            grouped = data.resample(time_window)
            vwaps = []
            for group in grouped:
                group_data = group[1]
                if not group_data.empty:
                    total_volume = group_data['volume'].sum()
                    if total_volume > 0:
                        vwap = (group_data['price'] * group_data['volume']).sum() / total_volume
                        vwaps.append(vwap)
            return np.mean(vwaps) if vwaps else 0

        # 全天VWAP
        total_volume = data['volume'].sum()
        if total_volume > 0:
            return (data['price'] * data['volume']).sum() / total_volume
        return 0

    def analyze_order_flow(self) -> Dict[str, Any]:
        """分析订单流特征"""
        if self.tick_data.empty or 'side' not in self.tick_data.columns:
            return {}

        analysis = {
            'total_trades': len(self.tick_data),
            'buy_trades': 0,
            'sell_trades': 0,
            'neutral_trades': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'buy_amount': 0,
            'sell_amount': 0,
            'order_flow_imbalance': 0,  # 订单流失衡
            'average_trade_size': 0,
            'volume_weighted_price': 0
        }

        buy_data = self.tick_data[self.tick_data['side'] == 'buy']
        sell_data = self.tick_data[self.tick_data['side'] == 'sell']
        neutral_data = self.tick_data[self.tick_data['side'] == 'neutral']

        analysis['buy_trades'] = len(buy_data)
        analysis['sell_trades'] = len(sell_data)
        analysis['neutral_trades'] = len(neutral_data)

        if not buy_data.empty:
            analysis['buy_volume'] = buy_data['volume'].sum()
            if 'amount' in buy_data.columns:
                analysis['buy_amount'] = buy_data['amount'].sum()

        if not sell_data.empty:
            analysis['sell_volume'] = sell_data['volume'].sum()
            if 'amount' in sell_data.columns:
                analysis['sell_amount'] = sell_data['amount'].sum()

        total_volume = analysis['buy_volume'] + analysis['sell_volume']
        if total_volume > 0:
            analysis['order_flow_imbalance'] = (analysis['buy_volume'] - analysis['sell_volume']) / total_volume

        if analysis['total_trades'] > 0:
            analysis['average_trade_size'] = total_volume / analysis['total_trades']

        # 计算成交量加权价格
        analysis['volume_weighted_price'] = self.calculate_vwap()

        return analysis

    def detect_large_trades(self, volume_threshold_pct: float = 95) -> pd.DataFrame:
        """检测大额交易

        Args:
            volume_threshold_pct: 成交量百分位阈值，默认95%分位

        Returns:
            大额交易DataFrame
        """
        if self.tick_data.empty or 'volume' not in self.tick_data.columns:
            return pd.DataFrame()

        threshold = np.percentile(self.tick_data['volume'], volume_threshold_pct)
        large_trades = self.tick_data[self.tick_data['volume'] >= threshold].copy()

        # 添加相对大小指标
        if not large_trades.empty:
            avg_volume = self.tick_data['volume'].mean()
            large_trades['size_multiple'] = large_trades['volume'] / avg_volume

        return large_trades.sort_values('volume', ascending=False)

    def calculate_trade_intensity(self, time_window: str = '1min') -> pd.DataFrame:
        """计算交易强度 (单位时间内的交易次数和成交量)

        Args:
            time_window: 时间窗口

        Returns:
            交易强度DataFrame
        """
        if self.tick_data.empty or 'time' not in self.tick_data.columns:
            return pd.DataFrame()

        data = self.tick_data.copy()
        data.set_index('time', inplace=True)

        # 按时间窗口聚合
        intensity = data.resample(time_window).agg({
            'price': ['count', 'mean'],  # 交易次数和平均价格
            'volume': ['sum', 'mean'],   # 总成交量和平均单笔量
            'amount': 'sum' if 'amount' in data.columns else lambda x: 0
        }).reset_index()

        # 重命名列
        intensity.columns = ['time', 'trade_count', 'avg_price', 'total_volume', 'avg_volume', 'total_amount']

        # 计算交易强度指标
        intensity['trades_per_minute'] = intensity['trade_count']  # 每分钟交易次数
        intensity['volume_per_minute'] = intensity['total_volume']  # 每分钟成交量

        return intensity


def analyze_order_book(symbol: str, market: str = 'CN') -> Optional[Dict[str, Any]]:
    """便捷函数：分析单只股票的订单簿

    Args:
        symbol: 股票代码
        market: 市场类型

    Returns:
        完整的订单簿分析结果
    """
    from .manager import get_order_book

    order_book_data = get_order_book(symbol, market)

    if not order_book_data:
        return None

    analyzer = OrderBookAnalyzer(order_book_data)

    return {
        'raw_data': order_book_data,
        'depth_metrics': analyzer.get_market_depth_metrics(),
        'spread_metrics': analyzer.calculate_spread_metrics(),
        'liquidity_metrics': analyzer.calculate_liquidity_metrics(),
        'distribution_analysis': analyzer.analyze_order_distribution(),
        'report': analyzer.generate_analysis_report()
    }


def analyze_tick_data(symbol: str, market: str = 'CN', trade_date: str = None) -> Optional[Dict[str, Any]]:
    """便捷函数：分析单只股票的逐笔数据

    Args:
        symbol: 股票代码
        market: 市场类型
        trade_date: 交易日期

    Returns:
        完整的逐笔数据分析结果
    """
    from .manager import get_tick_data

    tick_data = get_tick_data(symbol, market, trade_date)

    if tick_data is None or tick_data.empty:
        return None

    analyzer = TickDataAnalyzer(tick_data)

    return {
        'raw_data': tick_data,
        'vwap': analyzer.calculate_vwap(),
        'order_flow_analysis': analyzer.analyze_order_flow(),
        'large_trades': analyzer.detect_large_trades(),
        'trade_intensity': analyzer.calculate_trade_intensity()
    }