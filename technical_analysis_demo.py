#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标分析模块演示
展示如何使用quantlib.technical进行完整的技术分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# 导入我们的技术指标模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from quantlib.technical import TechnicalAnalyzer
    from quantlib.technical.trend import TrendIndicators
    from quantlib.technical.oscillator import OscillatorIndicators
    from quantlib.technical.volume import VolumeIndicators
    print("✅ 成功导入quantlib.technical模块")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保quantlib.technical模块已正确安装")

def get_sample_data(symbol: str = "AAPL", period: str = "1y") -> pd.DataFrame:
    """获取示例数据"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        # 标准化列名
        data.columns = [col.lower() for col in data.columns]
        data = data.reset_index()
        
        print(f"✅ 成功获取{symbol}数据，共{len(data)}条记录")
        print(f"数据范围: {data['date'].iloc[0].strftime('%Y-%m-%d')} 至 {data['date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        return data
        
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        print("使用模拟数据代替...")
        return generate_mock_data()

def generate_mock_data(days: int = 252) -> pd.DataFrame:
    """生成模拟股价数据"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='D')
    
    # 生成随机价格数据
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    price = 100
    prices = []
    
    for ret in returns:
        price *= (1 + ret)
        prices.append(price)
    
    data = pd.DataFrame({
        'date': dates,
        'open': np.array(prices) * (1 + np.random.normal(0, 0.01, len(dates))),
        'high': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
        'low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # 确保high >= low, high >= open, high >= close, low <= open, low <= close
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    print(f"✅ 生成模拟数据，共{len(data)}条记录")
    return data

def demo_individual_indicators(data: pd.DataFrame):
    """演示单个指标的使用"""
    print("\n" + "="*80)
    print("📊 单个技术指标演示")
    print("="*80)
    
    # 1. 趋势指标演示
    print("\n🔴 趋势指标 (Trend Indicators)")
    print("-" * 40)
    
    trend = TrendIndicators(data)
    
    # 移动平均线
    ma = trend.moving_averages(periods=[5, 10, 20, 50])
    print("移动平均线 (MA):")
    for key, value in ma.results.items():
        if not pd.isna(value.iloc[-1]):
            print(f"  {key}: {value.iloc[-1]:.2f}")
    
    # MACD
    macd = trend.macd()
    print(f"\nMACD:")
    print(f"  MACD: {macd.results['MACD'].iloc[-1]:.4f}")
    print(f"  Signal: {macd.results['Signal'].iloc[-1]:.4f}")
    print(f"  Histogram: {macd.results['Histogram'].iloc[-1]:.4f}")
    
    # 布林带
    bb = trend.bollinger_bands()
    print(f"\n布林带:")
    print(f"  上轨: {bb.results['Upper_Band'].iloc[-1]:.2f}")
    print(f"  中轨: {bb.results['Middle_Band'].iloc[-1]:.2f}")
    print(f"  下轨: {bb.results['Lower_Band'].iloc[-1]:.2f}")
    print(f"  位置: {bb.results['BB_Position'].iloc[-1]:.1f}%")
    
    # 2. 震荡指标演示
    print("\n🟡 震荡指标 (Oscillator Indicators)")
    print("-" * 40)
    
    osc = OscillatorIndicators(data)
    
    # RSI
    rsi = osc.rsi()
    print(f"RSI: {rsi.results['RSI'].iloc[-1]:.1f}")
    
    # KDJ
    kdj = osc.kdj()
    print(f"KDJ:")
    print(f"  K: {kdj.results['K'].iloc[-1]:.1f}")
    print(f"  D: {kdj.results['D'].iloc[-1]:.1f}")
    print(f"  J: {kdj.results['J'].iloc[-1]:.1f}")
    
    # 威廉指标
    williams = osc.williams()
    print(f"Williams %R: {williams.results['Williams_R'].iloc[-1]:.1f}")
    
    # 3. 成交量指标演示
    if 'volume' in data.columns:
        print("\n🟢 成交量指标 (Volume Indicators)")
        print("-" * 40)
        
        vol = VolumeIndicators(data)
        
        # OBV
        obv = vol.obv()
        print(f"OBV: {obv.results['OBV'].iloc[-1]:,.0f}")
        
        # VWAP
        vwap = vol.vwap(period=20)
        print(f"VWAP: {vwap.results['VWAP'].iloc[-1]:.2f}")
        
        # 蔡金资金流量
        cmf = vol.chaikin_money_flow()
        print(f"CMF: {cmf.results['CMF'].iloc[-1]:.3f}")

def demo_comprehensive_analysis(data: pd.DataFrame):
    """演示综合技术分析"""
    print("\n" + "="*80)
    print("🎯 综合技术分析演示")
    print("="*80)
    
    # 创建综合分析器
    analyzer = TechnicalAnalyzer(data)
    
    # 计算所有指标
    print("\n计算所有技术指标...")
    analyzer.calculate_all_indicators()
    
    # 生成交易信号
    print("生成交易信号...")
    analyzer.generate_all_signals()
    
    # 获取综合信号
    signal, strength, analysis = analyzer.get_consensus_signal()
    
    # 信号解读
    signal_meaning = {
        2: "🚀 强烈看涨",
        1: "📈 看涨", 
        0: "➡️  中性",
        -1: "📉 看跌",
        -2: "💥 强烈看跌"
    }
    
    print(f"\n📊 综合分析结果:")
    print(f"综合信号: {signal_meaning.get(signal, '未知')} (数值: {signal})")
    print(f"信号强度: {strength:.2f}")
    print(f"信号一致性: {analysis['signal_consistency']:.2f}")
    
    print(f"\n📈 信号分解:")
    print(f"趋势信号: {analysis['trend_signal']:.2f}")
    print(f"震荡信号: {analysis['oscillator_signal']:.2f}")
    print(f"成交量信号: {analysis['volume_signal']:.2f}")
    
    print(f"\n📊 信号统计:")
    print(f"看涨指标: {analysis['bullish_count']} 个")
    print(f"看跌指标: {analysis['bearish_count']} 个")
    print(f"中性指标: {analysis['neutral_count']} 个")
    print(f"总指标数: {analysis['total_indicators']} 个")
    
    # 支撑阻力位
    print(f"\n🎯 支撑阻力位分析:")
    levels = analyzer.identify_support_resistance()
    
    current_price = data['close'].iloc[-1]
    print(f"当前价格: {current_price:.2f}")
    
    if levels['support_levels']:
        nearest_support = max([level for level in levels['support_levels'] if level < current_price], default=None)
        if nearest_support:
            print(f"最近支撑位: {nearest_support:.2f} (距离: {((current_price - nearest_support) / current_price * 100):.1f}%)")
    
    if levels['resistance_levels']:
        nearest_resistance = min([level for level in levels['resistance_levels'] if level > current_price], default=None)
        if nearest_resistance:
            print(f"最近阻力位: {nearest_resistance:.2f} (距离: {((nearest_resistance - current_price) / current_price * 100):.1f}%)")

def demo_signal_analysis(data: pd.DataFrame):
    """演示信号分析和交易建议"""
    print("\n" + "="*80)
    print("🎯 交易信号分析和建议")
    print("="*80)
    
    analyzer = TechnicalAnalyzer(data)
    analyzer.calculate_all_indicators()
    analyzer.generate_all_signals()
    
    # 生成详细分析报告
    report = analyzer.generate_analysis_report()
    print(report)

def demo_visualization(data: pd.DataFrame):
    """演示技术分析可视化"""
    print("\n" + "="*80)
    print("📊 技术分析可视化演示")
    print("="*80)
    
    try:
        analyzer = TechnicalAnalyzer(data)
        analyzer.calculate_all_indicators()
        
        print("正在生成技术分析图表...")
        analyzer.plot_analysis()
        print("✅ 图表已显示")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        print("可能是matplotlib配置问题，请检查显示设置")

def demo_custom_analysis(data: pd.DataFrame):
    """演示自定义分析策略"""
    print("\n" + "="*80)
    print("🛠️  自定义分析策略演示")
    print("="*80)
    
    analyzer = TechnicalAnalyzer(data)
    analyzer.calculate_all_indicators()
    
    # 获取关键指标
    rsi = analyzer.indicators['rsi'].results['RSI'].iloc[-1]
    macd = analyzer.indicators['macd'].results['MACD'].iloc[-1]
    signal_line = analyzer.indicators['macd'].results['Signal'].iloc[-1]
    bb_position = analyzer.indicators['bb'].results['BB_Position'].iloc[-1]
    
    if analyzer.volume:
        cmf = analyzer.indicators['cmf'].results['CMF'].iloc[-1]
    else:
        cmf = 0
    
    print(f"当前关键指标:")
    print(f"RSI: {rsi:.1f}")
    print(f"MACD: {macd:.4f}")
    print(f"布林带位置: {bb_position:.1f}%")
    if analyzer.volume:
        print(f"资金流量CMF: {cmf:.3f}")
    
    # 自定义交易策略
    print(f"\n🎯 自定义交易策略判断:")
    
    buy_signals = 0
    sell_signals = 0
    
    # 超卖反弹策略
    if rsi < 30 and bb_position < 20:
        print("✅ 超卖反弹信号: RSI超卖 + 布林带下轨支撑")
        buy_signals += 1
    
    # MACD金叉确认
    if macd > signal_line and macd > 0:
        print("✅ MACD多头信号: MACD在零轴上方金叉")
        buy_signals += 1
    
    # 资金流入确认
    if analyzer.volume and cmf > 0.1:
        print("✅ 资金流入信号: CMF显示资金净流入")
        buy_signals += 1
    
    # 超买风险
    if rsi > 70 and bb_position > 80:
        print("⚠️ 超买风险: RSI超买 + 布林带上轨阻力")
        sell_signals += 1
    
    # MACD死叉
    if macd < signal_line and macd < 0:
        print("⚠️ MACD空头信号: MACD在零轴下方死叉")
        sell_signals += 1
    
    # 综合判断
    print(f"\n📊 策略综合评分:")
    print(f"看涨信号数: {buy_signals}")
    print(f"看跌信号数: {sell_signals}")
    
    if buy_signals > sell_signals:
        confidence = (buy_signals - sell_signals) / (buy_signals + sell_signals + 1) * 100
        print(f"🚀 建议: 看涨 (信心度: {confidence:.0f}%)")
    elif sell_signals > buy_signals:
        confidence = (sell_signals - buy_signals) / (buy_signals + sell_signals + 1) * 100
        print(f"💥 建议: 看跌 (信心度: {confidence:.0f}%)")
    else:
        print("➡️ 建议: 保持观望")

def main():
    """主演示函数"""
    print("🎯 quantlib技术指标分析模块完整演示")
    print("="*80)
    print("本演示将展示如何使用技术指标进行股票技术分析")
    print("包括趋势指标、震荡指标、成交量指标以及综合分析")
    
    # 获取数据
    print(f"\n📊 获取分析数据...")
    data = get_sample_data("AAPL", "6mo")  # 获取苹果股票6个月数据
    
    if data is None or data.empty:
        print("❌ 无法获取数据，演示终止")
        return
    
    try:
        # 1. 单个指标演示
        demo_individual_indicators(data)
        
        # 2. 综合分析演示  
        demo_comprehensive_analysis(data)
        
        # 3. 信号分析演示
        demo_signal_analysis(data)
        
        # 4. 自定义策略演示
        demo_custom_analysis(data)
        
        # 5. 可视化演示（可选）
        show_charts = input(f"\n是否显示技术分析图表? (y/n): ").lower().strip()
        if show_charts == 'y':
            demo_visualization(data)
        
        print(f"\n🎉 演示完成！")
        print("="*80)
        print("你现在已经了解了如何使用quantlib.technical模块进行技术分析")
        print("可以开始在你的量化投资项目中使用这些工具了！")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()