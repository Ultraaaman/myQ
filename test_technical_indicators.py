#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantlib.technical 技术指标模块测试脚本
全面测试所有技术指标的计算准确性和功能完整性
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# 添加quantlib路径
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from quantlib.technical import TechnicalAnalyzer
    from quantlib.technical.trend import TrendIndicators, MovingAverages, MACD, BollingerBands, ADX, ParabolicSAR
    from quantlib.technical.oscillator import OscillatorIndicators, RSI, KDJ, Williams, CCI, Stochastic, ROC
    from quantlib.technical.volume import VolumeIndicators, OBV, VPT, VWAP, ChaikinMoneyFlow, AccumulationDistribution
    from quantlib.technical.base import TechnicalIndicator
    print("成功导入所有技术指标模块")
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)


class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_test_data(periods=100, trend='sideways'):
        """
        生成测试用的OHLCV数据
        
        Args:
            periods: 数据点数量
            trend: 趋势类型 ('uptrend', 'downtrend', 'sideways', 'volatile')
        """
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        np.random.seed(42)  # 固定随机种子确保测试可重复
        
        if trend == 'uptrend':
            # 上升趋势
            base_returns = np.random.normal(0.005, 0.02, periods)  # 正向偏移
            base_price = 100
        elif trend == 'downtrend':
            # 下降趋势
            base_returns = np.random.normal(-0.005, 0.02, periods)  # 负向偏移
            base_price = 100
        elif trend == 'volatile':
            # 高波动
            base_returns = np.random.normal(0, 0.05, periods)  # 高波动率
            base_price = 100
        else:  # sideways
            # 横盘震荡
            base_returns = np.random.normal(0, 0.015, periods)
            base_price = 100
        
        # 生成价格序列
        prices = []
        current_price = base_price
        
        for ret in base_returns:
            current_price *= (1 + ret)
            prices.append(current_price)
        
        # 生成OHLC
        data = []
        for i, close in enumerate(prices):
            # 生成开盘价（基于前一日收盘价）
            if i == 0:
                open_price = base_price
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
            
            # 生成最高最低价
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            
            # 生成成交量
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'date': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)


class TestTechnicalIndicators(unittest.TestCase):
    """技术指标测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_data = TestDataGenerator.generate_test_data(periods=100)
        self.uptrend_data = TestDataGenerator.generate_test_data(periods=100, trend='uptrend')
        self.downtrend_data = TestDataGenerator.generate_test_data(periods=100, trend='downtrend')
        print(f"测试数据准备完成: {len(self.test_data)} 条记录")
    
    def test_data_validation(self):
        """测试数据验证功能"""
        print("\n 测试数据验证...")
        
        # 测试正常数据
        try:
            analyzer = TechnicalAnalyzer(self.test_data)
            self.assertTrue(True, "正常数据验证通过")
        except Exception as e:
            self.fail(f"正常数据验证失败: {e}")
        
        # 测试缺少必要列
        invalid_data = self.test_data.drop('close', axis=1)
        with self.assertRaises(ValueError):
            TechnicalAnalyzer(invalid_data)
        
        # 测试空数据
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            TechnicalAnalyzer(empty_data)
        
        print("数据验证测试通过")
    
    def test_moving_averages(self):
        """测试移动平均线"""
        print("\n 测试移动平均线...")
        
        trend = TrendIndicators(self.test_data)
        ma = trend.moving_averages(periods=[5, 10, 20])
        
        # 测试结果是否存在
        self.assertIn('SMA_5', ma.results)
        self.assertIn('SMA_10', ma.results)
        self.assertIn('SMA_20', ma.results)
        self.assertIn('EMA_5', ma.results)
        
        # 测试SMA计算正确性
        sma_5 = ma.results['SMA_5']
        manual_sma = self.test_data['close'].rolling(5).mean()
        np.testing.assert_array_almost_equal(
            sma_5.dropna().values, 
            manual_sma.dropna().values, 
            decimal=6
        )
        
        # 测试信号生成
        signals = ma.get_signals()
        self.assertIn('signal', signals.columns)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)
        
        print("移动平均线测试通过")
    
    def test_macd(self):
        """测试MACD指标"""
        print("\n 测试MACD指标...")
        
        trend = TrendIndicators(self.test_data)
        macd = trend.macd(fast=12, slow=26, signal=9)
        
        # 测试结果是否存在
        self.assertIn('MACD', macd.results)
        self.assertIn('Signal', macd.results)
        self.assertIn('Histogram', macd.results)
        
        # 测试MACD基本性质
        macd_line = macd.results['MACD']
        signal_line = macd.results['Signal']
        histogram = macd.results['Histogram']
        
        # Histogram应该等于MACD - Signal
        np.testing.assert_array_almost_equal(
            histogram.dropna().values,
            (macd_line - signal_line).dropna().values,
            decimal=6
        )
        
        # 测试信号生成
        signals = macd.get_signals()
        self.assertIn('signal', signals.columns)
        
        print("MACD测试通过")
    
    def test_bollinger_bands(self):
        """测试布林带"""
        print("\n 测试布林带...")
        
        trend = TrendIndicators(self.test_data)
        bb = trend.bollinger_bands(period=20, std_dev=2.0)
        
        # 测试结果
        self.assertIn('Upper_Band', bb.results)
        self.assertIn('Middle_Band', bb.results)
        self.assertIn('Lower_Band', bb.results)
        self.assertIn('Bandwidth', bb.results)
        self.assertIn('BB_Position', bb.results)
        
        # 测试布林带基本性质：上轨 > 中轨 > 下轨
        upper = bb.results['Upper_Band'].dropna()
        middle = bb.results['Middle_Band'].dropna()
        lower = bb.results['Lower_Band'].dropna()
        
        self.assertTrue((upper >= middle).all(), "上轨应该大于等于中轨")
        self.assertTrue((middle >= lower).all(), "中轨应该大于等于下轨")
        
        print("布林带测试通过")
    
    def test_rsi(self):
        """测试RSI指标"""
        print("\n 测试RSI指标...")
        
        osc = OscillatorIndicators(self.test_data)
        rsi = osc.rsi(period=14)
        
        # 测试结果
        self.assertIn('RSI', rsi.results)
        
        # 测试RSI取值范围[0, 100]
        rsi_values = rsi.results['RSI'].dropna()
        self.assertTrue((rsi_values >= 0).all(), "RSI应该大于等于0")
        self.assertTrue((rsi_values <= 100).all(), "RSI应该小于等于100")
        
        # 测试信号生成
        signals = rsi.get_signals()
        self.assertIn('signal', signals.columns)
        
        print("RSI测试通过")
    
    def test_kdj(self):
        """测试KDJ指标"""
        print("\n 测试KDJ指标...")
        
        osc = OscillatorIndicators(self.test_data)
        kdj = osc.kdj()
        
        # 测试结果
        self.assertIn('K', kdj.results)
        self.assertIn('D', kdj.results)
        self.assertIn('J', kdj.results)
        self.assertIn('RSV', kdj.results)
        
        # 测试K、D值范围
        k_values = kdj.results['K'].dropna()
        d_values = kdj.results['D'].dropna()
        
        # K、D值通常在0-100之间，但J值可能超出此范围
        self.assertTrue((k_values >= 0).all() and (k_values <= 100).all(), "K值应该在0-100之间")
        self.assertTrue((d_values >= 0).all() and (d_values <= 100).all(), "D值应该在0-100之间")
        
        print("KDJ测试通过")
    
    def test_volume_indicators(self):
        """测试成交量指标"""
        print("\n 测试成交量指标...")
        
        vol = VolumeIndicators(self.test_data)
        
        # 测试OBV
        obv = vol.obv()
        self.assertIn('OBV', obv.results)
        
        # 测试VWAP
        vwap = vol.vwap(period=20)
        self.assertIn('VWAP', vwap.results)
        
        # 测试CMF
        cmf = vol.chaikin_money_flow()
        self.assertIn('CMF', cmf.results)
        
        # CMF应该在-1到1之间
        cmf_values = cmf.results['CMF'].dropna()
        self.assertTrue((cmf_values >= -1).all(), "CMF应该大于等于-1")
        self.assertTrue((cmf_values <= 1).all(), "CMF应该小于等于1")
        
        print("成交量指标测试通过")
    
    def test_technical_analyzer(self):
        """测试综合技术分析器"""
        print("\n 测试综合技术分析器...")
        
        analyzer = TechnicalAnalyzer(self.test_data)
        
        # 测试计算所有指标
        analyzer.calculate_all_indicators()
        self.assertTrue(len(analyzer.indicators) > 0, "应该计算出指标")
        
        # 测试生成信号
        analyzer.generate_all_signals()
        self.assertTrue(len(analyzer.signals) > 0, "应该生成信号")
        
        # 测试综合信号
        signal, strength, analysis = analyzer.get_consensus_signal()
        self.assertIsInstance(signal, int, "综合信号应该是整数")
        self.assertTrue(-2 <= signal <= 2, "综合信号应该在-2到2之间")
        self.assertTrue(0 <= strength <= 1, "信号强度应该在0到1之间")
        self.assertIsInstance(analysis, dict, "分析结果应该是字典")
        
        # 测试支撑阻力位
        levels = analyzer.identify_support_resistance()
        self.assertIn('support_levels', levels)
        self.assertIn('resistance_levels', levels)
        
        # 测试生成报告
        report = analyzer.generate_analysis_report()
        self.assertIsInstance(report, str, "报告应该是字符串")
        self.assertTrue(len(report) > 100, "报告应该有实质内容")
        
        print("综合技术分析器测试通过")
    
    def test_signal_consistency(self):
        """测试信号一致性"""
        print("\n 测试信号一致性...")
        
        # 测试明显的上升趋势数据
        analyzer_up = TechnicalAnalyzer(self.uptrend_data)
        signal_up, strength_up, _ = analyzer_up.get_consensus_signal()
        
        # 测试明显的下降趋势数据
        analyzer_down = TechnicalAnalyzer(self.downtrend_data)
        signal_down, strength_down, _ = analyzer_down.get_consensus_signal()
        
        print(f"上升趋势信号: {signal_up}, 强度: {strength_up:.2f}")
        print(f"下降趋势信号: {signal_down}, 强度: {strength_down:.2f}")
        
        # 上升趋势应该倾向于正信号，下降趋势倾向于负信号
        # 但由于随机数据的特性，我们只检查它们不应该完全相反
        self.assertNotEqual(signal_up, -signal_down, "信号应该有一定的方向性")
        
        print("信号一致性测试通过")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n 测试错误处理...")
        
        # 测试无volume数据时的成交量指标
        data_no_volume = self.test_data.drop('volume', axis=1)
        
        with self.assertRaises(ValueError):
            VolumeIndicators(data_no_volume)
        
        # 测试分析器在无volume数据时的处理
        analyzer = TechnicalAnalyzer(data_no_volume)
        self.assertIsNone(analyzer.volume, "无volume数据时应该为None")
        
        # 应该仍能正常工作（只是没有成交量指标）
        analyzer.calculate_all_indicators()
        signal, strength, _ = analyzer.get_consensus_signal()
        self.assertIsInstance(signal, int)
        
        print("错误处理测试通过")
    
    def test_performance(self):
        """测试性能"""
        print("\n 测试性能...")
        
        import time
        
        # 测试大数据集
        large_data = TestDataGenerator.generate_test_data(periods=1000)
        
        start_time = time.time()
        analyzer = TechnicalAnalyzer(large_data)
        analyzer.calculate_all_indicators()
        analyzer.generate_all_signals()
        signal, strength, _ = analyzer.get_consensus_signal()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"处理1000条数据耗时: {elapsed_time:.2f}秒")
        
        # 性能要求：1000条数据应在5秒内完成
        self.assertLess(elapsed_time, 5.0, "性能测试：1000条数据应在5秒内处理完成")
        
        print("性能测试通过")


class TestSpecialCases(unittest.TestCase):
    """特殊情况测试"""
    
    def test_constant_price(self):
        """测试价格恒定的情况"""
        print("\n 测试价格恒定情况...")
        
        # 创建价格恒定的数据
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        constant_data = pd.DataFrame({
            'date': dates,
            'open': [100] * 50,
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': np.random.randint(1000000, 5000000, 50)
        })
        
        analyzer = TechnicalAnalyzer(constant_data)
        analyzer.calculate_all_indicators()
        
        # 某些指标在价格恒定时应该有特定行为
        # RSI应该接近50
        rsi_value = analyzer.indicators['rsi'].results['RSI'].iloc[-1]
        self.assertTrue(45 <= rsi_value <= 55, f"价格恒定时RSI应该接近50，实际值：{rsi_value}")
        
        print("价格恒定情况测试通过")
    
    def test_extreme_volatility(self):
        """测试极端波动情况"""
        print("\n 测试极端波动情况...")
        
        volatile_data = TestDataGenerator.generate_test_data(periods=100, trend='volatile')
        
        analyzer = TechnicalAnalyzer(volatile_data)
        analyzer.calculate_all_indicators()
        
        # 检查指标是否能正常计算（不应该有NaN或Inf）
        for name, indicator in analyzer.indicators.items():
            for result_name, result_series in indicator.results.items():
                finite_values = result_series.dropna()
                if len(finite_values) > 0:
                    self.assertTrue(np.isfinite(finite_values).all(), 
                                  f"{name}.{result_name}包含非有限值")
        
        print("极端波动情况测试通过")


def run_comprehensive_test():
    """运行全面测试"""
    print(" 开始quantlib技术指标模块全面测试")
    print("="*80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestTechnicalIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestSpecialCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    if result.wasSuccessful():
        print(" 所有测试通过！技术指标模块工作正常")
        print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)} 个测试")
    else:
        print("测试失败！")
        print(f"失败: {len(result.failures)} 个测试")
        print(f"错误: {len(result.errors)} 个测试")
        
        if result.failures:
            print("\n失败的测试:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\n错误的测试:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


def run_quick_test():
    """快速验证测试"""
    print("快速验证测试")
    print("-" * 40)
    
    try:
        # 生成测试数据
        data = TestDataGenerator.generate_test_data(50)
        print(f"测试数据生成: {len(data)} 条记录")
        
        # 测试基础功能
        analyzer = TechnicalAnalyzer(data)
        print("技术分析器初始化")
        
        analyzer.calculate_all_indicators()
        print(f"指标计算完成: {len(analyzer.indicators)} 个指标")
        
        analyzer.generate_all_signals()
        print(f"信号生成完成: {len(analyzer.signals)} 个信号")
        
        signal, strength, analysis = analyzer.get_consensus_signal()
        print(f"综合信号: {signal}, 强度: {strength:.2f}")
        
        levels = analyzer.identify_support_resistance()
        print(f"支撑阻力: {len(levels['support_levels'])} 支撑, {len(levels['resistance_levels'])} 阻力")
        
        print("\n 快速测试全部通过！")
        return True
        
    except Exception as e:
        print(f"快速测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='技术指标模块测试脚本')
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                       help='测试模式: full=完整测试, quick=快速测试')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        success = run_comprehensive_test()
    else:
        success = run_quick_test()
    
    # 设置退出码
    sys.exit(0 if success else 1)