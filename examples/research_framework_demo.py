#!/usr/bin/env python3
"""
研究框架完整演示

展示如何使用研究模块进行因子研究，包括：
1. 因子库管理和自定义因子添加
2. 因子有效性分析
3. 因子回测
4. 综合研究和报告生成
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入研究模块
from quantlib.research import (
    create_factor_library,
    create_factor_analyzer,
    create_research_framework,
    FactorCategory,
    BacktestConfig,
    create_research_report
)


def generate_sample_data(symbols: list, periods: int = 500) -> tuple:
    """生成示例数据"""
    print("📊 生成示例数据...")
    
    dates = pd.date_range(start='2022-01-01', periods=periods, freq='D')
    
    # 生成价格数据
    price_data = {}
    stock_data = {}
    returns_data = {}
    
    for symbol in symbols:
        # 模拟股价走势
        np.random.seed(hash(symbol) % 2**32)
        
        # 生成价格序列
        base_price = 10.0 + np.random.normal(0, 2)
        price_changes = np.random.normal(0.001, 0.02, periods)  # 年化约25%收益，50%波动
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        # 生成OHLCV数据
        stock_data[symbol] = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': 1000000 + np.random.randint(-300000, 300000, periods),
            # 模拟基本面数据
            'pe_ratio': 15 + np.random.normal(0, 5, periods).clip(5, 50),
            'pb_ratio': 2 + np.random.normal(0, 1, periods).clip(0.5, 10),
            'roe': 0.15 + np.random.normal(0, 0.05, periods).clip(0, 0.5)
        }, index=dates)
        
        price_data[symbol] = prices
        
        # 计算收益率
        returns_data[symbol] = pd.Series(price_changes, index=dates)
    
    # 转换为DataFrame格式
    price_df = pd.DataFrame(price_data, index=dates)
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # 合并所有股票数据（简化版，实际中每个股票应该分开）
    combined_data = pd.concat(stock_data, axis=1)
    
    print(f"✅ 生成完成: {len(symbols)}只股票, {periods}个交易日")
    
    return combined_data, price_df, returns_df


def demo_factor_library():
    """演示因子库功能"""
    print("\n🔬 演示1: 因子库管理")
    print("=" * 50)
    
    # 创建因子库
    factor_lib = create_factor_library("data/demo_factor_library")
    
    # 查看默认因子
    print("📋 默认因子列表:")
    factor_list = factor_lib.list_factors()
    for name, info in factor_list.items():
        print(f"  {name}: {info['description']} ({info['category']})")
    
    # 添加自定义因子
    print("\n➕ 添加自定义因子...")
    
    # 自定义因子1: 价格动量偏度
    def price_momentum_skew(data, period=20, **kwargs):
        """价格动量偏度因子"""
        returns = data['close'].pct_change()
        momentum = returns.rolling(period).mean()
        return momentum.rolling(period).skew()
    
    factor_lib.create_custom_factor(
        name="price_momentum_skew",
        calc_func=price_momentum_skew,
        description="价格动量偏度因子",
        category=FactorCategory.MOMENTUM
    )
    
    # 自定义因子2: 成交量价格相关性
    def volume_price_corr(data, period=30, **kwargs):
        """成交量价格相关性因子"""
        price_changes = data['close'].pct_change()
        volume_changes = data['volume'].pct_change()
        correlation = price_changes.rolling(period).corr(volume_changes)
        return correlation
    
    factor_lib.create_custom_factor(
        name="volume_price_corr",
        calc_func=volume_price_corr,
        description="成交量价格相关性因子",
        category=FactorCategory.ALTERNATIVE
    )
    
    # 自定义因子3: 波动率分解因子
    def volatility_decomp(data, short_period=5, long_period=20, **kwargs):
        """波动率分解因子"""
        returns = data['close'].pct_change()
        short_vol = returns.rolling(short_period).std()
        long_vol = returns.rolling(long_period).std()
        return short_vol / long_vol - 1
    
    factor_lib.create_custom_factor(
        name="volatility_decomp",
        calc_func=volatility_decomp,
        description="短期长期波动率比值因子",
        category=FactorCategory.VOLATILITY
    )
    
    print("✅ 自定义因子添加完成")
    
    # 保存因子库
    factor_lib.save_factor_library()
    print("💾 因子库已保存")
    
    # 显示因子库摘要
    summary = factor_lib.get_summary()
    print(f"\n📊 因子库摘要:")
    print(f"  总因子数: {summary['total_factors']}")
    print(f"  分类统计: {summary['categories']}")
    
    return factor_lib


def demo_factor_analysis(factor_lib, data, returns):
    """演示因子分析功能"""
    print("\n📈 演示2: 因子有效性分析")
    print("=" * 50)
    
    # 创建因子分析器
    analyzer = create_factor_analyzer(min_periods=30)
    
    # 选择要分析的因子
    factor_names = [
        'momentum_20d', 'rsi_14d', 'volatility_20d', 'pe_ratio', 'pb_ratio',
        'price_momentum_skew', 'volume_price_corr', 'volatility_decomp'
    ]
    
    print(f"🔍 分析因子: {factor_names}")
    
    # 计算因子值
    print("⚙️ 计算因子值...")
    factor_data = factor_lib.calculate_factors(factor_names, data)
    print(f"✅ 因子值计算完成，形状: {factor_data.shape}")
    
    # 使用第一只股票的收益率进行分析（简化）
    stock_returns = returns.iloc[:, 0].dropna()
    
    # 批量分析因子
    print("📊 进行因子有效性分析...")
    analysis_results = {}
    
    for factor_name in factor_names:
        if factor_name in factor_data.columns:
            factor_series = factor_data[factor_name].dropna()
            
            # 对齐数据
            common_index = factor_series.index.intersection(stock_returns.index)
            if len(common_index) > 50:  # 确保有足够数据
                try:
                    result = analyzer.comprehensive_factor_analysis(
                        factor_series.loc[common_index],
                        stock_returns.loc[common_index],
                        factor_name
                    )
                    analysis_results[factor_name] = result
                    
                    print(f"  ✅ {factor_name}:")
                    print(f"     IC均值: {result.ic_analysis.ic_mean:.4f}")
                    print(f"     IC信息比率: {result.ic_analysis.ic_ir:.4f}")
                    print(f"     多空收益: {result.long_short_return:.2%}")
                    print(f"     换手率: {result.turnover:.2%}")
                    
                except Exception as e:
                    print(f"  ❌ {factor_name}: 分析失败 - {str(e)[:50]}...")
    
    # 生成因子排名
    if analysis_results:
        ranking = analyzer.create_factor_ranking(analysis_results, 'ic_ir')
        print(f"\n🏆 因子排名 (按IC信息比率):")
        print(ranking[['factor_name', 'ic_mean', 'ic_ir', 'long_short_return']].head())
        
        # 相关性分析
        factor_correlation = analyzer.factor_correlation_analysis(
            {name: factor_data[name] for name in analysis_results.keys() if name in factor_data.columns}
        )
        print(f"\n🔗 因子相关性矩阵形状: {factor_correlation.shape}")
        print("前5x5相关性:")
        print(factor_correlation.iloc[:5, :5].round(3))
    
    return analysis_results, factor_data


def demo_factor_backtest(factor_data, price_data, analysis_results):
    """演示因子回测功能"""
    print("\n💹 演示3: 因子策略回测")
    print("=" * 50)
    
    # 配置回测参数
    backtest_config = BacktestConfig(
        start_date=price_data.index[100],  # 留出warm-up期
        end_date=price_data.index[-50],    # 留出验证期
        initial_capital=1000000,
        commission=0.001,
        long_pct=0.3,    # 做多30%
        short_pct=0.3,   # 做空30%
        rebalance_freq='M',  # 月度调仓
        min_stocks=2
    )
    
    print(f"📅 回测期间: {backtest_config.start_date.strftime('%Y-%m-%d')} 到 {backtest_config.end_date.strftime('%Y-%m-%d')}")
    print(f"💰 初始资金: ${backtest_config.initial_capital:,.0f}")
    
    # 创建研究框架进行回测
    research_framework = create_research_framework()
    
    # 选择表现较好的因子进行回测
    factor_names_to_backtest = list(analysis_results.keys())[:3]  # 前3个因子
    print(f"🎯 回测因子: {factor_names_to_backtest}")
    
    # 准备回测数据
    backtest_factor_data = factor_data[factor_names_to_backtest].loc[
        backtest_config.start_date:backtest_config.end_date
    ]
    backtest_price_data = price_data.loc[
        backtest_config.start_date:backtest_config.end_date
    ]
    
    print(f"📊 回测数据形状: 因子 {backtest_factor_data.shape}, 价格 {backtest_price_data.shape}")
    
    # 运行回测
    try:
        backtest_results = research_framework.conduct_factor_backtest(
            backtest_factor_data,
            backtest_price_data,
            backtest_config,
            factor_names_to_backtest,
            save_results=True
        )
        
        print(f"\n📈 回测结果摘要:")
        for factor_name, result in backtest_results.items():
            print(f"  {factor_name}:")
            print(f"    总收益率: {result.total_return:.2%}")
            print(f"    年化收益率: {result.annual_return:.2%}")
            print(f"    夏普比率: {result.sharpe_ratio:.2f}")
            print(f"    最大回撤: {result.max_drawdown:.2%}")
            print(f"    胜率: {result.win_rate:.2%}")
        
        return backtest_results
        
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        return {}


def demo_comprehensive_research():
    """演示综合研究功能"""
    print("\n🎓 演示4: 综合研究流程")
    print("=" * 50)
    
    # 1. 准备数据
    symbols = ['STOCK001', 'STOCK002', 'STOCK003', 'STOCK004', 'STOCK005']
    data, price_data, returns_data = generate_sample_data(symbols, periods=300)
    
    # 2. 创建研究框架
    research_framework = create_research_framework(storage_path="data/demo_research")
    
    # 3. 添加几个自定义因子
    print("➕ 添加自定义因子...")
    
    def macd_signal(data, fast=12, slow=26, signal=9, **kwargs):
        """MACD信号线因子"""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line
    
    research_framework.add_custom_factor(
        "macd_signal",
        macd_signal,
        "MACD信号线因子",
        FactorCategory.TECHNICAL
    )
    
    def earnings_quality(data, **kwargs):
        """盈利质量因子"""
        if 'roe' in data.columns:
            roe_ma = data['roe'].rolling(4).mean()  # 滚动4期ROE均值
            roe_stability = 1 / (data['roe'].rolling(4).std() + 0.01)  # ROE稳定性
            return roe_ma * roe_stability
        else:
            return pd.Series(index=data.index, data=0)
    
    research_framework.add_custom_factor(
        "earnings_quality",
        earnings_quality,
        "盈利质量因子",
        FactorCategory.QUALITY
    )
    
    # 4. 设置回测配置
    backtest_config = BacktestConfig(
        start_date=data.index[100],
        end_date=data.index[-30],
        initial_capital=1000000,
        commission=0.001,
        long_pct=0.2,
        short_pct=0.2,
        rebalance_freq='M'
    )
    
    # 5. 运行综合研究
    print("🔬 开始综合研究...")
    try:
        # 使用第一只股票的数据进行演示
        stock_data = data.iloc[:, data.columns.get_level_values(0) == symbols[0]]
        stock_data.columns = stock_data.columns.droplevel(0)
        stock_returns = returns_data[symbols[0]]
        
        comprehensive_results = research_framework.comprehensive_factor_study(
            stock_data,
            price_data[[symbols[0]]],  # 只使用第一只股票
            stock_returns,
            backtest_config,
            factor_names=['momentum_20d', 'rsi_14d', 'macd_signal', 'earnings_quality']
        )
        
        print("✅ 综合研究完成!")
        print(f"\n📊 研究摘要:")
        summary = comprehensive_results['summary']
        print(f"  分析因子数: {summary['total_factors']}")
        print(f"  平均IC均值: {summary['avg_ic_mean']:.4f}")
        print(f"  平均IC信息比率: {summary['avg_ic_ir']:.4f}")
        if summary.get('best_ic_factor'):
            print(f"  最佳IC因子: {summary['best_ic_factor']['name']} (IR: {summary['best_ic_factor']['ic_ir']:.4f})")
        
        return comprehensive_results
        
    except Exception as e:
        print(f"❌ 综合研究失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_report_generation(analysis_results, backtest_results=None):
    """演示报告生成功能"""
    print("\n📄 演示5: 研究报告生成")
    print("=" * 50)
    
    try:
        # 生成因子分析报告
        print("📝 生成因子分析报告...")
        report_path = create_research_report(
            analysis_results=analysis_results,
            title="因子研究分析报告",
            format="html",
            output_path="reports/demo"
        )
        print(f"✅ 因子分析报告已生成: {report_path}")
        
        # 如果有回测结果，生成回测报告
        if backtest_results:
            print("📝 生成回测报告...")
            backtest_report_path = create_research_report(
                backtest_results=backtest_results,
                title="因子策略回测报告",
                format="html",
                output_path="reports/demo"
            )
            print(f"✅ 回测报告已生成: {backtest_report_path}")
        
        # 生成Markdown版本
        print("📝 生成Markdown报告...")
        md_report_path = create_research_report(
            analysis_results=analysis_results,
            title="因子研究分析报告",
            format="markdown",
            output_path="reports/demo"
        )
        print(f"✅ Markdown报告已生成: {md_report_path}")
        
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")


def main():
    """主演示函数"""
    print("🚀 量化研究框架完整演示")
    print("=" * 60)
    
    try:
        # 1. 生成示例数据
        symbols = ['STOCK001', 'STOCK002', 'STOCK003']
        data, price_data, returns_data = generate_sample_data(symbols, periods=400)
        
        # 2. 演示因子库管理
        factor_lib = demo_factor_library()
        
        # 3. 演示因子分析
        # 使用第一只股票的数据进行演示
        stock_data = data.iloc[:, data.columns.get_level_values(0) == symbols[0]]
        stock_data.columns = stock_data.columns.droplevel(0)
        stock_returns = returns_data[symbols[0]]
        
        analysis_results, factor_data = demo_factor_analysis(factor_lib, stock_data, returns_data)
        
        # 4. 演示因子回测
        if analysis_results:
            backtest_results = demo_factor_backtest(factor_data, price_data, analysis_results)
        else:
            backtest_results = {}
        
        # 5. 演示综合研究
        comprehensive_results = demo_comprehensive_research()
        
        # 6. 演示报告生成
        if analysis_results:
            demo_report_generation(analysis_results, backtest_results)
        
        print("\n🎉 研究框架演示完成!")
        print("\n📋 演示总结:")
        print("✅ 因子库管理 - 创建、添加自定义因子、保存")
        print("✅ 因子有效性分析 - IC分析、多空收益、相关性等")
        print("✅ 因子策略回测 - 完整的回测流程和绩效评估")
        print("✅ 综合研究框架 - 一站式研究流程")
        print("✅ 研究报告生成 - HTML和Markdown格式")
        
        print(f"\n📁 输出文件位置:")
        print(f"  因子库: data/demo_factor_library/")
        print(f"  研究结果: data/demo_research/")
        print(f"  报告文件: reports/demo/")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()