#!/usr/bin/env python3
"""
增强版基本面分析演示脚本
展示新增指标的完整功能
"""

from quantlib.fundamental.analyzer import FundamentalAnalyzer

def demo_enhanced_analysis():
    """演示增强版基本面分析功能"""
    
    print("="*80)
    print("增强版基本面分析演示")
    print("="*80)
    
    # 分析美股示例
    print("\n🇺🇸 美股分析示例 - Apple Inc. (AAPL)")
    print("-" * 60)
    
    analyzer = FundamentalAnalyzer('AAPL', market='US')
    
    if analyzer.load_company_data():
        # 基本财务分析
        analyzer.analyze_financial_statements()
        
        # 计算增强版财务比率（包含新指标）
        analyzer.calculate_financial_ratios()
        
        # 显示详细分类指标汇总
        analyzer.print_detailed_ratios_summary()
        
        # 同行对比
        print(f"\n{'='*60}")
        print("同行对比分析")
        print('='*60)
        tech_peers = ['MSFT', 'GOOGL', 'NVDA']
        analyzer.peer_comparison_analysis(tech_peers)
        
        # 生成投资摘要（包含新指标的评估）
        analyzer.generate_investment_summary()
    
    print(f"\n\n{'='*80}")
    print("新增指标说明")
    print('='*80)
    print("""
    🚀 成长性指标:
    - Revenue Growth: 营收增长率
    - Net Income Growth: 净利润增长率  
    - EPS Growth: 每股收益增长率

    💰 现金流质量指标:
    - Operating CF to Net Income: 经营现金流/净利润比率
    - Free Cash Flow: 自由现金流
    - Free Cash Flow Yield: 自由现金流收益率

    🏗️ 资产质量指标:
    - Goodwill Ratio: 商誉占比
    - Intangible Assets Ratio: 无形资产比例
    - Tangible Book Value Ratio: 有形资产净值比率

    💵 股息指标:
    - Dividend Yield: 股息收益率
    - Payout Ratio: 派息比率

    📊 市场表现指标:
    - Beta: 系统性风险系数
    - Volatility: 年化波动率
    - Sharpe Ratio: 夏普比率
    - 1M/3M Price Change: 1个月/3个月价格变化

    ⚖️ 增强版评分系统:
    - 重新调整了各指标权重
    - 新增成长性评分 (15分)
    - 新增现金流质量评分 (5分)
    - 新增资产质量评分 (10分)
    - 新增市场风险评分 (5分)
    - 更全面的风险识别系统
    """)

if __name__ == "__main__":
    demo_enhanced_analysis()