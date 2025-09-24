#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化测试 - 验证系统核心功能
"""

import sys
from pathlib import Path
import pandas as pd

# 添加配置目录到路径
sys.path.append(str(Path(__file__).parent.parent / "config"))

def test_system():
    """测试系统核心功能"""
    try:
        print("🧪 开始测试每日新闻分析系统...")

        # 1. 测试配置加载
        print("\n📋 步骤 1: 测试配置加载...")
        from api_config import TUSHARE_TOKEN, OPENROUTER_API_KEY

        if TUSHARE_TOKEN == "your_tushare_token_here":
            print("⚠️  Warning: Tushare token 未配置")
        else:
            print("✅ Tushare token 已配置")

        if OPENROUTER_API_KEY == "your_openrouter_api_key_here":
            print("⚠️  Warning: OpenRouter API key 未配置")
        else:
            print("✅ OpenRouter API key 已配置")

        # 2. 测试分析器初始化
        print("\n📋 步骤 2: 测试分析器初始化...")
        from daily_news_analyzer import DailyNewsAnalyzer

        analyzer = DailyNewsAnalyzer(TUSHARE_TOKEN, OPENROUTER_API_KEY)
        print(f"✅ 分析器初始化成功")
        print(f"   - 股票池: {len(analyzer.stock_pool)} 只股票")
        print(f"   - 关键词: {len(analyzer.stock_keywords)} 个股票")

        # 3. 显示股票池样本
        if analyzer.stock_pool:
            print("\n📋 步骤 3: 股票池样本...")
            for i, stock in enumerate(analyzer.stock_pool[:3]):
                print(f"   {i+1}. {stock['stock_name']} ({stock['stock_code']}) - {stock['industry']}")

        # 4. 测试新闻获取（用历史日期）
        print("\n📋 步骤 4: 测试新闻获取（2024-12-20）...")
        news_df = analyzer.get_daily_news("2024-12-20")

        if not news_df.empty:
            print(f"✅ 成功获取 {len(news_df)} 条新闻")
            print("   前3条新闻标题:")
            for i, row in news_df.head(3).iterrows():
                title = str(row['original_title'])[:50] + "..." if len(str(row['original_title'])) > 50 else str(row['original_title'])
                print(f"   {i+1}. {title}")

            # 5. 测试新闻匹配
            print("\n📋 步骤 5: 测试新闻匹配...")
            matched_df = analyzer.match_news_to_stocks(news_df)

            if not matched_df.empty:
                print(f"✅ 成功匹配 {len(matched_df)} 条股票相关新闻")

                # 显示匹配结果样本
                unique_stocks = matched_df['stock_name'].unique()
                print(f"   涉及股票: {len(unique_stocks)} 只")
                for stock in unique_stocks[:5]:
                    count = len(matched_df[matched_df['stock_name'] == stock])
                    print(f"   - {stock}: {count} 条新闻")
            else:
                print("⚠️  未匹配到股票相关新闻")
        else:
            print("⚠️  未获取到新闻数据，可能该日期无新闻或API限制")

        print("\n🎉 系统测试完成！")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_system()