#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试股票池加载是否正常
"""

import sys
from pathlib import Path

# 添加配置目录到路径
sys.path.append(str(Path(__file__).parent.parent / "config"))

try:
    from api_config import TUSHARE_TOKEN, OPENROUTER_API_KEY
    from daily_news_analyzer import DailyNewsAnalyzer

    print("🧪 正在测试股票池加载...")

    # 初始化分析器
    analyzer = DailyNewsAnalyzer(TUSHARE_TOKEN, OPENROUTER_API_KEY)

    print(f"✅ 股票池加载成功: {len(analyzer.stock_pool)} 只股票")
    print(f"✅ 关键词提取成功: {len(analyzer.stock_keywords)} 个股票关键词")

    # 显示前几个股票信息
    if analyzer.stock_pool:
        print("\n📋 前5只股票：")
        for i, stock in enumerate(analyzer.stock_pool[:5]):
            print(f"  {i+1}. {stock['stock_name']} ({stock['stock_code']}) - {stock['industry']}")

    print("\n🎉 股票池测试通过！")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()