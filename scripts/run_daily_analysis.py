#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日新闻因子分析启动脚本
支持命令行参数指定分析日期
"""

import sys
import argparse
from datetime import datetime
from daily_news_analyzer import DailyNewsAnalyzer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='每日新闻因子分析')
    parser.add_argument('--date', '-d',
                       help='分析日期 (YYYY-MM-DD)，默认为今天',
                       default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--config', '-c',
                       help='配置文件路径',
                       default='../config/api_config.py')

    args = parser.parse_args()

    # 验证日期格式
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
        target_date = args.date
    except ValueError:
        print("错误：日期格式应为 YYYY-MM-DD")
        return

    # 导入配置
    try:
        sys.path.append(str(Path(__file__).parent.parent / "config"))
        from api_config import TUSHARE_TOKEN, OPENROUTER_API_KEY

        if TUSHARE_TOKEN == "your_tushare_token_here" or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
            print("错误：请在 config/api_config.py 中配置正确的API密钥")
            return

    except ImportError:
        print("错误：无法导入配置文件，请检查 config/api_config.py")
        return

    print(f"开始分析 {target_date} 的新闻因子...")

    try:
        # 初始化分析器
        analyzer = DailyNewsAnalyzer(TUSHARE_TOKEN, OPENROUTER_API_KEY)

        # 1. 获取当日新闻
        print("正在获取新闻数据...")
        news_df = analyzer.get_daily_news(target_date)
        if news_df.empty:
            print("未获取到新闻数据，程序结束")
            return

        # 2. 匹配新闻到股票
        print("正在匹配股票相关新闻...")
        matched_df = analyzer.match_news_to_stocks(news_df)
        if matched_df.empty:
            print("未匹配到相关股票新闻，程序结束")
            return

        print(f"找到 {len(matched_df)} 条股票相关新闻")

        # 3. 大模型评分
        print("正在使用大模型评分...")
        scored_df = analyzer.score_news_with_llm(matched_df)
        if scored_df.empty:
            print("评分失败，程序结束")
            return

        # 4. 保存结果
        print("正在保存结果...")
        analyzer.save_results(scored_df, target_date)

        # 5. 生成因子强度报告
        print("正在生成因子强度报告...")
        strong_stocks = analyzer.generate_factor_report(scored_df, target_date)

        print(f"\n✅ 分析完成！")
        print(f"📊 分析了 {len(scored_df)} 条股票新闻")

        if not strong_stocks.empty:
            print(f"🔥 发现 {len(strong_stocks)} 只强因子股票：")
            for idx, row in strong_stocks.head(5).iterrows():
                print(f"   {idx+1}. {row['stock_name']} ({row['stock_code']}): 因子强度 {row['factor_strength']:.2f}")
        else:
            print("📉 今日未发现强因子股票")

        print(f"\n📁 结果已保存到: D:/projects/q/myQ/output/daily_analysis/")

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()