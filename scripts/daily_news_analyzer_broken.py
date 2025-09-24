#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日新闻分析脚本（修复日期时间格式版本）
"""

import tushare as ts
import pandas as pd
import json
import requests
import os
import time
import random
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyNewsAnalyzer:
    def __init__(self, tushare_token, openrouter_api_key):
        """初始化新闻分析器 - 调试版"""
        print(f"🔧 正在初始化分析器...")

        # 初始化API
        print(f"🔧 初始化Tushare API...")
        self.ts_pro = ts.pro_api(tushare_token)
        print(f"✅ Tushare API初始化完成")

        self.api_key = openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }

        # 设置文件路径
        print(f"🔧 设置文件路径...")
        self.base_dir = Path("D:/projects/q/myQ")
        self.output_dir = self.base_dir / "output" / "daily_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 输出目录: {self.output_dir}")

        # 加载股票池
        print(f"🔧 加载股票池...")
        self.stock_pool = self._load_stock_pool()
        print(f"✅ 股票池加载完成: {len(self.stock_pool)} 只股票")

        print(f"🔧 提取关键词...")
        self.stock_keywords = self._extract_keywords()
        print(f"✅ 关键词提取完成: {len(self.stock_keywords)} 个股票")

    def _load_stock_pool(self):
        """加载股票池 - 调试版"""
        stock_pool_path = self.base_dir / "config" / "stock_pool.json"
        print(f"📁 读取股票池文件: {stock_pool_path}")

        with open(stock_pool_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📊 原始JSON数据键: {list(data.keys())}")

        # 支持不同的JSON格式
        if 'stocks' in data:
            stocks = data['stocks']
            print(f"✅ 使用 'stocks' 键")
        elif 'stock_database' in data:
            stocks = data['stock_database']
            print(f"✅ 使用 'stock_database' 键")
        else:
            stocks = data if isinstance(data, list) else []
            print(f"✅ 数据本身就是列表")

        print(f"📊 最终股票数量: {len(stocks)}")
        return stocks

    def _extract_keywords(self):
        """提取关键词 - 调试版"""
        keywords = {}

        for i, stock in enumerate(self.stock_pool):
            stock_keywords = []
            stock_keywords.append(stock['stock_name'])
            stock_keywords.append(stock['stock_code'])

            # 提取行业和主营业务关键词
            if 'industry' in stock:
                stock_keywords.extend(stock['industry'].split())
            if 'main_business' in stock:
                business_words = stock['main_business'].replace('，', ' ').replace('、', ' ').split()
                business_words = [w for w in business_words if len(w) >= 2][:5]
                stock_keywords.extend(business_words)

            # 去重
            keywords[stock['stock_code']] = list(set(stock_keywords))

            # 显示前3个股票的关键词用于调试
            if i < 3:
                print(f"  {stock['stock_name']} ({stock['stock_code']}): {keywords[stock['stock_code']]}")

        return keywords

    def get_daily_news(self, target_date=None):
        """获取当日新闻 - 修复日期时间格式版"""
        if target_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d 09:00:00')
            end_date = datetime.now().strftime('%Y-%m-%d 18:00:00')
        else:
            # 修正：使用完整的日期时间格式 YYYY-MM-DD HH:MM:SS
            start_date = f"{target_date} 09:00:00"
            end_date = f"{target_date} 18:00:00"

        print(f"📰 准备获取新闻数据...")
        print(f"   - 开始时间: {start_date}")
        print(f"   - 结束时间: {end_date}")
        print(f"   - API对象: {type(self.ts_pro)}")

        # 直接调用API，不使用try-except
        print(f"🔧 调用 self.ts_pro.news()...")
        news_df = self.ts_pro.news(
            src='sina',
            start_date=start_date,
            end_date=end_date
        )

        print(f"📊 API返回结果:")
        print(f"   - 数据类型: {type(news_df)}")
        print(f"   - 是否为空: {news_df.empty if hasattr(news_df, 'empty') else 'N/A'}")
        print(f"   - 数据形状: {news_df.shape if hasattr(news_df, 'shape') else 'N/A'}")

        if hasattr(news_df, 'columns'):
            print(f"   - 列名: {list(news_df.columns)}")

        if hasattr(news_df, 'empty') and not news_df.empty:
            print(f"   - 前3行数据:")
            print(news_df.head(3))

        return news_df

    def match_news_to_stocks(self, news_df):
        """匹配新闻到股票 - 调试版"""
        if news_df.empty:
            print("⚠️ 新闻数据为空，跳过匹配")
            return news_df

        print(f"🔍 开始匹配新闻到股票...")
        matched_records = []

        for i, (_, news_row) in enumerate(news_df.iterrows()):
            if i < 2:  # 只处理前2条新闻用于调试
                print(f"  处理第 {i+1} 条新闻:")
                print(f"    标题: {news_row['title'][:50]}...")

            news_title = str(news_row['title']).lower()
            news_content = str(news_row.get('content', '')).lower()
            full_text = f"{news_title} {news_content}"

            # 匹配股票
            matched_stocks = []
            for stock in self.stock_pool:
                stock_code = stock['stock_code']
                keywords = self.stock_keywords.get(stock_code, [])

                # 检查是否匹配
                matched = False
                matched_keyword = None
                for keyword in keywords:
                    if len(keyword) >= 2 and keyword.lower() in full_text:
                        matched = True
                        matched_keyword = keyword
                        break

                if matched:
                    matched_stocks.append((stock['stock_name'], matched_keyword))
                    record = news_row.to_dict()
                    record.update(stock)
                    matched_records.append(record)

            if i < 2 and matched_stocks:  # 调试信息
                print(f"    匹配到股票: {matched_stocks}")

        matched_df = pd.DataFrame(matched_records)
        print(f"🎯 匹配完成，找到 {len(matched_df)} 条股票相关新闻")

        return matched_df

def main():
    """简化的主函数用于调试"""
    print("🧪 开始调试测试...")

    # 导入配置
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "config"))
    from api_config import TUSHARE_TOKEN, OPENROUTER_API_KEY

    print(f"🔑 配置检查:")
    print(f"   - Tushare Token: {'已配置' if TUSHARE_TOKEN != 'your_tushare_token_here' else '未配置'}")
    print(f"   - OpenRouter Key: {'已配置' if OPENROUTER_API_KEY != 'your_openrouter_api_key_here' else '未配置'}")

    # 初始化分析器
    analyzer = DailyNewsAnalyzer(TUSHARE_TOKEN, OPENROUTER_API_KEY)

    # 测试新闻获取
    print(f"\n📰 测试新闻获取...")
    news_df = analyzer.get_daily_news("2024-12-20")

    # 如果有新闻，测试匹配
    if hasattr(news_df, 'empty') and not news_df.empty:
        print(f"\n🔍 测试新闻匹配...")
        matched_df = analyzer.match_news_to_stocks(news_df)
    else:
        print(f"❌ 没有新闻数据，跳过匹配测试")

    print(f"\n🎉 调试测试完成！")

if __name__ == "__main__":
    main()