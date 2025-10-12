#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间段新闻抓取脚本（精简版）

功能说明：
- 支持指定日期范围抓取新闻（从开始日期到结束日期）
- 自动匹配股票池中的股票
- 仅抓取和整理新闻数据，输出单个CSV汇总文件
- 自动过滤周末，只处理工作日

使用方法：
    python period_news_fetcher.py --start_date 2025-09-01 --end_date 2025-09-30

参数说明：
    --start_date: 开始日期 (格式: YYYY-MM-DD)
    --end_date: 结束日期 (格式: YYYY-MM-DD)

输出文件：
    news_YYYYMMDD_YYYYMMDD.csv - 新闻汇总文件
"""

import tushare as ts
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PeriodNewsFetcher:
    def __init__(self, tushare_token):
        """初始化新闻抓取器"""
        print(f"🔧 正在初始化新闻抓取器...")

        # 初始化Tushare API
        print(f"🔧 初始化Tushare API...")
        self.ts_pro = ts.pro_api(tushare_token)
        print(f"✅ Tushare API初始化完成")

        # 设置文件路径
        print(f"🔧 设置文件路径...")
        self.base_dir = Path("D:/projects/q/myQ")
        self.output_dir = self.base_dir / "output" / "period_news"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 输出目录: {self.output_dir}")

        # 加载股票池
        print(f"🔧 加载股票池...")
        self.stock_pool = self._load_stock_pool()
        print(f"✅ 股票池加载完成: {len(self.stock_pool)} 只股票")

        # 提取关键词
        print(f"🔧 提取关键词...")
        self.stock_keywords = self._extract_keywords()
        print(f"✅ 关键词提取完成: {len(self.stock_keywords)} 个股票")

    def _load_stock_pool(self):
        """加载股票池"""
        stock_pool_path = self.base_dir / "config" / "stock_pool.json"
        print(f"📁 读取股票池文件: {stock_pool_path}")

        with open(stock_pool_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 支持不同的JSON格式
        if 'stocks' in data:
            stocks = data['stocks']
        elif 'stock_database' in data:
            stocks = data['stock_database']
        else:
            stocks = data if isinstance(data, list) else []

        print(f"📊 最终股票数量: {len(stocks)}")
        return stocks

    def _extract_keywords(self):
        """提取股票关键词"""
        keywords = {}

        for stock in self.stock_pool:
            stock_keywords = []
            stock_keywords.append(stock['stock_name'])
            stock_keywords.append(stock['stock_code'])

            # 去重
            keywords[stock['stock_code']] = list(set(stock_keywords))

        return keywords

    def get_news_by_date(self, target_date):
        """获取指定日期的新闻"""
        start_datetime = f"{target_date} 09:00:00"
        end_datetime = f"{target_date} 18:00:00"

        try:
            news_df = self.ts_pro.news(
                src='sina',
                start_date=start_datetime,
                end_date=end_datetime
            )

            if news_df is not None and not news_df.empty:
                return news_df
            else:
                return pd.DataFrame()

        except Exception as e:
            return pd.DataFrame()

    def match_news_to_stocks(self, news_df, date_str=None):
        """匹配新闻到股票"""
        if news_df.empty:
            return pd.DataFrame()

        matched_records = []

        for _, news_row in news_df.iterrows():
            # 安全获取字段值
            title = news_row.get('title') or news_row.get('Title') or ''
            content = news_row.get('content') or news_row.get('Content') or ''

            news_title = str(title).lower() if title else ''
            news_content = str(content).lower() if content else ''
            full_text = f"{news_title} {news_content}"

            # 匹配股票
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
                    record = news_row.to_dict()
                    record.update({
                        'stock_code': stock['stock_code'],
                        'stock_name': stock['stock_name'],
                        'industry': stock.get('industry', ''),
                        'matched_keyword': matched_keyword
                    })

                    # 添加抓取日期（如果提供）
                    if date_str:
                        record['fetch_date'] = date_str

                    matched_records.append(record)

        matched_df = pd.DataFrame(matched_records)
        return matched_df

    def generate_date_range(self, start_date, end_date):
        """生成日期范围列表（仅工作日）"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        date_list = []
        current = start

        while current <= end:
            # 过滤周末 (0=周一, 6=周日)
            if current.weekday() < 5:  # 周一到周五
                date_list.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        return date_list

    def fetch_period_news(self, start_date, end_date):
        """
        抓取时间段内的新闻

        参数:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        print(f"\n🚀 开始抓取时间段新闻")
        print(f"   开始日期: {start_date}")
        print(f"   结束日期: {end_date}")

        # 生成日期列表
        date_list = self.generate_date_range(start_date, end_date)
        print(f"\n📅 共需抓取 {len(date_list)} 个交易日")

        # 设置输出文件名
        summary_filename = self.output_dir / f"news_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        print(f"💾 输出文件: {summary_filename}\n")

        total_matched = 0
        first_write = True

        # 逐日抓取并立即写入
        for i, date_str in enumerate(date_list, 1):
            print(f"处理进度: [{i}/{len(date_list)}] {date_str}", end=" ")

            # 获取当日新闻
            news_df = self.get_news_by_date(date_str)

            if news_df.empty:
                print("- 无数据")
                continue

            # 匹配股票
            matched_df = self.match_news_to_stocks(news_df, date_str)

            # 立即写入CSV文件
            if not matched_df.empty:
                if first_write:
                    # 第一次写入，创建文件并写入表头
                    matched_df.to_csv(summary_filename, mode='w', index=False, encoding='utf-8-sig')
                    first_write = False
                else:
                    # 后续写入，追加模式，不写表头
                    matched_df.to_csv(summary_filename, mode='a', header=False, index=False, encoding='utf-8-sig')

                total_matched += len(matched_df)
                print(f"- 匹配 {len(matched_df)} 条 ✓")
            else:
                print("- 无匹配")

            # 避免API限制，添加延迟
            if i < len(date_list):
                import time
                time.sleep(1)

        # 显示最终统计
        if total_matched > 0:
            print(f"\n🎉 抓取完成！")
            print(f"💾 汇总文件: {summary_filename}")
            print(f"📊 总新闻数: {total_matched}")

            # 读取文件统计涉及股票数
            final_df = pd.read_csv(summary_filename, encoding='utf-8-sig')
            print(f"📊 涉及股票: {final_df['stock_code'].nunique()}")
        else:
            print(f"\n⚠️ 未抓取到任何新闻数据")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='时间段新闻抓取工具')
    parser.add_argument('--start_date', type=str, default='2022-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-10-10', help='结束日期 (YYYY-MM-DD)')

    args = parser.parse_args()

    # 导入配置
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "config"))
    from api_config import TUSHARE_TOKEN

    print(f"🔑 配置检查:")
    print(f"   - Tushare Token: {'已配置' if TUSHARE_TOKEN != 'your_tushare_token_here' else '未配置'}")

    # 初始化抓取器
    fetcher = PeriodNewsFetcher(TUSHARE_TOKEN)

    # 执行抓取
    fetcher.fetch_period_news(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
