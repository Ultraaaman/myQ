#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版新闻爬虫 - 解决内容获取导致的数据丢失问题
"""

import requests
import json
import pandas as pd
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

class EastmoneyNewsCrawler:
    def __init__(self):
        self.api_url = "https://search-api-web.eastmoney.com/search/jsonp"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://so.eastmoney.com/',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
        }

    def get_news(self, keyword, sort_type='time', start_date=None, end_date=None, max_news=50):
        """
        获取指定时间范围内的新闻

        Args:
            keyword: 搜索关键词
            start_date: 开始日期，可以是字符串(YYYY-MM-DD)或datetime对象
            end_date: 结束日期，可以是字符串(YYYY-MM-DD)或datetime对象
            max_news: 最大新闻数量，默认50

        Returns:
            DataFrame: 包含时间和新闻内容的数据框
        """
        print(f"搜索关键词: {keyword}")

        # 处理时间范围
        if start_date is None or end_date is None:
            # 如果没有指定时间范围，默认最近7天
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            print(f"时间范围: 最近7天")
        else:
            # 转换时间格式
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')

            print(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

        # API参数
        param_data = {
            "uid": "",
            "keyword": keyword,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": sort_type,  # 按时间排序
                    "pageIndex": 1,
                    "pageSize": max_news,
                    "preTag": "<em>",
                    "postTag": "</em>"
                }
            }
        }

        params = {
            'cb': f'jQuery{int(time.time() * 1000)}_{int(time.time() * 1000) + 1}',
            'param': json.dumps(param_data)
        }

        try:
            response = requests.get(self.api_url, headers=self.headers, params=params, timeout=15)

            if response.status_code != 200:
                print(f"API请求失败，状态码: {response.status_code}")
                return pd.DataFrame()

            # 解析JSONP响应
            response_text = response.text
            if response_text.startswith('jQuery'):
                start = response_text.find('(') + 1
                end = response_text.rfind(')')
                json_str = response_text[start:end]
            else:
                json_str = response_text

            data = json.loads(json_str)

            if 'result' not in data or not data['result'] or 'cmsArticleWebOld' not in data['result']:
                print("未找到新闻数据")
                return pd.DataFrame()

            articles = data['result']['cmsArticleWebOld']
            print(f"API返回 {len(articles)} 条新闻")

            # 筛选时间范围内的新闻
            news_data = []
            content_success = 0
            content_failed = 0

            for article in articles:
                # 解析新闻时间
                date_str = article.get('showTime') or article.get('date', '')
                if not date_str:
                    continue

                try:
                    # 尝试多种时间格式
                    news_date = self._parse_date(date_str)
                    if not news_date:
                        continue

                    # 检查是否在时间范围内
                    if start_date <= news_date <= end_date:
                        title = self._clean_title(article.get('title', ''))
                        url = article.get('url', '')

                        # 修复：先添加基本信息，然后尝试获取内容
                        news_item = {
                            'date': news_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'title': title,
                            'url': url,
                            'content': f"{title}"  # 至少包含标题
                        }

                        # 尝试获取新闻内容（不强制要求）
                        try:
                            full_content = self._get_content(url)
                            if full_content and len(full_content.strip()) > 50:
                                news_item['content'] = f"{title}\n\n{full_content}"
                                content_success += 1
                            else:
                                # 使用API返回的摘要内容作为备选
                                if 'content' in article and article['content']:
                                    summary = self._clean_title(article['content'])
                                    news_item['content'] = f"{title}\n\n{summary}"
                                content_failed += 1
                        except Exception as e:
                            print(f"获取内容失败: {e}")
                            content_failed += 1

                        news_data.append(news_item)
                        print(f"✓ 获取新闻: {title[:30]}... ({news_date.strftime('%Y-%m-%d')})")

                except Exception as e:
                    print(f"处理新闻时出错: {e}")
                    continue

            if news_data:
                df = pd.DataFrame(news_data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False).reset_index(drop=True)
                print(f"\n成功获取 {len(df)} 条符合时间范围的新闻")
                print(f"内容获取统计: 成功 {content_success} 条，失败 {content_failed} 条")
                return df
            else:
                print("未获取到符合时间范围的新闻")
                return pd.DataFrame()

        except Exception as e:
            print(f"获取新闻失败: {e}")
            return pd.DataFrame()

    def _parse_date(self, date_str):
        """解析各种格式的日期字符串"""
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d %H:%M',
            '%Y/%m/%d',
            '%m-%d %H:%M',
            '%m/%d %H:%M'
        ]

        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                # 如果只有月日，补充年份
                if fmt in ['%m-%d %H:%M', '%m/%d %H:%M']:
                    parsed_date = parsed_date.replace(year=datetime.now().year)
                return parsed_date
            except ValueError:
                continue

        return None

    def _clean_title(self, title):
        """清理标题中的HTML标签"""
        return re.sub(r'<[^>]+>', '', title).strip()

    def _get_content(self, url):
        """获取新闻正文内容（增强版，更容错）"""
        if not url:
            return ""

        try:
            response = requests.get(url, headers=self.headers, timeout=5)  # 减少超时时间
            if response.status_code != 200:
                return ""

            soup = BeautifulSoup(response.text, 'html.parser')

            # 尝试常见的内容选择器
            selectors = [
                '.ArticleBody',
                '.content',
                '.article-content',
                '#ContentBody',
                'article',
                '.post-content',
                '.article_body',  # 新增
                '.text-content',  # 新增
                '#article_body'   # 新增
            ]

            for selector in selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    text = content_div.get_text().strip()
                    text = re.sub(r'\s+', ' ', text)
                    if len(text) > 50:
                        return text

            # 备用方案：获取所有段落
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = ' '.join([p.get_text().strip() for p in paragraphs])
                content = re.sub(r'\s+', ' ', content)
                if len(content) > 50:
                    return content

        except Exception as e:
            print(f"获取内容失败: {e}")

        return ""

    def save_to_csv(self, df, filename='news_data.csv', append=True):
        """
        保存数据到CSV文件

        Args:
            df: 要保存的DataFrame
            filename: 文件名
            append: 是否追加模式，True为追加，False为覆盖
        """
        import os

        if df.empty:
            print("没有数据可保存")
            return

        if append and os.path.exists(filename):
            # 追加模式：读取现有数据，合并去重
            try:
                existing_df = pd.read_csv(filename, encoding='utf-8-sig')
                existing_df['date'] = pd.to_datetime(existing_df['date'])

                # 合并数据
                combined_df = pd.concat([existing_df, df], ignore_index=True)

                # 去重（基于date和content的前100个字符）
                combined_df['content_short'] = combined_df['content'].str[:100]
                combined_df = combined_df.drop_duplicates(subset=['date', 'content_short'], keep='last')
                combined_df = combined_df.drop('content_short', axis=1)

                # 按时间排序
                combined_df = combined_df.sort_values('date', ascending=False).reset_index(drop=True)

                # 保存
                combined_df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"\n数据已追加到 {filename}")
                print(f"原有数据: {len(existing_df)} 条，新增数据: {len(df)} 条，合并后: {len(combined_df)} 条")

            except Exception as e:
                print(f"追加模式失败，改为覆盖模式: {e}")
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"\n数据已保存到 {filename}")
        else:
            # 覆盖模式
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n数据已保存到 {filename}")

# 使用示例
if __name__ == "__main__":
    crawler = EastmoneyNewsCrawler()

    # 示例1: 指定时间范围
    df = crawler.get_news("紫金矿业",
                         sort_type='default',
                         start_date='2025-03-01',  # 开始日期
                         end_date='2025-09-23',    # 结束日期
                         max_news=5000)

    # 示例2: 使用默认时间范围（最近7天）
    # df = crawler.get_news("紫金矿业", max_news=20)

    if not df.empty:
        print(f"\n=== 获取结果 ===")
        print(f"总计: {len(df)} 条新闻")
        print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")

        # 显示前几条
        print("\n=== 前3条新闻 ===")
        for i, row in df.head(3).iterrows():
            print(f"\n{i+1}. 时间: {row['date']}")
            print(f"标题: {row.get('title', 'N/A')}")
            print(f"内容长度: {len(row['content'])} 字符")
            print(f"内容预览: {row['content'][:100]}...")

        # 保存数据（追加模式）
        crawler.save_to_csv(df, 'news_data.csv', append=True)
    else:
        print("未获取到新闻数据")