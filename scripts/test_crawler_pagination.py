#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持分页的新闻爬虫 - 解决pageSize限制问题
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

    def get_news(self, keyword, sort_type='time', start_date=None, end_date=None, max_news=50, page_size=50):
        """
        获取指定时间范围内的新闻（支持分页）

        Args:
            keyword: 搜索关键词
            start_date: 开始日期，可以是字符串(YYYY-MM-DD)或datetime对象
            end_date: 结束日期，可以是字符串(YYYY-MM-DD)或datetime对象
            max_news: 最大新闻数量
            page_size: 每页大小（建议50-100，避免超出API限制）

        Returns:
            DataFrame: 包含时间和新闻内容的数据框
        """
        print(f"搜索关键词: {keyword}")
        print(f"目标数量: {max_news} 条新闻，每页 {page_size} 条")

        # 处理时间范围
        if start_date is None or end_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            print(f"时间范围: 最近7天")
        else:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            print(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

        all_news_data = []
        page_index = 1
        max_pages = (max_news // page_size) + 1

        print(f"\n开始分页查询，最多查询 {max_pages} 页...")

        while len(all_news_data) < max_news:
            print(f"\n--- 第 {page_index} 页 ---")

            # 获取当前页数据
            news_data = self._get_page_news(
                keyword, sort_type, start_date, end_date,
                page_index, page_size
            )

            if not news_data:
                print(f"第 {page_index} 页没有数据，停止查询")
                break

            print(f"第 {page_index} 页获取到 {len(news_data)} 条新闻")
            all_news_data.extend(news_data)

            # 检查是否还有更多页面
            if len(news_data) < page_size:
                print(f"当前页数据少于 {page_size} 条，可能已是最后一页")
                break

            page_index += 1

            # 避免请求过快
            time.sleep(0.5)

        # 限制总数量
        if len(all_news_data) > max_news:
            all_news_data = all_news_data[:max_news]

        if all_news_data:
            df = pd.DataFrame(all_news_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False).reset_index(drop=True)

            # 去重（基于时间和标题）
            df_unique = df.drop_duplicates(subset=['date', 'title'], keep='first').reset_index(drop=True)

            print(f"\n=== 查询完成 ===")
            print(f"查询页数: {page_index-1} 页")
            print(f"原始数据: {len(df)} 条")
            print(f"去重后: {len(df_unique)} 条")
            print(f"时间范围: {df_unique['date'].min()} 到 {df_unique['date'].max()}")

            return df_unique
        else:
            print("未获取到符合条件的新闻")
            return pd.DataFrame()

    def _get_page_news(self, keyword, sort_type, start_date, end_date, page_index, page_size):
        """获取指定页的新闻数据"""

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
                    "sort": sort_type,
                    "pageIndex": page_index,  # 页码
                    "pageSize": page_size,    # 每页大小
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
                return []

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
                print(f"第 {page_index} 页: 未找到新闻数据")
                return []

            articles = data['result']['cmsArticleWebOld']
            print(f"第 {page_index} 页: API返回 {len(articles)} 条新闻")

            # 筛选时间范围内的新闻
            news_data = []
            for article in articles:
                date_str = article.get('showTime') or article.get('date', '')
                if not date_str:
                    continue

                try:
                    news_date = self._parse_date(date_str)
                    if not news_date:
                        continue

                    # 检查时间范围
                    if start_date <= news_date <= end_date:
                        title = self._clean_title(article.get('title', ''))
                        url = article.get('url', '')

                        # 构建新闻数据
                        news_item = {
                            'date': news_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'title': title,
                            'url': url,
                            'content': title  # 基础内容
                        }

                        # 尝试获取完整内容
                        try:
                            full_content = self._get_content(url)
                            if full_content and len(full_content.strip()) > 50:
                                news_item['content'] = f"{title}\n\n{full_content}"
                            elif 'content' in article and article['content']:
                                # 使用API摘要
                                summary = self._clean_title(article['content'])
                                news_item['content'] = f"{title}\n\n{summary}"
                        except:
                            pass  # 保持基础内容

                        news_data.append(news_item)

                except Exception as e:
                    print(f"处理新闻时出错: {e}")
                    continue

            return news_data

        except Exception as e:
            print(f"获取第 {page_index} 页失败: {e}")
            return []

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
        """获取新闻正文内容"""
        if not url:
            return ""

        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code != 200:
                return ""

            soup = BeautifulSoup(response.text, 'html.parser')

            selectors = [
                '.ArticleBody', '.content', '.article-content', '#ContentBody',
                'article', '.post-content', '.article_body', '.text-content', '#article_body'
            ]

            for selector in selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    text = content_div.get_text().strip()
                    text = re.sub(r'\s+', ' ', text)
                    if len(text) > 50:
                        return text

            # 备用方案
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = ' '.join([p.get_text().strip() for p in paragraphs])
                content = re.sub(r'\s+', ' ', content)
                if len(content) > 50:
                    return content

        except:
            pass
        return ""

    def save_to_csv(self, df, filename='news_data.csv', append=True):
        """保存数据到CSV文件"""
        import os

        if df.empty:
            print("没有数据可保存")
            return

        if append and os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename, encoding='utf-8-sig')
                existing_df['date'] = pd.to_datetime(existing_df['date'])

                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df['content_short'] = combined_df['content'].str[:100]
                combined_df = combined_df.drop_duplicates(subset=['date', 'content_short'], keep='last')
                combined_df = combined_df.drop('content_short', axis=1)
                combined_df = combined_df.sort_values('date', ascending=False).reset_index(drop=True)

                combined_df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"\n数据已追加到 {filename}")
                print(f"原有数据: {len(existing_df)} 条，新增数据: {len(df)} 条，合并后: {len(combined_df)} 条")

            except Exception as e:
                print(f"追加模式失败，改为覆盖模式: {e}")
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"\n数据已保存到 {filename}")
        else:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n数据已保存到 {filename}")

# 使用示例
if __name__ == "__main__":
    crawler = EastmoneyNewsCrawler()

    # 测试分页功能
    df = crawler.get_news(
        keyword="紫金矿业",
        sort_type='default',
        start_date='2025-03-01',
        end_date='2025-09-23',
        max_news=5000,      # 想要的总数量
        page_size=50       # 每页大小（安全范围）
    )

    if not df.empty:
        print(f"\n=== 最终结果 ===")
        print(f"成功获取 {len(df)} 条新闻")
        print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")

        # 显示前几条
        print("\n=== 前3条新闻 ===")
        for i, row in df.head(3).iterrows():
            print(f"\n{i+1}. 时间: {row['date']}")
            print(f"标题: {row.get('title', 'N/A')}")
            print(f"内容长度: {len(row['content'])} 字符")

        # 保存数据
        crawler.save_to_csv(df, 'news_data.csv', append=True)
    else:
        print("未获取到新闻数据")