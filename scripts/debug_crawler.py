#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试版新闻爬虫 - 详细分析时间过滤问题
"""

import requests
import json
import pandas as pd
import time
from datetime import datetime, timedelta

def debug_news_crawler():
    """调试新闻爬虫的时间过滤问题"""
    print("=" * 60)
    print("调试新闻爬虫时间过滤问题")
    print("=" * 60)

    # API配置
    api_url = "https://search-api-web.eastmoney.com/search/jsonp"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://so.eastmoney.com/',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
    }

    # 测试时间范围设置
    keyword = "紫金矿业"

    # 问题范围：2025-06-01 到 2025-09-22
    start_date = datetime.strptime('2025-06-01', '%Y-%m-%d')
    end_date = datetime.strptime('2025-09-22', '%Y-%m-%d')

    print(f"搜索关键词: {keyword}")
    print(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
                "sort": "default",
                "pageIndex": 1,
                "pageSize": 50,  # 增加数量
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
        response = requests.get(api_url, headers=headers, params=params, timeout=15)

        if response.status_code != 200:
            print(f"API请求失败，状态码: {response.status_code}")
            return

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
            return

        articles = data['result']['cmsArticleWebOld']
        print(f"API返回 {len(articles)} 条新闻")

        # 详细分析每条新闻的时间
        print(f"\n详细分析前10条新闻的时间信息:")
        print("-" * 80)

        valid_count = 0
        invalid_count = 0
        time_parse_errors = 0

        for i, article in enumerate(articles[:10]):  # 只分析前10条
            print(f"\n第{i+1}条新闻:")

            # 获取时间字段
            date_str = article.get('showTime') or article.get('date', '')
            title = article.get('title', '').replace('<em>', '').replace('</em>', '')

            print(f"  标题: {title[:50]}...")
            print(f"  原始时间字段: showTime='{article.get('showTime')}', date='{article.get('date')}'")
            print(f"  使用时间字符串: '{date_str}'")

            if not date_str:
                print(f"  ✗ 没有时间信息")
                invalid_count += 1
                continue

            # 尝试解析时间
            try:
                news_date = parse_date_debug(date_str)
                if news_date:
                    print(f"  ✓ 解析后时间: {news_date.strftime('%Y-%m-%d %H:%M:%S')}")

                    # 检查是否在范围内
                    if start_date <= news_date <= end_date:
                        print(f"  ✓ 在时间范围内")
                        valid_count += 1
                    else:
                        print(f"  ✗ 不在时间范围内")
                        print(f"    范围: {start_date.strftime('%Y-%m-%d')} <= {news_date.strftime('%Y-%m-%d')} <= {end_date.strftime('%Y-%m-%d')}")
                        print(f"    比较结果: start_date <= news_date = {start_date <= news_date}")
                        print(f"    比较结果: news_date <= end_date = {news_date <= end_date}")
                        invalid_count += 1
                else:
                    print(f"  ✗ 时间解析失败")
                    time_parse_errors += 1
            except Exception as e:
                print(f"  ✗ 时间解析出错: {e}")
                time_parse_errors += 1

        print(f"\n" + "=" * 60)
        print(f"分析总结:")
        print(f"  API返回新闻总数: {len(articles)}")
        print(f"  分析的新闻数量: {min(10, len(articles))}")
        print(f"  时间范围内新闻: {valid_count}")
        print(f"  时间范围外新闻: {invalid_count}")
        print(f"  时间解析错误: {time_parse_errors}")

        # 如果没有在范围内的新闻，尝试最近几天的范围
        if valid_count == 0:
            print(f"\n尝试最近7天的范围:")
            recent_end = datetime.now()
            recent_start = recent_end - timedelta(days=7)
            print(f"最近7天范围: {recent_start.strftime('%Y-%m-%d')} 到 {recent_end.strftime('%Y-%m-%d')}")

            recent_valid = 0
            for article in articles[:10]:
                date_str = article.get('showTime') or article.get('date', '')
                if date_str:
                    try:
                        news_date = parse_date_debug(date_str)
                        if news_date and recent_start <= news_date <= recent_end:
                            recent_valid += 1
                    except:
                        pass

            print(f"最近7天范围内新闻: {recent_valid}")

    except Exception as e:
        print(f"调试过程中出错: {e}")

def parse_date_debug(date_str):
    """调试版时间解析函数"""
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

if __name__ == "__main__":
    debug_news_crawler()