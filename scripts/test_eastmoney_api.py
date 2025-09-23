#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试东方财富新闻API接口状态
"""

import requests
import json
import time
from datetime import datetime

def test_eastmoney_api():
    """测试东方财富API接口"""
    print("=" * 50)
    print("测试东方财富新闻API接口状态")
    print("=" * 50)

    # API配置
    api_url = "https://search-api-web.eastmoney.com/search/jsonp"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://so.eastmoney.com/',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
    }

    # 测试不同的搜索参数
    test_cases = [
        {
            "name": "原始参数",
            "keyword": "紫金矿业",
            "sort": "default",
            "page_size": 10
        },
        {
            "name": "按时间排序",
            "keyword": "紫金矿业",
            "sort": "time",
            "page_size": 10
        },
        {
            "name": "简化关键词",
            "keyword": "紫金",
            "sort": "default",
            "page_size": 5
        },
        {
            "name": "通用关键词",
            "keyword": "A股",
            "sort": "default",
            "page_size": 5
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n测试 {i+1}: {test_case['name']}")
        print(f"关键词: {test_case['keyword']}")

        # 构建API参数
        param_data = {
            "uid": "",
            "keyword": test_case['keyword'],
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": test_case['sort'],
                    "pageIndex": 1,
                    "pageSize": test_case['page_size'],
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
            print(f"请求URL: {api_url}")
            print(f"参数长度: {len(str(params))}")

            response = requests.get(api_url, headers=headers, params=params, timeout=10)

            print(f"HTTP状态码: {response.status_code}")
            print(f"响应长度: {len(response.text)}")

            if response.status_code == 200:
                # 尝试解析响应
                response_text = response.text
                print(f"响应开始: {response_text[:100]}...")

                if response_text.startswith('jQuery'):
                    try:
                        start = response_text.find('(') + 1
                        end = response_text.rfind(')')
                        json_str = response_text[start:end]

                        data = json.loads(json_str)
                        print(f"✓ JSON解析成功")

                        # 检查数据结构
                        if 'result' in data:
                            print(f"✓ 包含result字段")
                            if data['result']:
                                print(f"✓ result不为空")
                                if 'cmsArticleWebOld' in data['result']:
                                    articles = data['result']['cmsArticleWebOld']
                                    print(f"✓ 找到 {len(articles)} 条新闻")

                                    if articles:
                                        # 显示第一条新闻的结构
                                        first_article = articles[0]
                                        print(f"✓ 新闻字段: {list(first_article.keys())}")
                                        if 'title' in first_article:
                                            print(f"✓ 标题示例: {first_article['title'][:50]}...")
                                        if 'showTime' in first_article:
                                            print(f"✓ 时间示例: {first_article['showTime']}")
                                        if 'url' in first_article:
                                            print(f"✓ URL示例: {first_article['url'][:50]}...")
                                    else:
                                        print("× 新闻列表为空")
                                else:
                                    print("× 缺少cmsArticleWebOld字段")
                                    print(f"可用字段: {list(data['result'].keys()) if data['result'] else 'None'}")
                            else:
                                print("× result为空")
                        else:
                            print("× 缺少result字段")
                            print(f"响应字段: {list(data.keys())}")

                    except json.JSONDecodeError as e:
                        print(f"× JSON解析失败: {e}")
                        print(f"提取的JSON: {json_str[:200]}...")
                else:
                    print("× 响应不是JSONP格式")
                    try:
                        # 尝试直接解析JSON
                        data = json.loads(response_text)
                        print(f"✓ 直接JSON解析成功: {list(data.keys())}")
                    except:
                        print("× 无法解析为JSON")
            else:
                print(f"× HTTP请求失败")
                print(f"响应内容: {response.text[:200]}...")

        except requests.exceptions.Timeout:
            print("× 请求超时")
        except requests.exceptions.ConnectionError:
            print("× 连接错误")
        except Exception as e:
            print(f"× 其他错误: {e}")

    # 测试直接访问东方财富主页
    print("\n" + "=" * 50)
    print("测试东方财富主站连接")
    print("=" * 50)

    try:
        response = requests.get("https://www.eastmoney.com/", headers=headers, timeout=10)
        print(f"主站状态码: {response.status_code}")
        if response.status_code == 200:
            print("✓ 主站可访问")
        else:
            print("× 主站访问异常")
    except Exception as e:
        print(f"× 主站连接失败: {e}")

    # 测试搜索页面
    try:
        response = requests.get("https://so.eastmoney.com/", headers=headers, timeout=10)
        print(f"搜索页状态码: {response.status_code}")
        if response.status_code == 200:
            print("✓ 搜索页可访问")
        else:
            print("× 搜索页访问异常")
    except Exception as e:
        print(f"× 搜索页连接失败: {e}")

if __name__ == "__main__":
    test_eastmoney_api()