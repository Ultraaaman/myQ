#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 OpenRouter 上 DeepSeek 模型的可用性
"""

import requests
import json
import sys
import io
from pathlib import Path

# 设置标准输出为 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 导入配置
sys.path.append(str(Path(__file__).parent.parent / "config"))
from api_config import OPENROUTER_API_KEY

def test_model(model_name):
    """测试指定模型"""
    print(f"\n🔧 测试模型: {model_name}")

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "你好，请回复'测试成功'"}
        ],
        "max_tokens": 50
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)

        print(f"   状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"   ✅ 成功！响应: {content}")
            return True
        else:
            print(f"   ❌ 失败")
            try:
                error_info = response.json()
                print(f"   错误信息: {json.dumps(error_info, indent=2, ensure_ascii=False)}")
            except:
                print(f"   响应内容: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return False


def main():
    """测试多个可能的 DeepSeek 模型名称"""
    print("🚀 开始测试 OpenRouter 上的 DeepSeek 模型")
    print(f"🔑 API Key: {'已配置' if OPENROUTER_API_KEY else '未配置'}")

    # 可能的模型名称列表
    model_candidates = [
        # "deepseek/deepseek-chat",
        "deepseek/deepseek-chat-v3.1:free",
        "z-ai/glm-4.5-air:free",
        # "deepseek/deepseek-chat-v3",
        # "deepseek/deepseek-chat-v3.1",
        # "deepseek/deepseek-v3",
        # "deepseek/deepseek-chat:free",
        # "deepseek-ai/deepseek-chat",
        # "deepseek-ai/deepseek-chat-v3.1",
    ]

    print("\n" + "="*60)
    print("开始逐个测试模型...")
    print("="*60)

    working_models = []

    for model in model_candidates:
        if test_model(model):
            working_models.append(model)

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

    if working_models:
        print("\n✅ 可用的模型:")
        for model in working_models:
            print(f"   - {model}")
        print(f"\n💡 建议使用: {working_models[0]}")
    else:
        print("\n❌ 没有找到可用的模型")
        print("\n建议检查:")
        print("   1. API Key 是否正确")
        print("   2. OpenRouter 账户是否有余额")
        print("   3. 访问 https://openrouter.ai/models 查看可用模型列表")


if __name__ == "__main__":
    main()
