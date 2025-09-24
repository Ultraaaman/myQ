#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最原始的测试 - 直接测试Tushare API调用
"""

import sys
from pathlib import Path

# 添加配置目录
sys.path.append(str(Path(__file__).parent.parent / "config"))

print("🧪 开始原始API测试...")

# 1. 测试导入
print("1️⃣ 测试导入...")
import tushare as ts
print(f"✅ tushare版本: {ts.__version__}")

from api_config import TUSHARE_TOKEN
print(f"✅ Token配置: {'已配置' if TUSHARE_TOKEN != 'your_tushare_token_here' else '未配置'}")

# 2. 测试API初始化
print("\n2️⃣ 测试API初始化...")
pro_api = ts.pro_api(TUSHARE_TOKEN)
print(f"✅ API对象类型: {type(pro_api)}")

# 3. 测试API调用 - 你的格式
print("\n3️⃣ 测试API调用 - 你的格式...")
print("调用: pro_api.news(src='sina', start_date='2024-12-20 09:00:00', end_date='2024-12-20 18:00:00')")

df = pro_api.news(
    src='sina',
    start_date='2024-12-20 09:00:00',
    end_date='2024-12-20 18:00:00'
)

print(f"✅ 返回数据类型: {type(df)}")
print(f"✅ 数据形状: {df.shape}")
print(f"✅ 列名: {list(df.columns)}")
print(f"✅ 是否为空: {df.empty}")

if not df.empty:
    print("\n📊 前3行数据:")
    print(df.head(3))
    print("\n📋 数据信息:")
    print(df.info())
else:
    print("⚠️ 数据为空")

print("\n🎉 原始API测试完成！")