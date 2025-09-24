#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试每日新闻分析器是否正确配置
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试所有必需的导入"""
    try:
        import tushare as ts
        print("✅ tushare imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tushare: {e}")
        return False

    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pandas: {e}")
        return False

    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import requests: {e}")
        return False

    return True

def test_config():
    """测试配置文件"""
    try:
        # 添加配置目录到路径
        config_path = Path(__file__).parent.parent / "config"
        sys.path.append(str(config_path))

        from api_config import TUSHARE_TOKEN, OPENROUTER_API_KEY

        if TUSHARE_TOKEN == "your_tushare_token_here":
            print("⚠️  Warning: Please configure TUSHARE_TOKEN in config/api_config.py")
        else:
            print("✅ TUSHARE_TOKEN configured")

        if OPENROUTER_API_KEY == "your_openrouter_api_key_here":
            print("⚠️  Warning: Please configure OPENROUTER_API_KEY in config/api_config.py")
        else:
            print("✅ OPENROUTER_API_KEY configured")

        return True
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        return False

def test_stock_pool():
    """测试股票池文件"""
    try:
        stock_pool_path = Path(__file__).parent.parent / "config" / "stock_pool.json"

        if not stock_pool_path.exists():
            print(f"❌ Stock pool file not found: {stock_pool_path}")
            return False

        import json
        with open(stock_pool_path, 'r', encoding='utf-8') as f:
            stock_data = json.load(f)

        print(f"✅ Stock pool loaded with {len(stock_data.get('stocks', []))} stocks")
        return True

    except Exception as e:
        print(f"❌ Failed to load stock pool: {e}")
        return False

def test_output_dir():
    """测试输出目录创建"""
    try:
        output_dir = Path("D:/projects/q/myQ/output/daily_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory created: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to create output directory: {e}")
        return False

def main():
    """运行所有测试"""
    print("🧪 Testing Daily News Analyzer Configuration...")
    print("=" * 50)

    tests = [
        ("Import Dependencies", test_imports),
        ("Configuration", test_config),
        ("Stock Pool", test_stock_pool),
        ("Output Directory", test_output_dir),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the configuration.")

if __name__ == "__main__":
    main()