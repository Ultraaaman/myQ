#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API配置文件
请在这里配置你的API密钥
"""

# Tushare API配置
TUSHARE_TOKEN = "3b22745fa720272a07322298890b0835e74147be8419d0389d0d1be8"  # 在 https://tushare.pro/ 获取

# OpenRouter API配置
OPENROUTER_API_KEY = "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"  # 在 https://openrouter.ai/ 获取

# 模型配置
DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1:free"

# 分析参数
FACTOR_STRENGTH_THRESHOLD = 3.0  # 强因子阈值
MAX_NEWS_CONTENT_LENGTH = 1000   # 新闻内容最大长度
API_DELAY_RANGE = (0.5, 1.0)     # API调用间隔（秒）

# 输出配置
OUTPUT_DIR = "D:/projects/q/myQ/output/daily_analysis"
ENABLE_DETAILED_LOG = True