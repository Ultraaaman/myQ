#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试matplotlib颜色名称是否有效
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def test_colors():
    """测试颜色名称是否有效"""

    # 测试的颜色组合
    color_combinations = [
        ('blue', 'navy'),
        ('green', 'forestgreen'),
        ('orange', 'orangered'),
        ('purple', 'indigo'),
        ('red', 'crimson'),
        ('brown', 'saddlebrown')
    ]

    print("测试颜色组合:")
    for face_color, edge_color in color_combinations:
        try:
            # 尝试转换颜色
            face_rgba = mcolors.to_rgba(face_color)
            edge_rgba = mcolors.to_rgba(edge_color)
            print(f"✓ {face_color} + {edge_color}: 有效")
        except ValueError as e:
            print(f"✗ {face_color} + {edge_color}: 无效 - {e}")

    # 创建一个简单的测试图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.random.randn(50)
    y = np.random.randn(50)

    for i, (face_color, edge_color) in enumerate(color_combinations):
        try:
            offset_x = x + i * 2
            offset_y = y + i * 2
            ax.scatter(offset_x, offset_y, c=face_color, edgecolors=edge_color,
                      alpha=0.7, s=60, linewidth=0.5, label=f'{face_color}/{edge_color}')
        except Exception as e:
            print(f"绘图失败 {face_color}/{edge_color}: {e}")

    ax.set_title('Color Test')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 保存测试图片
    plt.savefig('color_test.png', dpi=100, bbox_inches='tight')
    print("\n测试图已保存为 color_test.png")
    plt.close()

if __name__ == "__main__":
    test_colors()