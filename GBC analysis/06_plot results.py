# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb

# Step 1: 读取 GBC 结果
df = pd.read_csv("GBC_results_step1.tsv", sep="\t", index_col=0)

# Step 2: 剔除低于阈值的比例并归一化
threshold = 0.08
df_filtered = df.apply(lambda col: col.where(col >= threshold, 0))
df_filtered = df_filtered.div(df_filtered.sum(axis=0), axis=1)

# Step 3: 转置为 Sample × Breed
df_plot = df_filtered.T  # 每行是一个样本，每列是一个品种

# Step 4: 统一排序所有品种（按总比例降序）
total_contributions = df_plot.sum(axis=0)
sorted_breeds = total_contributions[total_contributions > 0].sort_values(ascending=False).index.tolist()
df_plot = df_plot[sorted_breeds]

# Step 5: 生成明显不同且不重复的颜色
def generate_distinct_colors(n):
    # 使用 tab20 + tab20b + tab20c 共 60 个明显区分色
    base_cmaps = ['tab20', 'tab20b', 'tab20c']
    colors = []
    for cmap_name in base_cmaps:
        cmap = cm.get_cmap(cmap_name)
        for i in range(cmap.N):
            colors.append(to_rgb(cmap(i)))
    if n > len(colors):
        raise ValueError(f"当前最多可生成 {len(colors)} 个明显不同的颜色，但你需要 {n} 个。")
    return colors[:n]

num_breeds = len(sorted_breeds)
colors = generate_distinct_colors(num_breeds)

# Step 6: 绘图（堆叠柱状图）
plt.figure(figsize=(max(10, len(df_plot) * 0.4), 6))
bottom = pd.Series([0] * len(df_plot), index=df_plot.index)

for i, breed in enumerate(df_plot.columns):
    values = df_plot[breed]
    plt.bar(df_plot.index, values, bottom=bottom, label=breed, color=colors[i])
    bottom += values

# Step 7: 样式和图例
plt.xticks(rotation=90)
plt.ylabel("Proportion")
plt.xlabel("Sample")
plt.title("GBC Composition Across Samples")

if num_breeds > 20:
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        title='Breed',
        fontsize=7,
        ncol=max(1, num_breeds // 20)
    )
else:
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        title='Breed',
        fontsize=9
    )

plt.tight_layout()

# Step 8: 保存
plt.savefig("GBC_stacked_barplot_step2.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ 堆叠柱状图保存为 GBC_stacked_barplot_step2.png")
print(f"📊 共处理了 {num_breeds} 个品种，成功为每个品种分配了不重复且明显区分的颜色")
