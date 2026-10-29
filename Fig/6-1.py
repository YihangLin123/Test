import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 新的数据准备（2017至2025年速动比率数据）
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
guannong_data = [0.673, 0.607, 0.677, 0.561, 0.482, 0.775, 0.810, 0.743, 0.905]
industry_avg = [1.672, 0.742, 1.155, 0.919, 1.233, 1.079, 1.113, 1.189, 1.678]

# 创建画布
fig, ax = plt.subplots(figsize=(10, 8))

# 隐藏坐标轴自带的四周边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 绘制折线图，带数据点
ax.plot(years, guannong_data, color='blue', linestyle='-', marker='o', label='冠农股份数据')
ax.plot(years, industry_avg, color='green', linestyle='--', marker='o', label='行业平均值')

# 设置网格与不显示Y轴标签
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xticks(years)
ax.set_xticklabels([''] * len(years))  # 隐藏原X轴刻度文字，改用下方表格对应

# 构造下方嵌入的数据表格内容
cell_text = [
    [f"{val:.3f}" for val in guannong_data],
    [f"{val:.3f}" for val in industry_avg]
]
row_labels = ['冠农股份数据', '行业平均值']

# 在图表下方添加表格
table = ax.table(cellText=cell_text,
                 rowLabels=row_labels,
                 colLabels=[str(y) for y in years],
                 loc='bottom',
                 cellLoc='center',
                 bbox=[0.0, -0.25, 1.0, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)

# 图例放在最下方
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)

# 调整布局以完整容纳表格和图例
plt.subplots_adjust(bottom=0.35)

# 在整张图表外围绘制一个黑色矩形边框
rect = patches.Rectangle((0.01, 0.13), 0.92, 0.8, transform=fig.transFigure,
                         color="black", fill=False, linewidth=1.5)
fig.patches.append(rect)

# 保存为 6-7.png
plt.savefig('6-8.png', dpi=600, bbox_inches='tight')
plt.show()