import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义各模型的混淆矩阵数据格式：[[TN, FP], [FN, TP]]
# 按照您指定的顺序排列
models_data = [
    ("支持向量机", np.array([[524, 42], [31, 64]])),
    ("逻辑回归", np.array([[465, 101], [17, 78]])),
    ("随机森林", np.array([[502, 64], [14, 81]])),
    ("XGBoost", np.array([[531, 35], [26, 69]])),
    ("BP神经网络", np.array([[501, 65], [15, 80]]))
]

# 创建 2行3列 的子图画布（更适合横向展示）
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# 标签定义
xtick_labels = ['预测: 0 (无缺陷)', '预测: 1 (有缺陷)']
ytick_labels = ['真实: 0 (无缺陷)', '真实: 1 (有缺陷)']

# 循环绘制每个模型的混淆矩阵热力图
for i, (title, cm) in enumerate(models_data):
    ax = axes[i]

    # 使用 seaborn 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=xtick_labels,
                yticklabels=ytick_labels,
                ax=ax, annot_kws={"size": 13, "weight": "bold"},
                linewidths=1, linecolor='white')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('预测类别', fontsize=11)
    ax.set_ylabel('真实类别', fontsize=11)

# 将 2x3 网格中的第6个多余子图关闭隐藏
axes[-1].axis('off')

# 调整布局并高分辨率保存图片
plt.tight_layout()
plt.savefig('5-6.png', dpi=600, bbox_inches='tight')
print("✅ 2x3 布局的5个模型混淆矩阵综合对比图已成功保存至: '5-6.png'")
plt.show()