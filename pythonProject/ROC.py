# -*- coding: utf-8 -*-
"""
多模型 ROC 曲线汇总绘制脚本
功能：读取5个模型保存的真实标签与预测概率，在同一张 (8, 6) 画布中绘制对比 ROC 曲线
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置模型文件路径及显示名称（已剔除决策树）
# ============================================================
model_files = {
    '随机森林': 'rf.xlsx',
    'BP神经网络': 'bp.xlsx',
    '逻辑回归': 'lr.xlsx',
    '支持向量机': 'svm.xlsx',
    'XGBoost': 'xgb.xlsx'
}

# 自定义 5 个模型的线条颜色（对应5个模型）
colors = ['red', 'orange', 'green', 'blue', 'purple']

# ============================================================
# 创建画布并循环绘制各模型的 ROC 曲线
# ============================================================
plt.figure(figsize=(8, 6))

for (name, filepath), color in zip(model_files.items(), colors):
    try:
        # 读取保存的预测数据
        df_res = pd.read_excel(filepath)
        y_true = df_res['y_true'].values
        y_pred_proba = df_res['y_pred_proba'].values

        # 计算 AUC 值
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        # 计算 FPR 和 TPR
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

        # 绘制曲线
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    except FileNotFoundError:
        print(f"⚠️ 警告: 未找到文件 '{filepath}'，请确保该模型代码已成功运行并保存！")

# 绘制随机猜测参考线
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='随机猜测 (AUC = 0.5000)')

# 图表排版与美化设置（与原代码风格一致）
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存汇总大图
output_path = "5-7.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"\n✅ 五模型对比 ROC 曲线图已成功保存至: '{output_path}'")

# 显示图表
plt.show()