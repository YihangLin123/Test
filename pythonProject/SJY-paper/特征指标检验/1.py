import pandas as pd
import numpy as np

# 1. 读取清洗后的总数据集
file_path = "../data/cleaned_merged_data_final_processed_2.xlsx"
df = pd.read_excel(file_path)

print(f"数据读取成功！总样本数: {df.shape[0]}, 总列数: {df.shape[1]}")

# 2. 精准提取 G 列到 AG 列的 27 个候选指标特征（Python 索引 6 到 33，左闭右开）
feature_cols = df.columns[6:33].tolist()
X_features = df[feature_cols]

print(f"正在计算这 {len(feature_cols)} 个指标的 Spearman 秩相关系数矩阵...")

# 3. 计算 Spearman 秩相关系数矩阵
spearman_matrix = X_features.corr(method='spearman')

# 保留 3 位小数（学术论文标准格式）
spearman_matrix_rounded = spearman_matrix.round(3)

# 4. 导出为专用的 Excel 文件
output_path = "../data/spearman_correlation_table_thesis.xlsx"
spearman_matrix_rounded.to_excel(output_path)

print(f"\n✅ Spearman 相关系数矩阵已成功生成并导出至: '{output_path}'")
print("你可以直接打开该 Excel 文件，将其复制并粘贴到你的毕业论文中！")

# 5. 同时在控制台打印出部分预览（查看前 5 行前 5 列）
print("\n--- 矩阵预览（前 5 行、前 5 列） ---")
print(spearman_matrix_rounded.iloc[:5, :5])