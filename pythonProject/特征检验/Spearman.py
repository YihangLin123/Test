import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取双表头 Excel 文件（header=None，手动处理双表头）
file_path = '../data/C33_C40_mann.xlsx'  # 请根据您的实际文件名修改
df_raw = pd.read_excel(file_path, header=None)

# 第一行为英文名称，第二行为中文名称，第三行开始为数据
header_en = list(df_raw.iloc[0].values)
header_cn = list(df_raw.iloc[1].values) # 获取中文注释，方便在表格中对照
df = df_raw.iloc[2:].copy()
df.columns = header_en

# 2. 定义需要绘制和分析的指标列表
target_vars = [
    'X2', 'X3', 'X4', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13',
    'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22',
    'X23', 'X24', 'X25', 'X26', 'X27'
]

# 模糊匹配列名函数
def find_col(name, columns):
    for col in columns:
        col_str = str(col).strip()
        if col_str.lower() == name.lower() or col_str.lower() == name.replace('X0', 'X').lower():
            return col
    return None

selected_cols = []
actual_names = []
for var in target_vars:
    matched = find_col(var, df.columns)
    if matched is not None:
        selected_cols.append(matched)
        actual_names.append(var)

# 提取子集并转换为数值型
subset_df = df[selected_cols].apply(pd.to_numeric, errors='coerce')

# 3. 计算 Spearman 相关系数矩阵
corr_matrix = subset_df.corr(method='spearman')
corr_matrix.index = actual_names
corr_matrix.columns = actual_names


# ==================== 功能一：导出完整的 Spearman 相关系数数值表格 ====================
corr_output_path = 'Spearman_Correlation_Matrix.xlsx'
corr_matrix.to_excel(corr_output_path)
print(f"[1] 完整的 Spearman 相关系数矩阵已保存至: {corr_output_path}")


# ==================== 功能二：提取相关系数高的指标对（阈值 >= 0.85） ====================
threshold = 0.85  # 阈值已设置为 0.85

# 将矩阵转化为长格式，方便筛选和查看
corr_unstacked = corr_matrix.unstack().reset_index()
corr_unstacked.columns = ['Indicator_1', 'Indicator_2', 'Spearman_Corr']

# 剔除自身与自身对比以及重复的对称对
high_corr_df = corr_unstacked[
    (corr_unstacked['Indicator_1'] != corr_unstacked['Indicator_2']) &
    (corr_unstacked['Spearman_Corr'].abs() >= threshold)
].copy()

# 去除对称重复项
high_corr_df['Pair_Key'] = high_corr_df.apply(lambda row: tuple(sorted([row['Indicator_1'], row['Indicator_2']])), axis=1)
high_corr_df = high_corr_df.drop_duplicates(subset=['Pair_Key']).drop(columns=['Pair_Key'])

# 按相关系数绝对值从大到小排序
high_corr_df['Abs_Corr'] = high_corr_df['Spearman_Corr'].abs()
high_corr_df = high_corr_df.sort_values(by='Abs_Corr', ascending=False).drop(columns=['Abs_Corr'])
high_corr_df = high_corr_df.reset_index(drop=True)

# 匹配对应的中文注释
col_cn_map = {}
for var, col in zip(actual_names, selected_cols):
    col_idx = df_raw.columns.get_loc(col) if col in df_raw.columns else None
    if col_idx is not None and len(header_cn) > col_idx:
        col_cn_map[var] = header_cn[col_idx]
    else:
        col_cn_map[var] = ''

high_corr_df['Name_1'] = high_corr_df['Indicator_1'].map(col_cn_map)
high_corr_df['Name_2'] = high_corr_df['Indicator_2'].map(col_cn_map)

# 调整列顺序
high_corr_df = high_corr_df[['Indicator_1', 'Name_1', 'Indicator_2', 'Name_2', 'Spearman_Corr']]

# 保存高相关性指标表格
high_corr_output_path = 'High_Correlation_Pairs_0.85.xlsx'
high_corr_df.to_excel(high_corr_output_path, index=False)
print(f"[2] 相关系数绝对值 >= 0.85 的指标对表格已保存至: {high_corr_output_path}")

# 控制台预览
print("\n--- 高相关性指标对预览 (>= 0.85) ---")
if len(high_corr_df) > 0:
    print(high_corr_df.to_string(index=False))
else:
    print("未发现相关系数绝对值 >= 0.85 的指标对。")


# ==================== 功能三：绘制并保存高质量学术风格热力图 ====================
plt.figure(figsize=(10, 8), dpi=300)

# 设置全局字体与负号显示
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 绘制热力图
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    annot_kws={'size': 9},
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

# 保存高分辨率图片
output_img_path = '5-2.png'
plt.savefig(output_img_path, dpi=600, bbox_inches='tight')
plt.show()

print(f"\n[3] 相关性热力图已成功生成并保存至: {output_img_path}")