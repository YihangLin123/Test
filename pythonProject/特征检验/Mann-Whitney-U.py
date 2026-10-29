import numpy as np
import pandas as pd
from scipy import stats

# 1. 读取双表头 Excel 文件（header=None，手动处理双表头）
file_path = '../data/C33_C40.xlsx'
df_raw = pd.read_excel(file_path, header=None)

# 第一行为英文名称，第二行为中文名称，第三行开始为数据
header_en = list(df_raw.iloc[0].values)
header_cn = list(df_raw.iloc[1].values)
df = df_raw.iloc[2:].copy()
df.columns = header_en

# 2. 检查 Y 列是否存在
if 'Y' not in df.columns:
  raise ValueError("表格中未找到名为 'Y' 的分组列，请检查列名。")

# 将 Y 列转换为数值型并剔除缺失值
df['Y'] = pd.to_numeric(df['Y'], errors='coerce')

# 获取 Y 的两个分组类别（通常为 0 和 1）
y_unique = df['Y'].dropna().unique()
if len(y_unique) != 2:
  print(
      f'提示：Y 列的唯一值数量为 {len(y_unique)}，Mann-Whitney U'
      ' 检验通常适用于两组对比。'
  )

group1_val, group2_val = y_unique[0], y_unique[1]

# 3. 获取 E列到 AI列 的指标列 (索引 4 到 33 对应 E 到 AI)
target_columns = df.columns[4:31]

results = []

# 4. 遍历每一列进行 Mann-Whitney U 检验
for col in target_columns:
  # 提取当前指标与 Y 列，转换为数值型，并剔除含缺失值的行
  subset = df[[col, 'Y']].copy()
  subset[col] = pd.to_numeric(subset[col], errors='coerce')
  subset = subset.dropna()

  if len(subset) > 0:
    g1_data = subset[subset['Y'] == group1_val][col]
    g2_data = subset[subset['Y'] == group2_val][col]

    if len(g1_data) > 0 and len(g2_data) > 0:
      # 执行双侧 Mann-Whitney U 检验
      stat, p_value = stats.mannwhitneyu(
          g1_data, g2_data, alternative='two-sided'
      )

      # 记录结果（包含中位数和均值，修正原代码中组2均值误写为中位数的笔误）
      results.append({
          '指标名称': col,
          f'组别 {group1_val} 样本量': len(g1_data),
          f'组别 {group1_val} 中位数': g1_data.median(),
          f'组别 {group1_val} 均值': g1_data.mean(),
          f'组别 {group2_val} 样本量': len(g2_data),
          f'组别 {group2_val} 均值': g2_data.mean(),
          f'组别 {group2_val} 中位数': g2_data.median(),
          'Mann-Whitney U 统计量': stat,
          'P值': p_value,
          '显著性 (α=0.05)': (
              '显著 (*)' if p_value < 0.05 else '不显著'
          ),
      })

# 5. 转换为 DataFrame 并保存结果
results_df = pd.DataFrame(results)
output_path = 'Mann_whitney_U_C13_C14.xlsx'
results_df.to_excel(output_path, index=False)

print(f'Mann-Whitney U 检验完成！结果已保存至: {output_path}')
print(results_df.head(10).to_string(index=False))