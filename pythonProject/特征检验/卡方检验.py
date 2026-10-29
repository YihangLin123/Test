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
df = df.dropna(subset=['Y'])
df['Y'] = df['Y'].astype(int)

# 3. 指定需要进行卡方检验的二分类/离散特征列
# 对应文中的：两职合一（X1/X01）、股权性质（X2/X02）、是否变更审计机构（X25）、是否“四大”审计（X27）
potential_cols = ['X1', 'X01', 'X2', 'X02', 'X25', 'X27']
target_columns = [col for col in potential_cols if col in df.columns]

# 如果未完全匹配到预设列名，尝试通过关键字自动筛选
if not target_columns:
  print(
      '提示：未直接匹配到预设的特征列名，系统将尝试自动筛选包含目标编号的列。'
  )
  target_columns = [
      c
      for c in df.columns
      if any(k in str(c).upper() for k in ['X1', 'X2', 'X25', 'X27'])
  ]

print(f'成功识别并选定的卡方检验特征列: {target_columns}')

results = []

# 4. 遍历指定的离散特征列进行卡方检验
for col in target_columns:
  # 提取当前离散指标与 Y 列，转换为数值型，并剔除含缺失值的行
  subset = df[[col, 'Y']].copy()
  subset[col] = pd.to_numeric(subset[col], errors='coerce')
  subset = subset.dropna()

  if len(subset) > 0:
    # 构建交叉联表 (Contingency Table)
    contingency_table = pd.crosstab(subset[col], subset['Y'])

    # 执行卡方独立性检验
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(
        contingency_table
    )

    # 记录结果
    results.append({
        '指标名称': col,
        '卡方统计量 (Chi2)': chi2_stat,
        '自由度 (df)': dof,
        'P值': p_value,
        '显著性 (α=0.05)': ('显著 (*)' if p_value < 0.05 else '不显著'),
    })

# 5. 转换为 DataFrame 并保存结果
results_df = pd.DataFrame(results)
output_path = 'Chi_Square_Test_results.xlsx'
results_df.to_excel(output_path, index=False)

print(f'\n卡方检验完成！结果已保存至: {output_path}')
if not results_df.empty:
  print(results_df.to_string(index=False))
else:
  print(
      '警告：未找到任何匹配的特征列进行卡方检验，请检查数据表中的列名是否正确。'
  )