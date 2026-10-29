import numpy as np
import pandas as pd
from scipy import stats

# 1. 读取双表头 Excel 文件（header=None，手动处理双表头）
file_path = '../data/C33_C40.xlsx'
df_raw = pd.read_excel(file_path, header=None)

# 第一行为英文名称，第二行为中文名称，第三行开始为数据
header_en = list(df_raw.iloc[0].values)
header_cn = list(df_raw.iloc[1].values)
df_data = df_raw.iloc[2:].copy()
df_data.columns = header_en

# 2. 获取 E列到 AI列 的指标列 (索引 4 到 33 对应 E 到 AG)
target_columns = df_data.columns[4:31]

results = []

# 3. 遍历每一列进行 K-S 正态性检验
for col in target_columns:
  # 提取该列数据，转换为数值型并去除缺失值
  data = pd.to_numeric(df_data[col], errors='coerce').dropna()

  if len(data) > 1:  # 标准差计算至少需要2个样本
    # K-S 正态性检验要求数据标准化（均值为0，标准差为1）
    standardized_data = (data - data.mean()) / data.std(ddof=1)

    # 进行单样本 K-S 检验，检验其是否符合标准正态分布 N(0, 1)
    ks_stat, p_value = stats.kstest(standardized_data, 'norm')

    # 设定显著性水平 alpha = 0.05
    is_normal = p_value > 0.05

    results.append({
        '指标名称': col,
        '样本量 (N)': len(data),
        'KS统计量': ks_stat,
        'P值': p_value,
        '是否服从正态分布 (α=0.05)': '是' if is_normal else '否',
    })

# 4. 将结果转换为 DataFrame 并保存
results_df = pd.DataFrame(results)
output_path = 'K-S_C33_C40.xlsx'
results_df.to_excel(output_path, index=False)

print(f'K-S 正态性检验完成！结果已保存至: {output_path}')

# 5. 打印前 10 行结果查看
print(results_df.head(10).to_string(index=False))