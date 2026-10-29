import pandas as pd

# 1. 定义文件路径
file_input = './data/C33_C40_mann.xlsx'  # 输入的表格文件
output_file = './data/C33_C40_normalized.xlsx'  # 标准化后的输出新文件名

# 2. 读取表格（header=None，手动处理双表头）
df_raw = pd.read_excel(file_input, header=None)

# 第一行为英文名称，第二行为中文注释，第三行开始为数据
header_en = list(df_raw.iloc[0].values)
header_cn = list(df_raw.iloc[1].values)
df_data = df_raw.iloc[2:].copy()
df_data.columns = header_en

# 3. 定位 E 列至 X 列的特征指标范围
# 在 Python 索引中：E 列是第 5 列（索引 4），X 列是第 24 列（索引 23）
# 切片范围使用 4:24（左闭右开，即包含索引 4 到 23，对应 E 至 X）
# 如果您的 X 列恰好是第 25 列，可将切片调整为 4:25
feature_cols = df_data.columns[4:24]

print("即将进行 Min-Max 标准化的特征指标列（E至X列）：")
print(feature_cols.tolist())

# 4. 执行 Min-Max 标准化公式：X_norm = (X - X_min) / (X_max - X_min)
for col in feature_cols:
    # 强制转换为数值型，非数值转为 NaN
    df_data[col] = pd.to_numeric(df_data[col], errors='coerce')

    x_min = df_data[col].min()
    x_max = df_data[col].max()

    # 防止分母为 0（即该列所有值都相等的情况）
    if x_max != x_min:
        df_data[col] = (df_data[col] - x_min) / (x_max - x_min)
    else:
        # 如果最大值等于最小值，则全部归一化为 0 或 1（通常赋予 0）
        df_data[col] = 0.0

# 5. 写入并输出到新文件（完美保留原有的双行表头结构）
with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
    # 写入第1行：英文列名
    pd.DataFrame([header_en], columns=header_en).to_excel(writer, sheet_name='Sheet1', header=False, index=False,
                                                          startrow=0)
    # 写入第2行：中文注释
    pd.DataFrame([header_cn], columns=header_en).to_excel(writer, sheet_name='Sheet1', header=False, index=False,
                                                          startrow=1)
    # 写入从第3行开始的正文数据
    df_data.to_excel(writer, sheet_name='Sheet1', header=False, index=False, startrow=2)

print(f"\n处理完成！E至X列的特征指标已成功完成 Min-Max 标准化（映射至 [0, 1] 区间）。")
print(f"新文件已成功保存至: '{output_file}'，原文件未被覆盖。")