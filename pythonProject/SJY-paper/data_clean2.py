import pandas as pd

# 1. 读取 Excel 文件
file_name = 'cleaned_merged_data_final.xlsx'
df = pd.read_excel(file_name)

# 2. 去除任何一列中包含空值的行（实现每一列都为非空值）
df_cleaned = df.dropna()

# 3. 将清理后的数据保存到新文件
output_file = 'cleaned_merged_data_final_processed.xlsx'
df_cleaned.to_excel(output_file, index=False)

print(f"处理完成！已去除含有空值的行，新文件已保存为：{output_file}")