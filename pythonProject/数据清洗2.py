import pandas as pd

# 1. 读取 Excel 文件
file_path = './data/original_data_with_Y2.xlsx'
# 如果文件包含多个工作表，可以使用 sheet_name 指定，默认读取第一个工作表
df = pd.read_excel(file_path)

# 2. 筛选“行业代码”列中等于 'C13' 或 'C14' 的样本
# 使用 isin() 方法可以方便地匹配多个条件
filtered_df = df[df['行业代码'].isin(['C13', 'C14'])]

# 3. （可选）将筛选后的数据保存到新的 Excel 文件中
output_path = './data/original_data_cleaned_C13_C14.xlsx'
filtered_df.to_excel(output_path, index=False)

print(f"筛选完成！原始数据行数：{len(df)}，筛选后行数：{len(filtered_df)}")
print(f"已保存至: {output_path}")