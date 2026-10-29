import pandas as pd

# 读取Excel文件（仅读取表头，不加载全部数据，速度更快）
df = pd.read_excel("cleaned_merged_data_final_processed.xlsx", nrows=0)

# 打印全部表头
print("全部表头（共 {} 列）：".format(len(df.columns)))
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")