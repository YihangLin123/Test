import pandas as pd

# 读取Excel文件
file_path = './data/final_data.xlsx'
df = pd.read_excel(file_path)

# 定义列名（请根据实际表格列名核对调整）
year_col = 'year'          # 年份列名
symbol_col = '证券代码'     # 股票代码列名
industry_col = '行业代码'  # 行业代码列名（请检查Excel中实际列名，比如 '行业代码', 'IndCode' 等）

# 过滤出2017-2025年的数据
df = df[(df[year_col] >= 2017) & (df[year_col] <= 2025)]

# 1. 提取案例公司 (600251) 的各年度指标数据
target_data = df[df[symbol_col].astype(str).str.contains('600251')].sort_values(by=year_col)

# 2. 筛选行业代码为 'C13' 的数据来计算行业平均值
df_industry = df[df[industry_col].astype(str).str.upper() == 'C13']

# 排除不需要参与求平均值的非数值列或代码列
numeric_cols = df_industry.select_dtypes(include=['number']).columns.tolist()
if year_col in numeric_cols:
    numeric_cols.remove(year_col)

industry_avg = (
    df_industry.groupby(year_col)[numeric_cols]
    .mean()
    .reset_index()
    .sort_values(by=year_col)
)

# 3. 打印对比结果
print("========== 案例公司 (600251) 2017-2025 各年度指标 ==========")
print(target_data.to_string(index=False))

print("\n========== 行业 (C13) 平均值 2017-2025 各年度指标 ==========")
print(industry_avg.to_string(index=False))

# 4. 分别输出为两个独立的Excel文件
target_file = 'target_company_600251.xlsx'
industry_file = 'industry_average_C13.xlsx'

target_data.to_excel(target_file, index=False)
industry_avg.to_excel(industry_file, index=False)

print(f"\n文件已成功导出：\n1. {target_file}\n2. {industry_file}")