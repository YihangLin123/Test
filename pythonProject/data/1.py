import pandas as pd

print("正在读取表格数据...")
# 1. 读取两张 Excel 表格
df_deficiency = pd.read_excel('IC_DeficiencyInfo.xlsx')
df_industry = pd.read_excel('行业代码.xlsx')

print(f"IC_DeficiencyInfo 原始数据行数: {len(df_deficiency)}")
print(f"行业代码表原始数据行数: {len(df_industry)}")

# 2. 统一关键匹配字段的数据格式（去除前后空格，转为字符串或规范格式，防止因空格或浮点数导致匹配失败）
for col in ['Symbol', 'ShortName', 'EndDate']:
    if col in df_deficiency.columns:
        df_deficiency[col] = df_deficiency[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    if col in df_industry.columns:
        df_industry[col] = df_industry[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

# 确保两张表都包含所需的键字段
merge_keys = ['Symbol', 'ShortName', 'EndDate']
for key in merge_keys:
    if key not in df_deficiency.columns or key not in df_industry.columns:
        raise ValueError(f"错误：两张表中均必须包含 '{key}' 字段，请检查列名。")

# 3. 提取行业表中的核心列（假设行业代码列名为 'IndustryCode'，如果您的行业代码表里列名不同，请修改下方字典映射）
# 假设行业代码表里存有行业代码的列名叫 'IndustryCode' 或 'Industry'
# 如果行业列名是其他名字（比如 'IndCode'），请把下面代码中的 'IndustryCode' 改为实际列名
industry_code_col = 'IndustryCode'
if industry_code_col not in df_industry.columns:
    # 尝试寻找可能的列名
    possible_cols = [c for c in df_industry.columns if '行业' in str(c) or 'Code' in str(c)]
    if possible_cols:
        industry_code_col = possible_cols[0]
        print(f"⚠️ 自动识别到行业代码列为: '{industry_code_col}'")
    else:
        raise ValueError("未在《行业代码.xlsx》中找到行业代码相关列，请检查列名。")

# 为了安全合并，只保留用于匹配的键和行业列
df_industry_subset = df_industry[merge_keys + [industry_code_col]].drop_duplicates(subset=merge_keys)

# 4. 使用 pd.merge 进行多条件左连接（Left Join），以 IC_DeficiencyInfo 为主表
df_merged = pd.merge(
    df_deficiency,
    df_industry_subset,
    on=merge_keys,
    how='left'
)

# 如果合并后新列名不是 IndustryCode，将其重命名为 IndustryCode
if industry_code_col != 'IndustryCode':
    df_merged.rename(columns={industry_code_col: 'IndustryCode'}, inplace=True)

# 5. 检查匹配情况
unmatched_count = df_merged['IndustryCode'].isna().sum()
print(f"匹配完成！总行数: {len(df_merged)}")
print(f"成功匹配行数: {len(df_merged) - unmatched_count}")
print(f"未成功匹配（行业代码为空）行数: {unmatched_count}")

# 6. 保存结果到新文件
output_file = 'IC_DeficiencyInfo_with_Industry.xlsx'
df_merged.to_excel(output_file, index=False)
print(f"✅ 结果已成功保存至: '{output_file}'")