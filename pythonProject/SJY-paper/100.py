import pandas as pd

# 1. 读取原始大盘数据
file_path = './data/final1.xlsx'
df = pd.read_excel(file_path)

print(f"原始总样本数: {df.shape[0]}, 总列数: {df.shape[1]}")

# 统一 Symbol 格式
df['Symbol'] = df['Symbol'].astype(str).str.strip()

# 2. 定义目标年份范围
target_years = list(range(2017, 2026))  # 包含 2017 到 2025
print(f"目标连续考核年份: {target_years}")

# 3. 筛选出在 2017-2025 年间存在且 Y2 恒为 0 的公司
valid_symbols = []

# 按公司 Symbol 进行分组审查
for symbol, group in df.groupby('Symbol'):
    # 提取该公司的年份集合
    company_years = set(group['Year'].astype(int))

    # 检查是否完整包含了 2017-2025 所有年份
    if set(target_years).issubset(company_years):
        # 进一步检查：在 2017-2025 这几年间，其 Y2 是否全部等于 0
        sub_period = group[group['Year'].isin(target_years)]
        if (sub_period['Y2'] == 0).all():
            valid_symbols.append(symbol)

print(f"\n✅ 筛选完成！")
print(f"• 满足 2017-2025 连续存在且 Y2 恒为 0 的公司总数: {len(valid_symbols)} 家")

# 4. 提取这些合规公司的所有年度数据（或者只提取 2017-2025 期间的数据）
# 这里提取这些公司在 2017-2025 期间的数据
df_filtered_clean = df[
    (df['Symbol'].isin(valid_symbols)) &
    (df['Year'].isin(target_years))
    ].copy()

# 按 Symbol 和 Year 升序排序
df_filtered_clean = df_filtered_clean.sort_values(['Symbol', 'Year']).reset_index(drop=True)

# 5. 导出到指定的文件夹/文件路径
output_path = './data/balanced_control_companies_2017_2025.xlsx'
df_filtered_clean.to_excel(output_path, index=False)

print(f"• 最终导出的面板数据总行数: {df_filtered_clean.shape[0]} 行")
print(f"• 纯净对照组表格已成功保存至: '{output_path}'")