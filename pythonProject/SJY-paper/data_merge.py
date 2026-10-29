import os
import pandas as pd

data_dir = "raw_data"

print("=" * 60)
print("第一步：正在加载并预处理各分表...")
print("=" * 60)

# 1. 读取基础信息表作为主干表
base_path = os.path.join(data_dir, "基础信息.xlsx")
df_base = pd.read_excel(base_path, header=0)

# 清理基础信息表可能存在的CSMAR说明行
if len(df_base) > 0 and ('代码' in str(df_base.iloc[0].values) or '没有单位' in str(df_base.iloc[0].values)):
    df_base = df_base.drop(index=[0, 1]).reset_index(drop=True)

# 统一基础信息表的关键列名并提取年份
# 基础信息通常是每家公司每年一条记录，或者包含EndDate
code_col_base = 'Symbol'
date_col_base = 'EndDate'

df_base['Clean_Symbol'] = df_base[code_col_base].astype(str).str.strip().str.zfill(6)
df_base['Year'] = pd.to_datetime(df_base[date_col_base], errors='coerce').dt.year

# 确保主表是 “一个公司、一个年份唯一对应一行”
df_base = df_base.dropna(subset=['Year']).drop_duplicates(subset=['Clean_Symbol', 'Year']).reset_index(drop=True)
merged_df = df_base.copy()
print(f"主干表《基础信息.xlsx》加载完成，有效样本行数: {len(merged_df)}")

# 定义需要合并的常规表格（排除证监会处罚和基础信息）
# 财务表需要特殊处理，只留12-31
financial_files = ['偿债能力.xlsx', '盈利能力.xlsx', '成长能力.xlsx', '营运能力.xlsx', '现金流能力.xlsx']
other_files = ['公司治理.xlsx', '内控报告.xlsx', '审计.xlsx', '股权.xlsx', '高管.xlsx']

all_files_to_merge = financial_files + other_files

for file in all_files_to_merge:
    file_path = os.path.join(data_dir, file)
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file}，跳过。")
        continue

    df_curr = pd.read_excel(file_path, header=0)

    # 清理CSMAR垃圾行
    if len(df_curr) > 0:
        drop_indices = []
        for idx in range(min(3, len(df_curr))):
            row_str = str(df_curr.iloc[idx].values)
            if '没有单位' in row_str or '股票代码' in row_str or '统计截止日期' in row_str:
                drop_indices.append(idx)
        if drop_indices:
            df_curr = df_curr.drop(index=drop_indices).reset_index(drop=True)

    # 智能寻找当前表的 代码列、日期/年度列
    curr_code_col, curr_date_col = None, None
    for col in df_curr.columns:
        col_lower = str(col).lower()
        if col_lower in ['stkcd', 'symbol']:
            curr_code_col = col
        if col_lower in ['accper', 'enddate', 'reptdt', 'enddate']:
            curr_date_col = col

    if not curr_code_col:
        print(f"错误: 无法在 {file} 中识别出股票代码列，跳过。")
        continue

    # 统一代码格式
    df_curr['Clean_Symbol'] = df_curr[curr_code_col].astype(str).str.strip().str.zfill(6)

    # 处理日期与年度
    if curr_date_col:
        df_curr['Parsed_Date'] = pd.to_datetime(df_curr[curr_date_col], errors='coerce')
        df_curr['Year'] = df_curr['Parsed_Date'].dt.year

        # 核心要求：财务报表只保留 12月31日 的数据
        if file in financial_files:
            # 筛选月份 == 12 且 日 == 31 的行
            before_len = len(df_curr)
            df_curr = df_curr[(df_curr['Parsed_Date'].dt.month == 12) & (df_curr['Parsed_Date'].dt.day == 31)]
            print(f"【财务表过滤】《{file}》仅保留12-31日数据，行数从 {before_len} 过滤至 {len(df_curr)}")
    else:
        # 如果表里没有明确的日期列（例如某些治理或高管表可能只有年度字段）
        for col in df_curr.columns:
            if 'year' in str(col).lower() or '年度' in str(col):
                df_curr['Year'] = pd.to_numeric(df_curr[col], errors='coerce')
                break
        if 'Year' not in df_curr.columns:
            print(f"警告: 表格 {file} 中无法识别有效年份，可能导致匹配错误。")

    # 确保副表也是 “一个公司、一个年份唯一对应一行”，防止merge时行数暴增或出现重复
    if 'Year' in df_curr.columns:
        df_curr = df_curr.dropna(subset=['Year']).drop_duplicates(subset=['Clean_Symbol', 'Year']).reset_index(
            drop=True)

    # 执行左连接合并
    print(f"正在合并: {file} ...")
    merged_df = pd.merge(
        merged_df,
        df_curr,
        how='left',
        on=['Clean_Symbol', 'Year'],
        suffixes=('', '_dup')
    )

print("\n" + "=" * 60)
print("第二步：清理与后处理...")
print("=" * 60)

# 清理合并产生的重复列（带_dup后缀的）
dup_cols = [c for c in merged_df.columns if c.endswith('_dup')]
if dup_cols:
    merged_df.drop(columns=dup_cols, inplace=True)

# 最终检查粒度：必须保证 [Clean_Symbol, Year] 组合是唯一的
before_dedup = len(merged_df)
merged_df = merged_df.drop_duplicates(subset=['Clean_Symbol', 'Year']).reset_index(drop=True)
print(f"最终大宽表总行数: {len(merged_df)} (重复清理行数: {before_dedup - len(merged_df)})")
print(f"最终大宽表总列数: {len(merged_df.columns)}")

# 保存结果
output_path = "merged_data.xlsx"
merged_df.to_excel(output_path, index=False)
print("=" * 60)
print(f"【成功】合并完毕！已严格确保一个公司一个年份为一行，财务数据仅保留12-31。")
print(f"文件已保存至: {output_path}")
print("=" * 60)