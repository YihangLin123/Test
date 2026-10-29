import pandas as pd

# 1. 读取主表（此时的主表已包含之前处理好的所有特征及 Y1 标签）
df_main = pd.read_excel('cleaned_merged_data.xlsx')

# 读取托宾Q值补充表
df_tobin = pd.read_excel('托宾Q值.xlsx')

# 2. 自动清理两张表所有列名的前后空格
df_main.columns = df_main.columns.str.strip()
df_tobin.columns = df_tobin.columns.str.strip()

print("主表列名预览:", df_main.columns.tolist()[:5])
print("托宾Q表列名确认:", df_tobin.columns.tolist())

# 3. 定义统一的股票代码清洗函数（确保 Symbol 格式绝对一致）
def clean_symbol(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s

df_main['Symbol'] = df_main['Symbol'].apply(clean_symbol)
df_tobin['Symbol'] = df_tobin['Symbol'].apply(clean_symbol)

# 4. 处理托宾Q表的时间：从 EndDate 提取出年份 Year
date_col = 'EndDate'
tobin_val_col = 'F100901A' # 对应你表中的真实托宾Q值列名

df_tobin[date_col] = pd.to_datetime(df_tobin[date_col], errors='coerce')
df_tobin['Year'] = df_tobin[date_col].dt.year

# 5. 筛选年末（12月31日）数据
df_tobin_yearly = df_tobin[df_tobin[date_col].dt.month == 12].copy()

# 【关键调试】打印两张表的交集情况，看看有没有对上
main_set = set(zip(df_main['Symbol'], df_main['Year']))
tobin_set = set(zip(df_tobin_yearly['Symbol'], df_tobin_yearly['Year']))
print(f"主表 (Symbol, Year) 组合数: {len(main_set)}")
print(f"托宾Q表 (Symbol, Year) 组合数: {len(tobin_set)}")
print(f"两张表成功重合的 (Symbol, Year) 数量: {len(main_set.intersection(tobin_set))}")

# 6. 进行 merge
tobin_subset = df_tobin_yearly[['Symbol', 'Year', tobin_val_col]].copy()
df_merged = pd.merge(
    df_main,
    tobin_subset,
    on=['Symbol', 'Year'],
    how='left'
)

# 7. 检查并重命名托宾Q列，转为规范的 Tobin_Q 特征
if tobin_val_col in df_merged.columns:
    df_merged['Tobin_Q'] = df_merged[tobin_val_col]
    missing_count = df_merged['Tobin_Q'].isnull().sum()
    print(f"合并成功！托宾Q值 (F100901A) 缺失值数量: {missing_count}")
    # 如果原列名和新名不同，可选择把原列 drop 掉
    if tobin_val_col != 'Tobin_Q':
        df_merged.drop(columns=[tobin_val_col], inplace=True, errors='ignore')
else:
    print("警告：合并后未检测到 F100901A 列！")
    df_merged['Tobin_Q'] = None

# 8. 确保灵魂标签 Y1 稳稳呆在表格的最后一列
if 'Y1' in df_merged.columns:
    feature_cols = [col for col in df_merged.columns if col != 'Y1']
    df_merged = df_merged[feature_cols + ['Y1']]

# 9. 保存最终更新的大宽表
df_merged.to_excel('cleaned_merged_data_final.xlsx', index=False)
print("处理完毕！托宾Q值已成功植入，最终大宽表已更新并保存为 cleaned_merged_data_final.xlsx")