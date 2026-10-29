import pandas as pd

# ==========================================
# 第一步：读取原始大表与精炼表格
# ==========================================
# 包含原始 Dadtunit 数据的总表
merged_data_path = './data/merged_data.xlsx'
df_merged = pd.read_excel(merged_data_path)

# 你当前已经剔除过样本的目标精炼表格路径
fine_data_path = './data/final.xlsx'
df_fine = pd.read_excel(fine_data_path)

print(f"原始大表 merged_data 行数: {df_merged.shape[0]}")
print(f"精炼表行数: {df_fine.shape[0]}")

# 统一转换 Symbol 格式防匹配失败
df_merged['Symbol'] = df_merged['Symbol'].astype(str).str.strip()
df_fine['Symbol'] = df_fine['Symbol'].astype(str).str.strip()

# ==========================================
# 第二步：在原始大表中根据 Dadtunit 重新计算 IsAuditorChanged
# ==========================================
# 1. 确保数据按企业和年份升序排列，以便正确进行跨年对比
df_merged = df_merged.sort_values(by=['Symbol', 'Year']).reset_index(drop=True)

# 2. 获取去年的审计单位名称 (使用 Dadtunit 列进行 shift)
df_merged['Prev_Dadtunit'] = df_merged.groupby('Symbol')['Dadtunit'].shift(1)


# 3. 重新判定：当年与去年相同为 1，不同为 0
def calculate_auditor_changed(row):
    current = row['Dadtunit']
    prev = row['Prev_Dadtunit']

    if pd.isna(current) or pd.isna(prev):
        return 1  # 缺乏去年数据或缺失时，默认设为 0

    if str(current).strip() == str(prev).strip():
        return 1  # 相同为 1
    else:
        return 0  # 不同为 0


df_merged['IsAuditorChanged_Corrected'] = df_merged.apply(calculate_auditor_changed, axis=1)

# ==========================================
# 第三步：只提取 fine 表格已有的样本进行精准匹配更新
# ==========================================
# 提取出主键和修正后的列
df_updates = df_merged[['Symbol', 'Year', 'IsAuditorChanged_Corrected']].drop_duplicates()

# 如果精炼表中原本有 IsAuditorChanged 这一列，先将其删掉避免重复冲突
if 'IsAuditorChanged' in df_fine.columns:
    df_fine = df_fine.drop(columns=['IsAuditorChanged'])

# 严格按 fine 表格现有的 Symbol 和 Year 进行左连接（Left Join）匹配
df_fine_updated = pd.merge(
    df_fine,
    df_updates,
    on=['Symbol', 'Year'],
    how='left'
)

# 将新列重命名为标准的 IsAuditorChanged
df_fine_updated.rename(columns={'IsAuditorChanged_Corrected': 'IsAuditorChanged'}, inplace=True)

# 检查更新后的数值分布情况
print(f"\n更新后精炼表行数: {df_fine_updated.shape[0]}")
print(f"IsAuditorChanged 列的数值分布情况:\n{df_fine_updated['IsAuditorChanged'].value_counts(dropna=False)}")

# ==========================================
# 第四步：保存更新后的表格
# ==========================================
output_path = './data/final1.xlsx'
df_fine_updated.to_excel(output_path, index=False)
print(f"\n✅ 成功完成 IsAuditorChanged 的重新计算与精确匹配，已保存至: {output_path}")