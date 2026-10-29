import pandas as pd
import re

# ==========================================
# 第一步：读取总表格和处罚表
# ==========================================
main_data_path = './data/cleaned_merged_data_feature_selected_final.xlsx'
df_panel = pd.read_excel(main_data_path)

punish_data_path = './data/CMS_CSRCAdminPunish_with_Y2.xlsx'
df_punish = pd.read_excel(punish_data_path)

print(f"总表格原始行数: {df_panel.shape[0]}")
print(f"处罚表原始行数: {df_punish.shape[0]}")

df_panel['Symbol'] = df_panel['Symbol'].astype(str).str.strip()
df_punish['Symbol'] = df_punish['Symbol'].astype(str).str.strip()

# ==========================================
# 第二步：对 CMS_CSRCAdminPunish_with_Y2 表格进行过滤与文本定年
# ==========================================

year_col_in_punish = None
for col in ['Year', 'NoticeYear', 'PunishYear', 'AnnounceYear']:
    if col in df_punish.columns:
        year_col_in_punish = col
        break

if year_col_in_punish:
    df_punish = df_punish[df_punish[year_col_in_punish] >= 2016].copy()
    print(f"依据 '{year_col_in_punish}' 过滤掉 2016 年以前记录后，处罚表剩余行数: {df_punish.shape[0]}")


# 2. 【终极升级版】：彻底屏蔽公告日、立案调查等事后年份干扰
def extract_true_violation_years_smart(text):
    if pd.isna(text):
        return []
    text = str(text)

    sentences = re.split(r'[；。；\n]', text)
    target_years = set()

    for sent in sentences:
        # 【关键修复】：如果句子包含“发布、公告、立案、调查、告知、听证”等事后程序词，直接跳过！
        is_announcement_info = any(
            kw in sent for kw in ['发布', '公告', '立案', '调查', '告知', '听证', '办理终结', '作出'])
        is_tenure_info = any(kw in sent for kw in ['任', '担任', '离任', '生于', '身份证'])
        has_action_info = any(kw in sent for kw in
                              ['买入', '卖出', '交易', '违规', '持有', '减持', '增持', '虚增', '未披露', '占用', '期间',
                               '发生', '差错'])

        # 只要是公告日、立案调查、或纯任职背景，一律跳过不提取年份
        if is_announcement_info:
            continue
        if is_tenure_info and not has_action_info:
            continue

        # 在真正描述违规事实的句子中提取年份
        years_in_sent = re.findall(r'20[1-2][0-9]', sent)
        if years_in_sent:
            years_in_sent = [int(y) for y in years_in_sent]

            if "至" in sent or "到" in sent or "-" in sent or "间" in sent:
                min_y, max_y = min(years_in_sent), max(years_in_sent)
                for y in range(min_y, max_y + 1):
                    if y >= 2016:
                        target_years.add(y)
            else:
                for y in years_in_sent:
                    if y >= 2016:
                        target_years.add(y)

    return sorted(list(target_years))


# 3. 展开处罚表：根据 Activity 列提取真实发生年份
expanded_records = []

for idx, row in df_punish.iterrows():
    symbol = row['Symbol']
    activity_text = row['Activity']

    true_years = extract_true_violation_years_smart(activity_text)

    if true_years:
        for yr in true_years:
            expanded_records.append({'Symbol': symbol, 'True_Violation_Year': yr})
    else:
        if year_col_in_punish and pd.notna(row[year_col_in_punish]):
            fallback_yr = int(row[year_col_in_punish]) - 1
            if fallback_yr >= 2016:
                expanded_records.append({'Symbol': symbol, 'True_Violation_Year': fallback_yr})

df_true_violations = pd.DataFrame(expanded_records).drop_duplicates()
print(f"经过升级后 Activity 文本穿透解析，共生成 {df_true_violations.shape[0]} 条【公司-真实违规年】映射对。")

# ==========================================
# 第三步：将过滤定年后的标签合并到总表格中
# ==========================================
df_panel = df_panel[df_panel['Year'] >= 2016].reset_index(drop=True)
df_panel['Y2'] = 0

df_true_violations['Actual_Y2'] = 1

df_merged_final = pd.merge(
    df_panel,
    df_true_violations[['Symbol', 'True_Violation_Year', 'Actual_Y2']],
    left_on=['Symbol', 'Year'],
    right_on=['Symbol', 'True_Violation_Year'],
    how='left'
)

df_merged_final['Y2'] = df_merged_final['Actual_Y2'].fillna(0).astype(int)
df_merged_final.drop(columns=['True_Violation_Year', 'Actual_Y2'], inplace=True, errors='ignore')

print(f"\n合并完成！总表格当前样本数: {df_merged_final.shape[0]}")
print(f"修正定年后的正样本 (Y2 = 1) 总数: {df_merged_final['Y2'].sum()}")

# ==========================================
# 第四步：保存最终对齐的总表格
# ==========================================
output_final_path = './data/fine.xlsx'
df_merged_final.to_excel(output_final_path, index=False)
print(f"已成功生成最终清洗与定年对齐的总表格，并保存至: {output_final_path}")