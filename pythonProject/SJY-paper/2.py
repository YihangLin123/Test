import pandas as pd
import re

# 1. 读取带有 Y2 标签的处罚明细表
df_punish = pd.read_excel('CMS_CSRCAdminPunish_with_Y2.xlsx')

if 'Activity' not in df_punish.columns or 'Symbol' not in df_punish.columns or 'Y2' not in df_punish.columns:
    raise ValueError("错误：处罚表中缺少必要列！")

# 2. 定义重大缺陷的中文核心关键词（与你的 major_defect_codes 完美对应）
major_defect_keywords = [
    '虚构利润', '虚列资产', '虚假记载', '误导性陈述',
    '重大遗漏', '欺诈上市', '擅自改变资金用途', '占用公司资产',
    '内幕交易', '操纵股价', '违规担保', '虚增收入', '虚增利润', '虚增成本'
]


# 3. 核心智能提取函数：基于“重大缺陷关键词”或“财报语境”提取年份
def extract_precise_violation_years(row):
    text = row['Activity']
    y2_val = row['Y2']
    declare_date = row['DeclareDate']

    # 如果本来就不是重大缺陷 (Y2=0)，我们不需要费劲去对齐财报年，直接用公告年或跳过均可
    # 但为了保险，如果文本里有年份也可以提取
    if pd.isna(text):
        if pd.notna(declare_date):
            dt = pd.to_datetime(declare_date, errors='coerce', format='mixed')
            return [dt.year] if pd.notna(dt) else []
        return []

    text_str = str(text)
    matched_years = set()

    # 策略 A：寻找“年年度报告”、“年半年度报告”前面的年份（最稳健的财报年抓取）
    pattern_reports = r'(20\d{2})\s*年(?:度)?(?:半年度|季度)?报告'
    for y in re.findall(pattern_reports, text_str):
        matched_years.add(int(y))

    # 策略 B：如果在文本中同时出现了重大缺陷关键词（如“虚构利润”、“占用公司资产”等），
    # 我们可以放心地把文本中出现的、紧邻这些词附近的 20xx 年全部纳入
    has_major_keyword = any(kw in text_str for kw in major_defect_keywords)
    if has_major_keyword:
        # 抓取文本中所有形如 "20xx年" 或 "20xx年度" 的年份
        all_years_in_text = re.findall(r'(20\d{2})\s*年', text_str)
        for y in all_years_in_text:
            matched_years.add(int(y))

    # 如果上面两种方法都没抓到年份（比如文本非常简短），则退一步使用公告日期（DeclareDate）的年份
    if not matched_years and pd.notna(declare_date):
        dt = pd.to_datetime(declare_date, errors='coerce', format='mixed')
        if pd.notna(dt):
            matched_years.add(dt.year)

    return list(matched_years)


# 4. 逐行应用提取函数
extracted_records = []

for idx, row in df_punish.iterrows():
    symbol = row['Symbol']
    y2_val = row['Y2']
    declare_date = row['DeclareDate']

    # 获取该条处罚对应的所有违规所属年份列表
    years = extract_precise_violation_years(row)

    for yr in years:
        extracted_records.append({
            'Symbol': symbol,
            'Year': yr,
            'Y2': y2_val
        })

df_punish_years = pd.DataFrame(extracted_records)

if df_punish_years.empty:
    raise ValueError("警告：未能解析出任何年份！")

# 5. 同一年多条处罚取最大值 (max Y2) -> 确保只要当年有重大缺陷处罚，Y2 就是 1
df_punish_final = df_punish_years.groupby(['Symbol', 'Year'])['Y2'].max().reset_index()

# ==========================================
# 6. 和主大宽表进行精准左连接合并 (Left Merge)
# ==========================================
df_main = pd.read_excel('cleaned_merged_data.xlsx')

# 统一类型防报错
df_main['Symbol'] = df_main['Symbol'].astype(str).str.strip()
df_punish_final['Symbol'] = df_punish_final['Symbol'].astype(str).str.strip()
df_main['Year'] = pd.to_numeric(df_main['Year'], errors='coerce').astype(int)
df_punish_final['Year'] = pd.to_numeric(df_punish_final['Year'], errors='coerce').astype(int)

# 如果主表原本有 Y2，先清理掉
if 'Y2' in df_main.columns:
    df_main = df_main.drop(columns=['Y2'])

# 左连接并入主表
df_merged = pd.merge(df_main, df_punish_final, on=['Symbol', 'Year'], how='left')

# 未中招的年份（即没有重大缺陷违规的年份）自动赋值为 0
df_merged['Y2'] = df_merged['Y2'].fillna(0).astype(int)

# 确保 Y2 稳稳呆在表格的【最后一列】
feature_cols = [col for col in df_merged.columns if col != 'Y2']
df_merged = df_merged[feature_cols + ['Y2']]

# 7. 打印统计结果预览
print("--- 关键词匹配 + 财报语境提取后，主表 Y2 标签最终分布 ---")
print(df_merged['Y2'].value_counts(dropna=False))

# 8. 保存最终大宽表
df_merged.to_excel('cleaned_merged_data_final.xlsx', index=False)
print("\n太完美了！已成功通过“重大缺陷中文关键词 + 财报语境”双重判定提取年份，大宽表已成功更新并保存！")