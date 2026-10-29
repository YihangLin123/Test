import pandas as pd
import re

# 1. 读取证监会行政处罚表格
file_path = "CMS_CSRCAdminPunish.xlsx"
df = pd.read_excel(file_path)

print(f"原始处罚表格总行数: {len(df)}")

# 2. 定义需要检索的“内控缺陷/内控失效”核心关键词列表
# 导师要求：处罚意见或监管函中明确指出“内部控制失效”、“内部控制存在重大缺陷”、“内控机制不健全”等
keywords = [
    "内部控制缺陷",
    "内控缺陷",
    "内部控制失效",
    "内控失效",
    "内部控制不健全",
    "内控机制不健全",
    "内部控制存在重大缺陷",
    "内控存在重大缺陷",
    "内部控制执行不到位",
    "内控执行不到位",
    "未有效执行内部控制",
    "内部控制流于形式"
]

# 将关键词列表组合成正则表达式的“或”模式 (keyword1|keyword2|...)
pattern = '|'.join([re.escape(kw) for kw in keywords])

# 3. 编写筛选函数：检查 Activity 列中是否包含上述关键词
def check_internal_control_flaw(text):
    if pd.isna(text):
        return False
    # 转为字符串并搜索关键词
    if re.search(pattern, str(text)):
        return True
    return False

# 4. 执行筛选
# 假设文本内容在名为 'Activity' 的列中（如果列名不同，请自行修改）
filtered_df = df[df['Activity'].apply(check_internal_control_flaw)].copy()

print(f"筛选出包含明确‘内控缺陷/失效’关键词的处罚记录数: {len(filtered_df)}")

# 5. 将筛选结果保存为新的 Excel 表格，供你人工核对挑选最佳案例
output_path = "filtered_companies_with_ic_flaws.xlsx"
filtered_df.to_excel(output_path, index=False)
print(f"筛选结果已成功保存至 '{output_path}' 文件！")

# 6. 打印前几条结果预览（查看股票代码、简称和部分处罚案由）
if len(filtered_df) > 0:
    # 假设公司名称列叫 'StkName' 或 'Symbol'，视你的表格实际表头而定
    display_cols = [col for col in ['Symbol', 'StkName', 'PunishDate', 'Activity'] if col in filtered_df.columns]
    print("\n--- 筛选出的部分公司预览 ---")
    print(filtered_df[display_cols].head(5))
else:
    print("没有匹配到包含指定关键词的记录，建议检查 Activity 列名或适当放宽、调整关键词。")