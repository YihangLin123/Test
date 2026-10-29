import pandas as pd

# 1. 读取双表头 Excel 数据文件
file_path = './data/C33_C40_normalized.xlsx'
df_raw = pd.read_excel(file_path, header=None)

# 提取双表头中的英文列名（第 0 行）和正文数据（从第 2 行开始）
header_en = list(df_raw.iloc[0].values)
df = df_raw.iloc[2:].copy()
df.columns = header_en

# 2. 检查是否存在 'Stkname' 列
if 'Stkname' in df.columns:
    # 规范化公司名称（去除前后空格），并使用 drop_duplicates 或 nunique 统计不重复的上市公司总数
    stk_names = df['Stkname'].astype(str).str.strip()
    total_companies = stk_names.nunique()

    print(f"📊 统计结果：")
    print(f"该数据集中一共有 {total_companies} 家不重复的上市公司。")
else:
    raise ValueError("表格中未找到 'Stkname'（公司名称）列，请检查列名。")