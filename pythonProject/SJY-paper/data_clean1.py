import os
import pandas as pd

# 1. 读取刚刚合并好的大宽表
input_path = "merged_data.xlsx"
if not os.path.exists(input_path):
    print(f"错误：找不到文件 '{input_path}'，请先确认该文件与脚本在同一目录下。")
else:
    df = pd.read_excel(input_path)
    print(f"原始大宽表总行数: {len(df)}")

    # 核心设定：定义你的核心案例白名单（广汇物流股票代码 600603）
    case_symbols = ['600603']

    # 用于存放所有被剔除样本的列表
    dropped_records = []


    def record_dropped(df_dropped, reason):
        if len(df_dropped) > 0:
            temp_df = df_dropped.copy()
            temp_df['Drop_Reason'] = reason
            dropped_records.append(temp_df)


    # ---------------------------------------------------------
    # 第一步：剔除金融类上市公司
    # ---------------------------------------------------------
    industry_col = None
    for col in df.columns:
        if 'industry' in col.lower() or '行业' in col:
            industry_col = col
            break

    if industry_col:
        before_len = len(df)

        is_financial = df[industry_col].astype(str).str.contains('金融', na=False)
        for col in df.columns:
            if 'code' in col.lower() and ('ind' in col.lower() or '行业' in col):
                is_financial = is_financial | df[col].astype(str).str.upper().str.startswith('J', na=False)

        is_case_company = df['Clean_Symbol'].astype(str).str.zfill(6).isin(case_symbols)
        target_to_drop = is_financial & (~is_case_company)

        df_dropped = df[target_to_drop]
        df = df[~target_to_drop].reset_index(drop=True)  # 重置索引消除隐患

        record_dropped(df_dropped, "第一步：剔除金融类上市公司")
        print(f"【第一步：剔除金融业】过滤掉 {before_len - len(df)} 行，剩余行数: {len(df)}")
    else:
        print("提示：未检测到明显的行业列，跳过金融业剔除。")

    # ---------------------------------------------------------
    # 第二步：剔除 ST 或 *ST 公司（双重拦截：简称 + 上市状态，白名单保留）
    # ---------------------------------------------------------
    name_col = None
    for col in df.columns:
        if col.lower() in ['shortname', 'stknme', 'companyname', '公司简称']:
            name_col = col
            break

    before_len = len(df)

    is_st_name = pd.Series(False, index=df.index)
    if name_col:
        is_st_name = df[name_col].astype(str).str.upper().str.contains('ST', na=False)

    is_st_state = pd.Series(False, index=df.index)
    if 'LISTINGSTATE' in df.columns:
        is_st_state = df['LISTINGSTATE'].astype(str).str.upper().str.contains('ST|退市|风险', na=False)

    is_st_any = is_st_name | is_st_state
    is_case_company = df['Clean_Symbol'].astype(str).str.zfill(6).isin(case_symbols)

    target_to_drop = is_st_any & (~is_case_company)

    df_dropped = df[target_to_drop]
    df = df[~target_to_drop].reset_index(drop=True)  # 重置索引消除隐患

    record_dropped(df_dropped, "第二步：剔除ST/*ST公司(广汇物流已被白名单保护)")
    print(f"【第二步：剔除ST公司】过滤掉 {before_len - len(df)} 行（已双重过滤简称与状态，广汇物流受到白名单保护保留）")

    # ---------------------------------------------------------
    # 第三步：剔除上市时间不足两年的公司（含时间逻辑错误防穿越）
    # ---------------------------------------------------------
    list_date_col = None
    year_col = 'Year'

    for col in df.columns:
        if 'listingdate' in col.lower() or '上市日期' in col:
            list_date_col = col
            break

    if list_date_col and year_col in df.columns:
        before_len = len(df)

        # 显式对齐索引计算
        parsed_list_date = pd.to_datetime(df[list_date_col], errors='coerce')
        list_year = parsed_list_date.dt.year

        is_time_travel = (df[year_col] < list_year)
        is_young = (df[year_col] - list_year) < 2

        is_case_company = df['Clean_Symbol'].astype(str).str.zfill(6).isin(case_symbols)
        target_to_drop = (is_young | is_time_travel) & (~is_case_company)

        df_dropped = df[target_to_drop].copy()
        df = df[~target_to_drop].reset_index(drop=True)

        record_dropped(df_dropped, "第三步：上市时间不足两年或时间逻辑错误(广汇物流保护)")
        print(f"【第三步：剔除上市不足两年/时间错误】过滤掉 {before_len - len(df)} 行，剩余行数: {len(df)}")
    else:
        print("提示：未找到上市日期列或年份列，跳过上市年限筛选。")

    # ---------------------------------------------------------
    # 生成并导出被剔除样本的详细报告
    # ---------------------------------------------------------
    if dropped_records:
        all_dropped_df = pd.concat(dropped_records, ignore_index=True)
        dropped_output_path = "dropped_samples_report.xlsx"
        all_dropped_df.to_excel(dropped_output_path, index=False)
        print(f"\n【报告生成】共计剔除不合规样本 {len(all_dropped_df)} 行。")
        print(f"详细剔除清单及原因已保存至: {dropped_output_path}")

        print("\n被剔除样本示例预览：")
        show_cols = [c for c in ['Clean_Symbol', 'ShortName', 'Year', 'Drop_Reason'] if c in all_dropped_df.columns]
        print(all_dropped_df[show_cols].head(5))
    else:
        print("\n【提示】本次清洗没有发现被剔除的样本。")

    # ---------------------------------------------------------
    # 保存清洗后的最终大宽表
    # ---------------------------------------------------------
    output_cleaned_path = "cleaned_merged_data1111.xlsx"
    df.to_excel(output_cleaned_path, index=False)

    print("=" * 60)
    print(f"【全部完成】数据清洗完毕！(无任何索引对齐警告)")
    print(f"最终有效样本总行数: {len(df)}")
    print(f"清洗后的最终数据已保存至: {output_cleaned_path}")
    print("=" * 60)