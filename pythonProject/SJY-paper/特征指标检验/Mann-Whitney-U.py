import pandas as pd
from scipy.stats import mannwhitneyu

# 1. 读取清洗后的总数据集
file_path = "../data/cleaned_merged_data_final_processed.xlsx"
df = pd.read_excel(file_path)

# 2. 提取特征列（第 7 到第 46 列，共 40 个指标）
feature_cols = df.columns[6:46]

# 设定你要检验的目标标签（此处以 Y1 为例，若测 Y2 可直接改为 'Y2'）
target_col = 'Y1'
print(f"当前检验的目标标签: {target_col}")

# 3. 将数据集按标签拆分为两组
group_normal = df[df[target_col] == 0]  # 无缺陷组（负样本）
group_defect = df[df[target_col] == 1]  # 有缺陷组（正样本）

print(f"无缺陷组样本数: {len(group_normal)}, 有缺陷组样本数: {len(group_defect)}")

# 4. 逐个指标执行 Mann-Whitney U 检验
results = []
for col in feature_cols:
    x = group_normal[col].dropna()
    y = group_defect[col].dropna()

    if len(x) > 0 and len(y) > 0:
        # 执行双侧 Mann-Whitney U 检验
        stat, p_value = mannwhitneyu(x, y, alternative='two-sided')

        # 显著性判定标准：p < 0.05 视为差异显著
        is_significant = "是" if p_value < 0.05 else "否"

        results.append({
            'Indicator': col,
            'U_Statistic': stat,
            'P_Value': p_value,
            'Is_Significant': is_significant
        })

# 5. 转换为 DataFrame 并输出预览
mwu_df = pd.DataFrame(results)
print("\n--- Mann-Whitney U 显著性检验部分结果预览 ---")
print(mwu_df.head(10))

# 6. 统计汇总
sig_count = (mwu_df['Is_Significant'] == '是').sum()
insig_count = (mwu_df['Is_Significant'] == '否').sum()

print(f"\n【Mann-Whitney U 检验结论统计】")
print(f"总计检测指标: {len(mwu_df)} 个")
print(f"表现显著（P < 0.05，予以保留）的指标数: {sig_count} 个")
print(f"表现不显著（P >= 0.05，予以剔除）的指标数: {insig_count} 个")

# 7. 保存显著性检验结果到 Excel 文件（方便后续写入论文表格）
mwu_df.to_excel("mannwhitney_u_test_results.xlsx", index=False)
print("\n检验结果已成功保存至 'mannwhitney_u_test_results.xlsx' 文件中！")