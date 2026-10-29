import pandas as pd
import numpy as np
from scipy.stats import kstest

# 1. 读取清洗后的总数据集
file_path = "../data/cleaned_merged_data_final_processed.xlsx"
df = pd.read_excel(file_path)

print(f"数据读取成功！总样本数: {df.shape[0]}, 总列数: {df.shape[1]}")

# 2. 精准提取第 7 到第 46 列作为候选指标特征（索引 6 到 46，左闭右开）
feature_cols = df.columns[6:46]
print(f"参与 K-S 正态性检验的候选指标总数: {len(feature_cols)}")

# 3. 逐个指标进行 Kolmogorov-Smirnov (K-S) 正态性检验
results = []
for col in feature_cols:
    # 剔除空值 (NaN)
    data_clean = df[col].dropna()

    if len(data_clean) >= 3:
        # 【关键步骤】K-S 检验通常需要先将数据进行标准化（Z-score），
        # 使其均值为0，标准差为1，再与标准正态分布进行拟合检验
        data_normalized = (data_clean - data_clean.mean()) / (data_clean.std() + 1e-10)

        # 执行 K-S 检验 (对比标准正态分布 'norm')
        stat, p_value = kstest(data_normalized, 'norm')

        # 判断标准：若 p 值 > 0.05，则视为服从正态分布；若 p <= 0.05，则拒绝正态假设（不服从）
        is_normal = "是" if p_value > 0.05 else "否"

        results.append({
            'Indicator': col,
            'KS_Statistic': stat,
            'P_Value': p_value,
            'Is_Normal': is_normal
        })

# 4. 将结果转换为 DataFrame 并打印查看
normality_df = pd.DataFrame(results)
print("\n--- K-S 正态性检验部分结果预览 ---")
print(normality_df.head(10))

# 统计汇总
normal_count = (normality_df['Is_Normal'] == '是').sum()
non_normal_count = (normality_df['Is_Normal'] == '否').sum()
print(f"\n【K-S 检验结论统计】")
print(f"总计检测指标: {len(normality_df)} 个")
print(f"服从正态分布的指标数: {normal_count} 个")
print(f"不服从正态分布的指标数: {non_normal_count} 个")

# 5. 保存检验结果到 Excel 文件（方便直接复制到论文中作为表格使用）
normality_df.to_excel("ks_normality_test_results.xlsx", index=False)
print("\n完整 K-S 检验结果已成功保存至 'ks_normality_test_results.xlsx' 文件中！")