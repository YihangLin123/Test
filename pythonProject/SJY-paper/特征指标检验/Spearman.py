import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取清洗后的总数据集
file_path = "../data/cleaned_merged_data_final_processed_2.xlsx"
df = pd.read_excel(file_path)

print(f"数据读取成功！总样本数: {df.shape[0]}, 总列数: {df.shape[1]}")

# 2. 精准提取 G 列到 AG 列（第 7 到第 33 列，Python 索引 6 到 33）作为 27 个候选指标
feature_cols = df.columns[6:33].tolist()
print(f"参与 Spearman 多重共线性检验的指标总数: {len(feature_cols)} 个")

# ==================== 定义财务表现维度的英文列名（按序号从小到大） ====================
dim2_codes = [
    'F050201B', 'F050501B', 'F053301B', 'F051501B', 'F080601A',
    'F081001B', 'F081601B', 'F082601B', 'F040201B', 'F041701B',
    'F060201B', 'F061801B', 'F011201A', 'F010101A', 'F010201A',
    'F010701B', 'F011601A', 'F012301B', 'F012401B'
]

# 建立英文名到序号的映射字典，用于热力图坐标轴展示
code_mapping = {
    'F050201B': 'A10',
    'F050501B': 'A11',
    'F053301B': 'A12',
    'F051501B': 'A13',
    'F080601A': 'A14',
    'F081001B': 'A15',
    'F081601B': 'A16',
    'F082601B': 'A17',
    'F040201B': 'A18',
    'F041701B': 'A22',
    'F060201B': 'A23',
    'F061801B': 'A24',
    'F011201A': 'A25',
    'F010101A': 'A26',
    'F010201A': 'A27',
    'F010701B': 'A28',
    'F011601A': 'A29',
    'F012301B': 'A31',
    'F012401B': 'A32'
}

# 过滤出表格中实际存在的列
type2_cols = [c for c in dim2_codes if c in df.columns]

# ==================== 绘制“财务表现维度”热力图（无大标题、无小标题） ====================
if len(type2_cols) > 0:
    plt.figure(figsize=(10, 8))

    sub_df = df[type2_cols].copy()
    # 将列名替换为对应的序号（A10, A11 等）
    sub_df.columns = [code_mapping.get(c, c) for c in type2_cols]

    # 计算 Spearman 相关系数
    sub_corr = sub_df.corr(method='spearman')

    # 绘制热力图
    ax = sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                     cbar=True, annot_kws={"size": 10})
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()

    # 保存热力图图片
    heatmap_output_path = "../data/spearman_heatmap_financial_dimension.png"
    plt.savefig(heatmap_output_path, dpi=300, bbox_inches='tight')
    print(f"\n【成功】财务表现维度热力图已绘制并保存至: '{heatmap_output_path}'")
    plt.show()

# ==================== 3. 整体的多重共线性检验与清洗 ====================
X_features = df[feature_cols]
spearman_corr = X_features.corr(method='spearman').abs()

threshold = 0.7
to_drop = set()

for i in range(len(spearman_corr.columns)):
    for j in range(i):
        if spearman_corr.iloc[i, j] > threshold:
            colname = spearman_corr.columns[i]
            to_drop.add(colname)

final_features = [col for col in feature_cols if col not in to_drop]

print("\n" + "=" * 50)
print(f"--- Spearman 多重共线性清洗结果 ---")
print(f"设定相关性阈值         : {threshold}")
print(f"因多重共线性被剔除的指标数: {len(to_drop)} 个")
if len(to_drop) > 0:
    print(f"被剔除的冗余指标列名    : {list(to_drop)}")
print(f"最终锁定的核心特征数   : {len(final_features)} 个")
print("=" * 50)

# 4. 保存最终精简的数据集
base_cols = df.columns[:6].tolist()
label_cols = df.columns[33:].tolist()

final_cols_to_save = base_cols + final_features + label_cols
df_final_selected = df[final_cols_to_save]

output_file = "../data/cleaned_merged_data_feature_selected_final.xlsx"
df_final_selected.to_excel(output_file, index=False)
print(f"\n特征清洗全部完成！最终精简的数据集已成功保存至: '{output_file}'")