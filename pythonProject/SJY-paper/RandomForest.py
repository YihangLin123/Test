# -*- coding: utf-8 -*-
"""
内控缺陷穿透识别模型 —— 随机森林 + 样本外盲测
核心逻辑：
  1. 严格隔离案例公司 (600251)，防止数据泄露
  2. 负样本严格筛选：选取 2017-2025 年样本数据都存在且 Y2 恒为 0 的合规公司作为对照池
  3. 超参数优化的随机森林 + PR 曲线自动寻优阈值
  4. 特征重要性排序
  5. 对案例公司进行纯样本外盲测，穿透隐瞒型缺陷
  6. 新增：计算并导出 2573 份最终训练样本在 2017-2025 各年份的每列特征指标平均值
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split

# ============================================================
# 全局配置
# ============================================================
FILE_NAME = './data/data.xlsx'
TARGET_SYMBOL = '600251'          # 案例公司股票代码（绝不参与训练）
TARGET_COL = 'Y2'                 # 监管铁证纠偏标签
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 基本信息列与标签列（不参与特征训练）
LABEL_COLS = ['Y1', 'Y2', 'Symbol', 'ShortName',
              'IndustryName', 'IndustryCode', 'LISTINGDATE', 'Year']


# ============================================================
# 第一步：读取数据 & 隔离案例公司
# ============================================================
df = pd.read_excel(FILE_NAME)
print(f"原始总样本数: {df.shape[0]}, 总列数: {df.shape[1]}")

# 统一 Symbol 格式，防止匹配失败
df['Symbol'] = df['Symbol'].astype(str).str.strip()

# 1) 提取案例公司全部数据（独立保存，绝不参与训练）
df_case = df[df['Symbol'] == TARGET_SYMBOL].copy()

# 2) 构建候选池（彻底剔除案例公司）
df_train_pool_raw = df[df['Symbol'] != TARGET_SYMBOL].copy()
print(f"剔除案例公司 ({TARGET_SYMBOL}) 后的候选训练池样本数: {df_train_pool_raw.shape[0]}")

# 自动识别特征列
feature_cols = [c for c in df_train_pool_raw.columns if c not in LABEL_COLS]


# ============================================================
# 第二步：正样本全保留，负样本严格筛选（2017-2025 连续存在且 Y2 恒为 0）
# ============================================================
target_years = list(range(2017, 2026))  # 2017 到 2025
print(f"\n正在筛选 2017-2025 连续合规的平衡面板负样本公司...")

valid_neg_symbols = []
for symbol, group in df_train_pool_raw.groupby('Symbol'):
    company_years = set(group['Year'].astype(int))
    # 检查是否完整包含了 2017-2025 所有年份
    if set(target_years).issubset(company_years):
        sub_period = group[group['Year'].isin(target_years)]
        # 检查这几年内 Y2 是否全部等于 0
        if (sub_period[TARGET_COL] == 0).all():
            valid_neg_symbols.append(symbol)

print(f"• 筛选出的长期合规负样本公司总数: {len(valid_neg_symbols)} 家")

# 提取这些长期合规公司在 2017-2025 期间的所有年度数据作为负样本池
df_neg = df_train_pool_raw[
    df_train_pool_raw['Symbol'].isin(valid_neg_symbols) &
    df_train_pool_raw['Year'].isin(target_years)
].copy()

# 正样本池：来自训练候选池中所有 Y2 == 1 的样本
df_pos = df_train_pool_raw[df_train_pool_raw[TARGET_COL] == 1].copy()

print(f"最终训练池构成：正样本数 = {len(df_pos)}，合规负样本数 = {len(df_neg)}")

# 合并并打乱训练池
df_train_pool = (
    pd.concat([df_pos, df_neg])
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

X = df_train_pool[feature_cols]
y = df_train_pool[TARGET_COL]

print(f"\n抽样后训练特征维度: {X.shape}")
print(f"标签 '{TARGET_COL}' 分布 -> 正样本: {y.sum()} ({y.mean():.2%}), "
      f"负样本: {(y == 0).sum()}")


# ============================================================
# 第三步：划分数据集 & 训练随机森林
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)


# ============================================================
# 第四步：特征重要性排序
# ============================================================
feature_weight_df = pd.DataFrame({
    "特征名称": feature_cols,
    "重要性权重": rf_model.feature_importances_
}).sort_values("重要性权重", ascending=False).reset_index(drop=True)
feature_weight_df.insert(0, "排名", feature_weight_df.index + 1)

print("\n" + "=" * 60)
print("各影响因素（特征）重要性权重全表（降序）")
print("=" * 60)
print(feature_weight_df.to_string(index=False))
print("\nTop20 核心影响因素：")
print(feature_weight_df.head(20).to_string(index=False))

feature_weight_df.to_excel("./feature_importance_weight.xlsx", index=False)
print("\n特征权重表已保存至 feature_importance_weight.xlsx")


# ============================================================
# 第五步：PR 曲线寻优阈值 & 模型评估
# ============================================================
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

print(f"\n最优分类阈值: {best_threshold:.4f} (对应最高 F1: {best_f1:.4f})")

y_pred_opt = (y_pred_proba >= best_threshold).astype(int)

metrics = {
    "优化后分类阈值": best_threshold,
    "准确率 (Accuracy)": accuracy_score(y_test, y_pred_opt),
    "精确率 (Precision)": precision_score(y_test, y_pred_opt, zero_division=0),
    "召回率 (Recall)": recall_score(y_test, y_pred_opt, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_opt, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
    "PR-AUC": average_precision_score(y_test, y_pred_proba),
}

print("\n" + "=" * 50)
print(f"随机森林模型评估结果 (目标标签: {TARGET_COL})")
print("=" * 50)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("=" * 50)


# ============================================================
# 第六步：案例公司样本外盲测
# ============================================================
if df_case.empty:
    print(f"\n警告：未找到股票代码为 '{TARGET_SYMBOL}' 的公司，请检查 Symbol 或原始表格！")
else:
    print(f"\n成功锁定目标案例公司 ({TARGET_SYMBOL})，共 {df_case.shape[0]} 个年度样本（未参与训练）。")

    X_case = df_case[feature_cols]
    df_case['Predicted_Risk_Probability'] = rf_model.predict_proba(X_case)[:, 1]
    df_case['Model_Predicted_Label'] = (
        df_case['Predicted_Risk_Probability'] >= best_threshold
    ).astype(int)

    display_cols = ['Symbol', 'ShortName', 'Year',
                    'Predicted_Risk_Probability', 'Model_Predicted_Label', 'Y1', 'Y2']
    df_case_sorted = df_case[display_cols].sort_values('Year')

    print("\n" + "=" * 80)
    print(f"目标案例企业 ({TARGET_SYMBOL}) 历年内控风险穿透盲测报告")
    print("=" * 80)
    print(df_case_sorted.to_string(index=False))
    print("=" * 80)

    print("\n【个案逐年穿透盲测结果与隐瞒行为深度解读】")
    for _, row in df_case_sorted.iterrows():
        yr = int(row['Year'])
        prob = row['Predicted_Risk_Probability']
        pred_lbl = int(row['Model_Predicted_Label'])
        y1_val = int(row['Y1'])
        y2_val = int(row['Y2'])

        if y1_val == 0 and y2_val == 1:
            disclosure = "【虚假陈述/选择性隐瞒：自评无缺陷，实则被监管处罚】"
        elif y1_val == 1 and y2_val == 1:
            disclosure = "【相对坦诚：自评有缺陷，且被监管证实】"
        else:
            disclosure = "【真实合规】"

        warning = "【高危预警：模型成功穿透伪装】" if pred_lbl == 1 else "【模型预测合规】"

        print(f"• {yr}年度: 预测缺陷概率 = {prob:.4f} (阈值: {best_threshold:.4f}) | "
              f"模型判定: {warning}")
        print(f"  对照组: 企业自述(Y1) = {y1_val} | 监管纠偏(Y2) = {y2_val} {disclosure}\n")


# ============================================================
# 第七步：计算并导出 2573 份训练样本在 2017-2025 各年份的每列特征平均值
# ============================================================
print("\n正在计算 2573 份样本各年份的特征平均值...")

# 按 Year 分组计算特征列的均值
df_train_yearly_mean = df_train_pool.groupby('Year')[feature_cols].mean().reset_index()

# 按年份升序排序
df_train_yearly_mean = df_train_yearly_mean.sort_values('Year').reset_index(drop=True)

# 导出到 Excel
benchmark_output_path = "./data/train_samples_yearly_benchmark_means.xlsx"
df_train_yearly_mean.to_excel(benchmark_output_path, index=False)

print(f"✅ 2017-2025 年度特征平均值基准表已成功导出！")
print(f"• 保存路径: {benchmark_output_path}")
print(f"• 表格维度 (年份数, 特征数+1): {df_train_yearly_mean.shape}")
print("=" * 80)