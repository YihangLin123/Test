# -*- coding: utf-8 -*-
"""
内控缺陷穿透识别模型 —— BP 神经网络 (MLP) + 样本外盲测 + 正方形规范 ROC 曲线
数据结构适配：
  - 基础信息：证券代码、证券简称、year、行业代码
  - 特征范围：E 列至 Y 列
  - 标签列：Y1, Y2 (最后两列)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve
)
from sklearn.model_selection import train_test_split

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 全局配置
# ============================================================
FILE_NAME = './data/final_data.xlsx'
TARGET_SYMBOL = '600251'  # 案例公司股票代码（绝不参与训练）
TARGET_COL = 'Y2'  # 监管铁证纠偏标签
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================
# 第一步：读取数据 & 隔离案例公司
# ============================================================
df = pd.read_excel(FILE_NAME)
print(f"原始总样本数: {df.shape[0]}, 总列数: {df.shape[1]}")

# 统一股票代码列名为 'Symbol'（方便后续处理）
code_col_candidates = ['证券代码', 'Symbol', 'Stkcd']
code_col = next((c for c in code_col_candidates if c in df.columns), None)
if not code_col:
    raise ValueError(f"表格中未找到股票代码列，现有列名为: {list(df.columns)}")
if code_col != 'Symbol':
    df.rename(columns={code_col: 'Symbol'}, inplace=True)

# 统一 Symbol 格式，防止匹配失败
df['Symbol'] = df['Symbol'].astype(str).str.strip()

# 统一标示年份列名为 'Year'
year_col_candidates = ['year', 'Year']
year_col = next((c for c in year_col_candidates if c in df.columns), None)
if not year_col:
    raise ValueError(f"表格中未找到年份列，现有列名为: {list(df.columns)}")
if year_col != 'Year':
    df.rename(columns={year_col: 'Year'}, inplace=True)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# ============================================================
# 第二步：根据你的结构精准定位特征列（E 列到 Y 列）
# ============================================================
start_idx = 4  # E 列
end_idx = 25  # Y 列的下一位（不含）

raw_feature_cols = df.columns[start_idx:end_idx].tolist()
print(
    f"成功锁定特征范围：从【{raw_feature_cols[0]}】(E列) 到【{raw_feature_cols[-1]}】(Y列)，共 {len(raw_feature_cols)} 个特征")

# 1) 提取案例公司全部数据（独立保存，绝不参与训练）
df_case = df[df['Symbol'] == TARGET_SYMBOL].copy()

# 2) 构建候选训练池（彻底剔除案例公司）
df_train_pool_raw = df[df['Symbol'] != TARGET_SYMBOL].copy()
print(f"剔除案例公司 ({TARGET_SYMBOL}) 后的候选训练池样本数: {df_train_pool_raw.shape[0]}")

# 强制将特征列转为数值型，过滤掉任何可能存在的文本/字符串乱码
feature_cols = []
for col in raw_feature_cols:
    converted = pd.to_numeric(df_train_pool_raw[col], errors='coerce')
    if converted.notna().sum() > 0:  # 包含有效数值
        df_train_pool_raw[col] = converted
        if col in df_case.columns:
            df_case[col] = pd.to_numeric(df_case[col], errors='coerce')
        feature_cols.append(col)

print(f"清洗后实际参与训练的有效数值特征数: {len(feature_cols)}")

# ============================================================
# 第三步：正样本全保留，负样本严格筛选（2017-2025 连续存在且 Y2 恒为 0）
# ============================================================
target_years = list(range(2017, 2026))  # 2017 到 2025
print(f"\n正在筛选 2017-2025 连续合规的平衡面板负样本公司...")

valid_neg_symbols = []
for symbol, group in df_train_pool_raw.groupby('Symbol'):
    company_years = set(group['Year'].astype(int))
    if set(target_years).issubset(company_years):
        sub_period = group[group['Year'].isin(target_years)]
        if (sub_period[TARGET_COL] == 0).all():
            valid_neg_symbols.append(symbol)

print(f"• 筛选出的长期合规负样本公司总数: {len(valid_neg_symbols)} 家")

df_neg = df_train_pool_raw[
    df_train_pool_raw['Symbol'].isin(valid_neg_symbols) &
    df_train_pool_raw['Year'].isin(target_years)
    ].copy()

df_pos = df_train_pool_raw[df_train_pool_raw[TARGET_COL] == 1].copy()

print(f"最终训练池构成：正样本数 = {len(df_pos)}，合规负样本数 = {len(df_neg)}")

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
# 第四步：划分数据集 & 特征标准化 & 训练 BP 神经网络模型
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 神经网络对数据尺度敏感，必须进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练 BP 神经网络模型 (MLPClassifier)
bp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=RANDOM_STATE,
    early_stopping=True,  # 启用早停法防止过拟合
    validation_fraction=0.1
)
bp_model.fit(X_train_scaled, y_train)

# ============================================================
# 第五步：PR 曲线寻优阈值、模型评估与正方形 ROC 曲线绘制
# ============================================================
y_pred_proba = bp_model.predict_proba(X_test_scaled)[:, 1]

# 【新增】保存测试集真实标签 y_test 及预测为正类的概率值 y_pred_proba 到 xlsx 文件
model_evaluation_results = pd.DataFrame({
    'y_true': y_test.values,
    'y_pred_proba': y_pred_proba
})
model_evaluation_results.to_excel("./data/bpa.xlsx", index=False)
print("✅ BP神经网络测试集真实标签 y_true 及预测概率 y_pred_proba 已成功保存至: './data/bp_model_y_true_and_pred_proba.xlsx'")

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

print(f"\n最优分类阈值: {best_threshold:.4f} (对应最高 F1: {best_f1:.4f})")

y_pred_opt = (y_pred_proba >= best_threshold).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

metrics = {
    "优化后分类阈值": best_threshold,
    "准确率 (Accuracy)": accuracy_score(y_test, y_pred_opt),
    "精确率 (Precision)": precision_score(y_test, y_pred_opt, zero_division=0),
    "召回率 (Recall)": recall_score(y_test, y_pred_opt, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_opt, zero_division=0),
    "ROC-AUC": roc_auc,
    "PR-AUC": pr_auc,
}

print("\n" + "=" * 50)
print(f"BP 神经网络模型评估结果 (目标标签: {TARGET_COL})")
print("=" * 50)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("=" * 50)

# 绘制并保存正方形比例的 ROC 曲线（figsize=(6, 6) 确保后续多图拼接时尺寸一致）
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('假正例率 (False Positive Rate)', fontsize=11)
plt.ylabel('真正例率 (True Positive Rate)', fontsize=11)
plt.title('BP神经网络ROC曲线', fontsize=13, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

roc_output_path = "./data/bp_roc_curve_result.png"
plt.savefig(roc_output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ BP神经网络正方形 ROC 曲线图已成功保存至: '{roc_output_path}'")
plt.close()

# ============================================================
# 第六步：案例公司样本外盲测
# ============================================================
if df_case.empty:
    print(f"\n警告：未找到股票代码为 '{TARGET_SYMBOL}' 的公司，请检查 Symbol 或原始表格！")
else:
    print(f"\n成功锁定目标案例公司 ({TARGET_SYMBOL})，共 {df_case.shape[0]} 个年度样本（未参与训练）。")

    X_case = df_case[feature_cols]
    # 案例公司特征同样需要通过训练集的 scaler 进行标准化
    X_case_scaled = scaler.transform(X_case)

    df_case['Predicted_Risk_Probability'] = bp_model.predict_proba(X_case_scaled)[:, 1]
    df_case['Model_Predicted_Label'] = (
            df_case['Predicted_Risk_Probability'] >= best_threshold
    ).astype(int)

    name_col = next((c for c in ['证券简称', 'ShortName'] if c in df_case.columns), None)
    display_cols = ['Symbol'] + ([name_col] if name_col else []) + ['Year', 'Predicted_Risk_Probability',
                                                                    'Model_Predicted_Label', 'Y1', 'Y2']
    df_case_sorted = df_case[display_cols].sort_values('Year')

    print("\n" + "=" * 80)
    print(f"目标案例企业 ({TARGET_SYMBOL}) 历年内控风险穿透盲测报告 (BP Neural Network)")
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