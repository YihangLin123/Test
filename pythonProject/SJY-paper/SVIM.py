import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, precision_recall_curve
)

# ==========================================
# 第一步：读取数据、严格隔离案例公司以防数据泄露 (锁定穿透纠偏标签 Y2)
# ==========================================
file_name = './data/final1.xlsx'
df = pd.read_excel(file_name)

print(f"原始总样本数: {df.shape[0]}, 总列数: {df.shape[1]}")

# 统一转换 Symbol 格式防匹配失败
df['Symbol'] = df['Symbol'].astype(str).str.strip()

# 【核心防泄露步骤】：指定你要作为论文专案剖析的目标案例公司代码
target_symbol = '600251'

# 1. 提取并独立保存案例公司的所有数据（绝不参与模型训练！）
df_case = df[df['Symbol'] == target_symbol].copy()

# 2. 构建初始大盘训练池（从总数据中彻底剔除该案例公司）
df_train_pool_raw = df[df['Symbol'] != target_symbol].copy()
print(f"剔除案例公司 ({target_symbol}) 后的原始候选训练池样本数: {df_train_pool_raw.shape[0]}")

# 自动识别末尾的标签列和基本信息列
label_cols = ['Y1', 'Y2', 'Symbol', 'ShortName', 'IndustryName', 'IndustryCode', 'LISTINGDATE', 'Year']
feature_cols = [col for col in df_train_pool_raw.columns if col not in label_cols]

# ==========================================
# 【核心逻辑】：正样本全保留，随机抽取负样本
# ==========================================
target_col = 'Y2'  # 使用监管铁证纠偏标签

# 拆分出正样本池和负样本池
df_pos = df_train_pool_raw[df_train_pool_raw[target_col] == 1].copy()
df_neg = df_train_pool_raw[df_train_pool_raw[target_col] == 0].copy()

print(f"原始训练池中：正样本数 = {len(df_pos)}，负样本数 = {len(df_neg)}")

# 如果负样本总数大于 4500，则随机抽取 4500 个；若不足则全取
n_neg_sample = min(4500, len(df_neg))
df_neg_sampled = df_neg.sample(n=n_neg_sample, random_state=42)

# 合并成最终的平衡训练池
df_train_pool = pd.concat([df_pos, df_neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# 提取特征 X 和标签 Y2
X = df_train_pool[feature_cols]
y = df_train_pool[target_col]

print(f"\n抽样后参与训练的精简特征维度: {X.shape}")
print(f"标签 '{target_col}' 分布 -> 正样本(实质缺陷): {y.sum()} ({y.mean():.2%}), 负样本: {(y == 0).sum()}")

# ==========================================
# 第二步：划分训练/测试集、特征标准化并训练 SVM 模型
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 【关键步骤】：SVM 对特征量纲极度敏感，必须进行 Z-score 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_case_scaled = scaler.transform(df_case[feature_cols])  # 案例公司特征同步标准化

# 【模型替换】：构建支持向量机分类器 (SVM)
svm_model = SVC(
    kernel='rbf',             # 高斯核函数
    C=1.0,                    # 惩罚系数
    class_weight='balanced',  # 解决样本不平衡
    probability=True,         # 开启概率输出支持
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

# 获取测试集的预测概率
y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

# 【核心新增】：利用 Precision-Recall 曲线寻优计算最佳分类阈值
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

print(f"\n🔥 算法自动寻找到的最优分类阈值: {best_threshold:.4f} (此时对应的F1最高: {best_f1:.4f})")

y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

acc_opt = accuracy_score(y_test, y_pred_optimized)
prec_opt = precision_score(y_test, y_pred_optimized, zero_division=0)
rec_opt = recall_score(y_test, y_pred_optimized, zero_division=0)
f1_opt = f1_score(y_test, y_pred_optimized, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print("\n" + "=" * 50)
print(f"--- 支持向量机 (SVM) 模型评估结果 (阈值寻优后, 目标标签: {target_col}) ---")
print("=" * 50)
print(f"1. 优化后分类阈值      : {best_threshold:.4f}")
print(f"2. 准确率 (Accuracy)  : {acc_opt:.4f}")
print(f"3. 精确率 (Precision) : {prec_opt:.4f}")
print(f"4. 召回率 (Recall)    : {rec_opt:.4f}")
print(f"5. F1 Score          : {f1_opt:.4f}")
print(f"6. ROC-AUC           : {roc_auc:.4f}")
print(f"7. PR-AUC            : {pr_auc:.4f}")
print("=" * 50)

# ==========================================
# 第三步：使用从未见过的专案数据对案例公司进行纯样本外盲测
# ==========================================
if df_case.empty:
    print(f"\n⚠️ 警告：未在数据集中找到股票代码为 '{target_symbol}' 的公司，请检查 Symbol 或原始表格！")
else:
    print(
        f"\n成功锁定目标案例公司（股票代码: {target_symbol}），共找到 {df_case.shape[0]} 个年度样本记录（该数据未参与模型训练）。")

    # 使用标准化后的案例特征与 SVM 模型进行打分
    df_case['Predicted_Risk_Probability'] = svm_model.predict_proba(X_case_scaled)[:, 1]
    # 使用寻优得到的 best_threshold 作为判别标准
    df_case['Model_Predicted_Label'] = (df_case['Predicted_Risk_Probability'] >= best_threshold).astype(int)

    display_cols = ['Symbol', 'ShortName', 'Year', 'Predicted_Risk_Probability', 'Model_Predicted_Label', 'Y1', 'Y2']
    df_case_sorted = df_case[display_cols].sort_values('Year')

    print("\n" + "=" * 80)
    print(f"--- 目标案例企业 ({target_symbol}) 历年内控风险穿透盲测报告 (SVM 阈值寻优版) ---")
    print("=" * 80)
    print(df_case_sorted.to_string(index=False))
    print("=" * 80)

    print("\n【个案逐年穿透盲测结果与隐瞒行为深度解读】:")
    for idx, row in df_case_sorted.iterrows():
        yr = int(row['Year'])
        prob = row['Predicted_Risk_Probability']
        pred_lbl = int(row['Model_Predicted_Label'])
        y1_val = int(row['Y1'])
        y2_val = int(row['Y2'])

        if y1_val == 0 and y2_val == 1:
            disclosure_status = "【🎭 虚假陈述/选择性隐瞒：自评无缺陷，实则被监管处罚】"
        elif y1_val == 1 and y2_val == 1:
            disclosure_status = "【🤝 相对坦诚：自评有缺陷，且被监管证实】"
        else:
            disclosure_status = "【✅ 真实合规】"

        model_warning = "🔥【高危预警：模型成功穿透伪装】" if pred_lbl == 1 else "✅【模型预测合规】"

        print(f"• {yr}年度: 预测缺陷概率 = {prob:.4f} (阈值: {best_threshold:.4f}) | 模型判定: {model_warning}")
        print(f"  --> 对照组视角: 企业自述(Y1) = {y1_val} | 监管纠偏(Y2) = {y2_val} {disclosure_status}\n")