import os
import tempfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 【关键修复】解决 Windows 下路径含中文导致的 joblib 编码报错
os.environ['JOBLIB_TEMP_FOLDER'] = tempfile.gettempdir()
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
safe_temp_dir = os.path.join(current_dir, 'joblib_temp')
os.makedirs(safe_temp_dir, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = safe_temp_dir

# 1. 读取双表头 Excel 数据文件
file_path = './data/C33_C40_normalized.xlsx'
df_raw = pd.read_excel(file_path, header=None)

# 提取双表头与正文数据
header_en = list(df_raw.iloc[0].values)
df = df_raw.iloc[2:].copy()
df.columns = header_en

# 确保 Code 列格式规范
if 'Code' in df.columns:
    df['Code'] = df['Code'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
else:
    raise ValueError("表格中未找到 'Code'（股票代码）列，请检查列名。")

# 2. 确定特征变量 (X) 与标签变量 (Y)
feature_cols = df.columns[4:24]
target_col = 'Y'

print(f"选用的特征指标数量: {len(feature_cols)} 个（E至X列）")

# 转换为数值型并去除缺失值行
cols_to_keep = list(feature_cols) + [target_col, 'Code', 'EndDate']
df_clean = df[cols_to_keep].copy()
for col in list(feature_cols) + [target_col]:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna(subset=list(feature_cols) + [target_col])

# 排除独立案例（如 603169），保持与前文一致的建模池
case_code = '603169'
df_model_pool = df_clean[df_clean['Code'] != case_code]

X = df_model_pool[feature_cols]
y = df_model_pool[target_col].astype(int)

# 3. 样本划分（分层随机抽样，按 8:2 比例划分训练集与测试集）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. 样本不平衡处理：使用 SMOTE 过采样处理训练集
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. 逻辑回归模型参数设置与寻优（网格搜索 + 5折交叉验证）
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced']
}

# 实例化逻辑回归分类器
lr_clf = LogisticRegression(
    random_state=42,
    max_iter=1000
)

grid_search_lr = GridSearchCV(
    estimator=lr_clf,
    param_grid=param_grid_lr,
    cv=5,
    scoring='roc_auc',
    n_jobs=1
)

print("\n正在进行逻辑回归（Logistic Regression）网格搜索与 5 折交叉验证调参...")
grid_search_lr.fit(X_train_resampled, y_train_resampled)

# 获取最优模型与参数
best_lr_model = grid_search_lr.best_estimator_

# 6. 在常规测试集上进行模型评估
y_pred = best_lr_model.predict(X_test)
y_pred_proba = best_lr_model.predict_proba(X_test)[:, 1]

# 保存测试集真实标签 y_test 及预测概率 y_pred_proba 到 lr.xlsx
model_evaluation_results = pd.DataFrame({
    'y_true': y_test.values,
    'y_pred_proba': y_pred_proba
})
model_evaluation_results.to_excel("lr.xlsx", index=False)
print("✅ 测试集真实标签 y_true 及预测概率 y_pred_proba 已成功保存至: 'lr.xlsx'")

test_auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# ==================== 按照指定格式输出结果 ====================
print(f"\n[逻辑回归模型寻优结果]")
print(f"最优超参数组合：{grid_search_lr.best_params_}")
print(f"5折交叉验证 ROC-AUC 均值：{grid_search_lr.best_score_:.4f}")

print(f"\n[常规测试集评估表现]")
print(f"测试集 ROC-AUC: {test_auc:.4f}")
print(f"----------------------------------------")
print(f"准确率 (Accuracy):   {acc * 100:.2f}%")
print(f"精确率 (Precision):  {precision * 100:.2f}%")
print(f"召回率 (Recall):     {recall * 100:.2f}%")
print(f"F1 分数 (F1-Score):  {f1 * 100:.2f}%")
print(f"----------------------------------------")
print(f"[混淆矩阵 (Confusion Matrix)]")
print(f"                 预测: 0 (无缺陷)    预测: 1 (有缺陷)")
print(f"真实: 0 (无缺陷)      {tn:<15} {fp:<15}")
print(f"真实: 1 (有缺陷)      {fn:<15} {tp:<15}")
print(f"----------------------------------------")