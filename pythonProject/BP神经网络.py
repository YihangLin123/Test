import os
import tempfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 【关键修复】解决 Windows 下路径含中文导致的临时文件报错
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

print(f"选用的特征指标数量: {len(feature_cols)} 个")

# 转换为数值型并去除缺失值行
cols_to_keep = list(feature_cols) + [target_col, 'Code', 'EndDate']
df_clean = df[cols_to_keep].copy()
for col in list(feature_cols) + [target_col]:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna(subset=list(feature_cols) + [target_col])

# 排除独立案例（如 603169），保持与前文一致的建模池
case_code = '603169'
df_model_pool = df_clean[df_clean['Code'] != case_code]

X = df_model_pool[feature_cols].values.astype(np.float32)
y = df_model_pool[target_col].astype(int).values

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


# 5. 定义符合您架构的 PyTorch BP 神经网络模型类（输入层 - 双隐含层(64/32+Tanh+Dropout) - 输出层(Sigmoid)）
class CustomBPNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CustomBPNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


# 6. 设置选定的固定超参数
hyperparams = {
    'optimizer': 'adam',
    'lr': 0.01,
    'epochs': 50,
    'batch_size': 32
}

input_dim = X_train_resampled.shape[1]
model = CustomBPNetwork(input_dim)
criterion = nn.BCELoss()  # 二元交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

# 构建 DataLoader
train_dataset = TensorDataset(torch.tensor(X_train_resampled, dtype=torch.float32),
                              torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)

print("\n正在直接使用设定参数训练 BP 神经网络模型...")
model.train()
for epoch in range(hyperparams['epochs']):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# 7. 在常规测试集上进行模型评估
model.eval()
with torch.no_grad():
    y_pred_proba = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

y_pred = (y_pred_proba >= 0.5).astype(int)

# 保存测试集真实标签 y_test 及预测概率 y_pred_proba 到 bp.xlsx
model_evaluation_results = pd.DataFrame({
    'y_true': y_test,
    'y_pred_proba': y_pred_proba
})
model_evaluation_results.to_excel("bp.xlsx", index=False)
print("✅ 测试集真实标签 y_true 及预测概率 y_pred_proba 已成功保存至: 'bp.xlsx'")

test_auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# ==================== 按照指定格式输出结果 ====================
print(f"\n[BP神经网络模型参数设置]")
print(f"设定超参数组合：{hyperparams}")

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