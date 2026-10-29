import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
header_cn = list(df_raw.iloc[1].values)
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

# 转换为数值型并去除缺失值行（保留 Code 和 EndDate 列以便后续独立提取与打印年份）
cols_to_keep = list(feature_cols) + [target_col, 'Code', 'EndDate']
df_clean = df[cols_to_keep].copy()
for col in list(feature_cols) + [target_col]:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna(subset=list(feature_cols) + [target_col])

# ==================== 提取独立案例 (Code == '603169') ====================
case_code = '603169'
df_case = df_clean[df_clean['Code'] == case_code]
# 从主体数据中剔除该独立案例，使其完全不参与训练与常规测试
df_model_pool = df_clean[df_clean['Code'] != case_code]

print(f"有效样本总量（剔除独立案例后）: {len(df_model_pool)} 条")
print(f"独立测试案例 (Code: {case_code}) 提取数量: {len(df_case)} 条")

X = df_model_pool[feature_cols]
y = df_model_pool[target_col].astype(int)

# 3. 样本划分（分层随机抽样，按 8:2 比例划分训练集与测试集）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 保持训练集与测试集中内部控制重大缺陷样本比例一致
)

print(f"\n训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
print(f"训练集中缺陷样本占比: {y_train.mean():.4f}")

# 4. 样本不平衡处理：使用 SMOTE 过采样处理训练集
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"SMOTE 过采样后的训练集样本数: {len(X_train_resampled)}")

# 5. 直接使用表 5-13 中的最优超参数实例化并训练随机森林模型
best_hyperparams = {
    'n_estimators': 300,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

print(f"\n正在使用设定参数训练随机森林模型...")
best_rf_model = RandomForestClassifier(
    random_state=42,
    **best_hyperparams
)

# 直接在过采样训练集上拟合模型
best_rf_model.fit(X_train_resampled, y_train_resampled)

print(f"\n[随机森林模型设定参数]")
print(f"设定最优超参数组合：{best_hyperparams}")

# 6. 在常规测试集上进行模型评估（引入自定义阈值优化召回率）
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]

# 【核心优化】自定义分类阈值设为 0.33
threshold = 0.33
y_pred = (y_pred_proba >= threshold).astype(int)

# 保存测试集真实标签 y_test 及预测概率 y_pred_proba 到 ./data/rf.xlsx
os.makedirs('./data', exist_ok=True)
model_evaluation_results = pd.DataFrame({
    'y_true': y_test.values,
    'y_pred_proba': y_pred_proba
})
model_evaluation_results.to_excel("./data/rf.xlsx", index=False)
print("✅ 测试集真实标签 y_true 及预测概率 y_pred_proba 已成功保存至: './data/rf.xlsx'")

test_auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n[常规测试集评估表现 (判定阈值: {threshold})]")
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

# ============================================================
# 【模块一】全局 SHAP 特征解释分析
# ============================================================
print("\n" + "=" * 50)
print("【全局】SHAP 特征解释分析")
print("=" * 50)

# 特征标签中英文对照表
FEATURE_CN_MAP = {
    'X02': '股权性质', 'X03': '董事会规模', 'X04': '股权制衡度',
    'X07': '机构投资者持股比例', 'X08': '高管薪酬', 'X09': '总资产收益率',
    'X12': '总资产增长率', 'X13': '净利润增长率', 'X14': '营业收入增长率',
    'X15': '应收账款周转率', 'X16': '应付账款周转率', 'X17': '固定资产周转率',
    'X18': '总资产周转率', 'X21': '速动比率', 'X22': '权益乘数',
    'X23': '托宾Q值', 'X24': '上市年限', 'X25': '是否变更审计机构',
    'X26': '审计费用', 'X27': '是否四大审计',
    # 兼容不带 0 补位的写法
    'X2': '股权性质', 'X3': '董事会规模', 'X4': '股权制衡度',
    'X7': '机构投资者持股比例', 'X8': '高管薪酬', 'X9': '总资产收益率',
    'X12': '总资产增长率', 'X13': '净利润增长率', 'X14': '营业收入增长率',
    'X15': '应收账款周转率', 'X16': '应付账款周转率', 'X17': '固定资产周转率',
    'X18': '总资产周转率', 'X21': '速动比率', 'X22': '权益乘数',
    'X23': '托宾Q值', 'X24': '上市年限', 'X25': '是否变更审计机构',
    'X26': '审计费用', 'X27': '是否四大审计',
    # 英文简称兼容
    'SOE': '股权性质', 'Board': '董事会规模', 'Balance3': '股权制衡度',
    'INST': '机构投资者持股比例', 'TMTPay2': '高管薪酬', 'ROA': '总资产收益率',
    'AssetGrowth': '总资产增长率', 'NetProfitGrowth': '净利润增长率', 'Growth': '营业收入增长率',
    'ART': '应收账款周转率', 'APT': '应付账款周转率', 'FAT': '固定资产周转率',
    'ATO': '总资产周转率', 'Quick': '速动比率', 'EM1': '权益乘数',
    'TobinQ': '托宾Q值', 'ListAge': '上市年限', 'AuditSame': '是否变更审计机构',
    'AuditFee': '审计费用', 'Big4': '是否四大审计',
}


def _to_cn_label(col):
    key = str(col).strip()
    if key in FEATURE_CN_MAP:
        return FEATURE_CN_MAP[key]
    for k, v in FEATURE_CN_MAP.items():
        if k.lower() == key.lower():
            return v
    upper = key.upper()
    for a in range(1, 28):
        tag1 = f'X{a:02d}'
        tag2 = f'X{a}'
        if upper.startswith(tag1) or upper.startswith(tag2):
            return FEATURE_CN_MAP.get(tag1, FEATURE_CN_MAP.get(tag2, key))
    return key


feature_names_cn = [_to_cn_label(c) for c in feature_cols]

explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    sv_pos = shap_values[1]
elif shap_values.ndim == 3:
    sv_pos = shap_values[..., 1]
else:
    sv_pos = shap_values

# 全局图1：全局特征重要性条形图（横向长方形画布）
shap.summary_plot(sv_pos, X_test, feature_names=feature_names_cn,
                  plot_type="bar", show=False)
plt.gcf().set_size_inches(8, 6)
plt.tight_layout()
plt.savefig('shap_importance_bar.png', dpi=600, bbox_inches='tight')
plt.close()

# 全局图2：全局蜜蜂图（X轴已添加 impact on model output）
plt.figure(figsize=(9, 6))
shap.summary_plot(sv_pos, X_test, feature_names=feature_names_cn, show=False)
ax = plt.gca()
ax.set_xlabel('SHAP 值 (impact on model output)', fontsize=11)
ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4)
cb_ax = plt.gcf().axes[-1]
cb_ax.set_ylabel('特征取值', fontsize=11)
for _lab in cb_ax.get_yticklabels():
    _txt = _lab.get_text()
    if _txt == 'High':
        _lab.set_text('高')
    elif _txt == 'Low':
        _lab.set_text('低')
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=600, bbox_inches='tight')
plt.close()

# ============================================================
# 【模块二】独立案例（603169）预测与专属 SHAP 特征解释分析
# ============================================================
print(f"\n==========================================")
print(f"【独立测试案例预测与专属解释 (Code: {case_code})】")
print(f"==========================================")

if len(df_case) > 0:
    X_case = df_case[feature_cols]
    y_case_true = df_case[target_col].values

    case_proba = best_rf_model.predict_proba(X_case)[:, 1]
    case_pred = (case_proba >= threshold).astype(int)

    df_case_sorted_print = df_case.sort_values(by='EndDate')

    for index, row in df_case_sorted_print.iterrows():
        year_val = row['EndDate']
        idx_pos = list(df_case.index).index(index)
        p_val = case_proba[idx_pos]
        pred_val = case_pred[idx_pos]
        true_val = y_case_true[idx_pos]
        confidence = abs(p_val - threshold) / max(threshold, 1 - threshold) * 100

        print(f"年份 (EndDate): {year_val}")
        print(f"  - 真实标签 (True Label): {int(true_val)}")
        print(f"  - 预测概率 (predict_proba): {p_val:.4f} (即 {p_val * 100:.2f}%)")
        print(
            f"  - 预测结果 (predict):       {pred_val} ({'倾向正类-内部控制重大缺陷' if pred_val == 1 else '倾向负类-无重大缺陷'})")
        print(f"  - 判定阈值:               {threshold}")
        print(f"----------------------------------------")

    # ── 案例公司（603169）的专属 SHAP 特征解释图 ──
    print(f"\n正在生成案例公司 ({case_code}) 的专属分色 SHAP 特征解释图...")
    shap_values_case = explainer.shap_values(X_case)
    if isinstance(shap_values_case, list):
        sv_case_pos = shap_values_case[1]
    elif shap_values_case.ndim == 3:
        sv_case_pos = shap_values_case[..., 1]
    else:
        sv_case_pos = shap_values_case

    mean_abs_shap = np.mean(np.abs(sv_case_pos), axis=0)

    case_importance_df = pd.DataFrame({
        'feature_en': feature_cols,
        'feature_cn': feature_names_cn,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=True)


    def get_feature_color(col_name):
        c_upper = str(col_name).strip().upper()
        # 蓝色组：股权/治理 (X02, X03, X04, X07, X08)
        blue_set = {'X02', 'X03', 'X04', 'X07', 'X08', 'X2', 'X3', 'X4', 'X7', 'X8', 'SOE', 'BOARD', 'BALANCE3', 'INST',
                    'TMTPAY2'}
        # 红色组：财务/盈利/偿债/杠杆 (X09, X12-X18, X21, X22)
        red_set = {'X09', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X21', 'X22',
                   'X9', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X21', 'X22',
                   'ROA', 'ASSETGROWTH', 'NETPROFITGROWTH', 'GROWTH', 'ART', 'APT', 'FAT', 'ATO', 'QUICK', 'EM1'}

        if c_upper in blue_set:
            return '#1f77b4'  # 蓝色
        elif c_upper in red_set:
            return '#d62728'  # 红色
        else:
            return '#ff7f0e'  # 橙色


    bar_colors = [get_feature_color(f) for f in case_importance_df['feature_en']]

    # ── 案例专属图1：定制分色条形图（全包围矩形框坐标轴，右下角图例，600 DPI）──
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.barh(case_importance_df['feature_cn'], case_importance_df['importance'], color=bar_colors)
    plt.xlabel('mean(|SHAP value|) ', fontsize=11)

    # 设置坐标轴为全包围长方形框线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')

    # 右下角图例配置
    legend_elements = [
        Patch(facecolor='#1f77b4', label='内部治理制度维度指标'),
        Patch(facecolor='#d62728', label='财务表现维度指标'),
        Patch(facecolor='#ff7f0e', label='外部环境风险维度指标')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig('case_603169_shap_bar.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("✅ 案例专属分色条形图已保存至 -> 'case_603169_shap_bar.png'")

    # ── 案例专属图2：案例蜜蜂图（600 DPI）──
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv_case_pos, X_case, feature_names=feature_names_cn, show=False)
    ax = plt.gca()
    ax.set_xlabel('SHAP 值', fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4)
    cb_ax = plt.gcf().axes[-1]
    cb_ax.set_ylabel('特征取值', fontsize=11)
    for _lab in cb_ax.get_yticklabels():
        _txt = _lab.get_text()
        if _txt == 'High':
            _lab.set_text('高')
        elif _txt == 'Low':
            _lab.set_text('低')
    plt.tight_layout()
    plt.savefig('case_603169_shap_beeswarm.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("✅ 案例专属蜜蜂图已保存至 -> 'case_603169_shap_beeswarm.png'")

    # ── 案例专属图3：2020年度 SHAP 瀑布图（600 DPI，终极修复负号显示问题）──
    print(f"\n正在提取并生成案例公司 ({case_code}) 2020 年的 SHAP 瀑布图...")
    df_case_2020 = df_case[df_case['EndDate'].astype(str).str.contains('2020')]

    if len(df_case_2020) > 0:
        idx_2020 = df_case.index.get_loc(df_case_2020.index[0])
        exp_val = explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                            (list, np.ndarray)) else explainer.expected_value

        shap_exp = shap.Explanation(
            values=sv_case_pos[idx_2020],
            base_values=exp_val,
            data=X_case.iloc[idx_2020].values,
            feature_names=feature_names_cn
        )

        import matplotlib as mpl

        mpl.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(10, 6))

        # 绘制瀑布图
        shap.plots.waterfall(shap_exp, show=False)

        # 【终极补丁】遍历当前画布文本对象，将所有包含负号的数字文本动态设为西文字体（解决 SimHei 缺字形导致的负号空白）
        for txt in plt.gcf().findobj(match=lambda x: hasattr(x, 'get_text') and x.get_text()):
            try:
                current_text = txt.get_text()
                if '-' in current_text or '−' in current_text:
                    txt.set_fontfamily('DejaVu Sans')
            except:
                pass

        plt.tight_layout()
        plt.savefig('case_603169_2020_waterfall.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("✅ 案例公司 2020 年 SHAP 瀑布图已成功保存至 -> 'case_603169_2020_waterfall.png'")
    else:
        print(f"⚠️ 未找到 EndDate 为 2020 年的数据记录，无法生成 2020 年瀑布图。")

else:
    print(f"未在数据集中找到 Code 为 '{case_code}' 的企业样本，请检查代码或数据。")