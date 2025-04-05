import yaml
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, roc_auc_score
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def select_features(X, y, var_thresh=0.0, k_best=100, corr_thresh=0.95):
    # Step 1: 低方差筛选
    vt = VarianceThreshold(threshold=var_thresh)
    X_vt = vt.fit_transform(X)
    selected_var = X.columns[vt.get_support()]

    # Step 2: 卡方筛选
    chi = SelectKBest(score_func=chi2, k=min(k_best, len(selected_var)))
    X_chi = chi.fit_transform(X[selected_var], y)
    selected_chi = selected_var[chi.get_support()]

    # Step 3: 去除高相关性特征
    df_corr = X[selected_chi].corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    final_features = selected_chi.drop(to_drop)

    return X[final_features], final_features


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 模型配置
models = {
    'MultinomialNB': MultinomialNB(),
    'DecisionTree': DecisionTreeClassifier(criterion=config['model']['decision_tree']['criterion']),
    'RandomForest': RandomForestClassifier(n_estimators=config['model']['random_forest']['n_estimators']),
    'SVM': SVC(probability=True)
}

# 加载数据
df_train = pd.read_csv(config['dataset']['training_data_path'])
df_test = pd.read_csv(config['dataset']['test_data_path'])

X_train = df_train.iloc[:, :-2]
y_train = df_train['prognosis']

X_test = df_test.iloc[:, :-1]
y_test = df_test['prognosis']

# 特征选择
X_train_selected, selected_features = select_features(X_train, y_train, var_thresh=0.0, k_best=100, corr_thresh=0.95)
X_test_selected = X_test[selected_features]

# 多分类标签二值化（用于ROC AUC）
classes = sorted(list(set(y_train)))
y_test_bin = label_binarize(y_test, classes=classes)

# 存储结果
results = []

for model_name, model in models.items():
    print(f"\n Training and Evaluating: {model_name}")

    # 模型训练
    model.fit(X_train_selected, y_train)
    dump(model, f"{config['model_save_path']}{model_name}.joblib")

    # 预测
    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)

    # 指标计算
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')

    results.append((model_name, acc, prec, rec, f1, roc_auc))

# 结果展示
df_result = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
print("\n Model Performance on Selected Features:\n")
print(df_result.to_string(index=False))
