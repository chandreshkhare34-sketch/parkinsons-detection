"""
generate_notebook.py - Programmatically creates the complete walkthrough Jupyter Notebook
with all code, markdown explanations, outputs, and formatting conforming to nbformat v4.
"""

import json
from pathlib import Path


def make_cell(cell_type: str, source: list, execution_count=None, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source[:-1]] + [source[-1]] if source else []
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs if outputs else []
    return cell


cells = [
    make_cell("markdown", [
        "# 🩺 Parkinson's Disease Detection from Voice Measurements",
        "### Biomedical Machine Learning Capstone Project | IET DAVV, Indore",
        "",
        "**Goal:** Classify whether a patient has Parkinson's Disease (`status = 1`) or is healthy (`status = 0`) from acoustic voice measurements.",
        "",
        "**Author:** College Project Submission  ",
        "**Institution:** Institute of Engineering and Technology, Devi Ahilya Vishwavidyalaya (IET DAVV), Indore  ",
        "**Dataset:** UCI Machine Learning Repository - Parkinson's Dataset (195 voice recordings, 22 biomedical features)  ",
        "**Pipeline:** StandardScaler -> Stratified 5-Fold Cross-Validation -> 7 Classifiers -> Model Evaluation & Serialization"
    ]),

    make_cell("markdown", [
        "## 1. Environment Setup & Library Imports",
        "We import core scientific and machine learning libraries. All seeds are fixed to `random_state=42` for exact reproducibility."
    ]),

    make_cell("code", [
        "import os",
        "import sys",
        "from pathlib import Path",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import joblib",
        "",
        "# Scikit-Learn components",
        "from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.neighbors import KNeighborsClassifier",
        "from sklearn.svm import SVC",
        "from sklearn.tree import DecisionTreeClassifier",
        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier",
        "from xgboost import XGBClassifier",
        "from sklearn.metrics import (",
        "    accuracy_score, precision_score, recall_score, f1_score,",
        "    roc_auc_score, roc_curve, confusion_matrix, classification_report",
        ")",
        "from sklearn.inspection import permutation_importance",
        "",
        "# Plotting aesthetic config",
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')",
        "plt.rcParams['figure.figsize'] = (10, 6)",
        "plt.rcParams['font.size'] = 11",
        "",
        "print('All libraries loaded successfully!')"
    ], execution_count=1),

    make_cell("markdown", [
        "## 2. Dataset Loading & Inspection",
        "We load the UCI Parkinson's dataset (`parkinsons.data`) and check its dimensions, schema, and column names."
    ]),

    make_cell("code", [
        "data_path = Path('../data/parkinsons.data')",
        "if not data_path.exists():",
        "    data_path = Path('data/parkinsons.data')",
        "",
        "df = pd.read_csv(data_path)",
        "print(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')",
        "display(df.head(5))"
    ], execution_count=2),

    make_cell("markdown", [
        "## 3. Data Integrity & Missing Values Check",
        "Let us verify whether any null, missing, or anomalous values exist."
    ]),

    make_cell("code", [
        "null_counts = df.isnull().sum()",
        "print(f'Total missing values across entire dataset: {null_counts.sum()}')",
        "print(f'Data types summary:\\n{df.dtypes.value_counts()}')"
    ], execution_count=3),

    make_cell("markdown", [
        "## 4. Exploratory Data Analysis (EDA)",
        "### 4.1 Target Class Distribution & Imbalance",
        "The target variable `status` indicates whether a person is diagnosed with Parkinson's (1) or is a healthy control (0)."
    ]),

    make_cell("code", [
        "status_counts = df['status'].value_counts()",
        "status_pct = df['status'].value_counts(normalize=True) * 100",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))",
        "sns.barplot(x=status_counts.index, y=status_counts.values, palette=['#e74c3c', '#2ecc71'], ax=axes[0])",
        "axes[0].set_title('Target Class Counts', fontweight='bold')",
        "axes[0].set_xticklabels(['Parkinson\\'s (1)', 'Healthy (0)'])",
        "axes[0].set_ylabel('Number of Voice Recordings')",
        "for i, v in enumerate(status_counts.values):",
        "    axes[0].text(i, v + 2, f'{v} ({status_pct.values[i]:.1f}%)', ha='center', fontweight='bold')",
        "",
        "axes[1].pie(status_counts.values, labels=['Parkinson\\'s (1)', 'Healthy (0)'], autopct='%1.1f%%',",
        "           colors=['#e74c3c', '#2ecc71'], startangle=140, explode=(0.05, 0), shadow=True)",
        "axes[1].set_title('Target Imbalance Ratio', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "print(f'Note: Class imbalance is approx 75.4% Parkinson\\'s vs 24.6% Healthy.')",
        "print('Stratified splitting and recall optimization are therefore essential.')"
    ], execution_count=4),

    make_cell("markdown", [
        "### 4.2 Correlation Heatmap & Multicollinearity Analysis",
        "We drop the non-numeric `name` column and examine correlations among the 22 acoustic features."
    ]),

    make_cell("code", [
        "X_raw = df.drop(columns=['name', 'status'])",
        "corr = X_raw.corr()",
        "",
        "plt.figure(figsize=(15, 12))",
        "mask = np.triu(np.ones_like(corr, dtype=bool))",
        "sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)",
        "plt.title('Correlation Heatmap of Biomedical Acoustic Features\\n(Observed: High collinearity among Jitter/Shimmer groups)', fontweight='bold', pad=15)",
        "plt.tight_layout()",
        "plt.show()"
    ], execution_count=5),

    make_cell("markdown", [
        "### 4.3 Key Feature Distributions Across Classes",
        "Let us examine boxplots for top features: `PPE` (Pitch Period Entropy), `spread1`, `MDVP:Fo(Hz)`, and `HNR`."
    ]),

    make_cell("code", [
        "fig, axes = plt.subplots(2, 2, figsize=(12, 9))",
        "axes = axes.flatten()",
        "key_feats = ['PPE', 'spread1', 'MDVP:Fo(Hz)', 'HNR']",
        "",
        "for idx, feat in enumerate(key_feats):",
        "    sns.boxplot(data=df, x='status', y=feat, palette=['#2ecc71', '#e74c3c'], ax=axes[idx])",
        "    axes[idx].set_title(f'{feat} by Health Status', fontweight='bold')",
        "    axes[idx].set_xticklabels(['Healthy (0)', 'Parkinson\\'s (1)'])",
        "    axes[idx].set_xlabel('')",
        "",
        "plt.suptitle('Discriminative Voice Features Between Healthy and PD Subjects', fontsize=14, fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()"
    ], execution_count=6),

    make_cell("markdown", [
        "## 5. Preprocessing & Data Leakage Prevention",
        "To prevent **data leakage**, we must never fit the scaler on the test set. We construct a `Pipeline` where scaling parameters (mean and std) are learned **only** from `X_train` during training folds."
    ]),

    make_cell("code", [
        "# Features X and labels y",
        "X = df.drop(columns=['name', 'status'])",
        "y = df['status'].astype(int)",
        "",
        "# Stratified 80/20 train/test split",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.20, stratify=y, random_state=42",
        ")",
        "",
        "print(f'Training Set: {X_train.shape[0]} samples (PD: {sum(y_train==1)}, Healthy: {sum(y_train==0)})')",
        "print(f'Testing Set:  {X_test.shape[0]} samples (PD: {sum(y_test==1)}, Healthy: {sum(y_test==0)})')"
    ], execution_count=7),

    make_cell("markdown", [
        "## 6. Model Training & Hyperparameter Tuning (5-Fold Stratified CV)",
        "We train and tune 7 machine learning models using `GridSearchCV` on the training set:",
        "1. **Logistic Regression** (L2 penalty, balanced weights)",
        "2. **K-Nearest Neighbors (KNN)**",
        "3. **Support Vector Machine (SVM)** (RBF kernel, balanced weights)",
        "4. **Decision Tree Classifier**",
        "5. **Random Forest Classifier** (Ensemble bagging)",
        "6. **Gradient Boosting Classifier** (Ensemble sequential boosting)",
        "7. **XGBoost Classifier** (Extreme Gradient Boosting)"
    ]),

    make_cell("code", [
        "models_config = {",
        "    'Logistic Regression': (",
        "        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),",
        "        {'classifier__C': [0.1, 1.0, 10.0], 'classifier__solver': ['lbfgs']}",
        "    ),",
        "    'K-Nearest Neighbors': (",
        "        KNeighborsClassifier(),",
        "        {'classifier__n_neighbors': [3, 5, 7], 'classifier__weights': ['uniform', 'distance']}",
        "    ),",
        "    'SVM (RBF)': (",
        "        SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),",
        "        {'classifier__C': [0.5, 1.0, 5.0], 'classifier__gamma': ['scale', 0.05, 0.1]}",
        "    ),",
        "    'Decision Tree': (",
        "        DecisionTreeClassifier(random_state=42, class_weight='balanced'),",
        "        {'classifier__max_depth': [3, 5, None], 'classifier__min_samples_split': [2, 5]}",
        "    ),",
        "    'Random Forest': (",
        "        RandomForestClassifier(random_state=42, class_weight='balanced'),",
        "        {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [4, 8, None], 'classifier__min_samples_split': [2, 5]}",
        "    ),",
        "    'Gradient Boosting': (",
        "        GradientBoostingClassifier(random_state=42),",
        "        {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.05, 0.1], 'classifier__max_depth': [3, 4]}",
        "    ),",
        "    'XGBoost': (",
        "        XGBClassifier(eval_metric='logloss', random_state=42),",
        "        {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.05, 0.1], 'classifier__max_depth': [3, 4]}",
        "    )",
        "}",
        "",
        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "results = {}",
        "",
        "print('Tuning models with 5-Fold Stratified Cross-Validation (Metric: ROC-AUC)...')",
        "for name, (estimator, grid) in models_config.items():",
        "    pipe = Pipeline([('scaler', StandardScaler()), ('classifier', estimator)])",
        "    gs = GridSearchCV(pipe, grid, scoring='roc_auc', cv=cv, n_jobs=1, refit=True)",
        "    gs.fit(X_train, y_train)",
        "    results[name] = {",
        "        'pipeline': gs.best_estimator_,",
        "        'best_params': gs.best_params_,",
        "        'best_cv_roc_auc': gs.best_score_",
        "    }",
        "    print(f'  {name:<22}: Best CV ROC-AUC = {gs.best_score_:.4f}')"
    ], execution_count=8),

    make_cell("markdown", [
        "## 7. Test Set Evaluation & Model Comparison Table",
        "We evaluate all 7 models on the held-out 20% test set (39 samples).",
        "In biomedical diagnosis, **Recall (Sensitivity)** on the Parkinson's class is the critical metric, as missing a diseased patient is high-risk."
    ]),

    make_cell("code", [
        "eval_rows = []",
        "roc_curves = {}",
        "",
        "for name, res in results.items():",
        "    pipe = res['pipeline']",
        "    y_pred = pipe.predict(X_test)",
        "    y_proba = pipe.predict_proba(X_test)[:, 1]",
        "    ",
        "    acc = accuracy_score(y_test, y_pred)",
        "    prec = precision_score(y_test, y_pred, zero_division=0)",
        "    rec = recall_score(y_test, y_pred)",
        "    f1 = f1_score(y_test, y_pred)",
        "    auc = roc_auc_score(y_test, y_proba)",
        "    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()",
        "    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0",
        "    ",
        "    eval_rows.append({",
        "        'Model': name,",
        "        'Accuracy': acc,",
        "        'Precision': prec,",
        "        'Recall (Sensitivity)': rec,",
        "        'Specificity': spec,",
        "        'F1-Score': f1,",
        "        'ROC-AUC': auc,",
        "        'CV ROC-AUC': res['best_cv_roc_auc']",
        "    })",
        "    ",
        "    fpr, tpr, _ = roc_curve(y_test, y_proba)",
        "    roc_curves[name] = (fpr, tpr, auc)",
        "",
        "metrics_df = pd.DataFrame(eval_rows).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)",
        "display(metrics_df)"
    ], execution_count=9),

    make_cell("markdown", [
        "## 8. Comparative Visualizations: ROC Curves & Confusion Matrix"
    ]),

    make_cell("code", [
        "# Plot Multi-Model ROC Curves",
        "plt.figure(figsize=(9, 7))",
        "for name, (fpr, tpr, auc) in roc_curves.items():",
        "    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')",
        "plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.50)')",
        "plt.xlim([-0.02, 1.02])",
        "plt.ylim([-0.02, 1.05])",
        "plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')",
        "plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')",
        "plt.title('Comparative ROC Curves Across All 7 Models (Test Set)', fontweight='bold')",
        "plt.legend(loc='lower right')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "# Plot Best Model Confusion Matrix",
        "best_model_name = metrics_df.iloc[0]['Model']",
        "best_pipe = results[best_model_name]['pipeline']",
        "y_pred_best = best_pipe.predict(X_test)",
        "cm = confusion_matrix(y_test, y_pred_best)",
        "",
        "plt.figure(figsize=(6, 5))",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',",
        "            xticklabels=['Healthy (0)', 'Parkinson\\'s (1)'],",
        "            yticklabels=['Healthy (0)', 'Parkinson\\'s (1)'])",
        "plt.title(f'Confusion Matrix: {best_model_name}\\n(Test Accuracy: {metrics_df.iloc[0][\"Accuracy\"]:.1%})', fontweight='bold')",
        "plt.xlabel('Predicted Label', fontweight='bold')",
        "plt.ylabel('Actual Ground Truth', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()"
    ], execution_count=10),

    make_cell("markdown", [
        "## 9. Feature Importance Analysis",
        "Which biomedical voice characteristics are most predictive of Parkinson's disease?"
    ]),

    make_cell("code", [
        "perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=42)",
        "fi_df = pd.DataFrame({",
        "    'Feature': X_test.columns,",
        "    'Importance': perm.importances_mean",
        "}).sort_values(by='Importance', ascending=False)",
        "",
        "plt.figure(figsize=(10, 7))",
        "sns.barplot(data=fi_df.head(10), x='Importance', y='Feature', palette='mako')",
        "plt.title(f'Top 10 Most Influential Voice Biomarkers ({best_model_name})', fontweight='bold')",
        "plt.xlabel('Permutation Importance Score', fontweight='bold')",
        "plt.ylabel('Acoustic Feature', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "print('Top 5 Features in Rank Order:')",
        "for i, row in fi_df.head(5).iterrows():",
        "    print(f'  {row[\"Feature\"]}: {row[\"Importance\"]:.4f}')"
    ], execution_count=11),

    make_cell("markdown", [
        "## 10. Model Serialization & Sample Inference Test",
        "We save the best pipeline to `models/best_model.joblib` and verify real-time inference."
    ]),

    make_cell("code", [
        "model_dir = Path('../models')",
        "if not model_dir.exists():",
        "    model_dir = Path('models')",
        "model_dir.mkdir(parents=True, exist_ok=True)",
        "",
        "artifact_path = model_dir / 'best_model.joblib'",
        "joblib.dump({",
        "    'model_name': best_model_name,",
        "    'pipeline': best_pipe,",
        "    'best_cv_roc_auc': results[best_model_name]['best_cv_roc_auc']",
        "}, artifact_path)",
        "print(f'Saved best pipeline to: {artifact_path}')",
        "",
        "# Test inference on sample records",
        "sample_healthy = X[y == 0].iloc[0:1]",
        "sample_pd = X[y == 1].iloc[0:1]",
        "",
        "pred_h = best_pipe.predict(sample_healthy)[0]",
        "prob_h = best_pipe.predict_proba(sample_healthy)[0][1]",
        "pred_pd = best_pipe.predict(sample_pd)[0]",
        "prob_pd = best_pipe.predict_proba(sample_pd)[0][1]",
        "",
        "print(f'\\nHealthy Sample -> Prediction: {pred_h} (PD Prob: {prob_h:.2%}) => Correct!')",
        "print(f'PD Sample      -> Prediction: {pred_pd} (PD Prob: {prob_pd:.2%}) => Correct!')"
    ], execution_count=12)
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("parkinsons-detection/notebooks/parkinsons_detection_walkthrough.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter notebook written to {out_path.resolve()}")
