"""
evaluate.py - Test evaluation, metric calculation, comparison tables, and figure generation.

Produces:
1. Complete metrics table (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC)
2. High-resolution figures saved in reports/figures/:
   - 01_class_distribution.png
   - 02_correlation_heatmap.png
   - 03_key_features_boxplots.png
   - 04_roc_curves_comparison.png
   - 05_best_model_confusion_matrix.png
   - 06_feature_importance.png
   - 07_model_metrics_comparison.png
3. Model comparison summary exported to reports/model_comparison.csv
"""

import os
import sys
from pathlib import Path

# Configure Matplotlib writable directory before import
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".matplotlib_cache")

from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

from preprocess import load_dataset, clean_dataset, get_train_test_split, FEATURE_CATEGORIES
from train import train_and_tune_all, select_and_save_best_model

# Set publication-quality plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def plot_eda_figures(df: pd.DataFrame, output_dir: Path):
    """
    Generate and save EDA plots: class distribution, correlation heatmap, key feature boxplots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["status"].value_counts()
    labels = ["Parkinson's (1)", "Healthy (0)"]
    colors = ["#e74c3c", "#2ecc71"]

    # Bar plot
    sns.barplot(x=counts.index, y=counts.values, palette=colors, ax=axes[0], hue=counts.index, legend=False)
    axes[0].set_title("Class Frequency Count", fontweight="bold")
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Number of Voice Recordings")
    for i, count in enumerate(counts.values):
        axes[0].text(i, count + 2, f"{count} ({count/len(df)*100:.1f}%)", ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0.05, 0), shadow=True)
    axes[1].set_title("Class Imbalance Ratio", fontweight="bold")

    plt.suptitle("Dataset Target Distribution (UCI Parkinson's Dataset)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "01_class_distribution.png", bbox_inches="tight")
    plt.close()

    # 2. Correlation Heatmap
    X, _, _ = clean_dataset(df)
    corr = X.corr()
    plt.figure(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.7}, annot=False)
    plt.title("Correlation Heatmap of Biomedical Acoustic Features\n(Notice High Multicollinearity Among Jitter & Shimmer Variants)",
              fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / "02_correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # 3. Key Feature Boxplots
    key_features = ["PPE", "spread1", "MDVP:Fo(Hz)", "HNR"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for idx, feat in enumerate(key_features):
        sns.boxplot(data=df, x="status", y=feat, palette=["#2ecc71", "#e74c3c"], ax=axes[idx], hue="status", legend=False)
        axes[idx].set_title(f"Distribution of {feat} by Health Status", fontweight="bold")
        axes[idx].set_xticklabels(["Healthy (0)", "Parkinson's (1)"])
        axes[idx].set_xlabel("")
    plt.suptitle("Discriminative Power of Top Voice Features", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "03_key_features_boxplots.png", bbox_inches="tight")
    plt.close()


def evaluate_all_models(
    trained_results: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> pd.DataFrame:
    """
    Compute comprehensive metrics on test set for all models and generate comparative plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_records = []
    roc_data = {}

    for name, res in trained_results.items():
        pipeline = res["pipeline"]
        y_pred = pipeline.predict(X_test)

        # Probabilities for class 1
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_scores = pipeline.decision_function(X_test)
            y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Specificity: True Negatives / (True Negatives + False Positives)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall (Sensitivity)": rec,
            "Specificity": spec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "CV ROC-AUC": res["best_cv_score"]
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc, pipeline, y_pred)

    metrics_df = pd.DataFrame(metrics_records).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    # 4. Multi-Model ROC Curves
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(roc_data))
    for i, (name, (fpr, tpr, auc, _, _)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})", color=palette[i])
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    plt.title("Comparative ROC Curves Across All 7 Models (Test Set)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(output_dir / "04_roc_curves_comparison.png", bbox_inches="tight")
    plt.close()

    # Determine Best Model on Test Set (prioritizing ROC-AUC, then Recall)
    best_model_name = metrics_df.iloc[0]["Model"]
    _, _, best_auc, best_pipeline, best_y_pred = roc_data[best_model_name]

    # 5. Best Model Confusion Matrix
    cm = confusion_matrix(y_test, best_y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    labels_annot = np.array([
        [f"{cm[0, 0]}\n({cm_norm[0, 0]:.1%})", f"{cm[0, 1]}\n({cm_norm[0, 1]:.1%})"],
        [f"{cm[1, 0]}\n({cm_norm[1, 0]:.1%})", f"{cm[1, 1]}\n({cm_norm[1, 1]:.1%})"]
    ])

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=labels_annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=["Predicted Healthy (0)", "Predicted PD (1)"],
                yticklabels=["Actual Healthy (0)", "Actual PD (1)"],
                annot_kws={"size": 13, "fontweight": "bold"})
    plt.title(f"Confusion Matrix - Best Model: {best_model_name}\nTest Accuracy: {metrics_df.iloc[0]['Accuracy']:.1%}, Recall: {metrics_df.iloc[0]['Recall (Sensitivity)']:.1%}",
              fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Ground Truth Label", fontweight="bold")
    plt.xlabel("Model Prediction", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "05_best_model_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # 6. Feature Importance (Permutation importance for model-agnostic robustness)
    clf = best_pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        feature_names = X_test.columns
    else:
        perm_res = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_res.importances_mean
        feature_names = X_test.columns

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_df.head(12), x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top 12 Most Influential Voice Features ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score", fontweight="bold")
    plt.ylabel("Acoustic Feature", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "06_feature_importance.png", bbox_inches="tight")
    plt.close()

    # 7. Model Metrics Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    melted = metrics_df.melt(id_vars="Model", value_vars=["Accuracy", "Recall (Sensitivity)", "F1-Score", "ROC-AUC"],
                             var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Performance Comparison of All 7 Machine Learning Models", fontsize=13, fontweight="bold")
    plt.ylim(0.5, 1.05)
    plt.ylabel("Score", fontweight="bold")
    plt.xlabel("Algorithm", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output_dir / "07_model_metrics_comparison.png", bbox_inches="tight")
    plt.close()

    return metrics_df


def run_full_evaluation():
    """
    Main entry point for pipeline execution and evaluation.
    """
    df = load_dataset()
    reports_fig_dir = Path("reports/figures")
    reports_fig_dir.mkdir(parents=True, exist_ok=True)

    print("--> Generating Exploratory Data Analysis figures...")
    plot_eda_figures(df, reports_fig_dir)

    X, y, _ = clean_dataset(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    print("--> Training and hyperparameter tuning all 7 models...")
    trained_results = train_and_tune_all(X_train, y_train)

    print("--> Selecting best model and saving artifact...")
    best_name, artifact = select_and_save_best_model(trained_results, output_path="models/best_model.joblib")

    print("--> Evaluating on held-out test set and generating comparative figures...")
    metrics_df = evaluate_all_models(trained_results, X_test, y_test, reports_fig_dir)

    # Save metrics table
    csv_path = Path("reports/model_comparison.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Model comparison saved to {csv_path.resolve()}")

    print("\n" + "=" * 95)
    print("FINAL TEST SET PERFORMANCE COMPARISON (Sorted by ROC-AUC):")
    print("=" * 95)
    print(metrics_df.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    run_full_evaluation()
