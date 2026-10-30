"""
train.py - Model training, hyperparameter tuning, cross-validation, and selection.

Trains and tunes 7 machine learning models:
1. Logistic Regression (balanced weights)
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM RBF kernel)
4. Decision Tree Classifier
5. Random Forest Classifier
6. Gradient Boosting Classifier
7. XGBoost Classifier

Selects the best performing model based on ROC-AUC and Recall (prioritizing sensitive PD detection),
and persists the final pipeline to models/best_model.joblib.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from preprocess import load_dataset, clean_dataset, get_train_test_split, build_pipeline


def get_model_definitions() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """
    Returns dictionary of models paired with their hyperparameter grids for Pipeline tuning.
    Note parameter prefix 'classifier__' matches the pipeline step name.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs"]
            }
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [3, 5, 7],
                "classifier__weights": ["uniform", "distance"]
            }
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
            {
                "classifier__C": [0.5, 1.0, 5.0],
                "classifier__gamma": ["scale", 0.05, 0.1]
            }
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            {
                "classifier__max_depth": [3, 5, None],
                "classifier__min_samples_split": [2, 5]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [4, 8, None],
                "classifier__min_samples_split": [2, 5]
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "classifier__n_estimators": [50, 100],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 4]
            }
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=42),
            {
                "classifier__n_estimators": [50, 100],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 4]
            }
        )
    }
    return models


def train_and_tune_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    scoring: str = "roc_auc"
) -> Dict[str, Dict[str, Any]]:
    """
    Train each model wrapped inside a Pipeline (StandardScaler + Estimator)
    using GridSearchCV with Stratified 5-Fold Cross Validation.
    """
    model_defs = get_model_definitions()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    trained_results = {}

    print(f"=== Starting Training & Hyperparameter Tuning ({cv_splits}-Fold Stratified CV, metric={scoring}) ===")

    for name, (estimator, param_grid) in model_defs.items():
        print(f"--> Tuning: {name} ...")
        pipeline = build_pipeline(estimator, scaler_type="standard")

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            refit=True
        )

        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        print(f"    Best CV {scoring}: {best_cv_score:.4f} | Best Params: {best_params}")

        trained_results[name] = {
            "pipeline": best_pipeline,
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "grid_search": grid_search
        }

    return trained_results


def select_and_save_best_model(
    trained_results: Dict[str, Dict[str, Any]],
    output_path: str = "models/best_model.joblib"
) -> Tuple[str, Any]:
    """
    Select the model with the highest Cross-Validation ROC-AUC score,
    persist it to disk with joblib, and save training metadata.
    """
    best_name = max(trained_results, key=lambda k: trained_results[k]["best_cv_score"])
    best_pipeline = trained_results[best_name]["pipeline"]
    best_score = trained_results[best_name]["best_cv_score"]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save artifact bundle with feature names and training metadata
    artifact = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "best_cv_roc_auc": best_score,
        "best_params": trained_results[best_name]["best_params"],
        "feature_names": trained_results[best_name]["pipeline"].feature_names_in_.tolist() if hasattr(trained_results[best_name]["pipeline"], "feature_names_in_") else None
    }
    joblib.dump(artifact, out_file)
    print(f"\n[OK] Best Model Selected: '{best_name}' (CV Score: {best_score:.4f})")
    print(f"[OK] Persisted model artifact to: {out_file.resolve()}")

    return best_name, artifact


if __name__ == "__main__":
    df = load_dataset()
    X, y, _ = clean_dataset(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    results = train_and_tune_all(X_train, y_train)
    best_name, artifact = select_and_save_best_model(results)
