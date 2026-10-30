"""
predict.py - Inference engine for Parkinson's Disease classification from voice features.

Loads the saved best model pipeline and generates predictions with probabilities,
risk stratification, and contributing acoustic feature insights.
"""

from pathlib import Path
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd

from preprocess import ALL_FEATURE_COLUMNS, load_dataset, clean_dataset


def load_model(model_path: Union[str, Path] = "models/best_model.joblib") -> Dict[str, Any]:
    """
    Load serialized model artifact.
    """
    path = Path(model_path)
    if not path.exists():
        fallback = Path("parkinsons-detection/models/best_model.joblib")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at {path}")

    artifact = joblib.load(path)
    return artifact


def predict_sample(
    features: Union[Dict[str, float], pd.DataFrame, np.ndarray, list],
    model_artifact: Dict[str, Any] = None,
    model_path: Union[str, Path] = "models/best_model.joblib"
) -> Dict[str, Any]:
    """
    Run prediction on a single voice measurement sample.

    Args:
        features: Dictionary mapping feature names to numeric values, or DataFrame/array.
        model_artifact: Pre-loaded model artifact (optional).
        model_path: Path to serialized artifact if not passed.

    Returns:
        dict: Detailed prediction results including label, probability, risk level, and confidence.
    """
    if model_artifact is None:
        model_artifact = load_model(model_path)

    pipeline = model_artifact["pipeline"]

    # Convert features to DataFrame
    if isinstance(features, dict):
        df_input = pd.DataFrame([features])
    elif isinstance(features, (np.ndarray, list)):
        df_input = pd.DataFrame([features], columns=ALL_FEATURE_COLUMNS)
    elif isinstance(features, pd.DataFrame):
        df_input = features.copy()
    else:
        raise TypeError("Input features must be a dict, DataFrame, or numpy array.")

    # Ensure all required columns are present in exact order
    missing = set(ALL_FEATURE_COLUMNS) - set(df_input.columns)
    if missing:
        raise ValueError(f"Input features missing required columns: {missing}")

    df_input = df_input[ALL_FEATURE_COLUMNS]

    # Predict class & probability
    pred_class = int(pipeline.predict(df_input)[0])

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_input)[0]
        prob_healthy = float(probs[0])
        prob_parkinsons = float(probs[1])
    else:
        prob_parkinsons = float(pred_class)
        prob_healthy = 1.0 - prob_parkinsons

    # Risk level classification
    if prob_parkinsons < 0.35:
        risk_level = "Low Risk (Healthy Profile)"
    elif prob_parkinsons < 0.65:
        risk_level = "Moderate Risk (Indeterminate / Borderline)"
    else:
        risk_level = "High Risk (Voice Biomarkers Consistent with Parkinson's)"

    label = "Parkinson's Disease Detected" if pred_class == 1 else "Healthy (Normal Voice Pattern)"
    confidence = max(prob_healthy, prob_parkinsons) * 100.0

    return {
        "status": pred_class,
        "label": label,
        "prob_parkinsons": prob_parkinsons,
        "prob_healthy": prob_healthy,
        "confidence_percent": round(confidence, 2),
        "risk_level": risk_level,
        "model_used": model_artifact.get("model_name", "Unknown"),
        "disclaimer": "Educational & academic demonstration only; not a formal medical diagnosis."
    }


def get_sample_patient(sample_type: str = "healthy") -> Dict[str, float]:
    """
    Utility to fetch representative sample records from the dataset for demonstration.
    """
    df = load_dataset()
    X, y, _ = clean_dataset(df)

    if sample_type.lower() == "healthy":
        sample = X[y == 0].iloc[0].to_dict()
    elif sample_type.lower() in ["pd", "parkinsons"]:
        sample = X[y == 1].iloc[0].to_dict()
    else:
        sample = X.median().to_dict()

    return sample


if __name__ == "__main__":
    print("Testing inference engine...")
    sample_healthy = get_sample_patient("healthy")
    sample_pd = get_sample_patient("parkinsons")

    print("\n--- Evaluating Sample Healthy Patient ---")
    res_healthy = predict_sample(sample_healthy)
    for k, v in res_healthy.items():
        print(f"  {k}: {v}")

    print("\n--- Evaluating Sample Parkinson's Patient ---")
    res_pd = predict_sample(sample_pd)
    for k, v in res_pd.items():
        print(f"  {k}: {v}")
