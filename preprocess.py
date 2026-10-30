"""
preprocess.py - Data loading, inspection, cleaning, and preprocessing pipeline.

UCI Parkinson's Disease Dataset Preprocessing Module.
Handles stratified splitting, data integrity checks, and leakage-free pipeline creation.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline


# Feature taxonomy by acoustic category
FEATURE_CATEGORIES = {
    "Fundamental Frequency (Pitch)": [
        "MDVP:Fo(Hz)",   # Average vocal fundamental frequency
        "MDVP:Fhi(Hz)",  # Maximum vocal fundamental frequency
        "MDVP:Flo(Hz)"   # Minimum vocal fundamental frequency
    ],
    "Jitter (Frequency Variation)": [
        "MDVP:Jitter(%)",    # Percentage jitter
        "MDVP:Jitter(Abs)",  # Absolute jitter in microseconds
        "MDVP:RAP",          # Relative amplitude perturbation
        "MDVP:PPQ",          # 5-point period perturbation quotient
        "Jitter:DDP"         # Average absolute difference of differences between jitter cycles
    ],
    "Shimmer (Amplitude Variation)": [
        "MDVP:Shimmer",      # Local shimmer
        "MDVP:Shimmer(dB)",  # Local shimmer in dB
        "Shimmer:APQ3",      # 3-point amplitude perturbation quotient
        "Shimmer:APQ5",      # 5-point amplitude perturbation quotient
        "MDVP:APQ",          # 11-point amplitude perturbation quotient
        "Shimmer:DDA"        # Average absolute differences between consecutive amplitudes
    ],
    "Noise & Harmonics": [
        "NHR",  # Noise-to-harmonics ratio
        "HNR"   # Harmonics-to-noise ratio
    ],
    "Nonlinear Dynamical Complexity": [
        "RPDE",     # Recurrence period density entropy
        "DFA",      # Detrended fluctuation analysis
        "spread1",  # Nonlinear measure of fundamental frequency variation
        "spread2",  # Nonlinear measure of fundamental frequency variation
        "D2",       # Correlation dimension
        "PPE"       # Pitch period entropy
    ]
}

ALL_FEATURE_COLUMNS = [col for cols in FEATURE_CATEGORIES.values() for col in cols]
TARGET_COLUMN = "status"


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Parkinson's dataset from path or fallback default locations.
    """
    if filepath and Path(filepath).exists():
        return pd.read_csv(filepath)

    default_paths = [
        Path("data/parkinsons.data"),
        Path("parkinsons-detection/data/parkinsons.data"),
        Path("../data/parkinsons.data"),
        Path("c:/Users/HP/Desktop/parkinsons-voice-project/data/parkinsons.data")
    ]
    for p in default_paths:
        if p.exists():
            return pd.read_csv(p)

    raise FileNotFoundError("Could not find parkinsons.data in any default location.")


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Separate features and target, dropping the 'name' identifier column.
    Returns:
        X: DataFrame of 22 acoustic features
        y: Series of status labels (0 = Healthy, 1 = Parkinson's)
        subject_names: Series of original 'name' strings for subject-level analysis
    """
    df_clean = df.copy()
    subject_names = df_clean["name"] if "name" in df_clean.columns else None

    # Drop non-feature columns
    cols_to_drop = [c for c in ["name", TARGET_COLUMN] if c in df_clean.columns]
    X = df_clean.drop(columns=cols_to_drop)
    y = df_clean[TARGET_COLUMN].astype(int)

    return X, y, subject_names


def get_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split (80/20 by default).
    Stratification ensures both classes (75.4% PD, 24.6% Healthy) are represented proportionally.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(classifier, scaler_type: str = "standard") -> Pipeline:
    """
    Construct a scikit-learn Pipeline with scaler and estimator.
    Fitting the scaler strictly inside the pipeline prevents data leakage from the test set.
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", classifier)
    ])
    return pipeline


if __name__ == "__main__":
    df = load_dataset()
    print("Dataset loaded successfully. Shape:", df.shape)
    X, y, names = clean_dataset(df)
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print("Class breakdown:\n", y.value_counts(normalize=True).round(4) * 100)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
