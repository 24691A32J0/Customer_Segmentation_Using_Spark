"""
preprocessing.py
----------------
Module for loading, cleaning, exploring, and scaling customer data
for K-Means clustering.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads customer dataset from a CSV file.

    Parameters:
        file_path (str): Relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded dataset successfully from {file_path}. Shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspects and cleans the dataframe by handling missing values,
    removing duplicates, and standardizing column names.

    Parameters:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    cleaned_df = df.copy()

    # Standardize column naming if needed (strip whitespace)
    cleaned_df.columns = [col.strip() for col in cleaned_df.columns]

    # Check and remove duplicates
    initial_rows = len(cleaned_df)
    cleaned_df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(cleaned_df)
    if duplicates_removed > 0:
        print(f"[INFO] Removed {duplicates_removed} duplicate rows.")

    # Check and impute missing values (numerical -> median, categorical -> mode)
    missing_counts = cleaned_df.isnull().sum()
    if missing_counts.sum() > 0:
        print("[INFO] Missing values detected. Imputing...")
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                else:
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    else:
        print("[INFO] No missing values found in dataset.")

    return cleaned_df


def perform_eda_summary(df: pd.DataFrame) -> dict:
    """
    Calculates summary statistics and key metrics for exploratory data analysis.

    Parameters:
        df (pd.DataFrame): Cleaned dataframe.

    Returns:
        dict: Summary metrics including stats, correlations, and demographics.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().to_dict()
    corr_matrix = numeric_df.corr().to_dict()

    gender_distribution = (
        df['Gender'].value_counts().to_dict() if 'Gender' in df.columns else {}
    )

    summary = {
        "total_records": len(df),
        "columns": list(df.columns),
        "descriptive_stats": stats,
        "correlation_matrix": corr_matrix,
        "gender_distribution": gender_distribution,
    }
    return summary


def select_and_scale_features(
    df: pd.DataFrame,
    feature_cols: list = None,
    scaler: StandardScaler = None,
    fit_scaler: bool = True
) -> tuple:
    """
    Selects numerical features for clustering and applies StandardScaler.
    Feature scaling standardizes values to mean=0 and variance=1,
    which is essential for Euclidean distance-based algorithms like K-Means.

    Parameters:
        df (pd.DataFrame): Dataframe containing features.
        feature_cols (list): List of column names to scale.
                             Defaults to ['Annual Income (k$)', 'Spending Score (1-100)'].
        scaler (StandardScaler, optional): Pre-fitted scaler. If None, a new one is created.
        fit_scaler (bool): If True, fits and transforms. If False, only transforms.

    Returns:
        tuple: (scaled_data: np.ndarray, fitted_scaler: StandardScaler, selected_features_df: pd.DataFrame)
    """
    if feature_cols is None:
        feature_cols = ['Annual Income (k$)', 'Spending Score (1-100)']

    # Verify column existence
    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"Feature column '{col}' not found in DataFrame columns: {list(df.columns)}")

    features_raw = df[feature_cols].copy()

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        scaled_data = scaler.fit_transform(features_raw)
    else:
        scaled_data = scaler.transform(features_raw)

    return scaled_data, scaler, features_raw


def save_scaler(scaler: StandardScaler, output_path: str) -> None:
    """
    Saves fitted StandardScaler to disk using joblib.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(scaler, output_path)
    print(f"[INFO] Scaler saved successfully to: {output_path}")


def load_scaler(model_path: str) -> StandardScaler:
    """
    Loads saved StandardScaler from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Scaler file not found at: {model_path}")
    scaler = joblib.load(model_path)
    print(f"[INFO] Scaler loaded successfully from: {model_path}")
    return scaler


if __name__ == "__main__":
    # Self-test block
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_csv = os.path.join(base_dir, "data", "customer_data.csv")
    df = load_data(sample_csv)
    df_clean = clean_data(df)
    scaled_feats, fitted_scaler, raw_feats = select_and_scale_features(df_clean)
    print(f"[TEST] Scaled shape: {scaled_feats.shape}, Mean approx: {scaled_feats.mean(axis=0)}")
