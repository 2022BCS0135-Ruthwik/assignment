import pandas as pd

def load_data(data_path: str) -> pd.DataFrame:
    """Loads dataset from a given path."""
    return pd.read_csv(data_path)

def get_features_and_target(df: pd.DataFrame, target_col: str, feature_cols=None):
    """Separates features and target variable."""
    if feature_cols:
        X = df[feature_cols]
    else:
        X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
