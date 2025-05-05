import pandas as pd
from sklearn.impute import SimpleImputer


def simpleimput_mean(df: pd.DataFrame, col: list) -> pd.DataFrame:
    """
    Impute missing values in specified columns of a DataFrame using the mean.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to impute.

    Returns:
    - pd.DataFrame: DataFrame with imputed values.
    """
    imputer = SimpleImputer(strategy="mean")
    df[col] = imputer.fit_transform(df[col])
    return df


def simpleimput_median(df: pd.DataFrame, col: list) -> pd.DataFrame:
    """
    Impute missing values in specified columns of a DataFrame using the mean.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to impute.

    Returns:
    - pd.DataFrame: DataFrame with imputed values.
    """
    imputer = SimpleImputer(strategy="median")
    df[col] = imputer.fit_transform(df[col])
    return df
