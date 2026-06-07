import numpy as np
import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data."""
    cleaned_data = data.copy()
    cleaned_data.drop_duplicates(inplace=True)
    numeric_cols = cleaned_data.select_dtypes(include=np.number).columns
    for column in numeric_cols:
        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].median())
    return cleaned_data

def cap_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Handle outliers using IQR."""
    processed_data = data.copy()
    numeric_cols = processed_data.select_dtypes(include=np.number).columns
    for column in numeric_cols:
        q1 = processed_data[column].quantile(0.25)
        q3 = processed_data[column].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        processed_data[column] = processed_data[column].clip(lower_limit, upper_limit)
    return processed_data
