import os
import requests
import pandas as pd
import numpy as np

def download_raw_data(url: str, output_path: str) -> None:
    """
    Downloads raw CSV data from the provided URL and saves it to output_path.
    """
    if os.path.exists(output_path):
        print(f"File already exists at {output_path}. Skipping download.")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Successfully downloaded raw data and saved to {output_path}")

def load_data(path: str) -> pd.DataFrame:
    """
    Loads raw CSV data into a Pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the NYC Airbnb raw DataFrame by:
    - Filling missing names and host names
    - Handling missing reviews_per_month (filling with 0.0)
    - Filtering out rows where price is 0 or negative
    """
    df_clean = df.copy()
    
    # Fill missing text info
    df_clean['name'] = df_clean['name'].fillna('Unknown')
    df_clean['host_name'] = df_clean['host_name'].fillna('Unknown')
    
    # Missing reviews_per_month means 0 reviews, so fill with 0.0
    df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0.0)
    
    # Listings with price <= 0 are invalid data entry errors, drop them
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['price'] > 0]
    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with price <= 0.")
        
    return df_clean

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the cleaned DataFrame:
    - Converts last_review to datetime and calculates days_since_last_review
    - Calculates length of listing name
    - Adds binary has_reviews indicator
    - Creates log_price target variable
    """
    df_feat = df.copy()
    
    # Calculate name length (handy feature)
    df_feat['name_length'] = df_feat['name'].apply(lambda x: len(str(x)))
    
    # Binary indicator for whether the listing has reviews
    df_feat['has_reviews'] = (df_feat['number_of_reviews'] > 0).astype(int)
    
    # Date processing: days since last review relative to late 2019 (dataset's period)
    anchor_date = pd.to_datetime('2019-12-31')
    df_feat['last_review'] = pd.to_datetime(df_feat['last_review'])
    
    # Calculate difference in days. If no review, fill with 3650 days (10 years)
    days_diff = (anchor_date - df_feat['last_review']).dt.days
    df_feat['days_since_last_review'] = days_diff.fillna(3650)
    
    # Create the log_price target variable for regression models
    df_feat['log_price'] = np.log1p(df_feat['price'])
    
    return df_feat

def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Saves the processed DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")
