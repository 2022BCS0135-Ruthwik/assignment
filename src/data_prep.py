import pandas as pd
import os
import requests

def download_data():
    os.makedirs('data', exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    print(f"Downloading dataset from {url}...")
    res = requests.get(url)
    
    raw_path = "data/winequality-red.csv"
    with open(raw_path, "wb") as f:
        f.write(res.content)
        
    print("Processing datasets...")
    # The dataset uses semicolon as separator
    df = pd.read_csv(raw_path, sep=";")
    
    # Clean up column names by replacing spaces with underscores
    df.columns = [c.replace(' ', '_') for c in df.columns]
    
    # Version 2 is full dataset
    v2_path = "data/v2_dataset.csv"
    df.to_csv(v2_path, index=False)
    print(f"Version 2 (full dataset) saved to {v2_path}. Shape: {df.shape}")
    
    # Version 1 is subset
    df_subset = df.sample(frac=0.2, random_state=42)
    v1_path = "data/v1_dataset.csv"
    df_subset.to_csv(v1_path, index=False)
    print(f"Version 1 (subset dataset) saved to {v1_path}. Shape: {df_subset.shape}")
    
    print("Data preparation complete.")

if __name__ == "__main__":
    download_data()
