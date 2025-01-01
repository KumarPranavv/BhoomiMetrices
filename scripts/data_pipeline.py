import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def clean_and_process_data(raw_csv_path, output_csv_path):
    """
    Reads raw land price data, cleans it, and produces a processed CSV.
    """
    df = pd.read_csv(raw_csv_path, parse_dates=['date'])
    
    # Example cleaning steps:
    # 1. Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # 2. Sort by date
    df.sort_values(by='date', inplace=True)

    # 3. (Optional) Feature engineering
    # Let's create an example "year" column
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # 4. Fill or drop missing values (example)
    df.dropna(inplace=True)

    # 5. Save processed data
    df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Processed data saved to {output_csv_path}")

if __name__ == "__main__":
    # Define paths
    raw_csv = Path("data/raw/historical_land_prices.csv")
    processed_csv = Path("data/processed/processed_land_data.csv")

    clean_and_process_data(raw_csv, processed_csv)
