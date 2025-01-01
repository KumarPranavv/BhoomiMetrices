import pandas as pd

def create_target_shifts(df, horizon_5yr=60, horizon_10yr=120):
    """
    Create shifted price columns for 5-year and 10-year predictions.
    horizon_5yr=60 months, horizon_10yr=120 months (assuming monthly data).
    """
    df = df.copy()
    df['target_5yr'] = df.groupby('city')['price'].shift(-horizon_5yr)
    df['target_10yr'] = df.groupby('city')['price'].shift(-horizon_10yr)
    return df
