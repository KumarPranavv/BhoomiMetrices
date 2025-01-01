import pandas as pd
import joblib
from pathlib import Path

def batch_predict(input_csv, model_5yr_path, model_10yr_path, output_csv):
    df = pd.read_csv(input_csv)

    # Minimal feature generation: same as training
    # Convert city to dummies, ensure columns match training
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    X = pd.concat([df[['price','year','month']], city_dummies], axis=1)

    # Load models
    model_5yr = joblib.load(model_5yr_path)
    model_10yr = joblib.load(model_10yr_path)

    df['predicted_5yr'] = model_5yr.predict(X)
    df['predicted_10yr'] = model_10yr.predict(X)

    # Save
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Batch predictions saved to {output_csv}")

if __name__ == "__main__":
    # Example usage:
    batch_predict(
        input_csv="data/processed/processed_land_data.csv",
        model_5yr_path="models/model_5yr.pkl",
        model_10yr_path="models/model_10yr.pkl",
        output_csv="data/processed/predictions.csv"
    )
