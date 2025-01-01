# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

def train_models(processed_csv_path, models_folder):
    df = pd.read_csv(processed_csv_path, parse_dates=['date'])

    # --- REMOVE create_target_shifts call ---
    # df = create_target_shifts(df)

    # Create new target columns for 1yr (12 months) and 2yr (24 months)
    df['target_1yr'] = df.groupby('city')['price'].shift(-12)
    df['target_2yr'] = df.groupby('city')['price'].shift(-24)

    # Drop rows with NaN in our new target columns
    df.dropna(subset=['target_1yr', 'target_2yr'], inplace=True)

    # Example feature set: (price, year, month)
    features = ['price', 'year', 'month']

    # Convert 'city' to dummies
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    X = pd.concat([df[features], city_dummies], axis=1)

    # Targets
    y_1yr = df['target_1yr']
    y_2yr = df['target_2yr']

    print("X shape:", X.shape)
    print("Number of rows in y_1yr:", y_1yr.dropna().shape[0])
    print("Number of rows in y_2yr:", y_2yr.dropna().shape[0])

    # Split
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y_1yr, test_size=0.2, shuffle=False)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y_2yr, test_size=0.2, shuffle=False)

    # Initialize RandomForestRegressor
    model_1yr = RandomForestRegressor(n_estimators=50, random_state=42)
    model_2yr = RandomForestRegressor(n_estimators=50, random_state=42)

    # Train
    model_1yr.fit(X_train_1, y_train_1)
    model_2yr.fit(X_train_2, y_train_2)

    # Evaluate with MAE
    preds_1yr = model_1yr.predict(X_test_1)
    preds_2yr = model_2yr.predict(X_test_2)
    mae_1yr = mean_absolute_error(y_test_1, preds_1yr)
    mae_2yr = mean_absolute_error(y_test_2, preds_2yr)

    print(f"[INFO] 1yr Model MAE: {mae_1yr:.2f}")
    print(f"[INFO] 2yr Model MAE: {mae_2yr:.2f}")

    # Save
    models_folder = Path(models_folder)
    models_folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_1yr, models_folder / "model_1yr.pkl")
    joblib.dump(model_2yr, models_folder / "model_2yr.pkl")
    print("[INFO] Models saved.")

if __name__ == "__main__":
    processed_csv = "data/processed/processed_land_data.csv"
    models_dir = "models"
    train_models(processed_csv, models_dir)
