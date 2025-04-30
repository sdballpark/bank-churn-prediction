"""
data_preprocessing.py

Preprocesses the Bank Churn dataset:
- Loads raw CSV data
- Cleans and encodes categorical variables
- Handles missing values
- Scales features
- Saves processed data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath='../data/raw/Churn.csv'):
    # Load data
    df = pd.read_csv(filepath)

    # Convert TotalCharges to numeric and fill NaNs
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Drop customerID
    df.drop(['customerID'], axis=1, inplace=True)

    # Encode target label
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])

    # Encode other categorical variables
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    print("✅ Data preprocessing complete.")
