"""
model_training.py

Trains a RandomForestClassifier on preprocessed Bank Churn data.
Saves the trained model to disk as 'model.pkl'.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import load_and_preprocess

def train_model():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, '../model/model.pkl')
    print("✅ Model training complete and saved to '../model/model.pkl'.")

if __name__ == "__main__":
    train_model()
