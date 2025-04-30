"""
model_inference.py

Loads the trained model and makes predictions on new data.
"""

import joblib
import numpy as np

def load_model(model_path='../model/model.pkl'):
    return joblib.load(model_path)

def predict(model, sample_array):
    prediction = model.predict(sample_array)
    probability = model.predict_proba(sample_array)
    return prediction, probability

if __name__ == "__main__":
    model = load_model()
    sample_input = np.array([[0, 0, 1, 0, 1, 0, 25.3, 1, 0, 50.1, 1, 1, 0, 70.5]])  # Example scaled input
    pred, prob = predict(model, sample_input)
    print(f"Prediction: {pred[0]} | Probability: {prob[0]}")
