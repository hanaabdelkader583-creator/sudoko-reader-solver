"""
src/nn/baseline.py
Trains a Logistic Regression baseline on the raw MNIST data.
"""
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from .train import load_raw_mnist

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences >= bin_limits[i]) & (confidences < bin_limits[i+1])
        if np.any(bin_mask):
            acc = np.mean(y_true[bin_mask] == predictions[bin_mask])
            conf = np.mean(confidences[bin_mask])
            ece += np.sum(bin_mask) / len(y_true) * np.abs(acc - conf)
    return ece

def main():
    print("Loading raw MNIST data...")
    raw_dir = Path("data/MNIST/raw")
    splits = load_raw_mnist(raw_dir, val_fraction=0.1, seed=42)
    
    # Flatten images for scikit-learn (N, 28, 28) -> (N, 784)
    X_train = splits["train"][0].reshape(len(splits["train"][0]), -1) / 255.0
    y_train = splits["train"][1]
    X_val = splits["val"][0].reshape(len(splits["val"][0]), -1) / 255.0
    y_val = splits["val"][1]

    print("Training Logistic Regression (this will take 1-2 minutes)...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)

    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print(f"Calibration Error (ECE): {expected_calibration_error(y_val, y_prob):.4f}")

if __name__ == "__main__":
    main()