"""
Train a pure-NumPy Linear Regression (least-squares) model.
No scikit-learn, no scipy — avoids Application Control DLL blocks.
Saves model/weights.npy and model/mean_std.npy for inference.
"""

import os
import numpy as np
import pandas as pd

DATA_PATH = "data/electricity_data.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

FEATURES = ["consumption_kwh", "num_people", "daily_hours", "prev_consumption_kwh"]
TARGET   = "bill_inr"

X = df[FEATURES].values.astype(float)
y = df[TARGET].values.astype(float)

# ── simple 80/20 split ───────────────────────────────────────────────────────
np.random.seed(42)
idx   = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_tr, X_te = X[idx[:split]], X[idx[split:]]
y_tr, y_te = y[idx[:split]], y[idx[split:]]

# ── z-score normalisation ────────────────────────────────────────────────────
mean = X_tr.mean(axis=0)
std  = X_tr.std(axis=0) + 1e-8

X_tr_sc = (X_tr - mean) / std
X_te_sc = (X_te - mean) / std

# ── add polynomial features (degree-2 interactions) for better fit ───────────
def poly_features(X_sc):
    """Returns [X, X^2, pairwise products] — no sklearn needed."""
    n, d = X_sc.shape
    parts = [X_sc, X_sc ** 2]
    for i in range(d):
        for j in range(i + 1, d):
            parts.append((X_sc[:, i] * X_sc[:, j]).reshape(-1, 1))
    return np.hstack(parts)

X_tr_p = poly_features(X_tr_sc)
X_te_p = poly_features(X_te_sc)

# ── add bias column ──────────────────────────────────────────────────────────
ones = np.ones((X_tr_p.shape[0], 1))
X_tr_b = np.hstack([ones, X_tr_p])

ones_te = np.ones((X_te_p.shape[0], 1))
X_te_b  = np.hstack([ones_te, X_te_p])

# ── solve via normal equations (least-squares) ───────────────────────────────
weights, _, _, _ = np.linalg.lstsq(X_tr_b, y_tr, rcond=None)

# ── evaluation ───────────────────────────────────────────────────────────────
preds = X_te_b @ weights
mae   = np.mean(np.abs(preds - y_te))
ss_res = np.sum((preds - y_te) ** 2)
ss_tot = np.sum((y_te - y_te.mean()) ** 2)
r2    = 1 - ss_res / ss_tot

print(f"MAE : INR {mae:.2f}")
print(f"R2  : {r2:.4f}")

# ── save artifacts ────────────────────────────────────────────────────────────
np.save(os.path.join(MODEL_DIR, "weights.npy"), weights)
np.save(os.path.join(MODEL_DIR, "mean_std.npy"), np.vstack([mean, std]))
print("Model artifacts saved to model/")
