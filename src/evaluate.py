# src/evaluate.py
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main(data_csv, model_path, scaler_path, out_dir="outputs"):
    df = pd.read_csv(data_csv)
    target = "cpu_usage"
    X = df.drop(columns=[target])
    y = df[target]

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    os.makedirs(out_dir, exist_ok=True)
    # Residual plot
    residuals = y - preds
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=preds, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted CPU usage")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.savefig(os.path.join(out_dir, "residuals.png"))
    plt.close()

    # Predicted vs Actual
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=y, y=preds)
    mn, mx = min(y.min(), preds.min()), max(y.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], color='red', linestyle='--')
    plt.xlabel("Actual CPU usage")
    plt.ylabel("Predicted CPU usage")
    plt.title("Predicted vs Actual")
    plt.savefig(os.path.join(out_dir, "pred_vs_actual.png"))
    plt.close()

    # SHAP values (for tree-based models)
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_scaled)
        shap.summary_plot(shap_values, features=X, show=False)
        plt.savefig(os.path.join(out_dir, "shap_summary.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("SHAP failed:", e)

    # Save metrics
    metrics = {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    print("Saved evaluation artifacts to", out_dir)
