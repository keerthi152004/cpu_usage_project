# src/evaluate.py
import joblib, pandas as pd, os, matplotlib.pyplot as plt, seaborn as sns, json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate(data_csv, model_path, scaler_path, out_dir="outputs_eval"):
    df = pd.read_csv(data_csv)
    X = df.drop(columns=["cpu_usage"]); y = df["cpu_usage"]
    scaler = joblib.load(scaler_path); model = joblib.load(model_path)
    Xs = scaler.transform(X); preds = model.predict(Xs)

    mse = mean_squared_error(y, preds)
    rmse = mse**0.5
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    os.makedirs(out_dir, exist_ok=True)

    # residuals plot
    res = y - preds
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=preds, y=res)
    plt.axhline(0, color='red')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.savefig(os.path.join(out_dir, "residuals.png"))
    plt.close()

    # predicted vs actual
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y, y=preds)
    mn, mx = min(y.min(), preds.min()), max(y.max(), preds.max())
    plt.plot([mn,mx], [mn,mx], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(os.path.join(out_dir, "pred_vs_actual.png"))
    plt.close()

    # feature importance if RF
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(6,4))
        sns.barplot(x=importances[indices], y=X.columns[indices])
        plt.title("Feature Importance (Random Forest)")
        plt.savefig(os.path.join(out_dir, "feature_importance.png"))
        plt.close()

    # save metrics
    results = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        for k,v in results.items():
            f.write(f"{k}:{v}\n")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Saved evaluation artifacts to", out_dir)

if __name__ == "__main__":
    import sys
    data_csv, model_path, scaler_path = sys.argv[1], sys.argv[2], sys.argv[3]
    evaluate(data_csv, model_path, scaler_path, out_dir="outputs_eval")
