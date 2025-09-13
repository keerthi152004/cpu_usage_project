# src/evaluate.py
import joblib, pandas as pd, os, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(data_csv, model_path, scaler_path, out_dir="outputs_eval"):
    df = pd.read_csv(data_csv)
    X = df.drop(columns=["cpu_usage"]); y = df["cpu_usage"]
    scaler = joblib.load(scaler_path); model = joblib.load(model_path)
    Xs = scaler.transform(X); preds = model.predict(Xs)
    mse = mean_squared_error(y, preds); rmse = mse**0.5; mae = mean_absolute_error(y, preds); r2 = r2_score(y, preds)
    os.makedirs(out_dir, exist_ok=True)
    # residuals
    res = y - preds
    plt.figure(figsize=(6,4)); sns.scatterplot(x=preds, y=res); plt.axhline(0, color='red'); plt.xlabel("Pred"); plt.ylabel("Residual"); plt.savefig(os.path.join(out_dir,"residuals.png"))
    plt.close()
    # predicted vs actual
    plt.figure(figsize=(6,4)); sns.scatterplot(x=y, y=preds); mn, mx = min(y.min(), preds.min()), max(y.max(), preds.max()); plt.plot([mn,mx],[mn,mx],'r--'); plt.savefig(os.path.join(out_dir,"pred_vs_actual.png")); plt.close()
    # write metrics
    with open(os.path.join(out_dir,"metrics.txt"), "w") as f:
        f.write(f"mse:{mse}\nrmse:{rmse}\nmae:{mae}\nr2:{r2}\n")
    print("Saved evaluation artifacts to", out_dir)
out_dir = "outputs_eval"
os.makedirs(out_dir, exist_ok=True)

# Example saving a metrics file
with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(results, f, indent=4)