# src/train.py
import argparse, os, joblib, subprocess, json
import pandas as pd, numpy as np, mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml

def get_git_commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    except: return "unknown"

def load_params(pth="params.yaml"):
    with open(pth) as f: return yaml.safe_load(f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data_csv")
    p.add_argument("out_dir")
    p.add_argument("--params", default="params.yaml")
    args = p.parse_args()
    params = load_params(args.params)
    df = pd.read_csv(args.data_csv)
    X = df.drop(columns=["cpu_usage"])
    y = df["cpu_usage"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["train"]["test_size"], random_state=params["train"]["random_state"])
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    models = {
      "linear": LinearRegression(),
      "rf": RandomForestRegressor(n_estimators=params["train"]["rf"]["n_estimators"],
                                 max_depth=params["train"]["rf"]["max_depth"],
                                 random_state=params["train"]["random_state"])
    }
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.out_dir,"scaler.joblib"))
    mlflow.set_experiment("cpu-usage")
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            mse = mean_squared_error(y_test, preds); rmse = mse**0.5
            mae = mean_absolute_error(y_test, preds); r2 = r2_score(y_test, preds)
            model_path = os.path.join(args.out_dir, f"model_{name}.joblib")
            joblib.dump(model, model_path)
            mlflow.log_param("model", name)
            mlflow.log_metric("mse", float(mse)); mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("mae", float(mae)); mlflow.log_metric("r2", float(r2))
            mlflow.log_artifact(model_path, artifact_path="models")
            mlflow.log_artifact(os.path.join(args.out_dir,"scaler.joblib"), artifact_path="preproc")
            git_commit = get_git_commit()
            mlflow.log_param("git_commit", git_commit)
            # add data checksum if present
            checksum_file = args.data_csv + ".sha256"
            if os.path.exists(checksum_file):
                mlflow.log_artifact(checksum_file, artifact_path="data_info")
            print("Logged run:", run.info.run_id)
