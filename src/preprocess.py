# src/preprocess.py
import pandas as pd, joblib, os, argparse
from sklearn.preprocessing import OneHotEncoder

def main(in_path, out_path):
    df = pd.read_csv(in_path)
    required = ['cpu_request','mem_request','cpu_limit','mem_limit','runtime_minutes','controller_kind','cpu_usage']
    df = df[required].copy()
    df = df.dropna(subset=['cpu_usage'])
    num_cols = ['cpu_request','mem_request','cpu_limit','mem_limit','runtime_minutes']
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df['controller_kind'] = df['controller_kind'].fillna('UNKNOWN')
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat = ohe.fit_transform(df[['controller_kind']])
    cat_cols = [f"ck_{c}" for c in ohe.categories_[0]]
    df_ohe = pd.DataFrame(cat, columns=cat_cols, index=df.index)
    df = pd.concat([df.drop(columns=['controller_kind']), df_ohe], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    joblib.dump(ohe, os.path.join(os.path.dirname(out_path), "ohe_controller_kind.joblib"))
    print("Saved processed:", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("in_path")
    p.add_argument("out_path")
    args = p.parse_args()
    main(args.in_path, args.out_path)
