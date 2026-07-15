"""
predict.py

Required usage (per assignment spec):
  python predict.py --data_dir <folder> --out predictions.csv

<folder> has the same structure as eot_data/<lang>/ (audio/ + labels.csv),
possibly WITHOUT a 'label' column (inference mode) -- this script does not
require or read the 'label' column even if present.

Output: predictions.csv with columns turn_id,pause_index,p_eot
"""
import argparse
import joblib
import numpy as np
import pandas as pd

from eot_dataset import build_feature_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="model.joblib")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    feature_names = bundle["feature_names"]

    X, meta, _y = build_feature_matrix(args.data_dir)

    # align to training-time feature columns exactly (order + presence)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names].fillna(0.0)

    p_eot = clf.predict_proba(X.values)[:, 1]

    out = pd.DataFrame({
        "turn_id": meta["turn_id"].values,
        "pause_index": meta["pause_index"].values,
        "p_eot": p_eot,
    })
    out.to_csv(args.out, index=False)
    print(f"[predict] wrote {len(out)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
