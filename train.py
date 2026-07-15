"""
train.py

Usage:
  python train.py --data_dirs eot_data/english eot_data/hindi --model_out model.joblib

Trains a single multilingual classifier (pooling all provided language
folders) since the hidden test set is "mostly Hindi" while most labeled data
on hand is likely English-heavy -- pooling + language-agnostic prosodic
features generalizes better than fitting two separate small models.

Uses RandomForest wrapped in CalibratedClassifierCV so p_eot is an actual
calibrated probability (this matters a lot for score.py, which tunes a
threshold to hit a specific false-cutoff rate -- calibration quality
directly affects how well that threshold transfers to the hidden test set).
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

from eot_dataset import build_feature_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True,
                     help="one or more eot_data/<lang> folders WITH labels.csv (label column present)")
    ap.add_argument("--model_out", default="model.joblib")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=8)
    args = ap.parse_args()

    all_X, all_meta, all_y, all_lang = [], [], [], []
    for d in args.data_dirs:
        X, meta, y = build_feature_matrix(d)
        if y is None:
            raise ValueError(f"{d}/labels.csv has no 'label' column -- can't train on it.")
        lang = d.rstrip("/").split("/")[-1]
        meta = meta.copy()
        meta["lang"] = lang
        meta["group"] = lang + "_" + meta["turn_id"].astype(str)  # group by turn for CV split
        all_X.append(X)
        all_meta.append(meta)
        all_y.append(y)
        all_lang.append(np.full(len(y), lang))

    # align columns across languages (union of feature names, fill missing with 0)
    X = pd.concat(all_X, axis=0, ignore_index=True).fillna(0.0)
    meta = pd.concat(all_meta, axis=0, ignore_index=True)
    y = np.concatenate(all_y)
    lang_arr = np.concatenate(all_lang)

    feature_names = list(X.columns)
    print(f"[train] pooled: {len(X)} pauses, {y.sum()} eot / {len(y)-y.sum()} hold, "
          f"{X.shape[1]} features, langs={sorted(set(lang_arr))}")

    base_clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)

    # grouped CV (by turn) so pauses from the same turn never span train/val --
    # otherwise CV metrics would be optimistic (turn-level leakage)
    groups = meta["group"].values
    n_splits = min(5, pd.Series(groups).nunique())
    if n_splits >= 2 and len(np.unique(y)) == 2:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_pred = cross_val_predict(
            clf, X.values, y, cv=sgkf, groups=groups, method="predict_proba", n_jobs=-1
        )[:, 1]
        auc = roc_auc_score(y, oof_pred)
        brier = brier_score_loss(y, oof_pred)
        print(f"[train] out-of-fold AUC={auc:.3f}  Brier={brier:.3f}  (grouped {n_splits}-fold CV)")
        for lg in sorted(set(lang_arr)):
            mask = lang_arr == lg
            if len(set(y[mask])) == 2:
                auc_l = roc_auc_score(y[mask], oof_pred[mask])
                print(f"[train]   {lg}: AUC={auc_l:.3f}  n={mask.sum()}")
    else:
        print("[train] not enough groups/classes for CV -- skipping OOF metrics")

    # final fit on all data
    clf.fit(X.values, y)

    joblib.dump({"model": clf, "feature_names": feature_names}, args.model_out)
    print(f"[train] saved model -> {args.model_out}")


if __name__ == "__main__":
    main()
