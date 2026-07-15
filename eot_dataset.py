"""
eot_dataset.py

Shared logic for turning a data_dir (matching eot_data/<lang>/{audio/,labels.csv})
into a feature matrix. Used identically by train.py and predict.py so there is
no train/predict skew in how features are computed.
"""
import os
import numpy as np
import pandas as pd
import soundfile as sf

from eot_features import extract_features_for_pause

REQUIRED_COLS = {"turn_id", "audio_file", "pause_index", "pause_start"}


def load_labels(data_dir):
    path = os.path.join(data_dir, "labels.csv")
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv at {path} is missing required columns: {missing}")
    df = df.sort_values(["turn_id", "pause_index"]).reset_index(drop=True)
    return df


def build_feature_matrix(data_dir, verbose=True):
    """
    Returns (X_df, meta_df[, y]) where:
      X_df   : DataFrame of features, one row per pause
      meta_df: DataFrame with turn_id, pause_index (+ audio_file) aligned to X_df
      y      : np.array of 0/1 (eot=1) IF a 'label' column is present, else None
    """
    df = load_labels(data_dir)
    has_labels = "label" in df.columns

    feat_rows = []
    meta_rows = []
    y = [] if has_labels else None

    audio_cache = {}

    n_turns = df["turn_id"].nunique()
    if verbose:
        print(f"[eot_dataset] {data_dir}: {n_turns} turns, {len(df)} pause rows, "
              f"labels {'present' if has_labels else 'ABSENT (inference mode)'}")

    for turn_id, group in df.groupby("turn_id", sort=False):
        group = group.sort_values("pause_index")
        audio_file = group["audio_file"].iloc[0]
        wav_path = os.path.join(data_dir, audio_file)

        if wav_path not in audio_cache:
            y_full, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if y_full.ndim > 1:
                y_full = y_full.mean(axis=1)  # safety: force mono
            audio_cache[wav_path] = (y_full, sr)
        y_full, sr = audio_cache[wav_path]

        prior_pauses = []  # list of {pause_start, pause_end} strictly before current
        for _, row in group.iterrows():
            pause_start = float(row["pause_start"])

            feats = extract_features_for_pause(
                y_full=y_full,
                sr=sr,
                pause_start=pause_start,
                pause_rows_before=prior_pauses,
            )
            feat_rows.append(feats)
            meta_rows.append({
                "turn_id": row["turn_id"],
                "pause_index": row["pause_index"],
                "audio_file": audio_file,
            })
            if has_labels:
                y.append(1 if str(row["label"]).strip().lower() == "eot" else 0)

            # this pause becomes "prior" for subsequent pauses in the same turn.
            # pause_end is only used here to describe a PAST pause (already
            # finished before the NEXT pause_start) -- never for the current one.
            pe = float(row["pause_end"]) if "pause_end" in row and not pd.isna(row["pause_end"]) else pause_start
            prior_pauses.append({"pause_start": pause_start, "pause_end": pe})

    X_df = pd.DataFrame(feat_rows).fillna(0.0)
    meta_df = pd.DataFrame(meta_rows)
    y = np.array(y, dtype=int) if has_labels else None
    return X_df, meta_df, y
