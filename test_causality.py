"""
test_causality.py

Proof-of-causality check: for a batch of real (pause_start) points, we
extract features twice -- once on the true audio, once on a copy where
everything AFTER pause_start has been replaced with random noise / silence
/ or reversed audio. If the causality rule is respected, features must be
IDENTICAL in both cases, because extract_features_for_pause must never read
past pause_start.

Run: python test_causality.py --data_dir eot_data/english
"""
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

from eot_features import extract_features_for_pause
from eot_dataset import load_labels


def corrupt_future(y_full, sr, pause_start, mode="noise"):
    end_sample = int(pause_start * sr)
    y_corrupt = y_full.copy()
    future = y_corrupt[end_sample:]
    if mode == "noise":
        y_corrupt[end_sample:] = np.random.default_rng(0).normal(0, 1.0, size=future.shape).astype(y_full.dtype)
    elif mode == "silence":
        y_corrupt[end_sample:] = 0.0
    elif mode == "reverse":
        y_corrupt[end_sample:] = future[::-1]
    return y_corrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_checks", type=int, default=200)
    args = ap.parse_args()

    df = load_labels(args.data_dir)
    checks = 0
    max_abs_diff = 0.0
    failures = []

    audio_cache = {}
    for turn_id, group in df.groupby("turn_id", sort=False):
        group = group.sort_values("pause_index")
        audio_file = group["audio_file"].iloc[0]
        import os
        wav_path = os.path.join(args.data_dir, audio_file)
        if wav_path not in audio_cache:
            y_full, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if y_full.ndim > 1:
                y_full = y_full.mean(axis=1)
            audio_cache[wav_path] = (y_full, sr)
        y_full, sr = audio_cache[wav_path]

        prior_pauses = []
        for _, row in group.iterrows():
            if checks >= args.max_checks:
                break
            pause_start = float(row["pause_start"])

            feats_true = extract_features_for_pause(y_full, sr, pause_start, prior_pauses)

            for mode in ("noise", "silence", "reverse"):
                y_bad = corrupt_future(y_full, sr, pause_start, mode=mode)
                feats_bad = extract_features_for_pause(y_bad, sr, pause_start, prior_pauses)
                for k in feats_true:
                    d = abs(feats_true[k] - feats_bad.get(k, np.nan))
                    max_abs_diff = max(max_abs_diff, d)
                    if d > 1e-9:
                        failures.append((turn_id, row["pause_index"], mode, k, feats_true[k], feats_bad.get(k)))
            checks += 1

            pe = float(row["pause_end"]) if "pause_end" in row and not pd.isna(row["pause_end"]) else pause_start
            prior_pauses.append({"pause_start": pause_start, "pause_end": pe})
        if checks >= args.max_checks:
            break

    print(f"Checked {checks} pauses x 3 corruption modes.")
    print(f"Max absolute feature difference (should be 0.0): {max_abs_diff:.2e}")
    if failures:
        print(f"CAUSALITY VIOLATION: {len(failures)} feature(s) changed when future audio was corrupted.")
        for f in failures[:10]:
            print("  ", f)
        raise SystemExit(1)
    else:
        print("PASS: features are provably unaffected by anything after pause_start.")


if __name__ == "__main__":
    main()
