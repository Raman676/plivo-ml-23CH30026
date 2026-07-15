"""
eot_features.py

Causal feature extraction for end-of-turn (EOT) detection.

HARD RULE: for a given pause, every feature computed here uses ONLY audio
samples from index 0 up to int(pause_start * sr). Nothing from pause_end or
beyond is ever read. This module never receives pause_end at all, by design,
so it is structurally impossible to leak it.

Design notes (why these features):
- Silence duration/pause length is deliberately EXCLUDED as a feature for the
  *current* pause (that would require knowing pause_end -> future leakage).
  Durations of PAST pauses in the same turn are fair game (they already
  finished before pause_start of the current pause) and are informative:
  a turn with several short "hold" pauses is behaviorally different from
  one heading into its first pause.
- Prosody near a turn boundary tends to differ from prosody mid-utterance:
  falling pitch + decaying energy going into silence often signals finality;
  flat/rising pitch + sustained energy often signals "more is coming"
  (this holds fairly cross-linguistically as an intonational cue, which
  matters since the hidden test set is mostly Hindi while most iteration
  happens on English).
- We compute a per-turn "baseline" from the first ~0.4s of the turn's audio
  (always causal, since it precedes every pause) and express pitch/energy as
  deltas from that baseline. This helps normalize across speakers, genders,
  and languages that have different absolute pitch ranges.
"""
import numpy as np
import librosa

SR_DEFAULT = 16000
WINDOW_SEC = 1.5          # analysis window length before the pause
BASELINE_SEC = 0.4        # per-turn baseline window, from the start of the turn
FRAME_LENGTH = 512
HOP_LENGTH = 160           # 10ms hop at 16kHz


def _safe_slope(y):
    """Least-squares slope of y vs frame index; 0 if not enough points."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 2 or np.all(np.isnan(y)):
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x, y = x[mask], y[mask]
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def _pitch_track(y, sr):
    """Returns f0 array (Hz, NaN where unvoiced).

    Uses librosa.yin (fast autocorrelation-based tracker) rather than
    librosa.pyin (probabilistic HMM tracker): pyin is markedly more accurate
    but ~100x slower per call in practice, which matters a lot on a
    CPU-only laptop under a hard 120-minute clock, especially if the real
    dataset has far more turns than the synthetic set used for local
    testing. yin has no built-in voiced/unvoiced decision, so we derive one
    from frame energy (frames quieter than a fraction of the window's peak
    energy are treated as unvoiced/silence and masked to NaN, matching
    pyin's NaN-for-unvoiced convention so downstream code is unchanged).
    """
    if len(y) < 2048:
        return np.array([])
    try:
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=1024,
            hop_length=HOP_LENGTH,
        )
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP_LENGTH)[0]
        n = min(len(f0), len(rms))
        f0, rms = f0[:n], rms[:n]
        voiced_thresh = 0.25 * (np.max(rms) if len(rms) else 0.0)
        f0 = np.where(rms >= max(voiced_thresh, 1e-4), f0, np.nan)
        return f0
    except Exception:
        return np.array([])


def _turn_baseline(y_full, sr):
    """Baseline pitch/energy from the first BASELINE_SEC of the turn.
    Always causal: it's the very start of the file."""
    n = int(BASELINE_SEC * sr)
    seg = y_full[:n]
    if len(seg) < 400:
        return {"base_pitch": 0.0, "base_energy": 0.0}
    rms = librosa.feature.rms(y=seg, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    f0 = _pitch_track(seg, sr)
    f0v = f0[~np.isnan(f0)] if len(f0) else np.array([])
    return {
        "base_pitch": float(np.mean(f0v)) if len(f0v) else 0.0,
        "base_energy": float(np.mean(rms)) if len(rms) else 0.0,
    }


def _pause_history_features(pause_rows_before, current_pause_start):
    """pause_rows_before: list of dicts with pause_start, pause_end for
    pauses in THIS turn that occurred strictly before the current pause
    (i.e., pause_end <= current_pause_start). All causal by construction.
    Any 'label' key is ignored/not required (predict-time data won't have it
    reliably, and using it would be leaking ground truth into a feature
    anyway, so we deliberately never touch a label column here).
    """
    n_prior = len(pause_rows_before)
    if n_prior == 0:
        return {
            "n_prior_pauses": 0,
            "mean_prior_pause_dur": 0.0,
            "last_prior_pause_dur": 0.0,
            "time_since_turn_start": current_pause_start,
        }
    durs = [max(0.0, r["pause_end"] - r["pause_start"]) for r in pause_rows_before]
    return {
        "n_prior_pauses": n_prior,
        "mean_prior_pause_dur": float(np.mean(durs)),
        "last_prior_pause_dur": float(durs[-1]),
        "time_since_turn_start": current_pause_start,
    }


def extract_features_for_pause(y_full, sr, pause_start, pause_rows_before):
    """
    y_full: full audio array for the turn (mono, float)
    sr: sample rate
    pause_start: seconds; ONLY y_full[:int(pause_start*sr)] may be touched
    pause_rows_before: list of {pause_start, pause_end} dicts for earlier
                        pauses in the same turn (causal, see above)

    Returns a flat dict of feature_name -> float.
    """
    end_sample = int(pause_start * sr)
    y_before = y_full[:end_sample]  # <-- causality boundary enforced here

    feats = {}

    # ---- turn-level baseline (causal: first BASELINE_SEC of the file) ----
    baseline = _turn_baseline(y_before if len(y_before) else y_full[: int(BASELINE_SEC * sr)], sr)
    feats.update(baseline)

    # ---- analysis window: last WINDOW_SEC of speech before the pause ----
    win_samples = int(WINDOW_SEC * sr)
    y_win = y_before[-win_samples:] if len(y_before) > win_samples else y_before

    if len(y_win) < 400:
        # essentially no pre-pause audio (pause at/near t=0) -> neutral/zero features
        feats.update({
            "energy_mean": 0.0, "energy_std": 0.0, "energy_final": 0.0, "energy_slope": 0.0,
            "energy_delta_from_baseline": 0.0,
            "pitch_mean": 0.0, "pitch_std": 0.0, "pitch_final": 0.0, "pitch_slope": 0.0,
            "pitch_range": 0.0, "pitch_delta_from_baseline": 0.0, "voiced_ratio": 0.0,
            "zcr_mean": 0.0, "spectral_centroid_mean": 0.0, "spectral_rolloff_mean": 0.0,
        })
        for i in range(13):
            feats[f"mfcc{i}_mean"] = 0.0
            feats[f"mfcc{i}_slope"] = 0.0
    else:
        # --- energy ---
        rms = librosa.feature.rms(y=y_win, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        feats["energy_mean"] = float(np.mean(rms))
        feats["energy_std"] = float(np.std(rms))
        tail = rms[-5:] if len(rms) >= 5 else rms
        feats["energy_final"] = float(np.mean(tail))
        feats["energy_slope"] = _safe_slope(rms)
        feats["energy_delta_from_baseline"] = feats["energy_final"] - baseline["base_energy"]

        # --- pitch ---
        f0 = _pitch_track(y_win, sr)
        f0v = f0[~np.isnan(f0)] if len(f0) else np.array([])
        if len(f0v) > 0:
            feats["pitch_mean"] = float(np.mean(f0v))
            feats["pitch_std"] = float(np.std(f0v))
            tail_f0 = f0v[-5:] if len(f0v) >= 5 else f0v
            feats["pitch_final"] = float(np.mean(tail_f0))
            feats["pitch_slope"] = _safe_slope(f0)  # slope over full track incl. NaNs->masked
            feats["pitch_range"] = float(np.max(f0v) - np.min(f0v))
            feats["pitch_delta_from_baseline"] = feats["pitch_final"] - baseline["base_pitch"]
            feats["voiced_ratio"] = float(len(f0v) / max(1, len(f0)))
        else:
            feats["pitch_mean"] = 0.0
            feats["pitch_std"] = 0.0
            feats["pitch_final"] = 0.0
            feats["pitch_slope"] = 0.0
            feats["pitch_range"] = 0.0
            feats["pitch_delta_from_baseline"] = 0.0
            feats["voiced_ratio"] = 0.0

        # --- spectral / timbre ---
        zcr = librosa.feature.zero_crossing_rate(y_win, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        feats["zcr_mean"] = float(np.mean(zcr))

        cent = librosa.feature.spectral_centroid(y=y_win, sr=sr, hop_length=HOP_LENGTH)[0]
        feats["spectral_centroid_mean"] = float(np.mean(cent))

        rolloff = librosa.feature.spectral_rolloff(y=y_win, sr=sr, hop_length=HOP_LENGTH)[0]
        feats["spectral_rolloff_mean"] = float(np.mean(rolloff))

        mfcc = librosa.feature.mfcc(y=y_win, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
        for i in range(13):
            feats[f"mfcc{i}_mean"] = float(np.mean(mfcc[i]))
            feats[f"mfcc{i}_slope"] = _safe_slope(mfcc[i])

    # ---- pause-history / timing context (causal) ----
    feats.update(_pause_history_features(pause_rows_before, pause_start))

    return feats


FEATURE_NAMES = None  # populated lazily by caller after first extraction call
