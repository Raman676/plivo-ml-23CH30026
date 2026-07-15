The model extracts causal prosodic features (pitch trend, energy decay,
spectral/MFCC shape, per-turn baseline-normalized deltas) from the 1.5s of
audio before each pause, feeding a calibrated RandomForest trained on pooled
English+Hindi data. On a held-out 80/20 split by turn, it reaches AUC 0.82
and 640ms mean delay on Hindi (vs 1600ms baseline), but only AUC 0.53 and
~1100ms on English, barely above the 0.51 baseline. I ruled out three
explanations for the English gap: pooling hurting it (a solo English model
scored identically), window truncation (English actually has more pre-pause
audio than Hindi on average), and a pause-history shortcut (n_prior_pauses
doesn't appear in the top-15 feature importances). Feature importances are
flat and diffuse across all 47 features rather than concentrated on any
strong cue, suggesting the finality prosodic signal is genuinely weaker or
noisier in this English recording set. With more time I'd inspect the raw
English audio directly for compression artifacts, speaker-style variation,
or a systematically flatter intonation pattern that might be masking the cue,
and would try a language-conditioned feature (or separate calibration) rather
than more feature engineering, since the diagnostic trail points to a data
characteristic rather than a fixable modeling bug.
