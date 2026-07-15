# Run Log

| time | change | score.py result | notes |
|---|---|---|---|
| baseline | silence-only (given) | AUC=0.514, delay=1600ms @0% interrupted | confirms duration alone is weak |
| model v1 | causal prosody features (pitch/energy/mfcc/spectral) + calibrated RandomForest, pooled en+hi, held-out 80/20 val | english: AUC=0.527, delay~1175ms | hindi: AUC=0.824, delay=640ms | large en/hi gap found |
| diagnosis | tested 3 hypotheses for english gap | ruled out: pooling (solo-lang model = same AUC), window truncation (english has MORE pre-pause audio than hindi), pause-history shortcut (not in top-15 importances) | importances flat/diffuse -> genuinely weak signal in english audio, not a modeling bug |
| final | retrained on 100% of english+hindi data (train.py --data_dirs eot_handout/eot_data/english eot_handout/eot_data/hindi) | expected: english ~1100-1175ms, hindi ~640ms (from held-out val) | this is the submitted model.joblib and predictions_*.csv |
