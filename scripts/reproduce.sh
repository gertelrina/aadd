#!/usr/bin/env bash
# Full reproduction: LibriSpeech -> synthetic degradations -> features -> v2 model -> reports.
# ~25 min on an Apple M3 Pro (data 1 min, features 3 min, training ~15 min, eval ~5 min).
set -euo pipefail
DATA=${DATA:-data}
mkdir -p "$DATA" feats runs

for split in dev-clean test-clean; do
  if [ ! -d "$DATA/LibriSpeech/$split" ]; then
    curl -L -o "$DATA/$split.tar.gz" "https://www.openslr.org/resources/12/$split.tar.gz"
    tar xzf "$DATA/$split.tar.gz" -C "$DATA"
  fi
done

# v1/v2 protocol: 5 degraded copies + the clean original, one random degradation per second with p=0.5
aadd make-data "$DATA/LibriSpeech/dev-clean"  --out "$DATA/aadd_train_dev_clean" --copies 5 --seed 0
aadd make-data "$DATA/LibriSpeech/test-clean" --out "$DATA/aadd_test_clean"      --copies 5 --seed 1

aadd featurize "$DATA/aadd_train_dev_clean" --out feats/train_mel.npz --features mel
aadd featurize "$DATA/aadd_test_clean"      --out feats/test_mel.npz  --features mel

aadd train --train feats/train_mel.npz --test feats/test_mel.npz --out my_models/aadd_v2_retrained.pt \
           --epochs 10 --batch-size 128 --lr 5e-4 --log runs/v2_train_log.json

aadd eval "$DATA/aadd_test_clean" --model v2 --workers 8 --out runs/eval_v2_test_clean
aadd eval "$DATA/aadd_test_clean" --model v1 --workers 8 --out runs/eval_v1_test_clean
aadd eval "$DATA/aadd_test_clean" --baselines --out runs/eval_baselines_test_clean
