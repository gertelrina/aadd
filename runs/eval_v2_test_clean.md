# v2 on data/aadd_test_clean

model: `v2`  features: `mel`  data: `data/aadd_test_clean`  files: 15720

| class | precision | recall | F1 | ROC-AUC | AP | support |
|---|---|---|---|---|---|---|
| AddGaussianNoise | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 6384 |
| PitchShift | 0.839 | 0.998 | 0.912 | 1.000 | 0.993 | 6523 |
| TimeStretch | 0.989 | 0.948 | 0.968 | 0.996 | 0.984 | 6439 |
| Shift | 0.692 | 0.763 | 0.726 | 0.975 | 0.795 | 6522 |
| ClippingDistortion | 0.784 | 0.856 | 0.818 | 0.971 | 0.880 | 6417 |
| LowPassFilter | 0.999 | 0.996 | 0.997 | 1.000 | 1.000 | 6436 |
| HighPassFilter | 0.984 | 1.000 | 0.992 | 1.000 | 1.000 | 6606 |
| None | 0.960 | 0.925 | 0.942 | 0.982 | 0.983 | 63609 |

| metric | value |
|---|---|
| f1_micro | 0.9300 |
| f1_macro | 0.9193 |
| f1_macro_degradations | 0.9160 |
| f1_weighted | 0.9312 |
| f1_samples | 0.9288 |
| exact_match | 0.9267 |
| label_accuracy | 0.9825 |
| argmax_accuracy | 0.9301 |
| n_chunks | 108936 |

File level (true # degraded seconds vs predicted AADD count): n_files=15720, spearman=0.938, pearson=0.956, mae=0.404
