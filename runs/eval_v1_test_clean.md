# v1 on data/aadd_test_clean

model: `v1`  features: `png`  data: `data/aadd_test_clean`  files: 15720

| class | precision | recall | F1 | ROC-AUC | AP | support |
|---|---|---|---|---|---|---|
| AddGaussianNoise | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6384 |
| PitchShift | 0.883 | 0.992 | 0.934 | 1.000 | 0.994 | 6523 |
| TimeStretch | 0.913 | 0.972 | 0.941 | 0.995 | 0.985 | 6439 |
| Shift | 0.000 | 0.000 | 0.000 | 0.605 | 0.076 | 6522 |
| ClippingDistortion | 0.837 | 0.893 | 0.864 | 0.973 | 0.915 | 6417 |
| LowPassFilter | 0.983 | 1.000 | 0.991 | 1.000 | 1.000 | 6436 |
| HighPassFilter | 0.985 | 1.000 | 0.992 | 1.000 | 1.000 | 6606 |
| None | 0.896 | 0.957 | 0.926 | 0.918 | 0.894 | 63609 |

| metric | value |
|---|---|
| f1_micro | 0.9081 |
| f1_macro | 0.8312 |
| f1_macro_degradations | 0.8177 |
| f1_weighted | 0.8804 |
| f1_samples | 0.9063 |
| exact_match | 0.9057 |
| label_accuracy | 0.9771 |
| argmax_accuracy | 0.9084 |
| n_chunks | 108936 |

File level (true # degraded seconds vs predicted AADD count): n_files=15720, spearman=0.932, pearson=0.941, mae=0.552
