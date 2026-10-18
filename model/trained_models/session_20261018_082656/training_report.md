# Training Report — session_20261018_082656

**Generated at**: 2026-10-18 08:27:13  
**Model Type**: Multi-class Classification (กล่อง 1–9)  

---

## Dataset
| Split | Samples |
|---|:---:|
| Train (raw) | 501 |
| Train (effective, w/ aug) | 5010 |
| Validation | 107 |
| Test (Unseen) | 108 |
| **Total** | **716** |

---

## Feature Engineering
| Component | Dimension |
|---|---|
| SVD Components | 64 |
| Statistical Features | 3 (Edge Area, Contour Count, Aspect Ratio) |
| **Total Input Dim** | **67** |

---

## Search Configuration
| Category | Parameter | Value |
|---|---|---|
| **Tunable** | `clip_percentile` | `[95.0, 98.0, 99.0]` |
| | `learning_rate` | `[0.001, 0.005, 0.01, 0.05]` |
| | `batch_size` | `[8, 16, 32]` |
| | `hidden_dim` | `[32, 64, 128]` |
| **Fixed** | `n_classes` | `9` |
| | `n_svd_features` | `64` |
| | `blur_ksize` | `5` |
| | `max_epochs` | `500` |
| | `dropout_rate` | `0.3` |
| | `weight_init` | `xavier` |
| **Total Trials** | — | **108** |

---

## 🏆 Champion Model — Trial 61

| Parameter | Value |
|---|---|
| `clip_percentile` | `98.0` |
| `learning_rate` | `0.01` |
| `batch_size` | `32` |
| `hidden_dim` | `32` |
| Best Epoch | 62 / 92 |
| Early Stopped | Yes |

### Validation Metrics
| Metric | Value |
|---|---|
| Cross-Entropy | `1.351935` |
| Accuracy (Exact) | `52.34%` |
| Accuracy (±1 box) | `71.96%` |
| Macro F1 | `0.5166` |

### Test Set Metrics (Unseen Data)
| Metric | Value |
|---|---|
| Cross-Entropy | `1.654384` |
| Accuracy (Exact) | `41.67%` |
| Accuracy (±1 box) | `62.96%` |
| Macro F1 | `0.4253` |

### Confusion Matrix (Test Set)
*(Rows = True Label, Cols = Predicted Label, 1-indexed)*

| True \ Pred | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| **1** | 10 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 |
| **2** | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3** | 1 | 0 | 5 | 0 | 0 | 1 | 0 | 0 | 1 |
| **4** | 0 | 0 | 3 | 2 | 1 | 0 | 1 | 2 | 4 |
| **5** | 0 | 0 | 5 | 2 | 0 | 1 | 2 | 0 | 0 |
| **6** | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 0 | 0 |
| **7** | 0 | 0 | 0 | 5 | 0 | 1 | 1 | 6 | 6 |
| **8** | 0 | 0 | 0 | 1 | 2 | 2 | 3 | 3 | 4 |
| **9** | 0 | 0 | 0 | 1 | 2 | 0 | 1 | 1 | 11 |

---

## Artifacts
```
/content/parcel-box-counter/model/trained_models/session_20261018_082656/
├── best_weights.json
├── svd_model.npz
├── pipeline_config.json
├── manifest.json
├── leaderboard.csv
└── training_report.md
```