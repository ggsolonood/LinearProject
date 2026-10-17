# Training Report — session_20261017_033724

**Generated at**: 2026-10-17 03:37:37  

---

## Dataset
| Split | Samples |
|---|:---:|
| Train | 278 |
| Validation | 60 |
| Test (Unseen) | 59 |
| **Total** | **397** |

---

## Search Configuration
| Category | Parameter | Value |
|---|---|---|
| **Tunable** | `clip_percentile` | `[95.0, 98.0, 99.0]` |
| | `learning_rate` | `[0.001, 0.005, 0.01, 0.05, 0.1]` |
| | `batch_size` | `[4, 8, 16, 32]` |
| **Fixed** | `k` | `8.0` |
| | `n_svd_features` | `64` |
| | `blur_ksize` | `5` |
| | `max_epochs` | `500` |
| | `weight_init` | `xavier` |
| **Total Trials** | — | **60** |

---

## 🏆 Champion Model — Trial 12

| Parameter | Value |
|---|---|
| `clip_percentile` | `95.0` |
| `learning_rate` | `0.01` |
| `batch_size` | `32` |
| Best Epoch | 15 / 40 |
| Early Stopped | Yes |

### Validation Metrics
| Metric | Value |
|---|---|
| MSE | `1.075421` |
| MAE | `1.2062` กล่อง |
| R² | `0.5115` |
| Accuracy (Exact) | `23.33%` |
| Accuracy (±1 box) | `43.33%` |

### Test Set Metrics (Unseen Data)
| Metric | Value |
|---|---|
| MSE | `1.450944` |
| MAE | `1.4355` กล่อง |
| R² | `0.4099` |
| Accuracy (Exact) | `16.95%` |
| Accuracy (±1 box) | `37.29%` |

---

## Artifacts
```
model/trained_models/session_20261017_033724/
├── best_weights.json
├── svd_model.npz
├── pipeline_config.json
├── manifest.json
├── leaderboard.csv
└── training_report.md
```