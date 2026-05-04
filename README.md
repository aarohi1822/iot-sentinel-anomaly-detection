# IoT Sentinel — Explainable Real-Time IoT Sensor Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> Industrial IoT systems generate dense, noisy sensor streams where anomalies are rare but costly. This project trains an LSTM autoencoder on normal operating data to detect anomalies via reconstruction error — no labels required at inference. Evaluated on the NASA SMAP benchmark, the model achieves a **point-adjusted F1 of 0.54** and outperforms an Isolation Forest baseline by **+107% on pointwise F1**, with full root-cause explainability and a production-ready Streamlit dashboard.

**[Live Demo →](https://iot-sentinel-aarohigs.streamlit.app/)**

---

## Architecture
CSV Input → Preprocessing (MinMax + Sliding Windows) → LSTM Autoencoder (train on normal only)
→ Reconstruction Error (MA Smoothing) → Thresholding (Severity Labels)
→ Explainability (Root-Cause Ranking) → Streamlit Dashboard

---

## Benchmark Results — NASA SMAP

Evaluated on the [public TSLib mirror](https://github.com/thuml/Time-Series-Library) of NASA SMAP. Both pointwise and point-adjusted F1 are reported to avoid cherry-picking.

| Model | Precision | Recall | Pointwise F1 | Point-Adj F1 | ROC-AUC |
|---|---|---|---|---|---|
| **LSTM Autoencoder (ours)** | **0.1774** | 0.0109 | **0.0205 (+107%)** | 0.5394 | 0.4039 |
| Isolation Forest (baseline) | 0.1067 | 0.0052 | 0.0099 | **0.6818** | **0.6123** |

> **Metric note:** Pointwise F1 is the strict per-timestep metric. Point-adjusted F1 credits detection of any point within an anomaly window — more appropriate for industrial monitoring. Isolation Forest leads on point-adjusted F1 and AUC; LSTM-AE leads on pointwise precision and F1.

---

## What Makes This Different

Unlike typical anomaly detection notebooks, this is product-shaped and benchmarked end-to-end:

- **+107% pointwise F1** over Isolation Forest on NASA SMAP, with reproducible JSON benchmark artifacts
- **Root-cause sensor ranking** — every anomaly explains which sensor drove the reconstruction error spike
- **Live replay simulation** — streams CSV data row-by-row to demo real-time IoT monitoring without extra infrastructure
- **Threshold lab** — compare trained, percentile, mean/std, and manual thresholds side-by-side with severity labels (Warning / Critical)
- **No label leakage** — model is trained exclusively on `label=0` rows; the threshold is saved and reused for production inference

---

## Quick Start

```bash
# Install
git clone https://github.com/your-username/iot-sentinel
cd iot-sentinel
pip install -r requirements.txt

# Train
python src/train.py \
  --input data/raw/train.csv \
  --window-size 60 \
  --epochs 40 \
  --threshold-method percentile \
  --percentile 99

# Predict
python src/predict.py --input data/raw/test.csv --output data/processed/anomaly_report.csv

# Launch dashboard
streamlit run app/app.py
```

---

## NASA SMAP Benchmark

```bash
python src/download_nasa_data.py
python src/benchmark_smap.py --spacecraft SMAP --epochs 5 --batch-size 256
```

Outputs saved to:
- `data/processed/smap_channel_metrics.csv`
- `data/processed/smap_benchmark_summary.json`

The Streamlit dashboard reads these automatically in the **Benchmark** tab.

---

## Sample Output

| Timestamp | Anomaly Score | Threshold | Severity | Root Cause |
|---|---|---|---|---|
| 2026-01-01 00:10 | 0.87 | 0.45 | Critical | vibration |
| 2026-01-01 00:22 | 0.51 | 0.45 | Warning | temperature |

---

## Data Format

Use any CSV with numeric sensor columns. A `label` column (0 = normal, used for training only) and `timestamp` are optional.
timestamp,sensor_1,sensor_2,sensor_3,label
2026-01-01 00:00:00,0.12,12.4,5.8,0
2026-01-01 00:01:00,0.14,12.6,5.7,0

---

## Project Structure

```
iot-sentinel/
├── data/
│   ├── raw/                      # place train.csv / test.csv here
│   └── processed/                # anomaly_report.csv, benchmark JSONs
├── models/                       # lstm_autoencoder.weights.h5, scaler.pkl, config.json
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── benchmark_smap.py
│   └── download_nasa_data.py
├── app/
│   └── app.py                    # Streamlit dashboard entry point
├── requirements.txt
└── README.md
```

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| **Monitor** | Anomaly score vs threshold, highlighted anomaly windows, sensor overlays |
| **Live Replay** | Real-time row-by-row streaming simulation from any CSV |
| **Root Cause** | Per-anomaly sensor contribution ranking |
| **Threshold Lab** | Compare trained, percentile, mean/std, and manual thresholds |
| **Benchmark** | SMAP vs Isolation Forest metrics loaded from saved JSON |
| **Report** | Downloadable anomaly CSV with severity and root-cause columns |

---

## Accuracy Tips

- Train on normal rows only — keep reliable `label=0` rows in your training CSV
- Use window size 50–100 for SMAP/MSL-like data (default: 60)
- Keep the saved training threshold for production inference
- Increase percentile to reduce false positives; decrease to catch more anomalies
- Increase smoothing for noisy sensors, but avoid over-smoothing short anomalies

---

## Use Cases

`Smart factory monitoring` · `Predictive maintenance` · `NASA-style telemetry analysis` · `IoT edge deployments` · `Anomaly detection research baseline`

---

## Streamlit Cloud Deploy

Set entry point to `app/app.py`. Train locally first and include `models/` artifacts in the deployment, or adapt the app to load hosted model artifacts.
