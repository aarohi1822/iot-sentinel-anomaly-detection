# IoT Sentinel: Explainable IoT Sensor Anomaly Detection

Explainable Real-Time IoT Sensor Anomaly Detection with Root-Cause Analysis, Health Scoring, and Live Monitoring Dashboard.

## Features

- CSV loading for multivariate sensor streams
- MinMax normalization fitted on normal training rows only
- Sliding-window sequence generation
- LSTM autoencoder trained only on `label=0` rows
- Reconstruction-error anomaly scoring
- Moving-average smoothing to reduce noisy spikes
- Saved training threshold with optional manual tuning
- Streamlit dashboard with Plotly visualization
- Downloadable anomaly report
- Real-time replay mode for live-monitoring demos
- Root-cause sensor explanation for each anomaly
- Health score, warning/critical severity labels, and threshold comparison

## Dashboard

The Streamlit app is organized like a monitoring product:

- **Monitor**: anomaly score, threshold line, highlighted anomalies, and sensor overlay
- **Live Replay**: simulates real-time IoT sensor streaming from a CSV
- **Root Cause**: ranks sensors by reconstruction-error contribution
- **Threshold Lab**: compares trained, percentile, mean/std, and manual thresholds
- **Report**: downloadable anomaly report with severity and root-cause sensor

## Project Structure

```text
data/
  raw/
  processed/
models/
src/
  data_loader.py
  preprocess.py
  model.py
  train.py
  predict.py
  utils.py
app/
  app.py
requirements.txt
README.md
```
## What Makes This Different

Unlike typical anomaly detection projects:
- Detects anomalies AND explains root cause
- Simulates real-time IoT streaming
- Provides system health scoring
- Includes threshold experimentation lab

  
## Data Format

Use a CSV with numeric sensor columns. If a `label` column exists, `label=0` is treated as normal and used for training. A timestamp column is optional.

Example:

```csv
timestamp,sensor_1,sensor_2,sensor_3,label
2026-01-01 00:00:00,0.12,12.4,5.8,0
2026-01-01 00:01:00,0.14,12.6,5.7,0
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Place your training CSV at `data/raw/train.csv`, then run:

```bash
python src/train.py
```

Useful options:

```bash
python src/train.py \
  --input data/raw/train.csv \
  --window-size 60 \
  --epochs 40 \
  --batch-size 64 \
  --threshold-method percentile \
  --percentile 99
```

Training creates:

- `models/lstm_autoencoder.keras`
- `models/scaler.pkl`
- `models/config.json`

## Predict

```bash
python src/predict.py --input data/raw/test.csv --output data/processed/anomaly_report.csv
```

The report contains row indices, anomaly scores, threshold, and timestamp when available.

## Streamlit UI

```bash
streamlit run app/app.py
```

Use the included sample data or upload your own CSV. The dashboard includes:

- Monitor view with anomaly score and sensor overlays
- Live replay mode for real-time IoT simulation
- Root-cause view showing the sensor most responsible for each anomaly
- Threshold lab comparing trained, percentile, mean/std, and manual thresholds
- Anomaly report with severity and download support

## Accuracy Notes

- Train on normal rows only by keeping reliable labels in the training CSV.
- Use a window size between 50 and 100 for SMAP/MSL-like data; `60` is the default.
- Keep the saved training threshold for production inference when possible.
- Increase the percentile or threshold when false positives are high.
- Decrease the percentile or threshold when anomalies are missed.
- Increase smoothing slightly for noisy sensors, but avoid over-smoothing short anomalies.

## Streamlit Cloud

Deploy the repository and set the app entry point to:

```text
app/app.py
```

Train locally first and include the generated `models/` artifacts in the deployment, or adapt the app to load hosted model artifacts.
