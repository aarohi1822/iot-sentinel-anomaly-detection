IoT Sentinel: Explainable IoT Sensor Anomaly Detection
Explainable real-time IoT sensor anomaly detection with root-cause analysis, health scoring, live monitoring, and a reproducible NASA SMAP benchmark.

Features
CSV loading for multivariate sensor streams
MinMax normalization fitted on normal training rows only
Sliding-window sequence generation
LSTM autoencoder trained only on label=0 rows
Reconstruction-error anomaly scoring
Moving-average smoothing to reduce noisy spikes
Saved training threshold with optional manual tuning
Streamlit dashboard with Plotly visualization
Downloadable anomaly report
Real-time replay mode for live-monitoring demos
Root-cause sensor explanation for each anomaly
Health score, warning/critical severity labels, and threshold comparison
Dashboard
The Streamlit app is organized like a monitoring product:

Monitor: anomaly score, threshold line, highlighted anomalies, and sensor overlay
Live Replay: simulates real-time IoT sensor streaming from a CSV
Root Cause: ranks sensors by reconstruction-error contribution
Threshold Lab: compares trained, percentile, mean/std, and manual thresholds
Report: downloadable anomaly report with severity and root-cause sensor
Architecture
Data → Preprocessing → LSTM Autoencoder → Reconstruction Error → Thresholding → Explainability → Dashboard

Project Structure
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
Benchmark Results
This repository now includes a real SMAP benchmark run instead of only a demo dataset.

Latest recorded benchmark (data/processed/smap_benchmark_summary.json):

Dataset: NASA SMAP benchmark via the public TSLib mirror
LSTM Autoencoder: Precision 0.1774, Recall 0.0109, F1 0.0205, ROC-AUC 0.4039
Isolation Forest baseline: Precision 0.1067, Recall 0.0052, F1 0.0099, ROC-AUC 0.6123
Pointwise F1 lift vs Isolation Forest: +107.07%
Point-adjusted F1: LSTM Autoencoder 0.5394, Isolation Forest 0.6818
This project reports both raw pointwise metrics and point-adjusted F1 so the benchmark is explicit instead of cherry-picked.

What Makes This Different
Unlike typical anomaly detection projects, this one is both product-shaped and benchmarked:

Real benchmark artifacts for NASA SMAP, not just screenshots
Detects anomalies and explains likely root-cause sensors
Simulates real-time IoT streaming in the dashboard
Provides health scoring and anomaly severity levels
Includes threshold experimentation and baseline comparison
Data Format
Use a CSV with numeric sensor columns. If a label column exists, label=0 is treated as normal and used for training. A timestamp column is optional.

Example:

timestamp,sensor_1,sensor_2,sensor_3,label
2026-01-01 00:00:00,0.12,12.4,5.8,0
2026-01-01 00:01:00,0.14,12.6,5.7,0
Sample Output
timestamp | anomaly_score | threshold | severity | root_cause
2026-01-01 00:10 | 0.87 | 0.45 | Critical | vibration

Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Train
Place your training CSV at data/raw/train.csv, then run:

python src/train.py
Useful options:

python src/train.py \
  --input data/raw/train.csv \
  --window-size 60 \
  --epochs 40 \
  --batch-size 64 \
  --threshold-method percentile \
  --percentile 99
Training creates:

models/lstm_autoencoder.weights.h5
models/scaler.pkl
models/config.json
Predict
python src/predict.py --input data/raw/test.csv --output data/processed/anomaly_report.csv
The report contains row indices, anomaly scores, threshold, and timestamp when available.

Benchmark On NASA SMAP
Download the benchmark data:

python src/download_nasa_data.py
Run the SMAP benchmark:

python src/benchmark_smap.py --spacecraft SMAP --epochs 5 --batch-size 256
This creates:

data/processed/smap_channel_metrics.csv
data/processed/smap_benchmark_summary.json
The Streamlit dashboard reads these files automatically in the Benchmark tab.

Streamlit UI
streamlit run app/app.py
Use the included sample data or upload your own CSV. The dashboard includes:

Monitor view with anomaly score and sensor overlays
Live replay mode for real-time IoT simulation
Root-cause view showing the sensor most responsible for each anomaly
Threshold lab comparing trained, percentile, mean/std, and manual thresholds
Benchmark tab with SMAP vs Isolation Forest metrics
Anomaly report with severity and download support
Accuracy Notes
Train on normal rows only by keeping reliable labels in the training CSV.

Use a window size between 50 and 100 for SMAP/MSL-like data; 60 is the default.

Keep the saved training threshold for production inference when possible.

Increase the percentile or threshold when false positives are high.

Decrease the percentile or threshold when anomalies are missed.

Increase smoothing slightly for noisy sensors, but avoid over-smoothing short anomalies.

Use Cases
Smart factory monitoring

Predictive maintenance

IoT sensor anomaly detection

NASA-style telemetry analysis

Live Demo
https://iot-sentinel-aarohigs.streamlit.app/

Streamlit Cloud
Deploy the repository and set the app entry point to:

app/app.py
Train locally first and include the generated models/ artifacts in the deployment, or adapt the app to load hosted model artifacts.
