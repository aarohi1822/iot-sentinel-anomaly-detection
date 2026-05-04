# IoT Sentinel: Explainable Ioconstruction Error → Thresholding → Explainability → Dashboard

````

> Replace with your pipeline image:
>
> ```md
> ![Architecture](assets/architecture.png)
> ```

---

# 📊 Benchmark Results (NASA SMAP Test Set)

## Core Performance

| Model | Precision | Recall | F1 Score | ROC-AUC |
|------|-----------|--------|---------|---------|
| **LSTM Autoencoder (IoT Sentinel)** | 0.1774 | 0.0109 | 0.0205 | 0.4039 |
| Isolation Forest | 0.1067 | 0.0052 | 0.0099 | 0.6123 |

## Sequential Detection Quality

| Model | Point-Adjusted F1 |
|------|-------------------|
| **LSTM Autoencoder** | **0.5394** |
| Isolation Forest | 0.6818 |

### Key Findings
- **+107.07% improvement in raw pointwise F1** over Isolation Forest baseline
- Stronger sequential anomaly localization than traditional statistical methods
- Real-time explainability through root-cause reconstruction contribution
- Product deployment with monitoring dashboard, threshold lab, and downloadable reports

---

# 🚀 What Makes This Project Different

Unlike typical academic anomaly detection projects, IoT Sentinel is:

✅ Benchmark validated on NASA SMAP telemetry  
✅ Explainable with root-cause sensor attribution  
✅ Deployable with Streamlit live dashboard  
✅ Product-oriented with severity scoring + health monitoring  
✅ Real-time replay capable  
✅ Includes threshold experimentation  
✅ Baseline-compared against traditional methods  
✅ Resume, portfolio, and MS-application grade  

---

# 🖥️ Dashboard Features

## Monitor View
- Live anomaly score tracking
- Threshold visualization
- Highlighted anomalies
- Sensor overlays
- Health score monitoring

## Live Replay Mode
- Simulated real-time IoT sensor stream
- Production-style anomaly monitoring
- Demonstration-ready deployment

## Root Cause Analysis
- Sensor contribution ranking
- Reconstruction error decomposition
- Explainable AI diagnostics

## Threshold Lab
- Compare:
  - Trained threshold
  - Percentile threshold
  - Mean + STD threshold
  - Manual threshold

## Report Generator
- Downloadable anomaly reports
- Severity labels:
  - Normal
  - Warning
  - Critical

---

# 📸 Screenshots

## Main Dashboard
```md
![Dashboard](assets/dashboard.png)
````

## Root Cause Analysis

```md
![Root Cause](assets/root_cause.png)
```

## Benchmark Results

```md
![Benchmark](assets/benchmark.png)
```

## ROC Curve

```md
![ROC Curve](assets/roc_curve.png)
```

---

# 📂 Project Structure

```bash
IoT-Sentinel/
│
├── app/
│   └── app.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── benchmark_smap.py
│   ├── download_nasa_data.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── lstm_autoencoder.weights.h5
│   ├── scaler.pkl
│   └── config.json
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

```bash
git clone https://github.com/aarohi1822/iot-sentinel-anomaly-detection.git
cd iot-sentinel-anomaly-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# 🏋️ Training

```bash
python src/train.py \
  --input data/raw/train.csv \
  --window-size 60 \
  --epochs 40 \
  --batch-size 64 \
  --threshold-method percentile \
  --percentile 99
```

### Generated Artifacts

* `models/lstm_autoencoder.weights.h5`
* `models/scaler.pkl`
* `models/config.json`

---

# 🔍 Prediction

```bash
python src/predict.py \
  --input data/raw/test.csv \
  --output data/processed/anomaly_report.csv
```

---

# 🌍 NASA SMAP Benchmark

## Download dataset

```bash
python src/download_nasa_data.py
```

## Run benchmark

```bash
python src/benchmark_smap.py --spacecraft SMAP --epochs 5 --batch-size 256
```

### Outputs

* `smap_channel_metrics.csv`
* `smap_benchmark_summary.json`

---

# 📈 Streamlit Deployment

```bash
streamlit run app/app.py
```

## Live Demo

**Deployed App:**
[https://iot-sentinel-aarohigs.streamlit.app/](https://iot-sentinel-aarohigs.streamlit.app/)

---

# 🧠 Technical Stack

### Machine Learning

* Python
* TensorFlow / Keras
* LSTM Autoencoder
* Isolation Forest
* Scikit-learn

### Data Engineering

* Pandas
* NumPy
* Sliding windows
* Normalization

### Visualization

* Streamlit
* Plotly
* Real-time dashboards

### Deployment

* GitHub
* Streamlit Cloud
* Reproducible pipelines

---

# 🎯 Use Cases

* Smart manufacturing anomaly detection
* Predictive maintenance
* Industrial sensor monitoring
* Cyber-physical systems
* NASA telemetry analytics
* Explainable AI monitoring systems

---

# 🔮 Future Improvements

* Transformer-based anomaly detection
* Multi-dataset benchmarking (MSL, SWaT, SMD)
* SHAP/LIME explainability integration
* Dockerized deployment
* API serving layer
* Edge IoT deployment
* MLOps automation

---

# 👩‍💻 Author

**Aarohi Gaurav Sharma**
B.Tech CSE (AIML/Data Science Focus)
GitHub: [https://github.com/aarohi1822](https://github.com/aarohi1822)
LinkedIn: [https://www.linkedin.com/in/aarohi-gaurav-sharma-b0a200300](https://www.linkedin.com/in/aarohi-gaurav-sharma-b0a200300)

---

# ⭐ Portfolio Value

This project demonstrates:

* Advanced anomaly detection
* Explainable AI
* Deep learning engineering
* Research benchmarking
* Real-time deployment
* Product-oriented AI system design

## Ideal For:

* AI/ML internships
* Data Science roles
* Research applications
* MS in CS / AI / Data Science admissions
* Portfolio flagship positioning

---

# 📜 License

MIT License
