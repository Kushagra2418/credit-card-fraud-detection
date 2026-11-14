# 🛡️ SENTINEL AI — Credit Card Fraud Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PR-AUC](https://img.shields.io/badge/Champion%20PR--AUC-0.9669-brightgreen)]()
[![Recall](https://img.shields.io/badge/Fraud%20Recall-96.83%25-success)]()

A complete, production-grade **Machine Learning solution for real-time Credit Card Fraud Detection**.  
This project features an end-to-end pipeline: **Exploratory Data Analysis & Jupyter Notebook**, **Multi-Model Benchmark**, an **Automated Publication-Quality PDF Research Report**, a high-throughput **FastAPI REST Backend**, and a responsive **Web Dashboard** with real-time risk scoring and bulk CSV batch scanning.

---

## 📌 1. Problem Formulation & The "Accuracy Trap"

Financial institutions lose billions of dollars annually to credit card fraud while simultaneously struggling to avoid false alarms that decline innocent customers. In production transaction processing, fraudulent transactions account for **under 0.6%** of all activity.

### The "Accuracy Trap"
> **Warning:** A naive model that classifies *every single transaction as legitimate* trivially achieves **99.4% Accuracy**, but catches **0% of fraud**.

To build a reliable fraud detection engine, model selection must be driven by:
1. **PR-AUC (Precision-Recall Area Under the Curve):** The gold-standard metric for extreme class imbalance.
2. **Recall (Sensitivity):** The percentage of actual fraud caught (minimizing missed fraud / financial loss).
3. **Precision:** Minimizing false alarms to avoid cardholder friction.

---

## 📊 2. The 4-Model Benchmark Progression

We systematically benchmarked four algorithmic paradigms across **240,000+ realistic credit card transactions** using an identical out-of-sample test split:

| Model / Paradigm | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC (Gold Standard) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Logistic Regression (Unweighted)** | 97.75% | 84.47% | 42.61% | 0.567 | 0.935 | 0.611 |
| **2. Cost-Sensitive Logistic Regression (Balanced)** | 85.41% | 18.00% | 90.77% | 0.300 | 0.948 | 0.562 |
| **3. Balanced Random Forest** | 98.46% | 70.81% | 94.45% | 0.809 | 0.992 | 0.940 |
| **4. HistGradientBoosting (Champion)** 🏆 | **98.49%** | **70.41%** | **96.83%** | **0.815** | **0.998** | **0.967** |

### Key Findings & Architectural Evolution:
- **Model 1 Failure Mode:** Even with 97.75% accuracy, Model 1 missed **57.39% of all fraud attacks** due to majority-class dominance.
- **Model 2 Tradeoff:** Introducing balanced class penalties surged Recall to **90.77%**, but linear decision boundaries produced excessive false positives.
- **Model 3 & 4 Dominance:** Tree ensembles captured complex multi-variable interactions (e.g. night-time velocity, distance between cardholder and merchant, and high-risk merchant categories), with **HistGradientBoosting** delivering a staggering **0.9669 PR-AUC** and **96.83% Recall**.

---

## 🏗️ 3. System Architecture & Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1: JUPYTER LAB & ML RESEARCH                   │
│                                                                        │
│  File: notebooks/Fraud_Detection_Pipeline.ipynb                        │
│  ├── 1. Exploratory Data Analysis (EDA) & Imbalance Visualizations     │
│  ├── 2. Feature Engineering: Haversine distance, hour, age, categories │
│  ├── 3. Sequential 4-Model Training & Cross-Benchmarking               │
│  ├── 4. ROC, Precision-Recall & Confusion Matrix Generation            │
│  └── 5. Model Serialization ──► Saves .joblib & metadata.json          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Model Artifacts (.joblib)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             PHASE 2: PRODUCTION FULL-STACK DEPLOYMENT                  │
│                                                                        │
│  Backend (FastAPI REST API)         Frontend (Web Dashboard)           │
│  ├── POST /api/predict              ├── 1-Click Preset Loaders         │
│  ├── POST /api/batch-predict        ├── Dynamic Fraud Probability Bar  │
│  ├── GET  /api/metrics              ├── Real-Time Decision Verdict     │
│  └── GET  /api/download-report      └── Drag-and-Drop Batch CSV Scanner│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 4. Project Directory Structure

```text
credit-card-fraud-detection/
├── backend/
│   ├── main.py                     # FastAPI server & route handlers
│   ├── schemas.py                  # Pydantic request/response validation
│   └── inference_service.py        # Vectorized scaling & prediction pipeline
├── frontend/
│   ├── index.html                  # Modern fintech web dashboard
│   ├── css/style.css               # Custom animations & theme styles
│   └── js/app.js                   # API communication & dynamic UI rendering
├── notebooks/
│   └── Fraud_Detection_Pipeline.ipynb  # Interactive Jupyter Lab notebook
├── pipeline/
│   ├── train_and_benchmark.py      # Automated 4-model training script
│   └── generate_pdf_report.py      # Automated PDF report generator
├── models/
│   ├── champion_fraud_model.joblib # Serialized HistGradientBoosting model
│   ├── robust_scaler.joblib        # Preprocessing RobustScaler
│   ├── model_metadata.json         # Benchmark metrics & decision threshold
│   └── sample_presets.json         # Sample transactions for 1-click presets
├── reports/
│   ├── Fraud_Detection_Benchmark_Report.pdf  # Publication-grade PDF report
│   └── figures/                    # Saved ROC, PR, and Confusion Matrix plots
├── data/
│   └── sample_test_batch.csv       # 50-row sample batch for web testing
├── requirements.txt                # Python dependencies
├── run.py                          # 1-Click application launcher
└── README.md                       # Comprehensive documentation
```

---

## 💾 5. Dataset Source & Attribution

- **Dataset Name:** Credit Card Fraud Detection Dataset (Simulated transaction stream by Sparkov Data Generation).
- **Available Features:** `trans_date_trans_time`, `amt`, `category`, `gender`, `lat`, `long`, `merch_lat`, `merch_long`, `city_pop`, `dob`, `is_fraud`.
- **Dataset Source:**
  - Company Dataset Mirror: [Google Drive Link](https://drive.google.com/file/d/1CO2pD42N3MAPPxlBEo1OrCTfc07HFYuM/view?usp=sharing)
  - Kaggle Source: [Kaggle Fraud Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

*(Note: Raw `fraudTrain.csv` and `fraudTest.csv` files (~500 MB) are excluded from GitHub via `.gitignore` to comply with GitHub's file size limits. A sample batch `data/sample_test_batch.csv` is included for instant testing).*

---

## 🚀 6. Setup & Installation Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/kushagra2418/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create and Activate a Virtual Environment
**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 7. How to Run the Project

### Option A: Launch Full-Stack Web Application (Recommended)
Run the single-command launcher:
```bash
python run.py
```
Open your browser to:
- **Web UI Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Download Research PDF Report:** [http://127.0.0.1:8000/api/download-report](http://127.0.0.1:8000/api/download-report)

### Option B: Interactive Jupyter Lab
To inspect the exploratory data analysis, visual curves, and step-by-step model training:
```bash
jupyter lab
```
Navigate to `notebooks/Fraud_Detection_Pipeline.ipynb` and run all cells.

### Option C: Re-run the Automated Pipeline
To re-train models from raw data and re-generate the PDF report:
```bash
python pipeline/train_and_benchmark.py
python pipeline/generate_pdf_report.py
```

---

## 🌐 8. REST API Documentation

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/predict` | `POST` | Real-time transaction fraud scoring with sub-15ms latency. |
| `/api/batch-predict` | `POST` | Vectorized batch inference on uploaded CSV files. |
| `/api/sample-presets` | `GET` | Returns pre-validated test cases (Legit, Fraud, Borderline). |
| `/api/metrics` | `GET` | Returns full 4-model benchmark metrics, confusion matrices, and ROC/PR data. |
| `/api/download-report` | `GET` | Serves the generated PDF research report. |
| `/api/sample-batch-csv` | `GET` | Serves a 50-row CSV file ready for batch upload testing. |

### Example Request (`POST /api/predict`)
```json
{
  "amt": 980.20,
  "category": "shopping_net",
  "gender": "M",
  "city_pop": 45000,
  "lat": 34.0522,
  "long": -118.2437,
  "merch_lat": 40.7128,
  "merch_long": -74.0060,
  "trans_date_trans_time": "2026-11-13 03:15:00",
  "dob": "1975-08-22"
}
```

### Example Response
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9968,
  "risk_score_pct": 99.68,
  "risk_level": "HIGH",
  "action_recommendation": "DECLINE / IMMEDIATE CARD LOCK & SMS VERIFICATION REQUIRED",
  "decision_threshold": 0.35,
  "latency_ms": 11.4,
  "features_computed": {
    "distance_km": 3936.06,
    "hour": 3,
    "age": 51,
    "is_night": true,
    "category": "shopping_net"
  }
}
```

---

## 📄 9. Research PDF Report

A publication-quality 2-page executive and engineering report is automatically compiled in `reports/Fraud_Detection_Benchmark_Report.pdf`.  
It includes:
1. Executive Summary & Problem Formulation.
2. The Mathematical Failure of Accuracy in Extreme Class Imbalances.
3. 4-Model Comparative Table & Performance Bar Chart.
4. Embedded High-Resolution ROC Curves, Precision-Recall Curves, and Confusion Matrices.
5. Production Calibrated Thresholding (0.35) and False-Positive Friction Analysis.

---

## 👤 Author
- **Author:** Kushagra Jain
- **Email:** kushagrajainaj2003@gmail.com
- **GitHub:** [@kushagra2418](https://github.com/kushagra2418)
