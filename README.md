# 🩺 Parkinson's Disease Detection from Voice Measurements
### Biomedical Machine Learning Classification System | College Capstone Project

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Academic Affiliation & Project Overview
- **Institution:** Institute of Engineering and Technology, Devi Ahilya Vishwavidyalaya (**IET DAVV**), Indore (M.P.)
- **Department:** Computer Engineering / Information Technology
- **Course:** Bachelor of Technology (B.Tech) - Major Project / Capstone
- **Objective:** Classify whether a patient has Parkinson's disease (`status: 1`) or is healthy (`status: 0`) non-invasively using 22 acoustic biomedical voice measurements.

> ⚠️ **MEDICAL & CLINICAL DISCLAIMER:**  
> This software is an educational engineering project designed for academic demonstration and research purposes only. It is **NOT** a certified medical diagnostic instrument. It should never be used as a substitute for professional clinical diagnosis by a certified neurologist.

---

## 🔬 Clinical Problem & Motivation
Parkinson's Disease (PD) is a neurodegenerative disorder affecting dopaminergic neurons in the *substantia nigra*. Up to **90% of Parkinson's patients experience early vocal impairment (hypokinetic dysphonia)**—such as subtle frequency tremors, reduced amplitude, breathiness, and pitch instability—often occurring **years before severe motor tremors** appear.

Traditional clinical diagnostics require in-person neurological scoring (UPDRS) and expensive imaging (DaTscan SPECT). This project delivers an automated, non-invasive, tele-screening ML pipeline that evaluates sustained phonations of the vowel `/a/` in seconds.

---

## 📊 Dataset Summary
- **Source:** [UCI Machine Learning Repository - Parkinson's Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) (curated by Max Little, University of Oxford).
- **Samples:** 195 voice recordings from 31 subjects (23 Parkinson's patients, 8 healthy controls).
- **Features:** 22 continuous acoustic measurements spanning:
  1. **Pitch (Vocal Fundamental Frequency):** `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`.
  2. **Jitter (Frequency Perturbation):** `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`.
  3. **Shimmer (Amplitude Perturbation):** `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`.
  4. **Noise Ratios:** `NHR` (Noise-to-Harmonics), `HNR` (Harmonics-to-Noise).
  5. **Nonlinear Dynamic Complexity:** `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE` (Pitch Period Entropy).
- **Missing Values:** Exactly 0 missing values.
- **Class Imbalance:** 147 PD samples (75.4%) vs 48 Healthy samples (24.6%).

---

## 📁 Project Directory Structure
```
parkinsons-detection/
├── data/
│   └── parkinsons.data               # Raw UCI biomedical dataset (195 rows, 24 cols)
├── notebooks/
│   └── parkinsons_detection_walkthrough.ipynb  # End-to-end executable Jupyter walkthrough
├── src/
│   ├── preprocess.py                 # Data loading, cleaning, stratified split & pipeline
│   ├── train.py                      # 5-fold Stratified CV & hyperparameter optimization
│   ├── evaluate.py                   # Test metrics computation & publication-grade plots
│   └── predict.py                    # Real-time inference engine & sample presets
├── models/
│   └── best_model.joblib             # Serialized best trained pipeline artifact
├── reports/
│   ├── model_comparison.csv          # Quantitative metrics across 7 models
│   └── figures/                      # High-resolution (300 DPI) visual plots
│       ├── 01_class_distribution.png
│       ├── 02_correlation_heatmap.png
│       ├── 03_key_features_boxplots.png
│       ├── 04_roc_curves_comparison.png
│       ├── 05_best_model_confusion_matrix.png
│       ├── 06_feature_importance.png
│       └── 07_model_metrics_comparison.png
├── docs/
│   ├── PROJECT_REPORT.md             # Complete academic markdown report
│   └── Parkinsons_Disease_Detection_Project_Report.docx # Formatted Word report
├── app.py                            # Interactive Streamlit Web Application
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation & run instructions
```

---

## 🏆 Model Performance Benchmark
All seven algorithms were evaluated on the held-out 20% test partition (39 samples: 29 Parkinson's, 10 Healthy) using **5-Fold Stratified Cross-Validation**:

| Algorithm | Test Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | Test ROC-AUC | 5-Fold CV ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **94.87%** | **100.00%** | **93.10%** | **100.00%** | **96.43%** | **0.9828** | 0.9405 |
| **XGBoost** | **92.31%** | **93.33%** | **96.55%** | **80.00%** | **94.92%** | **0.9793** | **0.9595** |
| **K-Nearest Neighbors** | 87.18% | 92.86% | 89.66% | 80.00% | 91.23% | 0.9793 | 0.9580 |
| **Gradient Boosting** | 92.31% | 96.43% | 93.10% | 90.00% | 94.74% | 0.9759 | 0.9514 |
| **SVM (RBF Kernel)** | 89.74% | 96.30% | 89.66% | 90.00% | 92.86% | 0.9655 | 0.9499 |
| **Logistic Regression**| 82.05% | 92.31% | 82.76% | 80.00% | 87.27% | 0.9276 | 0.9024 |
| **Decision Tree** | 76.92% | 85.71% | 82.76% | 60.00% | 84.21% | 0.7448 | 0.8226 |

### 🔑 Critical Insights:
1. **Clinical Sensitivity (Recall):** Missing a Parkinson's patient (False Negative) delays medical intervention. **XGBoost achieved 96.55% Recall** (detecting 28 out of 29 Parkinson's cases).
2. **Top Discriminative Biomarkers:** Feature importance identified **`PPE` (Pitch Period Entropy)**, **`spread1`**, **`MDVP:Fo(Hz)`**, and **`HNR`** as the most decisive acoustic markers.

---

## 🚀 Quickstart & Setup Guide

### 1. Clone or Open the Repository
```bash
cd "c:/Users/HP/Desktop/ML PROJECT/parkinsons-detection"
```

### 2. Install Required Dependencies
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Training & Evaluation
To retrain all 7 models, run 5-fold CV, generate comparison tables, and save all 7 PNG figures:
```bash
python src/evaluate.py
```

### 4. Run CLI Sample Inference
Test the trained pipeline on sample healthy and Parkinson's records:
```bash
python src/predict.py
```

### 5. Launch the Streamlit Web Application
Launch the interactive web UI in your browser:
```bash
streamlit run app.py
```
*The app features patient preset buttons ("Load Healthy", "Load PD", "Reset Median"), intuitive feature sliders grouped by category, probabilistic risk meter, and benchmark charts.*

### 6. Run the Walkthrough Jupyter Notebook
Open the notebook in Jupyter Lab or VS Code:
```bash
jupyter notebook notebooks/parkinsons_detection_walkthrough.ipynb
```

---

## 📑 Project Reports & Documentation
Submission-ready documentation is located in the `docs/` folder:
- **Word Document:** [`docs/Parkinsons_Disease_Detection_Project_Report.docx`](docs/Parkinsons_Disease_Detection_Project_Report.docx)  
  *Fully formatted college project report complete with IET DAVV title page, executive abstract, methodology, embedded full-color figures, benchmark tables, limitations, and viva questions.*
- **Markdown Report:** [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)  
  *Complete GitHub-readable academic report.*

---

## 🎓 Viva Voce Preparation
The documentation includes a curated **Viva Q&A section** with 10 high-frequency viva questions covering:
- Why voice analysis is effective for Parkinson's screening.
- The physical difference between Jitter and Shimmer.
- Data leakage prevention using Scikit-Learn Pipelines.
- Why Recall is prioritized over Accuracy in healthcare.
- Mathematical meaning of Pitch Period Entropy (PPE).
- Limitations of record-level splitting vs subject-level `GroupKFold`.
- Real-world cloud deployment architecture.

---

## 👨‍💻 Project Metadata
- **Institute:** Institute of Engineering and Technology, Devi Ahilya Vishwavidyalaya (**IET DAVV**), Indore
- **Branch:** Computer Engineering / Information Technology
- **Session:** 2025–2026
