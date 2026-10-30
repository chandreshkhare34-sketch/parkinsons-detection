# PARKINSON'S DISEASE DETECTION FROM VOICE MEASUREMENTS
## A Biomedical Machine Learning Classification System

---

### **A Major Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of Bachelor of Technology (B.Tech)**

**Department of Computer Engineering / Information Technology**  
**Institute of Engineering and Technology, Devi Ahilya Vishwavidyalaya (IET DAVV), Indore (M.P.) - 452017**  
**Academic Year:** 2025–2026  

---

| **Student Details** | **Faculty Supervisor** |
|:---|:---|
| **Name:** [Student Name Placeholder] | **Guide:** [Guide Name / Faculty Designation Placeholder] |
| **Roll No:** [Roll Number Placeholder] | **Department:** Computer Engineering / IT |
| **Enrollment No:** [Enrollment Number Placeholder] | **Institute:** IET DAVV, Indore |

---

## Abstract

Parkinson's Disease (PD) is a progressive neurodegenerative disorder affecting motor faculties, speech articulation, and neurological coordination. Approximately 90% of individuals with Parkinson's experience early acoustic abnormalities—clinically known as *hypokinetic dysarthria* or dysphonia—often manifesting years prior to severe motor symptoms such as resting tremors, postural instability, and bradykinesia. In clinical practice, definitive neurological diagnosis requires expert neurological consultation and expensive clinical imaging (e.g., DaTscan SPECT). 

This project designs, implements, and validates a machine learning system for non-invasive, tele-monitoring-friendly Parkinson's disease detection utilizing 22 acoustic biomedical voice measurements extracted from sustained vowel `/a/` phonations. Using the benchmark UCI Parkinson's dataset (195 voice recordings across 31 subjects), we implement a pipeline comprising data validation, leak-free standard scaling, class imbalance mitigation, and 5-fold Stratified Cross-Validation across seven machine learning algorithms: Logistic Regression, K-Nearest Neighbors, Support Vector Machine (RBF), Decision Tree, Random Forest, Gradient Boosting, and Extreme Gradient Boosting (XGBoost).

On the held-out 20% test partition, **Random Forest** achieved **94.87% accuracy**, **100.0% precision**, and an **ROC-AUC of 0.9828**, while **XGBoost** demonstrated the highest clinical sensitivity (**Recall of 96.55%**, detecting 28 out of 29 Parkinson's test subjects) with a cross-validated ROC-AUC of **0.9595**. Feature importance analysis identified Pitch Period Entropy (**PPE**), fundamental frequency variation (**spread1**), vocal pitch (**MDVP:Fo**), and Harmonics-to-Noise Ratio (**HNR**) as the most significant non-invasive acoustic biomarkers. A companion Streamlit web application was developed to offer real-time risk stratification and patient record autofill, accompanied by an explicit clinical disclaimer.

---

## 1. Introduction and Problem Statement

### 1.1 Clinical Background
Parkinson's Disease is the second most common neurodegenerative disorder globally after Alzheimer's disease, affecting over 10 million individuals worldwide. The underlying pathology involves the progressive degeneration of dopaminergic neurons within the *substantia nigra pars compacta* in the basal ganglia. As dopamine neurotransmitter levels decline, neuromuscular control deteriorates, causing classical motor symptoms including rigidity, postural instability, and resting tremor.

Significantly, vocal cord impairment (dysphonia) occurs in up to 90% of PD patients during early stages. Subtle voice tremors, reduced pitch modulation, breathiness, and increased acoustic noise arise due to incomplete vocal fold adduction and laryngeal muscle rigidity.

### 1.2 Problem Statement
Traditional Parkinson's diagnostic protocols depend heavily on specialized clinical evaluations such as the Unified Parkinson's Disease Rating Scale (UPDRS) and expensive neuroimaging. These procedures are costly, geographically restricted to urban tertiary hospitals, and subjective in early stages. 

There is an urgent healthcare need for **objective, automated, non-invasive, and cost-effective tele-screening tools** that can analyze a patient's acoustic voice recordings to detect subtle dysphonic biomarkers, enabling timely medical intervention and longitudinal monitoring.

---

## 2. Project Objectives

1. **Acoustic Biomarker Analysis:** Analyze 22 acoustic speech parameters spanning vocal pitch, frequency perturbation (jitter), amplitude perturbation (shimmer), vocal noise ratios (NHR, HNR), and nonlinear dynamical complexity (RPDE, DFA, spread1, spread2, D2, PPE).
2. **Leakage-Free Preprocessing:** Construct a scikit-learn `Pipeline` that isolates standard scaling strictly within training folds, avoiding data leakage.
3. **Class Imbalance Handling:** Address the ~75% vs ~25% target imbalance through stratified sampling and cost-sensitive class weighting.
4. **Multi-Model Benchmark:** Implement, tune (via 5-fold Stratified GridSearchCV), and empirically compare seven algorithms:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM RBF)
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - XGBoost Classifier
5. **Sensitivity-Oriented Evaluation:** Emphasize **Recall (Sensitivity)** and **ROC-AUC** for the positive class (Parkinson's), minimizing dangerous False Negatives.
6. **Interpretability & Feature Importance:** Quantify the predictive influence of non-linear measures versus classical jitter/shimmer features.
7. **Interactive Deployment:** Develop a browser-based Streamlit web dashboard with instant feedback, preset autofill, and mandatory medical disclaimers.

---

## 3. Literature Review

The application of digital speech signal processing and statistical learning for neurological disorder screening has evolved substantially over the past two decades:

1. **Little et al. (2009):** *"Suitability of dysphonia measurements for telemonitoring of Parkinson's disease," IEEE Transactions on Biomedical Engineering.*  
   Introduced the primary dataset utilized in this project. The authors demonstrated that classical acoustic measures (jitter, shimmer) alone are insufficient due to background acoustic noise, whereas nonlinear dynamical measurements (RPDE, DFA, Pitch Period Entropy) successfully achieved over 91% classification accuracy using Support Vector Machines.

2. **Tsanas et al. (2012):** *"Novel speech signal processing algorithms for high-accuracy classification of Parkinson's disease," IEEE TBME.*  
   Expanded the voice feature repertoire to include wavelet transforms and recurrence quantification analysis, showing that multi-feature acoustic vectors can track UPDRS severity non-invasively over telephone networks.

3. **Sakar et al. (2013):** *"Collection and analysis of a Parkinson speech dataset with multiple types of sound recordings," IEEE JBHI.*  
   Examined sustained vowels, numbers, words, and short sentences. Concluded that sustained phonations of vowel `/a/` provide the most stable fundamental frequency tracking, devoid of linguistic and phonetic confounding factors.

4. **Vaiciukynas et al. (2017):** *"Parkinson's disease detection from speech using convolutional neural networks and acoustic features," Computers in Biology and Medicine.*  
   Compared traditional handcrafted acoustic features against deep audio spectrogram representations, validating that tree ensembles on handcrafted acoustic features perform comparably to deep learning on small sample cohorts without risk of extreme overfitting.

---

## 4. Dataset Description

The dataset was obtained from the **UCI Machine Learning Repository** (Parkinson's Disease Dataset, contributed by Max Little, University of Oxford).

### 4.1 Dataset Summary
- **Total Records:** 195 voice recordings
- **Subject Count:** 31 individuals (23 diagnosed with Parkinson's disease, 8 healthy controls)
- **Recordings per Subject:** Approximately 6 sustained vowel phonations per subject
- **Target Distribution:** 147 recordings from Parkinson's subjects (75.38%), 48 recordings from healthy controls (24.62%)
- **Missing Values:** Exactly 0 missing or NaN values across all 24 columns

### 4.2 Comprehensive Feature Dictionary

| Feature Name | Category | Unit / Scale | Clinical & Physical Meaning |
|:---|:---|:---|:---|
| **`MDVP:Fo(Hz)`** | Fundamental Frequency | Hertz (Hz) | Average fundamental frequency (vocal pitch). Healthy baseline: ~150–220 Hz. Depressed in rigid vocal cords. |
| **`MDVP:Fhi(Hz)`** | Fundamental Frequency | Hertz (Hz) | Maximum fundamental vocal frequency during sustained phonation. |
| **`MDVP:Flo(Hz)`** | Fundamental Frequency | Hertz (Hz) | Minimum fundamental vocal frequency during sustained phonation. |
| **`MDVP:Jitter(%)`** | Frequency Perturbation | Percentage (%) | Cycle-to-cycle variation of fundamental frequency. Pathological vocal fold tremor increases jitter. |
| **`MDVP:Jitter(Abs)`**| Frequency Perturbation | Microseconds (μs)| Absolute cycle-to-cycle fundamental frequency variation. |
| **`MDVP:RAP`** | Frequency Perturbation | Ratio | Relative Amplitude Perturbation; evaluates frequency perturbation across 3 consecutive cycles. |
| **`MDVP:PPQ`** | Frequency Perturbation | Ratio | Five-point Period Perturbation Quotient. |
| **`Jitter:DDP`** | Frequency Perturbation | Ratio | Average absolute difference of differences between cycles ($3 \times \text{RAP}$). |
| **`MDVP:Shimmer`** | Amplitude Perturbation | Ratio | Cycle-to-cycle variation in speech waveform amplitude. |
| **`MDVP:Shimmer(dB)`**| Amplitude Perturbation | Decibels (dB) | Local shimmer expressed logarithmically in decibels. |
| **`Shimmer:APQ3`** | Amplitude Perturbation | Ratio | Three-point Amplitude Perturbation Quotient. |
| **`Shimmer:APQ5`** | Amplitude Perturbation | Ratio | Five-point Amplitude Perturbation Quotient. |
| **`MDVP:APQ`** | Amplitude Perturbation | Ratio | 11-point Amplitude Perturbation Quotient. |
| **`Shimmer:DDA`** | Amplitude Perturbation | Ratio | Average absolute differences between consecutive amplitudes ($3 \times \text{APQ3}$). |
| **`NHR`** | Vocal Noise | Ratio | Noise-to-Harmonics Ratio. Indicates breathiness and turbulence from incomplete vocal cord adduction. |
| **`HNR`** | Vocal Noise | Decibels (dB) | Harmonics-to-Noise Ratio. Healthy speech has higher acoustic resonance ($> 20 \text{ dB}$). |
| **`RPDE`** | Nonlinear Dynamics | $[0, 1]$ Entropy | Recurrence Period Density Entropy; measures dynamical complexity and periodicity of vocal attractor. |
| **`DFA`** | Nonlinear Dynamics | Exponent | Detrended Fluctuation Analysis; characterizes stochastic fractal self-similarity of speech signal. |
| **`spread1`** | Nonlinear Dynamics | Log-scaled | Nonlinear fundamental frequency variation. Strongly shifted toward zero in Parkinson's pathology. |
| **`spread2`** | Nonlinear Dynamics | Dimensionless | Secondary nonlinear frequency variation parameter. |
| **`D2`** | Nonlinear Dynamics | Dimension | Correlation dimension; quantifies chaotic geometric complexity in reconstructed state space. |
| **`PPE`** | Nonlinear Dynamics | Entropy | Pitch Period Entropy; measures impairment in pitch control. Most discriminative biomarker in literature. |
| **`status`** | Target Variable | Binary (0 or 1) | **0 = Healthy Control**, **1 = Parkinson's Disease Diagnosed**. |

---

## 5. Methodology and Pipeline Architecture

The overall architecture is designed to prevent data leakage and ensure reproducible results:

```
+-------------------------------------------------------------+
|               UCI Parkinson's Voice Dataset                 |
|             195 Recordings (147 PD, 48 Healthy)             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              Data Ingestion & Integrity Check               |
|      - Drop non-predictive identifier ('name')              |
|      - Verify 0 missing values, validate data types         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|         Stratified Train/Test Partitioning (80 / 20)        |
|    - Training Set: 156 samples (118 PD, 38 Healthy)         |
|    - Held-Out Test Set: 39 samples (29 PD, 10 Healthy)      |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|         Scikit-Learn Preprocessing & Tuning Pipeline        |
|  For each fold in 5-Fold Stratified Cross-Validation:       |
|    [Training Fold]                                          |
|      1. StandardScaler (fit on train fold only)             |
|      2. Classifier Tuning via GridSearchCV                  |
|    [Validation Fold]                                        |
|      3. Transform using fitted fold scaler                  |
|      4. Evaluate ROC-AUC and Recall                         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|         Model Evaluation on Unseen Test Set (20%)           |
|    - Logistic Regression, KNN, SVM, Decision Tree,          |
|      Random Forest, Gradient Boosting, XGBoost              |
|    - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC      |
|    - Confusion Matrix & Multi-Model ROC Overlay             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|             Persistence & Interactive Application           |
|    - Serialize Best Pipeline -> models/best_model.joblib    |
|    - Streamlit Web Dashboard (app.py) with Risk Badges      |
+-------------------------------------------------------------+
```

### 5.1 Data Leakage Prevention
A frequent flaw in published academic student projects is computing `StandardScaler.fit()` over the entire dataset prior to train/test splitting. This leaks the mean $\mu$ and standard deviation $\sigma$ of the test set into training. In this project:
- We strictly employ scikit-learn's `Pipeline([('scaler', StandardScaler()), ('classifier', estimator)])`.
- During cross-validation, the scaler is fitted solely on the $k-1$ training folds.
- The test set is transformed strictly using the training set statistics.

### 5.2 Multicollinearity Considerations
Correlation analysis revealed high multicollinearity ($r > 0.90$) within the Jitter family (`MDVP:Jitter(%)`, `RAP`, `PPQ`, `Jitter:DDP`) and Shimmer family (`MDVP:Shimmer`, `APQ3`, `APQ5`, `DDA`). 
- While multicollinearity can distort coefficients in unregularized linear models, non-parametric tree ensembles (Random Forest, Gradient Boosting, XGBoost) and non-linear RBF-kernel SVMs are inherently robust to collinear feature subsets.
- Non-linear dynamic biomarkers (`PPE`, `spread1`, `RPDE`) provide complementary, orthogonal diagnostic signal.

---

## 6. Implementation Details

- **Programming Language:** Python 3.13.14
- **Primary Packages:** pandas (3.0.6), numpy (2.5.4), scikit-learn (1.9.1), xgboost (3.4.1), matplotlib (3.11.2), seaborn (0.13.2), joblib (1.6.0), streamlit (1.66.1), python-docx (1.2.0).
- **Random Seed:** Set uniformly to `random_state=42`.

### 6.1 Project Directory Structure
```
parkinsons-detection/
|-- data/
|   `-- parkinsons.data               # Raw UCI biomedical dataset
|-- notebooks/
|   `-- parkinsons_detection_walkthrough.ipynb  # Executable Jupyter walkthrough
|-- src/
|   |-- preprocess.py                 # Ingestion, validation & pipeline builders
|   |-- train.py                      # 5-fold CV & hyperparameter optimization
|   |-- evaluate.py                   # Metric computation & publication plots
|   `-- predict.py                    # Inference engine & patient presets
|-- models/
|   `-- best_model.joblib             # Serialized best pipeline artifact
|-- reports/
|   |-- model_comparison.csv          # Quantitative metrics across 7 models
|   `-- figures/                      # Generated PNG figures
|       |-- 01_class_distribution.png
|       |-- 02_correlation_heatmap.png
|       |-- 03_key_features_boxplots.png
|       |-- 04_roc_curves_comparison.png
|       |-- 05_best_model_confusion_matrix.png
|       |-- 06_feature_importance.png
|       `-- 07_model_metrics_comparison.png
|-- app.py                            # Streamlit web deployment application
|-- requirements.txt                  # Complete library dependencies
|-- README.md                         # Submission & execution guide
`-- docs/
    |-- PROJECT_REPORT.md             # Complete academic markdown report
    `-- Parkinsons_Disease_Detection_Project_Report.docx # Word submission report
```

---

## 7. Results and Experimental Analysis

### 7.1 Quantitative Benchmark Comparison
All models were evaluated on the identical 20% stratified test set ($N=39$ samples: 29 Parkinson's subjects, 10 healthy controls).

| Model Algorithm | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | Test ROC-AUC | 5-Fold CV ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **94.87%** | **100.00%** | **93.10%** | **100.00%** | **96.43%** | **0.9828** | 0.9405 |
| **XGBoost** | **92.31%** | **93.33%** | **96.55%** | **80.00%** | **94.92%** | **0.9793** | **0.9595** |
| **K-Nearest Neighbors** | 87.18% | 92.86% | 89.66% | 80.00% | 91.23% | 0.9793 | 0.9580 |
| **Gradient Boosting** | 92.31% | 96.43% | 93.10% | 90.00% | 94.74% | 0.9759 | 0.9514 |
| **SVM (RBF Kernel)** | 89.74% | 96.30% | 89.66% | 90.00% | 92.86% | 0.9655 | 0.9499 |
| **Logistic Regression**| 82.05% | 92.31% | 82.76% | 80.00% | 87.27% | 0.9276 | 0.9024 |
| **Decision Tree** | 76.92% | 85.71% | 82.76% | 60.00% | 84.21% | 0.7448 | 0.8226 |

### 7.2 Analysis of Findings
1. **Clinical Sensitivity (Recall):**  
   In biomedical screening, **False Negatives are far more dangerous than False Positives**. Classifying a Parkinson's patient as "healthy" delays diagnosis and medical intervention. **XGBoost achieved the highest sensitivity of 96.55%**, successfully identifying 28 out of 29 Parkinson's cases in the test cohort with only 1 false negative.
2. **Overall Accuracy & Specificity:**  
   **Random Forest** achieved the highest overall accuracy (**94.87%**) and perfect precision (**100.0%**), registering zero false positives (all 10 healthy control subjects were accurately classified as healthy).
3. **Cross-Validation Consistency:**  
   XGBoost achieved the highest cross-validation score (**CV ROC-AUC = 0.9595**), demonstrating excellent generalization without overfitting.

### 7.3 Feature Importance Ranking
Permutation feature importance and Gini impurity metrics revealed the primary acoustic indicators:
1. **`PPE` (Pitch Period Entropy):** Highest importance score. Measures logarithmic cycle irregularity; highly elevated in Parkinson's dysphonia.
2. **`spread1`:** Nonlinear fundamental frequency variation. Demonstrates strong bimodal separation between healthy and diseased individuals.
3. **`MDVP:Fo(Hz)` (Average Fundamental Frequency):** Healthy subjects maintain higher average pitch (~180 Hz) compared to Parkinson's subjects (~145 Hz).
4. **`HNR` (Harmonics-to-Noise Ratio):** Significantly diminished in Parkinson's voices ($< 20 \text{ dB}$) due to turbulent airflow through incompletely closed vocal folds.

---

## 8. Discussion: Limitations and Ethical Considerations

### 8.1 Methodological Limitations
1. **Small Dataset Size:** The dataset comprises 195 voice recordings across 31 unique subjects. While standard in academic literature, larger clinical cohorts are necessary before clinical translation.
2. **Repeated Subject Phonations (Record-Level vs Subject-Level Split):**  
   Each subject provided approximately 6 voice recordings. A standard random or stratified train/test split distributes recordings from the same individual across both training and test sets. Because individual acoustic vocal tracts possess unique voice signatures, models can partially learn subject-specific vocal characteristics rather than disease characteristics alone. In literature, grouped splitting (`GroupKFold` or `GroupShuffleSplit`) yields lower but more realistic clinical accuracy (~82–86%). This distinction is an excellent discussion point during academic viva evaluations.
3. **Environmental and Microphone Variances:** The recordings were captured under controlled studio acoustic conditions using high-end condenser microphones. Real-world smartphone microphone recordings may introduce background ambient noise, reverberation, and compression artifacts.

### 8.2 Ethical and Clinical Considerations
- **Non-Diagnostic Nature:** Machine learning models are decision-support tools, not diagnostic authorities. Diagnostic responsibility remains strictly with certified neurologists.
- **Psychological Impact:** False positive screenings can cause severe emotional distress to patients. Clear risk communication and probabilistic outputs are imperative.

---

## 9. Conclusion and Future Scope

### 9.1 Conclusion
This project designed and evaluated a machine learning classification framework for Parkinson's Disease detection using biomedical acoustic measurements. By implementing a leakage-free Scikit-Learn pipeline with 5-fold Stratified Cross-Validation, seven distinct algorithms were tuned and benchmarked. Random Forest achieved **94.87% accuracy** with **0.9828 ROC-AUC**, and XGBoost demonstrated a superior **96.55% Recall**. The companion Streamlit dashboard makes these models accessible for interactive demonstration.

### 9.2 Future Scope
1. **Mobile Application Integration:** Developing an Android/iOS mobile application allowing patients to record sustained vowels via smartphone microphones for continuous remote tele-monitoring.
2. **Deep Learning on Raw Audio Spectrograms:** Applying Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) directly to log-mel spectrograms of continuous conversational speech.
3. **Multimodal Diagnostics:** Integrating voice features with keystroke dynamics, gait accelerometry, and spiral drawing digital tablet tests for holistic motor and cognitive assessment.

---

## 10. References

1. Little, M. A., McSharry, P. E., Hunter, E. J., Spielman, J., & Ramig, L. O. (2009). Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. *IEEE Transactions on Biomedical Engineering*, 56(4), 1015-1022.
2. Tsanas, A., Little, M. A., McSharry, P. E., & Ramig, L. O. (2012). Novel speech signal processing algorithms for high-accuracy classification of Parkinson's disease. *IEEE Transactions on Biomedical Engineering*, 59(5), 1264-1271.
3. Sakar, B. E., Isenkul, M. E., Sakar, C. O., Sertbas, A., Gurgen, F., Delil, S., ... & Kursun, O. (2013). Collection and analysis of a Parkinson speech dataset with multiple types of sound recordings. *IEEE Journal of Biomedical and Health Informatics*, 17(4), 828-834.
4. Vaiciukynas, E., Verikas, A., Gelzinis, A., & Bacauskiene, M. (2017). Parkinson's disease detection from speech using convolutional neural networks and acoustic features. *Computers in Biology and Medicine*, 89, 21-28.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

---

## 11. Viva Voce Examination: Questions and Model Answers

Here are 10 likely questions asked during a college project viva voce examination, accompanied by comprehensive, high-scoring technical answers:

#### Q1. Why did you choose voice measurements to detect Parkinson's disease instead of physical motor tremors?
**Answer:** Approximately 90% of Parkinson's patients exhibit voice impairment (hypokinetic dysarthria) during the early stages of the disease, often years before severe motor tremors or rigidity emerge. Voice recording is completely non-invasive, takes only a few seconds, requires no specialized hardware beyond a microphone, and is well-suited for telemedicine and remote longitudinal monitoring.

#### Q2. What is the fundamental difference between Jitter and Shimmer?
**Answer:** 
- **Jitter** measures cycle-to-cycle perturbations in **frequency** (pitch variations in time).
- **Shimmer** measures cycle-to-cycle perturbations in **amplitude** (sound wave energy/loudness).  
In Parkinson's patients, neurological instability in laryngeal muscle innervation causes erratic fluctuations in both frequency and amplitude, elevating both jitter and shimmer measures.

#### Q3. What is data leakage, and how did your implementation prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (such as test set statistics) is inadvertently shared with the model during training, producing artificially optimistic evaluation metrics. We prevented data leakage by wrapping feature scaling (`StandardScaler`) and estimators inside a Scikit-Learn `Pipeline`. Consequently, scaling parameters ($\mu$ and $\sigma$) were computed strictly from the training folds during 5-fold cross-validation and applied to validation/test sets without recalculation.

#### Q4. Why is Recall (Sensitivity) more critical than overall Accuracy in this project?
**Answer:** In medical diagnostic screening, the cost of a **False Negative** is severe: misclassifying a diseased individual as healthy denies them timely treatment and neuroprotective therapy. Conversely, a **False Positive** merely leads to confirmatory clinical consultation. Because our dataset has a ~75% class imbalance, an uncalibrated model could achieve 75% accuracy simply by predicting Parkinson's for all samples. Recall specifically evaluates what percentage of true Parkinson's patients the model successfully identified.

#### Q5. How does Pitch Period Entropy (PPE) work and why is it your most important feature?
**Answer:** Standard frequency measures struggle with background acoustic noise and natural healthy prosodic pitch variations. PPE (Pitch Period Entropy) was designed by Little et al. to measure the entropy of fundamental frequency variations on a logarithmic pitch scale, explicitly filtering out healthy intentional modulation. It quantifies the inability of the patient to maintain steady vocal cord tension, making it the most discriminative acoustic biomarker in our feature importance analysis.

#### Q6. What is the difference between Bagging (Random Forest) and Boosting (XGBoost)?
**Answer:** 
- **Random Forest (Bagging):** Builds multiple independent decision trees in parallel using bootstrap samples with random feature subsets, aggregating their votes to reduce variance (overfitting).
- **XGBoost (Boosting):** Builds decision trees sequentially, where each successive tree is trained to correct the residual gradient errors of preceding trees. XGBoost incorporates second-order Taylor expansion gradients, shrinkage (learning rate), and L1/L2 regularization to prevent overfitting.

#### Q7. Your dataset has 195 samples. Why didn't you use Deep Learning (e.g., LSTMs or CNNs)?
**Answer:** Deep neural networks typically require tens of thousands of samples to learn representations without overfitting. With only 195 tabular records, deep models would memorize the training data. Handcrafted acoustic domain features paired with regularized tree ensembles (Random Forest, XGBoost) and kernel SVMs are mathematically suited for tabular biomedical datasets of this scale.

#### Q8. What is the significance of the ROC-AUC metric?
**Answer:** ROC-AUC measures the Area Under the Receiver Operating Characteristic curve, which plots the True Positive Rate (Sensitivity) against the False Positive Rate ($1 - \text{Specificity}$) across all classification thresholds from 0 to 1. An AUC of 1.0 represents a perfect classifier, 0.5 represents random guessing. Unlike accuracy, ROC-AUC is threshold-independent and insensitive to class imbalance.

#### Q9. What is a critical limitation of the train/test split on this dataset?
**Answer:** The dataset contains 195 recordings from 31 distinct individuals (~6 recordings per subject). A standard stratified split distributes recordings from the same individual across both the training and test sets. Because individuals have unique vocal tract characteristics, the model could partially learn person-specific voice signatures rather than generalized disease features. In future work, a **GroupKFold** or subject-level split should be utilized to test generalization on completely unseen patients.

#### Q10. How would you deploy this project into a real-world clinical telemedicine workflow?
**Answer:** The model pipeline (`best_model.joblib`) can be containerized using Docker and deployed as a microservice API (e.g., FastAPI on Google Cloud Run or AWS). A mobile application can guide patients to pronounce sustained vowel `/a/` for 3 seconds into their smartphone microphone. A lightweight audio signal processing module (using Parselmouth/Praat in Python) extracts the 22 acoustic features and transmits the JSON payload to the API, returning an immediate risk score and longitudinal tracking dashboard for their attending neurologist.
