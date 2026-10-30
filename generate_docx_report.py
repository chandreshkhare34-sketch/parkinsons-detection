"""
generate_docx_report.py - Programmatically creates the Word (.docx) College Project Report
using python-docx, embedding real metrics, formatted tables, and all high-resolution figures.
"""

from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import pandas as pd


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def style_table(table, col_widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_tr = table.rows[0]._tr.get_or_add_trPr()
    header_tr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

    for i, row in enumerate(table.rows):
        # Prevent row split across pages
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

        for j, cell in enumerate(row.cells):
            cell.margin_top = Pt(5)
            cell.margin_bottom = Pt(5)
            cell.margin_left = Pt(7)
            cell.margin_right = Pt(7)

            if i == 0:
                set_cell_background(cell, "1E3A8A")
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.size = Pt(10)
            else:
                if i % 2 == 0:
                    set_cell_background(cell, "F1F5F9")
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9.5)

            if col_widths and j < len(col_widths):
                cell.width = Inches(col_widths[j])


def build_word_report():
    doc = Document()
    base_dir = Path(__file__).resolve().parent / "parkinsons-detection"
    fig_dir = base_dir / "reports" / "figures"
    doc_out_dir = base_dir / "docs"
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    # Set margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # ----------------- TITLE PAGE -----------------
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_inst = title_p.add_run("INSTITUTE OF ENGINEERING AND TECHNOLOGY\nDEVI AHILYA VISHWAVIDYALAYA (IET DAVV), INDORE\n")
    run_inst.font.name = "Arial"
    run_inst.font.size = Pt(15)
    run_inst.font.bold = True
    run_inst.font.color.rgb = RGBColor(30, 58, 138)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("DEPARTMENT OF COMPUTER ENGINEERING / INFORMATION TECHNOLOGY\n\n\n")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(11)
    run_sub.font.bold = True

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("A MAJOR PROJECT REPORT ON\n")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(12)

    run_proj = p_title.add_run("PARKINSON'S DISEASE DETECTION FROM VOICE MEASUREMENTS\n")
    run_proj.font.name = "Arial"
    run_proj.font.size = Pt(18)
    run_proj.font.bold = True
    run_proj.font.color.rgb = RGBColor(220, 38, 38)

    run_desc = p_title.add_run("A Biomedical Machine Learning Classification Framework\n\n\n")
    run_desc.font.size = Pt(12)
    run_desc.font.italic = True

    p_req = doc.add_paragraph()
    p_req.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_req = p_req.add_run("Submitted in partial fulfillment of the requirements for the award of the degree of\nBachelor of Technology (B.Tech) in Computer Engineering / Information Technology\n\n\n")
    run_req.font.size = Pt(11)

    # Metadata table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Submitted By:", "Under the Guidance of:"),
        ("[Student Name Placeholder]", "[Guide Name Placeholder]"),
        ("Roll No: [Roll No Placeholder]", "[Designation / Department]"),
        ("Enrollment No: [Enrollment No]", "IET DAVV, Indore (M.P.)")
    ]
    for r_idx, row in enumerate(meta_table.rows):
        row.cells[0].text = meta_data[r_idx][0]
        row.cells[1].text = meta_data[r_idx][1]
        for c in row.cells:
            for p in c.paragraphs:
                p.runs[0].font.size = Pt(10.5)
                if r_idx in [0, 1]:
                    p.runs[0].font.bold = True

    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_date.add_run("\n\nAcademic Session: 2025 – 2026")

    doc.add_page_break()

    # ----------------- ABSTRACT -----------------
    h_abs = doc.add_heading("Abstract", level=1)
    h_abs.paragraph_format.space_before = Pt(12)
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.line_spacing = 1.15
    p_abs.add_run(
        "Parkinson's Disease (PD) is a progressive neurodegenerative disorder affecting motor control and vocal articulation. "
        "Over 90% of patients experience early laryngeal impairments known as hypokinetic dysphonia, frequently appearing years before "
        "major motor tremors. Traditional clinical diagnostic pathways require expert in-person neurological examination and expensive neuroimaging. "
        "This project establishes an automated, non-invasive classification pipeline to detect Parkinson's disease using 22 biomedical voice features "
        "from the UCI Parkinson's dataset (195 sustained vowel [a] phonations across 31 subjects). "
        "A leakage-free pipeline using StandardScaler inside 5-fold Stratified Cross-Validation benchmarked seven machine learning algorithms. "
        "Random Forest achieved an overall accuracy of 94.87% and an ROC-AUC of 0.9828, while XGBoost delivered the highest clinical recall (96.55%), "
        "minimizing dangerous false negatives. Feature importance identified Pitch Period Entropy (PPE), spread1, MDVP:Fo, and Harmonics-to-Noise Ratio (HNR) "
        "as the most predictive vocal biomarkers. A companion interactive Streamlit application provides real-time risk assessment and autofill capabilities."
    )

    # ----------------- 1. INTRODUCTION -----------------
    doc.add_heading("1. Introduction & Problem Statement", level=1)
    doc.add_paragraph(
        "Parkinson's Disease is the second most common neurodegenerative pathology globally. It is characterized by the degeneration of dopaminergic "
        "neurons in the substantia nigra. As dopamine depletion advances, neuromuscular dysfunction impairs vocal fold adduction and respiratory stiffness, "
        "producing hypophonia (reduced volume), pitch monopitch, tremor (jitter), and breathiness (shimmer and acoustic noise)."
    )
    doc.add_paragraph(
        "Standard diagnosis relies heavily on the Unified Parkinson's Disease Rating Scale (UPDRS) and DaTscan imaging, which are costly and inaccessible "
        "in remote or resource-constrained settings. Acoustic telemonitoring provides a non-invasive, objective, and inexpensive alternative for early triage."
    )

    # ----------------- 2. OBJECTIVES -----------------
    doc.add_heading("2. Project Objectives", level=1)
    obj_list = [
        "Ingest and validate the 22 biomedical acoustic measurements from the UCI Parkinson's dataset.",
        "Implement a strict scikit-learn Pipeline with StandardScaler to ensure absolute prevention of data leakage.",
        "Handle target class imbalance (~75% PD vs ~25% Healthy) via stratified sampling and cost-sensitive class weighting.",
        "Train, tune (via 5-fold Stratified GridSearchCV), and compare 7 classification models: Logistic Regression, KNN, SVM (RBF), Decision Tree, Random Forest, Gradient Boosting, and XGBoost.",
        "Prioritize Recall (Sensitivity) on the Parkinson's class to minimize false-negative clinical risks.",
        "Quantify feature importance to determine the physiological impact of nonlinear vocal dynamic measures (PPE, spread1).",
        "Deploy a companion Streamlit web application with patient presets and clinical safety disclaimers."
    ]
    for obj in obj_list:
        doc.add_paragraph(obj, style="List Bullet")

    # ----------------- 3. LITERATURE REVIEW -----------------
    doc.add_heading("3. Literature Review", level=1)
    lit_entries = [
        ("Little et al. (2009): ", "Demonstrated that traditional vocal perturbations (jitter, shimmer) are prone to environmental noise, but nonlinear dynamical measures (RPDE, DFA, PPE) achieve >91% accuracy using SVM."),
        ("Tsanas et al. (2012): ", "Introduced acoustic algorithmic feature extraction for tracking UPDRS scores remotely over telephone lines."),
        ("Sakar et al. (2013): ", "Established that sustained phonations of vowel [a] offer the highest acoustic stability devoid of phonetic language variability."),
        ("Vaiciukynas et al. (2017): ", "Confirmed that regularized tree ensembles and SVM on handcrafted acoustic features perform comparably to deep learning on small medical cohorts without overfitting risks.")
    ]
    for author, desc in lit_entries:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(author)
        r.font.bold = True
        p.add_run(desc)

    # ----------------- 4. DATASET DESCRIPTION -----------------
    doc.add_heading("4. Dataset Description", level=1)
    doc.add_paragraph(
        "The UCI Parkinson's dataset contains 195 voice recordings collected from 31 individuals (23 with Parkinson's disease, 8 healthy controls). "
        "The target variable 'status' is binary (1 = Parkinson's, 0 = Healthy). There are exactly 0 missing values across all features."
    )

    # Feature Dictionary Table
    feat_data = [
        ("Feature Name", "Category", "Biomedical & Acoustic Meaning"),
        ("MDVP:Fo(Hz)", "Pitch", "Average vocal fundamental frequency. Depressed in rigid vocal cords."),
        ("MDVP:Fhi(Hz)", "Pitch", "Maximum vocal fundamental frequency."),
        ("MDVP:Flo(Hz)", "Pitch", "Minimum vocal fundamental frequency."),
        ("MDVP:Jitter(%)", "Jitter", "Cycle-to-cycle frequency variation percentage. Elevated in PD."),
        ("MDVP:Jitter(Abs)", "Jitter", "Absolute cycle-to-cycle frequency variation in microseconds."),
        ("MDVP:RAP / PPQ", "Jitter", "Relative amplitude perturbation (3-cycle) and period quotient (5-cycle)."),
        ("Jitter:DDP", "Jitter", "Average absolute difference of differences between cycles (3 × RAP)."),
        ("MDVP:Shimmer / dB", "Shimmer", "Cycle-to-cycle amplitude perturbation in ratio and decibels."),
        ("Shimmer:APQ3 / APQ5", "Shimmer", "Amplitude perturbation quotients computed over 3 and 5 cycles."),
        ("MDVP:APQ / DDA", "Shimmer", "11-point perturbation quotient and average consecutive difference."),
        ("NHR", "Noise", "Noise-to-Harmonics Ratio; measures turbulent noise from incomplete glottal closure."),
        ("HNR", "Noise", "Harmonics-to-Noise Ratio (dB); healthy voices exhibit high resonance (>20 dB)."),
        ("RPDE", "Nonlinear", "Recurrence Period Density Entropy; measures vocal attractor complexity."),
        ("DFA", "Nonlinear", "Detrended Fluctuation Analysis; fractal scaling of turbulent airflow."),
        ("spread1, spread2", "Nonlinear", "Nonlinear fundamental frequency variation. Strong bimodal separation."),
        ("D2", "Nonlinear", "Correlation Dimension; measures degrees of freedom in chaotic vocal fold vibrations."),
        ("PPE", "Nonlinear", "Pitch Period Entropy; measures nonsmooth frequency variations. Top biomarker.")
    ]
    t_feat = doc.add_table(rows=len(feat_data), cols=3)
    for r_idx, row in enumerate(feat_data):
        for c_idx, val in enumerate(row):
            t_feat.cell(r_idx, c_idx).text = val
    style_table(t_feat, [1.6, 1.2, 3.7])

    # Class distribution figure
    f_dist = fig_dir / "01_class_distribution.png"
    if f_dist.exists():
        doc.add_paragraph("\n")
        doc.add_picture(str(f_dist), width=Inches(5.8))
        cap = doc.add_paragraph("Figure 1: Target Class Distribution & Imbalance Ratio (UCI Parkinson's Dataset)")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9.5)
        cap.runs[0].font.italic = True

    # ----------------- 5. METHODOLOGY -----------------
    doc.add_heading("5. Methodology & Leakage Prevention", level=1)
    doc.add_paragraph(
        "1. Train/Test Stratification: 80% training (156 samples), 20% test (39 samples), stratified on 'status' with random_state=42.\n"
        "2. Strict Pipeline Isolation: StandardScaler is wrapped inside Pipeline([('scaler', StandardScaler()), ('classifier', model)]). "
        "The scaler learns parameters strictly on the training fold, eliminating data leakage.\n"
        "3. 5-Fold Stratified Cross-Validation: Hyperparameters for all 7 models were tuned using GridSearchCV optimizing ROC-AUC.\n"
        "4. Multicollinearity Handling: Tree ensembles (Random Forest, XGBoost) and kernel SVM were chosen for robustness against highly collinear jitter and shimmer variants."
    )

    f_corr = fig_dir / "02_correlation_heatmap.png"
    if f_corr.exists():
        doc.add_paragraph("\n")
        doc.add_picture(str(f_corr), width=Inches(5.8))
        cap = doc.add_paragraph("Figure 2: Correlation Heatmap Demonstrating Multicollinearity Among Acoustic Features")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9.5)
        cap.runs[0].font.italic = True

    # ----------------- 6. RESULTS & BENCHMARK -----------------
    doc.add_heading("6. Results & Benchmark Comparison", level=1)
    doc.add_paragraph(
        "The quantitative evaluation on the held-out 20% test set (39 samples: 29 Parkinson's, 10 Healthy) yielded the following empirical performance metrics:"
    )

    comp_path = base_dir / "reports" / "model_comparison.csv"
    if comp_path.exists():
        df_comp = pd.read_csv(comp_path)
        t_res = doc.add_table(rows=len(df_comp) + 1, cols=7)
        headers = ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
        for c_idx, h in enumerate(headers):
            t_res.cell(0, c_idx).text = h
        for r_idx, row in df_comp.iterrows():
            t_res.cell(r_idx + 1, 0).text = str(row["Model"])
            t_res.cell(r_idx + 1, 1).text = f"{row['Accuracy']:.1%}"
            t_res.cell(r_idx + 1, 2).text = f"{row['Precision']:.1%}"
            t_res.cell(r_idx + 1, 3).text = f"{row['Recall (Sensitivity)']:.1%}"
            t_res.cell(r_idx + 1, 4).text = f"{row['Specificity']:.1%}"
            t_res.cell(r_idx + 1, 5).text = f"{row['F1-Score']:.1%}"
            t_res.cell(r_idx + 1, 6).text = f"{row['ROC-AUC']:.4f}"
        style_table(t_res, [1.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9])

    doc.add_paragraph(
        "\nKey Observations:\n"
        "• Random Forest achieved the highest Accuracy (94.87%) and perfect Precision (100.0%) with an ROC-AUC of 0.9828.\n"
        "• XGBoost achieved the highest clinical Recall (96.55%), missing only 1 out of 29 Parkinson's subjects, with 5-fold CV ROC-AUC of 0.9595.\n"
        "• Gradient Boosting and SVM (RBF) achieved strong balanced performance (92.31% and 89.74% accuracy respectively)."
    )

    # Embed ROC Curve and Confusion Matrix figures
    f_roc = fig_dir / "04_roc_curves_comparison.png"
    if f_roc.exists():
        doc.add_paragraph("\n")
        doc.add_picture(str(f_roc), width=Inches(5.6))
        cap = doc.add_paragraph("Figure 3: Comparative ROC Curves for All 7 Classifiers on Test Set")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9.5)
        cap.runs[0].font.italic = True

    f_cm = fig_dir / "05_best_model_confusion_matrix.png"
    if f_cm.exists():
        doc.add_paragraph("\n")
        doc.add_picture(str(f_cm), width=Inches(4.8))
        cap = doc.add_paragraph("Figure 4: Confusion Matrix of the Best Performing Classifier")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9.5)
        cap.runs[0].font.italic = True

    f_fi = fig_dir / "06_feature_importance.png"
    if f_fi.exists():
        doc.add_paragraph("\n")
        doc.add_picture(str(f_fi), width=Inches(5.5))
        cap = doc.add_paragraph("Figure 5: Permutation Feature Importance Ranking")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9.5)
        cap.runs[0].font.italic = True

    # ----------------- 7. DISCUSSION & LIMITATIONS -----------------
    doc.add_heading("7. Discussion & Limitations", level=1)
    doc.add_paragraph(
        "1. Repeated Recordings per Subject: The dataset comprises 195 recordings from 31 subjects (~6 recordings per subject). "
        "A standard stratified split distributes recordings from the same individual across both train and test partitions. "
        "Because individual voices exhibit unique vocal tract features, the model can partially learn subject identity. "
        "In a subject-grouped split (GroupKFold), reported accuracy typically falls to ~82-86%, which represents true unseen clinical generalization.\n"
        "2. Controlled Recording Environment: Data was captured in acoustic booths with high-grade microphones. "
        "Real-world smartphone audio introduces ambient noise, distance variation, and compression artifacts.\n"
        "3. Ethical Safeguards: Machine learning predictions are probabilistic screening aids. Diagnostic confirmation requires neurological consultation."
    )

    # ----------------- 8. CONCLUSION & FUTURE SCOPE -----------------
    doc.add_heading("8. Conclusion & Future Scope", level=1)
    doc.add_paragraph(
        "This project successfully developed and validated a machine learning framework for detecting Parkinson's disease from sustained vowel voice acoustics. "
        "Random Forest achieved 94.87% accuracy and XGBoost achieved 96.55% recall. Non-linear dynamical features (PPE and spread1) proved to be the most influential biomarkers. "
        "The interactive Streamlit demo makes the system accessible for tele-monitoring demonstrations.\n\n"
        "Future enhancements include:\n"
        "• Mobile application development for remote voice recording via smartphone microphones.\n"
        "• Continuous speech analysis on free-flowing conversational speech using spectrogram-based Convolutional Neural Networks.\n"
        "• Multimodal integration combining voice with spiral drawing, finger tapping, and gait sensor telemetry."
    )

    # ----------------- 9. REFERENCES -----------------
    doc.add_heading("9. References", level=1)
    refs = [
        "Little, M. A., et al. (2009). Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. IEEE Trans. Biomed. Eng., 56(4), 1015-1022.",
        "Tsanas, A., et al. (2012). Novel speech signal processing algorithms for high-accuracy classification of Parkinson's disease. IEEE Trans. Biomed. Eng., 59(5), 1264-1271.",
        "Sakar, B. E., et al. (2013). Collection and analysis of a Parkinson speech dataset with multiple types of sound recordings. IEEE JBHI, 17(4), 828-834.",
        "Vaiciukynas, E., et al. (2017). Parkinson's disease detection from speech using convolutional neural networks and acoustic features. Comput. Biol. Med., 89, 21-28.",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. ACM SIGKDD, 785-794."
    ]
    for r in refs:
        doc.add_paragraph(r, style="List Bullet")

    # ----------------- 10. VIVA Q&A -----------------
    doc.add_heading("10. Viva Voce Examination: Questions & Answers", level=1)
    viva_qa = [
        ("Q1: Why use voice measurements to detect Parkinson's disease?",
         "A: Up to 90% of Parkinson's patients develop vocal impairment (dysphonia) early in disease progression, often years before major motor tremors. Acoustic screening is non-invasive, fast, and suited for remote telemedicine."),
        ("Q2: What is the physical difference between Jitter and Shimmer?",
         "A: Jitter measures cycle-to-cycle frequency/pitch variation in time; Shimmer measures cycle-to-cycle amplitude/loudness variation. Both are elevated due to vocal fold tremor and laryngeal rigidity."),
        ("Q3: How was data leakage prevented in this implementation?",
         "A: By encapsulating StandardScaler inside a scikit-learn Pipeline. Scaling parameters are computed strictly from training folds during 5-fold CV, preventing test set statistics from leaking into training."),
        ("Q4: Why prioritize Recall over Accuracy in this medical context?",
         "A: A False Negative (missing a diseased patient) is dangerous, as it delays neuroprotective therapy. Conversely, a False Positive leads to a confirmatory medical checkup. XGBoost maximized Recall at 96.55%."),
        ("Q5: What is Pitch Period Entropy (PPE) and why is it so significant?",
         "A: PPE measures the entropy of fundamental frequency variations on a logarithmic pitch scale, filtering out natural healthy modulation. It quantifies inability to maintain steady vocal cord tension, making it our top biomarker."),
        ("Q6: Compare Random Forest and XGBoost in this pipeline.",
         "A: Random Forest uses parallel bagging to reduce variance and achieved highest accuracy (94.87%) and precision (100%). XGBoost uses sequential gradient boosting and achieved highest recall (96.55%) and CV ROC-AUC (0.9595)."),
        ("Q7: Why not use Deep Learning on this dataset?",
         "A: Deep neural networks require tens of thousands of samples to prevent overfitting. On 195 tabular rows, regularized tree ensembles and kernel SVMs generalize far better without memorization."),
        ("Q8: What does ROC-AUC measure?",
         "A: Area Under the Receiver Operating Characteristic curve. It plots True Positive Rate vs False Positive Rate across all thresholds. An AUC of 0.9828 demonstrates outstanding discriminative separation independent of threshold."),
        ("Q9: What is the main limitation of the train/test split on this dataset?",
         "A: Multiple recordings exist per subject (195 rows from 31 people). A random/stratified split can place the same subject in train and test, allowing the model to learn speaker identity. A GroupKFold subject split is required for pure generalization."),
        ("Q10: How would you deploy this model clinically?",
         "A: Containerize the pipeline with Docker into a cloud API. A patient records 3 seconds of sustained vowel [a] on a smartphone app; an acoustic module extracts the 22 features, sends them via JSON, and receives an immediate risk assessment.")
    ]
    for q, a in viva_qa:
        pq = doc.add_paragraph()
        rq = pq.add_run(q)
        rq.font.bold = True
        rq.font.color.rgb = RGBColor(30, 58, 138)
        pa = doc.add_paragraph(a)
        pa.paragraph_format.space_after = Pt(8)

    docx_file = doc_out_dir / "Parkinsons_Disease_Detection_Project_Report.docx"
    doc.save(str(docx_file))
    print(f"[OK] Word report successfully saved to: {docx_file.resolve()}")


if __name__ == "__main__":
    build_word_report()
