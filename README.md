# OncoDecision-AI

## Project Overview

Blood Cancer AI System is an end-to-end machine learning application designed to assist clinical decision-making for blood cancer cases in the Indian healthcare context. Given a patient's blood test results and symptoms, the system runs three sequential predictive models: it first determines whether cancer is present, then classifies the severity level if cancer is detected, and finally estimates the likely treatment cost based on the patient's profile and city.

The project covers the full machine learning lifecycle — from exploratory data analysis and feature engineering through model training, hyperparameter tuning, and deployment via an interactive Streamlit web application. A downloadable PDF report is generated for every patient consultation.

---

## Problem Statement

In India, early detection of blood cancer and accurate estimation of treatment costs remain significant challenges due to limited accessible diagnostic support tools and fragmented awareness across healthcare settings. Delayed diagnosis and unpredictable cost projections place a heavy burden on patients and clinicians alike.

This system addresses three specific clinical questions:

1. Does the patient's blood panel and symptom profile indicate blood cancer?
2. If cancer is present, what is the severity — Low, Medium, or High?
3. What is the estimated total treatment cost given the patient's severity level and city?

---

## Features

- Binary classification to detect cancer presence from haematological markers and symptoms
- Multi-class classification to categorise cancer severity as Low, Medium, or High
- Regression-based treatment cost estimation in Indian Rupees, adjusted by city
- Real-time laboratory gauge cards displaying WBC, RBC, Haemoglobin, and Platelet readings against clinical reference ranges
- Engineered clinical features computed on-the-fly including anemia flag, thrombocytopenia flag, WBC–platelet ratio, symptom count, and a composite clinical risk score
- Downloadable PDF medical report containing patient details, diagnosis, severity, cost estimate, and cost comparison chart
- Dark, Light, and System UI themes
- Stepwise workflow with progress indicator for guided clinical input
- City-aware cost estimation across 11 major Indian cities

---

## Technologies Used

| Category | Tools / Libraries |
|---|---|
| Language | Python 3 |
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn |
| Data Processing | pandas, numpy, pyarrow, fastparquet |
| Visualisation | matplotlib, seaborn |
| Web Application | Streamlit |
| Model Serialisation | joblib |
| PDF Generation | ReportLab |
| Notebook Environment | Jupyter |

---

## Project Architecture and Workflow

The project follows a structured pipeline from raw data to deployed application.

```
Raw Dataset (Parquet)
        |
        v
Exploratory Data Analysis (blood_cancer_eda.ipynb)
  - Data cleaning, duplicate removal, type optimisation
  - Outlier analysis, target variable analysis
  - Feature engineering (6 derived clinical features)
  - Dataset split into 3 task-specific subsets
        |
        v
Model Training Notebooks
  +----------------------------+----------------------------+---------------------------+
  | binary_classification      | multi_class_classification | regression                |
  | .ipynb                     | .ipynb                     | .ipynb                    |
  | Target: cancer_present     | Target: severity_level     | Target: treatment_cost    |
  | Model: Tuned XGBoost       | Model: Tuned XGBoost       | Model: Tuned Gradient     |
  |                            |                            | Boosting                  |
  +----------------------------+----------------------------+---------------------------+
        |
        v
Saved Model Pipelines (.pkl)
  - binary_classification_final_pipeline.pkl
  - multiclass_final_pipeline.pkl
  - regression_final_pipeline.pkl
        |
        v
Streamlit Application (app.py)
  - Patient input form
  - Live lab gauges
  - Sequential model inference
  - Cost estimation with city selection
  - PDF report generation and download
```

The three models are applied sequentially. The regression model is only invoked when the binary classifier confirms cancer presence, reflecting the intended clinical workflow.

---

## Dataset Information

| Property | Details |
|---|---|
| Format | Parquet |
| Original Size | 354,298 rows, 13 features |
| After Deduplication | 354,156 rows |
| Missing Values | None |
| Target Variables | `cancer_present` (binary), `severity_level` (multiclass), `treatment_cost` (continuous) |

**Input Features**

| Feature | Type | Description |
|---|---|---|
| age | Numerical | Patient age |
| gender | Categorical | Male / Female |
| wbc | Numerical | White blood cell count (cells/uL) |
| rbc | Numerical | Red blood cell count (M/uL) |
| hemoglobin | Numerical | Haemoglobin level (g/dL) |
| platelets | Numerical | Platelet count (cells/uL) |
| fever | Binary | Fever symptom present |
| fatigue | Binary | Fatigue symptom present |
| weight_loss | Binary | Weight loss symptom present |
| city | Categorical | One of 11 Indian cities |

**Engineered Features** (derived from the above)

| Feature | Description |
|---|---|
| symptom_count | Total number of symptoms present (0–3) |
| wbc_abnormal | Flag for WBC outside normal range (4,000–11,000 cells/uL) |
| wbc_platelet_ratio | Ratio of WBC to platelet count |
| anemia_flag | Low haemoglobin relative to gender-specific threshold |
| thrombocytopenia_flag | Platelets below 100,000 cells/uL |
| clinical_risk_score | Composite weighted score from WBC, haemoglobin, and platelets (0–1) |

**Dataset Split for Modelling**

- Binary classification: 354,156 rows (all patients), target = `cancer_present`
- Multi-class classification: 194,077 rows (cancer-positive only), target = `severity_level`
- Regression: 194,077 rows (cancer-positive only), target = `treatment_cost`

---

## Installation Instructions

**Prerequisites**

- Python 3.9 or higher
- pip

**Steps**

1. Clone the repository:

```bash
git clone https://github.com/VigneshKalla/Blood-Cancer-AI-System-Detection-Severity-Cost-Prediction-India-.git
cd blood-cancer-ai-system
```

2. Create and activate a virtual environment:

```bash
python -m venv bloodcancerenv
# Windows
bloodcancerenv\Scripts\activate
# macOS / Linux
source bloodcancerenv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
---

## How to Run the Project

**Running the Streamlit Application**

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

**Retraining the Models (Optional)**

If you want to retrain from scratch, run the Jupyter notebooks in the following order from the `notebooks/` directory:

```
1. blood_cancer_eda.ipynb          # EDA, feature engineering, dataset export
2. binary_classification.ipynb     # Cancer detection model
3. multi_class_classification.ipynb # Severity classification model
4. regression.ipynb                # Treatment cost regression model
```
---

## Project Structure

```
Blood Cancer AI System/
|
+-- app.py                                  # Streamlit application entry point
+-- requirements.txt                        # Python dependencies
+-- README.md                               # Project documentation
+-- .gitignore
|
+-- data/
|   +-- raw/
|   |   +-- blood_cancer_dataset.parquet    # Original dataset
|   +-- processed/
|       +-- blood_cancer_cleaned_dataset.parquet
|       +-- binary_classification_dataset.parquet
|       +-- multi_class_classification_dataset.parquet
|       +-- regression_dataset.parquet
|
+-- models/
|   +-- bc_best_model.pkl                           # Best binary model (pre-tuning)
|   +-- binary_classification_final_pipeline.pkl    # Final binary pipeline (deployed)
|   +-- mc_best_model.pkl                           # Best multiclass model (pre-tuning)
|   +-- multiclass_final_pipeline.pkl               # Final multiclass pipeline (deployed)
|   +-- reg_best_model.pkl                          # Best regression model (pre-tuning)
|   +-- regression_gb_pipeline.pkl                  # Final regression pipeline (deployed)
|
+-- notebooks/
|   +-- blood_cancer_eda.ipynb
|   +-- binary_classification.ipynb
|   +-- multi_class_classification.ipynb
|   +-- regression.ipynb
|
+-- plots/                                  # Saved visualisation outputs
```

---

## Usage Example

1. Launch the application with `streamlit run app.py`.
2. Enter the patient's name, age, and gender in the input form.
3. Input blood test values for WBC, RBC, Haemoglobin, and Platelets. The live lab gauge cards update in real time, highlighting any values outside the normal clinical range.
4. Select any symptoms present: Fever, Fatigue, Weight Loss.
5. Click **Run Diagnostic Prediction**. The binary model evaluates the inputs and returns a Cancer Detected or No Cancer result.
6. If cancer is detected, the multi-class model automatically classifies severity as Low, Medium, or High, and displays the clinical risk score, symptom count, and anemia flag.
7. Select the patient's city from the dropdown to account for regional cost variation.
8. Click **Estimate Treatment Cost**. The regression model returns a cost estimate in Indian Rupees with a contextual cost band interpretation.
9. Click **Download Full Medical Report (PDF)** to save a complete patient summary including all inputs, diagnosis, severity, cost estimate, and chart.

---

## Results and Output

### Binary Classification — Cancer Detection

| Metric | Value |
|---|---|
| Algorithm | Tuned XGBoost |
| Train Accuracy | ~99.21% |
| Test Accuracy | ~98.96% |
| F1 Score (weighted) | ~0.99 |
| ROC-AUC | 0.9997 |
| Train–Test Gap | ~0.25 (no overfitting) |

Top predictive features: `clinical_risk_score` (34.6%), `symptom_count`, `anemia_flag`, `wbc_abnormal`.

### Multi-Class Classification — Severity Level

| Metric | Value |
|---|---|
| Algorithm | Tuned XGBoost |
| Train Accuracy | 0.9979 |
| Test Accuracy | 0.9980 |
| F1 Score (macro) | 0.998 |
| ROC-AUC | 1.0 |
| Classes | Low / Medium / High |

Severity labels were re-derived from `clinical_risk_score` using 33rd and 67th percentile thresholds, producing a balanced three-class distribution. Primary driver: `clinical_risk_score`, followed by `wbc` and `hemoglobin`.

### Regression — Treatment Cost Estimation

| Metric | Value |
|---|---|
| Algorithm | Tuned Gradient Boosting |
| Test R² | ~0.84 |
| MAE | ~₹1,40,000 |
| MAPE | ~29% |
| Train–Test Gap | ~0 (no overfitting) |

Cost increases with severity: Low (~₹1.39L), Medium (~₹4.45L), High (~₹11.69L). City tier is a secondary contributor, with Mumbai and Delhi showing higher costs compared to Lucknow and Chandigarh.

### Application Output

- Green banner with routine follow-up recommendation when no cancer is detected
- Red banner with oncology referral guidance when malignancy indicators are present
- Severity pill (Low / Medium / High) with associated colour coding
- Cost estimate in INR with a cost band interpretation (Low / Mid / High / Critical)
- Downloadable PDF medical report

---

## Future Improvements

- Integrate real-world clinical datasets to replace synthetically structured data and improve generalisability
- Add SHAP-based explainability to provide per-prediction feature contribution breakdowns for clinicians
- Extend city coverage and incorporate hospital-level cost data for more granular treatment cost estimates
- Add support for additional blood cancer subtypes (ALL, CLL, AML, CML) with type-specific models
- Implement patient history tracking across consultations for longitudinal risk monitoring
- Deploy the application on a cloud platform with authentication for use in clinical settings
- Add model confidence intervals and prediction uncertainty quantification to the output
- Include a data upload interface to accept CBC report files directly rather than manual input

---

## Author and Credits

**Project:** OncoDecision AI: Blood Cancer Detection, Severity Analysis & Cost Prediction (India)

This project was developed as an end-to-end machine learning solution covering data analysis, feature engineering, model training, and production-ready deployment.

**Disclaimer:** This tool is intended for clinical decision support only. It does not replace professional medical diagnosis. All results must be confirmed by a qualified haematologist or oncologist before any clinical action is taken.
