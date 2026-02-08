# Credit Risk ML Pipeline

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting loan default probability using Lending Club loan data. The goal is to build a production-ready credit risk assessment system that can support lending decisions in a real-world financial environment.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Scope](#project-scope)
  - [Business Context](#business-context)
  - [Machine Learning Goals](#machine-learning-goals)
  - [Success Metrics](#success-metrics)
  - [Constraints and Assumptions](#constraints-and-assumptions)
- [Project Status](#project-status)
- [Results Summary](#results-summary)
- [Architecture Overview](#architecture-overview)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Documentation](#documentation)

---

## Project Scope

### Business Context

**Why This Problem Matters**

Credit risk assessment is fundamental to the lending industry. Accurate prediction of loan defaults enables:
- **Risk mitigation**: Lenders can avoid high-risk loans or adjust interest rates accordingly
- **Profitability optimization**: Better risk assessment leads to improved portfolio performance
- **Regulatory compliance**: Financial institutions must demonstrate sound risk management practices
- **Customer protection**: Responsible lending protects both lenders and borrowers

**Who Would Use This Model**

- **Lending institutions**: Banks, credit unions, and fintech companies making loan approval decisions
- **Risk management teams**: Analysts and managers monitoring portfolio risk
- **Underwriting departments**: Loan officers needing quick, data-driven risk assessments
- **Portfolio managers**: Professionals managing loan portfolios and setting pricing strategies

**What Decisions This Model Supports**

1. **Loan approval/rejection**: Binary decision on whether to approve a loan application
2. **Interest rate pricing**: Risk-based pricing where higher-risk loans receive higher rates
3. **Credit limit assignment**: Determining appropriate loan amounts based on risk profiles
4. **Portfolio risk monitoring**: Identifying high-risk segments for proactive management
5. **Regulatory reporting**: Providing quantitative risk metrics for compliance purposes

### Machine Learning Goals

**Primary Objective**
Build a supervised binary classification model that accurately predicts the probability of loan default (charged-off status) for approved loan applications.

**Secondary Objectives**
- Develop a reproducible, maintainable ML pipeline following industry best practices
- Create interpretable models that provide actionable insights (via SHAP analysis)
- Implement robust validation strategies that reflect real-world deployment scenarios
- Document all design decisions and trade-offs for portfolio demonstration

### Success Metrics

**Primary Decision Metric** (To be confirmed with stakeholder)
The primary metric will be determined based on business priorities. Candidates include:

- **ROC-AUC**: Measures ranking quality and ability to distinguish between defaulters and non-defaulters across all thresholds
- **Recall (Sensitivity)**: Critical for risk avoidance - minimizing false negatives (missed defaults)
- **Precision**: Important for customer impact - minimizing false positives (rejecting good loans)
- **Threshold-based cost metric**: Business-aligned metric incorporating financial costs of false positives/negatives

**Secondary Diagnostic Metrics**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis
- Feature importance rankings
- Model calibration (predicted probabilities vs. actual default rates)
- Performance across different loan segments (e.g., by grade, purpose, term)

### Constraints and Assumptions

**Data Constraints**
- **Dataset**: Lending Club Loan Data (Kaggle)
- **Data nature**: Real-world, noisy, partially inconsistent (intentionally embraced to reflect production conditions)
- **Temporal split**: Train on older vintages, test on recent ones (time-aware validation)
- **Rejected loans**: Included for feature engineering purposes only, not as training examples (to avoid selection bias)

**Technical Constraints**
- **Python environment**: `venv` virtual environment
- **Core ML library**: `scikit-learn` (mandatory)
- **Advanced models**: `XGBoost` (allowed and encouraged)
- **Validation strategy**: Advanced time-aware/out-of-time split
- **Testing**: Well-structured unit tests (pytest-style)
- **Code organization**: Production-ready Python modules (notebooks only for initial EDA)

**Modeling Assumptions**
1. Historical patterns in older vintages are predictive of future loan performance
2. The relationship between features and default probability is relatively stable over time (with monitoring for distribution shifts)
3. Approved loans in the dataset represent the population of interest for prediction
4. Rejected loans can provide valuable contextual features (e.g., rejection rates by segment) without introducing selection bias

**Business Assumptions**
1. The cost of a false negative (approving a loan that defaults) is significantly higher than a false positive (rejecting a good loan)
2. Model interpretability is valuable for regulatory compliance and stakeholder trust
3. The model will be used in a semi-automated decision support system (not fully automated approval/rejection)

### Project Deliverables

1. **Reproducible ML Pipeline**
   - Data ingestion and validation
   - Data cleaning and transformation
   - Feature engineering
   - Model training and evaluation
   - Model serialization and inference interface

2. **Model Interpretability**
   - SHAP analysis for feature importance and individual predictions
   - Model explainability documentation

3. **Production Readiness**
   - Training scripts
   - Inference interface
   - Unit tests
   - Documentation

4. **Portfolio Documentation**
   - Comprehensive README
   - Architecture overview
   - Results summary and analysis
   - Limitations and future improvements

---

## Project Status

**Current Phase**: Step 14 - Dockerization ✅

**Completed Steps**: 
- Step 0: Project Definition & Alignment ✅
- Step 1: Repository & Project Structure ✅
- Step 2: Project Scoping (Design First, Code Later) ✅
- Step 3: Dataset Selection & Validation ✅
- Step 4: Data Ingestion (ETL - Raw Layer) ✅
- Step 5: Data Cleaning & Transformation ✅
- Step 6: Feature Engineering ✅
- Step 7: Model Development ✅
- Step 8: Hyperparameter Tuning ✅
- Step 9: Model Evaluation ✅
- Step 10: Model Interpretability (SHAP analysis) ✅
- Step 11: Production Deployment (FastAPI REST API) ✅
- Step 12: Communication & Portfolio Polish ✅
- Step 13: Production Hardening (CI/CD, Monitoring, Versioning, Retraining) ✅
- Step 14: Dockerization (separate serving and training images, no all-in-one) ✅

---

## Results Summary

### Model Performance

We evaluated three models on a held-out test set of **1,206 samples** using a temporal split strategy (training on older vintages, testing on recent ones) to simulate real-world deployment scenarios.

#### Best Model: **Random Forest**

| Metric | Random Forest | Logistic Regression | XGBoost |
|--------|--------------|---------------------|---------|
| **ROC-AUC** | **0.9770** | 0.9722 | 0.9757 |
| **Average Precision** | **0.9280** | 0.9117 | 0.9242 |
| **Precision** | **0.8185** | 0.8000 | 0.8227 |
| **Recall** | **0.9157** | 0.9195 | 0.8889 |
| **F1-Score** | **0.8644** | 0.8556 | 0.8545 |

#### Confusion Matrix (Random Forest - Best Model)

| | Predicted: No Default | Predicted: Default |
|--|----------------------|---------------------|
| **Actual: No Default** | 892 (TN) | 53 (FP) |
| **Actual: Default** | 22 (FN) | 239 (TP) |

**Key Insights:**
- **High Recall (91.6%)**: The model successfully identifies 91.6% of actual defaults, minimizing false negatives (missed defaults)
- **Strong Precision (81.9%)**: When the model predicts default, it's correct 81.9% of the time
- **Low False Positive Rate (5.6%)**: Only 5.6% of good loans are incorrectly flagged as defaults
- **ROC-AUC of 0.977**: Excellent ranking ability - the model can distinguish between defaulters and non-defaulters with high confidence

#### Model Comparison

All three models performed exceptionally well, with Random Forest achieving the best overall performance:
- **Random Forest**: Best balance of precision and recall, highest ROC-AUC
- **Logistic Regression**: Strong interpretability, slightly lower performance but excellent for baseline
- **XGBoost**: Competitive performance, good for production with fast inference

### Feature Importance (SHAP Analysis)

The most important features for predicting loan default (based on SHAP analysis):
1. **FICO Score** (creditworthiness indicator)
2. **Debt-to-Income Ratio (DTI)** (borrower's financial burden)
3. **Revolving Utilization** (credit card usage)
4. **Loan Amount** (loan size)
5. **Interest Rate** (risk-based pricing indicator)

### Business Impact

- **Risk Mitigation**: High recall ensures most defaults are caught before approval
- **Profitability**: Strong precision minimizes rejection of good loans
- **Regulatory Compliance**: Interpretable models with SHAP analysis support explainable AI requirements
- **Scalability**: FastAPI deployment enables real-time predictions for loan applications

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CREDIT RISK ML PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Raw Data        │  Lending Club Loan Data (Kaggle)
│  (CSV/Parquet)   │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  • Data Ingestion (src/data/ingestion.py)                        │
│  • Data Cleaning (src/data/cleaning.py)                         │
│    - Missing value imputation                                    │
│    - Outlier detection & handling                                │
│    - Type consistency                                            │
│  • Feature Engineering (src/features/pipeline.py)                │
│    - Ratio features (DTI, utilization)                           │
│    - FICO midpoint calculation                                   │
│    - Credit history features                                    │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  • Model Training (src/models/trainers.py)                       │
│    - Logistic Regression (baseline)                              │
│    - Random Forest (ensemble)                                   │
│    - XGBoost (gradient boosting)                                │
│  • Hyperparameter Tuning (src/models/tuning.py)                 │
│    - RandomizedSearchCV with cross-validation                    │
│  • Model Evaluation (src/models/evaluation.py)                    │
│    - ROC-AUC, Precision, Recall, F1-Score                       │
│    - Confusion matrices, ROC/PR curves                          │
│  • Model Interpretability (src/models/interpretability.py)      │
│    - SHAP analysis for feature importance                       │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION SERVING LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  • FastAPI Application (src/api/app.py)                         │
│    - /predict endpoint for real-time inference                  │
│    - Input validation (Pydantic models)                         │
│    - Automatic feature engineering                              │
│    - Health check & model info endpoints                        │
│    - Monitoring endpoints (/metrics, /stats)                   │
│    - Data drift detection                                       │
│  • Model Serving (src/api/serving.py)                           │
│    - Model loading & caching                                    │
│    - Feature preparation                                        │
│    - Prediction logic                                           │
│  • Monitoring (src/api/monitoring.py)                           │
│    - Prometheus metrics                                         │
│    - Request tracking                                           │
│    - Data drift detection                                       │
│  • Artifact Registry (src/registry/)                           │
│    - Model versioning                                           │
│    - Training run tracking                                      │
│    - Model promotion workflow                                   │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Client          │  HTTP REST API
│  Applications    │  JSON Request/Response
└──────────────────┘
```

### Pipeline Flow

1. **Data Ingestion**: Load raw Lending Club data from CSV/Parquet files
2. **Data Cleaning**: Handle missing values, outliers, and type inconsistencies
3. **Feature Engineering**: Create derived features (ratios, aggregations)
4. **Temporal Split**: Split data by time (train on older, test on recent)
5. **Model Training**: Train multiple models with class imbalance handling
6. **Hyperparameter Tuning**: Optimize model parameters via cross-validation
7. **Model Evaluation**: Comprehensive evaluation on held-out test set
8. **Model Interpretability**: SHAP analysis for explainability
9. **Model Serialization**: Save best models for production
10. **API Deployment**: FastAPI server for real-time predictions
11. **Model Registry**: Register models with versioning and metadata
12. **Monitoring**: Track requests, metrics, and data drift
13. **Retraining**: Automated pipeline for model updates

### Key Design Decisions

- **Temporal Validation**: Time-aware splits prevent data leakage and simulate real-world deployment
- **Class Imbalance Handling**: Balanced class weights and scale_pos_weight for XGBoost
- **Pipeline Architecture**: Scikit-learn pipelines ensure consistent preprocessing in production
- **Interpretability**: SHAP analysis provides model explainability for regulatory compliance
- **Production-Ready**: FastAPI with input validation and automatic feature engineering
- **CI/CD**: Automated testing and linting via GitHub Actions
- **Monitoring**: Prometheus metrics, request tracking, and data drift detection
- **Model Versioning**: Artifact registry for tracking and promoting models
- **Automated Retraining**: Scripts for retraining and model promotion workflows

---

## Technical Stack

- **Python**: 3.10+
- **Core ML**: scikit-learn
- **Advanced Models**: XGBoost
- **API Framework**: FastAPI
- **Testing**: pytest, pytest-cov
- **Code Quality**: ruff, black, pre-commit
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus (prometheus-client)
- **Interpretability**: SHAP
- **Environment**: venv

---

## Project Structure

```
credit-risk-ml-pipeline/
├── data/
│   ├── raw/              # Raw, unprocessed data files
│   ├── processed/        # Cleaned and transformed data
│   └── external/         # External data sources
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory Data Analysis (to be created)
├── src/
│   ├── api/              # API for model serving (FastAPI)
│   ├── config/           # Configuration files and constants
│   ├── data/             # Data ingestion and processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   ├── pipelines/        # ML pipeline construction
│   ├── evaluation/       # Model evaluation and validation
│   └── utils/            # Utility functions and helpers
├── tests/                # Unit tests (pytest)
├── scripts/              # Executable scripts
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── .gitignore           # Git ignore rules
```

---

---

## Quick Start

### Running the API

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API server:**
   ```bash
   python scripts/run_api.py
   ```

3. **Access the API:**
   - API Base URL: http://localhost:8000
   - Interactive Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

4. **Make a prediction:**
   ```python
   import requests
   
   response = requests.post(
       "http://localhost:8000/predict",
       json={
           "loan_amnt": 10000,
           "funded_amnt": 10000,
           "funded_amnt_inv": 10000,
           "int_rate": 10.5,
           "installment": 300.0,
           "annual_inc": 50000,
           "dti": 15.5,
           "delinq_2yrs": 0,
           "fico_range_low": 700,
           "fico_range_high": 704,
           "inq_last_6mths": 1,
           "open_acc": 10,
           "pub_rec": 0,
           "revol_bal": 5000,
           "revol_util": 30.0,
           "total_acc": 20
       }
   )
   print(response.json())
   ```

For detailed API documentation, see [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md).

### Running with Docker (Step 14)

- **Serving**: Build and run the API in a container; mount your `models/` and set `MODEL_PATH`. See [docs/DOCKER.md](docs/DOCKER.md).
- **Training**: Build and run the retrain pipeline as a one-off job; mount `data/`, `models/`, and `artifacts/`.
- **Compose**: `docker compose up api` to start the API; `docker compose run train` to run training once.

---

## Documentation

### Production Documentation
- **[Production Checklist](docs/PRODUCTION_CHECKLIST.md)**: Pre-deployment checklist, deployment steps, and post-deployment validation
- **[Operations Runbook](docs/OPERATIONS_RUNBOOK.md)**: Troubleshooting guide, monitoring procedures, and emergency protocols
- **[Docker (Step 14)](docs/DOCKER.md)**: Build and run serving/training images; contracts and optional Compose

### API Documentation
- **[API Documentation](docs/API_DOCUMENTATION.md)**: Complete API reference with endpoints, request/response formats, and examples

### Git Workflow
- **[Git Workflow Guide](docs/GIT_WORKFLOW.md)**: Branching strategy, commit conventions, and useful Git commands
- **[GitHub Setup Guide](docs/GITHUB_SETUP.md)**: Instructions for setting up GitHub repository and SSH keys
- **[SSH Setup Guide](docs/SSH_SETUP.md)**: Detailed SSH key configuration for Windows

### Evaluation Results
- **Model Evaluation**: Results and visualizations saved in `docs/evaluation/`
  - `evaluation_summary.json`: Summary of all model metrics
  - `evaluation_results.json`: Detailed results per model
  - ROC curves, PR curves, and confusion matrices

### Model Interpretability
- **SHAP Analysis**: Feature importance and explainability results in `docs/interpretability/`
  - SHAP summary plots
  - Feature importance rankings
  - Individual prediction explanations

---

## Development Approach

This project follows a step-by-step development approach with explicit validation checkpoints after each major step. All code, comments, and documentation are written in English following industry-standard ML engineering practices.

### Key Principles
- **Reproducibility**: All data processing and model training steps are scripted and version-controlled
- **Production-Ready**: Code follows best practices for maintainability and deployment
- **Interpretability**: Models are explainable using SHAP analysis for regulatory compliance
- **Validation**: Temporal splits ensure realistic evaluation that simulates production deployment


