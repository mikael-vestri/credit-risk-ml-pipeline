# Credit Risk ML Pipeline

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting loan default probability using Lending Club loan data. The goal is to build a production-ready credit risk assessment system that can support lending decisions in a real-world financial environment.

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

**Current Phase**: Step 10 - Model Interpretability ✅

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

**Next Steps**: 
- Step 11: Production Deployment

---

## Technical Stack

- **Python**: 3.10+
- **Core ML**: scikit-learn
- **Advanced Models**: XGBoost
- **Testing**: pytest
- **Interpretability**: SHAP (optional but desired)
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

## Notes

This project follows a step-by-step development approach with explicit validation checkpoints after each major step. All code, comments, and documentation are written in English following industry-standard ML engineering practices.


