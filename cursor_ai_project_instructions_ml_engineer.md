# Cursor AI – Step-by-Step Instructions for an End-to-End Machine Learning Engineering Project

## Purpose of This Document

This document is **not** the project itself. It is a **meta-instruction manual** designed to be used **inside Cursor** as guidance for an AI coding assistant.

The goal is for the AI in Cursor to:
- Implement the project **step by step**
- **Pause after every major step** to ask for confirmation
- **Ask clarifying questions whenever assumptions are required**
- Follow **real-world Machine Learning Engineering best practices**
- Produce a **portfolio-grade, production-ready project**

This project is intended to validate the author as a **Machine Learning Engineer** in professional recruiting processes.

---

## Global Rules for the Cursor AI (Must Follow)

1. **Never implement multiple major steps at once**
2. After each step, explicitly ask:
   > "Can I proceed to the next step, or would you like to adjust something first?"
3. If **any requirement is ambiguous**, stop and ask for clarification
4. Prefer **clean, modular, production-ready code** over quick experiments
5. Favor **Python scripts and modules** over notebooks unless explicitly justified
6. Every decision must be **documented** (README, docstrings, comments)
7. Follow **industry-standard project structure**
8. All code, comments, documentation must be written **in English**

---

## Recommended Development Environment (Real-World Oriented)

### Fixed Technical Decisions (User-Approved)

These choices are **locked** and must be respected by the Cursor AI:

- **Python environment**: `venv`
- **Core ML library**: `scikit-learn` (mandatory)
- **Advanced models**: `XGBoost` (allowed and encouraged)
- **Validation strategy**: advanced (time-aware / out-of-time split where applicable)
- **Testing**: well-structured unit tests (pytest-style)

### Primary Environment
- **Cursor IDE** (instead of VSCode)
- Python 3.10+
- Virtual environment via `venv`

### Notebooks – When to Use
Notebooks **may be used**, but only for:
- Initial Exploratory Data Analysis (EDA)
- Visual inspection and hypothesis generation

All **final logic** (ETL, features, training, evaluation, pipelines) must be migrated to **Python modules**.

---

## Step 0 – Project Definition & Alignment (DO NOT CODE YET)

### Fixed Project Definition (User-Approved)

- **Problem type**: Supervised classification
- **Domain**: Financial risk / credit underwriting
- **Task**: Predict loan default probability
- **Dataset**: Lending Club (Kaggle)
- **Nature of data**: Real-world, noisy, partially inconsistent

This project intentionally embraces messy data to reflect real production conditions.

### AI Must Still Ask the User:
1. Preferred **time window** (e.g., train on older vintages, test on recent ones)
2. Whether rejected loans should be excluded or treated explicitly
3. Any constraints on explainability (e.g., SHAP required or optional)

⚠️ Do not proceed until these clarifications are confirmed.

---

## Step 1 – Repository & Project Structure

### Objective
Create a professional repository structure aligned with real ML engineering teams.

### Expected Folder Structure
```
ml-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   ├── evaluation/
│   └── utils/
├── tests/
├── scripts/
├── README.md
├── requirements.txt
├── pyproject.toml (optional)
└── .gitignore
```

### AI Tasks
- Initialize Git repository
- Create folders
- Write an **initial README.md** with:
  - Project motivation
  - Problem statement (draft)
  - High-level pipeline overview

⏸️ Pause and ask for approval before proceeding.

---

## Step 2 – Project Scoping (Design First, Code Later)

### Objectives
- Define **clear business and ML goals**
- Define **success metrics** (not only accuracy)
- Define **constraints and assumptions**

### AI Must Produce
- A dedicated **Project Scope section** in README.md
- Explicit answers to:
  - Why this problem matters
  - Who would use this model
  - What decisions it supports

⏸️ Pause for validation.

---

## Step 3 – Dataset Selection & Validation

### Dataset (Locked)
- **Source**: Lending Club Loan Data (Kaggle)
- **Target**: Loan default / charged-off indicator

### AI Responsibilities
- Validate schema consistency across vintages
- Identify target leakage columns
- Explicitly document dropped fields and justification

⚠️ AI must **not download or preprocess data** until user confirms time-split strategy.

---

## Step 4 – Data Ingestion (ETL – Raw Layer)

### Implementation Rules
- Raw data is **never modified**
- All ingestion logic goes to `src/data/ingestion.py`

### AI Tasks
- Implement reproducible data loading
- Log dataset version, schema, and basic stats

⏸️ Pause before transformations.

---

## Step 5 – Data Cleaning & Transformation

### Requirements
- Handle missing values explicitly
- Detect outliers
- Ensure type consistency
- Log all transformations

### Output
- Clean dataset saved in `data/processed/`

---

## Step 6 – Feature Engineering

### Rules
- Train/validation/test split **before** feature engineering
- Avoid leakage at all costs

### AI Must Implement
- Feature builders in `src/features/`
- Config-driven feature selection

---

## Step 7 – Model Development

### Mandatory Modeling Strategy

1. **Baseline models**
   - Logistic Regression
   - Decision Tree / Random Forest

2. **Advanced model**
   - XGBoost classifier

### Evaluation Strategy
- Multiple metrics may be computed, but the AI must:
  - Ask the user to define **one primary decision metric**
  - Justify secondary metrics as supporting diagnostics

Examples:
- ROC-AUC → ranking quality
- Recall (default) → risk avoidance
- Precision → customer impact
- Threshold-based cost metric → business alignment

---

## Step 8 – Hyperparameter Tuning

### Constraints
- Use GridSearch or RandomizedSearch
- Document trade-offs

---

## Step 9 – Pipeline Construction

### Mandatory
- Use `sklearn.pipeline.Pipeline`
- Single callable training entrypoint

---

## Step 10 – Evaluation & Validation

### Required Outputs
- Final metrics
- Error analysis
- Bias & variance discussion

---

## Step 11 – Production Readiness

### Must Include
- Reproducible training script
- Model serialization
- Clear inference interface

---

## Step 12 – Communication & Portfolio Polish

### Deliverables
- Clear README
- Architecture diagram (optional)
- Results summary

---

## Step 13 – Conclusion & Next Steps

### AI Must Help Write
- Limitations
- Future improvements
- Scaling considerations

---

## Final Rule

**The AI must behave like a senior ML Engineer mentoring a junior engineer, not like an auto-coder.**

If anything is unclear, stop and ask.

