# Dataset Validation Plan

## Dataset Information

- **Source**: Lending Club Loan Data (Kaggle)
- **Target Variable**: Loan default / charged-off indicator
- **Data Nature**: Real-world, noisy, partially inconsistent (intentionally embraced)

**⚠️ IMPORTANT**: This validation plan contains **hypothetical assumptions** about the dataset structure based on common Lending Club datasets. **All assumptions must be validated** once we have access to the actual dataset. Column names, schemas, and data structures may differ from what is documented here.

## Validation Objectives

### 1. Schema Consistency Across Vintages

**What to Check:**
- Column names consistency across different time periods/vintages
- Data types consistency for each column
- Missing column handling (columns that appear/disappear over time)
- Column order variations (if any)

**Expected Issues:**
- Lending Club dataset may have evolved over time with new columns added
- Some columns may have been deprecated or renamed
- Data types might have changed (e.g., string to numeric)

**Action Plan:**
- Load all available vintage files
- Compare schemas (column names, types, counts)
- Document schema evolution timeline
- Create unified schema mapping if needed

### 2. Target Leakage Identification

**Definition**: Features that contain information about the target that would not be available at prediction time.

**⚠️ IMPORTANT NOTE**: The columns listed below are **hypothetical examples** based on common Lending Club dataset structures. These are **educated guesses** that must be **validated against the actual dataset** once we have access to it. The actual column names, existence, and meanings may differ.

**How to Identify Leakage (General Principles):**

1. **Temporal Availability**: Ask "When is this information known?"
   - ✅ Available at application time → Safe to use
   - ❌ Only known after loan is issued/paid → Likely leakage

2. **Causal Direction**: Ask "Does this cause the outcome, or is it caused by the outcome?"
   - ✅ Causes outcome → Safe (e.g., credit score → default risk)
   - ❌ Caused by outcome → Leakage (e.g., payments made → default status)

3. **Post-Event Information**: Any data that only exists after the loan outcome is determined is leakage.

**Hypothetical High-Risk Leakage Columns (To Validate):**

*These are examples based on typical Lending Club datasets. Must verify against actual data.*

- `total_pymnt` - **Why leakage?** Only known after loan is issued and payments are made. If we're predicting at application time, this doesn't exist yet.
- `total_pymnt_inv` - **Why leakage?** Same as above - post-loan information.
- `total_rec_prncp` - **Why leakage?** "Received to date" means cumulative payments made over time. Only exists after loan starts.
- `total_rec_int` - **Why leakage?** Interest received accumulates over time, only known after payments.
- `recoveries` - **Why leakage?** Recovery happens AFTER a loan is charged off (defaulted). This is the outcome, not a predictor.
- `collection_recovery_fee` - **Why leakage?** Collection fees only exist after default occurs.
- `last_pymnt_d` - **Why leakage?** Last payment date only exists if payments have been made (post-loan).
- `last_pymnt_amnt` - **Why leakage?** Last payment amount only known after payment occurs.
- `out_prncp` - **Why leakage?** Outstanding principal decreases as payments are made. This changes over time and reflects payment history.
- Any columns with "recover", "collection", "payment", "pymnt" in the name - **Why?** These typically indicate post-loan events.

**Hypothetical Medium-Risk Leakage Columns (Review Carefully):**

*These need careful temporal analysis - depends on prediction timing.*

- `issue_d` - **Why medium risk?** 
  - If predicting at **application time** → Leakage (loan hasn't been issued yet)
  - If predicting at **approval time** (pre-issue) → May be acceptable
  - **Decision depends on use case**
- `loan_status` - **Why medium risk?** 
  - This is likely our **target variable** (converted to binary default indicator)
  - Or it may contain information that directly indicates the outcome
  - **Must verify what values this column contains**
- `pymnt_plan` - **Why medium risk?** 
  - Payment plan might be set at application (safe) or after approval (leakage)
  - **Need to verify when this is determined**

**Safe Columns (Available at Application Time):**
- Applicant demographics (income, employment, etc.)
- Loan characteristics (amount, term, purpose, grade)
- Credit history (FICO scores, credit lines, etc.)
- Application metadata (application type, etc.)

**Action Plan:**
- Systematically review each column
- Check column descriptions/documentation
- Verify temporal availability (when is this information known?)
- Document decision for each potentially leaky column

### 3. Rejected Loans Handling

**Decision**: Include rejected loans for feature engineering only, not as training examples.

**Why Selection Bias Occurs:**

Selection bias happens when the data we use to train the model is not representative of the population we want to predict on.

**The Problem with Rejected Loans:**

1. **Missing Labels**: Rejected loans never had a chance to default or be paid off. We have **no outcome** (no label) for them.
   - We can't say "this rejected loan would have defaulted" - we simply don't know.

2. **Different Population**: Rejected loans are fundamentally different from approved loans:
   - They were rejected for a reason (higher risk, insufficient income, etc.)
   - Approved loans passed certain criteria
   - These are two different populations

3. **What Would Happen If We Included Them?**
   - **Option A**: Label them arbitrarily (e.g., all as "would default") → **Wrong labels, model learns incorrect patterns**
   - **Option B**: Exclude them from having a label → **Can't use them for supervised learning**
   - **Option C**: Try to predict what would have happened → **Speculation, not data-driven**

4. **Selection Bias Example:**
   - Imagine we only train on approved loans, but in production we need to predict on all applications
   - The model learns patterns from a "selected" population (approved loans)
   - This creates bias because approved loans are not a random sample - they passed filters

**Our Solution:**

- **Training**: Use only approved loans with known outcomes (defaulted or paid off)
- **Feature Engineering**: Use rejected loans to create aggregate features:
  - Example: "rejection_rate_by_grade" - what % of grade A loans are rejected?
  - This provides context without introducing bias
  - These features are available at application time for all loans

**Action Plan:**
- Identify rejected loan records
- Separate them from approved loans
- Use approved loans for model training
- Use rejected loans to create aggregate features (e.g., rejection_rate_by_grade)

### 4. Temporal Split Strategy

**Strategy**: Train on older vintages, test on recent ones

**Implementation:**
- Identify date column(s) for temporal ordering
- Define train/test split date threshold
- Ensure no temporal leakage (no future information in training)
- Document the split rationale

**Considerations:**
- Ensure sufficient data in both train and test sets
- Monitor for distribution shifts between periods
- Consider validation set from middle period if needed

## Validation Checklist

- [ ] Dataset downloaded and accessible
- [ ] All vintage files identified and listed
- [ ] Schema comparison completed across vintages
- [ ] Schema inconsistencies documented
- [ ] Target variable identified and validated
- [ ] Leakage columns identified and documented
- [ ] Dropped columns list created with justifications
- [ ] Rejected loans identified and separated
- [ ] Temporal split strategy implemented
- [ ] Validation report generated

## Output Deliverables

1. **Schema Validation Report**: Document schema differences across vintages
2. **Leakage Analysis Document**: List of all columns with leakage assessment
3. **Dropped Columns Log**: Final list of dropped columns with justifications
4. **Data Dictionary**: Cleaned data dictionary for approved columns
5. **Temporal Split Documentation**: Train/test split details and rationale

## Next Steps

After validation is complete:
- Proceed to Step 4: Data Ingestion (ETL - Raw Layer)
- Implement reproducible data loading
- Log dataset version, schema, and basic stats

