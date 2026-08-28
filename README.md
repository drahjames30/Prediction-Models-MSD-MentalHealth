# Mental Health & MSD Prediction Models

Explainable machine learning models for predicting **musculoskeletal disorders (MSD)** and **mental health status**, using Multiple Logistic Regression (MLR), a standalone Multilayer Perceptron (MLP), and an **MLR-probability-enhanced MLP** hybrid model, interpreted using **SHAP** and **LIME**.

## Overview

Two independent, secondary health datasets are used as case studies to compare predictive performance and interpretability across three modelling approaches:

1. **MLR** – logistic regression baseline (coefficients, odds ratios)
2. **Standalone MLP** – neural network trained on the original predictors
3. **MLR-enhanced MLP** – the MLR-predicted probability is added as an extra input feature to the MLP, which then produces the final classification

SHAP and LIME are applied post-hoc to explain global feature importance and individual predictions for each model.

## Repository Contents

| File | Description |
|---|---|
| `CaseStudy_I_MSD.R` | Full R analysis script for the Musculoskeletal Disorders (MSD) case study |
| `CaseStudy_II_MentalHealth.R` | Full R analysis script for the Mental Health case study |
| `Dataset of MSD and Mental Health.csv` | Combined dataset used for both case studies |
| `README.md` | This file |

## Case Studies

### Case Study I: Musculoskeletal Disorders (MSD)
- **Outcome:** Binary MSD status
- **Predictors (17):** Gender, marital status, education, handedness, sports activity, working position, work environment, age, work experience, sleep hours, working days, working hours, musculoskeletal complaint history, BMI, and related sociodemographic/occupational variables

### Case Study II: Mental Health Status
- **Outcome:** Binary mental health status
- **Predictors (13):** Gender, age, marital status, education level, position status, socioeconomic status, religion, residence, income, history1, history2, duration

## Methods

- **Data split:** 70% training / 30% testing (`set.seed(123)` for reproducibility)
- **Preprocessing:** Min–max normalization
- **Models:** `glm()` (MLR), `neuralnet` / `nnet` (MLP)
- **Interpretability:** SHAP (via `iml`) and LIME (via `lime`)
- **Evaluation metrics:** Accuracy, sensitivity, specificity, PPV, NPV, balanced accuracy, AUC, Brier score, confusion matrix, McNemar's test, paired t-test

## Requirements

R (≥ 4.0) with the following packages:

```r
install.packages(c("caret", "neuralnet", "nnet", "ResourceSelection", 
                    "pROC", "iml", "lime", "dplyr", "ggplot2"))
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/drahjames30/Prediction-Models-MSD-MentalHealth.git
   ```
2. Open either `CaseStudy_I_MSD.R` or `CaseStudy_II_MentalHealth.R` in RStudio and run the sections in order:
   - Data loading & missing value check
   - Normalization
   - Train/test split
   - MLR model fitting
   - MLP model fitting
   - MLR-enhanced MLP model fitting
   - SHAP/LIME explainability
   - Performance evaluation (ROC, McNemar's test, paired t-test)

> **Note:** Some sections (e.g., the MLR-enhanced MLP and LIME/SHAP blocks) reference helper objects/functions defined earlier in the full pipeline (`safe_predict_glm`, `safe_train_nn`, `get_binary_metrics`, `select_explanation_cases`, `outcome_var`, `train_data`, `Training_mlr`, etc.). These should be defined in your session — or sourced from a shared helper script — before running those sections.

## Key Results

| Task | Model | Accuracy | Balanced Accuracy | AUC |
|---|---|---|---|---|
| MSD | MLR | 76.92% | – | – |
| MSD | Standalone MLP | 74.36% | Highest | – |
| MSD | MLR-enhanced MLP | **79.49%** | – | – |
| Mental Health | MLR-enhanced MLP | **88.20%** | 0.779 | 0.849 |

MSD results should be interpreted cautiously due to weak specificity across all models. Important predictors identified include BMI, work experience, working posture, physical activity, sleep duration, income, socioeconomic status, and psychosocial support.

## Data Source & Ethics

Both datasets are secondary data used for prediction model development and validation, not for prevalence estimation. Refer to the associated thesis for full ethical approval details, sampling methodology, and variable coding.

## Citation

If you use this repository, please cite the associated thesis/publications by **Arsalan Humayun** et al. (Universiti Sains Malaysia).

## License

For academic and research use. Contact the repository owner for other uses.
