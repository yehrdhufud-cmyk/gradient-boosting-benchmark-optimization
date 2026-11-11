# Gradient Boosting Benchmark & Hyperparameter Optimization System

A machine learning benchmarking and hyperparameter optimization system for evaluating and comparing four powerful gradient boosting algorithms using Optuna.

The project combines automated hyperparameter tuning, cross-validation, regression evaluation, feature importance analysis, and visualization in a unified workflow.

## Overview

This project benchmarks four gradient boosting models:

- **HistGradientBoosting (HGB)**
- **XGBoost**
- **LightGBM**
- **CatBoost**

Each model is optimized using **Optuna with the Tree-structured Parzen Estimator (TPE)** sampler.

The system evaluates model performance using multiple regression metrics and generates visual reports to support model comparison and analysis.

## Key Features

- Benchmarking of four gradient boosting algorithms.
- Automated hyperparameter optimization using Optuna.
- TPE-based optimization strategy.
- 3-fold cross-validation.
- Multi-output regression.
- Model evaluation using MAE, MSE, RMSE, and R².
- Training and testing performance gap analysis.
- Permutation feature importance.
- Actual vs. predicted value visualization.
- Residual analysis.
- Optuna optimization history visualization.
- Automatic model serialization using Joblib.
- CSV and JSON report generation.
- Automated output directory creation.

## Technologies & Libraries

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Hyperparameter Optimization | Optuna |
| Data Processing | NumPy, Pandas |
| Data Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |

## Project Architecture

The project is implemented in a single Python script:

```text
gradient-boosting-benchmark-optimization/
│
├── main.py
├── README.md
├── requirements.txt
│
└── boosting_outputs/
    ├── models/
    └── charts/
```

The output directory is generated automatically when the program runs.

## Models

| Model | Description |
|---|---|
| HistGradientBoosting | Scikit-learn's histogram-based gradient boosting implementation |
| XGBoost | Optimized gradient boosting framework |
| LightGBM | Efficient gradient boosting framework using histogram-based learning |
| CatBoost | Gradient boosting algorithm with ordered boosting techniques |

## Optimization Strategy

Optuna is used to search for suitable hyperparameters for each model.

The optimization workflow includes:

1. Defining model-specific hyperparameter search spaces.
2. Sampling hyperparameters using the TPE sampler.
3. Evaluating candidate configurations through 3-fold cross-validation.
4. Selecting the best-performing configuration.
5. Training the optimized model.
6. Comparing the optimized models using evaluation metrics.

## Evaluation Metrics

The system uses the following metrics:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination
- **Train-Test Gap:** Difference between training and testing performance, used to help identify potential overfitting.

## Visualizations

The system generates charts for:

- Target variable distributions.
- Feature correlation matrix.
- Model performance comparison.
- Actual vs. predicted values.
- Residual analysis.
- Permutation feature importance.
- Optuna optimization history.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/gradient-boosting-benchmark-optimization.git
```

Navigate to the project directory:

```bash
cd gradient-boosting-benchmark-optimization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

The program automatically:

1. Generates a synthetic regression dataset.
2. Trains and optimizes the four boosting models.
3. Evaluates model performance.
4. Generates visualization charts.
5. Saves trained models and reports.

## Output

Generated files are stored in:

```text
boosting_outputs/
├── models/
│   └── Trained model files
│
└── charts/
    └── Performance and optimization charts
```

The project also generates structured evaluation reports in CSV and JSON formats.

## Dataset

This project uses a synthetic housing dataset generated programmatically with NumPy and Scikit-learn.

It contains 1,200 samples, 11 input features, and 3 regression targets.

The synthetic dataset provides a controlled environment for benchmarking and testing optimization workflows.

**Note:** The reported results are based on synthetic data and should not be interpreted as real-world housing market predictions.

## Future Improvements

- Benchmarking on real-world datasets.
- Expanded hyperparameter search spaces.
- Advanced optimization algorithms.
- Early stopping and computational efficiency analysis.
- Additional regression models.
- Reproducible experiment tracking.

## Author

Developed as a machine learning portfolio project focused on gradient boosting, model benchmarking, and automated hyperparameter optimization.