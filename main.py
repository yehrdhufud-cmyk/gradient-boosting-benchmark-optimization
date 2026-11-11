# ============================================================
# ADVANCED BOOSTING BENCHMARK & OPTUNA OPTIMIZATION SYSTEM
# HGB | XGBoost | LightGBM | CatBoost | Optuna
# ============================================================

import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import optuna


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SAMPLES = 1200
TEST_SIZE = 0.20

CV_FOLDS = 3
OPTUNA_TRIALS = 15

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "boosting_outputs"
MODEL_DIR = OUTPUT_DIR / "models"
CHART_DIR = OUTPUT_DIR / "charts"

for directory in [OUTPUT_DIR, MODEL_DIR, CHART_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)

warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# 2. TARGETS AND FEATURES
# ============================================================

FEATURE_NAMES = [
    "Area_SQM",
    "Bedrooms",
    "Bathrooms",
    "Property_Age",
    "Distance_To_Center",
    "Transit_Access",
    "School_Rating",
    "Crime_Rate",
    "Neighborhood_Income",
    "Floor_Number",
    "Parking_Spaces"
]

TARGET_NAMES = [
    "Sale_Price",
    "Monthly_Rent",
    "Price_Per_SQM"
]


# ============================================================
# 3. VISUALIZATION THEME
# ============================================================

plt.rcParams.update({
    "figure.facecolor": "#080b12",
    "axes.facecolor": "#101725",
    "axes.edgecolor": "#35445c",
    "axes.labelcolor": "#dce6f2",
    "text.color": "#edf4ff",
    "xtick.color": "#b9c7da",
    "ytick.color": "#b9c7da",
    "grid.color": "#35445c",
    "grid.alpha": 0.35,
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "figure.titlesize": 17,
    "savefig.facecolor": "#080b12"
})

COLORS = [
    "#00d4ff",
    "#a78bfa",
    "#ff4d9d",
    "#00e5a8",
    "#ffb84d",
    "#ff6b6b"
]

MODEL_COLORS = {
    "HGB": "#00d4ff",
    "XGBoost": "#a78bfa",
    "LightGBM": "#00e5a8",
    "CatBoost": "#ff4d9d"
}


def save_chart(filename):

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="#080b12"
    )

    plt.show()
    plt.close()


# ============================================================
# 4. SYNTHETIC HOUSING DATASET
# ============================================================

print("\n" + "=" * 65)
print("ADVANCED BOOSTING BENCHMARK & OPTUNA OPTIMIZATION")
print("=" * 65)

print("\n[1/12] Generating synthetic housing dataset...")

rng = np.random.default_rng(RANDOM_STATE)

area = rng.uniform(40, 250, N_SAMPLES)

bedrooms = np.clip(
    np.round(area / 35 + rng.normal(0, 0.7, N_SAMPLES)),
    1,
    7
).astype(int)

bathrooms = np.clip(
    np.round(bedrooms * 0.7 + rng.normal(0, 0.5, N_SAMPLES)),
    1,
    5
).astype(int)

property_age = rng.integers(0, 60, N_SAMPLES)

distance = rng.uniform(0.5, 30, N_SAMPLES)

transit_access = rng.integers(1, 11, N_SAMPLES)

school_rating = rng.uniform(1, 10, N_SAMPLES)

crime_rate = rng.uniform(1, 100, N_SAMPLES)

neighborhood_income = rng.uniform(20000, 150000, N_SAMPLES)

floor_number = rng.integers(1, 31, N_SAMPLES)

parking_spaces = rng.integers(0, 4, N_SAMPLES)


# ------------------------------------------------------------
# Sale Price Generation
# ------------------------------------------------------------

price_per_sqm = (
    1800
    + neighborhood_income * 0.025
    + school_rating * 180
    + transit_access * 90
    - distance * 35
    - crime_rate * 8
    - property_age * 12
    + rng.normal(0, 180, N_SAMPLES)
)

price_per_sqm = np.maximum(price_per_sqm, 500)

sale_price = (
    area * price_per_sqm
    + bedrooms * 3500
    + bathrooms * 2200
    + parking_spaces * 4500
    + floor_number * 100
    + rng.normal(0, 12000, N_SAMPLES)
)

sale_price = np.maximum(sale_price, 20000)


# ------------------------------------------------------------
# Monthly Rent Generation
# ------------------------------------------------------------

monthly_rent = (
    sale_price * 0.004
    + bedrooms * 120
    + bathrooms * 80
    + transit_access * 35
    - property_age * 8
    + rng.normal(0, 100, N_SAMPLES)
)

monthly_rent = np.maximum(monthly_rent, 100)


# ------------------------------------------------------------
# Feature Dataset
# ------------------------------------------------------------

X = pd.DataFrame({
    "Area_SQM": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Property_Age": property_age,
    "Distance_To_Center": distance,
    "Transit_Access": transit_access,
    "School_Rating": school_rating,
    "Crime_Rate": crime_rate,
    "Neighborhood_Income": neighborhood_income,
    "Floor_Number": floor_number,
    "Parking_Spaces": parking_spaces
})

y = pd.DataFrame({
    "Sale_Price": sale_price,
    "Monthly_Rent": monthly_rent,
    "Price_Per_SQM": price_per_sqm
})

dataset = pd.concat([X, y], axis=1)

dataset.to_csv(
    OUTPUT_DIR / "housing_dataset.csv",
    index=False
)

print(f"Dataset shape: {dataset.shape}")
print(f"Features: {X.shape[1]}")
print(f"Targets: {y.shape[1]}")
print(f"Samples: {len(dataset)}")


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[2/12] Exploratory Data Analysis...")

print("\nDataset Information:")
print(dataset.info())

print("\nStatistical Summary:")
print(dataset.describe().round(2))

print("\nMissing Values:")
print(dataset.isnull().sum())


# ------------------------------------------------------------
# Target Distributions
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for i, target in enumerate(TARGET_NAMES):

    axes[i].hist(
        y[target],
        bins=35,
        color=COLORS[i],
        alpha=0.85,
        edgecolor="white"
    )

    axes[i].set_title(f"{target} Distribution")
    axes[i].set_xlabel(target)
    axes[i].set_ylabel("Frequency")
    axes[i].grid(True)

plt.suptitle("Housing Target Distributions")
save_chart("target_distributions.png")


# ------------------------------------------------------------
# Correlation Matrix
# ------------------------------------------------------------

correlation = dataset.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(13, 10))

image = ax.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.columns)))

ax.set_xticklabels(
    correlation.columns,
    rotation=90
)

ax.set_yticklabels(correlation.columns)

plt.colorbar(image, ax=ax, label="Correlation")

ax.set_title("Feature and Target Correlation")

save_chart("correlation_matrix.png")


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

print("\n[3/12] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 7. MODEL FACTORY
# ============================================================

def build_model(model_name, params=None):

    params = params or {}

    if model_name == "HGB":

        estimator = HistGradientBoostingRegressor(
            max_iter=params.get("max_iter", 200),
            learning_rate=params.get("learning_rate", 0.08),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            l2_regularization=params.get("l2_regularization", 0.0),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            random_state=RANDOM_STATE
        )

    elif model_name == "XGBoost":

        estimator = XGBRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            min_child_weight=params.get("min_child_weight", 1),
            reg_lambda=params.get("reg_lambda", 1.0),
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0
        )

    elif model_name == "LightGBM":

        estimator = LGBMRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", -1),
            num_leaves=params.get("num_leaves", 31),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            reg_lambda=params.get("reg_lambda", 0.0),
            verbosity=-1,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

    elif model_name == "CatBoost":

        estimator = CatBoostRegressor(
            iterations=params.get("iterations", 300),
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            loss_function="RMSE",
            verbose=False,
            random_seed=RANDOM_STATE,
            thread_count=1
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return MultiOutputRegressor(estimator, n_jobs=1)


MODEL_NAMES = [
    "HGB",
    "XGBoost",
    "LightGBM",
    "CatBoost"
]


# ============================================================
# 8. OPTUNA HYPERPARAMETER SEARCH SPACE
# ============================================================

def suggest_parameters(trial, model_name):

    if model_name == "HGB":

        return {
            "max_iter": trial.suggest_int(
                "max_iter", 100, 500
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),

            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", 15, 63
            ),

            "l2_regularization": trial.suggest_float(
                "l2_regularization", 0.0001, 10.0, log=True
            ),

            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 10, 40
            )
        }

    elif model_name == "XGBoost":

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 150, 600
            ),

            "max_depth": trial.suggest_int(
                "max_depth", 3, 10
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),

            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 10
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.01, 10.0, log=True
            )
        }

    elif model_name == "LightGBM":

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 150, 600
            ),

            "max_depth": trial.suggest_int(
                "max_depth", 3, 12
            ),

            "num_leaves": trial.suggest_int(
                "num_leaves", 15, 63
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),

            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.01, 10.0, log=True
            )
        }

    elif model_name == "CatBoost":

        return {
            "iterations": trial.suggest_int(
                "iterations", 150, 600
            ),

            "depth": trial.suggest_int(
                "depth", 4, 10
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),

            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 10.0
            )
        }


# ============================================================
# 9. OPTUNA OBJECTIVE FUNCTION
# ============================================================

def objective(trial, model_name):

    params = suggest_parameters(trial, model_name)

    kfold = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_scores = []

    for fold, (train_index, valid_index) in enumerate(
        kfold.split(X_train)
    ):

        X_fold_train = X_train.iloc[train_index]
        X_fold_valid = X_train.iloc[valid_index]

        y_fold_train = y_train.iloc[train_index]
        y_fold_valid = y_train.iloc[valid_index]

        model = build_model(model_name, params)

        model.fit(
            X_fold_train,
            y_fold_train
        )

        predictions = model.predict(X_fold_valid)

        score = r2_score(
            y_fold_valid,
            predictions,
            multioutput="uniform_average"
        )

        fold_scores.append(score)

    return float(np.mean(fold_scores))


# ============================================================
# 10. OPTIMIZATION AND MODEL TRAINING
# ============================================================

print("\n[4/12] Starting Optuna optimization...")

best_models = {}
studies = {}
best_parameters = {}

for model_name in MODEL_NAMES:

    print("\n" + "-" * 60)
    print(f"Optimizing: {model_name}")
    print("-" * 60)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        )
    )

    study.optimize(
        lambda trial: objective(trial, model_name),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True
    )

    studies[model_name] = study

    best_parameters[model_name] = study.best_params

    print(f"Best CV R2: {study.best_value:.5f}")
    print(f"Best Parameters: {study.best_params}")

    print(f"\nTraining optimized {model_name}...")

    best_model = build_model(
        model_name,
        study.best_params
    )

    best_model.fit(
        X_train,
        y_train
    )

    best_models[model_name] = best_model

    joblib.dump(
        best_model,
        MODEL_DIR / f"{model_name.lower()}_best_model.pkl"
    )


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

print("\n[5/12] Evaluating optimized models...")

results = []

predictions_by_model = {}

for model_name, model in best_models.items():

    train_predictions = model.predict(X_train)

    test_predictions = model.predict(X_test)

    predictions_by_model[model_name] = test_predictions

    train_r2 = r2_score(
        y_train,
        train_predictions,
        multioutput="uniform_average"
    )

    test_r2 = r2_score(
        y_test,
        test_predictions,
        multioutput="uniform_average"
    )

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    mse = mean_squared_error(
        y_test,
        test_predictions
    )

    rmse = np.sqrt(mse)

    results.append({
        "Model": model_name,
        "CV_R2": studies[model_name].best_value,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Overfitting_Gap": train_r2 - test_r2
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test_R2",
    ascending=False
).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\nMODEL PERFORMANCE")
print(results_df.round(5).to_string(index=False))


# ============================================================
# 12. BEST MODEL SELECTION
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = best_models[best_model_name]

best_predictions = predictions_by_model[best_model_name]

print("\n" + "=" * 65)
print("BEST MODEL")
print("=" * 65)

print(f"Model: {best_model_name}")
print(f"Test R2: {results_df.iloc[0]['Test_R2']:.5f}")
print(f"MAE: {results_df.iloc[0]['MAE']:.5f}")
print(f"RMSE: {results_df.iloc[0]['RMSE']:.5f}")


# ============================================================
# 13. MODEL COMPARISON VISUALIZATION
# ============================================================

print("\n[6/12] Creating model comparison charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

metrics = [
    ("Test_R2", "Test R2 Score"),
    ("MAE", "Mean Absolute Error"),
    ("RMSE", "Root Mean Squared Error"),
    ("Overfitting_Gap", "Overfitting Gap")
]

for ax, (metric, title) in zip(axes.flatten(), metrics):

    colors = [
        MODEL_COLORS[name]
        for name in results_df["Model"]
    ]

    ax.bar(
        results_df["Model"],
        results_df[metric],
        color=colors,
        edgecolor="white",
        alpha=0.9
    )

    ax.set_title(title)
    ax.grid(axis="y")

    ax.tick_params(axis="x", rotation=15)

plt.suptitle("Optimized Boosting Model Comparison")

save_chart("model_comparison.png")


# ============================================================
# 14. ACTUAL VS PREDICTED
# ============================================================

print("\n[7/12] Creating actual vs predicted charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, target in enumerate(TARGET_NAMES):

    actual = y_test[target].values

    predicted = best_predictions[:, i]

    axes[i].scatter(
        actual,
        predicted,
        color=COLORS[i],
        alpha=0.65,
        edgecolors="none"
    )

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    axes[i].plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        color="white",
        linewidth=1.5
    )

    axes[i].set_title(f"{target}: Actual vs Predicted")
    axes[i].set_xlabel("Actual")
    axes[i].set_ylabel("Predicted")
    axes[i].grid(True)

plt.suptitle(f"Best Model: {best_model_name}")

save_chart("actual_vs_predicted.png")


# ============================================================
# 15. RESIDUAL ANALYSIS
# ============================================================

print("\n[8/12] Residual analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, target in enumerate(TARGET_NAMES):

    actual = y_test[target].values

    predicted = best_predictions[:, i]

    residuals = actual - predicted

    axes[i].scatter(
        predicted,
        residuals,
        color=COLORS[i],
        alpha=0.65
    )

    axes[i].axhline(
        0,
        color="white",
        linestyle="--"
    )

    axes[i].set_title(f"{target} Residuals")
    axes[i].set_xlabel("Predicted Values")
    axes[i].set_ylabel("Residuals")
    axes[i].grid(True)

plt.suptitle("Residual Analysis")

save_chart("residual_analysis.png")


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

print("\n[9/12] Calculating feature importance...")

importance_result = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="r2",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Importance": importance_result.importances_mean,
    "Std": importance_result.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1],
    xerr=importance_df["Std"][::-1],
    color=COLORS[0],
    alpha=0.85
)

ax.set_title(f"Permutation Feature Importance - {best_model_name}")
ax.set_xlabel("Mean R2 Score Decrease")
ax.grid(axis="x")

save_chart("feature_importance.png")


# ============================================================
# 17. OPTUNA OPTIMIZATION HISTORY
# ============================================================

print("\n[10/12] Creating Optuna optimization charts...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for ax, model_name in zip(axes.flatten(), MODEL_NAMES):

    study = studies[model_name]

    trial_numbers = [
        trial.number + 1
        for trial in study.trials
        if trial.value is not None
    ]

    trial_scores = [
        trial.value
        for trial in study.trials
        if trial.value is not None
    ]

    best_so_far = np.maximum.accumulate(trial_scores)

    ax.plot(
        trial_numbers,
        trial_scores,
        marker="o",
        alpha=0.5,
        label="Trial Score",
        color=MODEL_COLORS[model_name]
    )

    ax.plot(
        trial_numbers,
        best_so_far,
        linewidth=2,
        label="Best Score",
        color="#ffffff"
    )

    ax.set_title(f"{model_name} Optimization History")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV R2")
    ax.legend()
    ax.grid(True)

plt.suptitle("Optuna Optimization History")

save_chart("optuna_optimization_history.png")


# ============================================================
# 18. TARGET-WISE MODEL PERFORMANCE
# ============================================================

print("\n[11/12] Calculating target-wise performance...")

target_results = []

for model_name, model_predictions in predictions_by_model.items():

    for i, target in enumerate(TARGET_NAMES):

        actual = y_test[target].values

        predicted = model_predictions[:, i]

        target_results.append({
            "Model": model_name,
            "Target": target,
            "MAE": mean_absolute_error(
                actual,
                predicted
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            ),
            "R2": r2_score(
                actual,
                predicted
            )
        })


target_results_df = pd.DataFrame(target_results)

target_results_df.to_csv(
    OUTPUT_DIR / "target_wise_performance.csv",
    index=False
)

print("\nTARGET-WISE PERFORMANCE")

print(
    target_results_df.round(5).to_string(index=False)
)


# ============================================================
# 19. SAVE PREDICTION REPORT
# ============================================================

prediction_report = pd.DataFrame(
    best_predictions,
    columns=TARGET_NAMES
)

prediction_report = prediction_report.add_prefix(
    "Predicted_"
)

actual_report = y_test.reset_index(drop=True).add_prefix(
    "Actual_"
)

prediction_report = pd.concat(
    [actual_report, prediction_report],
    axis=1
)

prediction_report.to_csv(
    OUTPUT_DIR / "best_model_predictions.csv",
    index=False
)


# ============================================================
# 20. SAVE OPTIMIZATION METADATA
# ============================================================

metadata = {
    "project": "Advanced Boosting Benchmark & Optuna Optimization",
    "random_state": RANDOM_STATE,
    "samples": N_SAMPLES,
    "features": FEATURE_NAMES,
    "targets": TARGET_NAMES,
    "test_size": TEST_SIZE,
    "cv_folds": CV_FOLDS,
    "optuna_trials_per_model": OPTUNA_TRIALS,
    "best_model": best_model_name,
    "best_model_test_r2": float(
        results_df.iloc[0]["Test_R2"]
    ),
    "best_parameters": best_parameters,
    "model_results": results_df.to_dict(orient="records")
}

with open(
    OUTPUT_DIR / "experiment_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n[12/12] Project completed!")

print("\n" + "=" * 65)
print("FINAL PROJECT SUMMARY")
print("=" * 65)

print(f"Dataset Samples: {N_SAMPLES}")
print(f"Number of Features: {len(FEATURE_NAMES)}")
print(f"Number of Targets: {len(TARGET_NAMES)}")

print("\nModels:")
for model_name in MODEL_NAMES:
    print(f"- {model_name}")

print("\nOptimization:")
print(f"- Optimizer: Optuna (TPE Sampler)")
print(f"- Trials per model: {OPTUNA_TRIALS}")
print(f"- Cross Validation: {CV_FOLDS}-Fold")

print("\nBest Model:")
print(f"- Name: {best_model_name}")
print(
    f"- Test R2: {results_df.iloc[0]['Test_R2']:.5f}"
)

print("\nOutput Directory:")
print(OUTPUT_DIR)

print("\nGenerated Files:")
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file():
        print(f"- {path.relative_to(OUTPUT_DIR)}")

print("\n" + "=" * 65)
print("ALL TASKS FINISHED SUCCESSFULLY")
print("=" * 65)