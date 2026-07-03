import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

RANDOM_STATE = 42
N_SPLITS = 5


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    df["aspect_ratio_px"] = df["height_px"] / (df["thickness_px"] + eps)
    df["area_px"] = df["height_px"] * df["thickness_px"]
    df["est_volume_proxy"] = df["height_mm_est"] * df["thickness_mm_est"] * df["depth_rs_mm"]
    df["thickness_to_depth"] = df["thickness_mm_est"] / (df["depth_rs_mm"] + eps)
    df["height_to_depth"] = df["height_mm_est"] / (df["depth_rs_mm"] + eps)

    return df


def build_preprocessor(numeric_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )


def get_models():
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "huber": HuberRegressor(max_iter=1000),
        "rf": RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "gbr": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def regression_metrics(y_true, y_pred):
    abs_err = np.abs(y_true - y_pred)
    within_50g = np.mean(abs_err <= 50)
    within_100g = np.mean(abs_err <= 100)
    within_150g = np.mean(abs_err <= 150)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "p90_abs_err": np.percentile(abs_err, 90),
        "within_50g": within_50g,
        "within_100g": within_100g,
        "within_150g": within_150g,
    }


def evaluate_models(df, feature_cols, target_col, label):
    X = df[feature_cols]
    y = df[target_col]

    preprocessor = build_preprocessor(feature_cols)
    models = get_models()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = []

    for model_name, model in models.items():
        fold_metrics = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            fold_metrics.append(regression_metrics(y_val, y_pred))

        summary = {"setting": label, "model": model_name}
        for key in fold_metrics[0]:
            summary[key] = np.mean([m[key] for m in fold_metrics])

        results.append(summary)

    return pd.DataFrame(results).sort_values(by="mae")


def train_final_model(df, feature_cols, target_col, model_name):
    X = df[feature_cols]
    y = df[target_col]

    preprocessor = build_preprocessor(feature_cols)
    model = get_models()[model_name]

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X, y)
    return pipe


def main():
    df = pd.read_csv("weight_dataset.csv")
    df = add_features(df)

    # merge predicted width from width stage
    width_preds = pd.read_csv("outputs/width_predictions.csv")[["filename", "pred_width_mm"]]
    df = df.merge(width_preds, on="filename", how="left")

    target_col = "mass"

    base_features = [
        "height_px",
        "thickness_px",
        "depth_rs_mm",
        "height_mm_est",
        "thickness_mm_est",
        "aspect_ratio_px",
        "area_px",
        "est_volume_proxy",
        "thickness_to_depth",
        "height_to_depth",
    ]

    features_no_width = base_features.copy()
    features_pred_width = base_features + ["pred_width_mm"]
    features_true_width = base_features + ["W_mm"]  # upper-bound only

    results_no_width = evaluate_models(df, features_no_width, target_col, "no_width")
    results_pred_width = evaluate_models(df, features_pred_width, target_col, "pred_width")
    results_true_width = evaluate_models(df, features_true_width, target_col, "true_width_upper_bound")

    all_results = pd.concat(
        [results_no_width, results_pred_width, results_true_width],
        ignore_index=True
    ).sort_values(by="mae")

    print("\nWEIGHT RESULTS")
    print(all_results)

    # separate deployable vs upper bound
    deployable_results = all_results[
        all_results["setting"].isin(["no_width", "pred_width"])
    ].sort_values(by="mae")

    upper_bound_results = all_results[
        all_results["setting"] == "true_width_upper_bound"
    ].sort_values(by="mae")

    print("\n=== DEPLOYABLE RESULTS ===")
    print(deployable_results)

    print("\n=== UPPER BOUND (ANALYSIS ONLY) ===")
    print(upper_bound_results)

    # choose only from deployable
    best_row = deployable_results.iloc[0]
    best_setting = best_row["setting"]
    best_model_name = best_row["model"]

    if best_setting == "no_width":
        final_features = features_no_width
    elif best_setting == "pred_width":
        final_features = features_pred_width

    final_model = train_final_model(df, final_features, target_col, best_model_name)
    df["pred_mass"] = final_model.predict(df[final_features])

    os.makedirs("outputs", exist_ok=True)
    all_results.to_csv("outputs/weight_model_results.csv", index=False)
    df.to_csv("outputs/weight_predictions.csv", index=False)
    joblib.dump(final_model, "outputs/weight_model.joblib")

    print(f"\nBest weight setting: {best_setting}")
    print(f"Best weight model: {best_model_name}")
    print("Saved weight model and predictions.")
    print(df["mass"].describe())


if __name__ == "__main__":
    main()