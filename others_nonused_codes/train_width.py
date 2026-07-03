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
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "p90_abs_err": np.percentile(abs_err, 90),
    }


def evaluate_models(df, feature_cols, target_col):
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

        summary = {"model": model_name}
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
    df = pd.read_csv("width_dataset.csv")
    df = add_features(df)

    feature_cols = [
        "height_px",
        "thickness_px",
        "depth_rs_mm",
        "height_mm_est",
        "thickness_mm_est",
        "aspect_ratio_px",
        "area_px",
        "thickness_to_depth",
        "height_to_depth",
    ]

    target_col = "W_mm"

    results = evaluate_models(df, feature_cols, target_col)
    print("\nWIDTH RESULTS")
    print(results)

    best_model_name = results.iloc[0]["model"]
    final_model = train_final_model(df, feature_cols, target_col, best_model_name)

    df["pred_width_mm"] = final_model.predict(df[feature_cols])

    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/width_model_results.csv", index=False)
    df.to_csv("outputs/width_predictions.csv", index=False)
    joblib.dump(final_model, "outputs/width_model.joblib")

    print(f"\nBest width model: {best_model_name}")
    print("Saved width model and predictions.")


if __name__ == "__main__":
    main()