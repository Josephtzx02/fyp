# train_width_weight.py

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# Optional
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor

RANDOM_STATE = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    eps = 1e-6
    df["aspect_ratio_px"] = df["height_px"] / (df["thickness_px"] + eps)
    df["area_px"] = df["height_px"] * df["thickness_px"]
    df["est_volume_proxy"] = df["height_mm_est"] * df["thickness_mm_est"] * df["depth_rs_mm"]
    df["thickness_to_depth"] = df["thickness_mm_est"] / (df["depth_rs_mm"] + eps)
    df["height_to_depth"] = df["height_mm_est"] / (df["depth_rs_mm"] + eps)

    return df


def build_preprocessor(numeric_features, categorical_features=None):
    if categorical_features is None:
        categorical_features = []

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
            ]
        )

    return preprocessor


def get_models():
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "huber": HuberRegressor(max_iter=1000),
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "gbr": GradientBoostingRegressor(random_state=RANDOM_STATE),
        # "xgb": XGBRegressor(...),
        # "lgbm": LGBMRegressor(...),
        # "catboost": CatBoostRegressor(verbose=0, random_state=RANDOM_STATE),
    }
    return models


def regression_metrics(y_true, y_pred):
    abs_err = np.abs(y_true - y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "p90_abs_err": np.percentile(abs_err, 90),
    }


def evaluate_models(df, feature_cols, target_col, categorical_cols=None, n_splits=5):
    if categorical_cols is None:
        categorical_cols = []

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = get_models()

    X = df[feature_cols]
    y = df[target_col]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
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

            m = regression_metrics(y_val, y_pred)
            fold_metrics.append(m)

        summary = {"model": model_name}
        for key in fold_metrics[0].keys():
            summary[key] = np.mean([fm[key] for fm in fold_metrics])

        results.append(summary)

    results_df = pd.DataFrame(results).sort_values(by="mae")
    return results_df


def train_final_model(df, feature_cols, target_col, model_name="rf", categorical_cols=None):
    if categorical_cols is None:
        categorical_cols = []

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = get_models()[model_name]

    X = df[feature_cols]
    y = df[target_col]

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X, y)
    return pipe


def main():
    df = pd.read_csv("book_dataset.csv")
    df = add_features(df)

    # ---------- WIDTH ----------
    width_features = [
        "height_px",
        "thickness_px",
        "depth_rs_mm",
        "height_mm_est",
        "thickness_mm_est",
        "aspect_ratio_px",
        "area_px"
    ]

    width_results = evaluate_models(
        df=df,
        feature_cols=width_features,
        target_col="W_mm",
        categorical_cols=[]
    )
    print("\nWIDTH RESULTS")
    print(width_results)

    best_width_model_name = width_results.iloc[0]["model"]
    width_pipe = train_final_model(
        df=df,
        feature_cols=width_features,
        target_col="W_mm",
        model_name=best_width_model_name,
        categorical_cols=[]
    )

    df["pred_width_mm"] = width_pipe.predict(df[width_features])

    # ---------- WEIGHT ----------
    weight_features = [
        "height_px",
        "thickness_px",
        "depth_rs_mm",
        "height_mm_est",
        "thickness_mm_est",
        "aspect_ratio_px",
        "area_px",
        "est_volume_proxy",
        "pred_width_mm"
    ]

    weight_results = evaluate_models(
        df=df,
        feature_cols=weight_features,
        target_col="mass",
        categorical_cols=[]
    )
    print("\nWEIGHT RESULTS")
    print(weight_results)

    best_weight_model_name = weight_results.iloc[0]["model"]
    weight_pipe = train_final_model(
        df=df,
        feature_cols=weight_features,
        target_col="mass",
        model_name=best_weight_model_name,
        categorical_cols=[]
    )

    # save outputs
    os.makedirs("outputs", exist_ok=True)
    width_results.to_csv("outputs/width_model_results.csv", index=False)
    weight_results.to_csv("outputs/weight_model_results.csv", index=False)
    df.to_csv("outputs/book_dataset_with_predictions.csv", index=False)

    print("\nDone.")
    print(f"Best width model: {best_width_model_name}")
    print(f"Best weight model: {best_weight_model_name}")


if __name__ == "__main__":
    main()