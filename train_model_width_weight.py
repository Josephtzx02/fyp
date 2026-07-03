import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score

csv_path = "weight_dataset.csv"
df = pd.read_csv(csv_path)

target_col = "mass" 

df = df.dropna(subset=[target_col]).copy()

if all(col in df.columns for col in ["H_mm", "T_mm", "W_mm"]):
    df["true_volume_proxy"] = df["H_mm"] * df["T_mm"] * df["W_mm"]

if all(col in df.columns for col in ["height_px", "thickness_px"]):
    df["pixel_area_proxy"] = df["height_px"] * df["thickness_px"]

df["aspect_ratio_px"] = df["height_px"] / df["thickness_px"]
df["scale_h"] = df["height_mm_est"] / df["height_px"]
df["thickness_ratio"] = df["thickness_mm_est"] / df["height_mm_est"]
df["pixel_area_proxy"] = df["height_px"] * df["thickness_px"]

df["wide_textbook_zone"] = (
    (df["height_mm_est"] >= 228) &
    (df["height_mm_est"] <= 245) &
    (df["thickness_mm_est"] >= 15)
).astype(int)

df["wide_textbook_score"] = (
    df["wide_textbook_zone"] *
    df["height_mm_est"] *
    df["thickness_mm_est"]
)

df["possible_hardcover_wide"] = (
    (df["height_mm_est"].between(228, 245)) &
    (df["thickness_mm_est"] >= 18)
).astype(int)

df["possible_thin_royal"] = (
    (df["height_mm_est"].between(228, 245)) &
    (df["thickness_mm_est"] < 18)
).astype(int)

df["height_x_thickness_est"] = df["height_mm_est"] * df["thickness_mm_est"]
df["height_x_prior_width"] = np.nan
df["thickness_x_prior_width"] = np.nan
df["depth_x_height"] = df["depth_rs_mm"] * df["height_mm_est"]
df["thinness_score"] = df["height_mm_est"] / df["thickness_mm_est"]

BOOK_SIZE_PRIORS = [
    ("US_Pocket", 175, 108),
    ("Mass_Market", 178, 110),
    ("B6", 176, 125),
    ("B6_Slim", 176, 110),
    ("US_Trade_Small", 203, 127),
    ("A5", 210, 148),
    ("Digest", 216, 140),
    ("Demy_UK", 216, 138),
    ("US_Standard", 229, 152),
    ("Royal_UK", 234, 156),
    ("B5", 250, 176),
    ("Executive", 267, 184),
    ("Large_Textbook", 280, 215),
    ("A4", 297, 210),
]

def infer_book_size_from_height(h_mm):
    if pd.isna(h_mm):
        return pd.Series(["Unknown", np.nan, np.nan])
    
    # Find the book size with the minimum absolute height difference
    best_match = min(BOOK_SIZE_PRIORS, key=lambda x: abs(x[1] - h_mm))
    family_name, std_height, std_width = best_match
    height_error = abs(h_mm - std_height)
    
    return pd.Series([family_name, std_width, height_error])

df[["auto_paper_family", "standard_width_prior", "height_to_standard_error"]] = (
    df["height_mm_est"].apply(infer_book_size_from_height)
)

df["height_x_prior_width"] = df["height_mm_est"] * df["standard_width_prior"]
df["thickness_x_prior_width"] = df["thickness_mm_est"] * df["standard_width_prior"]

# Distance to every standard height
for family, h_std, w_std in BOOK_SIZE_PRIORS:
    df[f"dist_h_{family}"] = abs(df["height_mm_est"] - h_std)

# Also add nearest standard width ratio
df["prior_width_ratio"] = df["standard_width_prior"] / df["height_mm_est"]


# =========================
# 2B. Train width models and create width_pred_mm
# =========================

width_target = "W_mm"

width_features = [
    "height_px",
    "thickness_px",
    "depth_rs_mm",
    "height_mm_est",
    "thickness_mm_est",
    "aspect_ratio_px",
    "scale_h",
    "thickness_ratio",
    "pixel_area_proxy",
    "standard_width_prior",
    "height_to_standard_error",
    "prior_width_ratio",
    "auto_paper_family",

    # new interaction features
    "height_x_thickness_est",
    "height_x_prior_width",
    "thickness_x_prior_width",
    "depth_x_height",
    "thinness_score",
]

# Add all distance-to-standard features
width_features += [f"dist_h_{family}" for family, _, _ in BOOK_SIZE_PRIORS]

width_features = [col for col in width_features if col in df.columns]

print("\nWidth experiment: Soft paper-size priors without hard snapping")

width_data = df[width_features + [width_target]].replace([np.inf, -np.inf], np.nan).dropna().copy()

X_w = width_data[width_features]
y_w = width_data[width_target]

width_cat_cols = ["auto_paper_family"] if "auto_paper_family" in X_w.columns else []
width_num_cols = [col for col in X_w.columns if col not in width_cat_cols]

width_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), width_num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), width_cat_cols)
    ],
    remainder="drop"
)

width_models = {
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),
    "Ridge": Ridge(alpha=1.0),
    "Huber": HuberRegressor(max_iter=5000)
}

width_results = []
best_width_model = None
best_width_mae = float("inf")
best_width_name = None

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_w, y_w,
    test_size=0.25,
    random_state=42
)

for name, regressor in width_models.items():
    pipe = Pipeline([
        ("preprocess", width_preprocessor),
        ("model", regressor)
    ])

    pipe.fit(X_train_w, y_train_w)
    y_pred_w = pipe.predict(X_test_w)

    mae_w = mean_absolute_error(y_test_w, y_pred_w)
    rmse_w = np.sqrt(mean_squared_error(y_test_w, y_pred_w))
    r2_w = r2_score(y_test_w, y_pred_w)
    p90_w = np.percentile(np.abs(y_test_w - y_pred_w), 90)

    width_results.append({
        "width_model": name,
        "mae_mm": mae_w,
        "rmse_mm": rmse_w,
        "r2": r2_w,
        "p90_abs_err_mm": p90_w
    })

    if mae_w < best_width_mae:
        best_width_mae = mae_w
        best_width_model = pipe
        best_width_name = name

width_results_df = pd.DataFrame(width_results).sort_values(by="mae_mm")

print("\n================ WIDTH MODEL COMPARISON ================")
print(width_results_df.to_string(index=False))
print("========================================================")
print(f"\nBest width model: {best_width_name}")
print(f"Best width MAE: {best_width_mae:.3f} mm\n")

width_results_df.to_csv("width_model_comparison.csv", index=False)

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error

def evaluate_subset_width(data_subset, feature_cols, target="W_mm"):
    data_subset = data_subset[feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = data_subset[feature_cols]
    y = data_subset[target]

    cat_cols = ["auto_paper_family"] if "auto_paper_family" in X.columns else []
    num_cols = [col for col in X.columns if col not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    preds = cross_val_predict(
        model,
        X,
        y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42)
    )

    mae = mean_absolute_error(y, preds)

    print(f"Subset samples: {len(data_subset)}")
    print(f"Width MAE: {mae:.3f} mm")

    return mae

# =========================
# OOF width prediction
# =========================
from sklearn.base import clone

df["width_pred_mm"] = np.nan
oof_pred = np.zeros(len(X_w))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Grab your best performing pipeline setup
best_pipe = Pipeline([
    ("preprocess", width_preprocessor),
    ("model", clone(width_models["Random Forest"]))
])

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_w), start=1):
    X_train_fold = X_w.iloc[train_idx]
    y_train_fold = y_w.iloc[train_idx]
    X_valid_fold = X_w.iloc[valid_idx]
    
    # Train the cloned pipeline on this fold's training data
    fold_pipe = clone(best_pipe)
    fold_pipe.fit(X_train_fold, y_train_fold)
    
    # Predict on the validation slice and store it
    oof_pred[valid_idx] = fold_pipe.predict(X_valid_fold)
    print(f"Fold {fold} width prediction completed.")

# Map predictions back safely using index matching
df.loc[X_w.index, "width_pred_mm"] = oof_pred

# Evaluate OOF width performance
mae_oof = mean_absolute_error(y_w, oof_pred)
rmse_oof = np.sqrt(mean_squared_error(y_w, oof_pred))
r2_oof = r2_score(y_w, oof_pred)
p90_oof = np.percentile(np.abs(y_w - oof_pred), 90)

print("\n================ OOF WIDTH MODEL PERFORMANCE ================")
print(f"OOF MAE: {mae_oof:.3f} mm")
print(f"OOF RMSE: {rmse_oof:.3f} mm")
print(f"OOF R2: {r2_oof:.4f}")
print(f"OOF P90 Error: {p90_oof:.3f} mm")
print("=============================================================\n")

# Deployment volume proxy
df["est_volume_proxy"] = (
    df["height_mm_est"] *
    df["thickness_mm_est"] *
    df["width_pred_mm"]
)

df.to_csv("weight_dataset_with_width_pred_mm.csv", index=False)

print("\nWidth prediction completed.")
print("Saved updated CSV as: weight_dataset_with_width_pred_mm.csv")
print("Width model features used:", width_features)

print("\nFirst 5 predicted widths:")
print(df[["W_mm", "width_pred_mm"]].head())

AMB_LOW = 228
AMB_HIGH = 242

df_ambiguous = df[
    df["height_mm_est"].between(AMB_LOW, AMB_HIGH)
].copy()

df_non_ambiguous = df[
    ~df["height_mm_est"].between(AMB_LOW, AMB_HIGH)
].copy()

# =========================
# Weight error breakdown by ambiguous width zone
# =========================

AMB_LOW = 228
AMB_HIGH = 242

df["ambiguous_height_zone"] = df["height_mm_est"].between(AMB_LOW, AMB_HIGH)

print("\n================ WEIGHT ZONE BREAKDOWN ================")
print("Ambiguous zone samples:", df["ambiguous_height_zone"].sum())
print("Non-ambiguous samples:", (~df["ambiguous_height_zone"]).sum())

# ==========================================
# Ambiguous-height ablation split
# ==========================================

print("\n===== DATASET SPLIT CHECK =====")
print("Total samples:", len(df))
print("Ambiguous zone:", len(df_ambiguous))
print("Non-ambiguous:", len(df_non_ambiguous))

print("\n=========== ABLATION TEST ===========")

print("\nFull Dataset:")
evaluate_subset_width(
    df,
    width_features
)

print("\nNon-Ambiguous Only:")
evaluate_subset_width(
    df_non_ambiguous,
    width_features
)

print("\nAmbiguous Zone Only:")
evaluate_subset_width(
    df_ambiguous,
    width_features
)

# =========================
# Width error analysis
# =========================
df["width_abs_error"] = abs(df["W_mm"] - df["width_pred_mm"])

df["width_class"] = pd.cut(
    df["W_mm"],
    bins=[0, 130, 150, 170, 190, 999],
    labels=["very_narrow", "narrow", "medium", "wide", "very_wide"]
)

print("\n================ WIDTH ERROR ANALYSIS ================")
print("Top 30 worst width predictions:")
print(
    df[[
        "filename",
        "H_mm",
        "W_mm",
        "height_mm_est",
        "thickness_mm_est",
        "auto_paper_family",
        "standard_width_prior",
        "width_pred_mm",
        "width_abs_error",
        "width_class"
    ]]
    .sort_values("width_abs_error", ascending=False)
    .head(30)
    .to_string(index=False)
)

print("\nWidth error by class:")
print(
    df.groupby("width_class", observed=False)["width_abs_error"]
    .agg(["count", "mean", "median", "max"])
)

print("\nWidth error by paper family:")
print(
    df.groupby("auto_paper_family", observed=False)["width_abs_error"]
    .agg(["count", "mean", "median", "max"])
    .sort_values("mean", ascending=False)
)

royal = df[df["auto_paper_family"] == "Royal_UK"].copy()

royal["is_true_royal_like"] = (
    (royal["H_mm"].between(229, 239)) &
    (royal["W_mm"].between(145, 170))
)

royal["is_wide_textbook_like"] = (
    (royal["H_mm"].between(229, 245)) &
    (royal["W_mm"].between(180, 200))
)

print("\nROYAL_UK BREAKDOWN")
print("Total Royal_UK:", len(royal))
print("True Royal-like:", royal["is_true_royal_like"].sum())
print("Wide textbook-like:", royal["is_wide_textbook_like"].sum())

print("\nRoyal_UK width distribution:")
print(royal["W_mm"].describe())

print("\nRoyal_UK grouped width bins:")
print(pd.cut(
    royal["W_mm"],
    bins=[0, 145, 170, 180, 200, 999],
    labels=["<145", "145-170 true royal", "170-180 wide", "180-200 textbook", ">200"]
).value_counts().sort_index())


royal = df[df["auto_paper_family"] == "Royal_UK"].copy()

royal["royal_subtype"] = np.select(
    [
        royal["W_mm"].between(145, 170),
        royal["W_mm"].between(180, 200),
        royal["W_mm"] > 200
    ],
    [
        "true_royal_like",
        "wide_textbook_like",
        "square_or_outlier"
    ],
    default="other"
)

print("\nRoyal_UK subtype thickness check:")
print(
    royal.groupby("royal_subtype")[["T_mm", "thickness_mm_est", "H_mm", "W_mm", "mass"]]
    .agg(["count", "mean", "median", "min", "max"])
)

from sklearn.model_selection import cross_val_predict

def get_oof_weight_predictions(df, feature_list, target_col="mass"):
    data = df[feature_list + [target_col, "ambiguous_height_zone"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()

    X = data[feature_list]
    y = data[target_col]

    cat_cols = [col for col in feature_list if X[col].dtype == "object"]
    num_cols = [col for col in feature_list if col not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2
        ))
    ])

    pred = cross_val_predict(
        model,
        X,
        y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42)
    )

    data["weight_pred_oof"] = pred
    data["weight_abs_error"] = abs(data[target_col] - data["weight_pred_oof"])

    return data

df["height_x_thickness_est"] = df["height_mm_est"] * df["thickness_mm_est"]
df["px_density_proxy"] = df["pixel_area_proxy"] / df["depth_rs_mm"]
df["thickness_to_height_est"] = df["thickness_mm_est"] / df["height_mm_est"]
df["thickness_x_depth"] = df["thickness_mm_est"] * df["depth_rs_mm"]
df["height_x_depth"] = df["height_mm_est"] * df["depth_rs_mm"]

# =========================
# 3. Define Model A, B, C feature sets
# =========================

# Model A: clean upper-bound model
# Uses true measured book geometry
features_A = [
    "H_mm",
    "T_mm",
    "W_mm",
    "true_volume_proxy"
]

# Model B: deployment model
# Uses camera-estimated height/thickness + predicted width
features_B = [
    "height_mm_est",
    "thickness_mm_est",
    "width_pred_mm",
    "depth_rs_mm",
    "est_volume_proxy"
]

# Model C: camera-only / vision-only model
# Uses raw camera observable features
features_C = [
    "height_px",
    "thickness_px",
    "depth_rs_mm",
    "pixel_area_proxy"
]

# Model D: deployment model without predicted width
# Tests whether width_pred_mm is adding noise into weight prediction
features_D = [
    "height_mm_est",
    "thickness_mm_est",
    "depth_rs_mm",
    "height_x_thickness_est",
    "pixel_area_proxy",
    "px_density_proxy",
    "thickness_to_height_est",
    "thickness_x_depth",
    "height_x_depth",
    "auto_paper_family"
]

# Optional categorical features if available
categorical_candidates = [
    "auto_paper_family",
    "paper_family",
    "paper_family_code",
    "book_type",
    "book_type_code"
]

# =========================
# 4. Helper function
# =========================
def prepare_features(df, feature_list):
    available_features = [col for col in feature_list if col in df.columns]
    available_cats = [col for col in categorical_candidates if col in df.columns]

    final_features = list(dict.fromkeys(
        available_features + available_cats
    ))

    data = df[final_features + [target_col]].dropna().copy()

    X = data[final_features]
    y = data[target_col]

    numeric_features = [col for col in final_features if col not in available_cats]
    categorical_features = [col for col in final_features if col in available_cats]

    return X, y, numeric_features, categorical_features, final_features

from sklearn.model_selection import KFold, cross_val_predict

def evaluate_model(model_name, feature_set_name, X, y, numeric_features, categorical_features):
    if len(X) < 10:
        print(f"\n{feature_set_name}: Not enough data. Samples = {len(X)}")
        return None

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
        "Ridge": Ridge(alpha=1.0),
        "Huber": HuberRegressor(max_iter=1000)
    }

    results = []

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, regressor in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", regressor)
        ])

        y_pred = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv
        )

        mae = mean_absolute_error(y, y_pred)
        medae = median_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        p90 = np.percentile(np.abs(y - y_pred), 90)

        results.append({
            "feature_set": feature_set_name,
            "model": name,
            "samples": len(X),
            "mae": mae,
            "median_ae": medae,
            "rmse": rmse,
            "r2": r2,
            "p90_abs_err": p90
        })

    return results
    
# =========================
# 5. Run Model A, B, C, D
# =========================
all_results = []

feature_sets = {
    "Model A - True Dimension Upper Bound": features_A,
    "Model B - Deployment Estimated Dimension": features_B,
    "Model C - Camera Only": features_C,
    "Model D - Deployment Without Predicted Width": features_D
}

for set_name, features in feature_sets.items():
    X, y, num_cols, cat_cols, used_cols = prepare_features(df, features)

    print("\n" + "=" * 60)
    print(set_name)
    print("Used features:", used_cols)
    print("Samples:", len(X))
    print("=" * 60)

    results = evaluate_model(
        model_name=None,
        feature_set_name=set_name,
        X=X,
        y=y,
        numeric_features=num_cols,
        categorical_features=cat_cols
    )

    if results:
        all_results.extend(results)


# =========================
# 6. Results table
# =========================
results_df = pd.DataFrame(all_results)

if not results_df.empty:
    results_df = results_df.sort_values(by=["feature_set", "mae"])

    print("\n\nFINAL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    results_df.to_csv("weight_model_ABCD_comparison.csv", index=False)
    print("\nSaved results to: weight_model_ABCD_comparison.csv")
    # ---------------- SAVE DEPLOYMENT MODELS ----------------

    best_width_model.fit(X_w, y_w)

    joblib.dump(best_width_model, "width_model.pkl")
    joblib.dump(width_features, "width_features.pkl")

    print("Saved width model.")

    X_D, y_D, num_D, cat_D, used_D = prepare_features(df, features_D)

    weight_preprocessor_D = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_D),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_D)
        ],
        remainder="drop"
    )

    final_weight_model_D = Pipeline([
        ("preprocess", weight_preprocessor_D),
        ("model", HuberRegressor(max_iter=1000))
    ])

    final_weight_model_D.fit(X_D, y_D)

    joblib.dump(final_weight_model_D, "weight_model_D_huber.pkl")
    joblib.dump(used_D, "weight_features_D.pkl")

    print("Saved weight model.")

else:
    print("No valid results generated. Check column names and sample count.")