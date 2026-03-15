"""
utils.py — Vi Telecom Churn Prediction
Data processing, feature engineering, model training, evaluation utilities.
"""

import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
try:
    import xgboost as xgb
    _USE_XGB = True
except ImportError:
    _USE_XGB = False
import joblib
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# STEP 1 · Column name normaliser
# ─────────────────────────────────────────

def to_snake_case(name: str) -> str:
    """Convert any column name to snake_case."""
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', name)
    name = name.replace(' ', '_').replace('-', '_')
    return name.lower().strip('_')


COLUMN_MAP = {
    "monthlychargesinr": "monthly_charges_inr",
    "contracttenuremonths": "contract_tenure_months",
    "datalimitgb": "data_limit_gb",
    "avgmonthlydatausedgb": "avg_monthly_data_used_gb",
    "avgmonthlycallsmins": "avg_monthly_calls_mins",
}

NUMERIC_COLS = [
    "age", "monthly_charges_inr", "contract_tenure_months",
    "data_limit_gb", "avg_monthly_data_used_gb", "avg_monthly_calls_mins",
    "complaints_last_6months", "customer_satisfaction_score",
    "customer_service_calls", "late_payment_count",
]

CATEGORICAL_COLS = [
    "gender", "state", "location_type", "plan_type", "payment_method",
]

BINARY_COLS = [
    "roaming_usage_flag", "international_calls_flag",
    "competitor_offer_received", "number_portability_enquiry",
]

# ─────────────────────────────────────────
# STEP 2 · Load & basic checks
# ─────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalise column names
    df.columns = [to_snake_case(c) for c in df.columns]
    df.rename(columns=COLUMN_MAP, inplace=True)
    # coerce numerics
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────
# STEP 3 · Missing value imputation
# ─────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    summary = []

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        flag_col = f"{col}_was_imputed"
        df[flag_col] = df[col].isnull().astype(int)
        skew = df[col].skew()
        if abs(skew) > 0.5:
            fill_val = df[col].median()
            method = "median"
        else:
            fill_val = df[col].mean()
            method = "mean"
        df[col] = df[col].fillna(fill_val)
        summary.append({"column": col, "missing": n_missing, "method": method, "value": round(fill_val, 4)})

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        flag_col = f"{col}_was_imputed"
        df[flag_col] = df[col].isnull().astype(int)
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        summary.append({"column": col, "missing": n_missing, "method": "mode", "value": mode_val})

    # rare states → Others
    if "state" in df.columns:
        freq = df["state"].value_counts(normalize=True)
        rare = freq[freq < 0.01].index.tolist()
        df["state"] = df["state"].apply(lambda x: "Others" if x in rare else x)
        if rare:
            summary.append({"column": "state", "missing": 0, "method": f"Grouped {len(rare)} rare states → 'Others'", "value": ""})

    print(pd.DataFrame(summary).to_string(index=False) if summary else "No missing values found.")
    return df


# ─────────────────────────────────────────
# STEP 4 · Encode categorical variables
# ─────────────────────────────────────────

def encode_features(df: pd.DataFrame, save_path: str = "data/encoding_mappings.csv") -> pd.DataFrame:
    df = df.copy()
    mappings = []

    # One-hot encode
    ohe_cols = [c for c in ["gender", "location_type", "plan_type", "payment_method"] if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)
    for col in ohe_cols:
        mappings.append({"column": col, "type": "one_hot"})

    # Ordinal encode satisfaction score
    if "customer_satisfaction_score" in df.columns:
        df["customer_satisfaction_score"] = df["customer_satisfaction_score"].astype(int)
        mappings.append({"column": "customer_satisfaction_score", "type": "ordinal_1_to_5"})

    # Frequency encode state
    if "state" in df.columns:
        freq_map = df["state"].value_counts(normalize=True).to_dict()
        df["state_freq_enc"] = df["state"].map(freq_map)
        df.drop(columns=["state"], inplace=True)
        mappings.append({"column": "state", "type": "frequency_encoding"})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(mappings).to_csv(save_path, index=False)
    return df


# ─────────────────────────────────────────
# STEP 5 · Feature engineering
# ─────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # usage ratio
    if "avg_monthly_data_used_gb" in df.columns and "data_limit_gb" in df.columns:
        df["usage_ratio"] = df["avg_monthly_data_used_gb"] / df["data_limit_gb"].replace(0, np.nan)
        df["usage_ratio"] = df["usage_ratio"].fillna(0).clip(0, 5)

    # complaint rate
    if "complaints_last_6months" in df.columns and "contract_tenure_months" in df.columns:
        tenure_periods = (df["contract_tenure_months"] / 6).replace(0, np.nan)
        df["complaint_rate"] = df["complaints_last_6months"] / tenure_periods
        df["complaint_rate"] = df["complaint_rate"].fillna(0)

    # tenure bucket
    if "contract_tenure_months" in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["contract_tenure_months"],
            bins=[-1, 3, 12, 36, 1000],
            labels=[0, 1, 2, 3]
        ).astype(int)

    # payment risk score
    if "late_payment_count" in df.columns:
        risky = 0
        if "payment_method_Cash" in df.columns:
            risky = df["payment_method_Cash"].fillna(0)
        df["payment_risk_score"] = df["late_payment_count"] + risky

    return df


# ─────────────────────────────────────────
# STEP 6 · Split
# ─────────────────────────────────────────

def split_data(df: pd.DataFrame, target: str = "churned", test_size: float = 0.2):
    drop_cols = [c for c in df.columns if c.endswith("_was_imputed")]
    X = df.drop(columns=[target] + drop_cols, errors='ignore')
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


# ─────────────────────────────────────────
# STEP 7 · Train models
# ─────────────────────────────────────────

if _USE_XGB:
    _xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, verbosity=0
    )
else:
    _xgb_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )

MODELS = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1),
    "XGBoost": _xgb_model,
}


def train_all_models(X_train, y_train):
    trained = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  ✓ {name} trained")
    return trained


def save_models(trained: dict, folder: str = "models"):
    os.makedirs(folder, exist_ok=True)
    for name, model in trained.items():
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(folder, fname))
    print(f"  Models saved to '{folder}/'")


def load_models(folder: str = "models") -> dict:
    trained = {}
    name_map = {"decision_tree": "Decision Tree", "random_forest": "Random Forest", "xgboost": "XGBoost"}
    for fname in os.listdir(folder):
        if fname.endswith(".joblib"):
            key = fname.replace(".joblib", "")
            nice = name_map.get(key, key)
            trained[nice] = joblib.load(os.path.join(folder, fname))
    return trained


# ─────────────────────────────────────────
# STEP 8 · Evaluate
# ─────────────────────────────────────────

def evaluate_models(trained: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Train Acc": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "Test Acc": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC AUC": round(roc_auc_score(y_test, y_prob), 4),
            "PR AUC": round(average_precision_score(y_test, y_prob), 4),
        })
    return pd.DataFrame(rows)


def get_roc_data(trained: dict, X_test, y_test) -> dict:
    roc = {}
    for name, model in trained.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        roc[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
    return roc


def get_pr_data(trained: dict, X_test, y_test) -> dict:
    pr = {}
    for name, model in trained.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        pr[name] = {"precision": prec, "recall": rec, "ap": ap}
    return pr


def get_confusion_matrices(trained: dict, X_test, y_test) -> dict:
    cms = {}
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        cms[name] = confusion_matrix(y_test, y_pred)
    return cms


def cross_val_best(trained: dict, X_train, y_train, best_name: str = "XGBoost") -> dict:
    model = trained[best_name]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(model, X_train, y_train, cv=cv,
                              scoring=['accuracy', 'f1', 'roc_auc'], return_train_score=False)
    return {k: (v.mean(), v.std()) for k, v in results.items() if k.startswith("test_")}


# ─────────────────────────────────────────
# STEP 9 · Feature importance
# ─────────────────────────────────────────

def get_feature_importance(trained: dict, feature_names: list) -> dict:
    importances = {}
    for name, model in trained.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            importances[name] = imp
    return importances


def get_permutation_importance(trained: dict, X_test, y_test, feature_names: list, best_name: str = "XGBoost"):
    from sklearn.inspection import permutation_importance
    model = trained[best_name]
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc')
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False)
    return imp


# ─────────────────────────────────────────
# PIPELINE RUNNER (for pre-training)
# ─────────────────────────────────────────

def run_pipeline(raw_csv: str = "data/VI_Customer_Churn.csv",
                 cleaned_csv: str = "data/vi_customer_churn_cleaned.csv",
                 models_folder: str = "models") -> dict:
    print("\n── Step 2: Load & clean ──")
    df = load_and_clean(raw_csv)

    print("\n── Step 3: Handle missing values ──")
    df = handle_missing(df)

    print("\n── Step 4: Encode ──")
    df_enc = encode_features(df, save_path="data/encoding_mappings.csv")

    print("\n── Step 5: Feature engineering ──")
    df_enc = engineer_features(df_enc)

    # save cleaned
    df.to_csv(cleaned_csv, index=False)
    print(f"\n  Cleaned dataset saved → {cleaned_csv}")

    print("\n── Step 6: Split ──")
    X_train, X_test, y_train, y_test = split_data(df_enc)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n── Step 7: Train models ──")
    trained = train_all_models(X_train, y_train)
    save_models(trained, models_folder)

    print("\n── Step 8: Evaluate ──")
    metrics_df = evaluate_models(trained, X_train, X_test, y_train, y_test)
    print(metrics_df.to_string(index=False))

    print("\n── Step 9: Feature importance ──")
    importances = get_feature_importance(trained, list(X_train.columns))
    for name, imp in importances.items():
        print(f"\n  {name} — Top 5:")
        print(imp.head(5).to_string())

    return {
        "df": df,
        "df_enc": df_enc,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "trained": trained,
        "metrics": metrics_df,
        "importances": importances,
        "feature_names": list(X_train.columns),
    }


if __name__ == "__main__":
    run_pipeline()
