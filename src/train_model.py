import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_PATH = "data/timeseries_final_v3.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_MONTHS = 3
CV_FOLDS = 5
MIN_PRECISION = 0.55
MODEL_VERSION = "v1"

HORIZONS = ["distress_t1", "distress_t2", "distress_t3"]

RISK_BINS = [0, 30, 60, 100]
RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]


# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    return df


# ─────────────────────────────────────────
# ENCODE
# ─────────────────────────────────────────
def encode(df):
    df = df.copy()

    # Encode shock
    if "shock_event" in df.columns or "shock" in df.columns:
        col = "shock_event" if "shock_event" in df.columns else "shock"
        le = LabelEncoder()
        df["shock_enc"] = le.fit_transform(df[col].astype(str))
        df.drop(columns=[col], inplace=True)
        joblib.dump(le, MODEL_DIR / "label_encoder_shock.pkl")

    # ✅ NEW: encode user_type
    if "user_type" in df.columns:
        le2 = LabelEncoder()
        df["user_type_enc"] = le2.fit_transform(df["user_type"].astype(str))
        df.drop(columns=["user_type"], inplace=True)
        joblib.dump(le2, MODEL_DIR / "label_encoder_user_type.pkl")

    return df


# ─────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────
def split(df):
    cutoff = df["month"].max() - TEST_MONTHS
    return df[df["month"] <= cutoff], df[df["month"] > cutoff]


# ─────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────
def get_features(df):
    exclude = {"user_id", "month", "cal_month",
               "distress_t1", "distress_t2", "distress_t3"}
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────
# THRESHOLD
# ─────────────────────────────────────────
def find_threshold(y, proba):
    p, r, t = precision_recall_curve(y, proba)

    # choose threshold where F1 is maximized
    best_t = 0.5
    best_f1 = 0

    for prec, rec, thr in zip(p[:-1], r[:-1], t):
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                best_t = thr

    print(f"Threshold={best_t:.3f}, Best F1={best_f1:.3f}")
    return best_t


# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────
def evaluate(name, horizon, y, proba, threshold):
    pred = (proba >= threshold).astype(int)

    print(f"\n{name} | {horizon}")
    print("ROC-AUC:", roc_auc_score(y, proba))
    print("PR-AUC :", average_precision_score(y, proba))
    print("F1     :", f1_score(y, pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y, pred))

    return {
        "model": name,
        "horizon": horizon,
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
        "threshold": threshold
    }


# ─────────────────────────────────────────
# CV
# ─────────────────────────────────────────
def run_cv(X, y, scale):
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    scores = []

    for tr, val in tscv.split(X):
        m = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.04,
            scale_pos_weight=scale,
            eval_metric="aucpr"
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[val], m.predict_proba(X.iloc[val])[:, 1]))

    print("CV ROC-AUC:", np.mean(scores), "±", np.std(scores))


# ─────────────────────────────────────────
# TRAIN LOGISTIC
# ─────────────────────────────────────────
def train_logistic(Xtr, Xte, ytr, yte, horizon):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xtr_s, ytr)

    proba = model.predict_proba(Xte_s)[:, 1]
    t = find_threshold(yte, proba)

    metrics = evaluate("Logistic", horizon, yte, proba, t)

    joblib.dump(model, MODEL_DIR / f"logistic_{horizon}_{MODEL_VERSION}.pkl")
    joblib.dump(scaler, MODEL_DIR / f"scaler_{horizon}_{MODEL_VERSION}.pkl")

    return metrics


# ─────────────────────────────────────────
# TRAIN XGB
# ─────────────────────────────────────────
def train_xgb(Xtr, Xte, ytr, yte, horizon):
    scale = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    run_cv(Xtr, ytr, scale)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        scale_pos_weight=scale,
        eval_metric="aucpr"
    )

    model.fit(Xtr, ytr)

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(Xtr, ytr)

    proba = calibrated.predict_proba(Xte)[:, 1]
    t = find_threshold(yte, proba)

    metrics = evaluate("XGBoost", horizon, yte, proba, t)

    joblib.dump(calibrated, MODEL_DIR / f"xgb_{horizon}_{MODEL_VERSION}.pkl")
    joblib.dump(t, MODEL_DIR / f"threshold_{horizon}_{MODEL_VERSION}.pkl")

    return calibrated, t, metrics


# ─────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────
def explain(model, X, horizon):
    try:
        base = model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base)
        vals = explainer.shap_values(X)

        df = pd.DataFrame({
            "feature": X.columns,
            "importance": np.abs(vals).mean(axis=0)
        }).sort_values("importance", ascending=False)

        df.to_csv(MODEL_DIR / f"shap_{horizon}.csv", index=False)
        print("\nTop SHAP features:\n", df.head(10))

    except Exception as e:
        print("SHAP failed:", e)


# ─────────────────────────────────────────
# RISK
# ─────────────────────────────────────────
def risk(df, models, thresholds, features):
    latest = df.sort_values("month").groupby("user_id").last().reset_index()

    result = latest[["user_id", "month"]].copy()

    for h, m in models.items():
        p = m.predict_proba(latest[features])[:, 1]
        result[f"risk_{h}"] = p * 100
        result[f"alert_{h}"] = (p >= thresholds[h]).astype(int)

    result["overall"] = result[[f"risk_{h}" for h in models]].max(axis=1)

    result["level"] = pd.cut(
        result["overall"],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        include_lowest=True
    )

    result.to_csv(MODEL_DIR / "risk_scores.csv", index=False)
    print("\nRisk scores saved")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("🚀 START\n")

    df = encode(load_data())
    train_df, test_df = split(df)

    features = get_features(df)
    joblib.dump(features, MODEL_DIR / "features.pkl")  # NEW

    models = {}
    thresholds = []
    results = []

    for h in HORIZONS:
        print("\n==========", h, "==========")

        Xtr = train_df[features]
        ytr = train_df[h]
        Xte = test_df[features]
        yte = test_df[h]

        results.append(train_logistic(Xtr, Xte, ytr, yte, h))

        model, t, metrics = train_xgb(Xtr, Xte, ytr, yte, h)
        models[h] = model
        thresholds.append(t)
        results.append(metrics)

        explain(model, Xtr.sample(min(1000, len(Xtr))), h)

    risk(test_df, models, dict(zip(HORIZONS, thresholds)), features)

    print("\n✅ DONE")


if __name__ == "__main__":
    main()