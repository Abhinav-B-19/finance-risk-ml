"""
==============================================================================
 AI-BASED EARLY WARNING SYSTEM — Prediction Module (REAL SYSTEM VERSION)
==============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")

RISK_BINS = [0, 30, 60, 100]
RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]

# 🔥 Only raw fields required now
REQUIRED_RAW_FIELDS = {
    "income": (int, float),
    "expenses": (int, float),
    "debt": (int, float),
    "shock": str,
}

CATEGORICAL_FALLBACKS = {
    "shock": "none",
    "user_type": "average",
}

# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
def _load_artifacts():
    features = joblib.load(MODEL_DIR / "features.pkl")
    model = joblib.load(MODEL_DIR / "xgb_distress_t2_v1.pkl")
    threshold = joblib.load(MODEL_DIR / "threshold_distress_t2_v1.pkl")
    le_shock = joblib.load(MODEL_DIR / "label_encoder_shock.pkl")

    le_user = None
    if (MODEL_DIR / "label_encoder_user_type.pkl").exists():
        le_user = joblib.load(MODEL_DIR / "label_encoder_user_type.pkl")

    return features, model, threshold, le_shock, le_user


_FEATURES, _MODEL, _THRESHOLD, _LE_SHOCK, _LE_USER = _load_artifacts()

# ─────────────────────────────────────────
# SHAP (load once)
# ─────────────────────────────────────────
try:
    import shap
    _BASE_MODEL = _MODEL.calibrated_classifiers_[0].estimator
    _EXPLAINER = shap.TreeExplainer(_BASE_MODEL)
except:
    _EXPLAINER = None


# ─────────────────────────────────────────
# SAFE ENCODE
# ─────────────────────────────────────────
def _safe_encode(le, value, field):
    if value in le.classes_:
        return int(le.transform([value])[0])

    fallback = CATEGORICAL_FALLBACKS.get(field, le.classes_[0])
    return int(le.transform([fallback])[0])


# ─────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────
def _validate(data):
    errors = []

    for field, types in REQUIRED_RAW_FIELDS.items():
        if field not in data:
            errors.append(f"Missing: {field}")
        elif not isinstance(data[field], types):
            errors.append(f"Wrong type: {field}")

    if "income" in data and data["income"] <= 0:
        errors.append("income must be > 0")

    if "expenses" in data and data["expenses"] < 0:
        errors.append("expenses must be >= 0")

    if "debt" in data and data["debt"] < 0:
        errors.append("debt must be >= 0")

    if errors:
        raise ValueError("\n".join(errors))


# ─────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────
def _preprocess(data):
    d = dict(data)

    income = float(d["income"])
    expenses = float(d["expenses"])
    debt = float(d["debt"])

    # derived features
    d["dti"] = debt / income if income > 0 else 0
    d["expense_ratio"] = expenses / income if income > 0 else 1
    d["savings_ratio"] = (income - expenses) / income if income > 0 else 0
    d["savings"] = income - expenses

    # clip
    d["dti"] = np.clip(d["dti"], 0, 3)
    d["expense_ratio"] = np.clip(d["expense_ratio"], 0, 3)
    d["savings_ratio"] = np.clip(d["savings_ratio"], -1, 1)

    # encode
    d["shock_enc"] = _safe_encode(_LE_SHOCK, d.get("shock", "none"), "shock")
    d.pop("shock", None)

    df = pd.DataFrame([d])

    # fill missing (lags etc.)
    for col in _FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df[_FEATURES]


# ─────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────
def _explain(X):
    if _EXPLAINER is None:
        return []

    shap_vals = _EXPLAINER.shap_values(X)[0]

    drivers = sorted(
        [
            {
                "feature": f,
                "value": float(X[f].iloc[0]),
                "shap": float(s),
                "direction": "risk↑" if s > 0 else "risk↓",
            }
            for f, s in zip(X.columns, shap_vals)
        ],
        key=lambda x: abs(x["shap"]),
        reverse=True,
    )

    return drivers[:5]


# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────
def predict(input_data, explain=False):

    try:
        _validate(input_data)
    except Exception as e:
        return {"status": "error", "errors": [str(e)]}

    try:
        X = _preprocess(input_data)

        proba = float(_MODEL.predict_proba(X)[0][1])
        risk_score = round(proba * 100, 2)

        result = {
            "status": "success",
            "probability": round(proba, 4),
            "risk_score": risk_score,
            "risk_level": (
                "LOW" if risk_score < 30 else
                "MEDIUM" if risk_score < 60 else
                "HIGH"
            ),
            "alert": int(proba >= _THRESHOLD),
        }

        if explain:
            result["drivers"] = _explain(X)

        return result

    except Exception as e:
        return {"status": "error", "errors": [str(e)]}


def predict_batch(inputs):
    return [predict(i, explain=False) for i in inputs]

    
# """
# ==============================================================================
#  AI-BASED EARLY WARNING SYSTEM — Prediction Module (FINAL VERSION)
# ==============================================================================
# """

# import pandas as pd
# import numpy as np
# import joblib
# import warnings
# from pathlib import Path

# warnings.filterwarnings("ignore")

# MODEL_DIR = Path("models")

# RISK_BINS   = [0, 30, 60, 100]
# RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]

# REQUIRED_RAW_FIELDS = {
#     "income":              (int, float),
#     "expenses":            (int, float),
#     "debt":                (int, float),
#     "shock":               str,
#     "dti_lag1":            (int, float),
#     "dti_lag2":            (int, float),
#     "savings_ratio_lag1":  (int, float),
#     "savings_ratio_lag2":  (int, float),
#     "debt_lag1":           (int, float),
#     "debt_lag2":           (int, float),
# }

# CATEGORICAL_FALLBACKS = {
#     "shock": "none",
#     "user_type": "average",
# }

# # ─────────────────────────────────────────
# # LOAD ARTIFACTS
# # ─────────────────────────────────────────
# def _load_artifacts():
#     features  = joblib.load(MODEL_DIR / "features.pkl")
#     model     = joblib.load(MODEL_DIR / "xgb_distress_t2_v1.pkl")
#     threshold = joblib.load(MODEL_DIR / "threshold_distress_t2_v1.pkl")
#     le_shock  = joblib.load(MODEL_DIR / "label_encoder_shock.pkl")

#     le_user = None
#     if (MODEL_DIR / "label_encoder_user_type.pkl").exists():
#         le_user = joblib.load(MODEL_DIR / "label_encoder_user_type.pkl")

#     # Feature validation
#     try:
#         base = model.calibrated_classifiers_[0].estimator
#         model_features = list(base.get_booster().feature_names)
#         if set(model_features) != set(features):
#             raise ValueError("Feature mismatch between model and features.pkl")
#     except:
#         pass

#     return features, model, threshold, le_shock, le_user


# _FEATURES, _MODEL, _THRESHOLD, _LE_SHOCK, _LE_USER = _load_artifacts()

# # ─────────────────────────────────────────
# # SHAP (optimized load once)
# # ─────────────────────────────────────────
# try:
#     import shap
#     _BASE_MODEL = _MODEL.calibrated_classifiers_[0].estimator
#     _EXPLAINER = shap.TreeExplainer(_BASE_MODEL)
# except:
#     _EXPLAINER = None

# # ─────────────────────────────────────────
# # SAFE ENCODE
# # ─────────────────────────────────────────
# def _safe_encode(le, value, field):
#     if value in le.classes_:
#         return int(le.transform([value])[0])

#     fallback = CATEGORICAL_FALLBACKS.get(field, le.classes_[0])
#     if fallback not in le.classes_:
#         fallback = le.classes_[0]

#     warnings.warn(f"{field}='{value}' unknown → fallback='{fallback}'")
#     return int(le.transform([fallback])[0])

# # ─────────────────────────────────────────
# # VALIDATION
# # ─────────────────────────────────────────
# def _validate(data):
#     errors = []

#     for field, types in REQUIRED_RAW_FIELDS.items():
#         if field not in data:
#             errors.append(f"Missing: {field}")
#         elif not isinstance(data[field], types):
#             errors.append(f"Wrong type: {field}")

#     # 🔥 strict numeric validation
#     if "income" in data:
#         if data["income"] <= 0:
#             errors.append("income must be > 0")

#     if "expenses" in data:
#         if data["expenses"] < 0:
#             errors.append("expenses must be >= 0")

#     if "debt" in data:
#         if data["debt"] < 0:
#             errors.append("debt must be >= 0")

#     if errors:
#         raise ValueError("Input validation failed:\n" + "\n".join(errors))

# # ─────────────────────────────────────────
# # PREPROCESS
# # ─────────────────────────────────────────
# def _preprocess(data):
#     d = dict(data)

#     income = float(d["income"])
#     expenses = float(d["expenses"])
#     debt = float(d["debt"])

#     d["dti"] = debt / income if income > 0 else 0
#     d["expense_ratio"] = expenses / income if income > 0 else 1
#     d["savings_ratio"] = (income - expenses) / income if income > 0 else 0
#     d["savings"] = income - expenses

#     # stability clipping
#     d["dti"] = np.clip(d["dti"], 0, 3)
#     d["expense_ratio"] = np.clip(d["expense_ratio"], 0, 3)
#     d["savings_ratio"] = np.clip(d["savings_ratio"], -1, 1)

#     # encode
#     d["shock_enc"] = _safe_encode(_LE_SHOCK, d["shock"], "shock")
#     d.pop("shock", None)

#     if _LE_USER and "user_type" in d:
#         d["user_type_enc"] = _safe_encode(_LE_USER, d["user_type"], "user_type")
#         d.pop("user_type", None)

#     df = pd.DataFrame([d])

#     # fill missing
#     missing = [c for c in _FEATURES if c not in df.columns]
#     if missing:
#         warnings.warn(f"Missing features filled with 0: {missing}")

#     for c in _FEATURES:
#         if c not in df.columns:
#             df[c] = 0

#     return df[_FEATURES]

# # ─────────────────────────────────────────
# # SHAP EXPLAIN
# # ─────────────────────────────────────────
# def _explain(X):
#     if _EXPLAINER is None:
#         return [{"error": "SHAP not available"}]

#     shap_vals = _EXPLAINER.shap_values(X)[0]

#     drivers = sorted(
#         [
#             {
#                 "feature": f,
#                 "value": float(X[f].iloc[0]),
#                 "shap": float(s),
#                 "direction": "risk↑" if s > 0 else "risk↓",
#             }
#             for f, s in zip(X.columns, shap_vals)
#         ],
#         key=lambda x: abs(x["shap"]),
#         reverse=True,
#     )

#     return drivers[:5]

# # ─────────────────────────────────────────
# # PREDICT
# # ─────────────────────────────────────────
# def predict(input_data, explain=True):

#     try:
#         _validate(input_data)
#     except ValueError as e:
#         return {
#             "status": "error",
#             "errors": str(e).split("\n")[1:]  # clean list of errors
#         }

#     try:
#         X = _preprocess(input_data)

#         proba = float(_MODEL.predict_proba(X)[0][1])
#         risk_score = round(proba * 100, 2)

#         result = {
#             "status": "success",
#             "probability": round(proba, 4),
#             "risk_score": risk_score,
#             "risk_level": (
#                 "LOW" if risk_score < 30 else
#                 "MEDIUM" if risk_score < 60 else
#                 "HIGH"
#             ),
#             "alert": int(proba >= _THRESHOLD),
#             "threshold": round(float(_THRESHOLD), 3),
#         }

#         if explain:
#             result["drivers"] = _explain(X)

#         return result

#     except Exception as e:
#         return {
#             "status": "error",
#             "errors": [f"Prediction failed: {str(e)}"]
#         }

# # ─────────────────────────────────────────
# # BATCH
# # ─────────────────────────────────────────
# def predict_batch(inputs):
#     return [predict(i, explain=False) for i in inputs]

# # ─────────────────────────────────────────
# # TEST
# # ─────────────────────────────────────────
# if __name__ == "__main__":

#     sample = {
#          "income": 100000,
#  "expenses": 20000,
#  "debt": 20000,
#         "shock": "medical",
#         "dti_lag1": 0.35,
#         "dti_lag2": 0.30,
#         "savings_ratio_lag1": 0.08,
#         "savings_ratio_lag2": 0.1,
#         "debt_lag1": 180000,
#         "debt_lag2": 160000,
#     }

#     print("\nPrediction:")
#     print(predict(sample, explain=False))