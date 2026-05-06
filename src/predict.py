"""
==============================================================================
 AI-BASED EARLY WARNING SYSTEM — FINAL HYBRID VERSION
==============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")

REQUIRED_RAW_FIELDS = {
    "income": (int, float),
    "expenses": (int, float),
    "debt": (int, float),
    "shock": str,
}

CATEGORICAL_FALLBACKS = {
    "shock": "none",
}

# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
def _load_artifacts():
    features = joblib.load(MODEL_DIR / "features.pkl")

    models = {
        "month_1": joblib.load(MODEL_DIR / "xgb_distress_t1_v1.pkl"),
        "month_2": joblib.load(MODEL_DIR / "xgb_distress_t2_v1.pkl"),
        "month_3": joblib.load(MODEL_DIR / "xgb_distress_t3_v1.pkl"),
    }

    thresholds = {
        "month_1": joblib.load(MODEL_DIR / "threshold_distress_t1_v1.pkl"),
        "month_2": joblib.load(MODEL_DIR / "threshold_distress_t2_v1.pkl"),
        "month_3": joblib.load(MODEL_DIR / "threshold_distress_t3_v1.pkl"),
    }

    le_shock = joblib.load(MODEL_DIR / "label_encoder_shock.pkl")

    return features, models, thresholds, le_shock


_FEATURES, _MODELS, _THRESHOLDS, _LE_SHOCK = _load_artifacts()

# ─────────────────────────────────────────
# SAFE ENCODE
# ─────────────────────────────────────────
def _safe_encode(le, value):
    if value in le.classes_:
        return int(le.transform([value])[0])
    return int(le.transform([CATEGORICAL_FALLBACKS["shock"]])[0])


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

    if data.get("income", 0) <= 0:
        errors.append("income must be > 0")

    if data.get("expenses", 0) < 0:
        errors.append("expenses must be >= 0")

    if data.get("debt", 0) < 0:
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

    # Derived features
    d["dti"] = debt / income
    d["expense_ratio"] = expenses / income
    d["savings_ratio"] = (income - expenses) / income
    d["savings"] = income - expenses

    # Clip values (important for model stability)
    d["dti"] = np.clip(d["dti"], 0, 3)
    d["expense_ratio"] = np.clip(d["expense_ratio"], 0, 3)
    d["savings_ratio"] = np.clip(d["savings_ratio"], -1, 1)

    # Encode
    d["shock_enc"] = _safe_encode(_LE_SHOCK, d.get("shock", "none"))
    d.pop("shock", None)

    df = pd.DataFrame([d])

    # Fill missing features (lags)
    for col in _FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df[_FEATURES]


# ─────────────────────────────────────────
# 🚨 HYBRID RISK OVERRIDE (CRITICAL FIX)
# ─────────────────────────────────────────
def _apply_risk_override(data, risk_score):

    income = data["income"]
    expenses = data["expenses"]
    debt = data["debt"]

    dti = debt / income if income > 0 else 0
    savings_ratio = (income - expenses) / income if income > 0 else 0

    # 🔥 HARD RISK CONDITIONS
    if expenses > income:
        return max(risk_score, 75)

    if debt > 5 * income:
        return max(risk_score, 80)

    if dti > 1.0:
        return max(risk_score, 70)

    if savings_ratio < -0.2:
        return max(risk_score, 70)

    return risk_score

def _build_prediction_result(proba, threshold):
    risk_score = round(proba * 100, 2)

    risk_score = max(0, min(risk_score, 100))

    risk_level = (
        "LOW" if risk_score < 30 else
        "MEDIUM" if risk_score < 60 else
        "HIGH"
    )

    return {
        "probability": round(proba, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "alert": int(proba >= threshold),
    }

# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────
def predict(input_data, explain=False):

    try:
        _validate(input_data)
    except Exception as e:
        return {
            "status": "error",
            "errors": [str(e)]
        }

    try:
        X = _preprocess(input_data)

        predictions = {}

        for horizon, model in _MODELS.items():

            threshold = _THRESHOLDS[horizon]

            proba = float(model.predict_proba(X)[0][1])

            risk_score = round(proba * 100, 2)

            # Hybrid override
            risk_score = _apply_risk_override(
                input_data,
                risk_score
            )

            adjusted_proba = risk_score / 100

            predictions[horizon] = {
                "probability": round(adjusted_proba, 4),
                "risk_score": risk_score,
                "risk_level": (
                    "LOW" if risk_score < 30 else
                    "MEDIUM" if risk_score < 60 else
                    "HIGH"
                ),
                "alert": int(risk_score >= 60)
            }

        return {
            "status": "success",
            "predictions": predictions
        }

    except Exception as e:
        return {
            "status": "error",
            "errors": [str(e)]
        }


def predict_batch(inputs):
    return [predict(i, explain=False) for i in inputs]