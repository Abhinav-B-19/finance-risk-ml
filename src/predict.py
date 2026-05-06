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
from datetime import datetime
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ─────────────────────────────────────────
# REQUIRED INPUTS
# ─────────────────────────────────────────
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

    return int(
        le.transform([CATEGORICAL_FALLBACKS["shock"]])[0]
    )

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

    # Stability clipping
    d["dti"] = np.clip(d["dti"], 0, 3)
    d["expense_ratio"] = np.clip(d["expense_ratio"], 0, 3)
    d["savings_ratio"] = np.clip(d["savings_ratio"], -1, 1)

    # Encode categorical
    d["shock_enc"] = _safe_encode(
        _LE_SHOCK,
        d.get("shock", "none")
    )

    d.pop("shock", None)

    df = pd.DataFrame([d])

    # Fill missing lag features
    for col in _FEATURES:

        if col not in df.columns:
            df[col] = 0.0

    return df[_FEATURES]

# ─────────────────────────────────────────
# HYBRID RISK OVERRIDE
# ─────────────────────────────────────────
def _apply_risk_override(data, risk_score):

    income = data["income"]
    expenses = data["expenses"]
    debt = data["debt"]

    dti = debt / income if income > 0 else 0

    savings_ratio = (
        (income - expenses) / income
        if income > 0 else 0
    )

    # Hard risk rules

    if expenses > income:
        return max(risk_score, 75)

    if debt > 5 * income:
        return max(risk_score, 80)

    if dti > 1.0:
        return max(risk_score, 70)

    if savings_ratio < -0.2:
        return max(risk_score, 70)

    return risk_score

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

        current_month = datetime.utcnow().replace(day=1)

        for idx, (horizon, model) in enumerate(
            _MODELS.items(),
            start=1
        ):

            threshold = _THRESHOLDS[horizon]

            proba = float(
                model.predict_proba(X)[0][1]
            )

            risk_score = round(proba * 100, 2)

            # Hybrid override
            risk_score = _apply_risk_override(
                input_data,
                risk_score
            )

            # Clamp
            risk_score = max(
                0,
                min(risk_score, 100)
            )

            adjusted_proba = risk_score / 100

            risk_level = (
                "LOW" if risk_score < 30 else
                "MEDIUM" if risk_score < 60 else
                "HIGH"
            )

            forecast_month = (
                current_month +
                relativedelta(months=idx)
            ).strftime("%Y-%m")

            predictions[forecast_month] = {
                "probability": round(adjusted_proba, 4),
                "risk_score": risk_score,
                "risk_level": risk_level,
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

# ─────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────
def predict_batch(inputs):

    return [
        predict(i, explain=False)
        for i in inputs
    ]