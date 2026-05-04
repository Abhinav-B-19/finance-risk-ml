import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

N_MONTHS = 12
DISTRESS_HORIZONS = [1, 2, 3]

SEASONAL_SPEND = {
    1: 1.02, 2: 1.00, 3: 1.01, 4: 1.03, 5: 1.04, 6: 1.02,
    7: 1.01, 8: 1.02, 9: 1.01, 10: 1.08, 11: 1.12, 12: 1.06
}

SHOCK_EVENTS = {
    "job_loss": {"income_mult": 0.3, "expense_mult": 1.10, "debt_spike": 0, "prob": 0.02},
    "income_cut": {"income_mult": 0.75, "expense_mult": 1.00, "debt_spike": 0, "prob": 0.04},
    "medical": {"income_mult": 1.0, "expense_mult": 1.25, "debt_spike": 25000, "prob": 0.05},
    "festival": {"income_mult": 1.0, "expense_mult": 1.20, "debt_spike": 8000, "prob": 0.06},
    "new_loan": {"income_mult": 1.0, "expense_mult": 1.05, "debt_spike": 40000, "prob": 0.04},
    "bonus": {"income_mult": 1.25, "expense_mult": 1.00, "debt_spike": -5000, "prob": 0.08},
    "none": {"income_mult": 1.0, "expense_mult": 1.00, "debt_spike": 0, "prob": 0.71},
}

EXPENSE_COLUMNS = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries",
    "Transport", "Eating_Out", "Entertainment", "Utilities",
    "Healthcare", "Education", "Miscellaneous"
]

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def sample_event():
    events = list(SHOCK_EVENTS.keys())
    probs = np.array([SHOCK_EVENTS[e]["prob"] for e in events])
    probs /= probs.sum()
    e = rng.choice(events, p=probs)
    return e, SHOCK_EVENTS[e]


def simulate_debt(debt, emi, shock):
    rate = 0.01
    interest = debt * rate
    repayment = min(emi, debt)
    debt = max(debt + interest - repayment + shock, 0)
    return debt * rng.normal(1.0, 0.015)


def compute_distress(dti, savings_ratio, expense_ratio, momentum):
    if dti > 0.5 or savings_ratio < 0.05:
        return 1
    if dti > 0.4 or expense_ratio > 1.2 or momentum > 0.1:
        return int(rng.random() < 0.7)
    if dti > 0.3:
        return int(rng.random() < 0.4)
    return 0


# ─────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────
def generate_timeseries(df):
    all_data = []

    for uid, row in df.iterrows():

        base_income = float(row["Income"])
        base_emi = float(row.get("Loan_Repayment", 0))
        base_exp = sum(float(row[c]) for c in EXPENSE_COLUMNS if c in row)

        user_type = rng.choice(["disciplined", "average", "risky"])
        drift = {"disciplined": 0.995, "average": 1.0, "risky": 1.01}[user_type]

        debt = max(base_emi * rng.uniform(8, 15), 5000)

        records = []

        for m in range(N_MONTHS):
            cal_month = (m % 12) + 1

            event, shock = sample_event()

            income = base_income * rng.normal(1.0, 0.03) * shock["income_mult"]

            inflation = rng.normal(1.005, 0.002)

            expenses = base_exp * SEASONAL_SPEND[cal_month] * inflation
            expenses *= drift * shock["expense_mult"]

            expenses *= rng.uniform(0.95, 1.5)
            expenses = max(expenses, base_exp * 0.6)

            debt = simulate_debt(debt, base_emi, shock["debt_spike"])

            savings = income - expenses
            savings_ratio = savings / income if income > 0 else 0
            expense_ratio = expenses / income if income > 0 else 1

            # Improved DTI
            dti = (base_emi + 0.01 * debt) / income if income > 0 else 0

            records.append({
                "user_id": uid,
                "month": m + 1,
                "income": income,
                "expenses": expenses,
                "debt": debt,
                "dti": dti,
                "savings_ratio": savings_ratio,
                "expense_ratio": expense_ratio,
                "shock": event,
                "user_type": user_type
            })

        df_user = pd.DataFrame(records)

        # Rolling + momentum
        df_user["dti_roll"] = df_user["dti"].rolling(3, 1).mean()
        df_user["momentum"] = df_user["dti"] - df_user["dti_roll"]

        # Labels
        for h in DISTRESS_HORIZONS:
            labels = []
            for i in range(len(df_user)):
                f = df_user.iloc[i + h] if i + h < len(df_user) else df_user.iloc[i]

                label = compute_distress(
                    f["dti"],
                    f["savings_ratio"],
                    f["expense_ratio"],
                    f["momentum"]
                )
                labels.append(label)

            df_user[f"distress_t{h}"] = labels

        all_data.append(df_user)

    return pd.concat(all_data, ignore_index=True)


# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def add_features(df):
    df = df.sort_values(["user_id", "month"])
    g = df.groupby("user_id")

    for col in ["dti", "savings_ratio", "debt", "income"]:
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_lag2"] = g[col].shift(2)

    df["dti_roll3"] = g["dti"].transform(lambda x: x.rolling(3, 1).mean())
    df["income_vol"] = g["income"].transform(lambda x: x.rolling(3, 1).std().fillna(0))
    df["debt_change"] = df["debt"] - df["debt_lag1"]

    # Financial Stress Index (NEW)
    df["financial_stress"] = (
        0.5 * df["dti"] +
        0.3 * df["expense_ratio"] -
        0.2 * df["savings_ratio"]
    )

    return df


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
def main():
    df = pd.read_csv("data/finance_data.csv")

    ts = generate_timeseries(df)
    ts = add_features(ts)

    # Clip extreme values (NEW)
    ts["dti"] = ts["dti"].clip(0, 2)
    ts["savings_ratio"] = ts["savings_ratio"].clip(-1, 1)
    ts["expense_ratio"] = ts["expense_ratio"].clip(0, 3)

    # Drop NaNs (NEW)
    ts = ts.dropna()

    ts.to_csv("data/timeseries_final_v3.csv", index=False)

    print("✅ FINAL DATASET READY (v3)")
    print(ts.head())


if __name__ == "__main__":
    main()