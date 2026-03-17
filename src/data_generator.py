import numpy as np
import pandas as pd
from src.config import RANDOM_SEED, N_BORROWERS


def generate_portfolio():
    np.random.seed(RANDOM_SEED)

    credit_score = np.random.normal(680, 60, N_BORROWERS).clip(300, 850)
    income = np.random.normal(75000, 25000, N_BORROWERS).clip(20000, 200000)
    dti = np.random.normal(0.30, 0.12, N_BORROWERS).clip(0.05, 0.80)
    loan_amount = np.random.normal(25000, 12000, N_BORROWERS).clip(3000, 100000)
    interest_rate = np.random.normal(0.08, 0.03, N_BORROWERS).clip(0.02, 0.25)

    industries = np.random.choice(
        ["Technology", "Retail", "Healthcare", "Energy", "Manufacturing", "Finance"],
        size=N_BORROWERS
    )

    df = pd.DataFrame({
        "credit_score": credit_score,
        "income": income,
        "dti": dti,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "industry": industries
    })

    # Create a synthetic default probability
    score_term = (700 - df["credit_score"]) / 100
    dti_term = df["dti"] * 2.5
    rate_term = df["interest_rate"] * 3.0
    loan_term = df["loan_amount"] / 100000

    industry_risk = df["industry"].map({
        "Technology": 0.10,
        "Retail": 0.35,
        "Healthcare": 0.15,
        "Energy": 0.30,
        "Manufacturing": 0.25,
        "Finance": 0.12
    })

    risk_score = -2.8 + score_term + dti_term + rate_term + loan_term + industry_risk
    default_probability = 1 / (1 + np.exp(-risk_score))

    df["default_probability_true"] = default_probability
    df["default"] = np.random.binomial(1, default_probability)

    return df