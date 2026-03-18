import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import STRESS_SCENARIOS

st.set_page_config(page_title="RiskLab Dashboard", layout="wide")

st.title("RiskLab: Credit Risk & Portfolio Stress Testing")
st.write(
    "This dashboard shows borrower risk, model performance, expected loss, "
    "and interactive stress scenario results."
)

portfolio = pd.read_csv("results/portfolio_with_pd.csv")
model_metrics = pd.read_csv("results/model_metrics.csv")

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    auc_value = model_metrics.loc[model_metrics["Metric"] == "AUC", "Value"].iloc[0]
    st.metric("AUC Score", f"{auc_value:.3f}")

with col2:
    default_rate = model_metrics.loc[
        model_metrics["Metric"] == "Default Rate", "Value"
    ].iloc[0]
    st.metric("Portfolio Default Rate", f"{default_rate:.2%}")

st.subheader("Portfolio Overview")

col1, col2, col3 = st.columns(3)

with col1:
    total_exposure = portfolio["loan_amount"].sum()
    st.metric("Total Exposure", f"${total_exposure:,.0f}")

with col2:
    avg_pd = portfolio["predicted_pd"].mean()
    st.metric("Average Predicted PD", f"{avg_pd:.2%}")

with col3:
    total_expected_loss = portfolio["expected_loss"].sum()
    st.metric("Base Expected Loss", f"${total_expected_loss:,.0f}")

st.subheader("Interactive Stress Scenario")

scenario_name = st.selectbox(
    "Choose a stress scenario",
    list(STRESS_SCENARIOS.keys()),
    index=2
)

default_pd_multiplier = STRESS_SCENARIOS[scenario_name]["pd_multiplier"]
default_lgd_multiplier = STRESS_SCENARIOS[scenario_name]["lgd_multiplier"]

pd_multiplier = st.slider(
    "PD Multiplier",
    min_value=0.5,
    max_value=3.0,
    value=float(default_pd_multiplier),
    step=0.05
)

lgd_multiplier = st.slider(
    "LGD Multiplier",
    min_value=0.5,
    max_value=2.0,
    value=float(default_lgd_multiplier),
    step=0.05
)

interactive_df = portfolio.copy()
interactive_df["lgd"] = 0.45
interactive_df["stressed_pd"] = (
    interactive_df["predicted_pd"] * pd_multiplier
).clip(0, 1)
interactive_df["stressed_lgd"] = (
    interactive_df["lgd"] * lgd_multiplier
).clip(0, 1)
interactive_df["stressed_expected_loss"] = (
    interactive_df["stressed_pd"]
    * interactive_df["stressed_lgd"]
    * interactive_df["loan_amount"]
)

base_loss = portfolio["expected_loss"].sum()
stressed_loss = interactive_df["stressed_expected_loss"].sum()
loss_change = stressed_loss - base_loss

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Scenario", scenario_name)

with col2:
    st.metric(
        "Stressed Expected Loss",
        f"${stressed_loss:,.0f}",
        f"${loss_change:,.0f}"
    )

with col3:
    stress_ratio = stressed_loss / base_loss if base_loss != 0 else 0
    st.metric("Stress / Base Loss", f"{stress_ratio:.2f}x")

st.subheader("Expected Loss Comparison")

comparison_df = pd.DataFrame({
    "Scenario": ["Base", "Interactive Stress"],
    "Expected Loss": [base_loss, stressed_loss]
})
st.bar_chart(comparison_df.set_index("Scenario"))

st.subheader("Predicted PD Distribution")
pd_bins = portfolio["predicted_pd"].round(2).value_counts().sort_index()
st.bar_chart(pd_bins)

st.subheader("Top 20 Riskiest Borrowers")
top_risky = interactive_df.sort_values("stressed_pd", ascending=False).head(20)
st.dataframe(
    top_risky[
        [
            "credit_score",
            "income",
            "dti",
            "loan_amount",
            "interest_rate",
            "predicted_pd",
            "stressed_pd",
            "expected_loss",
            "stressed_expected_loss",
        ]
    ],
    use_container_width=True
)

st.subheader("Risk by Industry")
industry_cols = [col for col in interactive_df.columns if col.startswith("industry_")]

industry_risk = []
for col in industry_cols:
    industry_name = col.replace("industry_", "")
    subset = interactive_df[interactive_df[col] == 1]
    if len(subset) > 0:
        industry_risk.append({
            "Industry": industry_name,
            "Average PD": subset["predicted_pd"].mean(),
            "Average Stressed PD": subset["stressed_pd"].mean(),
            "Total Expected Loss": subset["expected_loss"].sum(),
            "Total Stressed Expected Loss": subset["stressed_expected_loss"].sum(),
        })

industry_risk_df = pd.DataFrame(industry_risk)

if not industry_risk_df.empty:
    st.dataframe(industry_risk_df, use_container_width=True)
    st.bar_chart(
        industry_risk_df.set_index("Industry")["Total Stressed Expected Loss"]
    )

st.subheader("Recent Portfolio Data")
st.dataframe(interactive_df.head(20), use_container_width=True)