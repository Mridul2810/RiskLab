from src.config import DEFAULT_LGD, STRESS_SCENARIOS


def add_expected_loss(df):
    df = df.copy()

    df["lgd"] = DEFAULT_LGD
    df["expected_loss"] = df["predicted_pd"] * df["lgd"] * df["loan_amount"]

    return df


def apply_stress_scenario(df, scenario_name):
    df = df.copy()

    scenario = STRESS_SCENARIOS[scenario_name]
    pd_multiplier = scenario["pd_multiplier"]
    lgd_multiplier = scenario["lgd_multiplier"]

    df["stressed_pd"] = (df["predicted_pd"] * pd_multiplier).clip(0, 1)
    df["stressed_lgd"] = (df["lgd"] * lgd_multiplier).clip(0, 1)
    df["stressed_expected_loss"] = df["stressed_pd"] * df["stressed_lgd"] * df["loan_amount"]

    return df


def summarize_stress_results(df):
    summary = {
        "Total Exposure": df["loan_amount"].sum(),
        "Average Predicted PD": df["predicted_pd"].mean(),
        "Total Expected Loss": df["expected_loss"].sum(),
    }

    if "stressed_expected_loss" in df.columns:
        summary["Total Stressed Expected Loss"] = df["stressed_expected_loss"].sum()

    return summary