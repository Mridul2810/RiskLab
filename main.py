import os
import pandas as pd
from src.data_generator import generate_portfolio
from src.preprocess import preprocess_portfolio
from src.model import train_default_model, add_predicted_pd
from src.stress_test import add_expected_loss, apply_stress_scenario, summarize_stress_results


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    portfolio = generate_portfolio()
    portfolio.to_csv("data/raw/borrower_portfolio.csv", index=False)

    processed_portfolio = preprocess_portfolio(portfolio)
    processed_portfolio.to_csv("data/processed/borrower_portfolio_processed.csv", index=False)

    model, X_train, X_test, y_train, y_test, y_prob, auc, report = train_default_model(processed_portfolio)

    modeled_portfolio = add_predicted_pd(processed_portfolio, model)
    modeled_portfolio = add_expected_loss(modeled_portfolio)
    modeled_portfolio.to_csv("results/portfolio_with_pd.csv", index=False)

    stressed_portfolio = apply_stress_scenario(modeled_portfolio, "Severe Recession")
    stressed_portfolio.to_csv("results/stressed_portfolio.csv", index=False)

    stress_summary = summarize_stress_results(stressed_portfolio)
    stress_summary_df = pd.DataFrame(list(stress_summary.items()), columns=["Metric", "Value"])
    stress_summary_df.to_csv("results/stress_summary.csv", index=False)

    with open("results/model_report.txt", "w") as f:
        f.write(f"AUC: {auc}\n\n")
        f.write(report)

    metrics_df = pd.DataFrame({
        "Metric": ["AUC", "Default Rate"],
        "Value": [auc, portfolio["default"].mean()]
    })
    metrics_df.to_csv("results/model_metrics.csv", index=False)

    print("RiskLab pipeline complete.")
    print("\nAUC:", auc)
    print("\nStress Summary:")
    print(stress_summary_df)

    print("\nSaved:")
    print("- data/raw/borrower_portfolio.csv")
    print("- data/processed/borrower_portfolio_processed.csv")
    print("- results/portfolio_with_pd.csv")
    print("- results/stressed_portfolio.csv")
    print("- results/stress_summary.csv")
    print("- results/model_report.txt")
    print("- results/model_metrics.csv")


if __name__ == "__main__":
    main()