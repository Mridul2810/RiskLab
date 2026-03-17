RANDOM_SEED = 42
N_BORROWERS = 500

DEFAULT_LGD = 0.45

STRESS_SCENARIOS = {
    "Base": {
        "pd_multiplier": 1.0,
        "lgd_multiplier": 1.0
    },
    "Mild Recession": {
        "pd_multiplier": 1.25,
        "lgd_multiplier": 1.10
    },
    "Severe Recession": {
        "pd_multiplier": 1.60,
        "lgd_multiplier": 1.25
    },
    "Rate Shock": {
        "pd_multiplier": 1.20,
        "lgd_multiplier": 1.05
    }
}