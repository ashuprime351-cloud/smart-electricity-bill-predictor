"""
Script to generate a realistic synthetic electricity dataset.
Run once to produce data/electricity_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

consumption       = np.random.uniform(50, 800, N)          # kWh / month
people            = np.random.randint(1, 8, N)             # household size
daily_hours       = np.random.uniform(2, 16, N)            # avg daily usage hrs
prev_consumption  = consumption * np.random.uniform(0.7, 1.3, N)

# Realistic bill formula (approximate Indian domestic tariff tiers)
# 0–100 kWh  → ₹3.50/unit
# 101–300     → ₹5.00/unit
# 301–500     → ₹6.50/unit
# >500        → ₹8.00/unit  + fixed charges + noise

def calc_bill(kwh):
    if kwh <= 100:
        return kwh * 3.50
    elif kwh <= 300:
        return 100 * 3.50 + (kwh - 100) * 5.00
    elif kwh <= 500:
        return 100 * 3.50 + 200 * 5.00 + (kwh - 300) * 6.50
    else:
        return 100 * 3.50 + 200 * 5.00 + 200 * 6.50 + (kwh - 500) * 8.00

base_bill = np.array([calc_bill(k) for k in consumption])
# Add fixed charge + people-based load + noise
bill = (base_bill
        + 50                                    # fixed meter charge
        + people * 15                           # per-person load
        + daily_hours * 5                       # daily usage factor
        + np.random.normal(0, 30, N))           # realistic noise

bill = np.clip(bill, 50, None)

df = pd.DataFrame({
    "consumption_kwh":      np.round(consumption, 2),
    "num_people":           people,
    "daily_hours":          np.round(daily_hours, 2),
    "prev_consumption_kwh": np.round(prev_consumption, 2),
    "bill_inr":             np.round(bill, 2),
})

df.to_csv("data/electricity_data.csv", index=False)
print(f"Dataset saved — {len(df)} rows")
print(df.describe())
