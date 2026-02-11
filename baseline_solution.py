import pandas as pd

df = pd.read_csv("data/validation_scenario.csv")

actions = []

for _, row in df.iterrows():
    price = row["price"]

    if price < 40:
        charge = 1.5
        discharge = 0
    elif price > 60:
        charge = 0
        discharge = 1.5
    else:
        charge = 0
        discharge = 0

    actions.append({
        "charge": charge,
        "discharge": discharge,
        "grid_import": max(row["demand"] - row["solar"] - row["wind"], 0),
        "grid_export": 0
    })

pd.DataFrame(actions).to_csv("baseline_submission.csv", index=False)
