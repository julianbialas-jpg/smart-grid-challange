import numpy as np
import pandas as pd
import os

HOURS = 60 * 24  # 1440
np.random.seed(42)

def generate_scenario(seed):
    np.random.seed(seed)

    t = np.arange(HOURS)

    # Solar generation (MW)
    solar = 3 * np.maximum(0, np.sin((t % 24 - 6) / 12 * np.pi))
    solar += np.random.normal(0, 0.2, HOURS)

    # Wind generation (MW)
    wind = 2 + 0.5 * np.sin(t / 48) + np.random.normal(0, 0.3, HOURS)
    wind = np.clip(wind, 0, None)

    # Demand (MW)
    demand = 5 + 1.5 * np.sin((t % 24) / 24 * 2 * np.pi)
    demand += np.random.normal(0, 0.5, HOURS)

    # Random demand spikes
    spike_indices = np.random.choice(HOURS, 20, replace=False)
    demand[spike_indices] += np.random.uniform(2, 4, 20)

    # Electricity price ($/MWh)
    price = 50 + 20 * np.sin((t % 24) / 24 * 2 * np.pi)
    price += np.random.normal(0, 5, HOURS)

    # Carbon intensity (kg/MWh)
    carbon = 0.4 + 0.2 * np.sin(t / 100)

    df = pd.DataFrame({
        "time": t,
        "solar": solar,
        "wind": wind,
        "demand": demand,
        "price": price,
        "carbon_intensity": carbon
    })

    return df


def create_datasets():
    os.makedirs("data", exist_ok=True)

    for i in range(1, 4):
        df = generate_scenario(seed=i)
        df.to_csv(f"data/train_scenario_{i}.csv", index=False)

    val = generate_scenario(seed=100)
    val.to_csv("data/validation_scenario.csv", index=False)


if __name__ == "__main__":
    create_datasets()
