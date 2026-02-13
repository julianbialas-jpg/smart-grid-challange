import numpy as np
import pandas as pd
import os

HOURS = 60 * 24  # 1440
np.random.seed(42)

def generate_scenario(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(HOURS)

    # -------------------------------------------------
    # SLOW WEATHER INDEX (multi-day persistence)
    # -------------------------------------------------
    weather = np.zeros(HOURS)
    weather[0] = 0

    for i in range(1, HOURS):
        weather[i] = 0.98 * weather[i-1] + rng.normal(0, 0.15)

    weather = np.clip(weather, -2, 2)

    # -------------------------------------------------
    # SOLAR
    # daily shape BUT modulated by weather
    # -------------------------------------------------
    daylight = np.maximum(0, np.sin((t % 24 - 6) / 12 * np.pi))

    cloud_factor = 1 - 0.5 * (weather > 0) * weather
    solar = 3.5 * daylight * cloud_factor

    solar += rng.normal(0, 0.12, HOURS)
    solar = np.clip(solar, 0, None)

    # occasional multi-day renewable drought
    if rng.random() < 0.4:
        start = rng.integers(0, HOURS-120)
        solar[start:start+120] *= 0.3

    # -------------------------------------------------
    # WIND (strong autocorrelation)
    # -------------------------------------------------
    wind = np.zeros(HOURS)
    wind[0] = 2

    for i in range(1, HOURS):
        wind[i] = (
            0.94 * wind[i-1]
            + 0.4 * weather[i]
            + rng.normal(0, 0.25)
        )

    wind = np.clip(wind, 0, 6)

    # wind storm
    if rng.random() < 0.3:
        start = rng.integers(0, HOURS-72)
        wind[start:start+72] += rng.uniform(1.5, 3)

    # -------------------------------------------------
    # TEMPERATURE (hidden driver of demand)
    # -------------------------------------------------
    temp = np.zeros(HOURS)
    temp[0] = 15

    for i in range(1, HOURS):
        temp[i] = 0.995 * temp[i-1] + rng.normal(0, 0.4)

    # heat wave / cold snap
    if rng.random() < 0.5:
        start = rng.integers(0, HOURS-150)
        temp[start:start+150] += rng.choice([-8, 8])

    # -------------------------------------------------
    # DEMAND
    # -------------------------------------------------
    daily = 1.2 * np.sin((t % 24) / 24 * 2*np.pi)

    heating = np.maximum(0, 16 - temp) * 0.18
    cooling = np.maximum(0, temp - 24) * 0.22

    demand = 4.8 + daily + heating + cooling

    # evening ramp (duck curve)
    ramp = ((t % 24) >= 18) & ((t % 24) <= 21)
    demand[ramp] += rng.uniform(1.0, 2.0)

    # industrial shocks
    spikes = rng.choice(HOURS, 30, replace=False)
    demand[spikes] += rng.uniform(2, 5, len(spikes))

    demand += rng.normal(0, 0.25, HOURS)
    demand = np.clip(demand, 2, None)

    # -------------------------------------------------
    # PRICE — driven by NET LOAD
    # -------------------------------------------------
    net_load = demand - (solar + wind)

    fuel_price = np.zeros(HOURS)
    fuel_price[0] = 1

    for i in range(1, HOURS):
        fuel_price[i] = 0.999 * fuel_price[i-1] + rng.normal(0, 0.01)

    price = 35 + 14 * net_load + 25 * fuel_price
    price += rng.normal(0, 5, HOURS)

    # scarcity spikes
    scarcity = net_load > 5
    price[scarcity] += rng.uniform(60, 180, scarcity.sum())

    # negative prices when excess renewables
    oversupply = net_load < -2
    price[oversupply] -= rng.uniform(20, 70, oversupply.sum())

    price = np.clip(price, -80, 500)

    # -------------------------------------------------
    # CARBON
    # correlated with fossil usage
    # -------------------------------------------------
    carbon = 0.18 + 0.45 * (net_load > 1.5)
    carbon += rng.normal(0, 0.025, HOURS)
    carbon = np.clip(carbon, 0.1, 0.75)

    return pd.DataFrame({
        "time": t,
        "solar": solar,
        "wind": wind,
        "demand": demand,
        "price": price,
        "carbon_intensity": carbon
    })



def create_datasets():
    os.makedirs("data", exist_ok=True)

    for i in range(1, 4):
        df = generate_scenario(seed=i)
        df.to_csv(f"data/train_scenario_{i}.csv", index=False)

    val = generate_scenario(seed=100)
    val.to_csv("data/validation_scenario.csv", index=False)


if __name__ == "__main__":
    create_datasets()
