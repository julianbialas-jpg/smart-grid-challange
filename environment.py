import numpy as np

BATTERY_CAPACITY = 5.0  # MWh
MAX_CHARGE = 2.0
MAX_DISCHARGE = 2.0
EFFICIENCY = 0.95

BLACKOUT_PENALTY = 500
OVERLOAD_PENALTY = 100
DEGRADATION_ALPHA = 5
CARBON_COST = 50

class SmartGridEnv:

    def __init__(self, df):
        self.df = df
        self.reset()

    def reset(self):
        self.soc = 2.5
        self.total_cost = 0
        self.degradation = 0
        self.blackouts = 0
        self.overloads = 0

    def step(self, t, action):
        charge = np.clip(action["charge"], 0, MAX_CHARGE)
        discharge = np.clip(action["discharge"], 0, MAX_DISCHARGE)
        grid_import = max(action["grid_import"], 0)
        grid_export = max(action["grid_export"], 0)

        solar = self.df.loc[t, "solar"]
        wind = self.df.loc[t, "wind"]
        demand = self.df.loc[t, "demand"]
        price = self.df.loc[t, "price"]
        carbon = self.df.loc[t, "carbon_intensity"]

        net_supply = solar + wind + discharge - charge + grid_import - grid_export

        # Battery update
        self.soc += charge * EFFICIENCY
        self.soc -= discharge / EFFICIENCY

        if self.soc > BATTERY_CAPACITY:
            self.overloads += 1
            self.soc = BATTERY_CAPACITY

        if self.soc < 0:
            self.blackouts += 1
            self.soc = 0

        # Degradation
        self.degradation += DEGRADATION_ALPHA * abs(discharge)

        # Power balance penalty
        if net_supply < demand:
            self.blackouts += 1

        # Costs
        energy_cost = grid_import * price - grid_export * price
        carbon_cost = grid_import * carbon * CARBON_COST

        self.total_cost += (
            energy_cost +
            carbon_cost +
            self.blackouts * BLACKOUT_PENALTY +
            self.overloads * OVERLOAD_PENALTY +
            self.degradation
        )

    def run(self, actions):
        self.reset()
        for t in range(len(actions)):
            self.step(t, actions[t])

        return -self.total_cost  # Higher is better
