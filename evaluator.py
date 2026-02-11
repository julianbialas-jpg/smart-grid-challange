import pandas as pd
from environment import SmartGridEnv

def evaluate_submission(submission_file, scenario_file):
    df = pd.read_csv(scenario_file)
    actions_df = pd.read_csv(submission_file)

    if len(actions_df) != len(df):
        raise ValueError("Submission length mismatch.")

    actions = []

    for _, row in actions_df.iterrows():
        actions.append({
            "charge": row["charge"],
            "discharge": row["discharge"],
            "grid_import": row["grid_import"],
            "grid_export": row["grid_export"]
        })

    env = SmartGridEnv(df)
    score = env.run(actions)

    return score
