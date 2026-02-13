import streamlit as st
import pandas as pd
import sqlite3
import os
import numpy as np
from datetime import datetime
from evaluator import evaluate_submission
from environment import SmartGridEnv

DB_FILE = "leaderboard.db"
SCENARIO = "data/validation_scenario.csv"

REQUIRED_COLUMNS = ["charge", "discharge", "grid_import", "grid_export"]

# ==========================================================
# DATABASE
# ==========================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                name TEXT,
                score REAL,
                timestamp TEXT
            )
        """)

    conn.commit()
    conn.close()


def add_score(name, score):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO leaderboard VALUES (?,?,?)",
        (name, score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def get_leaderboard():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(
        "SELECT * FROM leaderboard ORDER BY score DESC",
        conn
    )
    conn.close()
    return df

init_db()

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def validate_submission(df, scenario_length):
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"CSV must contain columns exactly: {REQUIRED_COLUMNS}"
        )

    if len(df) != scenario_length:
        raise ValueError(
            f"Submission must have {scenario_length} rows."
        )

    if df.isnull().values.any():
        raise ValueError("Submission contains NaN values.")

def simulate_for_plot(actions_df, scenario_df):
    env = SmartGridEnv(scenario_df)
    soc_history = []
    soc = 2.5

    for t in range(len(actions_df)):
        action = actions_df.iloc[t].to_dict()
        soc += action["charge"] * 0.95
        soc -= action["discharge"] / 0.95
        soc = np.clip(soc, 0, 5)
        soc_history.append(soc)

    return soc_history

# ==========================================================
# STREAMLIT APP
# ==========================================================

st.set_page_config(page_title="Smart Grid Challenge", layout="wide")

st.title("⚡ Smart Grid Energy Management Challenge")

tabs = st.tabs([
    "📖 Problem Description",
    "📂 Download Data",
    "📤 Submit Solution",
    "🏆 Leaderboard"
])

# ==========================================================
# TAB 1 — PROBLEM DESCRIPTION
# ==========================================================

with tabs[0]:

    st.header("Overview")

    st.markdown("""
You operate a **microgrid system** over **60 days (1440 hours)**.

The grid contains:

- Solar generation (stochastic)
- Wind generation (stochastic)
- Battery storage (5 MWh capacity)
- Grid import/export with dynamic pricing
- Time-varying carbon intensity
- Industrial + residential demand
""")

    st.header("Decision Variables (each hour t)")

    st.markdown("""
You must output:

- charge[t]
- discharge[t]
- grid_import[t]
- grid_export[t]
""")

    st.header("Battery Constraints")

    st.markdown("""
- Capacity: 5 MWh  
- Max charge rate: 2 MW  
- Max discharge rate: 2 MW  
- Efficiency: 95%  
- SOC must stay between 0 and 5  
""")

    st.header("Objective Function")

    st.markdown("""
Total Cost =

Energy Cost  
+ Carbon Cost  
+ 500 × Blackouts  
+ 100 × Overloads  
+ 5 × |Discharge|  

Carbon Cost = 50 × carbon_intensity × grid_import  

Final Score = − Total Cost  

Higher score = better.
""")

    st.header("Submission Format")

    st.markdown("""
Upload a CSV file with exactly 1440 rows and the columns:

charge,discharge,grid_import,grid_export

Example:

charge,discharge,grid_import,grid_export  
1.5,0,2.3,0  
0,1.2,1.0,0  
...
""")

    st.markdown("""
Allowed solution methods:

- Reinforcement Learning (PPO, DQN, ...)
- Particle Swarm Optimization
- Genetic Algorithms
- NN
- ACO
- ABC

""")

# ==========================================================
# TAB 2 — DOWNLOAD DATA
# ==========================================================

with tabs[1]:

    st.header("Validation Dataset")

    if os.path.exists(SCENARIO):
        df = pd.read_csv(SCENARIO)
        st.dataframe(df.head())

        st.download_button(
            label="Download validation_scenario.csv",
            data=df.to_csv(index=False),
            file_name="validation_scenario.csv",
            mime="text/csv"
        )
    else:
        st.warning("Dataset not found.")

# ==========================================================
# TAB 3 — SUBMISSION
# ==========================================================

with tabs[2]:

    st.header("Submit Your Solution")

    name = st.text_input("Your Name")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file and name:

        try:
            # Save uploaded file once
            temp_path = "temp_submission.csv"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Now read from saved file
            submission_df = pd.read_csv(temp_path)
            scenario_df = pd.read_csv(SCENARIO)

            validate_submission(submission_df, len(scenario_df))

            # Evaluate using file path (not uploaded_file object)
            score = evaluate_submission(
                temp_path,
                SCENARIO
            )

            add_score(name, score)

            st.success(f"Submission successful! Score: {score:.2f}")

            # Visualization
            st.subheader("Battery State of Charge")
            soc_history = simulate_for_plot(
                submission_df,
                scenario_df
            )
            st.line_chart(soc_history)

            st.subheader("Charge / Discharge Profile")
            st.line_chart(submission_df[["charge", "discharge"]])

            st.subheader("Grid Import / Export")
            st.line_chart(submission_df[["grid_import", "grid_export"]])

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ==========================================================
# TAB 4 — LEADERBOARD
# ==========================================================

with tabs[3]:

    st.header("Leaderboard")

    leaderboard = get_leaderboard()

    if len(leaderboard) > 0:
        leaderboard.index += 1
        leaderboard.index.name = "Rank"
        st.dataframe(leaderboard)
    else:
        st.info("No submissions yet.")
