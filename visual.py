# ==========================================================
# Exploratory Data Analysis (EDA)
# Space Mission Outcome Prediction Project
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

# ------------------ Load Dataset ------------------
df = pd.read_csv("C:/Users/acer/Downloads/dataset/space_missions_dataset.csv")

# ------------------ Create Target Variable ------------------
# Convert Mission Success (%) into classification label
df["Success"] = df["Mission Success (%)"].apply(lambda x: 1 if x >= 80 else 0)

# Column reference
cost_col = "Mission Cost (billion USD)"

# Create folder to save figures (optional)
import os
os.makedirs("EDA_Figures", exist_ok=True)

# ==========================================================
# 1. Bar Chart: Successful vs Failed Missions
# ==========================================================

counts = df["Success"].value_counts().sort_index()

plt.figure(figsize=(7,5))
bars = plt.bar(["Failed", "Successful"], counts)

plt.title("Distribution of Space Mission Outcomes", fontsize=14, fontweight="bold")
plt.xlabel("Mission Outcome", fontsize=12)
plt.ylabel("Number of Missions", fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             str(height), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig("EDA_Figures/mission_outcome_distribution.png", dpi=300)
plt.show()


# ==========================================================
# 2. Histogram: Mission Cost Distribution
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(df[cost_col],
         bins=20,
         edgecolor='black',
         alpha=0.85)

plt.title("Distribution of Mission Cost", fontsize=14, fontweight="bold")
plt.xlabel("Mission Cost (billion USD)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.grid(linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("EDA_Figures/mission_cost_distribution.png", dpi=300)
plt.show()


# ==========================================================
# 3. Scatter Plot: Mission Cost vs Mission Outcome
# ==========================================================

plt.figure(figsize=(8,5))

plt.scatter(df[cost_col],
            df["Success"],
            alpha=0.7)

plt.title("Mission Cost vs Mission Success", fontsize=14, fontweight="bold")
plt.xlabel("Mission Cost (billion USD)", fontsize=12)
plt.ylabel("Mission Outcome (1 = Success, 0 = Failure)", fontsize=12)

plt.grid(linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("EDA_Figures/cost_vs_success.png", dpi=300)
plt.show()


# ==========================================================
# Optional: Display Dataset Summary (Useful for Report)
# ==========================================================

print("\nDataset Shape:", df.shape)
print("\nClass Distribution:\n", df["Success"].value_counts())
print("\nDataset Description:\n", df.describe())