import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/CleanedLifeInsurance.csv", encoding="utf-8-sig")
df.head()

# Create binary variable for substandard risk
df["SUBSTD_RISK"] = df["SUBSTANDARD RISK"].apply(
    lambda x: "Y" if x > 0 else "N")
print(df["SUBSTD_RISK"].value_counts())

# Calculate lapse rate
lapse_summary = df.groupby("SUBSTD_RISK")["is_lapse"].agg(
    ["mean", "count"]
).reset_index()
lapse_summary.columns = ["SUBSTD_RISK", "lapse_rate", "count"]
print(lapse_summary)

# Plotting the lapse rate by substandard risk
plt.figure(figsize=(7, 5))

ax = sns.barplot(
    data=lapse_summary,
    x="SUBSTD_RISK",
    y="lapse_rate",
    order=["N", "Y"]
)

plt.title("Lapse Rate by Substandard Risk")
plt.xlabel("Substandard Risk")
plt.ylabel("Lapse Rate")
plt.ylim(0, 0.7)

for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.1%}",
        ha="center"
    )
plt.tight_layout()
plt.show()