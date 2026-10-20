import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("predicting-life-insurance-lapse/data/CleanedLifeInsurance.csv")

sns.violinplot(x = "Policy Year (Decimal)", y = "POLICY STATUS", data = data,
            hue = "POLICY STATUS")
plt.show()