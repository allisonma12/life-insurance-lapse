import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score, RocCurveDisplay)

# Import your model here
from sklearn.linear_model import LogisticRegression

# Import train_test_split, seaborn, matplotlib
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Change to your model's name (used in graphs)
MODEL_NAME = "Logistic Regression"

# Read dataset
df = pd.read_csv("data/CleanedLifeInsurance.csv")

# Define features and target
numeric_features = ["ENTRY AGE", "Policy Year", "BENEFIT", "Premium", "SUBSTANDARD RISK"]
categorical_features = ["PAYMENT MODE"]

X = df[numeric_features + categorical_features]
y = df["is_lapse"]

# Split data into train and test sets (keep random_state = 42 to ensure you get the same data splits every time.)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocess features
# numeric: fill missing values (Premium, BENEFIT) with the median, add a column flagging
# which values were missing, then scale. Scaling doesn't matter for tree models but doesn't hurt.
# categorical: one-hot encode PAYMENT MODE into 0/1 columns.
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
])
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Instantiate model
# Replace with your model and baseline hyperparameters. Keep random_state = 42 for reproducibility.
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
)

# Train model on training data
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate performance
print(f"{MODEL_NAME} Performance on Test Set:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nMacro F1:", f1_score(y_test, y_pred, average="macro"))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not lapse", "Lapse"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Graph 1: Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
class_names = ["Not lapse", "Lapse"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={"label": "% of Actual Class"}
)
plt.title(f"Confusion Matrix - {MODEL_NAME}")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

# Graph 2: ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob, name=MODEL_NAME, plot_chance_level=True)
plt.title(f"ROC Curve - {MODEL_NAME}")
plt.tight_layout()
plt.show()
