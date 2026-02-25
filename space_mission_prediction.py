import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------- 1. Load Dataset --------------------
df = pd.read_csv("C:/Users/acer/Downloads/dataset/space_missions_dataset.csv")

print("First 5 Rows of Original Dataset:")
print(df.head())

# -------------------- 2. Target Creation --------------------


df["Success"] = df["Mission Success (%)"].apply(lambda x: 1 if x >= 80 else 0)

# Drop unnecessary columns
df.drop(
    ["Mission ID", "Mission Name", "Launch Date", "Mission Success (%)"],
    axis=1,
    inplace=True,
    errors="ignore"
)

print("\nCleaned Dataset:")
print(df.head())

# -------------------- 3. Encode Categorical Features --------------------
encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# -------------------- 4. Train-Test Split (Stratified) --------------------
X = df.drop("Success", axis=1)
y = df["Success"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   
)

# -------------------- 5. Model Training --------------------

# Decision Tree (Baseline Model)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest (Main Model)
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)



# -------------------- 6. Evaluation --------------------

print("\n--- Random Forest Results ---\n")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

print("\n--- Decision Tree Results ---\n")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# -------------------- 7. Confusion Matrix --------------------

cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Unsuccessful", "Successful"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# -------------------- 8. Feature Importance --------------------

importances = pd.Series(rf.feature_importances_, index=X.columns)

print("\n--- Feature Importance ---")
print(importances.sort_values(ascending=False))

# Plot feature importance
importances.sort_values(ascending=False).plot(kind="bar")
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# -------------------- 9. CLI Prediction Interface --------------------

def predict_mission(model):
    print("\n--- Space Mission Outcome Prediction ---")
    user_input = []

    for col in X.columns:
        value = float(input(f"Enter {col}: "))
        user_input.append(value)

    prediction = model.predict([user_input])[0]

    if prediction == 1:
        print("\nPrediction: SUCCESSFUL Mission 🚀")
    else:
        print("\nPrediction: UNSUCCESSFUL Mission ❌")


# Run CLI with Random Forest
predict_mission(rf)