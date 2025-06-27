import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# List of symptoms
symptoms = [
    "fever",
    "headache",
    "nausea",
    "vomiting",
    "fatigue",
    "joint_pain",
    "skin_rash",
    "cough",
    "weight_loss",
    "yellow_eyes"
]

# Load dataset
df = pd.read_csv("improved_disease_dataset.csv")

# Split data
X = df.drop('disease', axis=1)
y = df['disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Get user input
print("\nPlease enter 1 if you have the symptom or 0 if you don't:\n")

user_input = []
for symptom in symptoms:
    while True:
        try:
            val = int(input(f"{symptom.capitalize()}: "))
            if val in [0, 1]:
                user_input.append(val)
                break
            else:
                print("Please enter 1 or 0 only.")
        except ValueError:
            print("Please enter a valid number (1 or 0).")
