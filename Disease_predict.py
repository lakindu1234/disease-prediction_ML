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


