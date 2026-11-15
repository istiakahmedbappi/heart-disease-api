from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/heart.csv")
MODEL_PATH = Path("model/heart_model.joblib")

# 1. Load the data. Last column "target": 1 = heart disease, 0 = no heart disease
df = pd.read_csv(DATA_PATH)
print("Rows, columns:", df.shape)

X = df.drop(columns="target")  # the 13 input features
y = df["target"]               # what we want to predict

# 2. Split: 80% to learn from, 20% to check on data the model has never seen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train the classifier (the goal of this assignment is Docker, not high accuracy)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Quick check
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {accuracy:.3f}")

# 5. Save. Because we trained on a DataFrame, the model remembers the column names.
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("Saved model to", MODEL_PATH)