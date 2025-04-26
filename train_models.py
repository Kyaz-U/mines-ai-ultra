import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from modules.logger import log_info

data = pd.read_csv("data/mines_data.csv")
X = data.drop("bombs_count", axis=1)
y = data["bombs_count"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "models/mines_rf_model.pkl")
log_info("Model saqlandi: models/mines_rf_model.pkl")
