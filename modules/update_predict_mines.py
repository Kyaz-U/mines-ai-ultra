import pandas as pd
import matplotlib.pyplot as plt
import joblib
from modules.model_guard import safe_predict
from modules.logger import log_info, log_error

MODEL_PATH = "models/mines_rf_model.pkl"
CSV_PATH = "data/mines_data.csv"
CHART_PATH = "data/chart.png"

def validate_all_models(models, X_row):
    results = {}
    for i, model in enumerate(models):
        prob, err = safe_predict(model, X_row)
        results[f"cell_{i+1}"] = prob if prob is not None else 0
    return results

def update_model_and_predict():
    try:
        data = pd.read_csv(CSV_PATH)
        avg_row = data.drop("bombs_count", axis=1).tail(5).mean().values.reshape(1, -1)
        model = joblib.load(MODEL_PATH)
        predictions = validate_all_models([model]*25, avg_row)

        draw_chart(predictions)
        sorted_cells = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [cell for cell, _ in sorted_cells[:7]], None
    except Exception as e:
        return None, str(e)

def draw_chart(predictions):
    try:
        keys = list(predictions.keys())
        vals = [v * 100 for v in predictions.values()]
        plt.figure(figsize=(10, 4))
        plt.bar(keys, vals)
        plt.xticks(rotation=90)
        plt.ylabel("%Xavfsizlik")
        plt.title("25 ta katak bo‘yicha bashorat")
        plt.tight_layout()
        plt.savefig(CHART_PATH)
        plt.close()
    except Exception as e:
        log_error(f"Grafik chizishda xatolik: {e}")
