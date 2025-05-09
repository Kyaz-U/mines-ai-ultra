def validate_input_vector(X_row, expected_features=25):
    if len(X_row[0]) != expected_features:
        return False, f"Xatolik: {len(X_row[0])} ta feature, lekin model {expected_features} ta kutmoqda."
    return True, "Input to‘g‘ri"

def safe_predict(model, X_row):
    valid, msg = validate_input_vector(X_row)
    if not valid:
        return None, msg
    try:
        prob = model.predict_proba(X_row)[0][1]
        return prob, None
    except Exception as e:
        return None, f"Model predictda xatolik: {str(e)}"
