
def predict_student(model, df):
    # Aquí puedes hacer transformaciones necesarias
    y_pred = model.predict(df)
    return y_pred.tolist()
