from flask import Flask, request, jsonify
import pandas as pd
import joblib

from utils import predict_student

app = Flask(__name__)

# Cargar modelo
model = joblib.load("app/model.pkl")

# Endpoint de prueba
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Servicio de predicción activo 🚀"})

# Endpoint para predicción
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])  # Convertir entrada a DataFrame
    prediction = predict_student(model, df)
    return jsonify({"prediccion": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
