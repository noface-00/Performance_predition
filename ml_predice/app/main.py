from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Cargar los modelos y columnas
artefacto = joblib.load("app/model.pkl")
modelo_compresion = artefacto["modelo_compresion"]
modelo_exito = artefacto["modelo_exito"]
columnas = artefacto["columnas"]

# Inicializar Flask
app = Flask(__name__)

# Reutilizamos tu función de recomendación
def recomendar_actividad_dict(datos_estudiante, columnas, modelo_compresion, modelo_exito):
    nuevo_estudiante = pd.DataFrame([datos_estudiante])

    # Binarizar tipo_actividad
    if "tipo_actividad" in nuevo_estudiante.columns:
        nuevo_estudiante = pd.get_dummies(nuevo_estudiante, columns=["tipo_actividad"], drop_first=True)
    
    # Aseguramos que tenga todas las columnas que vio en el entrenamiento
    for col in columnas:
        if col not in nuevo_estudiante.columns:
            nuevo_estudiante[col] = 0
    nuevo_estudiante = nuevo_estudiante[columnas]

    # Predicciones
    # Columnas para compresión (sin 'nivel_comprension')
    X_comp_modelo = [c for c in columnas if c != "nivel_comprension"]
    comprension = modelo_compresion.predict(nuevo_estudiante[X_comp_modelo])[0]

    # Agregar predicción al DataFrame para el modelo de éxito
    nuevo_estudiante["nivel_comprension"] = comprension
    prob_exito = modelo_exito.predict_proba(nuevo_estudiante)[0, 1]


    # Reglas de recomendación
    recomendaciones = []
    if comprension < 0.4:
        recomendaciones.append("Actividades de repaso básico")
    elif comprension < 0.7:
        recomendaciones.append("Actividades intermedias con apoyo visual")
    else:
        recomendaciones.append("Actividades avanzadas con menos pistas")

    if prob_exito < 0.5:
        recomendaciones.append("Reducir dificultad o dar más pistas")
    else:
        recomendaciones.append("Incrementar dificultad progresivamente")

    return {
        "nivel_comprension": float(comprension),
        "prob_exito": float(prob_exito),
        "recomendaciones": recomendaciones
    }

# Endpoint para predicción
@app.route("/predict", methods=["POST"])
def predict():
    try:
        datos = request.get_json()
        resultado = recomendar_actividad_dict(datos, columnas, modelo_compresion, modelo_exito)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
