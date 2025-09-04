import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pandas as pd
import joblib
import train_model  # tu función de entrenamiento actualizada

app = Flask(__name__)

UPLOAD_FOLDER = "datos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "app/model.pkl"

# ============================
# 1️⃣ Cargar modelo existente si hay
# ============================
if os.path.exists(MODEL_PATH):
    artefacto = joblib.load(MODEL_PATH)
    modelo_compresion = artefacto["modelo_compresion"]
    modelo_exito = artefacto["modelo_exito"]
    columnas = artefacto["columnas"]
else:
    modelo_compresion = modelo_exito = columnas = None

# ============================
# 2️⃣ Endpoint para predecir
# ============================
def recomendar_actividad_dict(datos_estudiante, columnas, modelo_compresion, modelo_exito):
    nuevo_estudiante = pd.DataFrame([datos_estudiante])

    if "tipo_actividad" in nuevo_estudiante.columns:
        nuevo_estudiante = pd.get_dummies(nuevo_estudiante, columns=["tipo_actividad"], drop_first=True)

    for col in columnas:
        if col not in nuevo_estudiante.columns:
            nuevo_estudiante[col] = 0
    nuevo_estudiante = nuevo_estudiante[columnas]

    X_comp_modelo = [c for c in columnas if c != "nivel_comprension"]
    comprension = modelo_compresion.predict(nuevo_estudiante[X_comp_modelo])[0]

    nuevo_estudiante["nivel_comprension"] = comprension
    prob_exito = modelo_exito.predict_proba(nuevo_estudiante)[0,1]

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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if modelo_compresion is None or modelo_exito is None:
            return jsonify({"error": "Modelo no entrenado"}), 400
        datos = request.get_json()
        resultado = recomendar_actividad_dict(datos, columnas, modelo_compresion, modelo_exito)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============================
# 3️⃣ Endpoint para subir CSV
# ============================
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No se encontró archivo"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({"mensaje": f"Archivo recibido: {file.filename}", "ruta": filepath})

# ============================
# 4️⃣ Endpoint para reentrenar acumulando CSVs
# ============================
@app.route("/retrain", methods=["POST"])
def retrain():
    global modelo_compresion, modelo_exito, columnas
    try:
        data = request.json
        filepath = data.get("filepath")

        # Verificar archivo
        if not filepath or not os.path.exists(filepath):
            return jsonify({"error": "Archivo no encontrado"}), 400

        # Entrenar acumulando todos los CSV
        train_model.train_model(ruta_csv=filepath, carpeta_dataset=UPLOAD_FOLDER, guardar_modelo=MODEL_PATH)

        # Recargar modelo en memoria
        artefacto = joblib.load(MODEL_PATH)
        modelo_compresion = artefacto["modelo_compresion"]
        modelo_exito = artefacto["modelo_exito"]
        columnas = artefacto["columnas"]

        return jsonify({"mensaje": f"Modelo reentrenado con {filepath}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # Página principal con formulario


@app.route("/", methods=["GET", "POST"])
def index_page():
    if request.method == "POST":
        try:
            datos = {
                "edad": int(request.form["edad"]),
                "grado": int(request.form["grado"]),
                "tipo_actividad": request.form["tipo_actividad"],
                "dificultad": int(request.form.get("dificultad", 1)),
                "tiempo_seg": int(request.form.get("tiempo_seg", 100)),
                "intentos": int(request.form.get("intentos", 1)),
                "pistas": int(request.form.get("pistas", 0)),
                "correcto": int(request.form.get("correcto", 1)),
                "nota": int(request.form.get("nota", 7)),
                "secuencia_actividades": int(request.form.get("secuencia_actividades",1)),
                "evolucion_desempeno": float(request.form.get("evolucion_desempeno",0)),
                "nivel_concentracion": float(request.form.get("nivel_concentracion",0.7)),
                "comparacion_historial": float(request.form.get("comparacion_historial",0))
            }
            resultado = recomendar_actividad_dict(datos, columnas, modelo_compresion, modelo_exito)
            return render_template("index.html", resultado=resultado)
        except Exception as e:
            flash(f"Error: {e}")
            return redirect(url_for("index_page"))
    return render_template("index.html", resultado=None)

# Página para reentrenar subiendo CSV
@app.route("/retrain_page", methods=["GET", "POST"])
def retrain_page():
    global modelo_compresion, modelo_exito, columnas
    if request.method == "POST":
        if "file" not in request.files:
            flash("No se seleccionó archivo")
            return redirect(url_for("retrain_page"))
        file = request.files["file"]
        if file.filename == "":
            flash("Nombre de archivo vacío")
            return redirect(url_for("retrain_page"))
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        train_model.train_model(ruta_csv=filepath, carpeta_dataset=UPLOAD_FOLDER, guardar_modelo=MODEL_PATH)

        # Recargar modelo
        artefacto = joblib.load(MODEL_PATH)
        modelo_compresion = artefacto["modelo_compresion"]
        modelo_exito = artefacto["modelo_exito"]
        columnas = artefacto["columnas"]

        flash("Modelo reentrenado correctamente ✅")
        return redirect(url_for("retrain_page"))

    return render_template("retrain.html")
# ============================
# 5️⃣ Run Flask
# ============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
