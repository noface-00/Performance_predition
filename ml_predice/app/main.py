import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pandas as pd
import joblib
import train_model  # tu función de entrenamiento actualizada
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'  # Cambiar en producción

# Configuración
UPLOAD_FOLDER = "app/datos"
MODEL_PATH = "app/model.pkl"

# Crear carpetas necesarias
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ============================
# 1️⃣ Cargar modelo existente si hay
# ============================
def cargar_modelo():
    """Función para cargar el modelo y manejar errores"""
    global modelo_compresion, modelo_exito, columnas
    
    if os.path.exists(MODEL_PATH):
        try:
            artefacto = joblib.load(MODEL_PATH)
            modelo_compresion = artefacto["modelo_compresion"]
            modelo_exito = artefacto["modelo_exito"]
            columnas = artefacto["columnas"]
            logger.info("Modelo cargado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            modelo_compresion = modelo_exito = columnas = None
            return False
    else:
        logger.warning("No se encontró modelo previo")
        modelo_compresion = modelo_exito = columnas = None
        return False

# Inicializar modelo
modelo_cargado = cargar_modelo()

# ============================
# 2️⃣ Función mejorada de predicción
# ============================
def recomendar_actividad_dict(datos_estudiante, columnas, modelo_compresion, modelo_exito):
    """
    Función mejorada para generar recomendaciones con mejor manejo de errores
    """
    try:
        nuevo_estudiante = pd.DataFrame([datos_estudiante])
        
        # Validar datos de entrada
        campos_requeridos = ['edad', 'grado', 'tipo_actividad']
        for campo in campos_requeridos:
            if campo not in nuevo_estudiante.columns:
                raise ValueError(f"Campo requerido faltante: {campo}")

        # Preprocesamiento
        if "tipo_actividad" in nuevo_estudiante.columns:
            nuevo_estudiante = pd.get_dummies(nuevo_estudiante, columns=["tipo_actividad"], drop_first=True)

        # Asegurar que tenemos todas las columnas necesarias
        for col in columnas:
            if col not in nuevo_estudiante.columns:
                nuevo_estudiante[col] = 0
        nuevo_estudiante = nuevo_estudiante[columnas]

        # Predicción de comprensión
        X_comp_modelo = [c for c in columnas if c != "nivel_comprension"]
        comprension = modelo_compresion.predict(nuevo_estudiante[X_comp_modelo])[0]

        # Predicción de éxito
        nuevo_estudiante["nivel_comprension"] = comprension
        prob_exito = modelo_exito.predict_proba(nuevo_estudiante)[0, 1]

        # Generar recomendaciones inteligentes
        recomendaciones = generar_recomendaciones_avanzadas(datos_estudiante, comprension, prob_exito)

        return {
            "nivel_comprension": float(comprension),
            "prob_exito": float(prob_exito),
            "recomendaciones": recomendaciones,
            "confianza": float(max(modelo_exito.predict_proba(nuevo_estudiante)[0]))
        }
    
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise e

def generar_recomendaciones_avanzadas(datos, comprension, prob_exito):
    """
    Genera recomendaciones más específicas y personalizadas
    """
    recomendaciones = []
    
    # Análisis por nivel de comprensión
    if comprension < 0.4:
        recomendaciones.append("🔴 Nivel bajo: Reforzar conceptos básicos con actividades guiadas")
        recomendaciones.append("📚 Recomendar sesiones de tutoría personalizada")
    elif comprension < 0.7:
        recomendaciones.append("🟡 Nivel medio: Continuar con actividades intermedias")
        recomendaciones.append("💡 Incluir más ejemplos prácticos en las explicaciones")
    else:
        recomendaciones.append("🟢 Nivel alto: Introducir desafíos más complejos")
        recomendaciones.append("🎯 Considerar rol de mentor para otros estudiantes")

    # Análisis por probabilidad de éxito
    if prob_exito < 0.3:
        recomendaciones.append("⚠️ Riesgo alto: Reducir dificultad y aumentar apoyo")
    elif prob_exito < 0.7:
        recomendaciones.append("⚡ Probabilidad moderada: Mantener estrategias actuales")
    else:
        recomendaciones.append("✅ Alta probabilidad de éxito: Incrementar complejidad gradualmente")

    # Análisis específico por actividad
    tipo_actividad = datos.get('tipo_actividad', '')
    if tipo_actividad == 'sopa_letras':
        if datos.get('tiempo_seg', 0) > 300:
            recomendaciones.append("⏰ Practicar reconocimiento rápido de patrones")
    elif tipo_actividad == 'crucigrama':
        if datos.get('pistas', 0) > 3:
            recomendaciones.append("📖 Ampliar vocabulario con lecturas dirigidas")
    elif tipo_actividad == 'relacionar':
        recomendaciones.append("🔗 Fortalecer pensamiento analítico con ejercicios de lógica")
    elif tipo_actividad == 'memoria':
        recomendaciones.append("🧠 Aplicar técnicas de memorización espacial")

    # Análisis temporal
    if datos.get('tiempo_seg', 0) > 600:
        recomendaciones.append("⏱️ Trabajar en gestión del tiempo con cronómetro")
    
    # Análisis de intentos
    if datos.get('intentos', 1) > 5:
        recomendaciones.append("🎯 Mejorar precisión con práctica focalizada")

    return recomendaciones[:5]  # Limitar a 5 recomendaciones principales

# ============================
# 3️⃣ Endpoints mejorados
# ============================
@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint mejorado para predicciones con mejor validación"""
    try:
        if modelo_compresion is None or modelo_exito is None:
            return jsonify({
                "error": "Modelo no entrenado. Por favor, entrena el modelo primero.",
                "redirect": "/retrain_page"
            }), 400

        # Obtener datos del request
        if request.is_json:
            datos = request.get_json()
        else:
            return jsonify({"error": "Se requiere contenido JSON"}), 400

        # Validar datos requeridos
        campos_requeridos = ['edad', 'grado', 'tipo_actividad']
        for campo in campos_requeridos:
            if campo not in datos:
                return jsonify({"error": f"Campo requerido faltante: {campo}"}), 400

        # Validaciones específicas
        if not (5 <= int(datos['edad']) <= 25):
            return jsonify({"error": "La edad debe estar entre 5 y 25 años"}), 400
        
        if not (1 <= int(datos['grado']) <= 12):
            return jsonify({"error": "El grado debe estar entre 1 y 12"}), 400

        tipos_validos = ['sopa_letras', 'crucigrama', 'relacionar', 'memoria']
        if datos['tipo_actividad'] not in tipos_validos:
            return jsonify({"error": f"Tipo de actividad debe ser uno de: {tipos_validos}"}), 400

        # Realizar predicción
        resultado = recomendar_actividad_dict(datos, columnas, modelo_compresion, modelo_exito)
        
        # Registrar predicción
        logger.info(f"Predicción realizada: comprensión={resultado['nivel_comprension']:.3f}, éxito={resultado['prob_exito']:.3f}")
        
        return jsonify(resultado)

    except ValueError as ve:
        logger.warning(f"Error de validación: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Endpoint mejorado para subir CSV con mejor validación"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se encontró archivo en la petición"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Solo se permiten archivos CSV"}), 400

        # Generar nombre único para evitar conflictos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_safe = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename_safe)
        
        # Guardar archivo
        file.save(filepath)
        
        # Validar contenido del CSV
        try:
            df = pd.read_csv(filepath)
            filas, columnas_count = df.shape
            
            # Verificar columnas mínimas
            columnas_minimas = ['edad', 'grado', 'tipo_actividad']
            columnas_faltantes = [col for col in columnas_minimas if col not in df.columns]
            
            if columnas_faltantes:
                os.remove(filepath)  # Limpiar archivo inválido
                return jsonify({
                    "error": f"Columnas faltantes: {columnas_faltantes}",
                    "columnas_encontradas": list(df.columns),
                    "columnas_requeridas": columnas_minimas
                }), 400

            logger.info(f"CSV subido exitosamente: {filename_safe} ({filas} filas, {columnas_count} columnas)")
            
            return jsonify({
                "mensaje": f"Archivo recibido exitosamente: {file.filename}",
                "ruta": filepath,
                "filas": filas,
                "columnas": columnas_count,
                "columnas_encontradas": list(df.columns),
                "timestamp": timestamp
            })
            
        except Exception as e:
            # Limpiar archivo si hay error de lectura
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Error leyendo CSV: {str(e)}"}), 400

    except Exception as e:
        logger.error(f"Error subiendo archivo: {e}")
        return jsonify({"error": f"Error interno subiendo archivo: {str(e)}"}), 500

@app.route("/retrain", methods=["POST"])
def retrain():
    """Endpoint mejorado para reentrenamiento con mejor manejo de errores"""
    global modelo_compresion, modelo_exito, columnas
    
    try:
        data = request.json
        if not data or 'filepath' not in data:
            return jsonify({"error": "Se requiere la ruta del archivo"}), 400

        filepath = data.get("filepath")
        
        # Verificar que el archivo existe
        if not filepath or not os.path.exists(filepath):
            return jsonify({"error": "Archivo no encontrado"}), 404

        logger.info(f"Iniciando reentrenamiento con: {filepath}")

        # Crear backup del modelo actual si existe
        if os.path.exists(MODEL_PATH):
            backup_path = f"{MODEL_PATH}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copy2(MODEL_PATH, backup_path)
            logger.info(f"Backup creado: {backup_path}")

        # Entrenar modelo
        train_model.train_model(
            ruta_csv=filepath, 
            carpeta_dataset=UPLOAD_FOLDER, 
            guardar_modelo=MODEL_PATH
        )

        # Recargar modelo en memoria
        if cargar_modelo():
            logger.info("Modelo reentrenado y recargado exitosamente")
            
            # Obtener estadísticas del modelo
            archivos_csv = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
            
            return jsonify({
                "mensaje": f"Modelo reentrenado exitosamente con {os.path.basename(filepath)}",
                "timestamp": datetime.now().isoformat(),
                "archivos_utilizados": len(archivos_csv),
                "modelo_activo": True,
                "backup_creado": os.path.exists(backup_path) if 'backup_path' in locals() else False
            })
        else:
            return jsonify({"error": "Modelo reentrenado pero falló la recarga"}), 500

    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
        return jsonify({"error": "Archivo de datos no encontrado"}), 404
    except Exception as e:
        logger.error(f"Error en reentrenamiento: {e}")
        return jsonify({"error": f"Error en reentrenamiento: {str(e)}"}), 500

# ============================
# 4️⃣ Rutas de páginas mejoradas
# ============================
@app.route("/", methods=["GET", "POST"])
def index_page():
    """Página principal mejorada con mejor manejo de errores"""
    if request.method == "POST":
        try:
            # Validar que el modelo esté disponible
            if modelo_compresion is None or modelo_exito is None:
                flash("⚠️ Modelo no disponible. Por favor, entrena el modelo primero.", "warning")
                return redirect(url_for("retrain_page"))

            # Recopilar datos del formulario con valores por defecto
            datos = {
                "edad": int(request.form.get("edad", 0)),
                "grado": int(request.form.get("grado", 0)),
                "tipo_actividad": request.form.get("tipo_actividad", ""),
                "dificultad": int(request.form.get("dificultad", 1)),
                "tiempo_seg": int(request.form.get("tiempo_seg", 100)),
                "intentos": int(request.form.get("intentos", 1)),
                "pistas": int(request.form.get("pistas", 0)),
                "correcto": int(request.form.get("correcto", 1)),
                "nota": int(request.form.get("nota", 7)),
                "secuencia_actividades": int(request.form.get("secuencia_actividades", 1)),
                "evolucion_desempeno": float(request.form.get("evolucion_desempeno", 0.0)),
                "nivel_concentracion": float(request.form.get("nivel_concentracion", 0.7)),
                "comparacion_historial": float(request.form.get("comparacion_historial", 0.0))
            }

            # Validaciones básicas
            if not (5 <= datos["edad"] <= 25):
                flash("❌ La edad debe estar entre 5 y 25 años", "error")
                return redirect(url_for("index_page"))
            
            if not (1 <= datos["grado"] <= 12):
                flash("❌ El grado debe estar entre 1 y 12", "error")
                return redirect(url_for("index_page"))

            if datos["tipo_actividad"] not in ['sopa_letras', 'crucigrama', 'relacionar', 'memoria']:
                flash("❌ Tipo de actividad no válido", "error")
                return redirect(url_for("index_page"))

            # Realizar predicción
            resultado = recomendar_actividad_dict(datos, columnas, modelo_compresion, modelo_exito)
            
            # Mensaje de éxito
            flash("✅ Predicción realizada exitosamente", "success")
            
            return render_template("index.html", resultado=resultado)

        except ValueError as ve:
            logger.warning(f"Error de validación: {ve}")
            flash(f"❌ Error de validación: {ve}", "error")
            return redirect(url_for("index_page"))
        except Exception as e:
            logger.error(f"Error en predicción web: {e}")
            flash(f"❌ Error interno: {e}", "error")
            return redirect(url_for("index_page"))

    return render_template("index.html", resultado=None)

@app.route("/retrain_page", methods=["GET", "POST"])
def retrain_page():
    """Página de reentrenamiento mejorada"""
    global modelo_compresion, modelo_exito, columnas
    
    if request.method == "POST":
        try:
            if "file" not in request.files:
                flash("❌ No se seleccionó archivo", "error")
                return redirect(url_for("retrain_page"))
            
            file = request.files["file"]
            if file.filename == "":
                flash("❌ Nombre de archivo vacío", "error")
                return redirect(url_for("retrain_page"))

            if not file.filename.lower().endswith('.csv'):
                flash("❌ Solo se permiten archivos CSV", "error")
                return redirect(url_for("retrain_page"))

            # Generar nombre único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_safe = f"{timestamp}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename_safe)
            
            # Guardar y validar archivo
            file.save(filepath)
            
            # Validar CSV antes de entrenar
            try:
                df = pd.read_csv(filepath)
                columnas_minimas = ['edad', 'grado', 'tipo_actividad']
                columnas_faltantes = [col for col in columnas_minimas if col not in df.columns]
                
                if columnas_faltantes:
                    os.remove(filepath)
                    flash(f"❌ Columnas faltantes: {columnas_faltantes}", "error")
                    return redirect(url_for("retrain_page"))

                flash(f"📁 Archivo validado: {len(df)} filas, {len(df.columns)} columnas", "info")
                
            except Exception as e:
                os.remove(filepath)
                flash(f"❌ Error leyendo CSV: {e}", "error")
                return redirect(url_for("retrain_page"))

            # Entrenar modelo
            train_model.train_model(
                ruta_csv=filepath, 
                carpeta_dataset=UPLOAD_FOLDER, 
                guardar_modelo=MODEL_PATH
            )

            # Recargar modelo
            if cargar_modelo():
                flash("✅ Modelo reentrenado correctamente", "success")
                logger.info(f"Modelo reentrenado exitosamente con {filename_safe}")
            else:
                flash("⚠️ Modelo entrenado pero falló la recarga", "warning")

            return redirect(url_for("retrain_page"))

        except Exception as e:
            logger.error(f"Error en reentrenamiento web: {e}")
            flash(f"❌ Error en reentrenamiento: {e}", "error")
            return redirect(url_for("retrain_page"))

    # Obtener estadísticas para mostrar en la página
    archivos_csv = []
    if os.path.exists(UPLOAD_FOLDER):
        archivos_csv = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    
    stats = {
        "modelo_activo": modelo_compresion is not None and modelo_exito is not None,
        "archivos_dataset": len(archivos_csv),
        "ultima_actualizacion": None
    }
    
    if os.path.exists(MODEL_PATH):
        stats["ultima_actualizacion"] = datetime.fromtimestamp(
            os.path.getmtime(MODEL_PATH)
        ).strftime("%Y-%m-%d %H:%M:%S")

    return render_template("retrain.html", stats=stats)

# ============================
# 5️⃣ Endpoints adicionales de información
# ============================
@app.route("/api/model_info", methods=["GET"])
def model_info():
    """Información detallada del modelo"""
    try:
        archivos_csv = []
        if os.path.exists(UPLOAD_FOLDER):
            archivos_csv = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]

        info = {
            "modelo_activo": modelo_compresion is not None and modelo_exito is not None,
            "timestamp": datetime.now().isoformat(),
            "archivos_dataset": len(archivos_csv),
            "tipos_actividad_soportados": ['sopa_letras', 'crucigrama', 'relacionar', 'memoria'],
            "version": "2.0.0",
            "ultima_actualizacion": None
        }
        
        if os.path.exists(MODEL_PATH):
            info["ultima_actualizacion"] = datetime.fromtimestamp(
                os.path.getmtime(MODEL_PATH)
            ).isoformat()
            info["tamaño_modelo"] = f"{os.path.getsize(MODEL_PATH) / 1024:.2f} KB"

        return jsonify(info)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """Endpoint de verificación de salud del sistema"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "modelo_disponible": modelo_compresion is not None and modelo_exito is not None,
            "version": "2.0.0",
            "uptime": "OK"
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "error": str(e)
        }), 500

# ============================
# 6️⃣ Manejo de errores mejorado
# ============================
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno del servidor: {error}")
    return render_template('500.html'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Excepción no manejada: {e}")
    return jsonify({"error": "Error interno del servidor"}), 500

# ============================
# 7️⃣ Inicialización y ejecución
# ============================
if __name__ == "__main__":
    print("=" * 60)
    print("🎓 SISTEMA DE PREDICCIÓN DE COMPORTAMIENTO ESTUDIANTIL")
    print("=" * 60)
    print(f"📅 Iniciando servidor: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Modelo cargado: {'✅ Sí' if modelo_cargado else '❌ No'}")
    print(f"📁 Carpeta de datos: {UPLOAD_FOLDER}")
    print(f"🔧 Ruta del modelo: {MODEL_PATH}")
    print("=" * 60)
    print("🌐 Endpoints disponibles:")
    print("  GET  /                    - Página principal de predicción")
    print("  POST /predict             - API de predicción (JSON)")
    print("  GET  /retrain_page        - Página de reentrenamiento") 
    print("  POST /upload_csv          - Subir archivo CSV")
    print("  POST /retrain             - Reentrenar modelo")
    print("  GET  /api/model_info      - Información del modelo")
    print("  GET  /api/health          - Estado del sistema")
    print("=" * 60)
    print("🚀 Iniciando en http://localhost:5000")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)