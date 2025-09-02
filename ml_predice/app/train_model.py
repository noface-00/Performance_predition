import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

df = pd.read_csv("/Users/km/Documents/VSPROJ/ML_Predidtion_rend/Performance_predition/datos/datos_actividades_1000.csv")

# ============================
# 2. Preprocesamiento
# ============================
le = LabelEncoder()
df["tipo_actividad"] = le.fit_transform(df["tipo_actividad"])

# Crear variables dummy (binarias)
X = pd.get_dummies(df.drop("nivel_comprension", axis=1),  # eliminamos target
                   columns=["tipo_actividad"],            # columna a binarizar
                   drop_first=True)                        # evita multicolinealidad

# Target
y = df["nivel_comprension"]


# División entrenamiento-prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar Random Forest
modelo_compresion = RandomForestRegressor(n_estimators=300, random_state=42)
modelo_compresion.fit(X_train, y_train)

# Evaluar R²
r2 = modelo_compresion.score(X_test, y_test)
print("R²:", r2)

# Creamos la variable "exito"
df["exito"] = ((df["nota"] >= 7) & (df["correcto"] == 1)).astype(int)
mask = np.random.rand(len(df)) < 0.05
df.loc[mask, "exito"] = 1 - df.loc[mask, "exito"]
# Convertir 'tipo_actividad' en variables binarias
X = pd.get_dummies(df.drop("exito", axis=1), columns=["tipo_actividad"], drop_first=True)
y = df["exito"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_exito = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
modelo_exito.fit(X_train, y_train)

# Probabilidad de éxito
probs = modelo_exito.predict_proba(X_test)[:,1]
print("Ejemplo de probabilidad de éxito:", probs[:5])

# Predicciones binarias
y_pred = modelo_exito.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC ROC:", roc_auc_score(y_test, probs))

def recomendar_actividad_dict(datos_estudiante, X, modelo_compresion, modelo_exito):
    nuevo_estudiante = pd.DataFrame([datos_estudiante])

    # Binarizar tipo_actividad
    if "tipo_actividad" in nuevo_estudiante.columns:
        nuevo_estudiante = pd.get_dummies(nuevo_estudiante, columns=["tipo_actividad"], drop_first=True)
    
    # Separar columnas para modelo_compresion (sin nivel_comprension)
    X_comp_modelo = X.drop(columns=["nivel_comprension"], errors='ignore')
    for col in X_comp_modelo.columns:
        if col not in nuevo_estudiante.columns:
            nuevo_estudiante[col] = 0
    nuevo_estudiante_comp = nuevo_estudiante[X_comp_modelo.columns]
    
    # Predicción comprensión
    comprension = modelo_compresion.predict(nuevo_estudiante_comp)[0]

    # Agregar nivel_comprension al DataFrame antes de predecir éxito
    nuevo_estudiante["nivel_comprension"] = comprension

    # Separar columnas para modelo_exito (X usado en entrenamiento de exito)
    X_exito_modelo = X.drop(columns=["exito"], errors='ignore')
    for col in X_exito_modelo.columns:
        if col not in nuevo_estudiante.columns:
            nuevo_estudiante[col] = 0
    nuevo_estudiante_exito = nuevo_estudiante[X_exito_modelo.columns]

    # Predicción probabilidad de éxito
    prob_exito = modelo_exito.predict_proba(nuevo_estudiante_exito)[:,1][0]

    # Recomendaciones
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
        "nivel_comprension": comprension,
        "prob_exito": prob_exito,
        "recomendaciones": recomendaciones
    }


# Guardar los dos modelos y las columnas de entrenamiento
artefacto = {
    "modelo_compresion": modelo_compresion,
    "modelo_exito": modelo_exito,
    "columnas": X.columns.tolist()
}

joblib.dump(artefacto, "/Users/km/Documents/VSPROJ/ML_Predidtion_rend/Performance_predition/ml_predice/app/model.pkl")
print("✅ Modelos guardados en model.pkl")