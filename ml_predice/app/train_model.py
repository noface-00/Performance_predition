import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_model(ruta_csv=None, carpeta_dataset="datos", guardar_modelo="app/model.pkl"):
    """
    Entrena modelos usando todos los CSV en carpeta_dataset + ruta_csv opcional.
    """
    # 1️⃣ Crear carpeta si no existe
    os.makedirs(carpeta_dataset, exist_ok=True)

    # 2️⃣ Guardar CSV nuevo si se pasa
    if ruta_csv:
        nombre = os.path.basename(ruta_csv)
        destino = os.path.join(carpeta_dataset, nombre)
        os.rename(ruta_csv, destino)
        print(f"Archivo {nombre} agregado a {carpeta_dataset}")

    # 3️⃣ Cargar todos los CSV acumulados
    archivos = [os.path.join(carpeta_dataset, f) for f in os.listdir(carpeta_dataset) if f.endswith(".csv")]
    if not archivos:
        raise FileNotFoundError("No hay archivos CSV en la carpeta de dataset.")

    dfs = [pd.read_csv(f) for f in archivos]
    df_total = pd.concat(dfs, ignore_index=True)
    print(f"Entrenando con {len(df_total)} filas de {len(archivos)} archivos")

    # 4️⃣ Preprocesamiento
    le = LabelEncoder()
    df_total["tipo_actividad"] = le.fit_transform(df_total["tipo_actividad"])

    # ===== Modelo nivel_comprension =====
    X_comp = pd.get_dummies(df_total.drop("nivel_comprension", axis=1), columns=["tipo_actividad"], drop_first=True)
    y_comp = df_total["nivel_comprension"]

    X_train, X_test, y_train, y_test = train_test_split(X_comp, y_comp, test_size=0.2, random_state=42)
    modelo_compresion = RandomForestRegressor(n_estimators=300, random_state=42)
    modelo_compresion.fit(X_train, y_train)
    print("R² nivel_comprension:", modelo_compresion.score(X_test, y_test))

    # ===== Modelo exito =====
    df_total["exito"] = ((df_total["nota"] >= 7) & (df_total["correcto"] == 1)).astype(int)
    mask = np.random.rand(len(df_total)) < 0.05
    df_total.loc[mask, "exito"] = 1 - df_total.loc[mask, "exito"]

    X_exito = pd.get_dummies(df_total.drop("exito", axis=1), columns=["tipo_actividad"], drop_first=True)
    y_exito = df_total["exito"]

    X_train, X_test, y_train, y_test = train_test_split(X_exito, y_exito, test_size=0.2, random_state=42)
    modelo_exito = LogisticRegression(max_iter=1000, solver='liblinear')
    modelo_exito.fit(X_train, y_train)
    probs = modelo_exito.predict_proba(X_test)[:,1]
    y_pred = modelo_exito.predict(X_test)
    from sklearn.metrics import accuracy_score, roc_auc_score
    print("Accuracy exito:", accuracy_score(y_test, y_pred))
    print("AUC ROC exito:", roc_auc_score(y_test, probs))

    # 5️⃣ Guardar modelo
    os.makedirs(os.path.dirname(guardar_modelo), exist_ok=True)
    # ⚡ Opcional: guardar copia de seguridad antes de sobrescribir
    if os.path.exists(guardar_modelo):
        import shutil, time
        timestamp = int(time.time())
        shutil.copy(guardar_modelo, f"{guardar_modelo}_{timestamp}.bak")
    artefacto = {
        "modelo_compresion": modelo_compresion,
        "modelo_exito": modelo_exito,
        "columnas": X_exito.columns.tolist()
    }
    joblib.dump(artefacto, guardar_modelo)
    print(f"✅ Modelos guardados en {guardar_modelo}")
