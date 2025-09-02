import pandas as pd
import numpy as np
import random

# Definir tipos de actividad
tipos_actividad = ["sopa_letras", "crucigrama", "relacionar", "memoria"]

# Crear 1000 registros simulados
data = []
for _ in range(1000):
    registro = {
        "edad": random.randint(9, 12),
        "grado": random.randint(4, 6),
        "tipo_actividad": random.choice(tipos_actividad),
        "dificultad": random.randint(1, 5),
        "tiempo_seg": random.randint(80, 300),
        "intentos": random.randint(1, 5),
        "pistas": random.randint(0, 3),
        "correcto": random.randint(0, 1),
        "nota": random.randint(5, 10),
        "secuencia_actividades": random.randint(1, 5),
        "evolucion_desempeno": round(random.uniform(-0.2, 0.5), 2),
        "nivel_concentracion": round(random.uniform(0.5, 1.0), 2),
        "comparacion_historial": round(random.uniform(-0.2, 0.3), 2)
    }
    data.append(registro)

df = pd.DataFrame(data)

# Crear 'nivel_comprension' como combinación de otras variables
noise = np.random.normal(0, 0.05, size=len(df))
df["nivel_comprension"] = (
    0.25*df["correcto"] + 
    0.2*df["nivel_concentracion"] + 
    0.15*df["nota"]/10 + 
    0.1*df["dificultad"]/5 + 
    0.1*df["evolucion_desempeno"] +
    noise
)
# Asegurar valores entre 0 y 1
df["nivel_comprension"] = df["nivel_comprension"].clip(0, 1)

# Guardar CSV
file_path = "/Users/km/Downloads/prueba_flask1000.csv"
df.to_csv(file_path, index=False)
file_path