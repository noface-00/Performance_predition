import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for student performance prediction
n_samples = 1000

# Create synthetic features
data = {
    'tiempo_actividad': np.random.normal(30, 10, n_samples),  # Average 30 minutes
    'pistas_usadas': np.random.poisson(3, n_samples),  # Average 3 hints used
    'intentos': np.random.poisson(2, n_samples) + 1,  # At least 1 attempt
    'dificultad': np.random.choice([1, 2, 3, 4, 5], n_samples),  # Difficulty 1-5
    'tipo_actividad_ejercicio': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'tipo_actividad_quiz': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'tipo_actividad_proyecto': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
}

df = pd.DataFrame(data)

# Generate target variables
# Nivel de comprensión (0-1, continuous)
df['nivel_comprension'] = np.clip(
    0.8 - df['intentos'] * 0.1 + 
    df['tiempo_actividad'] * 0.01 - 
    df['pistas_usadas'] * 0.05 + 
    df['tipo_actividad_proyecto'] * 0.2 +
    np.random.normal(0, 0.1, n_samples),
    0, 1
)

# Éxito (binary: 0 or 1)
prob_exito = (
    df['nivel_comprension'] * 0.8 +
    (5 - df['dificultad']) * 0.1 +
    df['tipo_actividad_ejercicio'] * 0.1
)
df['exito'] = np.random.binomial(1, np.clip(prob_exito, 0, 1), n_samples)

# Define columns for training (excluding target variables)
columnas = [col for col in df.columns if col not in ['nivel_comprension', 'exito']]

# Prepare features for compression model (without nivel_comprension)
X_comprension = df[columnas]
y_comprension = df['nivel_comprension']

# Prepare features for success model (including nivel_comprension)
X_exito = df[columnas + ['nivel_comprension']]
y_exito = df['exito']

# Split data for compression model
X_comp_train, X_comp_test, y_comp_train, y_comp_test = train_test_split(
    X_comprension, y_comprension, test_size=0.2, random_state=42
)

# Split data for success model
X_exit_train, X_exit_test, y_exit_train, y_exit_test = train_test_split(
    X_exito, y_exito, test_size=0.2, random_state=42
)

# Train models
print("Training compression model...")
modelo_compresion = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_compresion.fit(X_comp_train, y_comp_train)

print("Training success model...")
modelo_exito = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_exito.fit(X_exit_train, y_exit_train)

# Create artifact dictionary
artefacto = {
    "modelo_compresion": modelo_compresion,
    "modelo_exito": modelo_exito,
    "columnas": columnas + ['nivel_comprension']  # Include nivel_comprension for success model
}

# Save the artifact
joblib.dump(artefacto, 'app/model.pkl')

print("Model artifact created and saved successfully!")
print(f"Compression model R²: {modelo_compresion.score(X_comp_test, y_comp_test):.3f}")
print(f"Success model accuracy: {modelo_exito.score(X_exit_test, y_exit_test):.3f}")
print(f"Columns: {columnas}")
print(f"Model file size: {len(open('app/model.pkl', 'rb').read())} bytes")
