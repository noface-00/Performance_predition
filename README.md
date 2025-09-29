# Predicción de Comportamiento en Tareas de Estudiantes  

#### Video Demo:  <https://youtu.be/d7YW9oBosL8>
#### Descricion:
Este proyecto implementa un sistema de **machine learning** para predecir el comportamiento de estudiantes en la realización de tareas. El objetivo es ayudar a docentes y administradores académicos a identificar patrones de estudio, prever el cumplimiento de entregas y detectar estudiantes en riesgo de bajo rendimiento.  

---

## Estructura de proyecto
├── app/
│ ├── datos/ # Datasets para entrenamiento y prueba
│ ├── static/ # Archivos estáticos (CSS, JS)
│ │ ├── main.js
│ │ ├── style_main.css
│ │ └── style_retrain.css
│ ├── templates/ # Vistas HTML (Flask)
│ │ ├── index.html
│ │ └── retrain.html
│ ├── main.py # Script principal - servidor Flask
│ └── train_model.py # Script para entrenar el modelo
├── model.pkl # Modelo entrenado
├── requirements.txt # Dependencias del proyecto
└── README.md # Documentación

---

## Descripcion

El sistema cuenta con dos principales componentes:

**Backend (Flask + ML)**  
   - Entrena y guarda un modelo de predicción en `train_model.py`.  
   - Expone un servidor **Flask** local  en `main.py` con endpoints para cargar datos y obtener predicciones.

**Frontend (HTML + CSS + JS)** 
   - `index.html`: Interfaz principal donde el usuario ingresa datos de estudiantes.  
   - `retrain.html`: Interfaz para reentrenar el modelo desde la web.  
   - Archivos estáticos (`main.js`, `style_main.css`) para manejo de la lógica y estilo.


## Objetivos

- Predecir la probabilidad de que un estudiante entregue o no una tarea.  
- Mostrar los resultados en una **interfaz web amigable**.  
- Convertir en una API funcional para otros sistemas
- Permitir reentrenar el modelo desde la propia aplicación. 

## Tecnologías Utilizadas  

- **Python 3.x**  
- **Flask** → Backend y servidor web  
- **Scikit-learn** → Modelado y entrenamiento de ML  
- **Pandas / NumPy** → Manejo y procesamiento de datos  
- **Joblib** → Guardado y carga del modelo  
- **HTML / CSS / JavaScript** → Interfaz de usuario  

## Instalación  

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/noface-00/Performance_predition.git

   cd prediccion-estudiantes
2. **Crear un entorno:**
    python -m venv venv
    source venv/bin/activate   # En Linux/Mac
    venv\Scripts\activate      # En Windows

    pip install -r requirements.txt

