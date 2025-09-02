# Imagen base
FROM python:3.13.7

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY app/ ./app/

# Exponer puerto
EXPOSE 5000

# Comando para correr el servicio
CMD ["python", "app/main.py"]
