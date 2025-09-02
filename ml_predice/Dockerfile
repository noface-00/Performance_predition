# 1️⃣ Imagen base
FROM python:3.13-slim

# 2️⃣ Crear carpeta de trabajo
WORKDIR /app

# 3️⃣ Copiar requirements y código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app

# 4️⃣ Crear carpeta para CSVs subidos
RUN mkdir -p /app/datos

# 5️⃣ Exponer puerto Flask
EXPOSE 5000

# 6️⃣ Entrypoint
CMD ["python", "app/main.py"]
