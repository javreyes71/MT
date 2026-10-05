FROM python:3.10-slim

# Instalar SUMO y dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    sumo sumo-tools \
    && rm -rf /var/lib/apt/lists/*

ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Crear directorios necesarios
RUN mkdir -p models results tensorboard

# Comando por defecto
CMD ["python", "scripts/train.py"]
