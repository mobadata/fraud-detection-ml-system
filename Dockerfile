# Dockerfile pour l'API de détection de fraude
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]



