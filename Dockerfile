FROM python:3.11-slim

# Éviter les buffers et les prompts interactifs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    CHAINLIT_SERVER_HOST=0.0.0.0

WORKDIR /app

# Installation des dépendances système (gcc nécessaire pour ChromaDB sous Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python (cache optimisé)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Créer un utilisateur non-root (sécurité)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Créer les dossiers avec les bonnes permissions AVANT de copier le code
RUN mkdir -p /app/data /app/vectorstore /app/memory_sessions /app/logs && \
    chown -R appuser:appuser /app

# Copier le code avec les bonnes permissions
COPY --chown=appuser:appuser . .

# Passer à l'utilisateur non-root
USER appuser

EXPOSE 8000

# Healthcheck pour éviter le redémarrage en boucle silencieux
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz') || exit 1"

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]