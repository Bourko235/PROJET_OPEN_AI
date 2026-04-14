# Utilisation d'une image Python légère
FROM python:3.12-slim

# Éviter la génération de fichiers .pyc et forcer l'affichage des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définition du dossier de travail
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de l'intégralité du projet
COPY . .

# Création des dossiers pour la persistance des données
RUN mkdir -p data vectorstore memory_sessions

# Exposition du port utilisé par Chainlit
EXPOSE 8002

# Commande par défaut : Lancer l'interface Web
CMD ["python", "main.py", "--web", "--port", "8002"]