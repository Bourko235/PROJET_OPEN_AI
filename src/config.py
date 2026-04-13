import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# On charge le .env une seule fois au sommet
load_dotenv()

@dataclass(frozen=True) # frozen=True empêche de modifier la config par erreur
class Config:
    """Configuration centralisée pour Hémo-Expert"""
    
    # --- API KEYS ---
    # On utilise os.getenv avec une valeur par défaut None ou une chaîne vide
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    
    # --- MODÈLES ---
    # On permet l'override via le .env sinon on prend le défaut
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.0)) # 0.0 pour le médical
    
    # --- CHEMINS ---
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH: str = os.path.join(BASE_DIR, "data")
    VECTORSTORE_PATH: str = os.path.join(BASE_DIR, "vectorstore")
    
    # --- RAG SETTINGS ---
    # Chunk size de 1100 est bon pour capturer des paragraphes médicaux entiers
    CHUNK_SIZE: int = 1100
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    
    # --- AGENT SETTINGS ---
    MAX_ITERATIONS: int = 5

    @classmethod
    def load(cls):
        """Valide et initialise la configuration"""
        instance = cls()
        
        # Validation critique
        if not instance.OPENAI_API_KEY:
            print("❌ ERREUR FATALE : OPENAI_API_KEY manquante dans le fichier .env")
            raise ValueError("Une clé API OpenAI est requise pour démarrer l'assistant.")
            
        # Avertissements non-bloquants
        if not instance.TAVILY_API_KEY:
            print("⚠️ WARNING : TAVILY_API_KEY absente. La recherche web sera désactivée.")
            
        return instance

# Singleton utilisable partout dans le projet
CONFIG = Config.load()