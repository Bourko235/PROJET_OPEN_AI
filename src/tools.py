import os
import math
import logging
from datetime import datetime
from langchain.tools import tool
from src.config import CONFIG

# Configuration du logging pour suivre l'utilisation des outils en console
logger = logging.getLogger(__name__)

# --- RECHERCHE & MÉTÉO ---

@tool
def web_search(query: str) -> str:
    """
    Recherche des informations médicales ou actualités à jour sur internet.
    À utiliser UNIQUEMENT si le RAG (documents internes) ne trouve pas la réponse.
    """
    try:
        # Priorité à Tavily si la clé est présente (beaucoup plus précis pour l'IA)
        if CONFIG.TAVILY_API_KEY:
            from langchain_community.tools.tavily_search import TavilySearchResults
            search = TavilySearchResults(api_key=CONFIG.TAVILY_API_KEY, k=5)
            return search.run(query)
        else:
            # Fallback sur DuckDuckGo
            from langchain_community.tools import DuckDuckGoSearchResults
            search = DuckDuckGoSearchResults()
            # On ajoute des mots clés pour filtrer les résultats non-médicaux
            return search.run(f"{query} medical oncology hematology")
    except Exception as e:
        logger.error(f"Erreur web_search: {e}")
        return f"Désolé, la recherche en ligne a échoué : {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Récupère la météo actuelle. Utile pour les conseils de santé publique."""
    if not CONFIG.OPENWEATHER_API_KEY:
        return "Clé OpenWeather non configurée."
    try:
        from langchain_community.utilities import OpenWeatherMapAPIWrapper
        weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=CONFIG.OPENWEATHER_API_KEY)
        return weather.run(city)
    except Exception as e:
        return f"Erreur météo (Vérifiez si pyowm est installé) : {str(e)}"

# --- CALCULATEURS MÉDICAUX (SÉCURISÉS) ---

@tool
def calculate_bsa(height_cm: int, weight_kg: float) -> str:
    """
    CALCULATEUR OBLIGATOIRE pour la Surface Corporelle (BSA).
    Formule de Mosteller. Ne jamais calculer manuellement.
    """
    if height_cm <= 0 or weight_kg <= 0:
        return "Erreur : Taille et poids doivent être positifs."
    
    bsa = math.sqrt((height_cm * weight_kg) / 3600)
    formula = r"$$\text{BSA} = \sqrt{\frac{\text{Poids (kg)} \times \text{Taille (cm)}}{3600}}$$"
    
    return (
        f"La Surface Corporelle (BSA) est de **{bsa:.2f} m²**.\n\n"
        f"**Rendu mathématique :**\n{formula}"
    )

@tool
def calculate_creatinine_clearance(age: int, weight_kg: float, creatinine_umol: float, is_female: bool) -> str:
    """
    CALCULATEUR OBLIGATOIRE pour la clairance de la créatinine (Cockcroft-Gault).
    Indispensable pour l'adaptation des doses de chimiothérapie.
    """
    if creatinine_umol <= 0: return "Erreur : Valeur de créatinine invalide."
    
    # Constante pour conversion µmol/L
    factor = 0.814
    clearance = ((140 - age) * weight_kg) / (factor * creatinine_umol)
    
    if is_female:
        clearance *= 0.85
    
    status = "Normale (>60)" if clearance >= 60 else "Modérée (30-60)" if clearance >= 30 else "Sévère (<30)"
    formula = r"$$\text{Cl} = \frac{(140 - \text{âge}) \times \text{Poids} \times k}{0,814 \times \text{Créat}(\mu mol/L)}$$"
    
    return (
        f"Clairance de la créatinine : **{clearance:.1f} ml/min**.\n"
        f"Interprétation : **{status}**.\n\n"
        f"**Formule utilisée :**\n{formula}"
    )

# --- GESTION DES TÂCHES ---

@tool
def save_to_todo(task: str, priority: str = "normal") -> str:
    """Sauvegarde une note clinique ou une tâche de suivi dans la todo list."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open("todo_list.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{priority.upper()}] {task}\n")
        return f"✅ Tâche enregistrée avec succès : {task}"
    except Exception as e:
        return f"Erreur d'écriture fichier : {str(e)}"

@tool
def read_todo() -> str:
    """Lit l'intégralité de la todo list du praticien."""
    if not os.path.exists("todo_list.txt"):
        return "Aucune tâche enregistrée pour le moment."
    try:
        with open("todo_list.txt", "r", encoding="utf-8") as f:
            content = f.read()
        return f"📋 **Liste des rappels :**\n\n{content}" if content else "La liste est vide."
    except Exception as e:
        return f"Erreur de lecture : {str(e)}"

# --- UTILITAIRES ---

@tool
def get_current_date() -> str:
    """Donne la date et l'heure système."""
    return f"Nous sommes le {datetime.now().strftime('%d/%m/%Y')} et il est {datetime.now().strftime('%H:%M')}."

# Liste des outils pour l'agent
tools = [
    web_search, 
    get_weather, 
    calculate_bsa, 
    calculate_creatinine_clearance, 
    save_to_todo, 
    read_todo, 
    get_current_date
]