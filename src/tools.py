import os
import math
import logging
from datetime import datetime
from langchain.tools import tool
from src.config import CONFIG

logger = logging.getLogger(__name__)

# --- RECHERCHE & MÉTÉO ---

@tool
def web_search(query: str) -> str:
    """
    Recherche des informations médicales ou actualités à jour sur internet.
    À utiliser UNIQUEMENT si le RAG (documents internes) ne trouve pas la réponse.
    """
    try:
        if CONFIG.TAVILY_API_KEY:
            from langchain_community.tools.tavily_search import TavilySearchResults
            search = TavilySearchResults(api_key=CONFIG.TAVILY_API_KEY, k=5)
            return search.run(query)
        else:
            from langchain_community.tools import DuckDuckGoSearchResults
            search = DuckDuckGoSearchResults()
            return search.run(f"{query} medical oncology hematology")
    except Exception as e:
        logger.error(f"Erreur web_search: {e}")
        return f"❌ Erreur recherche web : {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Récupère la météo actuelle."""
    if not CONFIG.OPENWEATHER_API_KEY:
        return "⚠️ Clé OpenWeather non configurée."
    try:
        from langchain_community.utilities import OpenWeatherMapAPIWrapper
        weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=CONFIG.OPENWEATHER_API_KEY)
        return weather.run(city)
    except Exception as e:
        return f"❌ Erreur météo : {str(e)}"

# --- CALCULATEURS MÉDICAUX AMÉLIORÉS ---

@tool
def calculate_bsa(height_cm: int, weight_kg: float) -> str:
    """
    Calcule la Surface Corporelle (BSA) - Formule de Mosteller.
    Essentiel pour le dosage des chimiothérapies (mg/m²).
    
    Args:
        height_cm: Taille en centimètres (ex: 175)
        weight_kg: Poids en kilogrammes (ex: 70)
    """
    if height_cm <= 0 or weight_kg <= 0:
        return "⚠️ Erreur : Les valeurs doivent être strictement positives."
    
    bsa = math.sqrt((height_cm * weight_kg) / 3600)
    
    # Format optimisé pour Chainlit avec rendu LaTeX
    return f"""## 🧮 Surface Corporelle (BSA)

**Résultat :** `{bsa:.2f} m²`

---

### 📐 Formule de Mosteller
```
BSA (m²) = √[(Taille(cm) × Poids(kg)) / 3600]
```

**En LaTeX :** $BSA = \\sqrt{{\\frac{{{height_cm} \\times {weight_kg}}}{{3600}}}}$

---

### 🔢 Détail du calcul
- Taille : **{height_cm} cm**
- Poids : **{weight_kg} kg**
- Calcul : √({height_cm} × {weight_kg} / 3600)
- **Résultat : {bsa:.2f} m²**

> 💡 *Référence : Mosteller RD. Simplified calculation of body-surface area. N Engl J Med. 1987*
"""

@tool
def calculate_creatinine_clearance(age: int, weight_kg: float, creatinine_umol: float, is_female: bool) -> str:
    """
    Calcule la Clairance de la Créatinine (Cockcroft-Gault).
    Crucial pour l'ajustement des doses de chimiothérapie et la fonction rénale.
    
    Args:
        age: Âge en années
        weight_kg: Poids en kg
        creatinine_umol: Créatinine sérique en µmol/L
        is_female: True si patient féminin, False si masculin
    """
    if creatinine_umol <= 0 or age <= 0 or weight_kg <= 0:
        return "⚠️ Erreur : Toutes les valeurs doivent être positives."
    
    if age > 120:
        return "⚠️ Erreur : Vérifiez l'âge saisi (semble incohérent)."
    
    # Calcul
    clearance = ((140 - age) * weight_kg) / (0.814 * creatinine_umol)
    gender_factor = 0.85 if is_female else 1.0
    clearance_adj = clearance * gender_factor
    
    # Interprétation
    if clearance_adj >= 60:
        interpretation = "✅ Fonction rénale normale"
        color = "green"
    elif clearance_adj >= 30:
        interpretation = "⚠️ Insuffisance rénale modérée (Stade 3)"
        color = "orange"
    else:
        interpretation = "🚨 Insuffisance rénale sévère (Stade 4-5)"
        color = "red"
    
    gender_text = "Féminin (×0.85)" if is_female else "Masculin (×1.0)"
    
    return f"""## 🩺 Clairance de la Créatinine (Cockcroft-Gault)

**Résultat :** `{clearance_adj:.1f} mL/min`

**Interprétation :** {interpretation}

---

### 📐 Formule utilisée
```
Clairance = [(140 - Âge) × Poids] / (0,814 × Créatinine) × K
```

**LaTeX :** $Cl = \\frac{{(140 - {age}) \\times {weight_kg}}}{{0.814 \\times {creatinine_umol}}} \\times {gender_factor}$

---

### 🔢 Données de calcul
| Paramètre | Valeur |
|-----------|---------|
| **Âge** | {age} ans |
| **Poids** | {weight_kg} kg |
| **Créatinine** | {creatinine_umol} µmol/L |
| **Sexe** | {gender_text} |
| **Clairance brute** | {clearance:.1f} mL/min |
| **Clairance ajustée** | **{clearance_adj:.1f} mL/min** |

---

### 📊 Classification (MDRD)
- **> 60 mL/min** : Fonction normale
- **30-59 mL/min** : Insuffisance modérée
- **< 30 mL/min** : Insuffisance sévère

> ⚠️ *Cette formule est indicative. Ajustez les doses selon les recommandations du produit et l'état clinique.*
"""

@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """
    Calcule l'Indice de Masse Corporelle (IMC).
    Permet d'évaluer le statut nutritionnel du patient.
    """
    if height_cm <= 0 or weight_kg <= 0:
        return "⚠️ Erreur : Valeurs positives requises."
    
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    # Classification OMS
    if bmi < 18.5:
        status = "Maigreur"
        emoji = "🔵"
    elif bmi < 25:
        status = "Corpulence normale"
        emoji = "🟢"
    elif bmi < 30:
        status = "Surpoids"
        emoji = "🟡"
    else:
        status = "Obésité"
        emoji = "🔴"
    
    return f"""## ⚖️ Indice de Masse Corporelle (IMC)

**Résultat :** `{bmi:.1f} kg/m²`

**Classification :** {emoji} {status}

---

### 📐 Formule
$IMC = \\frac{{\\text{{Poids (kg)}}}}{{\\text{{Taille (m)}}^2}} = \\frac{{{weight_kg}}}{{({height_cm/100})^2}}$

---

### 🔢 Détail
- Poids : **{weight_kg} kg**
- Taille : **{height_cm} cm** ({height_m:.2f} m)
- IMC : **{bmi:.1f}**

### 📊 Référence OMS
| Catégorie | IMC |
|-----------|-----|
| 🔵 Maigreur | < 18,5 |
| 🟢 Normal | 18,5 - 24,9 |
| 🟡 Surpoids | 25,0 - 29,9 |
| 🔴 Obésité | ≥ 30,0 |
"""

# --- GESTION DES TÂCHES ---

@tool
def save_to_todo(task: str, priority: str = "normal") -> str:
    """Sauvegarde une note clinique dans la todo list."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        priority_emoji = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(priority.lower(), "⚪")
        
        with open("todo_list.txt", "a", encoding="utf-8") as f:
            f.write(f"{priority_emoji} [{timestamp}] [{priority.upper()}] {task}\n")
            
        return f"✅ Tâche enregistrée : {task}"
    except Exception as e:
        return f"❌ Erreur : {str(e)}"

@tool
def read_todo() -> str:
    """Lit la todo list."""
    if not os.path.exists("todo_list.txt"):
        return "📋 Aucune tâche enregistrée."
    try:
        with open("todo_list.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            return "📋 La liste est vide."
            
        return f"""## 📋 Todo List Médicale

{content}

---
*Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}*
"""
    except Exception as e:
        return f"❌ Erreur de lecture : {str(e)}"

@tool
def get_current_date() -> str:
    """Donne la date et l'heure actuelles."""
    now = datetime.now()
    return f"📅 Date : {now.strftime('%d/%m/%Y')}\n🕐 Heure : {now.strftime('%H:%M')}"

# Liste des outils pour l'agent
tools = [
    web_search, 
    get_weather, 
    calculate_bsa, 
    calculate_creatinine_clearance,
    calculate_bmi,  
    save_to_todo, 
    read_todo, 
    get_current_date
]

