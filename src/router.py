import logging
from enum import Enum
from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.config import CONFIG

# Configuration du logger
logger = logging.getLogger(__name__)

class QueryType(str, Enum):
    DOCUMENT = "document"  # RAG : Protocoles, HAS, SFH, Théorie
    TOOL = "tool"          # AGENTS : Calculs, Web, Météo, Todo, Date
    CHAT = "chat"          # LLM : Salutations, aide générale

class SemanticRouter:
    """
    Routeur sémantique de grade médical.
    Précision accrue par Few-Shot Prompting pour distinguer Savoir vs Calcul.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=CONFIG.MODEL_NAME, 
            temperature=0, # Crucial pour la stabilité du format JSON
            api_key=CONFIG.OPENAI_API_KEY
        )
        
        # Structure de message "System/Human" pour une meilleure adhésion aux instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le Dispatcher de 'Hémo-Expert'. Ton rôle est de classer la requête de l'utilisateur.

LOGIQUE DE CLASSIFICATION :
1. 'document' : Recherche dans la base documentaire locale (PDF). Concerne les protocoles officiels, les recommandations SFH/HAS, les mécanismes d'action, les fiches maladies (LAL, LLC, Myélome).
2. 'tool' : Utilisation d'un outil spécifique. Concerne :
   - Les calculs (BSA, Clairance, PNN, IMC).
   - La météo ou la date.
   - La recherche web (Tavily/DuckDuckGo) pour l'actualité immédiate.
   - La gestion de la Todo List.
3. 'chat' : Interactions sociales, explications sur tes capacités ou remerciements.

RÈGLE D'OR : Si la requête contient un calcul ET une question de protocole, choisis 'tool' (l'agent gérera l'enchaînement).

Réponds exclusivement en JSON avec cette structure :
{{
    "type": "document|tool|chat",
    "confidence": 0.0-1.0,
    "reasoning": "Brève explication"
}}"""),
            ("human", "Exemples : \n- 'Bonjour' -> chat\n- 'Calcule le BSA' -> tool\n- 'Traitement LAL senior' -> document\n- 'Météo à Lille' -> tool\n\nRequête à classer : {query}")
        ])
        
        self.chain = self.prompt | self.llm | JsonOutputParser()
    
    def route(self, query: str) -> Tuple[QueryType, float, str]:
        """Analyse et route la requête avec gestion d'erreurs."""
        try:
            logger.info(f"Routing query: {query[:50]}...")
            result = self.chain.invoke({"query": query})
            
            # Extraction sécurisée
            q_type = result.get("type", "chat")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "No reason provided")

            # Fallback si confiance trop faible
            if confidence < 0.6:
                logger.warning(f"Low confidence routing ({confidence}). Defaulting to TOOL.")
                return QueryType.TOOL, confidence, "Ambiguïté détectée : redirection vers l'Agent pour analyse profonde."

            return QueryType(q_type), confidence, reasoning

        except Exception as e:
            logger.error(f"Router Error: {str(e)}")
            # Fallback de secours ultime
            return QueryType.CHAT, 0.0, f"Erreur de routage : {str(e)}"

# Singleton
router = SemanticRouter()