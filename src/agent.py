import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.config import CONFIG 
from src.tools import tools 
from src.query_engine import rag_engine
from langchain.tools import tool

# Configuration du logging
logger = logging.getLogger(__name__)

# --- WRAPPER DE L'OUTIL RAG ---
@tool
def medical_knowledge_retrieval(query: str) -> str:
    """
    RECHERCHE DOCUMENTAIRE : Utilise cet outil pour consulter les protocoles internes, 
    les recommandations de la HAS, de la SFH et toute théorie médicale validée.
    """
    logger.info(f"Appel RAG avec la requête : {query}")
    result = rag_engine.query(query)
    return result["answer"]

class HemoAgent:
    """
    Agent spécialisé 'Hémo-Expert'. 
    Capacité d'enchaînement d'outils (Calcul -> Recherche -> Rapport).
    """
    
    def __init__(self):
        # 1. Modèle paramétré via CONFIG
        self.llm = ChatOpenAI(
            model=CONFIG.MODEL_NAME, 
            temperature=CONFIG.TEMPERATURE, 
            api_key=CONFIG.OPENAI_API_KEY
        )

        # 2. Agrégation des outils
        self.all_tools = tools + [medical_knowledge_retrieval]

        # 3. Prompt Systémique de Grade Médical
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es 'Hémo-Expert', l'assistant d'aide à la décision clinique.
            
            DIRECTIVES CRUCIALES :
            - CALCULS : N'effectue JAMAIS de calcul mental. Utilise les outils de calcul dédiés. 
            - SOURCE : Si une information provient d'un document, cite toujours [Source: Nom, Page].
            - CHAÎNAGE : Tu peux utiliser plusieurs outils pour une seule réponse (ex: calculer un BSA puis chercher une dose).
            - ABSENCE D'INFO : Si un outil ne donne rien, précise-le et propose une alternative (web_search).
            - UNITÉS : Sois rigoureux sur les unités (m², ml/min, µmol/L)."""),
            
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 4. Initialisation de l'Agent
        agent = create_openai_tools_agent(self.llm, self.all_tools, self.prompt)
        
        # 5. Exécuteur de l'Agent
        self.executor = AgentExecutor(
            agent=agent, 
            tools=self.all_tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=CONFIG.MAX_ITERATIONS
        )

    def run(self, user_input: str, history: list) -> dict:
        """
        Exécute l'agent avec l'historique fourni par le MemoryManager.
        """
        try:
            response = self.executor.invoke({
                "input": user_input,
                "chat_history": history
            })
            return response
        except Exception as e:
            logger.error(f"Erreur AgentExecutor : {str(e)}")
            return {"output": f"Désolé, une erreur technique est survenue lors de l'exécution : {str(e)}"}

# Instance unique (Singleton)
hemo_agent = HemoAgent()