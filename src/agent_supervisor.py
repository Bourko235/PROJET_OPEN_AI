import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# Imports internes
from src.config import CONFIG
from src.router import router, QueryType
from src.query_engine import rag_engine
from src.tools import tools
from src.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """
    Orchestrateur central 'Hémo-Expert'.
    Décide du meilleur chemin (RAG, Outils, ou Chat) pour répondre avec précision.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=CONFIG.MODEL_NAME,
            temperature=0,  # Zéro créativité pour une sécurité maximale
            api_key=CONFIG.OPENAI_API_KEY
        )
        self.router = router
        # L'agent est créé une seule fois à l'initialisation
        self.tools_agent = self._create_tools_agent()
        
    def _create_tools_agent(self):
        """Configure l'agent d'exécution des tâches complexes."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le module 'Action' de Hémo-Expert. 
            Ton rôle est d'exécuter des calculs cliniques ou des recherches web.
            
            RÈGLES DE CONDUITE :
            1. CALCULS : Utilise TOUJOURS les outils. Ne fais jamais de calcul mental.
            2. FORMULES : Affiche systématiquement la formule LaTeX utilisée.
            3. PRÉCISION : Si une donnée manque (ex: poids), demande-la explicitement.
            4. WEB : Synthétise les infos web en restant factuel et médical."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=CONFIG.MAX_ITERATIONS
        )
    
    async def process(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Point d'entrée principal. Gère le cycle de vie d'une requête.
        """
        # 0. Récupération de l'historique
        chat_history = memory_manager.get_messages(session_id)
        
        # 1. Routage intelligent
        query_type, confidence, reasoning = self.router.route(user_input)
        logger.info(f"Route choisie : {query_type} ({confidence*100:.0f}%) | Raison : {reasoning}")
        
        response_data = {
            "input": user_input,
            "route_type": query_type.value,
            "confidence": confidence,
            "output": "",
            "citations": []
        }
        
        try:
            # --- CAS A : SAVOIR MÉDICAL (RAG) ---
            if query_type == QueryType.DOCUMENT:
                # Le RAG est synchrone dans notre implémentation actuelle
                rag_result = rag_engine.query(user_input)
                response_data["output"] = rag_result["answer"]
                response_data["citations"] = rag_result.get("citations", [])

            # --- CAS B : ACTION OU CALCUL (TOOLS) ---
            elif query_type == QueryType.TOOL:
                # On utilise ainvoke pour l'asynchronisme de l'agent
                result = await self.tools_agent.ainvoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                response_data["output"] = result["output"]

            # --- CAS C : DIALOGUE GÉNÉRAL (CHAT) ---
            else:
                messages = [
                    SystemMessage(content="Tu es Hémo-Expert, assistant médical. Réponds de façon concise et polie."),
                    *chat_history,
                    HumanMessage(content=user_input)
                ]
                response = await self.llm.ainvoke(messages)
                response_data["output"] = response.content
        
        except Exception as e:
            logger.error(f"Erreur Supervisor : {str(e)}")
            response_data["output"] = f"Désolé, une erreur technique a perturbé l'analyse : {str(e)}"
            response_data["route_type"] = "error"
        
        # 3. Mise à jour de la mémoire persistante
        memory_manager.add_exchange(session_id, user_input, response_data["output"])
        
        return response_data

# Singleton pour l'import dans app.py
supervisor = SupervisorAgent()