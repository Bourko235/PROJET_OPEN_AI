import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import CONFIG
from src.router import router, QueryType
from src.query_engine import rag_engine
from src.tools import tools
from src.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class SupervisorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=CONFIG.MODEL_NAME,
            temperature=0,
            api_key=CONFIG.OPENAI_API_KEY
        )
        self.router = router
        self.tools_agent = self._create_tools_agent()
        
    def _create_tools_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le module 'Action' de Hémo-Expert..."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # CORRECTION ICI : create_tool_calling_agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=CONFIG.MAX_ITERATIONS
        )
    
    async def process(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        chat_history = memory_manager.get_messages(session_id)
        query_type, confidence, reasoning = self.router.route(user_input)
        
        response_data = {
            "input": user_input,
            "route_type": query_type.value,
            "confidence": confidence,
            "output": "",
            "citations": []
        }
        
        try:
            if query_type == QueryType.DOCUMENT:
                rag_result = rag_engine.query(user_input)
                response_data["output"] = rag_result["answer"]
                response_data["citations"] = rag_result.get("citations", [])

            elif query_type == QueryType.TOOL:
                result = await self.tools_agent.ainvoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                response_data["output"] = result["output"]

            else:
                messages = [
                    SystemMessage(content="Tu es Hémo-Expert..."),
                    *chat_history,
                    HumanMessage(content=user_input)
                ]
                response = await self.llm.ainvoke(messages)
                response_data["output"] = response.content
        
        except Exception as e:
            logger.error(f"Erreur : {str(e)}")
            response_data["output"] = f"Erreur technique : {str(e)}"
            response_data["route_type"] = "error"
        
        memory_manager.add_exchange(session_id, user_input, response_data["output"])
        return response_data

supervisor = SupervisorAgent()