import os
import json
import logging
from typing import List, Dict
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.config import CONFIG

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Gestionnaire de mémoire persistante. 
    Assure la continuité du dialogue médical entre les redémarrages.
    """
    
    def __init__(self, window_size: int = 10):
        self.persist_dir = "memory_sessions"
        self.window_size = window_size
        self.memories: Dict[str, ConversationBufferWindowMemory] = {}
        
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)

    def _get_memory_instance(self, session_id: str) -> ConversationBufferWindowMemory:
        """Récupère ou initialise la mémoire en cache pour une session."""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferWindowMemory(
                k=self.window_size,
                memory_key="chat_history",
                return_messages=True
            )
            # Chargement automatique de l'historique si le fichier existe
            self._load_from_disk(session_id)
            
        return self.memories[session_id]

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """Récupère les messages formatés pour le LLM."""
        memory = self._get_memory_instance(session_id)
        return memory.load_memory_variables({})["chat_history"]

    def add_exchange(self, session_id: str, user_input: str, ai_output: str):
        """Enregistre un nouvel échange et le persiste immédiatement."""
        memory = self._get_memory_instance(session_id)
        memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )
        self._save_to_disk(session_id)

    def _save_to_disk(self, session_id: str):
        """Sérialise l'historique en JSON."""
        filepath = os.path.join(self.persist_dir, f"{session_id}.json")
        memory = self.memories[session_id]
        messages = memory.load_memory_variables({})["chat_history"]
        
        serializable_data = []
        for msg in messages:
            role = "human" if isinstance(msg, HumanMessage) else "ai"
            serializable_data.append({"role": role, "content": msg.content})
            
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde mémoire ({session_id}): {e}")

    def _load_from_disk(self, session_id: str):
        """Restaure l'historique depuis le JSON."""
        filepath = os.path.join(self.persist_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                memory = self.memories[session_id]
                for item in data:
                    if item["role"] == "human":
                        memory.chat_memory.add_user_message(item["content"])
                    else:
                        memory.chat_memory.add_ai_message(item["content"])
                logger.info(f"Mémoire restaurée pour la session {session_id}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement mémoire ({session_id}): {e}")

    def clear(self, session_id: str):
        """Efface définitivement une session (Fin de consultation)."""
        if session_id in self.memories:
            self.memories[session_id].clear()
            del self.memories[session_id]
        
        filepath = os.path.join(self.persist_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)

# Instance globale
memory_manager = MemoryManager()