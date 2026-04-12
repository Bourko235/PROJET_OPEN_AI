import chainlit as cl
from chainlit import make_async
from agent_supervisor import supervisor
from memory_manager import MemoryManager

memory_manager = MemoryManager()

@cl.on_chat_start
async def start():
    """Initialisation de la session avec un accueil professionnel"""
    await cl.Message(
        content="""👋 **Bienvenue sur Hémo-Expert Pro**

Je vous assiste dans vos décisions cliniques via :
1. 📚 **Analyse de documents** (HAS, SFH, Protocoles)
2. 🛠️ **Outils Cliniques** (Calculs de scores, dosages)
3. 💬 **Différenciation diagnostique**

*Posez votre question ci-dessous.*""",
        author="Système"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Gestion des messages avec suivi des étapes de réflexion"""
    session_id = cl.user_session.get("id")
    user_input = message.content
    
    # On crée une étape "Pensée" pour l'ergonomie
    async with cl.Step(name="Hémo-Expert réfléchit...") as step:
        try:
            # On rend l'appel au superviseur asynchrone pour ne pas geler l'UI
            # result = await make_async(supervisor.process)(user_input, session_id)
            # Si supervisor est déjà asynchrone, utilisez simplement :
            result = await supervisor.process(user_input, session_id)
            
            output = result.get("output", "Désolé, je n'ai pas pu générer de réponse.")
            
            # Gestion des sources (uniformisation sur la clé 'citations')
            citations = result.get("citations", [])
            if citations:
                sources_text = "\n\n---\n📚 **Sources consultées :**\n"
                for i, c in enumerate(citations[:3], 1):
                    sources_text += f"{i}. {c['source']} (p.{c.get('page', 'NC')})\n"
                output += sources_text

            # Badge de confiance
            route = result.get("route_type", "chat")
            conf = result.get("confidence", 0)
            route_badge = {"document": "📚 RAG", "tool": "🛠️ Outils", "chat": "💬 Chat"}.get(route, "❓")
            
            final_content = f"{output}\n\n*{route_badge} | Fiabilité : {conf:.0%}*"
            
            # Envoi de la réponse finale
            msg = cl.Message(content=final_content)
            
            # Ajout d'une action suggérée si le RAG est vide sur une question doc
            if route == "document" and not citations:
                msg.actions = [
                    cl.Action(name="web_search", value=user_input, label="🔍 Chercher sur le Web")
                ]
            
            await msg.send()
            
        except Exception as e:
            await cl.Message(content=f"❌ Erreur lors du traitement : {str(e)}").send()

@cl.action_callback("web_search")
async def on_action(action: cl.Action):
    """Recherche web de secours via Tavily"""
    from tools_extended import web_search # Assure-toi que cet outil est prêt
    
    query = action.value
    await cl.Message(content=f"🔍 *Extension de la recherche sur internet pour : {query}*").send()
    
    # Exécution de la recherche web
    result = await make_async(web_search.invoke)(query)
    await cl.Message(content=f"**Résultats du web :**\n{result}").send()