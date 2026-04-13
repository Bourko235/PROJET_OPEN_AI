import chainlit as cl
import asyncio
from src.agent_supervisor import supervisor
from src.tools import web_search

@cl.on_chat_start
async def start():
    """Initialisation de la session avec un accueil professionnel"""
    # Dans Chainlit, l'ID de session est accessible via cl.user_session.get("id")
    session_id = cl.context.session.id 
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content="""👋 **Bienvenue sur Hémo-Assist Pro**

Je vous assiste dans vos décisions cliniques via :
1. 📚 **Analyse de documents** (HAS, SFH, Protocoles)
2. 🛠️ **Outils Cliniques** (Calculs de scores, dosages)
3. 💬 **Différenciation diagnostique**

*Posez votre question ci-dessous.*""",
        author="Système"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Gestion des messages avec exécution asynchrone"""
    session_id = cl.user_session.get("session_id")
    user_input = message.content
    
    # On utilise cl.Step pour montrer que l'IA "réfléchit"
    async with cl.Step(name="Hémo-Expert analyse la requête...") as step:
        try:
            # (Si supervisor.process est déjà async, utilise simplement 'await')
            result = await supervisor.process(user_input, session_id)
            
            output = result.get("output", "Erreur de génération")
            citations = result.get("citations", [])
            
            # Construction du bloc de sources
            if citations:
                sources_text = "\n\n---\n📚 **Sources consultées:**\n"
                for i, c in enumerate(citations[:3], 1):
                    sources_text += f"{i}. {c['source']} (p.{c.get('page', 'NC')})\n"
                output += sources_text
            
            # Badge de route
            route_map = {"document": "📚 RAG", "tool": "🛠️ Outils", "chat": "💬 Chat"}
            badge = route_map.get(result.get("route_type"), "❓")
            confidence = result.get("confidence", 0)
            
            final_content = f"{output}\n\n*{badge} (confiance: {confidence:.0%})*"
            
            # Envoi de la réponse
            res_msg = cl.Message(content=final_content)
            
            # ✅ On vérifie la même clé 'citations' pour le bouton web
            if result.get("route_type") == "document" and not citations:
                res_msg.actions = [
                    cl.Action(
                        name="web_search", 
                        value=user_input, 
                        label="🔍 Rechercher en ligne"
                    )
                ]
            
            await res_msg.send()

        except Exception as e:
            await cl.Message(content=f"❌ Erreur système: {str(e)}").send()

@cl.action_callback("web_search")
async def on_action(action: cl.Action):
    """Callback pour la recherche web via Tavily"""
    query = action.value
    msg = cl.Message(content=f"🔍 Recherche web en cours pour : *{query}*...")
    await msg.send()
    
    try:
        # Si ton tool est synchrone, on utilise aussi make_async
        result = await asyncio.to_thread(web_search.invoke, query)
        await cl.Message(content=f"**Résultats du Web :**\n\n{result}").send()
    except Exception as e:
        await cl.Message(content=f"❌ Échec de la recherche : {str(e)}").send()