import argparse
import sys
import subprocess
import asyncio
import logging
from pathlib import Path

# Ajout du dossier racine au path pour la résolution des modules
sys.path.insert(0, str(Path(__file__).parent))

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def cli_mode():
    """Mode interactif en ligne de commande avec support asynchrone."""
    # IMPORT LOCAL : On ne charge l'agent que si on entre en mode CLI
    from src.agent_supervisor import supervisor
    from src.memory_manager import memory_manager
    
    print("\n" + "="*60)
    print("🩸 HÉMO-EXPERT : INTERFACE CONSOLE (DEBUG)")
    print("="*60)
    print("Commandes spéciales : \n - /reset : Effacer la mémoire session\n - /exit  : Quitter")
    print("-"*60)
    
    session_id = "cli_debug_session"
    
    while True:
        try:
            user_input = input("\n👤 Médecin > ").strip()
            if not user_input:
                continue
            
            if user_input.lower() == "/exit":
                print("👋 Fermeture sécurisée. Au revoir Docteur.")
                break
            
            if user_input.lower() == "/reset":
                memory_manager.clear(session_id)
                print("🧹 [Système] Mémoire de la session réinitialisée.")
                continue
            
            # Traitement asynchrone du superviseur
            result = await supervisor.process(user_input, session_id)
            
            # Affichage de la réponse
            print(f"\n🤖 Hémo-Expert :\n{result['output']}")
            
            # Logs de routage pour le développeur
            route = result.get('route_type', 'unknown')
            conf = result.get('confidence', 0)
            print(f"\n--- [Log: Route={route} | Confidence={conf:.0%}] ---")
            
            # Citations
            citations = result.get("citations", [])
            if citations:
                sources = list(set([c['source'] for c in citations]))
                print(f"📚 Sources : {', '.join(sources)}")
                
        except KeyboardInterrupt:
            print("\nArrêt forcé.")
            break
        except Exception as e:
            logger.error(f"Erreur en mode CLI : {e}")

def main():
    parser = argparse.ArgumentParser(description="Hémo-Expert : Orchestrateur")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ingest", action="store_true", help="Indexer les PDF")
    group.add_argument("--web", action="store_true", help="Lancer Chainlit")
    group.add_argument("--cli", action="store_true", help="Mode console")
    
    parser.add_argument("-p", "--port", type=int, default=8002, help="Port web")
    
    args = parser.parse_args()
    
    if args.ingest:
        # IMPORT LOCAL : On ne charge le script d'ingestion que si nécessaire
        # Cela évite de verrouiller la base de données au démarrage du main
        from src.ingestion import ingest_documents
        print("📥 [Ingestion] Analyse et indexation des référentiels...")
        ingest_documents()
        
    elif args.web:
        print(f"🚀 [Web] Lancement de l'interface sur http://localhost:{args.port}")
        try:
            # Chainlit tourne dans son propre processus, aucun conflit d'import ici
            subprocess.run(["chainlit", "run", "app.py", "--port", str(args.port)], check=True)
        except KeyboardInterrupt:
            print("\nServeur arrêté.")
            
    elif args.cli:
        asyncio.run(cli_mode())
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()