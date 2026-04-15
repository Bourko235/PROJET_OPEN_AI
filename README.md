
# 🩸 Hémo-Expert : Assistant d'Aide à la Décision en Onco-Hématologie

## 📖 Résumé du Projet

**Hémo-Assit Pro** est une solution d'intelligence artificielle conçue pour assister les cliniciens en onco-hématologie. Face à la complexité croissante des protocoles de chimiothérapie et à la densité des recommandations (SFH, HAS), cet outil offre un support cognitif permettant de sécuriser les dosages et d'accéder instantanément aux référentiels de soins.

> **Impact Clinique :** Réduction du risque temps de recherche grâce à un double contrôle automatisé (Calculateurs + RAG) et une navigation facilitée dans les protocoles et recommandations.

-----

## 🛠️ Stack Technique

  * **Orchestration IA :** `LangChain` (Agents autonomes & Router sémantique).
  * **Modèles de Langage :** `OpenAI GPT-4o` (Raisonnement médical complexe).
  * **Vector Database :** `ChromaDB` (Stockage et recherche sémantique des protocoles).
  * **Recherche Web :** `Tavily API` & `DuckDuckGo` (Veille sur les actualités et ruptures de stocks, recherche).
  * **Interface Utilisateur :** `Chainlit` (Dashboard asynchrone pour une interaction fluide).
  * **Calculs Médicaux :** `Math` & `Pylatexenc` (Rendu LaTeX des formules BSA et Cockcroft-Gault).

-----

## 🏗️ Architecture du Projet

Le projet utilise une architecture agentique modulaire permettant une séparation entre le savoir (RAG) et l'action (Outils) :

```text
├── app.py                # Point d'entrée Interface Web (Chainlit)
├── main.py               # Poste de pilotage (Ingest, CLI, Web)
├── data/                 # Référentiels médicaux (PDF HAS, SFH)
├── memory_sessions/      # Persistance des historiques patients (JSON)
├── vectorstore/          # Index vectoriel ChromaDB (Persistant)
├── src/                  
│   ├── agent_supervisor.py # Orchestrateur 
│   ├── query_engine.py     # Moteur RAG & Expansion de requêtes
│   ├── router.py           # Routeur sémantique (Document vs Tool)
│   ├── tools.py            # Outils cliniques (Calculs, Web, Todo)
│   ├── memory_manager.py   # Gestionnaire de mémoire persistante
│   ├── ingestion.py        # Pipeline d'indexation des PDF
│   └── config.py           # Paramètres système et clés API
├── requirements.txt      # Dépendances figées
└── .env                  # Clés API (OpenAI, Tavily, OWM)
```

-----

## 🚀 Installation et Utilisation

```bash
# 1. Préparer l'environnement
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Indexer les protocoles (Dossier data/)
python main.py --ingest

# 4. Lancer l'interface médicale
python main.py --web
```

-----

## 🤖 Capacités de l'Agent Expert

Le système ne se contente pas de répondre, il **agit** selon les besoins du clinicien :

1.  **RAG de Précision :** Recherche dans les PDF avec extension de requête pour les acronymes (LAL, LLC, Myélome).
2.  **Chaînage Logique :** Calcul du BSA $\rightarrow$ Recherche de dose $\rightarrow$ Rapport de synthèse.
3.  **Sécurité des Données :** Mémoire locale persistante.

-----

## 🧪 Robustesse et Sécurité

Chaque module est conçu pour garantir la fiabilité des informations :

  * **Eviter des hallucinations :** Le routeur force l'usage du RAG ou des outils pour les données critiques.
  * **Transparence :** Chaque réponse documentaire est accompagnée de ses sources précises.

-----

## ⚠️ Disclaimer Médical

Cet outil est un **dispositif d'assistance à la décision** destiné aux professionnels de santé. Les prédictions et calculs doivent impérativement être validés  avant toute application clinique.

-----

**Equipe:** Ba Ibrahima | Mahamat Sultan | Moustapha Mendy
