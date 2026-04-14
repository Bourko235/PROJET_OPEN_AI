import os
import shutil
import time
import gc
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import CONFIG
import logging

logger = logging.getLogger(__name__)

def clean_vectorstore_folder(folder_path):
    """
    Supprime le contenu du dossier sans supprimer le dossier lui-même.
    Ceci évite l'erreur 'Device or resource busy' sur les volumes Docker.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return

    logger.info(f"🧹 Vidage sécurisé du contenu de : {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"Impossible de supprimer {file_path}: {e}")

def ingest_documents(data_path=None, persist_directory=None, clear_existing=True):
    """
    Pipeline d'ingestion avec nettoyage sécurisé pour Docker.
    """
    # Définition des chemins avec fallback sur CONFIG
    data_path = Path(data_path or CONFIG.DATA_PATH)
    persist_directory = Path(persist_directory or CONFIG.VECTORSTORE_PATH)
    
    logger.info("📥 Indexation des nouveaux documents en cours...")

    # 🔴 1. Nettoyage AVANT toute utilisation de Chroma
    if clear_existing:
        clean_vectorstore_folder(str(persist_directory))

    # 🔴 Libération mémoire préventive
    gc.collect()

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Dossier {data_path} créé. Ajoutez vos PDF/TXT/MD ici.")
        return

    print(f"--- 📂 Chargement depuis {data_path} ---")
    documents = []

    # 📄 PDF
    try:
        pdf_loader = DirectoryLoader(
            str(data_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents.extend(pdf_loader.load())
    except Exception as e:
        logger.warning(f"Erreur chargement PDF: {e}")

    # 📄 TXT / MD
    for ext in ["*.txt", "*.md"]:
        try:
            text_loader = DirectoryLoader(
                str(data_path),
                glob=f"**/{ext}",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                show_progress=True
            )
            documents.extend(text_loader.load())
        except Exception as e:
            logger.warning(f"Erreur chargement {ext}: {e}")

    if not documents:
        print("⚠️ Aucun document trouvé. Ingestion annulée.")
        return

    print(f"✅ {len(documents)} fichiers chargés.")

    # ✂️ 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.CHUNK_SIZE,
        chunk_overlap=CONFIG.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✂️ {len(chunks)} fragments (chunks) créés.")

    # 🧠 3. Embeddings avec batching pour éviter l'erreur 400 tokens
    embeddings = OpenAIEmbeddings(
        model=CONFIG.EMBEDDING_MODEL,
        openai_api_key=CONFIG.OPENAI_API_KEY,
        chunk_size=100  # Limite le nombre de docs par appel API
    )

    # 📦 4. Vector store avec batching manuel
    print("🧠 Indexation dans ChromaDB...")
    
    # Création incrémentale pour éviter les limites tokens
    vectorstore = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings
    )
    
    # Batching manuel
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vectorstore.add_documents(batch)
        if (i // batch_size) % 10 == 0:
            print(f"   Progression: {min(i+batch_size, len(chunks))}/{len(chunks)}")

    print(f"--- ✅ Indexation réussie ---")
    print(f"📍 Dossier : {persist_directory}")
    print(f"🔢 Total fragments indexés : {vectorstore._collection.count()}")

    # 🔒 5. Libération propre
    vectorstore.persist()
    vectorstore = None
    gc.collect()

if __name__ == "__main__":
    ingest_documents()