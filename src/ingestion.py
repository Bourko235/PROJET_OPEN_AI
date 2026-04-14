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
                os.unlink(file_path)  # Supprime fichier ou lien
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) # Supprime sous-dossier
        except Exception as e:
            logger.error(f"Impossible de supprimer {file_path}: {e}")

def ingest_documents():
    persist_directory = "vectorstore"
    
    # Correction : On vide le contenu au lieu de supprimer le dossier
    clean_vectorstore_folder(persist_directory)

    data_path = Path(data_path or CONFIG.DATA_PATH)
    persist_directory = Path(persist_directory or CONFIG.VECTORSTORE_PATH)

    # 🔴 1. Nettoyage AVANT toute utilisation de Chroma
    if clear_existing and persist_directory.exists():
        print(f"🧹 Nettoyage de l'ancienne base : {persist_directory}")
        clean_vectorstore(persist_directory)

    # 🔴 Libération mémoire préventive (important sous Windows)
    gc.collect()

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Dossier {data_path} créé. Ajoutez vos PDF/TXT/MD ici.")
        return

    print(f"--- 📂 Chargement depuis {data_path} ---")
    documents = []

    # 📄 PDF
    pdf_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents.extend(pdf_loader.load())

    # 📄 TXT / MD
    for ext in ["*.txt", "*.md"]:
        text_loader = DirectoryLoader(
            str(data_path),
            glob=f"**/{ext}",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'},
            show_progress=True
        )
        try:
            documents.extend(text_loader.load())
        except Exception as e:
            print(f"⚠️ Erreur sur {ext}: {e}")

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

    # 🧠 3. Embeddings
    embeddings = OpenAIEmbeddings(
        model=CONFIG.EMBEDDING_MODEL,
        openai_api_key=CONFIG.OPENAI_API_KEY,
        chunk_size=250
    )

    # 📦 4. Vector store
    print("🧠 Indexation dans ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"--- ✅ Indexation réussie ---")
    print(f"📍 Dossier : {persist_directory}")
    print(f"🔢 Total fragments indexés : {vectorstore._collection.count()}")

    # 🔒 5. Libération propre (CRUCIAL pour éviter le bug au prochain run)
    vectorstore = None
    gc.collect()
    
logger.info("📥 Indexation des nouveaux documents en cours...")

if __name__ == "__main__":
    ingest_documents()