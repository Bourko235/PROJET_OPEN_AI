import os
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CONFIG

class RAGEngine:
    """
    Moteur RAG de précision pour Hémo-Expert.
    Gère l'expansion sémantique médicale, le sourçage par page et l'ingestion à la volée.
    """
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=CONFIG.EMBEDDING_MODEL,
            api_key=CONFIG.OPENAI_API_KEY
        )
        
        if not os.path.exists(CONFIG.VECTORSTORE_PATH):
            # Si la base n'existe pas, on la crée vide plutôt que de planter
            os.makedirs(CONFIG.VECTORSTORE_PATH, exist_ok=True)
            logger_info = "Base vectorielle créée (vide)."
        else:
            logger_info = "Base vectorielle chargée."
        
        self.vectorstore = Chroma(
            persist_directory=CONFIG.VECTORSTORE_PATH,
            embedding_function=self.embeddings
        )
        
        self.llm = ChatOpenAI(
            model=CONFIG.MODEL_NAME,
            temperature=0,
            api_key=CONFIG.OPENAI_API_KEY
        )

        # 1. Prompt d'expansion médicale (Optimisé pour l'onco-hématologie)
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Tu es un expert en hématologie. 
            Génère 3 variantes de cette question pour optimiser la recherche de protocoles.
            Traduis impérativement les acronymes (ex: LAL, LLC, MM, RCP, IPI).
            Question originale : {question}"""
        )
        
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": CONFIG.TOP_K_RETRIEVAL}
        )
        
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
            prompt=QUERY_PROMPT
        )
        
        # 2. Prompt de génération "Hémo-Expert"
        self.prompt = ChatPromptTemplate.from_template("""
Tu es 'Hémo-Expert', un assistant médical spécialisé. 
Utilise les extraits de recommandations (HAS, SFH, Protocoles) ci-dessous pour répondre.

RÈGLES CRUCIALES :
1. RÉPONSE SOURCÉE : Cite la source à la fin de chaque paragraphe important, ex: [Source: nom_du_fichier.pdf, p.12].
2. ABSENCE D'INFO : Si le contexte ne contient pas la réponse, dis explicitement que le référentiel local est muet.
3. PRÉCISION : Ne modifie pas les dosages ou les scores pronostiques.

Contexte :
{context}

Question : {question}

Réponse experte :""")
    
    def _clean_source_name(self, path: str) -> str:
        """Extrait le nom du fichier du chemin complet ou retourne 'Inconnu'"""
        if not path: return "Inconnu"
        return os.path.basename(path)

    def format_docs(self, docs: List[Any]) -> str:
        """Formate les documents pour le contexte du LLM"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = self._clean_source_name(doc.metadata.get('source', 'Inconnu'))
            page = doc.metadata.get('page', 'NC')
            formatted.append(f"Document {i} (Source: {source}, Page: {page}):\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Exécute la chaîne RAG complète et retourne les citations pour l'UI"""
        docs = self.retriever.invoke(question)
        
        if not docs:
            return {
                "answer": "Le référentiel local est muet sur ce sujet. Souhaitez-vous une recherche sur internet ?",
                "citations": [],
                "sources": []
            }
        
        context = self.format_docs(docs)
        
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(question)
        
        # Préparation des citations pour l'interface Chainlit
        citations = []
        for doc in docs:
            citations.append({
                "source": self._clean_source_name(doc.metadata.get('source', 'Inconnu')),
                "page": doc.metadata.get('page', 'NC'),
                "content": doc.page_content[:200] + "..." # Aperçu pour l'UI
            })
            
        return {
            "answer": answer,
            "citations": citations,
            "sources": list(set([c["source"] for c in citations]))
        }

    def add_file_to_index(self, file_path: str):
        """Indexation immédiate d'un nouveau fichier (Trombone)"""
        try:
            # 1. Chargement
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # 2. Split (On utilise les mêmes réglages que l'ingestion globale)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CONFIG.CHUNK_SIZE,
                chunk_overlap=CONFIG.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            
            # 3. Enrichissement des métadonnées (optionnel mais recommandé)
            for chunk in chunks:
                chunk.metadata["source"] = self._clean_source_name(file_path)
                chunk.metadata["doc_type"] = ".pdf"

            # 4. Ajout à Chroma
            self.vectorstore.add_documents(chunks)
            print(f"✅ Document {self._clean_source_name(file_path)} indexé ({len(chunks)} fragments).")
            
            return len(chunks)
        except Exception as e:
            print(f"❌ Erreur lors de l'indexation flash : {str(e)}")
            return 0

# Instance unique exportée
rag_engine = RAGEngine()