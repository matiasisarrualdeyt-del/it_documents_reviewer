import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import settings

class VectorStoreService:
    """
    Servicio encargado de gestionar los embeddings nativos de Gemini,
    la ingesta de documentos técnicos (PDF, HTML, Markdown, logs) y la 
    recuperación de fragmentos para el RAG.
    """
    def __init__(self):
        # 1. Inicializamos el modelo de embeddings oficial de Google Gemini
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )
        
        # 2. Directorio de persistencia local para Chroma
        persist_directory = "./chroma_db"
        
        # 3. Inicializamos Chroma como base vectorial persistente
        self.vector_store = Chroma(
            collection_name="it_documents_collection",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def add_documents(self, texts: list[str], metadatas: list[dict]):
        """
        Recibe fragmentos de texto planos y sus metadatos (con doc_id obligatorio)
        para indexarlos directamente en Chroma.
        """
        docs = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(texts, metadatas)
        ]
        self.vector_store.add_documents(docs)
        return len(docs)

    def ingest_file(self, file_path: str, doc_id: str):
        """
        Procesa un archivo real (PDF, HTML, Markdown o texto/log), 
        aplica chunking y lo indexa en Chroma.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo en la ruta: {file_path}")

        # 1. Selección del loader según la extensión del archivo técnico
        ext = path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        elif ext in [".html", ".htm"]:
            loader = UnstructuredHTMLLoader(str(path))
        elif ext in [".md", ".txt", ".log"]:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Formato de archivo no soportado: {ext}")

        raw_docs = loader.load()

        # 2. Configurar el splitter basado en caracteres con solapamiento
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,     # Tamaño óptimo de caracteres por fragmento
            chunk_overlap=50    # Solapamiento para preservar contexto técnico
        )

        # 3. Fragmentar el documento
        chunks = text_splitter.split_documents(raw_docs)

        # 4. Preparar textos y metadatos estructurados
        texts = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            texts.append(chunk.page_content)
            metadatas.append({
                "doc_id": doc_id,
                "source": path.name,
                "chunk_index": i
            })

        # 5. Indexar en la base vectorial
        indexed_count = self.add_documents(texts, metadatas)
        return indexed_count

    def similarity_search_with_score(self, query: str, k: int = 4):
        """
        Busca los k documentos más similares a la consulta del usuario 
        y devuelve una lista de tuplas: (Document, score_de_similitud).
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

# Instancia global del servicio vectorial
vector_store_service = VectorStoreService()