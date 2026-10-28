import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field
from vector_store import vector_store_service
from rag_graph import rag_app, AgentState

app = FastAPI(
    title="IT Documents Reviewer API",
    description="Backend RAG modular impulsado por LangGraph y Gemini para revisión de documentación técnica.",
    version="1.0.0"
)

# --- MODELOS PYDANTIC PARA LAS PETICIONES Y RESPUESTAS ---

class QueryRequest(BaseModel):
    question: str = Field(..., description="Pregunta técnica del desarrollador")

class QueryResponse(BaseModel):
    question: str
    generation: str
    documents_used: int


# --- ENDPOINTS ---

@app.get("/")
def root():
    """Endpoint de comprobación de estado."""
    return {
        "status": "online",
        "service": "IT Documents Reviewer API",
        "engine": "FastAPI + LangGraph + Gemini"
    }

@app.post("/ingest", summary="Subir e ingestar un documento técnico desde el cliente")
async def ingest_document(
    file: UploadFile = File(..., description="Archivo técnico (PDF, Markdown, HTML, Log)"),
    doc_id: str = Form(..., description="Identificador único para las citas (ej: doc_01)")
):
    """
    Recibe un archivo enviado desde el cliente, lo almacena temporalmente en la 
    carpeta data/ y ejecuta la ingesta y chunking en ChromaDB.
    """
    try:
        # 1. Definir ruta de destino dentro de nuestra carpeta data/
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Validamos que el archivo tenga nombre para evitar problemas de tipos
        filename = file.filename or "uploaded_file.bin"
        file_path = data_dir / filename
        
        # 2. Guardar el archivo físico en el servidor
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Llamar al servicio vectorial pasándole la ruta real del archivo guardado
        chunks_count = vector_store_service.ingest_file(
            file_path=str(file_path), 
            doc_id=doc_id
        )
        
        return {
            "message": "Archivo subido e indexado exitosamente",
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks_created": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")

@app.post("/query", response_model=QueryResponse, summary="Consultar al asistente técnico")
def query_agent(payload: QueryRequest):
    """
    Ejecuta el grafo de LangGraph: recupera, evalúa contexto, reescribe si es necesario 
    y genera una respuesta técnica con citas obligatorias.
    """
    try:
        # Estado inicial tipado explícitamente como AgentState para Pylance
        initial_state: AgentState = {
            "question": payload.question,
            "documents": [],
            "generation": "",
            "loop_step": 0,
            "context_grade":""
        }
        
        # Invocamos el flujo completo de LangGraph
        final_state = rag_app.invoke(initial_state)
        
        return QueryResponse(
            question=payload.question,
            generation=final_state.get("generation", "No se pudo generar una respuesta."),
            documents_used=len(final_state.get("documents", []))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ejecutando el agente RAG: {str(e)}")