from typing import List, Literal, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph, START
from .vector_store import vector_store_service
from .config import settings

# 1. Definición del Estado del Agente
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    context_grade: str  # Para almacenar el resultado de la evaluación de calidad
    loop_step: int      # Para controlar bucles de reescritura infinitos

# 2. Inicializar el LLM de Google Gemini para el agente
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.1
)


# --- NODOS DEL GRAFO ---

def retrieve_node(state: AgentState):
    """Nodo 1: Recupera documentos técnicos desde ChromaDB."""
    print("--- [NODO] RECUPERANDO DOCUMENTOS ---")
    question = state["question"]
    
    results = vector_store_service.similarity_search_with_score(question, k=3)
    documents = [doc for doc, _ in results]
    
    return {"documents": documents}

def grade_context_node(state: AgentState):
    """Nodo 2: Evalúa si los documentos recuperados son relevantes para la pregunta."""
    print("--- [NODO] EVALUANDO CALIDAD DE CONTEXTO ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        print("--> No se encontraron documentos. Marcando para reescritura.")
        return {"context_grade": "rewrite"}

    grading_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un evaluador técnico estricto. Tu trabajo es determinar si los fragmentos de documentación técnica provistos contienen información útil para responder la consulta del desarrollador. Responde unicamente con 'yes' o 'no'."),
        ("human", "Consulta: {question}\n\nFragmentos de documentación:\n{context}")
    ])
    
    context_text = "\n\n".join([doc.page_content for doc in documents])
    chain = grading_prompt | llm
    
    response = chain.invoke({"question": question, "context": context_text})
    
    content_str = response.content if isinstance(response.content, str) else str(response.content)
    grade = content_str.strip().lower()
    
    print(f"--> Resultado de evaluación del contexto: {grade}")
    
    if "yes" in grade:
        return {"context_grade": "generate"}
    else:
        return {"context_grade": "rewrite"}

def route_after_grading(state: AgentState) -> Literal["generate", "rewrite"]:
  """Función de enrutamiento condicional basada en el estado y límite de intentos."""
  grade = state.get("context_grade", "rewrite")
  loop_step = state.get("loop_step", 0)

  MAX_LOOPS = 3
  if loop_step >= MAX_LOOPS:
    print(
        f"--- [ENRUTAMIENTO] Límite de reintentos alcanzado ({loop_step})."
        " Forzando paso a generación ---"
    )
    return "generate"

  if grade == "generate":
    return "generate"
  return "rewrite"

def rewrite_node(state: AgentState):
    """Nodo 3: Reescribe la pregunta del usuario para mejorar la búsqueda técnica."""
    print("--- [NODO] REESCRIBIENDO CONSULTA ---")
    question = state["question"]
    loop_step = state.get("loop_step", 0)
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente de ingeniería de software experto. Tu tarea es reformular la consulta de un desarrollador para que sea más precisa, técnica y efectiva para buscar en una base de datos vectorial de documentación técnica."),
        ("human", "Consulta original: {question}")
    ])
    
    chain = rewrite_prompt | llm
    response = chain.invoke({"question": question})
    
    content_str = response.content if isinstance(response.content, str) else str(response.content)
    new_question = content_str.strip()
    
    print(f"--> Pregunta reescrita: {new_question}")
    
    return {
        "question": new_question,
        "loop_step": loop_step + 1
    }

def generate_node(state: AgentState):
    """Nodo 4: Genera la respuesta técnica final citando obligatoriamente los documentos."""
    print("--- [NODO] GENERANDO RESPUESTA TÉCNICA ---")
    question = state["question"]
    documents = state["documents"]
    
    context_parts = []
    for i, doc in enumerate(documents):
        doc_id = doc.metadata.get("doc_id", "unknown_doc")
        source = doc.metadata.get("source", "unknown_source")
        context_parts.append(f"[Documento ID: {doc_id} | Fuente: {source}]\n{doc.page_content}")
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres IT Documents Reviewer, un asistente de IA experto para desarrolladores. 
Responde la consulta técnica utilizando EXCLUSIVAMENTE el contexto provisto. 
Es una REGLA OBLIGATORIA incluir citas explícitas a los documentos fuente usando el formato [doc_id] cada vez que utilices información de un fragmento.
Si la respuesta no se encuentra en el contexto, indícalo claramente sin inventar información."""),
        ("human", "Contexto técnico:\n{context}\n\nConsulta del desarrollador: {question}")
    ])
    
    chain = generation_prompt | llm
    response = chain.invoke({"context": context_text, "question": question})
    
    # Manejo robusto: convertimos content a string si viene como lista o dict
    raw_content = response.content
    if isinstance(raw_content, list):
        # Si es una lista de fragmentos, extraemos el texto de cada uno
        generation_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in raw_content])
    else:
        generation_text = str(raw_content)
    
    return {"generation": generation_text}


# --- CONSTRUCCIÓN DEL GRAFO (WORKFLOW) ---

workflow = StateGraph(AgentState)

# Añadir nodos
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_context", grade_context_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)

# Conectar el flujo
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_context")

# Arista condicional utilizando la función de enrutamiento separada
workflow.add_conditional_edges(
    "grade_context",
    route_after_grading,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)

workflow.add_edge("rewrite", "retrieve")  # Bucle de reintento con la pregunta mejorada
workflow.add_edge("generate", END)

# Compilar el grafo
rag_app = workflow.compile()