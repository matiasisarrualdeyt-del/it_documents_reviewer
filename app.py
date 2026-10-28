import os
import streamlit as st
from app.rag_graph import AgentState, rag_app
# Configuración de la página
st.set_page_config(
    page_title="IT Documents Reviewer", page_icon="🛠️", layout="wide"
)

st.title("🛠️ IT Documents Reviewer")
st.markdown(
    "Asistente técnico inteligente para la revisión de documentación y soporte de desarrollo."
)

# --- BARRA LATERAL (SIDEBAR): Ingesta de Documentos ---
with st.sidebar:
  st.header("📂 Ingesta de Documentos")
  uploaded_files = st.file_uploader(
      "Selecciona archivos", type=["pdf", "txt", "md"], accept_multiple_files=True
  )

  if uploaded_files:
    if st.button("Procesar e Ingestar"):
      with st.spinner("Procesando documentos..."):
        # Lógica de ingesta aquí
        pass
        st.success("¡Archivos ingetados correctamente!")

  st.markdown("---")
  st.markdown("### Estado del Sistema")
  st.info("Grafo activo: Self-RAG con reescritura y validación de contexto.")

# --- INTERFAZ PRINCIPAL: Chat Técnico ---

# 1. Inicializar el historial del chat
if "messages" not in st.session_state:
  st.session_state.messages = []

# 2. Mostrar mensajes anteriores
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# 3. Entrada del usuario (Aquí es donde 'prompt' nace de forma segura)
if prompt := st.chat_input("Escribe tu consulta técnica..."):
  # Guardar y mostrar mensaje del usuario
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # Invocar al grafo de LangGraph
  with st.chat_message("assistant"):
    with st.spinner("Analizando documentación técnica..."):
      try:
        initial_state: AgentState = {
            "question": prompt,
            "documents": [],
            "generation": "",
            "context_grade": "",
            "loop_step": 0,
        }

        result = rag_app.invoke(initial_state)
        response_text = result.get(
            "generation", "No se pudo generar una respuesta."
        )

        st.markdown(response_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

      except Exception as e:
        st.error(f"Ocurrió un error al ejecutar el grafo: {str(e)}")