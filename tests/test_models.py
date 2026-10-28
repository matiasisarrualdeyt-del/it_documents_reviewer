import os
from dotenv import load_dotenv
from google import genai

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: No se encontró la GEMINI_API_KEY en tu archivo .env")
else:
    print(f"🔑 API Key detectada. Listando todos los modelos disponibles...")
    
    client = genai.Client(api_key=api_key)
    
    try:
        print("\n--- Lista completa de modelos en tu cuenta ---")
        for m in client.models.list():
            # Imprimimos el nombre completo del modelo
            print(f"🔹 {m.name}")
    except Exception as e:
        print(f"❌ Error al conectar con la API de Google: {e}")