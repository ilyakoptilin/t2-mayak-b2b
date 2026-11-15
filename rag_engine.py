import requests
import chromadb
from chromadb.utils import embedding_functions

# --- НАСТРОЙКИ ---
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "b2b_mayak_docs"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"

# Инициализация базы данных с НОВОЙ моделью эмбеддингов
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, 
    embedding_function=ef
)

def generate_rag_response(query: str, model_name: str = DEFAULT_MODEL):
    # ВАЖНО: Модель e5 требует префикс "query: " для входящих вопросов при поиске
    search_query = f"query: {query}"
    
    # Ищем 5 самых релевантных чанков
    results = collection.query(query_texts=[search_query], n_results=5)
    
    context_blocks = []
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        source = results['metadatas'][0][i].get('source', 'Неизвестный источник')
        context_blocks.append(f"Документ: {source}\nТекст:\n{text}")
        
    full_context = "\n\n".join(context_blocks)
    
    # Мощный системный промпт для защиты от галлюцинаций
    system_prompt = (
        "Ты — корпоративный B2B-ассистент 'Маяк'. Твоя задача — помогать менеджерам быстро и точно находить информацию по бизнес-процессам.\n\n"
        "ПРАВИЛА ОТВЕТА:\n"
        "1. СТРОГАЯ ДОСТОВЕРНОСТЬ: Отвечай только на основе предоставленного КОНТЕКСТА. Категорически запрещено выдумывать цены, тарифы, регламенты, параметры или шаги, которых нет в тексте.\n"
        "2. ЧАСТИЧНЫЙ ОТВЕТ: Если в контексте есть хотя бы часть информации (например, только название системы, куда подать заявку, или одно условие из трех), обязательно выдай её. Не пиши «информации недостаточно», если можешь дать хоть какой-то полезный факт из контекста.\n"
        "3. ОТСУТСТВИЕ ДАННЫХ: Если контекст вообще не относится к вопросу, прямо и коротко ответь: «В текущих регламентах базы знаний нет точной информации по этому запросу».\n"
        "4. ФОРМАТИРОВАНИЕ: Пиши в деловом стиле, без лишней воды. Обязательно используй списки и выделяй **жирным** шрифтом названия ИТ-систем (BPM7, VAS B2B и др.), названия услуг и критические условия.\n\n"
        f"КОНТЕКСТ:\n{full_context}"
    )

    payload = {
        "model": model_name, 
        "messages": [
            {"role": "system", "content": system_prompt},
            # В саму LLM отправляем чистый вопрос от пользователя, без префикса
            {"role": "user", "content": query}
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        return {
            "answer": response.json()["message"]["content"],
            "sources": context_blocks 
        }
    except Exception as e:
        return {"answer": f"Ошибка генерации: {str(e)}", "sources": []}