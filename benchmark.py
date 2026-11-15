import time
import requests
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# --- Настройки ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODELS_TO_TEST = ["qwen2.5:3b", "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"] 
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "b2b_mayak_docs"

TEST_QUESTIONS = [
    "Как подключить Выделенный APN и Фиксированный IP?",
    "Какие условия и стоимость у услуги SMS-Таргет?",
    "Как происходит миграция клиентов с М2М Система на М2М Платформа 2.0?",
    "Какой регламент снятия блокировки с номера корпоративного клиента?",
    "Что входит в услугу Защита сайта от киберугроз (WAF Lite)?"
]

def setup_chroma():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="cointegrated/rubert-tiny2")
    return chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

def get_rag_context(collection, query: str, n_results: int = 3):
    results = collection.query(query_texts=[query], n_results=n_results)
    
    context_blocks = []
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        source = results['metadatas'][0][i].get('source', 'Неизвестно')
        context_blocks.append(f"[Источник: {source}]\n{text}")
        
    return "\n\n".join(context_blocks)

def ask_ollama(model_name: str, query: str, context: str):
    system_prompt = (
        "Ты — корпоративный B2B-ассистент 'Маяк'. Отвечай на вопросы менеджеров "
        "строго на основе предоставленного контекста. Не придумывай информацию.\n\n"
        f"КОНТЕКСТ:\n{context}"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "stream": False
    }

    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        latency = time.time() - start_time
        
        eval_count = data.get("eval_count", 0)
        eval_duration_sec = data.get("eval_duration", 0) / 1e9
        tps = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0

        return {
            "answer": data["message"]["content"],
            "latency_sec": round(latency, 2),
            "tps": round(tps, 2),
            "error": None
        }
    except Exception as e:
        return {"answer": "", "latency_sec": 0, "tps": 0, "error": str(e)}

def main():
    print("Инициализация базы данных...")
    collection = setup_chroma()
    results_data = []

    for model in MODELS_TO_TEST:
        print(f"\n{'='*50}")
        print(f"🚀 Запуск тестов для модели: {model}")
        print(f"{'='*50}")

        for i, query in enumerate(TEST_QUESTIONS, 1):
            print(f"\n[Вопрос {i}/{len(TEST_QUESTIONS)}]: {query}")
            
            # 1. Достаем контекст
            context = get_rag_context(collection, query)
            
            # ВЫВОДИМ ЧАНКИ В ТЕРМИНАЛ
            print("-" * 40)
            print("🔍 ИЗВЛЕЧЕННЫЙ КОНТЕКСТ:")
            print(context)
            print("-" * 40)
            
            # 2. Генерируем ответ
            print("⏳ Генерация ответа...")
            metrics = ask_ollama(model, query, context)
            
            if metrics["error"]:
                print(f"❌ Ошибка Ollama: {metrics['error']}")
                continue

            print(f"✅ Готово! Время: {metrics['latency_sec']}с | Скорость: {metrics['tps']} т/с")
            
            # 3. Сохраняем результат С КОНТЕКСТОМ
            results_data.append({
                "Модель": model,
                "Вопрос": query,
                "Время_ответа_сек": metrics["latency_sec"],
                "Скорость_токенов_в_сек": metrics["tps"],
                "Ответ_модели": metrics["answer"],
                "Контекст_из_базы": context  # <--- Теперь чанки сохранятся в таблицу
            })

    # Выгрузка результатов в CSV
    df = pd.DataFrame(results_data)
    output_file = "benchmark_results.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n🎉 Тестирование завершено! Результаты с чанками сохранены в {output_file}")

if __name__ == "__main__":
    main()