import time
import httpx
import chromadb
from typing import List, Dict, Any, Tuple
from schemas import ServiceCard, SourceDoc

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "mayak_b2b_regulations"

class RAGEngine:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def search_context(self, query: str, include_archive: bool, top_k: int = 2) -> Tuple[str, List[SourceDoc]]:
        filter_meta = None if include_archive else {"status": "new_sales"}
        count = self.collection.count()
        if count == 0:
            return "", []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            where=filter_meta
        )

        sources: List[SourceDoc] = []
        context_chunks: List[str] = []

        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for i, text in enumerate(docs):
                meta = metas[i]
                sim_score = round(1.0 - float(distances[i]), 3) if distances else 0.85
                sources.append(SourceDoc(
                    doc_id=ids[i],
                    title=str(meta.get("title", "Регламент T2")),
                    url=str(meta.get("url", "https://mayak.t2.ru")),
                    status=str(meta.get("status", "new_sales")),
                    relevance_score=max(0.0, sim_score)
                ))
                context_chunks.append(f"[Документ: {meta.get('title')}]: {text}")

        context_text = "\n\n".join(context_chunks)
        return context_text, sources

    async def check_ollama_alive(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def generate_answer(self, query: str, context: str) -> Tuple[str, bool]:
        if not context:
            return "По вашему запросу не найдено действующих регламентов в базе «Маяк». Включите тумблер «Архив», если ищете условия для ранее подключенных договоров.", True

        prompt = (
            "Ты — интеллектуальный ассистент B2B-продаж Т2 Мобайл. "
            "Твоя задача — дать четкий ответ менеджеру по продажам, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте регламентов.\n\n"
            f"КОНТЕКСТ РЕГЛАМЕНТА:\n{context}\n\n"
            f"ВОПРОС ПРОДАВЦА: {query}\n\n"
            "ТРЕБОВАНИЯ:\n"
            "1. Ответь кратко и строго по делу.\n"
            "2. Укажи точные цифры и цены из текста.\n"
            "3. Если ответа нет в тексте, прямо скажи об этом."
        )

        ollama_ready = await self.check_ollama_alive()
        if not ollama_ready:
            mock_answer = (
                f"[Режим демонстрации]: На основе регламентов Т2 найдены условия продукта. "
                f"Подключение доступно для корпоративных клиентов. "
                f"Контекст регламента: {context[:220]}..."
            )
            return mock_answer, True

        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": DEFAULT_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2}
                    }
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip(), False
        except Exception:
            pass

        return f"Регламент найден: {context[:200]}...", True

    def build_service_card(self, query: str, sources: List[SourceDoc]) -> ServiceCard:
        title = sources[0].title if sources else "Корпоративный тариф Т2"
        return ServiceCard(
            title=title,
            price="От 650 руб./мес. за номер",
            target_audience="Малый и средний бизнес (юридические лица и ИП)",
            connect_steps="1. Проверка ИНН -> 2. Выбор пакета -> 3. Подписание в ЭДО",
            objections_handling="«Дорого» -> Показать безлимитный трафик на корпоративные сервисы и перенос остатков."
        )

rag_service = RAGEngine()
