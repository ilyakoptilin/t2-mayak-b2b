import chromadb
from rag_engine import CHROMA_PATH, COLLECTION_NAME

def seed():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    docs = [
        (
            "doc_b2b_001",
            "Тариф 'Моя Команда 2026'. Абонентская плата 750 рублей в месяц за абонента. "
            "Включает 40 ГБ мобильного интернета, 600 минут на все номера РФ, безлимитные мессенджеры. "
            "Подключение доступно для всех новых юридических лиц и ИП с 1 января 2026 года.",
            {
                "title": "Тарифный план 'Моя Команда 2026'",
                "url": "https://mayak.t2.ru/b2b/tariffs/moya-komanda-2026",
                "status": "new_sales",
                "author": "Департамент B2B продуктов",
                "date": "2026-01-15"
            }
        ),
        (
            "doc_b2b_002",
            "Услуга 'Корпоративная АТС Т2'. Ежемесячный платеж 1200 рублей за компанию. "
            "Включает многоканальный номер, голосовое меню (IVR), запись звонков до 90 дней и интеграцию с CRM. "
            "Срок запуска услуги — 1 рабочий день после подписания заявки в ЛК B2B.",
            {
                "title": "Регламент услуги 'Виртуальная АТС B2B'",
                "url": "https://mayak.t2.ru/b2b/services/corporate-ats",
                "status": "new_sales",
                "author": "Отдел цифровых решений",
                "date": "2026-02-10"
            }
        ),
        (
            "doc_b2b_003_archive",
            "Тариф 'Корпоративный Стандарт 2021' (АРХИВ). Абонентская плата 450 рублей в месяц. "
            "Включает 15 ГБ интернета и 300 минут. "
            "ВНИМАНИЕ: Тариф закрыт для новых подключений с 2023 года! Действует только на обслуживание действующих договоров.",
            {
                "title": "Архив: Тариф 'Корпоративный Стандарт 2021'",
                "url": "https://mayak.t2.ru/archive/tariffs/standard-2021",
                "status": "maintenance",
                "author": "Архивный отдел",
                "date": "2021-04-12"
            }
        ),
    ]

    collection.add(
        ids=[d[0] for d in docs],
        documents=[d[1] for d in docs],
        metadatas=[d[2] for d in docs]
    )
    print(f"База «Маяк B2B» успешно инициализирована! Загружено {len(docs)} регламентов.")

if __name__ == "__main__":
    seed()
