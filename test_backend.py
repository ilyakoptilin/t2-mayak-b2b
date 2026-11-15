import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)



def test_api_ask_success_full_answer():
    """Тест: Идеальный сценарий, база нашла ответ"""
    with patch('main.generate_rag_response') as mock_generate:
        mock_generate.return_value = {
            "answer": "Для подключения APN создайте заявку в BPM7.",
            "sources": ["Документ: Docs_sxema.xlsx\nТекст:\nСистема: VAS B2B"]
        }
        
        response = client.post("/api/ask", json={"query": "Как подключить APN?"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Для подключения APN создайте заявку в BPM7."
        assert len(data["sources"]) == 1
        mock_generate.assert_called_once_with("Как подключить APN?")

def test_api_ask_no_data_in_db():
    """Тест: Идеальный запрос, но в базе нет документов (проверка промпта)"""
    with patch('main.generate_rag_response') as mock_generate:
        mock_generate.return_value = {
            "answer": "В текущих регламентах базы знаний нет точной информации по этому запросу.",
            "sources": []
        }
        
        response = client.post("/api/ask", json={"query": "Как запустить адронный коллайдер?"})
        
        assert response.status_code == 200
        data = response.json()
        assert "нет точной информации" in data["answer"]
        assert len(data["sources"]) == 0





def test_api_ask_empty_string():
    """Тест: Защита от пустых строк (проверка нашей логики в main.py)"""
    response = client.post("/api/ask", json={"query": "   "})
    assert response.status_code == 422
    assert response.json()["detail"] == "Пустой запрос"

def test_api_ask_missing_field():
    """Тест: Защита от кривого JSON без ключа 'query' (проверка Pydantic)"""
    response = client.post("/api/ask", json={"wrong_key": "Как подключить APN?"})
    assert response.status_code == 422
    # Pydantic должен вернуть ошибку missing
    assert response.json()["detail"][0]["type"] == "missing"

def test_api_ask_wrong_data_type():
    """Тест: Защита от отправки чисел или списков вместо строки"""
    response = client.post("/api/ask", json={"query": 12345})
    # Хотя Pydantic может привести 12345 к строке, проверим отправку чистого массива
    response = client.post("/api/ask", json={"query": ["это", "массив"]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "string_type"





def test_api_ask_ollama_offline():
    """Тест: Отработка падения локальной LLM или ChromaDB"""
    with patch('main.generate_rag_response') as mock_generate:
        # Имитируем, что requests.post внутри движка выбросил ошибку
        mock_generate.return_value = {
            "answer": "Ошибка генерации: Connection refused",
            "sources": []
        }
        
        response = client.post("/api/ask", json={"query": "Что-то сломалось?"})
        
        assert response.status_code == 500
        assert "Connection refused" in response.json()["detail"]