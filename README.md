# Т2 Маяк B2B — Search Assistant

MVP сервиса интеллектуального поиска по регламентам Т2 Мобайл с префильтрацией архивных тарифов.

## Быстрый запуск бэкенда:
1. `python3 -m venv venv && source venv/bin/activate` (на Windows: `venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `python seed_data.py`
4. `uvicorn main:app --reload --port 8000`

Документация Swagger: http://localhost:8000/docs
