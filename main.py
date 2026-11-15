from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from rag_engine import generate_rag_response

app = FastAPI(title="Маяк - B2B Ассистент", version="1.0.0")

# Схемы данных для валидации Pydantic
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/api/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Пустой запрос")
    
    # Вызываем нашу функцию (Qwen 3 подставится по умолчанию)
    result = generate_rag_response(request.query)
    
    if "Ошибка генерации" in result["answer"]:
        raise HTTPException(status_code=500, detail=result["answer"])
        
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"]
    )