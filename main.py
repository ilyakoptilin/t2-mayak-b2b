import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import SearchRequest, SearchResponse, HealthResponse
from rag_engine import rag_service, DEFAULT_MODEL

app = FastAPI(
    title="Т2 Маяк 2.0 — Интеллектуальный помощник B2B",
    description="RAG-бэкенд для семантического поиска по регламентам Т2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Система"])
async def health_check():
    ollama_ready = await rag_service.check_ollama_alive()
    doc_count = rag_service.collection.count()
    return HealthResponse(
        status="ok" if (ollama_ready and doc_count > 0) else "degraded",
        chroma_ready=True,
        chroma_documents_count=doc_count,
        ollama_ready=ollama_ready,
        model_name=DEFAULT_MODEL
    )

@app.post("/api/v1/search", response_model=SearchResponse, tags=["Поиск B2B"])
async def search_regulation(req: SearchRequest):
    start_time = time.time()
    context, sources = rag_service.search_context(
        query=req.query,
        include_archive=req.include_archive,
        top_k=req.top_k
    )
    answer, is_mock = await rag_service.generate_answer(
        query=req.query,
        context=context
    )
    card = rag_service.build_service_card(req.query, sources) if sources else None
    latency = int((time.time() - start_time) * 1000)

    return SearchResponse(
        query=req.query,
        answer=answer,
        service_card=card,
        sources=sources,
        is_mock=is_mock,
        latency_ms=latency
    )
