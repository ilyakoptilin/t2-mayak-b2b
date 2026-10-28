from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос продавца на естественном языке")
    include_archive: bool = Field(default=False, description="Флаг поиска: учитывать архивные регламенты")
    top_k: int = Field(default=2, ge=1, le=5, description="Количество чанков для контекста")

class ServiceCard(BaseModel):
    title: str
    price: str
    target_audience: str
    connect_steps: str
    objections_handling: str

class SourceDoc(BaseModel):
    doc_id: str
    title: str
    url: str
    status: str
    relevance_score: float

class SearchResponse(BaseModel):
    query: str
    answer: str
    service_card: Optional[ServiceCard] = None
    sources: List[SourceDoc]
    is_mock: bool = False
    latency_ms: int

class HealthResponse(BaseModel):
    status: str
    chroma_ready: bool
    chroma_documents_count: int
    ollama_ready: bool
    model_name: str
