"""
Pydantic-Modelle für die API (Cohere- und OpenAI-kompatibel).
"""
from pydantic import BaseModel, Field
from typing import List, Optional

# ===== Reranker-Modelle (Cohere-Format) =====

class RerankResult(BaseModel):
    """Einzelnes Rerank-Ergebnis (Cohere-Format)."""
    index: int
    relevance_score: float

class ApiVersion(BaseModel):
    """API-Version (Cohere-Format)."""
    version: str
    is_experimental: bool

class BilledUnits(BaseModel):
    """Billing-Einheiten (Cohere-Format)."""
    search_units: int

class Meta(BaseModel):
    """Meta-Informationen (Cohere-Format)."""
    api_version: ApiVersion
    billed_units: BilledUnits

# ===== Embeddings-Modelle (OpenAI-Format) =====

class EmbeddingData(BaseModel):
    """Einzelnes Embedding-Datum (OpenAI-Format)."""
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    """Token-Usage (OpenAI-Format)."""
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    """Embedding-Response (OpenAI-Format)."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
