"""
Datenmodelle für die Reranker-API.
Kompatibel mit Cohere Rerank API Format.
"""
from pydantic import BaseModel
from typing import Optional


class RerankResult(BaseModel):
    """Einzelnes Reranking-Ergebnis (Cohere-kompatibel)."""
    index: int
    relevance_score: float


class ApiVersion(BaseModel):
    """API-Versionsinformationen."""
    version: str
    is_experimental: bool


class BilledUnits(BaseModel):
    """Abrechnungseinheiten."""
    search_units: int


class Meta(BaseModel):
    """Metadaten für die Response."""
    api_version: ApiVersion
    billed_units: BilledUnits

