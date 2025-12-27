"""
FastAPI-Anwendung für lokales Reranking auf CPU.
Verwendet sentence-transformers CrossEncoder für effizientes Reranking.
Kompatibel mit Cohere Rerank API Format.
"""
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from app.reranker import RerankerService
from app.models import RerankResult, Meta, ApiVersion, BilledUnits

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Local Reranker API",
    description="Lokale Reranking-API für CPU-basierte Neuordnung von Dokumenten (Cohere-kompatibel)",
    version="1.0.0"
)

# Reranker-Service initialisieren
reranker_service = RerankerService()


class RerankRequest(BaseModel):
    """Request-Modell für Reranking (Cohere-kompatibel)."""
    query: str = Field(..., description="Die Suchanfrage")
    documents: List[str] = Field(..., description="Liste der zu rerankenden Dokumente")
    top_n: Optional[int] = Field(None, description="Anzahl der Top-Ergebnisse (optional)")
    model: Optional[str] = Field(None, description="Modellname (optional, überschreibt ENV-Variable)")


class RerankResponse(BaseModel):
    """Response-Modell für Reranking (Cohere-kompatibel)."""
    results: List[RerankResult]
    id: str
    meta: Meta


@app.on_event("startup")
async def startup_event():
    """Initialisiert den Reranker beim Start."""
    logger.info("Initialisiere Reranker-Service...")
    await reranker_service.initialize()
    logger.info("Reranker-Service erfolgreich initialisiert")


@app.get("/")
async def root():
    """Health-Check Endpoint."""
    return {
        "status": "online",
        "service": "AI Local Reranker API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detaillierter Health-Check."""
    cached_models = reranker_service.get_cached_models()
    return {
        "status": "healthy",
        "reranker_loaded": reranker_service.is_loaded(),
        "model_name": reranker_service.get_model_name(),
        "cached_models": cached_models,
        "cache_size": len(cached_models)
    }


@app.get("/model/info")
async def model_info():
    """Informationen über das verwendete Modell."""
    from app.reranker import AVAILABLE_MODELS
    
    return {
        "current_model": reranker_service.get_model_name(),
        "available_models": {
            alias: model_name 
            for alias, model_name in AVAILABLE_MODELS.items() 
            if alias != "default"
        },
        "note": "Verwenden Sie die Umgebungsvariable RERANKER_MODEL, um das Modell zu ändern. "
                "Mögliche Werte: 'fast', 'balanced', 'bge-v2', 'bge-large', 'zerank-1', 'zerank-1-small', "
                "'qwen3-0.6b', 'qwen3-4b', 'qwen3-8b', 'bert-german' oder direkter Modellname von HuggingFace."
    }


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Führt Reranking für eine Query und eine Liste von Dokumenten durch.
    Kompatibel mit Cohere Rerank API Format.
    
    Args:
        request: RerankRequest mit query, documents, optional top_n und model
        
    Returns:
        RerankResponse mit gerankten Ergebnissen im Cohere-Format
    """
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="Dokumentenliste darf nicht leer sein")
        
        if not request.query:
            raise HTTPException(status_code=400, detail="Query darf nicht leer sein")
        
        # Verwende angegebenes Modell oder Standard
        model_name = request.model if request.model else None
        
        # Reranking durchführen
        results = await reranker_service.rerank(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n,
            model_name=model_name
        )
        
        # Generiere UUID für Request-ID
        request_id = str(uuid.uuid4())
        
        # Erstelle Meta-Informationen
        meta = Meta(
            api_version=ApiVersion(
                version="1.0",
                is_experimental=False
            ),
            billed_units=BilledUnits(
                search_units=1
            )
        )
        
        return RerankResponse(
            results=results,
            id=request_id,
            meta=meta
        )
    
    except Exception as e:
        logger.error(f"Fehler beim Reranking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Interner Serverfehler: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8888,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120
    )

