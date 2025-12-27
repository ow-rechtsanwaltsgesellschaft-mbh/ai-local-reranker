"""
FastAPI-Anwendung für lokales Reranking und Embeddings.
Modular aufgebaut mit separaten Services für Reranking und Embeddings.
Kompatibel mit Cohere Rerank API und OpenAI Embeddings API.
"""
import uuid
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import logging
from app.reranker import RerankerService
from app.embeddings import EmbeddingsService
from app.models import (
    RerankResult, Meta, ApiVersion, BilledUnits,
    EmbeddingResponse, EmbeddingData, EmbeddingUsage
)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Local Reranker & Embeddings API",
    description="Lokale Reranking- und Embeddings-API (Cohere- und OpenAI-kompatibel)",
    version="1.0.0"
)

# Services initialisieren
reranker_service = RerankerService()
embeddings_service = EmbeddingsService()


# ===== Reranker Request/Response Models =====

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


# ===== Embeddings Request Model =====

class EmbeddingRequest(BaseModel):
    """Request-Modell für Embeddings (OpenAI-Format)."""
    model: str = Field(..., description="Modellname (erforderlich, z.B. 'bge-base', 'bge-large', 'jina-de', 'smollm3-de')")
    input: Union[str, List[str]] = Field(..., description="Text oder Liste von Texten zum Embedden")


# ===== Startup Event =====

@app.on_event("startup")
async def startup_event():
    """Lädt die Modelle beim Start."""
    logger.info("Starte Services...")
    await reranker_service.initialize()
    await embeddings_service.initialize()
    logger.info("Alle Services bereit!")


# ===== Health & Info Endpoints =====

@app.get("/")
async def root():
    """Health-Check Endpoint."""
    return {
        "status": "online",
        "service": "AI Local Reranker & Embeddings API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detaillierter Health-Check."""
    cached_reranker_models = reranker_service.get_cached_models()
    cached_embedding_models = embeddings_service.get_cached_models()
    return {
        "status": "healthy",
        "reranker_loaded": reranker_service.is_loaded(),
        "reranker_model": reranker_service.get_model_name(),
        "embeddings_loaded": embeddings_service.is_loaded(),
        "embeddings_model": embeddings_service.get_model_name(),
        "cached_reranker_models": cached_reranker_models,
        "cached_embedding_models": cached_embedding_models
    }


@app.get("/model/info")
async def model_info():
    """Informationen über die verwendeten Modelle."""
    from app.reranker import AVAILABLE_MODELS
    from app.embeddings import AVAILABLE_EMBEDDING_MODELS
    
    return {
        "reranker": {
            "current_model": reranker_service.get_model_name(),
            "available_models": {
                alias: model_name 
                for alias, model_name in AVAILABLE_MODELS.items() 
                if alias != "default"
            }
        },
        "embeddings": {
            "current_model": embeddings_service.get_model_name(),
            "available_models": {
                alias: model_name 
                for alias, model_name in AVAILABLE_EMBEDDING_MODELS.items() 
                if alias != "default"
            }
        }
    }


# ===== Reranker Endpoints (Cohere-Format) =====

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Führt Reranking für eine Query und eine Liste von Dokumenten durch.
    Kompatibel mit Cohere Rerank API Format.
    
    Endpoint: POST /v1/rerank
    
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


# ===== Embeddings Endpoints (OpenAI-Format) =====

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Erstellt Embeddings für einen Text oder eine Liste von Texten (OpenAI-Format).
    
    Endpoint: POST /v1/embeddings
    
    Request-Format (OpenAI-kompatibel):
    {
      "model": "bge-base",
      "input": "Text zum Embedden" oder ["Text1", "Text2"]
    }
    
    Unterstützte Modelle:
    - bge-large: BAAI/bge-large-en-v1.5
    - bge-base: BAAI/bge-base-en-v1.5
    - jina-de: jinaai/jina-embeddings-v2-base-de
    - smollm3-de: mayflowergmbh/smollm3-3b-german-embed
    
    Args:
        request: EmbeddingRequest mit model (Modellname) und input (Text oder Liste)
        
    Returns:
        EmbeddingResponse im OpenAI-Format
    """
    try:
        # Modellname kommt aus Request-Body (OpenAI-Standard, erforderlich)
        final_model_name = request.model
        
        # Normalisiere input zu Liste
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise HTTPException(status_code=400, detail="Input darf nicht leer sein")
        
        # Erstelle Embeddings
        embeddings = await embeddings_service.embed(
            texts=texts,
            model_name=final_model_name
        )
        
        # Erstelle Response im OpenAI-Format
        embedding_data = [
            EmbeddingData(
                object="embedding",
                embedding=emb,
                index=i
            )
            for i, emb in enumerate(embeddings)
        ]
        
        # Verwende Modellname für Response
        used_model = final_model_name or embeddings_service.get_model_name()
        
        # Schätze Token-Usage (vereinfacht: ~1 Token pro Wort)
        total_tokens = sum(len(text.split()) for text in texts)
        
        return EmbeddingResponse(
            object="list",
            data=embedding_data,
            model=used_model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )
    
    except Exception as e:
        logger.error(f"Fehler beim Erstellen von Embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Interner Serverfehler: {str(e)}")


@app.get("/v1/models")
async def list_models():
    """
    Listet verfügbare Embedding-Modelle auf (OpenAI-Format).
    """
    from app.embeddings import AVAILABLE_EMBEDDING_MODELS
    
    models = []
    for alias, model_id in AVAILABLE_EMBEDDING_MODELS.items():
        if alias != "default":
            models.append({
                "id": model_id,
                "object": "model",
                "created": 1677610602,  # Placeholder timestamp
                "owned_by": "local"
            })
    
    return {
        "object": "list",
        "data": models
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8888,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120
    )
