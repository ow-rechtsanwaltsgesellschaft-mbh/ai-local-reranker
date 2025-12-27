"""
FastAPI-Anwendung für Docling-Dokumentenverarbeitung.
Läuft auf Port 8890.
"""
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
from app.docling import DoclingService

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Docling Document Processing API",
    description="API für Dokumentenverarbeitung mit Docling (PDF, Word, etc.)",
    version="1.0.0"
)

# Service initialisieren
docling_service = DoclingService()


# ===== Request/Response Models =====

class ConvertRequest(BaseModel):
    """Request-Modell für Dokumentenkonvertierung."""
    file_path: Optional[str] = Field(None, description="Pfad zur Datei (optional, wenn file_bytes verwendet wird)")
    output_format: str = Field("markdown", description="Ausgabeformat: markdown, json, text")
    filename: Optional[str] = Field(None, description="Dateiname (für Format-Erkennung bei file_bytes)")


class ConvertResponse(BaseModel):
    """Response-Modell für Dokumentenkonvertierung."""
    success: bool
    content: Optional[str] = None
    metadata: dict
    error: Optional[str] = None


# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    """Initialisiert Docling-Service beim Start."""
    logger.info("Starte Docling-Service...")
    try:
        await docling_service.initialize()
        logger.info("Docling-Service erfolgreich initialisiert")
    except Exception as e:
        logger.error(f"Fehler beim Initialisieren des Docling-Services: {str(e)}")
        raise


# ===== Health & Info Endpoints =====

@app.get("/")
async def root():
    """Health-Check Endpoint."""
    return {
        "status": "online",
        "service": "Docling Document Processing API",
        "version": "1.0.0"
    }


@app.get("/docling/health")
async def health():
    """Detaillierter Health-Check."""
    return {
        "status": "healthy" if docling_service.is_initialized() else "initializing",
        "initialized": docling_service.is_initialized()
    }


# ===== Document Processing Endpoints =====

@app.post("/docling/v1/convert", response_model=ConvertResponse)
async def convert_document(
    file: UploadFile = File(..., description="Zu konvertierende Datei"),
    output_format: str = Form("markdown", description="Ausgabeformat: markdown, json, text")
):
    """
    Konvertiert ein hochgeladenes Dokument in ein strukturiertes Format.
    
    Unterstützte Formate:
    - PDF
    - DOCX, DOC
    - TXT
    - Und weitere von Docling unterstützte Formate
    
    Ausgabeformate:
    - markdown: Markdown-Format
    - json: JSON-Struktur
    - text: Reiner Text
    """
    try:
        if not docling_service.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="Docling-Service ist noch nicht initialisiert. Bitte versuchen Sie es später erneut."
            )
        
        # Datei lesen
        file_bytes = await file.read()
        filename = file.filename or "document"
        
        # Konvertierung durchführen
        result = await docling_service.convert_from_bytes(
            file_bytes=file_bytes,
            filename=filename,
            output_format=output_format
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Fehler beim Konvertieren: {result.get('error', 'Unbekannter Fehler')}"
            )
        
        # Content formatieren
        content = result.get("content")
        if output_format.lower() == "json" and isinstance(content, dict):
            # JSON als String zurückgeben
            import json
            content = json.dumps(content, ensure_ascii=False, indent=2)
        
        return ConvertResponse(
            success=True,
            content=content,
            metadata=result.get("metadata", {}),
            error=None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Konvertieren des Dokuments: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Interner Serverfehler: {str(e)}"
        )


@app.post("/docling/v1/convert/path", response_model=ConvertResponse)
async def convert_document_from_path(
    request: ConvertRequest
):
    """
    Konvertiert ein Dokument von einem Dateipfad.
    
    Endpoint: POST /docling/v1/convert/path
    
    Hinweis: Funktioniert nur, wenn der Dateipfad innerhalb des Containers erreichbar ist.
    """
    try:
        if not docling_service.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="Docling-Service ist noch nicht initialisiert. Bitte versuchen Sie es später erneut."
            )
        
        if not request.file_path:
            raise HTTPException(
                status_code=400,
                detail="file_path ist erforderlich"
            )
        
        # Konvertierung durchführen
        result = await docling_service.convert_document(
            file_path=request.file_path,
            output_format=request.output_format
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Fehler beim Konvertieren: {result.get('error', 'Unbekannter Fehler')}"
            )
        
        # Content formatieren
        content = result.get("content")
        if request.output_format.lower() == "json" and isinstance(content, dict):
            # JSON als String zurückgeben
            import json
            content = json.dumps(content, ensure_ascii=False, indent=2)
        
        return ConvertResponse(
            success=True,
            content=content,
            metadata=result.get("metadata", {}),
            error=None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Konvertieren des Dokuments: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Interner Serverfehler: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8890,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120
    )

