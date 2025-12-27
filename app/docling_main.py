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
    include_images_base64: bool = Field(False, description="Bilder als Base64 kodieren (wie Mistral OCR)")


class TableData(BaseModel):
    """Tabellen-Daten im Mistral OCR Format."""
    bbox: Optional[List[float]] = None
    rows: List[List[str]] = Field(default_factory=list)
    columns: Optional[List[str]] = None


class PageData(BaseModel):
    """Seiten-Daten im Mistral OCR Format."""
    index: int
    markdown: str
    dimensions: Optional[dict] = None
    images: List[dict] = Field(default_factory=list)
    tables: List[TableData] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Dokument-Metadaten im Mistral OCR Format."""
    title: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: Optional[int] = None


class ConvertResponse(BaseModel):
    """Response-Modell für Dokumentenkonvertierung (Mistral OCR Format)."""
    pages: List[PageData]
    metadata: Optional[DocumentMetadata] = None


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
    output_format: str = Form("markdown", description="Ausgabeformat: markdown, json, text"),
    include_images_base64: bool = Form(False, description="Bilder als Base64 kodieren (wie Mistral OCR)")
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
            output_format=output_format,
            include_images_base64=include_images_base64
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Fehler beim Konvertieren: {result.get('error', 'Unbekannter Fehler')}"
            )
        
        # Response im Mistral OCR Format zurückgeben
        pages_data = []
        for page_dict in result.get("pages", []):
            # Tabellen-Daten konvertieren
            tables_data = []
            for table_dict in page_dict.get("tables", []):
                tables_data.append(TableData(**table_dict))
            page_dict["tables"] = tables_data
            pages_data.append(PageData(**page_dict))
        
        # Metadaten konvertieren
        metadata = None
        if result.get("metadata"):
            metadata = DocumentMetadata(**result["metadata"])
        
        return ConvertResponse(pages=pages_data, metadata=metadata)
    
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
            output_format=request.output_format,
            include_images_base64=request.include_images_base64
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Fehler beim Konvertieren: {result.get('error', 'Unbekannter Fehler')}"
            )
        
        # Response im Mistral OCR Format zurückgeben
        pages_data = []
        for page_dict in result.get("pages", []):
            # Tabellen-Daten konvertieren
            tables_data = []
            for table_dict in page_dict.get("tables", []):
                tables_data.append(TableData(**table_dict))
            page_dict["tables"] = tables_data
            pages_data.append(PageData(**page_dict))
        
        # Metadaten konvertieren
        metadata = None
        if result.get("metadata"):
            metadata = DocumentMetadata(**result["metadata"])
        
        return ConvertResponse(pages=pages_data, metadata=metadata)
    
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

