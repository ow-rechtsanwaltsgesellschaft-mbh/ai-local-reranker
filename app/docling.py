"""
Docling-Service für Dokumentenverarbeitung (PDF, Word, etc.).
Konvertiert Dokumente in strukturierte Formate (Markdown, JSON, etc.).
"""
import os
from typing import Optional, Dict, Any, List
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


def get_hf_token() -> Optional[str]:
    """
    Liest den Hugging Face Token aus Umgebungsvariablen.
    
    Returns:
        HF Token oder None
    """
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


class DoclingService:
    """
    Service für Dokumentenverarbeitung mit Docling.
    
    Unterstützt verschiedene Dokumentformate (PDF, DOCX, etc.)
    und konvertiert sie in strukturierte Formate.
    """
    
    def __init__(self):
        """Initialisiert den Docling-Service."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialized = False
        self.pipeline = None
    
    async def initialize(self):
        """Lädt Docling-Pipeline asynchron."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._load_pipeline)
    
    def _load_pipeline(self):
        """Lädt die Docling-Pipeline."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            
            logger.info("Initialisiere Docling-Pipeline...")
            
            # Konfiguration für Docling
            config = {
                "doctr_model": "db_resnet50",  # OCR-Modell
                "tables": {
                    "detection_mode": "auto",
                    "structure_mode": "auto"
                }
            }
            
            # Pipeline erstellen
            self.pipeline = DocumentConverter(
                format=InputFormat.AUTO,
                **config
            )
            
            self._initialized = True
            logger.info("Docling-Pipeline erfolgreich initialisiert")
            
        except ImportError as e:
            logger.error(f"Docling ist nicht installiert: {str(e)}")
            logger.error("Bitte installieren Sie: pip install docling")
            raise
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren der Docling-Pipeline: {str(e)}")
            raise
    
    def is_initialized(self) -> bool:
        """Prüft, ob die Pipeline initialisiert ist."""
        return self._initialized
    
    async def convert_document(
        self,
        file_path: str,
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Konvertiert ein Dokument in ein strukturiertes Format.
        
        Args:
            file_path: Pfad zur Datei
            output_format: Ausgabeformat ("markdown", "json", "text")
            
        Returns:
            Dictionary mit konvertiertem Inhalt und Metadaten
        """
        if not self._initialized:
            raise RuntimeError("Docling-Pipeline ist nicht initialisiert")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        loop = asyncio.get_event_loop()
        
        def convert():
            try:
                # Dokument konvertieren
                result = self.pipeline.convert(file_path)
                
                # Format-spezifische Ausgabe
                if output_format.lower() == "markdown":
                    content = result.document.export_to_markdown()
                elif output_format.lower() == "json":
                    content = result.document.export_to_dict()
                elif output_format.lower() == "text":
                    content = result.document.export_to_text()
                else:
                    content = result.document.export_to_markdown()
                
                # Metadaten extrahieren
                metadata = {
                    "file_path": file_path,
                    "format": output_format,
                    "pages": len(result.document.pages) if hasattr(result.document, 'pages') else None,
                    "tables": len(result.document.tables) if hasattr(result.document, 'tables') else 0,
                }
                
                return {
                    "content": content,
                    "metadata": metadata,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Fehler beim Konvertieren des Dokuments: {str(e)}")
                return {
                    "content": None,
                    "metadata": {"file_path": file_path},
                    "success": False,
                    "error": str(e)
                }
        
        return await loop.run_in_executor(self.executor, convert)
    
    async def convert_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Konvertiert ein Dokument aus Bytes in ein strukturiertes Format.
        
        Args:
            file_bytes: Dateiinhalt als Bytes
            filename: Dateiname (für Format-Erkennung)
            output_format: Ausgabeformat ("markdown", "json", "text")
            
        Returns:
            Dictionary mit konvertiertem Inhalt und Metadaten
        """
        if not self._initialized:
            raise RuntimeError("Docling-Pipeline ist nicht initialisiert")
        
        # Temporäre Datei erstellen
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        try:
            result = await self.convert_document(tmp_path, output_format)
            return result
        finally:
            # Temporäre Datei löschen
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

