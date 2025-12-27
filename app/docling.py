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
            
            logger.info("Initialisiere Docling-Pipeline...")
            
            # Pipeline erstellen (ohne spezifische Konfiguration, verwendet Standardeinstellungen)
            self.pipeline = DocumentConverter()
            
            self._initialized = True
            logger.info("Docling-Pipeline erfolgreich initialisiert")
            
        except ImportError as e:
            logger.error(f"Docling ist nicht installiert: {str(e)}")
            logger.error("Bitte installieren Sie: pip install docling")
            raise
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren der Docling-Pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def is_initialized(self) -> bool:
        """Prüft, ob die Pipeline initialisiert ist."""
        return self._initialized
    
    async def convert_document(
        self,
        file_path: str,
        output_format: str = "markdown",
        include_images_base64: bool = False
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
                import time
                start_time = time.time()
                
                # Dokument konvertieren
                result = self.pipeline.convert(file_path)
                
                processing_time = time.time() - start_time
                logger.debug(f"Dokument konvertiert in {processing_time:.2f}s")
                
                # Zuerst das gesamte Dokument als Markdown extrahieren
                try:
                    full_markdown = result.document.export_to_markdown()
                    logger.debug(f"Gesamtes Dokument extrahiert: {len(full_markdown)} Zeichen")
                except Exception as e:
                    logger.warning(f"Fehler beim Extrahieren des gesamten Markdowns: {str(e)}")
                    full_markdown = ""
                
                # Seiteninformationen im Mistral OCR Format extrahieren
                pages_data = []
                if hasattr(result.document, 'pages') and result.document.pages:
                    total_pages = len(result.document.pages)
                    logger.debug(f"Gefundene Seiten: {total_pages}")
                    
                    # Versuche, den Markdown-Inhalt nach Seiten aufzuteilen
                    # Docling fügt manchmal Seitenumbrüche ein, die wir nutzen können
                    markdown_parts = []
                    if full_markdown and total_pages > 0:
                        # Versuche nach Seitenumbrüchen zu splitten (falls vorhanden)
                        if "\n\n---\n\n" in full_markdown:
                            parts = full_markdown.split("\n\n---\n\n")
                            if len(parts) == total_pages:
                                markdown_parts = parts
                            else:
                                # Gleichmäßige Aufteilung
                                chunk_size = len(full_markdown) // total_pages
                                markdown_parts = [full_markdown[i:i+chunk_size] for i in range(0, len(full_markdown), chunk_size)]
                                if len(markdown_parts) > total_pages:
                                    markdown_parts = markdown_parts[:total_pages]
                        else:
                            # Gleichmäßige Aufteilung
                            chunk_size = len(full_markdown) // total_pages
                            markdown_parts = [full_markdown[i:i+chunk_size] for i in range(0, len(full_markdown), chunk_size)]
                            if len(markdown_parts) > total_pages:
                                markdown_parts = markdown_parts[:total_pages]
                    
                    # Falls keine Aufteilung möglich, verwende gesamten Inhalt für jede Seite
                    if not markdown_parts:
                        markdown_parts = [full_markdown] * total_pages
                    
                    # Stelle sicher, dass wir genug Teile haben
                    while len(markdown_parts) < total_pages:
                        markdown_parts.append("")
                    
                    for i, page in enumerate(result.document.pages):
                        # Verwende den entsprechenden Markdown-Teil
                        page_markdown = markdown_parts[i] if i < len(markdown_parts) else full_markdown
                        
                        # Dimensions-Informationen (falls verfügbar)
                        dimensions = None
                        if hasattr(page, 'bbox') and page.bbox:
                            bbox = page.bbox
                            if len(bbox) >= 4:
                                width = bbox[2] - bbox[0] if bbox[2] > bbox[0] else None
                                height = bbox[3] - bbox[1] if bbox[3] > bbox[1] else None
                                if width and height:
                                    dimensions = {
                                        "dpi": 200,  # Standard-DPI, kann aus Metadaten extrahiert werden
                                        "height": int(height),
                                        "width": int(width)
                                    }
                        
                        # Bilder extrahieren (falls verfügbar)
                        images = []
                        if hasattr(page, 'images') and page.images:
                            for img in page.images:
                                img_data = {}
                                if hasattr(img, 'bbox'):
                                    img_data["bbox"] = list(img.bbox) if img.bbox else []
                                
                                # Base64-Kodierung (falls aktiviert)
                                if include_images_base64:
                                    try:
                                        import base64
                                        from io import BytesIO
                                        from PIL import Image
                                        
                                        # Versuche, das Bild zu extrahieren und zu kodieren
                                        if hasattr(img, 'image') and img.image:
                                            # Konvertiere PIL Image zu Base64
                                            buffered = BytesIO()
                                            img.image.save(buffered, format="PNG")
                                            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                            img_data["base64"] = img_base64
                                        elif hasattr(img, 'data') and img.data:
                                            # Falls Bilddaten direkt verfügbar sind
                                            img_data["base64"] = base64.b64encode(img.data).decode('utf-8')
                                    except Exception as e:
                                        logger.warning(f"Fehler beim Base64-Kodieren des Bildes: {str(e)}")
                                
                                images.append(img_data)
                        
                        page_info = {
                            "index": i,
                            "markdown": page_markdown,
                            "dimensions": dimensions,
                            "images": images
                        }
                        pages_data.append(page_info)
                else:
                    # Falls keine Seiteninformationen verfügbar, gesamtes Dokument als eine Seite
                    try:
                        full_markdown = result.document.export_to_markdown()
                        pages_data.append({
                            "index": 0,
                            "markdown": full_markdown,
                            "dimensions": None,
                            "images": []
                        })
                    except Exception as e:
                        logger.warning(f"Fehler beim Extrahieren des gesamten Dokuments: {str(e)}")
                        pages_data.append({
                            "index": 0,
                            "markdown": "",
                            "dimensions": None,
                            "images": []
                        })
                
                return {
                    "pages": pages_data,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Fehler beim Konvertieren des Dokuments: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    "pages": [],
                    "success": False,
                    "error": str(e)
                }
        
        return await loop.run_in_executor(self.executor, convert)
    
    async def convert_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        output_format: str = "markdown",
        include_images_base64: bool = False
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
            result = await self.convert_document(tmp_path, output_format, include_images_base64)
            return result
        finally:
            # Temporäre Datei löschen
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

