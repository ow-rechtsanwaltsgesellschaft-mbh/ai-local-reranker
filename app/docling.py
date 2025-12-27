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
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            logger.info("Initialisiere Docling-Pipeline...")
            
            # Pipeline-Optionen konfigurieren, um Bilder zu extrahieren
            try:
                # Versuche, Pipeline-Optionen zu konfigurieren
                pipeline_options = PdfPipelineOptions()
                # Stelle sicher, dass Bilder extrahiert werden
                if hasattr(pipeline_options, 'do_ocr'):
                    pipeline_options.do_ocr = True
                if hasattr(pipeline_options, 'do_table_structure'):
                    pipeline_options.do_table_structure = True
                
                self.pipeline = DocumentConverter(pipeline_options=pipeline_options)
                logger.info("Docling-Pipeline mit erweiterten Optionen initialisiert")
            except Exception as config_error:
                logger.debug(f"Konfiguration mit Pipeline-Optionen fehlgeschlagen, verwende Standard: {str(config_error)}")
                # Fallback: Standard-Pipeline ohne spezielle Konfiguration
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
                
                # Dokument-Metadaten extrahieren
                metadata = {}
                try:
                    doc = result.document
                    if hasattr(doc, 'metadata') and doc.metadata:
                        metadata = {
                            "title": getattr(doc.metadata, 'title', None),
                            "author": getattr(doc.metadata, 'author', None),
                            "language": getattr(doc.metadata, 'language', None),
                            "subject": getattr(doc.metadata, 'subject', None),
                            "creator": getattr(doc.metadata, 'creator', None),
                            "producer": getattr(doc.metadata, 'producer', None),
                            "creation_date": str(getattr(doc.metadata, 'creation_date', None)) if hasattr(doc.metadata, 'creation_date') else None,
                            "modification_date": str(getattr(doc.metadata, 'modification_date', None)) if hasattr(doc.metadata, 'modification_date') else None,
                        }
                    # Seitenanzahl hinzufügen
                    if hasattr(doc, 'pages') and doc.pages:
                        metadata["page_count"] = len(doc.pages)
                except Exception as e:
                    logger.warning(f"Fehler beim Extrahieren der Metadaten: {str(e)}")
                
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
                        try:
                            # Debug: Prüfe verfügbare Attribute
                            logger.debug(f"Seite {i} - Verfügbare Attribute: {dir(page)}")
                            
                            # Versuche verschiedene Quellen für Bilder
                            page_images = []
                            
                            # 1. Prüfe page.images
                            if hasattr(page, 'images') and page.images:
                                logger.debug(f"Seite {i} - Gefundene Bilder in page.images: {len(page.images)}")
                                page_images.extend(page.images)
                            
                            # 2. Prüfe page.content für Bilder
                            if hasattr(page, 'content') and page.content:
                                logger.debug(f"Seite {i} - page.content vorhanden: {type(page.content)}")
                                if hasattr(page.content, 'images') and page.content.images:
                                    logger.debug(f"Seite {i} - Gefundene Bilder in page.content.images: {len(page.content.images)}")
                                    page_images.extend(page.content.images)
                            
                            # 3. Prüfe das gesamte Dokument für seiten-spezifische Bilder
                            if hasattr(result.document, 'images') and result.document.images:
                                logger.debug(f"Seite {i} - Gefundene Bilder im Dokument: {len(result.document.images)}")
                                # Filtere Bilder, die zu dieser Seite gehören
                                for doc_img in result.document.images:
                                    # Prüfe, ob das Bild zu dieser Seite gehört (z.B. über bbox)
                                    if hasattr(doc_img, 'page') and doc_img.page == i:
                                        page_images.append(doc_img)
                                    elif hasattr(doc_img, 'bbox'):
                                        # Prüfe, ob bbox innerhalb der Seiten-Bbox liegt
                                        if hasattr(page, 'bbox') and page.bbox:
                                            page_bbox = page.bbox
                                            img_bbox = doc_img.bbox
                                            # Einfache Überlappungsprüfung
                                            if (img_bbox[1] >= page_bbox[1] and img_bbox[1] <= page_bbox[3]):
                                                page_images.append(doc_img)
                            
                            # 4. Prüfe page.items für Bilder
                            if hasattr(page, 'items') and page.items:
                                logger.debug(f"Seite {i} - Gefundene Items: {len(page.items)}")
                                for item in page.items:
                                    if hasattr(item, 'type') and 'image' in str(item.type).lower():
                                        logger.debug(f"Seite {i} - Bild gefunden in items (type): {item.type}")
                                        page_images.append(item)
                                    elif hasattr(item, '__class__') and 'image' in str(item.__class__).lower():
                                        logger.debug(f"Seite {i} - Bild gefunden in items (class): {item.__class__}")
                                        page_images.append(item)
                            
                            logger.debug(f"Seite {i} - Gesamt gefundene Bilder: {len(page_images)}")
                            
                            # Verarbeite gefundene Bilder
                            for img in page_images:
                                img_data = {}
                                
                                # Bounding Box extrahieren
                                if hasattr(img, 'bbox') and img.bbox:
                                    img_data["bbox"] = list(img.bbox) if img.bbox else []
                                elif hasattr(img, 'bounds') and img.bounds:
                                    img_data["bbox"] = list(img.bounds) if img.bounds else []
                                elif hasattr(img, 'rect') and img.rect:
                                    # Konvertiere rect zu bbox
                                    rect = img.rect
                                    if hasattr(rect, 'x') and hasattr(rect, 'y') and hasattr(rect, 'width') and hasattr(rect, 'height'):
                                        img_data["bbox"] = [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height]
                                
                                # Base64-Kodierung (falls aktiviert)
                                if include_images_base64:
                                    try:
                                        import base64
                                        from io import BytesIO
                                        from PIL import Image
                                        
                                        # Versuche verschiedene Wege, das Bild zu extrahieren
                                        image_data = None
                                        
                                        # 1. Direktes PIL Image
                                        if hasattr(img, 'image') and img.image:
                                            image_data = img.image
                                        # 2. Bilddaten als Bytes
                                        elif hasattr(img, 'data') and img.data:
                                            image_data = Image.open(BytesIO(img.data))
                                        # 3. Bildpfad
                                        elif hasattr(img, 'path') and img.path:
                                            image_data = Image.open(img.path)
                                        # 4. Bild-URL oder Pfad als String
                                        elif hasattr(img, 'src') and img.src:
                                            try:
                                                image_data = Image.open(img.src)
                                            except:
                                                pass
                                        
                                        if image_data:
                                            buffered = BytesIO()
                                            # Konvertiere zu RGB falls nötig
                                            if image_data.mode in ('RGBA', 'LA', 'P'):
                                                rgb_image = Image.new('RGB', image_data.size, (255, 255, 255))
                                                if image_data.mode == 'P':
                                                    image_data = image_data.convert('RGBA')
                                                rgb_image.paste(image_data, mask=image_data.split()[-1] if image_data.mode == 'RGBA' else None)
                                                image_data = rgb_image
                                            image_data.save(buffered, format="PNG")
                                            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                            img_data["base64"] = img_base64
                                    except Exception as e:
                                        logger.debug(f"Fehler beim Base64-Kodieren des Bildes: {str(e)}")
                                
                                # Nur hinzufügen, wenn mindestens bbox oder base64 vorhanden
                                if img_data.get("bbox") or img_data.get("base64"):
                                    images.append(img_data)
                        
                        except Exception as e:
                            logger.debug(f"Fehler beim Extrahieren der Bilder von Seite {i}: {str(e)}")
                        
                        # Tabellen extrahieren (falls verfügbar)
                        tables = []
                        try:
                            # Prüfe, ob die Seite Tabellen hat
                            if hasattr(page, 'tables') and page.tables:
                                for table in page.tables:
                                    table_data = {}
                                    
                                    # Bounding Box
                                    if hasattr(table, 'bbox') and table.bbox:
                                        table_data["bbox"] = list(table.bbox)
                                    
                                    # Tabellen-Struktur extrahieren
                                    rows = []
                                    columns = None
                                    
                                    # Versuche, Tabellenstruktur zu extrahieren
                                    if hasattr(table, 'rows') and table.rows:
                                        for row in table.rows:
                                            row_data = []
                                            if hasattr(row, 'cells') and row.cells:
                                                for cell in row.cells:
                                                    cell_text = ""
                                                    if hasattr(cell, 'text'):
                                                        cell_text = cell.text
                                                    elif hasattr(cell, 'content'):
                                                        cell_text = str(cell.content)
                                                    row_data.append(cell_text)
                                            rows.append(row_data)
                                            
                                            # Erste Zeile als Spaltenüberschriften verwenden
                                            if columns is None and row_data:
                                                columns = row_data
                                    elif hasattr(table, 'cells'):
                                        # Alternative Struktur: direkter Zugriff auf Zellen
                                        # Versuche, eine Matrix zu erstellen
                                        cells = table.cells
                                        if isinstance(cells, list) and len(cells) > 0:
                                            # Annahme: erste Zeile enthält Spaltenüberschriften
                                            if len(cells) > 0:
                                                first_row = cells[0] if isinstance(cells[0], list) else [cells[0]]
                                                columns = [str(cell) for cell in first_row]
                                                # Rest als Zeilen
                                                for row_cells in cells[1:]:
                                                    if isinstance(row_cells, list):
                                                        rows.append([str(cell) for cell in row_cells])
                                                    else:
                                                        rows.append([str(row_cells)])
                                    
                                    # Falls keine Struktur gefunden, versuche Markdown zu parsen
                                    if not rows and hasattr(table, 'to_markdown'):
                                        try:
                                            table_markdown = table.to_markdown()
                                            # Einfacher Parser für Markdown-Tabellen
                                            lines = table_markdown.strip().split('\n')
                                            if len(lines) > 1:
                                                # Erste Zeile als Spalten
                                                columns = [col.strip() for col in lines[0].split('|') if col.strip()]
                                                # Rest als Zeilen
                                                for line in lines[2:]:  # Überspringen der Trennlinie
                                                    row = [col.strip() for col in line.split('|') if col.strip()]
                                                    if row:
                                                        rows.append(row)
                                        except Exception as e:
                                            logger.debug(f"Fehler beim Parsen der Tabelle als Markdown: {str(e)}")
                                    
                                    table_data["rows"] = rows
                                    if columns:
                                        table_data["columns"] = columns
                                    
                                    tables.append(table_data)
                        except Exception as e:
                            logger.warning(f"Fehler beim Extrahieren der Tabellen: {str(e)}")
                        
                        page_info = {
                            "index": i,
                            "markdown": page_markdown,
                            "dimensions": dimensions,
                            "images": images,
                            "tables": tables
                        }
                        pages_data.append(page_info)
                else:
                    # Falls keine Seiteninformationen verfügbar, gesamtes Dokument als eine Seite
                    try:
                        full_markdown = result.document.export_to_markdown()
                        
                        # Versuche, Tabellen aus dem gesamten Dokument zu extrahieren
                        tables = []
                        try:
                            if hasattr(result.document, 'tables') and result.document.tables:
                                for table in result.document.tables:
                                    table_data = {}
                                    if hasattr(table, 'bbox') and table.bbox:
                                        table_data["bbox"] = list(table.bbox)
                                    
                                    rows = []
                                    columns = None
                                    if hasattr(table, 'rows') and table.rows:
                                        for row in table.rows:
                                            row_data = []
                                            if hasattr(row, 'cells') and row.cells:
                                                for cell in row.cells:
                                                    cell_text = ""
                                                    if hasattr(cell, 'text'):
                                                        cell_text = cell.text
                                                    elif hasattr(cell, 'content'):
                                                        cell_text = str(cell.content)
                                                    row_data.append(cell_text)
                                            rows.append(row_data)
                                            if columns is None and row_data:
                                                columns = row_data
                                    table_data["rows"] = rows
                                    if columns:
                                        table_data["columns"] = columns
                                    tables.append(table_data)
                        except Exception as e:
                            logger.debug(f"Fehler beim Extrahieren der Tabellen aus dem Dokument: {str(e)}")
                        
                        pages_data.append({
                            "index": 0,
                            "markdown": full_markdown,
                            "dimensions": None,
                            "images": [],
                            "tables": tables
                        })
                    except Exception as e:
                        logger.warning(f"Fehler beim Extrahieren des gesamten Dokuments: {str(e)}")
                        pages_data.append({
                            "index": 0,
                            "markdown": "",
                            "dimensions": None,
                            "images": [],
                            "tables": []
                        })
                
                return {
                    "pages": pages_data,
                    "metadata": metadata if metadata else None,
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

