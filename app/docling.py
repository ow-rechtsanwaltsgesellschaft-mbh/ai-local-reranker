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
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfFormatOption
                
                # WICHTIG: Bilder müssen explizit generiert werden (wie im Beispiel)
                pipeline_options = PdfPipelineOptions()
                pipeline_options.images_scale = 2.0  # Höhere Auflösung für Bilder
                pipeline_options.generate_page_images = True  # Seitenbilder generieren
                pipeline_options.generate_picture_images = True  # Bild-Elemente generieren
                
                # Erstelle DocumentConverter mit konfigurierten Optionen (wie im Beispiel)
                self.pipeline = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                logger.info("Docling-Pipeline mit Bild-Generierung konfiguriert")
            except Exception as config_error:
                logger.warning(f"Konfiguration mit Bild-Optionen fehlgeschlagen, verwende Standard: {str(config_error)}")
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
                
                # DEBUG: Prüfe die Struktur des result-Objekts
                logger.info(f"Result-Typ: {type(result)}")
                if hasattr(result, 'document'):
                    logger.info(f"Document-Attribute: {[attr for attr in dir(result.document) if not attr.startswith('_')]}")
                    # Prüfe pictures (nicht images!)
                    if hasattr(result.document, 'pictures'):
                        pic_count = len(result.document.pictures) if result.document.pictures else 0
                        logger.info(f"Anzahl Bilder (pictures) im Dokument: {pic_count}")
                        if result.document.pictures and len(result.document.pictures) > 0:
                            for idx, pic in enumerate(result.document.pictures[:3]):  # Erste 3 Bilder
                                logger.info(f"Bild {idx} - Typ: {type(pic)}, Attribute: {[attr for attr in dir(pic) if not attr.startswith('_')]}")
                                if hasattr(pic, 'bbox'):
                                    logger.info(f"Bild {idx} - BBox: {pic.bbox}")
                                if hasattr(pic, 'page'):
                                    logger.info(f"Bild {idx} - Page: {pic.page}")
                                elif hasattr(pic, 'page_index'):
                                    logger.info(f"Bild {idx} - Page Index: {pic.page_index}")
                                # Prüfe auch prov für Seitenzuordnung
                                if hasattr(pic, 'prov') and pic.prov:
                                    logger.info(f"Bild {idx} - Prov: {pic.prov}")
                                    if isinstance(pic.prov, dict):
                                        logger.info(f"Bild {idx} - Prov Keys: {list(pic.prov.keys())}")
                                        if 'page' in pic.prov:
                                            logger.info(f"Bild {idx} - Prov.page: {pic.prov.get('page')}")
                                        if 'page_index' in pic.prov:
                                            logger.info(f"Bild {idx} - Prov.page_index: {pic.prov.get('page_index')}")
                                # Prüfe auch content_layer für BBox und Image
                                if hasattr(pic, 'content_layer'):
                                    logger.info(f"Bild {idx} - Content Layer vorhanden: {type(pic.content_layer)}")
                                    if hasattr(pic.content_layer, 'bbox'):
                                        logger.info(f"Bild {idx} - Content Layer BBox: {pic.content_layer.bbox}")
                                    if hasattr(pic.content_layer, 'image'):
                                        logger.info(f"Bild {idx} - Content Layer Image: {type(pic.content_layer.image) if pic.content_layer.image else None}")
                                    if hasattr(pic.content_layer, 'data'):
                                        logger.info(f"Bild {idx} - Content Layer Data: {type(pic.content_layer.data) if pic.content_layer.data else None}")
                                # Prüfe get_image() direkt
                                if hasattr(pic, 'get_image') and callable(pic.get_image):
                                    try:
                                        test_img = pic.get_image()
                                        logger.info(f"Bild {idx} - get_image() Ergebnis: {type(test_img) if test_img else None}")
                                    except Exception as e:
                                        logger.info(f"Bild {idx} - get_image() Fehler: {str(e)}")
                    # Prüfe auch pages
                    if hasattr(result.document, 'pages'):
                        if isinstance(result.document.pages, dict):
                            logger.info(f"Pages ist ein Dictionary mit {len(result.document.pages)} Einträgen")
                            if result.document.pages:
                                first_key = list(result.document.pages.keys())[0]
                                logger.info(f"Erster Seiten-Key: {first_key}, Typ: {type(first_key)}, Wert: {result.document.pages[first_key]}")
                        elif isinstance(result.document.pages, (list, tuple)):
                            logger.info(f"Anzahl Seiten: {len(result.document.pages)}")
                            if result.document.pages and len(result.document.pages) > 0:
                                logger.info(f"Erste Seite - Typ: {type(result.document.pages[0])}, Wert: {result.document.pages[0]}")
                        else:
                            logger.info(f"Pages-Typ: {type(result.document.pages)}, Wert: {result.document.pages}")
                
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
                
                # Zuerst das gesamte Dokument extrahieren (je nach output_format)
                logger.info(f"output_format Parameter: {output_format}")
                full_markdown = ""
                try:
                    if output_format.lower() == "json":
                        # JSON-Format
                        full_markdown = result.document.export_to_markdown()  # Docling hat kein export_to_json, verwende Markdown
                        logger.debug(f"Gesamtes Dokument als JSON/Markdown extrahiert: {len(full_markdown)} Zeichen")
                    elif output_format.lower() == "text":
                        # Text-Format
                        full_markdown = result.document.export_to_text()
                        logger.debug(f"Gesamtes Dokument als Text extrahiert: {len(full_markdown)} Zeichen")
                    else:
                        # Markdown-Format (Standard)
                        full_markdown = result.document.export_to_markdown()
                        logger.debug(f"Gesamtes Dokument als Markdown extrahiert: {len(full_markdown)} Zeichen")
                except Exception as e:
                    logger.warning(f"Fehler beim Extrahieren des Dokuments: {str(e)}")
                    full_markdown = ""
                
                # DEBUG: Prüfe include_images_base64
                logger.info(f"include_images_base64 Parameter: {include_images_base64}")
                
                # Seiteninformationen im Mistral OCR Format extrahieren
                pages_data = []
                if hasattr(result.document, 'pages') and result.document.pages:
                    # pages könnte ein Dictionary oder eine Liste sein
                    if isinstance(result.document.pages, dict):
                        # Sortiere Keys numerisch (0, 1, 2, ...) für korrekte Reihenfolge
                        page_keys = sorted([k for k in result.document.pages.keys() if isinstance(k, (int, float))], key=lambda x: float(x))
                        total_pages = len(page_keys)
                        logger.debug(f"Gefundene Seiten (Dict): {total_pages}, Keys: {page_keys}")
                        pages_are_objects = True
                    else:
                        total_pages = len(result.document.pages)
                        logger.debug(f"Gefundene Seiten (List): {total_pages}")
                        first_page = result.document.pages[0] if result.document.pages else None
                        pages_are_objects = first_page and not isinstance(first_page, (int, float))
                        page_keys = list(range(total_pages))
                    
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
                    
                    # Iteriere über Seiten
                    for idx, page_key in enumerate(page_keys):
                        i = int(page_key)  # Seiten-Key (kann 0, 1, 2, ... oder 1, 2, 3, ... sein)
                        if isinstance(result.document.pages, dict):
                            page = result.document.pages[page_key]  # Page-Objekt
                        else:
                            page = result.document.pages[idx] if pages_are_objects else i
                        
                        # Berechne den Markdown-Index: Wenn Keys bei 1 beginnen, mappe auf 0-basierten Index
                        # Keys: [1, 2] -> markdown_index: [0, 1]
                        # Keys: [0, 1] -> markdown_index: [0, 1]
                        if page_keys and min(page_keys) > 0:
                            # Keys beginnen bei 1 oder höher, mappe auf 0-basierten Index
                            markdown_index = i - min(page_keys)
                        else:
                            # Keys beginnen bei 0
                            markdown_index = i
                        
                        # Verwende den entsprechenden Markdown-Teil
                        page_markdown = markdown_parts[markdown_index] if 0 <= markdown_index < len(markdown_parts) else full_markdown
                        
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
                        
                        # Seitenbild extrahieren (falls verfügbar und include_images_base64 aktiviert)
                        # Seiten haben ein image-Attribut mit pil_image (wie im Beispiel)
                        page_image = None
                        if include_images_base64 and hasattr(page, 'image') and page.image:
                            if hasattr(page.image, 'pil_image') and page.image.pil_image:
                                try:
                                    page_image = page.image.pil_image
                                    logger.info(f"Seite {i} - Seitenbild gefunden: {type(page_image)}")
                                except Exception as e:
                                    logger.debug(f"Seite {i} - Fehler beim Zugriff auf page.image.pil_image: {str(e)}")
                        
                        # Bilder extrahieren (falls verfügbar)
                        images = []
                        try:
                            # Versuche verschiedene Quellen für Bilder
                            page_images = []
                            
                            # 1. Prüfe result.document.pictures (gesamtes Dokument) - PRIORITÄT
                            # Docling verwendet 'pictures' statt 'images'!
                            if hasattr(result.document, 'pictures') and result.document.pictures:
                                logger.info(f"Seite {i} - Gefundene Bilder (pictures) im Dokument: {len(result.document.pictures)}")
                                # Filtere Bilder, die zu dieser Seite gehören
                                for pic in result.document.pictures:
                                    # Prüfe, ob das Bild zu dieser Seite gehört
                                    pic_page = None
                                    
                                    # Versuche verschiedene Wege, die Seitenzuordnung zu finden
                                    # 1. Prüfe prov (Provenance) zuerst - das ist oft die zuverlässigste Quelle
                                    if hasattr(pic, 'prov') and pic.prov:
                                        try:
                                            # prov kann eine Liste von ProvenanceItem-Objekten sein
                                            if isinstance(pic.prov, list) and len(pic.prov) > 0:
                                                # Nimm das erste ProvenanceItem
                                                prov_item = pic.prov[0]
                                                if hasattr(prov_item, 'page_no'):
                                                    pic_page = prov_item.page_no
                                                elif hasattr(prov_item, 'page'):
                                                    pic_page = prov_item.page
                                                elif hasattr(prov_item, 'page_index'):
                                                    pic_page = prov_item.page_index
                                            elif isinstance(pic.prov, dict):
                                                pic_page = pic.prov.get('page') or pic.prov.get('page_index') or pic.prov.get('page_num') or pic.prov.get('page_no')
                                            elif hasattr(pic.prov, 'page_no'):
                                                pic_page = pic.prov.page_no
                                            elif hasattr(pic.prov, 'page'):
                                                pic_page = pic.prov.page
                                            elif hasattr(pic.prov, 'page_index'):
                                                pic_page = pic.prov.page_index
                                        except Exception as e:
                                            logger.debug(f"Fehler beim Extrahieren der Seitenzuordnung aus prov: {str(e)}")
                                            pass
                                    
                                    # 2. Prüfe meta
                                    if pic_page is None and hasattr(pic, 'meta') and pic.meta:
                                        if isinstance(pic.meta, dict):
                                            pic_page = pic.meta.get('page') or pic.meta.get('page_index') or pic.meta.get('page_num')
                                        elif hasattr(pic.meta, 'page'):
                                            pic_page = pic.meta.page
                                    
                                    # 3. Prüfe direkte Attribute
                                    if pic_page is None:
                                        if hasattr(pic, 'page'):
                                            pic_page = pic.page
                                        elif hasattr(pic, 'page_index'):
                                            pic_page = pic.page_index
                                        elif hasattr(pic, 'page_num'):
                                            pic_page = pic.page_num
                                    
                                    # Seitenzuordnung prüfen
                                    if pic_page is not None:
                                        try:
                                            # Konvertiere zu int falls nötig
                                            pic_page_int = int(pic_page) if pic_page is not None else None
                                            page_int = int(i) if i is not None else None
                                            if pic_page_int == page_int:
                                                logger.info(f"Bild gehört zu Seite {i} (page-Attribut: {pic_page})")
                                                page_images.append(pic)
                                        except (ValueError, TypeError):
                                            # Falls Konvertierung fehlschlägt, vergleiche direkt
                                            if pic_page == i:
                                                logger.info(f"Bild gehört zu Seite {i} (page-Attribut: {pic_page})")
                                                page_images.append(pic)
                                    else:
                                        # Wenn keine Seitenzuordnung, prüfe über prov (Provenance)
                                        if hasattr(pic, 'prov') and pic.prov:
                                            try:
                                                # prov könnte Seiteninformationen enthalten
                                                if isinstance(pic.prov, dict):
                                                    prov_page = pic.prov.get('page') or pic.prov.get('page_index')
                                                    if prov_page is not None:
                                                        try:
                                                            if int(prov_page) == int(i):
                                                                logger.info(f"Bild gehört zu Seite {i} (prov.page: {prov_page})")
                                                                page_images.append(pic)
                                                        except:
                                                            pass
                                            except:
                                                pass
                                        
                                        # Wenn immer noch keine Zuordnung, füge zu allen Seiten hinzu (Fallback)
                                        # Aber nur, wenn wir noch keine Bilder für diese Seite haben
                                        if pic not in page_images:
                                            # Fallback: Füge Bild zu allen Seiten hinzu, wenn keine Zuordnung möglich
                                            logger.info(f"Bild ohne Seitenzuordnung - wird Seite {i} hinzugefügt (Fallback)")
                                            page_images.append(pic)
                            
                            # 2. Prüfe page.images (falls page ein Objekt ist)
                            if pages_are_objects and hasattr(page, 'images') and page.images:
                                logger.info(f"Seite {i} - Gefundene Bilder in page.images: {len(page.images)}")
                                page_images.extend(page.images)
                            
                            # 3. Prüfe page.content für Bilder (falls page ein Objekt ist)
                            if pages_are_objects and hasattr(page, 'content') and page.content:
                                logger.info(f"Seite {i} - page.content vorhanden: {type(page.content)}")
                                if hasattr(page.content, 'images') and page.content.images:
                                    logger.info(f"Seite {i} - Gefundene Bilder in page.content.images: {len(page.content.images)}")
                                    page_images.extend(page.content.images)
                            
                            # 4. Prüfe page.items für Bilder (falls page ein Objekt ist)
                            if pages_are_objects and hasattr(page, 'items') and page.items:
                                logger.info(f"Seite {i} - Gefundene Items: {len(page.items)}")
                                for item in page.items:
                                    item_type = str(type(item))
                                    if 'image' in item_type.lower() or 'picture' in item_type.lower():
                                        logger.info(f"Seite {i} - Bild gefunden in items")
                                        page_images.append(item)
                            
                            logger.info(f"Seite {i} - Gesamt gefundene Bilder: {len(page_images)}")
                            
                            # Verarbeite gefundene Bilder
                            for img_idx, img in enumerate(page_images):
                                logger.info(f"Seite {i} - Verarbeite Bild {img_idx}: Typ={type(img)}")
                                logger.info(f"Seite {i} - Bild {img_idx} - Alle Attribute: {[attr for attr in dir(img) if not attr.startswith('_')]}")
                                
                                img_data = {}
                                
                                # Bounding Box extrahieren
                                bbox = None
                                
                                # Prüfe verschiedene Quellen für BBox
                                # 1. Direktes bbox-Attribut
                                if hasattr(img, 'bbox') and img.bbox:
                                    bbox = list(img.bbox) if img.bbox else []
                                elif hasattr(img, 'bounds') and img.bounds:
                                    bbox = list(img.bounds) if img.bounds else []
                                elif hasattr(img, 'rect') and img.rect:
                                    # Konvertiere rect zu bbox
                                    rect = img.rect
                                    if hasattr(rect, 'x') and hasattr(rect, 'y') and hasattr(rect, 'width') and hasattr(rect, 'height'):
                                        bbox = [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height]
                                    elif isinstance(rect, (list, tuple)) and len(rect) >= 4:
                                        bbox = list(rect)
                                
                                # 2. Prüfe prov (Provenance) für BBox - WICHTIG: prov ist eine Liste von ProvenanceItem-Objekten
                                if bbox is None and hasattr(img, 'prov') and img.prov:
                                    try:
                                        # prov kann eine Liste von ProvenanceItem-Objekten sein
                                        if isinstance(img.prov, list) and len(img.prov) > 0:
                                            # Nimm das erste ProvenanceItem
                                            prov_item = img.prov[0]
                                            if hasattr(prov_item, 'bbox') and prov_item.bbox:
                                                # BoundingBox-Objekt hat l, t, r, b Attribute
                                                if hasattr(prov_item.bbox, 'l') and hasattr(prov_item.bbox, 't') and hasattr(prov_item.bbox, 'r') and hasattr(prov_item.bbox, 'b'):
                                                    bbox = [prov_item.bbox.l, prov_item.bbox.t, prov_item.bbox.r, prov_item.bbox.b]
                                                elif isinstance(prov_item.bbox, (list, tuple)) and len(prov_item.bbox) >= 4:
                                                    bbox = list(prov_item.bbox)
                                        elif isinstance(img.prov, dict):
                                            prov_bbox = img.prov.get('bbox') or img.prov.get('bounds')
                                            if prov_bbox and isinstance(prov_bbox, (list, tuple)) and len(prov_bbox) >= 4:
                                                bbox = list(prov_bbox)
                                    except Exception as e:
                                        logger.debug(f"Fehler beim Extrahieren der BBox aus prov: {str(e)}")
                                
                                # 3. Prüfe auch meta für BBox
                                if bbox is None and hasattr(img, 'meta') and img.meta:
                                    if isinstance(img.meta, dict):
                                        bbox = img.meta.get('bbox') or img.meta.get('bounds') or img.meta.get('rect')
                                        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                            bbox = list(bbox)
                                    
                                if bbox:
                                    img_data["bbox"] = bbox
                                    logger.info(f"Bild-BBox extrahiert: {bbox}")
                                else:
                                    # Auch ohne BBox hinzufügen, wenn Base64 vorhanden ist
                                    logger.debug(f"Keine BBox für Bild gefunden")
                                
                                # Base64-Kodierung (falls aktiviert)
                                if include_images_base64:
                                    try:
                                        import base64
                                        from io import BytesIO
                                        from PIL import Image
                                        
                                        # Debug: Prüfe alle möglichen Bildquellen
                                        logger.info(f"Seite {i} - Bild {img_idx} - Prüfe Bildquellen...")
                                        logger.info(f"Seite {i} - Bild {img_idx} - hasattr(img, 'get_image'): {hasattr(img, 'get_image')}")
                                        logger.info(f"Seite {i} - Bild {img_idx} - hasattr(img, 'image'): {hasattr(img, 'image')}")
                                        logger.info(f"Seite {i} - Bild {img_idx} - hasattr(img, 'data'): {hasattr(img, 'data')}")
                                        logger.info(f"Seite {i} - Bild {img_idx} - hasattr(img, 'path'): {hasattr(img, 'path')}")
                                        logger.info(f"Seite {i} - Bild {img_idx} - hasattr(img, 'content_layer'): {hasattr(img, 'content_layer')}")
                                        if hasattr(img, 'image'):
                                            logger.info(f"Seite {i} - Bild {img_idx} - img.image Wert: {img.image}")
                                        if hasattr(img, 'content_layer') and img.content_layer:
                                            logger.info(f"Seite {i} - Bild {img_idx} - content_layer Typ: {type(img.content_layer)}")
                                            logger.info(f"Seite {i} - Bild {img_idx} - content_layer Attribute: {[attr for attr in dir(img.content_layer) if not attr.startswith('_')]}")
                                        
                                        # Versuche verschiedene Wege, das Bild zu extrahieren
                                        image_data = None
                                        
                                        # 1. get_image(document) Methode (PictureItem hat diese)
                                        # WICHTIG: get_image() benötigt das document-Objekt als Parameter!
                                        if hasattr(img, 'get_image') and callable(img.get_image):
                                            try:
                                                # Versuche mit document-Parameter (wie im Beispiel)
                                                image_data = img.get_image(result.document)
                                                if image_data is not None:
                                                    logger.info(f"Seite {i} - Bild {img_idx} - Bild über get_image(document) extrahiert: {type(image_data)}, Größe: {image_data.size if hasattr(image_data, 'size') else 'N/A'}")
                                                else:
                                                    logger.warning(f"Seite {i} - Bild {img_idx} - get_image(document) gab None zurück")
                                            except TypeError as te:
                                                # Falls get_image() keinen Parameter benötigt
                                                logger.debug(f"Seite {i} - Bild {img_idx} - get_image() benötigt keinen Parameter, versuche ohne: {str(te)}")
                                                try:
                                                    image_data = img.get_image()
                                                    if image_data is not None:
                                                        logger.info(f"Seite {i} - Bild {img_idx} - Bild über get_image() extrahiert: {type(image_data)}")
                                                    else:
                                                        logger.warning(f"Seite {i} - Bild {img_idx} - get_image() gab None zurück")
                                                except Exception as e:
                                                    logger.warning(f"Seite {i} - Bild {img_idx} - get_image() fehlgeschlagen: {str(e)}")
                                            except Exception as e:
                                                logger.warning(f"Seite {i} - Bild {img_idx} - get_image(document) fehlgeschlagen: {str(e)}")
                                                import traceback
                                                logger.debug(traceback.format_exc())
                                        
                                        # 2. Direktes image Attribut
                                        if image_data is None and hasattr(img, 'image') and img.image:
                                            image_data = img.image
                                            logger.info(f"Bild über image-Attribut extrahiert: {type(image_data)}")
                                        
                                        # 3. content_layer.image (PictureItem hat oft content_layer)
                                        if image_data is None and hasattr(img, 'content_layer') and img.content_layer:
                                            if hasattr(img.content_layer, 'image') and img.content_layer.image:
                                                image_data = img.content_layer.image
                                                logger.info(f"Bild über content_layer.image extrahiert: {type(image_data)}")
                                            elif hasattr(img.content_layer, 'data') and img.content_layer.data:
                                                image_data = Image.open(BytesIO(img.content_layer.data))
                                                logger.info(f"Bild über content_layer.data extrahiert")
                                        
                                        # 4. Bilddaten als Bytes
                                        if image_data is None and hasattr(img, 'data') and img.data:
                                            image_data = Image.open(BytesIO(img.data))
                                            logger.info(f"Bild über data-Attribut extrahiert")
                                        
                                        # 5. Bildpfad
                                        if image_data is None and hasattr(img, 'path') and img.path:
                                            try:
                                                image_data = Image.open(img.path)
                                                logger.info(f"Bild über path-Attribut extrahiert")
                                            except Exception as e:
                                                logger.debug(f"Fehler beim Öffnen des Bildpfads: {str(e)}")
                                        
                                        # 6. meta.path oder meta.src
                                        if image_data is None and hasattr(img, 'meta') and img.meta:
                                            if isinstance(img.meta, dict):
                                                path = img.meta.get('path') or img.meta.get('src')
                                                if path:
                                                    try:
                                                        image_data = Image.open(path)
                                                        logger.info(f"Bild über meta.path/meta.src extrahiert")
                                                    except Exception as e:
                                                        logger.debug(f"Fehler beim Öffnen von meta.path: {str(e)}")
                                        
                                        # 7. prov.path oder prov.src (Provenance)
                                        if image_data is None and hasattr(img, 'prov') and img.prov:
                                            if isinstance(img.prov, dict):
                                                path = img.prov.get('path') or img.prov.get('src') or img.prov.get('file')
                                                if path:
                                                    try:
                                                        image_data = Image.open(path)
                                                        logger.info(f"Bild über prov.path/prov.src extrahiert")
                                                    except Exception as e:
                                                        logger.debug(f"Fehler beim Öffnen von prov.path: {str(e)}")
                                        
                                        if image_data:
                                            try:
                                                buffered = BytesIO()
                                                # Prüfe, ob image_data ein PIL Image ist
                                                if not hasattr(image_data, 'save'):
                                                    logger.warning(f"Seite {i} - Bild {img_idx} - image_data ist kein PIL Image: {type(image_data)}")
                                                    # Versuche es zu konvertieren, falls es ein numpy array oder ähnliches ist
                                                    if hasattr(image_data, 'shape'):
                                                        # Numpy array
                                                        image_data = Image.fromarray(image_data)
                                                    else:
                                                        image_data = None
                                                
                                                if image_data and hasattr(image_data, 'save'):
                                                    # Konvertiere zu RGB falls nötig
                                                    if hasattr(image_data, 'mode') and image_data.mode in ('RGBA', 'LA', 'P'):
                                                        rgb_image = Image.new('RGB', image_data.size, (255, 255, 255))
                                                        if image_data.mode == 'P':
                                                            image_data = image_data.convert('RGBA')
                                                        rgb_image.paste(image_data, mask=image_data.split()[-1] if image_data.mode == 'RGBA' else None)
                                                        image_data = rgb_image
                                                    
                                                    image_data.save(buffered, format="PNG")
                                                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                                    img_data["base64"] = img_base64
                                                    logger.info(f"Seite {i} - Bild {img_idx} - Bild erfolgreich als Base64 kodiert ({len(img_base64)} Zeichen)")
                                                else:
                                                    logger.warning(f"Seite {i} - Bild {img_idx} - Konnte Bild nicht als Base64 kodieren - kein gültiges PIL Image")
                                            except Exception as e:
                                                logger.warning(f"Seite {i} - Bild {img_idx} - Fehler beim Base64-Kodieren: {str(e)}")
                                                import traceback
                                                logger.debug(traceback.format_exc())
                                        else:
                                            logger.warning(f"Seite {i} - Bild {img_idx} - Konnte Bild nicht extrahieren - keine verfügbare Quelle")
                                    except Exception as e:
                                        logger.warning(f"Fehler beim Base64-Kodieren des Bildes: {str(e)}")
                                        import traceback
                                        logger.debug(traceback.format_exc())
                                
                                # Nur hinzufügen, wenn mindestens BBox oder Base64 vorhanden ist
                                if img_data.get("bbox") or img_data.get("base64"):
                                    images.append(img_data)
                                    logger.info(f"Bild hinzugefügt: bbox={img_data.get('bbox') is not None}, base64={img_data.get('base64') is not None}")
                                else:
                                    logger.warning(f"Bild nicht hinzugefügt - weder BBox noch Base64 vorhanden. Bild-Typ: {type(img)}")
                        
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
                        
                        # Seitenbild als Base64 kodieren (falls vorhanden)
                        if include_images_base64 and page_image:
                            try:
                                import base64
                                from io import BytesIO
                                from PIL import Image
                                buffered = BytesIO()
                                # Konvertiere zu RGB falls nötig
                                if page_image.mode in ('RGBA', 'LA', 'P'):
                                    rgb_image = Image.new('RGB', page_image.size, (255, 255, 255))
                                    if page_image.mode == 'P':
                                        page_image = page_image.convert('RGBA')
                                    rgb_image.paste(page_image, mask=page_image.split()[-1] if page_image.mode == 'RGBA' else None)
                                    page_image = rgb_image
                                page_image.save(buffered, format="PNG")
                                page_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                # Füge Seitenbild als erstes Bild hinzu
                                images.insert(0, {
                                    "base64": page_image_base64,
                                    "bbox": None  # Seitenbild hat keine BBox
                                })
                                logger.info(f"Seite {i} - Seitenbild als Base64 kodiert ({len(page_image_base64)} Zeichen)")
                            except Exception as e:
                                logger.warning(f"Seite {i} - Fehler beim Base64-Kodieren des Seitenbilds: {str(e)}")
                        
                        page_info = {
                            "index": idx,  # Verwende idx statt i für korrekte Reihenfolge
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

