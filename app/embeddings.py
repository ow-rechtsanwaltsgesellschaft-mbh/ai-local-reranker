"""
Embeddings-Service für lokales CPU/GPU-basiertes Embedding.
Unterstützt verschiedene Embedding-Modelle im OpenAI-Format.
"""
import os
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


def get_hf_token() -> Optional[str]:
    """
    Liest den Hugging Face Token aus Umgebungsvariablen.
    
    Returns:
        HF Token oder None
    """
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# Verfügbare Embedding-Modelle
AVAILABLE_EMBEDDING_MODELS = {
    "bge-large": "BAAI/bge-large-en-v1.5",  # BGE Large English
    "bge-base": "BAAI/bge-base-en-v1.5",  # BGE Base English
    "jina-de": "jinaai/jina-embeddings-v2-base-de",  # Jina German
    "smollm3-de": "mayflowergmbh/smollm3-3b-german-embed",  # SmolLM German
    # Aliase
    "default": "BAAI/bge-base-en-v1.5"
}

# Standard-Modell
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def resolve_embedding_model_name(model_name: Optional[str] = None) -> str:
    """
    Löst einen Modellnamen oder Alias auf den echten Modellnamen auf.
    
    Args:
        model_name: Modellname oder Alias (None = Standard)
    
    Returns:
        Echter Modellname für SentenceTransformer
    """
    if model_name:
        model_input = model_name.strip()
    else:
        return DEFAULT_EMBEDDING_MODEL
    
    # Prüfe ob es ein Alias ist (case-insensitive)
    model_lower = model_input.lower()
    if model_lower in AVAILABLE_EMBEDDING_MODELS:
        return AVAILABLE_EMBEDDING_MODELS[model_lower]
    
    # Prüfe auch direkten Modellnamen (case-sensitive)
    if model_input in AVAILABLE_EMBEDDING_MODELS:
        return AVAILABLE_EMBEDDING_MODELS[model_input]
    
    # Direkter Modellname (wird so verwendet wie angegeben)
    return model_input


class EmbeddingsService:
    """
    Service für lokales Embedding auf CPU/GPU.
    
    Verwendet SentenceTransformer für Embeddings im OpenAI-Format.
    """
    
    # Klassenweiter Cache für geladene Modelle
    _model_cache: dict[str, SentenceTransformer] = {}
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialisiert den Embeddings-Service.
        
        Args:
            model_name: Name des Modells (optional, Standard: bge-base)
        """
        self.model_name = resolve_embedding_model_name(model_name)
        self.model: Optional[SentenceTransformer] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._loaded = False
    
    async def initialize(self):
        """Lädt das Modell asynchron."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._load_model)
    
    def _load_model(self):
        """Lädt das SentenceTransformer-Modell (mit Caching)."""
        try:
            # Prüfe ob Modell bereits im Cache ist
            if self.model_name in EmbeddingsService._model_cache:
                logger.info(f"Verwende gecachtes Embedding-Modell: {self.model_name}")
                self.model = EmbeddingsService._model_cache[self.model_name]
            else:
                logger.info(f"Lade Embedding-Modell: {self.model_name}")
                
                # GPU-Status prüfen
                try:
                    import torch
                    if torch.cuda.is_available():
                        logger.info(f"GPU verfügbar: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
                    else:
                        logger.info("GPU nicht verfügbar, verwende CPU")
                except Exception:
                    logger.debug("Konnte GPU-Status nicht prüfen")
                
                # SentenceTransformer lädt automatisch auf GPU, wenn verfügbar
                self.model = SentenceTransformer(self.model_name)
                
                # Speichere im Cache für zukünftige Verwendung
                EmbeddingsService._model_cache[self.model_name] = self.model
                logger.info(f"Embedding-Modell erfolgreich geladen und gecacht: {self.model_name}")
            
            self._loaded = True
        except Exception as e:
            logger.error(f"Fehler beim Laden des Embedding-Modells: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Prüft, ob das Modell geladen ist."""
        return self._loaded
    
    def get_model_name(self) -> str:
        """Gibt den Namen des verwendeten Modells zurück."""
        return self.model_name
    
    @classmethod
    def get_cached_models(cls) -> list[str]:
        """Gibt eine Liste aller gecachten Modellnamen zurück."""
        return list(cls._model_cache.keys())
    
    @classmethod
    def clear_cache(cls):
        """Löscht den Modell-Cache (nur für Tests/Debugging)."""
        cls._model_cache.clear()
        logger.info("Embedding-Modell-Cache wurde geleert")
    
    async def embed(
        self,
        texts: List[str],
        model_name: Optional[str] = None
    ) -> List[List[float]]:
        """
        Erstellt Embeddings für eine Liste von Texten (OpenAI-Format).
        
        Args:
            texts: Liste der zu embeddenden Texte
            model_name: Optionaler Modellname (überschreibt aktuelles Modell temporär)
            
        Returns:
            Liste von Embedding-Vektoren (jeder Vektor ist eine Liste von Floats)
        """
        # Verwende temporäres Modell falls angegeben
        model_to_use = self.model
        if model_name:
            resolved_model_name = resolve_embedding_model_name(model_name)
            if resolved_model_name != self.model_name:
                # Prüfe Cache zuerst
                if resolved_model_name in EmbeddingsService._model_cache:
                    logger.debug(f"Verwende gecachtes temporäres Modell: {resolved_model_name}")
                    model_to_use = EmbeddingsService._model_cache[resolved_model_name]
                else:
                    logger.info(f"Lade temporäres Embedding-Modell: {resolved_model_name}")
                    hf_token = get_hf_token()
                    if hf_token:
                        temp_model = SentenceTransformer(
                            resolved_model_name,
                            token=hf_token
                        )
                    else:
                        temp_model = SentenceTransformer(resolved_model_name)
                    EmbeddingsService._model_cache[resolved_model_name] = temp_model
                    model_to_use = temp_model
        elif not self.model:
            raise RuntimeError("Modell ist nicht geladen. Bitte warten Sie auf die Initialisierung.")
        
        if not texts:
            return []
        
        # Führe Embedding in Thread-Pool aus (nicht-blockierend)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            model_to_use.encode,
            texts,
            {
                "normalize_embeddings": True,  # Normalisiert für bessere Vergleichbarkeit
                "show_progress_bar": False
            }
        )
        
        # Konvertiere numpy array zu Liste von Listen
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        return embeddings_list

