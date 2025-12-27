"""
Embeddings-Service für lokales CPU/GPU-basiertes Embedding.
Unterstützt verschiedene Embedding-Modelle im OpenAI-Format.
Unterstützt sowohl SentenceTransformer-Modelle als auch Qwen Embedding-Modelle.
"""
import os
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union, Tuple
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

# Verfügbare Embedding-Modelle (sentence-transformers kompatibel)
AVAILABLE_EMBEDDING_MODELS = {
    # BGE Modelle
    "bge-large": "BAAI/bge-large-en-v1.5",  # BGE Large English v1.5
    "bge-base": "BAAI/bge-base-en-v1.5",  # BGE Base English v1.5
    # Jina Modelle
    "jina-de": "jinaai/jina-embeddings-v2-base-de",  # Jina German Embeddings
    # SmolLM Modelle
    "smollm3-de": "mayflowergmbh/smollm3-3b-german-embed",  # SmolLM German Embeddings
    # Aliase
    "default": "BAAI/bge-base-en-v1.5"
}

# Qwen Embedding-Modelle (benötigen spezielle Behandlung)
# Diese Modelle verwenden einen anderen Ansatz und werden nicht über SentenceTransformer geladen
QWEN_EMBEDDING_MODELS = {
    "qwen-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen-embedding-8b": "Qwen/Qwen3-Embedding-8B",
}

# Standard-Modell
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def is_qwen_embedding_model(model_name: str) -> bool:
    """Prüft ob es sich um ein Qwen Embedding-Modell handelt."""
    model_lower = model_name.lower()
    return (
        "qwen" in model_lower and 
        "embedding" in model_lower and 
        "reranker" not in model_lower
    ) or model_lower in QWEN_EMBEDDING_MODELS


def resolve_embedding_model_name(model_name: Optional[str] = None) -> str:
    """
    Löst einen Modellnamen oder Alias auf den echten Modellnamen auf.
    
    Args:
        model_name: Modellname oder Alias (None = Standard)
    
    Returns:
        Echter Modellname für SentenceTransformer oder Qwen-Modell
    """
    if model_name:
        model_input = model_name.strip()
    else:
        return DEFAULT_EMBEDDING_MODEL
    
    # Prüfe ob es ein Qwen Embedding Alias ist
    model_lower = model_input.lower()
    if model_lower in QWEN_EMBEDDING_MODELS:
        return QWEN_EMBEDDING_MODELS[model_lower]
    
    # Prüfe ob es ein Standard Alias ist (case-insensitive)
    if model_lower in AVAILABLE_EMBEDDING_MODELS:
        return AVAILABLE_EMBEDDING_MODELS[model_lower]
    
    # Prüfe auch direkten Modellnamen (case-sensitive)
    if model_input in AVAILABLE_EMBEDDING_MODELS:
        return AVAILABLE_EMBEDDING_MODELS[model_input]
    
    if model_input in QWEN_EMBEDDING_MODELS:
        return QWEN_EMBEDDING_MODELS[model_input]
    
    # Direkter Modellname (wird so verwendet wie angegeben)
    return model_input


class EmbeddingsService:
    """
    Service für lokales Embedding auf CPU/GPU.
    
    Verwendet SentenceTransformer für Embeddings im OpenAI-Format.
    """
    
    # Klassenweiter Cache für geladene Modelle
    # Unterstützt sowohl SentenceTransformer als auch Qwen-Modelle (tuple = (model, tokenizer, device))
    _model_cache: dict[str, Union[SentenceTransformer, Tuple]] = {}
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialisiert den Embeddings-Service.
        
        Args:
            model_name: Name des Modells (optional, Standard: bge-base)
        """
        resolved = resolve_embedding_model_name(model_name)
        logger.debug(f"Embedding-Modell aufgelöst: {model_name} -> {resolved}")
        self.model_name = resolved
        self.is_qwen = is_qwen_embedding_model(resolved)
        self.model: Optional[Union[SentenceTransformer, Tuple]] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._loaded = False
    
    async def initialize(self):
        """Lädt das Modell asynchron."""
        loop = asyncio.get_event_loop()
        if self.is_qwen:
            model_data = await loop.run_in_executor(self.executor, self._load_qwen_model)
            self.model = model_data
            EmbeddingsService._model_cache[self.model_name] = model_data
            self._loaded = True
        else:
            await loop.run_in_executor(self.executor, self._load_model)
    
    def _load_qwen_model(self, model_name: Optional[str] = None):
        """
        Lädt ein Qwen Embedding-Modell mit AutoModel.
        Qwen Embedding-Modelle benötigen spezielle Behandlung.
        
        Args:
            model_name: Optionaler Modellname (Standard: self.model_name)
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Modellname bestimmen
            target_model_name = model_name if model_name else self.model_name
            
            # Device bestimmen
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Lade Qwen Embedding-Modell: {target_model_name} auf {device}")
            
            # HF Token für private Modelle
            hf_token = get_hf_token()
            token_kwargs = {"token": hf_token} if hf_token else {}
            
            # Tokenizer laden
            tokenizer = AutoTokenizer.from_pretrained(target_model_name, **token_kwargs)
            
            # Padding für decoder-only Modelle (Pflicht)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.debug("Pad-Token auf EOS-Token gesetzt")
            
            # Modell laden
            if device == "cuda":
                model = AutoModel.from_pretrained(
                    target_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    **token_kwargs
                ).eval()
            else:
                model = AutoModel.from_pretrained(
                    target_model_name,
                    **token_kwargs
                ).eval()
            
            logger.info(f"Qwen Embedding-Modell erfolgreich geladen auf {device}")
            return model, tokenizer, device
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Qwen Embedding-Modells: {str(e)}")
            raise
    
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
                # Verwende HF Token falls vorhanden
                hf_token = get_hf_token()
                if hf_token:
                    logger.debug(f"Verwende Hugging Face Token für Modell: {self.model_name}")
                    self.model = SentenceTransformer(
                        self.model_name,
                        token=hf_token
                    )
                else:
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
    
    async def _embed_qwen(
        self,
        texts: List[str],
        model_data: Tuple
    ) -> List[List[float]]:
        """
        Erstellt Embeddings mit Qwen Embedding-Modell.
        Qwen Embedding-Modelle verwenden AutoModel und benötigen spezielle Behandlung.
        
        Args:
            texts: Liste der zu embeddenden Texte
            model_data: Tuple (model, tokenizer, device)
            
        Returns:
            Liste von Embedding-Vektoren
        """
        model, tokenizer, device = model_data
        
        loop = asyncio.get_event_loop()
        
        def tokenize_and_encode():
            import torch  # Local import for thread safety
            import torch.nn.functional as F
            
            # Batch-Verarbeitung für bessere Performance
            # Tokenisiere alle Texte auf einmal
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(device)
            
            # Erstelle Embeddings für alle Texte auf einmal
            with torch.no_grad():
                outputs = model(**inputs)
                # Qwen Embedding-Modelle: Verwende mean pooling über sequence length
                # last_hidden_state shape: (batch_size, seq_len, hidden_size)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
                
                # Normalisiere Embeddings (wie bei SentenceTransformer)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Konvertiere zu Liste
                embeddings_list = embeddings.cpu().numpy().tolist()
            
            return embeddings_list
        
        embeddings = await loop.run_in_executor(self.executor, tokenize_and_encode)
        return embeddings
    
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
        is_qwen_model_use = self.is_qwen
        
        if model_name:
            resolved_model_name = resolve_embedding_model_name(model_name)
            is_qwen_model_use = is_qwen_embedding_model(resolved_model_name)
            
            if resolved_model_name != self.model_name:
                # Prüfe Cache zuerst
                if resolved_model_name in EmbeddingsService._model_cache:
                    logger.debug(f"Verwende gecachtes temporäres Modell: {resolved_model_name}")
                    model_to_use = EmbeddingsService._model_cache[resolved_model_name]
                else:
                    logger.info(f"Lade temporäres Embedding-Modell: {resolved_model_name}")
                    if is_qwen_model_use:
                        # Qwen Embedding-Modell laden
                        loop_temp = asyncio.get_event_loop()
                        temp_model_data = await loop_temp.run_in_executor(
                            self.executor, 
                            self._load_qwen_model, 
                            resolved_model_name
                        )
                        EmbeddingsService._model_cache[resolved_model_name] = temp_model_data
                        model_to_use = temp_model_data
                    else:
                        # SentenceTransformer-Modell laden
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
        
        # Führe Embedding aus (Qwen vs. SentenceTransformer)
        if is_qwen_model_use and isinstance(model_to_use, tuple):
            embeddings = await self._embed_qwen(texts, model_to_use)
        else:
            # SentenceTransformer-Modell
            loop = asyncio.get_event_loop()
            
            # Wrapper-Funktion für encode mit Parametern
            def encode_with_params():
                return model_to_use.encode(
                    texts,
                    normalize_embeddings=True,  # Normalisiert für bessere Vergleichbarkeit
                    show_progress_bar=False
                )
            
            embeddings_array = await loop.run_in_executor(
                self.executor,
                encode_with_params
            )
            
            # Konvertiere numpy array zu Liste von Listen
            embeddings = embeddings_array.tolist() if isinstance(embeddings_array, np.ndarray) else embeddings_array
        
        return embeddings

