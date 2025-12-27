"""
Reranker-Service für lokales CPU/GPU-basiertes Reranking.
Unterstützt CrossEncoder und Qwen3-Reranker mit spezieller Behandlung.
"""
import os
from sentence_transformers import CrossEncoder
from typing import List, Optional, Tuple, Union
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from app.models import RerankResult

logger = logging.getLogger(__name__)


def get_hf_token() -> Optional[str]:
    """
    Liest den Hugging Face Token aus Umgebungsvariablen.
    
    Returns:
        HF Token oder None
    """
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# Prüfe ob Qwen-Modell
def is_qwen_model(model_name: str) -> bool:
    """Prüft ob es sich um ein Qwen3-Reranker-Modell handelt."""
    return "qwen" in model_name.lower() and "reranker" in model_name.lower()

# Verfügbare Reranker-Modelle (von schnell/klein zu langsam/groß)
AVAILABLE_MODELS = {
    # CrossEncoder Modelle (MiniLM basiert auf BERT)
    "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # ~80MB, sehr schnell, BERT-basiert
    "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # ~120MB, guter Kompromiss, BERT-basiert
    "accurate": "cross-encoder/ms-marco-electra-base",  # ~450MB, sehr genau, ELECTRA-basiert
    
    # BGE Reranker Modelle
    "bge-v2": "BAAI/bge-reranker-v2-m3",  # BGE Reranker v2, mehrsprachig, beste Balance
    "bge-large": "BAAI/bge-reranker-large-v2",  # BGE Reranker Large, höchste Genauigkeit
    
    # ZeroEntropy Reranker Modelle
    "zerank-1": "zeroentropy/zerank-1",  # ZeroEntropy zerank-1, sehr leistungsstark (nicht-kommerziell)
    "zerank-1-small": "zeroentropy/zerank-1-small",  # ZeroEntropy zerank-1-small, Apache 2.0 Lizenz
    
    # Qwen3 Reranker Modelle (Alibaba, Apache 2.0, mehrsprachig)
    "qwen3-0.6b": "Qwen/Qwen3-Reranker-0.6B",  # Qwen3-Reranker 0.6B, schnell, mehrsprachig
    "qwen3-4b": "Qwen/Qwen3-Reranker-4B",  # Qwen3-Reranker 4B, beste Balance, mehrsprachig
    "qwen3-8b": "Qwen/Qwen3-Reranker-8B",  # Qwen3-Reranker 8B, höchste Genauigkeit, mehrsprachig
    
    # BERT-basierte Reranker Modelle
    "bert-base": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Alias für fast (BERT-basiert)
    "bert-german": "deepset/gbert-base-germandpr-reranking",  # German BERT für deutsche Texte
    "bert-multilingual": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Alias für balanced (mehrsprachig)
    
    # Standard
    "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Standard
}

# Standard-Modell
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def resolve_model_name(model_name: Optional[str] = None) -> str:
    """
    Löst einen Modellnamen oder Alias auf den echten Modellnamen auf.
    
    Args:
        model_name: Modellname oder Alias (None = aus ENV-Variable lesen)
    
    Returns:
        Echter Modellname für CrossEncoder
    """
    if model_name:
        model_input = model_name.strip()
    else:
        model_input = os.getenv("RERANKER_MODEL", "").strip()
    
    if not model_input:
        return DEFAULT_MODEL
    
    # Prüfe ob es ein Alias ist
    if model_input.lower() in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_input.lower()]
    
    # Sonst verwende den direkten Modellnamen
    return model_input


def get_model_name() -> str:
    """
    Liest den Modellnamen aus Umgebungsvariable oder verwendet Standard.
    
    Unterstützt sowohl direkte Modellnamen als auch Aliase (fast, balanced, accurate).
    
    Returns:
        Modellname für CrossEncoder
    """
    return resolve_model_name()


class RerankerService:
    """
    Service für lokales Reranking auf CPU.
    
    Verwendet CrossEncoder von sentence-transformers, der speziell
    für Reranking optimiert ist und effizient auf CPU läuft.
    
    Das Modell kann über die Umgebungsvariable RERANKER_MODEL ausgewählt werden:
    - "fast" oder "cross-encoder/ms-marco-MiniLM-L-6-v2" (Standard, schnell, BERT-basiert)
    - "balanced" oder "cross-encoder/ms-marco-MiniLM-L-12-v2" (gute Balance, BERT-basiert)
    - "accurate" oder "cross-encoder/ms-marco-electra-base" (sehr genau, ELECTRA-basiert)
    - "bge-v2" oder "BAAI/bge-reranker-v2-m3" (EMPFOHLEN: beste Balance, mehrsprachig)
    - "bge-large" oder "BAAI/bge-reranker-large-v2" (höchste Genauigkeit, mehrsprachig)
    - "zerank-1" oder "zeroentropy/zerank-1" (ZeroEntropy, sehr leistungsstark, nicht-kommerziell)
    - "zerank-1-small" oder "zeroentropy/zerank-1-small" (ZeroEntropy, Apache 2.0 Lizenz)
    - "qwen3-0.6b" oder "Qwen/Qwen3-Reranker-0.6B" (Qwen3 0.6B, schnell, mehrsprachig, Apache 2.0)
    - "qwen3-4b" oder "Qwen/Qwen3-Reranker-4B" (Qwen3 4B, beste Balance, mehrsprachig, Apache 2.0)
    - "qwen3-8b" oder "Qwen/Qwen3-Reranker-8B" (Qwen3 8B, höchste Genauigkeit, mehrsprachig, Apache 2.0)
    - "bert-german" oder "deepset/gbert-base-germandpr-reranking" (German BERT für deutsche Texte)
    - Oder direkter Modellname von HuggingFace (z.B. "bert-base-uncased" für Reranking)
    """
    
    # Klassenweiter Cache für geladene Modelle (shared across instances)
    # Unterstützt sowohl CrossEncoder als auch Qwen-Modelle (tuple = (model, tokenizer, device))
    _model_cache: dict[str, Union[CrossEncoder, tuple]] = {}
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialisiert den Reranker-Service.
        
        Args:
            model_name: Name des Modells (optional, überschreibt ENV-Variable)
        """
        self.model_name = resolve_model_name(model_name) if model_name else get_model_name()
        self.model: Optional[Union[CrossEncoder, tuple]] = None
        self.is_qwen = is_qwen_model(self.model_name)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._loaded = False
    
    async def initialize(self):
        """Lädt das Modell asynchron."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._load_model)
    
    def _load_qwen_model(self, model_name: Optional[str] = None):
        """
        Lädt ein Qwen3-Reranker-Modell mit AutoModelForSequenceClassification.
        Qwen-Modelle benötigen spezielle Behandlung.
        
        Args:
            model_name: Optionaler Modellname (Standard: self.model_name)
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Modellname bestimmen
            target_model_name = model_name if model_name else self.model_name
            
            # Device bestimmen
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Lade Qwen3-Reranker-Modell: {target_model_name} auf {device}")
            
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
                model = AutoModelForSequenceClassification.from_pretrained(
                    target_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    **token_kwargs
                ).eval()
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    target_model_name,
                    **token_kwargs
                ).eval()
            
            # WICHTIG: auch im Model-Config setzen
            model.config.pad_token_id = tokenizer.pad_token_id
            
            logger.info(f"Qwen3-Reranker-Modell erfolgreich geladen auf {device}")
            return model, tokenizer, device
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Qwen-Modells: {str(e)}")
            raise
    
    async def _rerank_qwen(
        self,
        query: str,
        documents: List[str],
        model_data: tuple
    ) -> List[float]:
        """
        Führt Reranking mit Qwen3-Reranker durch.
        Verwendet das spezielle Instruction-Format.
        
        Args:
            query: Die Suchanfrage
            documents: Liste der Dokumente
            model_data: Tuple (model, tokenizer, device)
            
        Returns:
            Liste von Scores
        """
        import torch
        model, tokenizer, device = model_data
        
        # Qwen3-korrektes Instruction-Format
        # Wichtig: Qwen3-Reranker sind instruktionssensitiv.
        # Ohne Instruction bewerten sie Stil statt Relevanz.
        pairs = [
            (
                (
                    "Given the query, judge whether the document is relevant.\n\n"
                    f"Query: {query}\n"
                    f"Document: {doc}\n"
                    "Relevance:"
                ),
                ""
            )
            for doc in documents
        ]
        
        # Tokenisieren (Batch) - in Thread-Pool ausführen
        loop = asyncio.get_event_loop()
        
        def tokenize_and_score():
            import torch
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Forward-Pass
            with torch.no_grad():
                logits = model(**inputs).logits
                # Qwen3-Reranker: logits.shape == (batch_size, 2)
                # Index 1 = relevant
                scores = logits[:, 1].cpu().numpy().tolist()
            
            return scores
        
        scores = await loop.run_in_executor(self.executor, tokenize_and_score)
        return scores
    
    def _configure_tokenizer(self, model: CrossEncoder):
        """
        Konfiguriert den Tokenizer für spezielle Modelle (z.B. Qwen).
        
        Args:
            model: Das geladene CrossEncoder-Modell
        """
        try:
            # Zugriff auf den Tokenizer über das interne Modell
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                tokenizer = model.tokenizer
                # Qwen-Modelle benötigen pad_token = eos_token
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                        logger.debug(f"Pad-Token auf EOS-Token gesetzt für Modell: {self.model_name}")
                    else:
                        logger.warning(f"Kein EOS-Token gefunden für Modell: {self.model_name}")
        except Exception as e:
            logger.warning(f"Konnte Tokenizer nicht konfigurieren: {str(e)}")
    
    def _load_model(self):
        """Lädt das CrossEncoder-Modell (mit Caching)."""
        try:
            # Prüfe ob Modell bereits im Cache ist
            if self.model_name in RerankerService._model_cache:
                logger.info(f"Verwende gecachtes Modell: {self.model_name}")
                self.model = RerankerService._model_cache[self.model_name]
            else:
                logger.info(f"Lade Reranker-Modell: {self.model_name}")
                
                # GPU-Status prüfen
                try:
                    import torch
                    if torch.cuda.is_available():
                        logger.info(f"GPU verfügbar: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
                    else:
                        logger.info("GPU nicht verfügbar, verwende CPU")
                except Exception:
                    logger.debug("Konnte GPU-Status nicht prüfen")
                
                # Qwen3-Modelle benötigen spezielle Behandlung
                if self.is_qwen:
                    model, tokenizer, device = self._load_qwen_model()
                    self.model = (model, tokenizer, device)
                else:
                    # CrossEncoder erkennt automatisch GPU, wenn verfügbar
                    # Zerank-Modelle benötigen trust_remote_code=True
                    hf_token = get_hf_token()
                    if "zerank" in self.model_name.lower():
                        if hf_token:
                            self.model = CrossEncoder(self.model_name, trust_remote_code=True, token=hf_token)
                        else:
                            self.model = CrossEncoder(self.model_name, trust_remote_code=True)
                    else:
                        if hf_token:
                            self.model = CrossEncoder(self.model_name, token=hf_token)
                        else:
                            self.model = CrossEncoder(self.model_name)
                    
                    # Konfiguriere Tokenizer für spezielle Modelle
                    self._configure_tokenizer(self.model)
                
                # Speichere im Cache für zukünftige Verwendung
                RerankerService._model_cache[self.model_name] = self.model
                logger.info(f"Reranker-Modell erfolgreich geladen und gecacht: {self.model_name}")
            
            self._loaded = True
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {str(e)}")
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
        logger.info("Modell-Cache wurde geleert")
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> List[RerankResult]:
        """
        Führt Reranking für eine Query und Dokumente durch (Cohere-kompatibel).
        
        Args:
            query: Die Suchanfrage
            documents: Liste der zu rerankenden Dokumente
            top_n: Anzahl der Top-Ergebnisse (None = alle)
            model_name: Optionaler Modellname (überschreibt aktuelles Modell temporär)
            
        Returns:
            Liste von RerankResult mit index und relevance_score, sortiert nach Score (höchste zuerst)
        """
        # Verwende temporäres Modell falls angegeben
        model_to_use = self.model
        is_qwen_model_use = self.is_qwen
        
        if model_name:
            # Löse Alias auf echten Modellnamen auf
            resolved_model_name = resolve_model_name(model_name)
            is_qwen_model_use = is_qwen_model(resolved_model_name)
            
            if resolved_model_name != self.model_name:
                # Prüfe Cache zuerst
                if resolved_model_name in RerankerService._model_cache:
                    logger.debug(f"Verwende gecachtes temporäres Modell: {resolved_model_name} (Alias: {model_name})")
                    model_to_use = RerankerService._model_cache[resolved_model_name]
                else:
                    logger.info(f"Lade temporäres Modell: {resolved_model_name} (Alias: {model_name})")
                    
                    # Qwen-Modelle benötigen spezielle Behandlung
                    if is_qwen_model_use:
                        temp_model_data = self._load_qwen_model(resolved_model_name)
                        model_to_use = temp_model_data
                        RerankerService._model_cache[resolved_model_name] = temp_model_data
                    else:
                        # Zerank-Modelle benötigen trust_remote_code=True
                        if "zerank" in resolved_model_name.lower():
                            temp_model = CrossEncoder(resolved_model_name, trust_remote_code=True)
                        else:
                            temp_model = CrossEncoder(resolved_model_name)
                        
                        # Konfiguriere Tokenizer für spezielle Modelle
                        self._configure_tokenizer(temp_model)
                        
                        # Speichere im Cache
                        RerankerService._model_cache[resolved_model_name] = temp_model
                        model_to_use = temp_model
        elif not self.model:
            raise RuntimeError("Modell ist nicht geladen. Bitte warten Sie auf die Initialisierung.")
        
        if not documents:
            return []
        
        # Qwen-Modelle benötigen spezielle Behandlung
        if is_qwen_model_use and isinstance(model_to_use, tuple):
            # Qwen3-Reranker mit Instruction-Format
            scores = await self._rerank_qwen(query, documents, model_to_use)
        else:
            # Standard CrossEncoder
            # Erstelle Query-Dokument-Paare für CrossEncoder
            pairs = [[query, doc] for doc in documents]
            
            # Führe Scoring in Thread-Pool aus (nicht-blockierend)
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                self.executor,
                model_to_use.predict,
                pairs
            )
        
        # Konvertiere zu numpy array für Verarbeitung
        scores_array = np.array(scores, dtype=np.float32)
        
        # Prüfe Score-Bereich für Debugging
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        mean_score = np.mean(scores_array)
        logger.info(f"Roh-Scores - Min: {min_score:.6f}, Max: {max_score:.6f}, Mean: {mean_score:.6f}")
        logger.info(f"Roh-Scores Details: {scores_array.tolist()}")
        
        # Verwende Roh-Scores direkt ohne Normalisierung
        # Die Modelle geben bereits interpretierbare Scores aus
        # Normalisierung würde die relativen Unterschiede verzerren
        scores_normalized = scores_array
        
        # Erstelle Liste von (index, score) Tupeln
        indexed_scores = [
            (index, float(score))
            for index, score in enumerate(scores_normalized)
        ]
        
        # Sortiere nach Score (absteigend)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Top-N filtern falls angegeben
        if top_n is not None and top_n > 0:
            indexed_scores = indexed_scores[:top_n]
        
        # Erstelle Ergebnisse im Cohere-Format
        results = [
            RerankResult(
                index=index,
                relevance_score=score
            )
            for index, score in indexed_scores
        ]
        
        return results

