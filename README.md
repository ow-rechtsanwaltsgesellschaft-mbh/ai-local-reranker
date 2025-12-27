# AI Local Reranker & Embeddings API

[![Runpod](https://api.runpod.io/badge/ow-rechtsanwaltsgesellschaft-mbh/ai-local-reranker)](https://console.runpod.io/hub/ow-rechtsanwaltsgesellschaft-mbh/ai-local-reranker)

Eine modulare Python-API für lokales Reranking und Embeddings auf CPU/GPU mit FastAPI und sentence-transformers.

## Features

- 🚀 **Lokales Reranking auf CPU/GPU** - Keine externe API erforderlich
- 🔤 **Lokale Embeddings** - OpenAI-kompatible Embeddings-API
- 📄 **Dokumentenverarbeitung** - Docling-API für PDF, Word, etc. (Port 8890)
- ⚡ **Schnell und effizient** - Optimiert für CPU- und GPU-Performance
- 🐳 **Docker-ready** - Einfache Bereitstellung mit Docker Compose
- ☁️ **RunPod-ready** - GPU-optimiertes Deployment auf RunPod
- 📊 **RESTful API** - Standardisierte Endpoints
- 🔧 **Best Practices** - Production-ready Code
- 🔄 **Cohere-kompatibel** - Gleiches Request/Response-Format wie Cohere Rerank API
- 🤖 **OpenAI-kompatibel** - Gleiches Request/Response-Format wie OpenAI Embeddings API

## Technologie-Stack

- **FastAPI** - Moderne, schnelle Web-Framework
- **sentence-transformers** - CrossEncoder für Reranking, SentenceTransformer für Embeddings
- **PyTorch** - CPU/GPU-optimiertes Machine Learning
- **transformers** - Für spezielle Modelle (Qwen3-Reranker)
- **Docker** - Containerisierung
- **RunPod** - GPU-Cloud-Deployment (optional)

## Schnellstart

### Mit Docker Compose (Empfohlen)

```bash
# Container bauen und starten
docker-compose up --build

# Im Hintergrund starten
docker-compose up -d

# Logs anzeigen
docker-compose logs -f

# Container stoppen
docker-compose down
```

Die APIs sind dann unter folgenden URLs erreichbar:
- **Reranker & Embeddings API**: `http://localhost:8888`
- **Docling Document Processing API**: `http://localhost:8890`

### API-Dokumentation

Nach dem Start können Sie die interaktive API-Dokumentation unter folgenden URLs aufrufen:

**Reranker & Embeddings API (Port 8888):**
- **Swagger UI**: http://localhost:8888/docs
- **ReDoc**: http://localhost:8888/redoc

**Docling API (Port 8890):**
- **Swagger UI**: http://localhost:8890/docs
- **ReDoc**: http://localhost:8890/redoc

## API-Endpoints

### Reranker & Embeddings API (Port 8888)

#### Health Check

```bash
GET /health
```

#### Modell-Informationen

```bash
GET /model/info
GET /v1/models
```

Beide Endpoints listen verfügbare Modelle im OpenAI-Format auf (Reranker und Embeddings).

**Response (OpenAI-Format):**

```json
{
  "object": "list",
  "data": [
    {
      "id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "object": "model",
      "created": 1677610602,
      "owned_by": "local",
      "type": "reranker",
      "alias": "fast"
    },
    {
      "id": "BAAI/bge-base-en-v1.5",
      "object": "model",
      "created": 1677610602,
      "owned_by": "local",
      "type": "embedding",
      "alias": "bge-base"
    }
  ]
}
```

### Docling Document Processing API (Port 8890)

#### Health Check

```bash
GET http://localhost:8890/docling/health
```

#### Dokumentenkonvertierung

```bash
POST http://localhost:8890/docling/v1/convert
Content-Type: multipart/form-data
```

**Request:**
- `file`: Datei (PDF, DOCX, DOC, TXT, etc.)
- `output_format`: Ausgabeformat (`markdown`, `json`, `text`) - Standard: `markdown`
- `include_images_base64`: Bilder als Base64 kodieren (wie Mistral OCR) - Standard: `false`

**Beispiel mit curl:**
```bash
# Standard (ohne Base64-Bilder)
curl -X POST "http://localhost:8890/docling/v1/convert" \
  -F "file=@document.pdf" \
  -F "output_format=markdown"

# Mit Base64-kodierten Bildern (wie Mistral OCR)
curl -X POST "http://localhost:8890/docling/v1/convert" \
  -F "file=@document.pdf" \
  -F "output_format=markdown" \
  -F "include_images_base64=true"
```

#### Dokumentenkonvertierung von Dateipfad

```bash
POST http://localhost:8890/docling/v1/convert/path
Content-Type: application/json
```

**Request:**
```json
{
  "file_path": "/app/documents/document.pdf",
  "output_format": "markdown",
  "include_images_base64": false
}
```

**Response (Mistral OCR-kompatibel):**
```json
{
  "pages": [
    {
      "index": 0,
      "markdown": "# Dokumenttitel\n\nExtrahierter Textinhalt...",
      "dimensions": {
        "dpi": 200,
        "height": 2200,
        "width": 1700
      },
      "images": [
        {
          "bbox": [100, 150, 400, 450],
          "base64": "iVBORw0KGgoAAAANSUhEUgAA..."  // Nur wenn include_images_base64=true
        }
      ],
      "tables": [
        {
          "bbox": [100, 200, 500, 400],
          "columns": ["Spalte 1", "Spalte 2"],
          "rows": [
            ["Wert 1", "Wert 2"],
            ["Wert 3", "Wert 4"]
          ]
        }
      ]
    }
  ],
  "metadata": {
    "title": "Dokumenttitel",
    "author": "Autor",
    "language": "de",
    "subject": "Betreff",
    "creator": "Erstellungsprogramm",
    "producer": "Produktionsprogramm",
    "creation_date": "2024-01-01T00:00:00",
    "modification_date": "2024-01-02T00:00:00",
    "page_count": 5
  }
}
```

**Unterstützte Formate:**
- **Eingabe:** PDF, DOCX, DOC, TXT, HTML, und weitere von Docling unterstützte Formate
- **Ausgabe:** Markdown, JSON, Text

## Reranking API (Cohere-kompatibel)

### Endpoint

```bash
POST /v1/rerank
Content-Type: application/json
```

```bash
POST /v1/rerank
Content-Type: application/json

{
  "query": "Was ist Machine Learning?",
  "documents": [
    "Machine Learning ist ein Teilbereich der künstlichen Intelligenz.",
    "Python ist eine Programmiersprache.",
    "Deep Learning verwendet neuronale Netze."
  ],
  "top_n": 2,  # Optional: Anzahl der Top-Ergebnisse
  "model": "balanced"  # Optional: Modellname (überschreibt ENV-Variable)
}
```

**Response (Cohere-kompatibel):**

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95
    },
    {
      "index": 2,
      "relevance_score": 0.72
    }
  ],
  "id": "07734bd2-2473-4f07-94e1-0d9f0e6843cf",
  "meta": {
    "api_version": {
      "version": "1.0",
      "is_experimental": false
    },
    "billed_units": {
      "search_units": 1
    }
  }
}
```

**Hinweis:** Die `index`-Werte verweisen auf die Position im ursprünglichen `documents`-Array.

### Verfügbare Reranker-Modelle

| Alias | Modellname | Größe | Geschwindigkeit | Genauigkeit | Empfohlen für |
|-------|------------|-------|-----------------|-------------|---------------|
| `fast` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB | ⚡⚡⚡ Sehr schnell | ⭐⭐⭐ Gut | Standard, hoher Durchsatz (BERT-basiert) |
| `balanced` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~120 MB | ⚡⚡ Schnell | ⭐⭐⭐⭐ Sehr gut | Gute Balance (BERT-basiert) |
| `accurate` | `cross-encoder/ms-marco-electra-base` | ~450 MB | ⚡ Langsam | ⭐⭐⭐⭐⭐ Ausgezeichnet | Sehr genau (ELECTRA-basiert) |
| `bge-v2` | `BAAI/bge-reranker-v2-m3` | ~560 MB | ⚡⚡ Schnell | ⭐⭐⭐⭐⭐ Ausgezeichnet | **Empfohlen: Beste Balance** |
| `bge-large` | `BAAI/bge-reranker-large-v2` | ~1.3 GB | ⚡ Langsam | ⭐⭐⭐⭐⭐⭐ Hervorragend | Höchste Genauigkeit, mehrsprachig |
| `zerank-1` | `zeroentropy/zerank-1` | ~1.1 GB | ⚡ Langsam | ⭐⭐⭐⭐⭐⭐ Hervorragend | ZeroEntropy, sehr leistungsstark (nicht-kommerziell) |
| `zerank-1-small` | `zeroentropy/zerank-1-small` | ~440 MB | ⚡⚡ Schnell | ⭐⭐⭐⭐⭐ Ausgezeichnet | ZeroEntropy, Apache 2.0 Lizenz |
| `qwen3-0.6b` | `Qwen/Qwen3-Reranker-0.6B` | ~1.2 GB | ⚡⚡ Schnell | ⭐⭐⭐⭐⭐ Ausgezeichnet | Qwen3 0.6B, schnell, mehrsprachig, Apache 2.0 |
| `qwen3-4b` | `Qwen/Qwen3-Reranker-4B` | ~8 GB | ⚡ Langsam | ⭐⭐⭐⭐⭐⭐ Hervorragend | Qwen3 4B, beste Balance, mehrsprachig, Apache 2.0 |
| `qwen3-8b` | `Qwen/Qwen3-Reranker-8B` | ~16 GB | ⚡⚡⚡ Sehr langsam | ⭐⭐⭐⭐⭐⭐⭐ Extrem hoch | Qwen3 8B, höchste Genauigkeit, mehrsprachig, Apache 2.0 |
| `bert-german` | `deepset/gbert-base-germandpr-reranking` | ~440 MB | ⚡⚡ Schnell | ⭐⭐⭐⭐⭐ Ausgezeichnet | **Deutsche Texte** (German BERT) |

**Standard:** `fast` (cross-encoder/ms-marco-MiniLM-L-6-v2)

## Embeddings API (OpenAI-kompatibel)

### Endpoint

```bash
POST /v1/embeddings
Content-Type: application/json
```

### Request-Format

```json
{
  "model": "bge-base",
  "input": "Text zum Embedden"
}
```

Oder für mehrere Texte:

```json
{
  "model": "bge-large",
  "input": [
    "Erster Text",
    "Zweiter Text",
    "Dritter Text"
  ]
}
```

**Response (OpenAI-kompatibel):**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    }
  ],
  "model": "BAAI/bge-base-en-v1.5",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

### Verfügbare Embedding-Modelle

| Alias | Modellname | Größe | Geschwindigkeit | Genauigkeit | Empfohlen für |
|-------|------------|-------|-----------------|-------------|---------------|
| Kurzname     | HuggingFace-Modell                              | Größe   | Geschwindigkeit | Qualität                  | Einsatz                                   | Dimensionen |
|--------------|------------------------------------------------|---------|-----------------|---------------------------|-------------------------------------------|-------------|
| `bge-base`   | `BAAI/bge-base-en-v1.5`                         | ~130 MB | ⚡⚡ Schnell     | ⭐⭐⭐⭐⭐ Ausgezeichnet       | **Standard**, englische Texte, beste Balance | **768**     |
| `bge-large`  | `BAAI/bge-large-en-v1.5`                        | ~335 MB | ⚡ Langsam      | ⭐⭐⭐⭐⭐⭐ Hervorragend      | Höchste Genauigkeit, englische Texte        | **1024**    |
| `jina-de`    | `jinaai/jina-embeddings-v2-base-de`             | ~130 MB | ⚡⚡ Schnell     | ⭐⭐⭐⭐⭐ Ausgezeichnet       | **Deutsche Texte**, mehrsprachig            | **768**     |
| `smollm3-de` | `mayflowergmbh/smollm3-3b-german-embed`         | ~6 GB   | ⚡⚡⚡ Sehr langsam | ⭐⭐⭐⭐⭐⭐ Hervorragend    | Deutsche Texte, sehr große Dimensionen      | **4096**    |


**Standard:** `bge-base` (BAAI/bge-base-en-v1.5)

**Hinweis:** Der Modellname muss im Request-Body angegeben werden (OpenAI-Standard).

## Beispiel-Requests

### Reranking

#### Reranking mit cURL

```bash
curl -X POST "http://localhost:8888/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python Programmierung",
    "documents": [
      "Python ist eine interpretierte Programmiersprache.",
      "Java ist eine objektorientierte Sprache.",
      "Python wird häufig für Data Science verwendet."
    ],
    "top_n": 2,
    "model": "bge-v2"
  }'
```

#### Reranking mit Python (Cohere-kompatibel)

```python
import requests

response = requests.post(
    "http://localhost:8888/v1/rerank",
    json={
        "query": "Python Programmierung",
        "documents": [
            "Python ist eine interpretierte Programmiersprache.",
            "Java ist eine objektorientierte Sprache.",
            "Python wird häufig für Data Science verwendet."
        ],
        "top_n": 2
    }
)

result = response.json()
print(f"Request ID: {result['id']}")
for item in result['results']:
    doc_index = item['index']
    score = item['relevance_score']
    print(f"Index {doc_index}: Score {score}")
```

### Cohere SDK-kompatibel

Das API-Format ist kompatibel mit Cohere, sodass Sie den Code nahezu unverändert verwenden können:

```python
# Statt: co = cohere.ClientV2()
# Verwenden Sie: requests.post("http://localhost:8888/v1/rerank", ...)

docs = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands...",
    "Capitalization or capitalisation in English grammar...",
    "Washington, D.C. is the capital of the United States...",
    "Capital punishment has existed in the United States...",
]

response = requests.post(
    "http://localhost:8888/v1/rerank",
    json={
        "query": "What is the capital of the United States?",
        "documents": docs,
        "top_n": 3,
    }
)

print(response.json())
```

### Embeddings

#### Embeddings mit cURL

```bash
curl -X POST "http://localhost:8888/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-base",
    "input": "Was ist Machine Learning?"
  }'
```

Oder für mehrere Texte:

```bash
curl -X POST "http://localhost:8888/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-large",
    "input": [
      "Machine Learning ist ein Teilbereich der KI.",
      "Python ist eine Programmiersprache.",
      "Deep Learning verwendet neuronale Netze."
    ]
  }'
```

#### Embeddings mit Python (OpenAI-kompatibel)

```python
import requests

# Einzelner Text
response = requests.post(
    "http://localhost:8888/v1/embeddings",
    json={
        "model": "bge-base",
        "input": "Was ist Machine Learning?"
    }
)

result = response.json()
print(f"Modell: {result['model']}")
print(f"Embedding-Dimension: {len(result['data'][0]['embedding'])}")
print(f"Token-Usage: {result['usage']}")

# Mehrere Texte
response = requests.post(
    "http://localhost:8888/v1/embeddings",
    json={
        "model": "jina-de",  # Deutsches Modell
        "input": [
            "Machine Learning ist ein Teilbereich der KI.",
            "Python ist eine Programmiersprache.",
            "Deep Learning verwendet neuronale Netze."
        ]
    }
)

result = response.json()
for item in result['data']:
    print(f"Index {item['index']}: Embedding mit {len(item['embedding'])} Dimensionen")
```

#### OpenAI SDK-kompatibel

Das API-Format ist kompatibel mit OpenAI, sodass Sie den Code nahezu unverändert verwenden können:

```python
# Statt: openai.Embedding.create(...)
# Verwenden Sie: requests.post("http://localhost:8888/v1/embeddings", ...)

import requests

response = requests.post(
    "http://localhost:8888/v1/embeddings",
    json={
        "model": "bge-base",
        "input": "Text zum Embedden"
    }
)

result = response.json()
embedding = result['data'][0]['embedding']
print(f"Embedding-Vektor: {embedding[:5]}...")  # Erste 5 Werte
```

## Lokale Entwicklung (ohne Docker)

```bash
# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# PyTorch CPU-only installieren (wichtig!)
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Server starten
uvicorn app.main:app --reload --host 0.0.0.0 --port 8888
```

### Schnellstart in der Konsole

```bash
# 1. Virtual Environment aktivieren (falls vorhanden)
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. API starten
uvicorn app.main:app --reload --host 0.0.0.0 --port 8888

# 3. In einem anderen Terminal: API testen
python test_api.py
```

Die APIs sind dann unter folgenden URLs erreichbar:
- **Reranker & Embeddings API**: `http://localhost:8888`
- **Docling Document Processing API**: `http://localhost:8890`

## Modell-Konfiguration

### Reranker-Modell auswählen

Das Reranker-Modell kann über die Umgebungsvariable `RERANKER_MODEL` ausgewählt werden. Siehe Tabelle oben für verfügbare Modelle.

**Standard:** `fast` (cross-encoder/ms-marco-MiniLM-L-6-v2)

**Empfehlung für beste CPU-Performance:** `bge-v2` (BAAI/bge-reranker-v2-m3) - bietet die beste Balance zwischen Genauigkeit und Geschwindigkeit auf CPU.

### Embedding-Modell auswählen

Das Embedding-Modell wird im Request-Body angegeben (OpenAI-Standard). Siehe Tabelle oben für verfügbare Modelle.

**Standard:** `bge-base` (BAAI/bge-base-en-v1.5)

**Empfehlung:**
- **Englische Texte:** `bge-base` oder `bge-large`
- **Deutsche Texte:** `jina-de` oder `smollm3-de`

### Hugging Face Token

Für private Modelle oder Modelle mit Zugriffsbeschränkungen können Sie einen Hugging Face Token über die Umgebungsvariable `HF_TOKEN` angeben:

```bash
export HF_TOKEN=your_hf_token_here
```

Token erhalten Sie unter: https://huggingface.co/settings/tokens

### Modell auswählen

#### Mit Docker Compose

```bash
# .env-Datei erstellen (optional)
# Kopieren Sie .env.example zu .env und passen Sie die Werte an
cp .env.example .env
# Oder manuell:
echo "RERANKER_MODEL=balanced" > .env
echo "HF_TOKEN=your_hf_token_here" >> .env  # Optional: Für private Modelle

# Oder direkt beim Start
RERANKER_MODEL=balanced docker-compose up
```

Oder in der `docker-compose.yml`:

```yaml
environment:
  - RERANKER_MODEL=balanced  # oder "fast", "accurate" oder direkter Modellname
  - HF_TOKEN=your_hf_token_here  # Optional: Für private Modelle
```

#### Lokale Entwicklung

```bash
# Umgebungsvariable setzen
export RERANKER_MODEL=balanced

# Hugging Face Token für private Modelle (optional)
export HF_TOKEN=your_hf_token_here

# Server starten
uvicorn app.main:app --reload
```

#### Modell-Informationen abrufen

```bash
# Aktuelles Modell und verfügbare Optionen anzeigen
curl http://localhost:8888/model/info
```

### Direkter Modellname

Sie können auch direkt einen Modellnamen verwenden:

```bash
# Reranker
RERANKER_MODEL=cross-encoder/stsb-roberta-base docker-compose up

# Embeddings (im Request-Body)
# Verwenden Sie den vollständigen Modellnamen im "model"-Feld
```

## Performance-Optimierungen

- **Model-Caching**: Modelle werden in einem Docker-Volume gecacht
- **Thread-Pool**: Asynchrone Verarbeitung für bessere Performance
- **CPU-Optimierung**: PyTorch läuft standardmäßig auf CPU

## Projektstruktur

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI-Anwendung
│   ├── reranker.py      # Reranker-Service
│   ├── embeddings.py     # Embeddings-Service
│   └── models.py         # Datenmodelle (Reranker + Embeddings)
├── Dockerfile            # CPU-optimiertes Dockerfile
├── Dockerfile.runpod     # GPU-optimiertes Dockerfile für RunPod
├── docker-compose.yml
├── requirements.txt
├── runpod_template.json  # RunPod Template-Konfiguration (Legacy)
├── RUNPOD_DEPLOYMENT.md  # RunPod Deployment-Anleitung
├── handler.py           # RunPod Serverless Handler (im Arbeitsverzeichnis)
├── .runpod/
│   ├── hub.json         # RunPod Hub-Konfiguration
│   ├── tests.json       # RunPod Test-Konfiguration
│   └── start.sh         # Start-Script für RunPod
└── README.md
```

## RunPod Deployment

Die API kann auch auf RunPod mit GPU-Unterstützung bereitgestellt werden für deutlich bessere Performance.

Siehe [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) für eine detaillierte Anleitung.

### RunPod Hub

Die API ist für RunPod Hub vorbereitet mit:
- ✅ `.runpod/hub.json` - Hub-Konfiguration mit Presets
- ✅ `.runpod/tests.json` - Test-Konfiguration
- ✅ `.runpod/handler.py` - Serverless Handler
- ✅ `Dockerfile.runpod` - GPU-optimiertes Dockerfile

### Schnellstart RunPod

```bash
# 1. GPU-fähiges Image bauen
docker build -f Dockerfile.runpod -t your-username/ai-local-reranker:latest .

# 2. Zu Docker Hub pushen
docker push your-username/ai-local-reranker:latest

# 3. RunPod Hub Template erstellen (siehe RUNPOD_DEPLOYMENT.md)
# Oder verwenden Sie die .runpod/hub.json Konfiguration
```

**Vorteile von RunPod:**
- ⚡ **GPU-Beschleunigung** - 10-100x schneller als CPU
- 💰 **Pay-per-use** - Nur zahlen, wenn aktiv
- 🔄 **Auto-Scaling** - Automatische Skalierung bei Last
- 🚀 **Schnelle Deployment** - In Minuten live
- 🏪 **RunPod Hub** - Einfache Veröffentlichung und Nutzung

## Lizenz

MIT

