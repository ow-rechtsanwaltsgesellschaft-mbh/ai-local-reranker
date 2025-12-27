# AI Local Reranker API

[![Runpod](https://api.runpod.io/badge/ow-rechtsanwaltsgesellschaft-mbh/ai-local-reranker)](https://console.runpod.io/hub/ow-rechtsanwaltsgesellschaft-mbh/ai-local-reranker)

Eine Python-API für lokales Reranking auf CPU/GPU mit FastAPI und sentence-transformers.

## Features

- 🚀 **Lokales Reranking auf CPU/GPU** - Keine externe API erforderlich
- ⚡ **Schnell und effizient** - Optimiert für CPU- und GPU-Performance
- 🐳 **Docker-ready** - Einfache Bereitstellung mit Docker Compose
- ☁️ **RunPod-ready** - GPU-optimiertes Deployment auf RunPod
- 📊 **RESTful API** - Standardisierte Endpoints
- 🔧 **Best Practices** - Production-ready Code
- 🔄 **Cohere-kompatibel** - Gleiches Request/Response-Format wie Cohere Rerank API

## Technologie-Stack

- **FastAPI** - Moderne, schnelle Web-Framework
- **sentence-transformers** - CrossEncoder für Reranking
- **PyTorch** - CPU/GPU-optimiertes Machine Learning
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

Die API ist dann unter `http://localhost:8888` erreichbar.

### API-Dokumentation

Nach dem Start können Sie die interaktive API-Dokumentation unter folgenden URLs aufrufen:

- **Swagger UI**: http://localhost:8888/docs
- **ReDoc**: http://localhost:8888/redoc

## API-Endpoints

### Health Check

```bash
GET /health
```

### Modell-Informationen

```bash
GET /model/info
```

Zeigt das aktuell verwendete Modell und verfügbare Optionen an.

### Reranking (Cohere-kompatibel)

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

## Beispiel-Requests

### Mit cURL

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
    "top_n": 2
  }'
```

### Mit Python (Cohere-kompatibel)

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

Die API ist dann unter `http://localhost:8888` erreichbar.

## Modell-Konfiguration

Das Reranker-Modell kann über die Umgebungsvariable `RERANKER_MODEL` ausgewählt werden. Es stehen mehrere Modelle zur Verfügung, die unterschiedliche Balance zwischen Geschwindigkeit und Genauigkeit bieten:

### Verfügbare Modelle

| Alias | Modellname | Größe | Geschwindigkeit | Genauigkeit | Empfohlen für |
|-------|------------|-------|-----------------|-------------|---------------|
| `fast` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB | ⚡⚡⚡ Sehr schnell | ⭐⭐⭐ Gut | Standard, hoher Durchsatz (BERT-basiert) |
| `balanced` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~120 MB | ⚡⚡ Schnell | ⭐⭐⭐⭐ Sehr gut | Gute Balance (BERT-basiert) |
| `bge-v2` | `BAAI/bge-reranker-v2-m3` | ~560 MB | ⚡⚡ Schnell | ⭐⭐⭐⭐⭐ Ausgezeichnet | **Empfohlen: Beste Balance** |
| `bge-large` | `BAAI/bge-reranker-large-v2` | ~1.3 GB | ⚡ Langsam | ⭐⭐⭐⭐⭐⭐ Hervorragend | Höchste Genauigkeit, mehrsprachig |
| `zerank-1` | `zeroentropy/zerank-1` | ~1.1 GB | ⚡ Langsam | ⭐⭐⭐⭐⭐⭐ Hervorragend | ZeroEntropy, sehr leistungsstark (nicht-kommerziell) |
| `zerank-1-small` | `zeroentropy/zerank-1-small` | ~440 MB | ⚡⚡ Schnell | ⭐⭐⭐⭐⭐ Ausgezeichnet | ZeroEntropy, Apache 2.0 Lizenz |
| `qwen3-0.6b` | `Qwen/Qwen3-Reranker-0.6B` | ~1.2 GB | ⚡⚡ Schnell | ⭐⭐⭐⭐⭐ Ausgezeichnet | Qwen3 0.6B, mehrsprachig (119 Sprachen), Apache 2.0 |
| `qwen3-4b` | `Qwen/Qwen3-Reranker-4B` | ~8 GB | ⚡ Langsam | ⭐⭐⭐⭐⭐⭐ Hervorragend | Qwen3 4B, beste Balance, mehrsprachig, Apache 2.0 |
| `qwen3-8b` | `Qwen/Qwen3-Reranker-8B` | ~16 GB | ⚡⚡⚡ Sehr langsam | ⭐⭐⭐⭐⭐⭐⭐ Exzellent | Qwen3 8B, höchste Genauigkeit, mehrsprachig, Apache 2.0 |
| `bert-german` | `deepset/gbert-base-germandpr-reranking` | ~440 MB | ⚡⚡ Schnell | ⭐⭐⭐⭐⭐ Ausgezeichnet | **Deutsche Texte** (German BERT) |

**Standard:** `fast` (cross-encoder/ms-marco-MiniLM-L-6-v2)

**Empfehlung für beste CPU-Performance:** `bge-v2` (BAAI/bge-reranker-v2-m3) - bietet die beste Balance zwischen Genauigkeit und Geschwindigkeit auf CPU.

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
RERANKER_MODEL=cross-encoder/stsb-roberta-base docker-compose up
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
│   └── models.py        # Datenmodelle
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

