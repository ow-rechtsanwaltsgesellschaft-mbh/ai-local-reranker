# RunPod Deployment Guide

Anleitung zur Bereitstellung der AI Local Reranker API auf RunPod.

## Voraussetzungen

1. RunPod-Konto erstellen: https://runpod.io/
2. Docker Hub-Konto (oder andere Container-Registry)
3. RunPod API-Key (in den Einstellungen generieren)

## Schritt 1: Docker Image erstellen und hochladen

### Lokal bauen und testen

```bash
# GPU-fähiges Image bauen
docker build -f Dockerfile.runpod -t your-dockerhub-username/ai-local-reranker:latest .

# Optional: Lokal testen (benötigt NVIDIA Docker)
docker run --gpus all -p 8000:8000 \
  -e RERANKER_MODEL=bge-v2 \
  your-dockerhub-username/ai-local-reranker:latest
```

### Zu Docker Hub pushen

```bash
# Bei Docker Hub anmelden
docker login

# Image taggen (falls noch nicht geschehen)
docker tag your-dockerhub-username/ai-local-reranker:latest your-dockerhub-username/ai-local-reranker:latest

# Image hochladen
docker push your-dockerhub-username/ai-local-reranker:latest
```

## Schritt 2: RunPod Template erstellen

### Option A: Über RunPod Web-Interface

1. Gehen Sie zu https://www.runpod.io/console/serverless
2. Klicken Sie auf "New Template"
3. Füllen Sie die folgenden Felder aus:
   - **Template Name**: `AI Local Reranker API`
   - **Container Image**: `your-dockerhub-username/ai-local-reranker:latest`
   - **Container Disk**: `20 GB` (für Modelle)
   - **Volume**: `10 GB` (für Model-Cache)
   - **Volume Mount Path**: `/root/.cache`
   - **Port**: `8000/http`
   - **Environment Variables**:
     - `RERANKER_MODEL=bge-v2` (oder gewünschtes Modell)
     - `PYTHONUNBUFFERED=1`

### Option B: Über RunPod API

```bash
# RunPod CLI installieren
pip install runpod

# Template erstellen (verwenden Sie Ihre API-Key)
runpod create_template \
  --name "AI Local Reranker API" \
  --image "your-dockerhub-username/ai-local-reranker:latest" \
  --container-disk 20 \
  --volume 10 \
  --volume-mount-path "/root/.cache" \
  --ports "8000/http" \
  --env "RERANKER_MODEL=bge-v2" \
  --env "PYTHONUNBUFFERED=1"
```

## Schritt 3: Endpoint erstellen

1. Gehen Sie zu https://www.runpod.io/console/serverless
2. Klicken Sie auf "New Endpoint"
3. Wählen Sie das erstellte Template
4. Konfigurieren Sie:
   - **GPU Type**: Empfohlen: `RTX 3090`, `RTX 4090` oder `A100` (je nach Modell)
   - **Max Workers**: `1-3` (je nach GPU-Speicher)
   - **Idle Timeout**: `5 Minuten`
   - **Flashboot**: Aktiviert (für schnelleres Starten)

## Schritt 4: API verwenden

Nach dem Erstellen des Endpoints erhalten Sie eine URL wie:
```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID
```

### Beispiel-Request

```python
import requests

endpoint_url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/rerank"

response = requests.post(
    endpoint_url,
    json={
        "query": "What is machine learning?",
        "documents": [
            "Machine Learning is a subset of AI.",
            "Python is a programming language.",
            "Deep Learning uses neural networks."
        ],
        "top_n": 3,
        "model": "bge-v2"
    },
    headers={
        "Authorization": "Bearer YOUR_RUNPOD_API_KEY"
    }
)

print(response.json())
```

### Mit cURL

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "query": "What is machine learning?",
    "documents": [
      "Machine Learning is a subset of AI.",
      "Python is a programming language."
    ],
    "top_n": 3
  }'
```

## Modell-Auswahl für GPU

Empfohlene Modelle für GPU (schneller als CPU):

- **`bge-v2`** oder **`bge-large`**: Sehr gute Performance auf GPU
- **`qwen3-4b`** oder **`qwen3-8b`**: Beste Genauigkeit, benötigt mehr VRAM
- **`zerank-1`**: Sehr leistungsstark auf GPU

Kleinere Modelle wie `fast` oder `balanced` laufen auch auf GPU, profitieren aber weniger davon.

## GPU-Anforderungen

| Modell | Empfohlene GPU | VRAM |
|--------|----------------|------|
| `fast`, `balanced` | RTX 3060 / RTX 3090 | 8-12 GB |
| `bge-v2`, `bge-large` | RTX 3090 / RTX 4090 | 12-16 GB |
| `qwen3-0.6b` | RTX 3090 | 12 GB |
| `qwen3-4b` | RTX 4090 / A100 | 16-24 GB |
| `qwen3-8b` | A100 40GB | 32+ GB |
| `zerank-1` | RTX 4090 / A100 | 16-24 GB |

## Kosten-Optimierung

1. **Flashboot aktivieren**: Reduziert Cold-Start-Zeit
2. **Idle Timeout**: Auf 5 Minuten setzen (spart Kosten bei Inaktivität)
3. **Kleineres Modell wählen**: Wenn Genauigkeit ausreicht
4. **Volume für Model-Cache**: Modelle werden zwischen Starts gecacht

## Troubleshooting

### Modell lädt zu langsam
- Erhöhen Sie `Container Disk` auf 30-50 GB
- Verwenden Sie Volume für Model-Cache
- Wählen Sie ein kleineres Modell

### Out of Memory (OOM)
- Wählen Sie eine GPU mit mehr VRAM
- Reduzieren Sie `Max Workers` auf 1
- Verwenden Sie ein kleineres Modell

### Timeout-Fehler
- Erhöhen Sie `Handler Timeout` auf 300+ Sekunden
- Verwenden Sie Flashboot für schnelleres Starten

## Weitere Ressourcen

- [RunPod Dokumentation](https://docs.runpod.io/)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
- [RunPod Discord Community](https://discord.gg/runpod)

