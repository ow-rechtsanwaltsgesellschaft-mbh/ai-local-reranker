# Multi-Stage Build für optimale Image-Größe
FROM python:3.11-slim as builder

# Arbeitsverzeichnis setzen
WORKDIR /app

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten kopieren und installieren
# PyTorch CPU-only installieren (kleineres Image, keine GPU-Abhängigkeiten)
COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    --index-url https://download.pytorch.org/whl/cpu \
    torch>=2.2.0 && \
    pip install --no-cache-dir --user -r requirements.txt

# Production-Stage
FROM python:3.11-slim

# Arbeitsverzeichnis setzen
WORKDIR /app

# Python-Abhängigkeiten vom Builder-Stage kopieren
COPY --from=builder /root/.local /root/.local

# Umgebungsvariablen setzen
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES="" \
    TORCH_DEVICE=cpu

# App-Code kopieren
COPY app/ ./app/

# Port freigeben
EXPOSE 8000

# Health-Check mit erhöhten Timeouts für Modell-Laden
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Uvicorn starten mit erhöhten Timeouts für große Modelle
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120", "--timeout-graceful-shutdown", "120"]

