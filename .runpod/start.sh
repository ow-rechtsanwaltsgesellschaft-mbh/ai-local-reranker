#!/bin/bash
# Start-Script für RunPod: Startet API und Handler parallel

# Starte FastAPI im Hintergrund
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --timeout-keep-alive 120 \
    --timeout-graceful-shutdown 120 &

# Warte kurz, damit API startet
sleep 5

# Starte RunPod Handler
python handler.py

