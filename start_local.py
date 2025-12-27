#!/usr/bin/env python3
"""
Lokaler Start der Reranker-API für Tests.
"""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starte Reranker API lokal...")
    print("📡 API erreichbar unter: http://localhost:8000")
    print("📚 Dokumentation: http://localhost:8000/docs")
    print("🛑 Beenden mit Ctrl+C")
    print()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Automatisches Neuladen bei Code-Änderungen
        log_level="info",
        timeout_keep_alive=120,  # Erhöhtes Timeout für große Modelle
        timeout_graceful_shutdown=120
    )

