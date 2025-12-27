"""
RunPod Serverless Handler für AI Local Reranker API.
Dieser Handler ermöglicht die Verwendung der API auf RunPod Serverless.
"""
import runpod
import requests
import json
import time
from typing import Dict, Any

# API-URL (läuft im gleichen Container)
API_URL = "http://localhost:8000"
MAX_RETRIES = 3
RETRY_DELAY = 5


def wait_for_api(max_wait=120):
    """Wartet bis die API verfügbar ist."""
    for _ in range(max_wait // 2):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless Handler.
    
    Args:
        event: Event-Daten von RunPod
        
    Returns:
        Response-Daten
    """
    # Warte auf API, falls sie noch startet
    if not wait_for_api():
        return {
            "status": "error",
            "error": "API ist nicht verfügbar - Modell lädt möglicherweise noch"
        }
    
    try:
        # Prüfe ob es ein Health-Check ist
        if "endpoint" in event and event.get("endpoint") == "/health":
            response = requests.get(f"{API_URL}/health", timeout=10)
            return {
                "status": "success",
                "status_code": response.status_code,
                "data": response.json()
            }
        
        # Standard: Rerank-Request
        if "endpoint" in event:
            endpoint = event["endpoint"]
            method = event.get("method", "POST")
            body = event.get("body", {})
            
            if method == "GET":
                response = requests.get(f"{API_URL}{endpoint}", timeout=30)
            else:
                response = requests.post(
                    f"{API_URL}{endpoint}",
                    json=body,
                    timeout=120  # Erhöhtes Timeout für große Modelle
                )
            
            return {
                "status": "success",
                "status_code": response.status_code,
                "data": response.json()
            }
        else:
            # Direkter Rerank-Request (Cohere-Format)
            rerank_data = {
                "query": event.get("query", ""),
                "documents": event.get("documents", []),
                "top_n": event.get("top_n"),
                "model": event.get("model")
            }
            
            response = requests.post(
                f"{API_URL}/rerank",
                json=rerank_data,
                timeout=120  # Erhöhtes Timeout für große Modelle
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "data": response.json()
                }
            else:
                return {
                    "status": "error",
                    "error": response.text,
                    "status_code": response.status_code
                }
    
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error": "Request timeout - Modell verarbeitet möglicherweise noch"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# RunPod Handler registrieren
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

