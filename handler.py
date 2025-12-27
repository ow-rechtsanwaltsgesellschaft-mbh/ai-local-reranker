"""
RunPod Serverless Handler für AI Local Reranker API.
Dieser Handler ermöglicht die Verwendung der API auf RunPod Serverless.
"""
import runpod  # Required
import requests
import json
import time
import sys
import os
from typing import Dict, Any

# Füge app-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# API-URL (läuft im gleichen Container)
API_URL = "http://localhost:8000"
MAX_WAIT_TIME = 180  # 3 Minuten für Modell-Laden

# Globale Variable für API-Status
_api_ready = False


def wait_for_api(max_wait=MAX_WAIT_TIME):
    """Wartet bis die API verfügbar ist."""
    global _api_ready
    if _api_ready:
        return True
    
    for i in range(max_wait // 2):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                _api_ready = True
                return True
        except:
            pass
        if i % 5 == 0:  # Log alle 10 Sekunden
            print(f"Warte auf API... ({i * 2}s)")
        time.sleep(2)
    return False


def handler(event):
    """
    RunPod Serverless Handler.
    
    Extrahiert Eingabedaten aus event["input"] und verarbeitet Rerank-Requests.
    
    Unterstützt folgende Input-Formate:
    1. Direkter Rerank-Request (Cohere-Format):
       {
         "input": {
           "query": "What is machine learning?",
           "documents": ["...", "..."],
           "top_n": 3,
           "model": "bge-v2"
         }
       }
    
    2. Endpoint-basierter Request:
       {
         "input": {
           "endpoint": "/rerank",
           "method": "POST",
           "body": {...}
         }
       }
    
    3. Health-Check:
       {
         "input": {
           "endpoint": "/health"
         }
       }
    
    Args:
        event: Event-Daten von RunPod mit "input" Key
        
    Returns:
        Response-Daten im Cohere-Format oder Error
    """
    # Extract input data from the request
    input_data = event.get("input", {})
    
    # Warte auf API, falls sie noch startet
    if not wait_for_api():
        return {
            "error": "API ist nicht verfügbar - Modell lädt möglicherweise noch. Bitte versuchen Sie es erneut."
        }
    
    try:
        # Prüfe ob es ein Health-Check ist
        if "endpoint" in input_data and input_data.get("endpoint") == "/health":
            response = requests.get(f"{API_URL}/health", timeout=10)
            return {
                "status": "success",
                "status_code": response.status_code,
                "data": response.json()
            }
        
        # Standard: Rerank-Request
        if "endpoint" in input_data:
            endpoint = input_data["endpoint"]
            method = input_data.get("method", "POST")
            body = input_data.get("body", {})
            
            if method == "GET":
                response = requests.get(f"{API_URL}{endpoint}", timeout=30)
            else:
                response = requests.post(
                    f"{API_URL}{endpoint}",
                    json=body,
                    timeout=180  # Erhöhtes Timeout für große Modelle
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": response.text,
                    "status_code": response.status_code
                }
        else:
            # Direkter Rerank-Request (Cohere-Format)
            query = input_data.get("query", "")
            documents = input_data.get("documents", [])
            
            if not query:
                return {
                    "error": "Query darf nicht leer sein"
                }
            
            if not documents:
                return {
                    "error": "Dokumentenliste darf nicht leer sein"
                }
            
            rerank_data = {
                "query": query,
                "documents": documents,
                "top_n": input_data.get("top_n"),
                "model": input_data.get("model")
            }
            
            response = requests.post(
                f"{API_URL}/rerank",
                json=rerank_data,
                timeout=180  # Erhöhtes Timeout für große Modelle
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": response.text,
                    "status_code": response.status_code
                }
    
    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout - Modell verarbeitet möglicherweise noch. Bitte versuchen Sie es mit einem kleineren Modell oder weniger Dokumenten."
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "Verbindungsfehler zur API. Bitte versuchen Sie es erneut."
        }
    except Exception as e:
        return {
            "error": f"Unerwarteter Fehler: {str(e)}"
        }


# RunPod Handler registrieren
runpod.serverless.start({"handler": handler})  # Required

