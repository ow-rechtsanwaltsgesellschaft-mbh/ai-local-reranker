"""
Einfaches Test-Skript für die Reranker-API.
"""
import requests
import json

# API-URL
API_URL = "http://localhost:8888"

def test_health():
    """Testet den Health-Check Endpoint."""
    print("🔍 Teste Health-Check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_model_info():
    """Testet den Model-Info Endpoint."""
    print("🔍 Teste Model-Info...")
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_rerank():
    """Testet den Rerank Endpoint."""
    print("🔍 Teste Reranking...")
    
    test_data = {
        "query": "What is the capital of the United States?",
        "documents": [
            "Carson City is the capital city of the American state of Nevada.",
            "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
            "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
            "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
            "Capital punishment has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
        ],
        "top_n": 3
    }
    
    response = requests.post(
        f"{API_URL}/v1/rerank",
        json=test_data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Request ID: {result['id']}")
        print(f"\nTop {len(result['results'])} Ergebnisse:")
        for i, item in enumerate(result['results'], 1):
            doc_index = item['index']
            score = item['relevance_score']
            document = test_data['documents'][doc_index]
            print(f"\n{i}. Index {doc_index} (Score: {score:.4f})")
            print(f"   Dokument: {document[:80]}...")
    else:
        print(f"Fehler: {response.text}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Reranker API Test")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_model_info()
        test_rerank()
        print("✅ Alle Tests erfolgreich!")
    except requests.exceptions.ConnectionError:
        print("❌ Fehler: API ist nicht erreichbar!")
        print("   Stellen Sie sicher, dass die API läuft:")
        print("   python -m app.main")
        print("   oder")
        print("   uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Fehler: {e}")

