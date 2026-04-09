import requests
import json

def test_articles():
    url = "http://localhost:8000/api/news"
    payload = {
        "query": "Get the latest news about Tesla and Elon Musk.",
        "reset": True
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        answer = data.get("answer", "No answer")
        articles = data.get("top_articles", [])
        
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Top Articles ---")
        print(f"Count: {len(articles)}")
        for i, art in enumerate(articles, 1):
            print(f"{i}. {art.get('headline')} ({art.get('source')})")
            
        if len(articles) >= 5:
            print("\n✅ SUCCESS: Returned at least 5 articles.")
        else:
            print(f"\n❌ FAILURE: Returned only {len(articles)} articles.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_articles()
