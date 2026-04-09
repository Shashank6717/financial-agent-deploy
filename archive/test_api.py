import requests

base_url = "http://localhost:8000/api/news"

queries = [
    "What is the capital of France?",
    "What is the latest news for Apple?",
    "What is the current stock price of Apple?"
]

for query in queries:
    print(f"\nTesting Query: {query}")
    try:
        response = requests.post(base_url, json={"query": query})
        print(f"Response: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")
