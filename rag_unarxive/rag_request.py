import requests
import json

def rag_request(query: str, k: int, server_addr: str = "http://127.0.0.1", port: str = "8001", endpoint: str = "rag_retrieve"):
    # Define the server URL
    server_url = f"{server_addr}:{port}/{endpoint}/"
    
    # Query data
    query_json = {"query": query, "k": k}
    
    # Make a POST request to the server
    print("sending RAG request...")
    response = requests.post(server_url, json=query_json)
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code, response.json()}")
        return None