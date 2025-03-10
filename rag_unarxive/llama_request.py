import requests
import json

def llama_request(chat, server_addr: str = "http://127.0.0.1", port: str = "8002", endpoint: str = "llama_generate"):
    # Define the server URL
    server_url = f"{server_addr}:{port}/{endpoint}/"
    
    
    # Make a POST request to the server
    print("sending LLM request")
    response = requests.post(server_url, json=chat)
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code, response.json()}")
        return None
