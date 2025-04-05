import requests
import json

def check_server_health_request(server_addr: str = "http://127.0.0.1", port: int = 8003, endpoint: str = "healthy") -> bool:
    # Define the server URL
    server_url = f"{server_addr}:{port}/{endpoint}/"
    
    try:
        response = requests.get(server_url)
    except:
        return False
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        if "status" in result and result["status"] == "OK":
            return True
        else:
            return False
    else:
        print(f"Error: {response.status_code, response.json()}")
        return False
