import requests
import json

TOGETHER_API_KEY = ""  # reomve 

headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json",
}

resp = requests.get("https://api.together.ai/v1/models", headers=headers)

print("Status Code:", resp.status_code)
print("Content-Type:", resp.headers.get("Content-Type", ""))

data = resp.json()

print("\n=== Available models (id — display_name — type) ===\n")
for m in data:
    print(f"{m.get('id')}  —  {m.get('display_name')}  —  {m.get('type')}")
