import requests

api_key = "tgp_v1_SUu3yVIii0xrg9Xp_FPcQW-LXU-hRZr1nJd8u1phyZw"  # Replace this

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get("https://api.together.xyz/models", headers=headers)

# ✅ Show the response content even if it's not JSON
print("Status Code:", response.status_code)
print("Headers:", response.headers)
print("Text Response:")
print(response.text)

