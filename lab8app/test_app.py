import requests

url = "http://localhost:8000/predict"

data = {
    "danceability": 0.7,
    "energy": 0.8,
    "key": 5,
    "loudness": -5.5,
    "mode": 1,
    "speechiness": 0.05,
    "acousticness": 0.12,
    "instrumentalness": 0.0,
    "liveness": 0.1,
    "valence": 0.65,
    "tempo": 115.0,
    "duration_ms": 210000,
    "explicit": 0
}

response = requests.post(url, json=data)
print(response.json())
