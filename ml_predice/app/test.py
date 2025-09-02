import requests

url = "http://127.0.0.1:5000/predict"
data = {"edad": 25, "grado": 4, "tiempo_seg": 150, "intentos": 2, "pistas": 1}

response = requests.post(url, json=data)
print(response.json())
