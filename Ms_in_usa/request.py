import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'gre_score':700, 'gpa':5, 'ses':2,'gender':0,'race':2,'rank':3})

print(r.json())