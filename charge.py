import requests
import random
from concurrent.futures import ThreadPoolExecutor

def send_request(index):
    data = {
        "sepal_length": round(random.uniform(4.3, 7.9), 1),
        "sepal_width": round(random.uniform(2.0, 4.4), 1),
        "petal_length": round(random.uniform(1.0, 6.9), 1),
        "petal_width": round(random.uniform(0.1, 2.5), 1)
    }
    url = 'https://grp6-container-app.salmonisland-ae09a70e.francecentral.azurecontainerapps.io/predict'
    
    response = requests.post(url, json=data)
    print(f"Réponse du serveur pour la requête {index}: {response.text}")

num_requests = 100

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(send_request, i) for i in range(num_requests)]

    for future in futures:
        future.result()
