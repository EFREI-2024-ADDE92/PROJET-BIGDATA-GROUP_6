import requests
from concurrent.futures import ThreadPoolExecutor

def send_request(index):
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    url = 'https://grp6-container-app.salmonisland-ae09a70e.francecentral.azurecontainerapps.io/predict'
    
    response = requests.post(url, json=data)
    print(f"Réponse du serveur pour la requête {index}: {response.text}")

num_requests = 100

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(send_request, i) for i in range(num_requests)]

    for future in futures:
        future.result()
