import joblib
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter
from fastapi.responses import StreamingResponse



app = FastAPI()
model = joblib.load("knn-model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

setosa_counter = Counter("model_prediction_setosa_total", "Number of predictions for setosa")
versicolor_counter = Counter("model_prediction_versicolor_total", "Number of predictions for versicolor")
virginica_counter = Counter("model_prediction_virginica_total", "Number of predictions for virginica")

@app.post("/predict")
def pred_iris(iris_input: IrisInput):
    features = [
        [iris_input.sepal_length, iris_input.sepal_width, iris_input.petal_length, iris_input.petal_width]
    ]
    #pred = model.predict(features)
    pred_class = int(model.predict(features)[0])

    pred_name = class_names.get(pred_class, 'Classe inconnue')

    
    if pred_name == 'setosa':
        setosa_counter.inc()
    elif pred_name == 'versicolor':
        versicolor_counter.inc()
    elif pred_name == 'virginica':
        virginica_counter.inc()

    #return {"prediction": int(pred[0])}
    return {"prediction": pred_name}

@app.get("/metrics")
async def get_metrics():
    async def generate():
        metrics_data = generate_latest()
        yield metrics_data

    return StreamingResponse(content=generate(), media_type=CONTENT_TYPE_LATEST)




instrumentator = Instrumentator()
instrumentator.instrument(app)


instrumentator.expose(app, include_in_schema=False, should_gzip=True)





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
