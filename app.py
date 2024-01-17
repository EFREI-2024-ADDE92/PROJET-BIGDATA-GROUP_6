import joblib
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()
model = joblib.load("knn-model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def pred_iris(iris_input: IrisInput):
    features = [
        [iris_input.sepal_length, iris_input.sepal_width, iris_input.petal_length, iris_input.petal_width]
    ]
    pred = model.predict(features)
    return {"prediction": int(pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
