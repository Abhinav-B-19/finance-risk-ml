from fastapi import FastAPI
from src.predict import predict
from src.schemas.prediction import PredictionInput

app = FastAPI()

@app.post("/predict")
def predict_api(data: PredictionInput):
    return predict(data.dict(), explain=False)