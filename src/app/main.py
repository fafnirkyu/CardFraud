# src/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import os

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.json")

app = FastAPI(title="Churn predictor")

class Features(BaseModel):
    # create dynamic fields or a single list/ dict depending on your schema
    features: dict

model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)

@app.post("/predict")
def predict(payload: Features):
    try:
        df = pd.DataFrame([payload.features])
        dmat = xgb.DMatrix(df)
        prob = model.predict(dmat)[0]
        return {"probability": float(prob), "prediction": int(prob > 0.5)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
