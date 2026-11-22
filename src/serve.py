import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Credit Scoring API", version="0.1")

MODEL_FILE = os.path.join("models", "final_model.pkl")
_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_FILE)
    return _model


class InputData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int = 0
    PAY_2: int = 0
    PAY_3: int = 0
    PAY_4: int = 0
    PAY_5: int = 0
    PAY_6: int = 0
    BILL_AMT1: float = 0
    BILL_AMT2: float = 0
    BILL_AMT3: float = 0
    BILL_AMT4: float = 0
    BILL_AMT5: float = 0
    BILL_AMT6: float = 0
    PAY_AMT1: float = 0
    PAY_AMT2: float = 0
    PAY_AMT3: float = 0
    PAY_AMT4: float = 0
    PAY_AMT5: float = 0
    PAY_AMT6: float = 0


@app.get("/")
def root():
    return {"msg": "credit scoring api работает"}


@app.post("/predict")
def predict(payload: InputData):
    try:
        model = _load_model()
        row = pd.DataFrame([payload.dict()])

        # добавляем фичи
        from src.fe import engineer_features

        row = engineer_features(row)

        proba = float(model.predict_proba(row)[:, 1][0])
        label = int(model.predict(row)[0])

        return {"class": label, "probability": round(proba, 5)}
    except FileNotFoundError:
        raise HTTPException(503, "модель ещё не обучена, запустите dvc repro")
    except Exception as e:
        raise HTTPException(500, str(e))
