import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CREDIT RISK API")


pipeline = joblib.load("models/model.pkl")


class CreditInput(BaseModel):
    duration: float
    credit_amount: float
    installment_commitment: float
    residence_since: float
    age: float
    existing_credits: float
    num_dependents: float


@app.get("/")
def root():
    return {"title": "Credit Risk Api", "message": "Running"}


@app.post("/predict")
def predict(data: CreditInput):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    return {
        "prediction": int(prediction),
        "risk": "bad" if prediction == 1 else "good",
        "probability": round(float(probability), 4),
    }
