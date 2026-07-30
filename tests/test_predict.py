import joblib 
import pandas as pd



pipeline =  joblib.load("models/model.pkl")

def predict(data: dict) -> dict:
    input_df = pd.DataFrame([data])
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    return{
        "prediction": int(prediction),
        "risk": "bad" if prediction == 1 else "good",
        "probability": round(float(probability), 4)
    }
    

def test_low_risk_prediction():
    data = {
        "duration": 12,
        "credit_amount": 3000,
        "installment_commitment": 2,
        "residence_since": 3,
        "age": 30,
        "existing_credits": 1,
        "num_dependents": 1
    }
    result = predict(data)
    assert result["prediction"] in [0, 1]
    assert result["risk"] in ["good", "bad"]
    assert 0.0 <= result["probability"] <= 1.0
    
    
def test_high_risk_prediction():
    data = {
        "duration": 48,
        "credit_amount": 15000,
        "installment_commitment": 4,
        "residence_since": 1,
        "age": 22,
        "existing_credits": 3,
        "num_dependents": 2
    }
    result = predict(data)
    assert result["prediction"] in [0, 1]
    assert result["risk"] in ["good", "bad"]
    assert 0.0 <= result["probability"] <=  1.0
    
    
def test_probability_range():
    data = {
        "duration": 6,
        "credit_amount": 1000,
        "installment_commitment": 1,
        "residence_since": 4,
        "age": 45,
        "existing_credits": 1,
        "num_dependents": 1
    }
    result = predict(data)
    assert 0.0 <= result["probability"] <=  1.0



def test_prediction_is_binary():
    data = {
        "duration": 24,
        "credit_amount": 5000,
        "installment_commitment": 3,
        "residence_since": 2,
        "age": 35,
        "existing_credits": 1,
        "num_dependents": 1
    }
    result = predict(data)
    assert result["prediction"] in [0, 1]