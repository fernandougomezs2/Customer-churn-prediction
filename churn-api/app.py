from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Cargar modelo (Pipeline completo)
model = joblib.load("churn_model.pkl")

app = FastAPI(title="Churn Prediction API", version="1.0")

# === Input schema (debe coincidir con columnas X del entrenamiento) ===
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Churn API running. Go to /docs to test."}

@app.post("/predict")
def predict_churn(customer: Customer):
    # Convertir a DataFrame (1 fila)
    df = pd.DataFrame([customer.model_dump()])

    # Probabilidad y predicción
    proba = float(model.predict_proba(df)[0, 1])  # prob de churn (clase 1)
    pred = int(proba >= 0.5)  # umbral (puedes cambiarlo)

    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": pred,   # 1 = churn, 0 = no churn
        "threshold": 0.5
    }
