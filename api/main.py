from fastapi import FastAPI

from src.predict import predict_single
from src.schemas import CustomerInput

app = FastAPI(title="Customer Subscription Prediction API")


@app.get("/")
def root():
    return {"message": "Customer Subscription Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerInput):
    result = predict_single(customer.model_dump())
    return result