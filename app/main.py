from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Wine Quality Prediction API")

NAME = "M.Ruthwik"
ROLLNO = "2022BCS0135"

MODEL_PATH = "models/model.joblib"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return {
        "name": NAME,
        "roll_no": ROLLNO,
        "message": "Welcome to Wine Quality Prediction API"
    }

@app.post("/predict")
def predict(data: WineData):
    if model is None:
        return {"error": "Model not loaded", "name": NAME, "roll_no": ROLLNO}
    
    # Handle dict compatibility across Pydantic versions
    data_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)[0]
    
    return {
        "prediction": float(prediction),
        "name": NAME,
        "roll_no": ROLLNO
    }
