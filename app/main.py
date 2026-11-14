# Creating the API for the model using FastAIP

from pathlib import Path

import joblib

import pandas as pd

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field

from app.crop_service import FEATURES, CropService, ValidationError
from app.config import get_settings

BASE_DIR = Path(__file__).resolve().parent.parent



MODEL_VERSION = "1.0.0"
 

app = FastAPI(
    title="Pulse Crop Recommendation API",
    version=MODEL_VERSION
)
settings = get_settings()
MODEL = joblib.load(settings.model_path)
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    return {
        "message": "Welcome to Pulse Crop Recommendation API",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
    }

class PredictionInput(BaseModel):

    N: float = Field(..., ge=0, description="Nitrogen content")

    P: float = Field(..., ge=0, description="Phosphorus content")

    K: float = Field(..., ge=0, description="Potassium content")

    temperature: float = Field(..., description="Temperature in Celsius")

    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")

    ph: float = Field(..., ge=0, le=14, description="Soil pH")

    rainfall: float = Field(..., ge=0, description="Rainfall in mm")
    

class PredictionOutput(BaseModel):

    recommended_crop: str

    confidence: float

    model_version: str
    
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_version": MODEL_VERSION
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(item: PredictionInput):

    try:

        df = pd.DataFrame([item.model_dump()])

        # Ensure the features are in the correct order
        df = df[[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]]

        # Predict the recommended crop
        prediction = MODEL.predict(df)[0]

        # Calculate prediction probabilities
        probabilities = MODEL.predict_proba(df)[0]

        # Get the highest probability
        confidence = float(max(probabilities))

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc

    return PredictionOutput(
        recommended_crop=str(prediction),
        confidence=round(confidence, 4),
        model_version=MODEL_VERSION
    )
