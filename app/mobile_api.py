"""
mobile_api.py - FastAPI backend for the Expo web/mobile app (and any other client).

Run locally:   uvicorn mobile_api:app --host 0.0.0.0 --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from crop_service import CropService, ValidationError

app = FastAPI(
    title="Pulse Crop Recommendation API",
    description="Prototype prediction API. Trained on a public dataset; not validated on local farms.",
    version="1.1.0",
)

# CORS lets the Expo *web* build (served from another origin) call this API.
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

service = CropService()


class CropInput(BaseModel):
    N: float = Field(ge=0, description="Nitrogen")
    P: float = Field(ge=0, description="Phosphorus")
    K: float = Field(ge=0, description="Potassium")
    temperature: float = Field(description="Temperature (degrees Celsius)")
    humidity: float = Field(ge=0, le=100, description="Relative humidity (%)")
    ph: float = Field(ge=0, le=14, description="Soil pH")
    rainfall: float = Field(ge=0, description="Rainfall (dataset unit; assumed mm)")


@app.get("/")
def root():
    return {"status": "ok", "service": "Pulse Crop Recommendation API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/meta")
def meta():
    return service.info()


@app.post("/predict")
def predict(payload: CropInput):
    try:
        return service.predict(payload.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error"
        )
