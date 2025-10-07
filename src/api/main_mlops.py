"""
FastAPI REST API for geometric shape detection with MLOps monitoring
"""

from src.evaluation.predictor import ShapePredictor
from src.mlops.model_monitor import ModelMonitor
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
from PIL import Image
import io
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    """Response model for predictions"""
    class_name: str = Field(..., description="Predicted class name")
    class_idx: int = Field(..., description="Predicted class index")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="All class probabilities")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_images: int


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_info: Optional[Dict] = None


# Initialize FastAPI app
app = FastAPI(
    title="Geometric Shape Detection API",
    description="Advanced ML API for detecting geometric shapes in images with MLOps monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
predictor: Optional[ShapePredictor] = None
monitor: Optional[ModelMonitor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and monitor on startup"""
    global predictor, monitor
    
    model_path = os.getenv('MODEL_PATH', 'models/saved_models/best_model.h5')
    config_path = os.getenv('CONFIG_PATH', 'models/saved_models/model_config.json')
    
    predictor = ShapePredictor(
        model_path=model_path,
        config_path=config_path
    )
    
    monitor = ModelMonitor(model_name='ml-geometry')
    
    print(f"✅ Model loaded from: {model_path}")
    print(f"✅ Monitor initialized")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Geometric Shape Detection API with MLOps",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring
    """
    model_loaded = predictor is not None and predictor.model is not None
    
    model_info = None
    if model_loaded:
        model_info = {
            "num_classes": len(predictor.class_names),
            "input_shape": list(predictor.model.input_shape[1:])
        }
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_info=model_info
    )


@app.post("/predict")
async def predict_shape(file: UploadFile = File(...)):
    """
    Predict the geometric shape in an uploaded image with monitoring
    """
    start_time = time.time()
    
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Get prediction
        result = predictor.predict_image(image_np)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log to monitor
        if monitor:
            monitor.log_prediction(
                predicted_class=result['class_name'],
                confidence=result['confidence'],
                latency=latency
            )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        if monitor:
            monitor.log_error('prediction_error', str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict geometric shapes for multiple images
    """
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        
        for file in files:
            start_time = time.time()
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_np = np.array(image)
            
            result = predictor.predict_image(image_np)
            predictions.append(result)
            
            # Log each prediction
            if monitor:
                latency = time.time() - start_time
                monitor.log_prediction(
                    predicted_class=result['class_name'],
                    confidence=result['confidence'],
                    latency=latency
                )
        
        return JSONResponse(content={
            "predictions": predictions,
            "total_images": len(predictions)
        })
    
    except Exception as e:
        if monitor:
            monitor.log_error('batch_prediction_error', str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    """
    Get list of available shape classes
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": predictor.class_names,
        "num_classes": len(predictor.class_names)
    }


@app.get("/model/info")
async def get_model_info():
    """
    Get model information
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": predictor.model_path,
        "num_classes": len(predictor.class_names),
        "classes": predictor.class_names,
        "input_shape": list(predictor.model.input_shape[1:])
    }


@app.get("/metrics")
async def get_metrics():
    """
    Get monitoring metrics in Prometheus format
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    return Response(
        content=monitor.get_prometheus_metrics(),
        media_type="text/plain"
    )


@app.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get metrics summary in JSON format
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    return monitor.get_metrics()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
