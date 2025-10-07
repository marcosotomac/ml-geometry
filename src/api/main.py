"""
FastAPI REST API for geometric shape detection
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
    description="Advanced ML API for detecting geometric shapes in images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = ShapePredictor(
    model_path=os.getenv('MODEL_PATH', 'models/saved_models/best_model.h5'),
    config_path=os.getenv('CONFIG_PATH', 'models/saved_models/model_config.json')
)

# Initialize monitor
monitor = ModelMonitor(model_name='ml-geometry')
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

# Global predictor instance
predictor: Optional[ShapePredictor] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor

    # Default model path (can be configured via environment variable)
    model_path = os.getenv('MODEL_PATH', 'models/saved_models/best_model.h5')
    config_path = os.getenv(
        'CONFIG_PATH', 'models/saved_models/model_config.json')

    try:
        if os.path.exists(model_path):
            predictor = ShapePredictor(
                model_path, config_path if os.path.exists(config_path) else None)
            print(f"✅ Model loaded successfully from {model_path}")
        else:
            print(f"⚠️  Warning: Model not found at {model_path}")
            print("   API will run but predictions will fail until model is loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   API will run but predictions will fail")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Geometric Shape Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    model_loaded = predictor is not None

    model_info = None
    if model_loaded:
        model_info = {
            "input_shape": predictor.input_shape,
            "num_classes": len(predictor.class_names),
            "class_names": predictor.class_names
        }

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_info=model_info
    )


@app.post("/predict")
async def predict_shape(file: UploadFile = File(...)):
    """
    Predict the geometric shape in an uploaded image.

    Returns:
        dict: Prediction results including class name, confidence, and probabilities
    """
    start_time = time.time()
    
    try:
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to numpy array
        image_np = np.array(image)

        # Get prediction
        result = predictor.predict_image(image_np)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log to monitor
        monitor.log_prediction(
            predicted_class=result['class_name'],
            confidence=result['confidence'],
            latency=latency
        )

        return JSONResponse(content=result)

    except Exception as e:
        monitor.log_error('prediction_error', str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(...)
):
    """
    Predict geometric shapes in multiple images

    Args:
        files: List of image files

    Returns:
        Batch prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 50:
        raise HTTPException(
            status_code=400, detail="Maximum 50 images per batch")

    try:
        # Read all images
        images = []
        for file in files:
            if not file.content_type.startswith('image/'):
                continue

            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            images.append(image)

        if len(images) == 0:
            raise HTTPException(
                status_code=400, detail="No valid images provided")

        # Make batch prediction
        results = predictor.predict_batch(images)

        # Format response
        predictions = [
            PredictionResponse(
                class_name=result.get('class', f"Class {result['class_idx']}"),
                class_idx=result['class_idx'],
                confidence=result['confidence']
            )
            for result in results
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            total_images=len(images)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/classes", tags=["Model Info"])
async def get_classes():
    """Get list of available classes"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "classes": predictor.class_names,
        "num_classes": len(predictor.class_names)
    }


@app.get("/model/info", tags=["Model Info"])
async def get_model_info():
    """Get detailed model information"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "input_shape": predictor.input_shape,
        "num_classes": len(predictor.class_names),
        "class_names": predictor.class_names,
        "config": predictor.config
    }


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv('PORT', 8000))

    print(f"\n{'='*60}")
    print(f"🚀 Starting Geometric Shape Detection API")
    print(f"{'='*60}")
    print(f"📡 Server: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"📖 ReDoc: http://localhost:{port}/redoc")
    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
