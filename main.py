from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, List
import uvicorn
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tea Disease Classification API",
    description="API for classifying tea leaf diseases",
    version="1.0.0",
    root_path="/predict"
    
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disease classes
DISEASE_CLASSES = {
    0: "Brown Blight",
    1: "Gray Blight", 
    2: "Tea Leaf Algal",
    3: "Healthy Leaf"
}


model = None


class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    all_predictions: Dict[str, float]
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

def load_model():
    """Load the TensorFlow Lite model"""
    global model
    try:

        model = tf.lite.Interpreter(model_path="tea_disease_model.tflite")
        model.allocate_tensors()
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image: Image.Image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    try:
       
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
      
        image = image.resize(target_size)
        
       
        image_array = np.array(image)
        
        image_array = image_array.astype(np.float32) / 255.0
 
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing image")

def predict_disease(image_array: np.ndarray) -> Dict:
    """Make prediction using the loaded model"""
    try:
       
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        
        model.set_tensor(input_details[0]['index'], image_array)
        
        
        model.invoke()
        
        
        output_data = model.get_tensor(output_details[0]['index'])
        
       
        predictions = tf.nn.softmax(output_data[0]).numpy()
        
     
        predicted_class_idx = np.argmax(predictions)
        predicted_disease = DISEASE_CLASSES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        
        
        all_predictions = {
            DISEASE_CLASSES[i]: float(predictions[i]) 
            for i in range(len(DISEASE_CLASSES))
        }
        
        return {
            "disease": predicted_disease,
            "confidence": confidence,
            "all_predictions": all_predictions
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error making prediction")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.warning("Model failed to load on startup")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for health check"""
    return HealthResponse(
        status="Tea Disease Classification API is running",
        model_loaded=model is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "model not loaded",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict tea disease from uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        PredictionResponse with disease classification and confidence
    """
   
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
  
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        
        processed_image = preprocess_image(image)
        
     
        prediction_result = predict_disease(processed_image)
        
        return PredictionResponse(
            disease=prediction_result["disease"],
            confidence=prediction_result["confidence"],
            all_predictions=prediction_result["all_predictions"],
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict tea disease from multiple uploaded images
    
    Args:
        files: List of image files
    
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
    
    results = []
    
    for i, file in enumerate(files):
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "File must be an image",
                "status": "failed"
            })
            continue
        
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            processed_image = preprocess_image(image)
            prediction_result = predict_disease(processed_image)
            
            results.append({
                "filename": file.filename,
                "disease": prediction_result["disease"],
                "confidence": prediction_result["confidence"],
                "all_predictions": prediction_result["all_predictions"],
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return {"results": results}

@app.get("/classes")
async def get_classes():
    """Get available disease classes"""
    return {
        "classes": DISEASE_CLASSES,
        "total_classes": len(DISEASE_CLASSES)
    }

@app.post("/reload_model")
async def reload_model():
    """Reload the model (useful for model updates)"""
    success = load_model()
    return {
        "status": "success" if success else "failed",
        "message": "Model reloaded successfully" if success else "Failed to reload model"
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )