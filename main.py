from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define custom layers (same as in your training code)
class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, transformer_units, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim//num_heads, dropout=0.1)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(transformer_units, activation='gelu'),
            layers.Dropout(0.1),
            layers.Dense(projection_dim),
            layers.Dropout(0.1),
        ])

    def call(self, inputs, training=None):
        attention = self.attention(inputs, inputs)
        x = self.norm1(inputs + attention)
        output = self.mlp(x)
        return self.norm2(x + output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "num_heads": self.num_heads,
            "transformer_units": self.transformer_units,
        })
        return config

class VisionTransformer(layers.Layer):
    def __init__(self, patch_size=8, projection_dim=256, num_heads=8,
                 transformer_units=1024, transformer_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.transformer_layers = transformer_layers

        self.patch_proj = layers.Conv2D(
            filters=projection_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid'
        )
        self.transformers = [
            TransformerBlock(projection_dim, num_heads, transformer_units)
            for _ in range(transformer_layers)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(0.1)

    def build(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        self.num_patches = (height // self.patch_size) * (width // self.patch_size)

        self.cls_token = self.add_weight(
            shape=(1, 1, self.projection_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name='cls_token'
        )

        self.pos_embedding = self.add_weight(
            shape=(1, self.num_patches + 1, self.projection_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name='pos_embedding'
        )

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        x = self.patch_proj(inputs)
        x = tf.reshape(x, [batch_size, self.num_patches, self.projection_dim])

        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embedding

        for transformer_block in self.transformers:
            x = transformer_block(x, training=training)

        x = self.norm(x[:, 0])
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "projection_dim": self.projection_dim,
            "num_heads": self.num_heads,
            "transformer_units": self.transformer_units,
            "transformer_layers": self.transformer_layers,
        })
        return config

# Initialize FastAPI app
app = FastAPI(
    title="Disease Prediction API",
    description="AI-powered disease prediction from medical images",
    version="1.0.0",
    root_path="/predict"
)

# Global model variable
model = None

# Class names - Update these based on your actual tea disease classes

CLASS_NAMES = [
    "Tea Algal Leaf Spot",
    "Brown Blight",
    "Gray Blight",
    "Healthy"
]

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    try:
        logger.info("Loading model...")
        
        # Define custom objects for model loading
        custom_objects = {
            'VisionTransformer': VisionTransformer,
            'TransformerBlock': TransformerBlock
        }
        
        # Load model with custom objects
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model("hybrid_cnn_vit_model2.h5")
        
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def preprocess_image(image: Image.Image, target_size=(256, 256)):
    """
    Preprocess the uploaded image for model prediction
    
    Args:
        image: PIL Image object
        target_size: Target size for resizing (width, height) - matches your training size
    
    Returns:
        Preprocessed image array ready for prediction
    """
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for disease prediction",
            "/health": "GET - Health check",
            "/classes": "GET - Get available disease classes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global model
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    return {"status": "healthy", "message": "Model loaded and ready"}

@app.get("/classes")
async def get_classes():
    """Get available disease classes"""
    return {"classes": CLASS_NAMES, "total_classes": len(CLASS_NAMES)}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict disease from uploaded image
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
    
    Returns:
        Prediction results with probabilities
    """
    global model
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read and process the uploaded file
        logger.info(f"Processing uploaded file: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        logger.info("Making prediction...")
        predictions = model.predict(processed_image)
        
        # Get prediction probabilities
        probabilities = predictions[0].tolist()
        
        # Get the highest confidence score and its corresponding class index
        confidence = float(np.max(predictions[0]))
        predicted_class_idx = np.argmax(predictions[0])

        # --- OOD Image Handling Logic ---
        # Define the confidence threshold
        confidence_threshold = 0.70 # You can adjust this value

        if confidence < confidence_threshold:
            # If confidence is below the threshold, classify as OOD
            predicted_class = "Non-Tea Leaf"
        else:
            # Otherwise, use the model's prediction
            predicted_class = CLASS_NAMES[predicted_class_idx]

        # Prepare response
        result = {
            "filename": file.filename,
            "predicted_disease": predicted_class,
            "confidence": confidence,
            "all_probabilities": {
                CLASS_NAMES[i]: prob for i, prob in enumerate(probabilities)
            }
        }
        
        logger.info(f"Prediction complete: {predicted_class} ({confidence:.4f})")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)