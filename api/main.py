# ============================================
# FastAPI Backend for Chest X-Ray Diagnosis
# ============================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import io
import json
import time
import logging
from pathlib import Path

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="AI-powered pneumonia detection from chest X-rays using EfficientNetB0",
    version="1.0.0",
)

# CORS middleware — allows frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODEL CONFIGURATION
# ============================================
CLASSES      = ['NORMAL', 'PNEUMONIA']
IMAGE_SIZE   = 224
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]
MODEL_PATH = str(Path(__file__).parent.parent / "experiments" / "best_model.pth")
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# MODEL DEFINITION
# ============================================
class ChestXRayModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ChestXRayModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features   = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ============================================
# PREPROCESSING
# ============================================
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

# ============================================
# LOAD MODEL ON STARTUP
# ============================================
@app.on_event("startup")
async def load_model():
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = ChestXRayModel(num_classes=2, dropout_rate=0.3)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(DEVICE)
        model.eval()
        logger.info(f"✅ Model loaded successfully on {DEVICE}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise e

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint — API info"""
    return {
        "message":     "Chest X-Ray Pneumonia Detection API",
        "version":     "1.0.0",
        "status":      "running",
        "device":      str(DEVICE),
        "classes":     CLASSES,
        "docs":        "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status":  "healthy",
        "model":   "loaded",
        "device":  str(DEVICE)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict pneumonia from chest X-ray image

    Args:
        file: X-ray image file (JPG, PNG)

    Returns:
        prediction, confidence scores, inference time
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload JPG or PNG image."
        )

    try:
        start_time = time.time()

        # Read image
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert('RGB')
        image    = np.array(image)

        # Preprocess
        image = apply_clahe(image)
        transformed = transform(image=image)
        tensor = transformed['image'].unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs     = model(tensor)
            probs       = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)

        # Results
        pred_class   = CLASSES[predicted.item()]
        confidence   = confidence.item()
        all_probs    = probs[0].cpu().numpy().tolist()
        inference_ms = (time.time() - start_time) * 1000

        logger.info(f"Prediction: {pred_class} ({confidence:.3f}) in {inference_ms:.1f}ms")

        return JSONResponse({
            "prediction":     pred_class,
            "confidence":     round(confidence, 4),
            "probabilities": {
                "NORMAL":    round(all_probs[0], 4),
                "PNEUMONIA": round(all_probs[1], 4),
            },
            "inference_time_ms": round(inference_ms, 2),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get model information"""
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "architecture":   "EfficientNetB0",
        "total_params":   total_params,
        "input_size":     f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "classes":        CLASSES,
        "device":         str(DEVICE),
        "preprocessing":  "CLAHE + ImageNet Normalization",
        "test_accuracy":  "93.37%",
        "auc_roc":        "0.9705",
    }