# ============================================
# Pydantic Schemas for API
# ============================================
from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    prediction:        str
    confidence:        float
    probabilities:     Dict[str, float]
    inference_time_ms: float
    status:            str

class HealthResponse(BaseModel):
    status: str
    model:  str
    device: str

class ModelInfoResponse(BaseModel):
    architecture:  str
    total_params:  int
    input_size:    str
    classes:       list
    device:        str
    test_accuracy: str
    auc_roc:       str