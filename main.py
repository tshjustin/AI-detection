from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from transformers import pipeline

import tempfile
import shutil
import os

app = FastAPI(title="Deepfake Detector API")

detector = None

@app.on_event("startup")
async def load_model():
    global detector
    detector = pipeline("video-classification", model="tayyabimam/Deepfake")
    print("Model loaded ")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    
    Args:   
        file: MP4 video 

    Returns:
        Json of the payload
        {
            "prediction": "REAL/FAKE", 
            "confidence": float,
            "prediction_code": int
        }
    """

    # HF needs a file handler 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        results = detector(tmp_path)
        return results
    finally:
        os.unlink(tmp_path) 

@app.post('/ping')
def ping():
    return {"status": "ok"}