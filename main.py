from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File

import shutil
import tempfile

from transformers import pipeline

detector = pipeline("video-classification", model="tayyabimam/Deepfake")

app = FastAPI(title="Deepfake Detector API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # HF endpoint requires file handler 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    results = detector(tmp_path)
    
    return results
