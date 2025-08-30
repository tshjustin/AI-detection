from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
import tempfile
import shutil
import os
import httpx
import cv2
import base64
from typing import List, Dict, Any
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="")

deepfake_detector = None

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

@app.on_event("startup")
async def load_model():
    global deepfake_detector
    deepfake_detector = pipeline("video-classification", model="tayyabimam/Deepfake")
    print("Deepfake detection model loaded")

def extract_frames(video_path: str, num_frames: int = 8) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        raise ValueError("can not read video frames")
    
    frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    
    frames_b64 = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(frame_b64)
    
    cap.release()
    return frames_b64

async def call_qwen_video_model(frames: List[str]) -> Dict[str, Any]:
    content = [
        {
            "type": "text",
            "text": "Analyze this video and rate its genuinity on a scale of 1-10, where 1 is completely artificial/animated/movie content (anime, CGI, movie clips, cartoons) and 10 is authentic human-created real-world content. Consider if this appears to be from movies, anime, cartoons, or other non-human created media. Respond with just a number 1-10."
        }
    ]
    
    for i, frame_b64 in enumerate(frames):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}"
            }
        })
    
    payload = {
        "model": "qwen/qwen-2-vl-72b-instruct",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            import re
            score_match = re.search(r'(\d+)', content)
            genuinity_score = int(score_match.group(1)) if score_match else 5
            genuinity_score = max(1, min(10, genuinity_score))
            
            return {"genuinity": genuinity_score}
            
        except Exception as e:
            print(f"qwen API error: {e}")
            return {"genuinity": 0}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="invalid file format")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        deepfake_task = asyncio.create_task(run_deepfake_detection(tmp_path))
        qwen_task = asyncio.create_task(run_qwen_analysis(tmp_path))
        
        deepfake_result, qwen_result = await asyncio.gather(
            deepfake_task, qwen_task, return_exceptions=True
        )
        
        if isinstance(deepfake_result, Exception):
            print(f"Deepfake detection error: {deepfake_result}")
            deepfake_result = []
            
        if isinstance(qwen_result, Exception):
            print(f"Qwen analysis error: {qwen_result}")
            qwen_result = {"genuinity": 5}
        
        return {
            "deepfake_result": deepfake_result,
            "genuinity": qwen_result.get("genuinity", 0)
        }
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

async def run_deepfake_detection(video_path: str) -> List[Dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, deepfake_detector, video_path)

async def run_qwen_analysis(video_path: str) -> Dict[str, Any]:
    try:
        frames = extract_frames(video_path, num_frames=8)
        if not frames:
            raise ValueError("Could not extract frames from video")
        
        result = await call_qwen_video_model(frames)
        return result
        
    except Exception as e:
        print(f"Qwen analysis error: {e}")
        return {"genuinity": 5}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "deepfake_detector": "loaded" if deepfake_detector else "not_loaded",
            "qwen_integration": "enabled"
        }
    }

@app.post("/ping")
def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)