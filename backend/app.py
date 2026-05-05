from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

# We will load models in another file to keep things clean.
from models.emotion_model import AdvancedEmotionModel

app = FastAPI(
    title="Advanced Emotion Detection API",
    description="Deep Learning API for multi-label emotion and intensity detection.",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev. In production, specify Vercel/Render URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmotionRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    emotions: List[str]            # Because it's multi-label
    confidences: Dict[str, float]  # All confidences
    intensity: str                 # Low, Medium, High
    primary_emotion: str           # The one with the highest confidence
    macro_sentiment: str           # Positive, Negative, Ambiguous, Neutral
    empathy_response: str          # Dynamic empathy reply
    
class ExplainRequest(BaseModel):
    text: str

class ExplainResponse(BaseModel):
    important_words: Dict[str, float]  # Word -> score

# Initialize the model globally
emotion_model = None

@app.on_event("startup")
async def startup_event():
    global emotion_model
    print("Loading Advanced Emotion Model (Transformer pipeline)...")
    emotion_model = AdvancedEmotionModel()
    print("Model Loaded successfully.")

@app.post("/predict", response_model=EmotionResponse)
async def predict_emotion(req: EmotionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        result = emotion_model.predict(req.text)
        return EmotionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplainResponse)
async def explain_emotion(req: ExplainRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        words_importance = emotion_model.explain(req.text)
        return ExplainResponse(important_words=words_importance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
