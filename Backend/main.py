import pickle
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model import SentimentRNN
from preprocessing import preprocess_review

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

MODEL_PATH = ARTIFACTS_DIR / "rnn_sentiment.pth"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="IMDB Sentiment RNN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model: SentimentRNN | None = None
vectorizer = None
label_encoder = None


class PredictRequest(BaseModel):
    review: str = Field(..., min_length=1, description="Raw movie review text")


class PredictResponse(BaseModel):
    sentiment: str
    positive_probability: float
    confidence: float


@app.on_event("startup")
def load_artifacts():
    global model, vectorizer, label_encoder

    missing = [
        p.name for p in (MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH) if not p.exists()
    ]
    if missing:
        raise RuntimeError(
            f"Missing artifact file(s) in {ARTIFACTS_DIR}: {missing}. "
            "Run save_artifacts.py in your notebook and copy the files there "
            "(see backend/README instructions)."
        )

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    input_size = len(vectorizer.get_feature_names_out())
    model = SentimentRNN(input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None or vectorizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    cleaned = preprocess_review(payload.review)
    if not cleaned.strip():
        raise HTTPException(
            status_code=400,
            detail="Review became empty after preprocessing (e.g. only stopwords/punctuation). "
            "Try a longer review.",
        )

    features = vectorizer.transform([cleaned]).toarray() 
    tensor = torch.from_numpy(features).float().to(device)
    tensor = tensor.unsqueeze(1)  

    with torch.no_grad():
        logits = model(tensor)
        prob_label1 = torch.sigmoid(logits.squeeze()).item()  
    positive_index = list(label_encoder.classes_).index("positive")
    prob_positive = prob_label1 if positive_index == 1 else 1 - prob_label1

    predicted_label = "positive" if prob_positive > 0.5 else "negative"
    confidence = prob_positive if predicted_label == "positive" else 1 - prob_positive

    return PredictResponse(
        sentiment=predicted_label,
        positive_probability=round(prob_positive, 4),
        confidence=round(confidence, 4),
    )


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)