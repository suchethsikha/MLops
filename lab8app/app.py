from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Spotify Popularity Predictor",
    description="Predicts the popularity of a Spotify track based on acousticness.",
    version="1.0"
)

class RequestBody(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: float
    explicit: int


@app.get("/")
def read_root():
    return {"message": "This is a model for predicting Spotify track popularity."}

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("models/lab2_best_model.pkl")

@app.post("/predict")
def predict(data: RequestBody):
    X = [[
        data.danceability,
        data.energy,
        data.key,
        data.loudness,
        data.mode,
        data.speechiness,
        data.acousticness,
        data.instrumentalness,
        data.liveness,
        data.valence,
        data.tempo,
        data.duration_ms,
        data.explicit
    ]]
    prediction = model.predict(X)
    return {"Predicted Popularity": float(prediction[0])}

