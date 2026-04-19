from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Force CPU — as VRAM is too small for reliable GPU inference
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

class TextInput(BaseModel):
    text: str

@app.post("/classify")
def classify_text(input: TextInput):
    result = classifier(input.text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4)
    }
