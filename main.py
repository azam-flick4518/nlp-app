from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json

app = FastAPI()

class TextInput(BaseModel):
    text: str
    labels: list[str]

@app.post("/classify")
def classify_text(input: TextInput):
    prompt = f"""You are a text classifier. Classify the following text into exactly one of these labels: {', '.join(input.labels)}.

Text: "{input.text}"

Respond with JSON only, no explanation. Format:
{{"label": "chosen_label", "confidence": 0.95, "reasoning": "one sentence"}}"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    raw = response.json()["response"].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # LLM sometimes adds markdown fences — strip them
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

    return {
        "text": input.text,
        "labels": input.labels,
        "result": result
    }