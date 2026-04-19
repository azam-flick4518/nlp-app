# NLP Classifier — Sentiment & Zero-Shot
A simple AI-powered web app that classifies text sentiment using a pre-trained HuggingFace model, served via FastAPI and visualized through Streamlit.
A web app demonstrating two approaches to text classification:
1. **Sentiment Analysis** — fast, binary (distilbert)
2. **Zero-Shot Classification** — flexible, any labels you define (Llama 3.2 via Ollama)

## Stack

- **FastAPI** — REST API backend
- **HuggingFace Transformers** — distilbert for sentiment (runs on CPU)
- **Ollama + Llama 3.2 3B** — zero-shot classification via local LLM
- **Streamlit** — frontend UI
- **PyTorch** — model inference engine

## Why Ollama for Zero-Shot?

The canonical HuggingFace zero-shot model is `facebook/bart-large-mnli` (~1.6GB).
We used Ollama instead because `llama3.2:3b` was already available locally — no extra download needed.

### About facebook/bart-large-mnli

`facebook/bart-large-mnli` is a BART model fine-tuned on the Multi-Genre Natural Language 
Inference (MNLI) dataset. It works by framing classification as an entailment problem:
given a text and a candidate label, it scores how likely the text "entails" that label.

**Characteristics:**
- Single forward pass — faster than generative models
- Returns probability scores for all labels simultaneously
- No reasoning or explanation, just scores
- Better for high-volume production classification

**To use it instead of Ollama**, replace `main.py` with:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Downloads ~1.6GB on first run
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=-1)

class TextInput(BaseModel):
    text: str
    labels: list[str]

@app.post("/classify")
def classify_text(input: TextInput):
    result = classifier(input.text, candidate_labels=input.labels)
    return {
        "text": input.text,
        "results": [
            {"label": l, "score": round(s, 4)}
            for l, s in zip(result["labels"], result["scores"])
        ]
    }
```

**Key difference from Ollama approach:**
- No reasoning field — just label + probability score
- Faster inference (single pass vs token generation)
- Deterministic — same input always gives same output
- Ollama gives reasoning, bart-large-mnli gives calibrated probabilities

## Project Structure
```
nlp-app/
├── main.py         # FastAPI backend (zero-shot via Ollama)
├── app.py          # Streamlit frontend
├── requirements.txt
├── README.md
└── .venv/          # virtual environment (not committed)
```

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- `llama3.2:3b` pulled: `ollama pull llama3.2:3b`

### 1. Activate virtual environment
.venv\Scripts\activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Start Ollama (Terminal 1)
ollama serve

### 4. Run the backend (Terminal 2)
uvicorn main:app --reload

### 5. Run the frontend (Terminal 3)
streamlit run app.py

### 6. Open the app
http://localhost:8501

## API

### POST /classify

**Request:**
```json
{
  "text": "this is ok, but kind of worst",
  "labels": ["positive", "negative", "neutral"]
}
```

**Response:**
```json
{
  "text": "this is ok, but kind of worst",
  "labels": ["positive", "negative", "neutral"],
  "result": {
    "label": "negative",
    "confidence": 0.90,
    "reasoning": "contains the word 'worst'"
  }
}
```

Interactive API docs: `http://127.0.0.1:8000/docs`

## Model Comparison

| | distilbert (sentiment) | Llama 3.2 via Ollama | bart-large-mnli |
|---|---|---|---|
| Type | Fine-tuned classifier | Generative LLM | NLI-based classifier |
| Labels | Fixed (pos/neg only) | Any you define | Any you define |
| Output | Score only | Label + confidence + reasoning | Scores per label |
| Speed | Fastest | Slowest | Fast |
| Size | 260MB | 2GB (already local) | 1.6GB |
| Best for | Binary sentiment at scale | Learning, explainability | Production zero-shot |

## Known Limitations

- distilbert is trained on movie reviews — struggles with neutral/ambiguous text
- Llama 3.2 via Ollama is slower than dedicated classifiers (~3-5 seconds per request)
- All models run on CPU — inference is slower than GPU

## Notes

- HuggingFace cache redirected to `C:\huggingface_cache`
- Forced CPU inference (device=-1) due to limited VRAM(2GB)
- Close unused apps before running — models need ~2GB RAM combined
