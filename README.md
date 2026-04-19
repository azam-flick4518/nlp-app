# NLP Sentiment Analyzer

A simple AI-powered web app that classifies text sentiment using a pre-trained HuggingFace model, served via FastAPI and visualized through Streamlit.

## Stack

- **FastAPI** — REST API backend
- **HuggingFace Transformers** — distilbert sentiment model (runs on CPU)
- **Streamlit** — frontend UI
- **PyTorch** — model inference engine

## Project Structure
```text
nlp-app/
├── main.py       # FastAPI backend
├── app.py        # Streamlit frontend
├── README.md
└── .venv/        # virtual environment (not committed)
```

## Setup

### 1. Activate virtual environment
.venv\Scripts\activate

### 2. Install dependencies
pip install fastapi uvicorn transformers torch streamlit

### 3. Run the backend (Terminal 1)
uvicorn main:app --reload

### 4. Run the frontend (Terminal 2)
streamlit run app.py

### 5. Open the app
http://localhost:8501

## API

### POST /classify

**Request:**
{
  "text": "your text here"
}

**Response:**
{
  "label": "POSITIVE",
  "score": 0.9998
}

Interactive API docs available at: http://127.0.0.1:8000/docs

## Notes

- Model downloads ~260MB on first run — this is normal
- Forced to CPU (device=-1) due to limited VRAM.
- Close unused apps before running — model needs ~1.5GB RAM

Terminal 1: uvicorn main:app --reload
Terminal 2: streamlit run app.py