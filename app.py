import streamlit as st
import requests

st.title("🧠 Zero-Shot Classifier")
st.write("Classify text into any categories you define — powered by Llama 3.2 via Ollama.")

text_input = st.text_area("Your text", placeholder="Type something here...")
labels_input = st.text_input(
    "Labels (comma-separated)",
    placeholder="positive, negative, neutral",
    value="positive, negative, neutral"
)

if st.button("Classify"):
    if not text_input.strip() or not labels_input.strip():
        st.warning("Fill in both fields.")
    else:
        labels = [l.strip() for l in labels_input.split(",")]

        with st.spinner("Classifying..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/classify",
                    json={"text": text_input, "labels": labels},
                    timeout=30  # Ollama can be slow on CPU
                )
                data = response.json()
                result = data["result"]

                st.subheader("Result")
                st.markdown(f"**Label:** `{result['label']}`")
                st.markdown(f"**Confidence:** {result['confidence']:.0%}")
                st.markdown(f"**Reasoning:** {result['reasoning']}")

            except requests.exceptions.ConnectionError:
                st.error("FastAPI server not running. Start: uvicorn main:app --reload")
            except requests.exceptions.Timeout:
                st.error("Ollama took too long. Make sure Ollama is running: ollama serve")
            except Exception as e:
                st.error(f"Unexpected error: {e}")