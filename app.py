import streamlit as st
import requests

st.title("🧠 Sentiment Analyzer")
st.write("Enter text below to classify its sentiment.")

text_input = st.text_area("Your text", placeholder="Type something here...")

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/classify",
                json={"text": text_input}
            )
            result = response.json()
            label = result["label"]
            score = result["score"]

            color = "green" if label == "POSITIVE" else "red"
            st.markdown(f"**Result:** :{color}[{label}]")
            st.markdown(f"**Confidence:** {score:.2%}")

        except requests.exceptions.ConnectionError:
            st.error("FastAPI server not running. Start it with: uvicorn main:app --reload")