import streamlit as st
import requests
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Thai News Classifier",
    page_icon="📰",
    layout="centered"
)

# --- FastAPI Backend URL ---
BACKEND_URL = "http://127.0.0.1:8000/predict"

# --- UI Elements ---
st.title("📰 Thai News Article Classifier")
st.markdown(
    "Enter a piece of Thai news text below, and the model will predict the probability "
    "of it belonging to one of the 12 categories."
)

# Text area for user input
input_text = st.text_area("Enter news text here:", height=250, placeholder="ตัวอย่าง: รัฐบาลประกาศนโยบายใหม่เพื่อกระตุ้นเศรษฐกิจ...")

# Submit button
if st.button("Classify Text"):
    # Show a spinner while waiting for the API response
    with st.spinner('Analyzing the text...'):
        # --- API Request ---
        payload = {"text": input_text}
        response = requests.post(BACKEND_URL, json=payload, timeout=30)

        # --- Display Results ---
        st.subheader("Classification Probabilities")
        results = response.json()
        predictions = results["predictions"]

        df = pd.DataFrame(predictions.items(), columns=['Category', 'Probability'])
        df = df.sort_values(by='Probability', ascending=False).reset_index(drop=True)
        
        st.bar_chart(df.set_index('Category'))
        st.markdown("### Detailed Scores")
        st.table(df)