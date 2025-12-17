import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Twitter–Reddit Sentiment Classifier",
    layout="centered"
)

st.title("😄😑😠Multi-Model Sentiment Classification System")
st.write("Analyze social media text using multiple machine learning models and compare predictions in real time.")

@st.cache_data
def fetch_models():
    response = requests.get(f"{API_URL}/models")
    return response.json()["models"]

models = fetch_models()


model_name = st.selectbox(
    "Choose a Model",
    models
)

text = st.text_area(
    "Enter text for classification",
    height=150,
    placeholder="Type or paste tweet/reddit comment here..."
)


if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        payload = {
            "model_name": model_name,
            "text": text
        }

        with st.spinner("Predicting..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload
            )


        if response.status_code == 200:
            result = response.json()

            st.success("Prediction Successful!")
            st.write("### Result")
            st.write(f"**Model Used:** {result['model_used']}")
            st.write(f"**Prediction:** {result['prediction']}")
        else:
            st.error(response.json()["detail"])




